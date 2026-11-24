from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pyharness.runtime.event_journal import EventJournal
from pyharness.runtime.governance import (
    ApprovalBroker,
    CancellationToken,
    ContextCompressor,
    TokenBudgetGuard,
    TurnCancelledError,
)
from pyharness.runtime.llm_client import BaseLLMClient
from pyharness.runtime.schemas import (
    EventType,
    Message,
    RiskLevel,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnResult,
    TurnStatus,
)

ToolHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Centralized registry with JSON-Schema required-arg validation and risk metadata."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
    ) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return [item.spec for item in self._tools.values()]

    @staticmethod
    def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
        required = spec.parameters.get("required") or []
        missing = [k for k in required if k not in arguments]
        if missing:
            return f"Missing required parameter(s) for '{spec.name}': {', '.join(missing)}"
        return None


@dataclass
class HarnessConfig:
    max_steps: int = 10
    max_parallel_tools: int = 4
    max_consecutive_errors: int = 3
    max_tokens_per_turn: int = 32000
    approval_timeout_seconds: float = 120.0
    context_compress_threshold: int = 6000
    keep_recent_messages: int = 4


class AgentHarnessEngine:
    """
    Deterministic Enterprise Agent Harness Loop.

    Key production guarantees:
    - Cooperative session cancellation across LLM waits, approvals, and tool execution.
    - Bounded parallel tool execution (`asyncio.gather` + `asyncio.Semaphore`).
    - Human-in-the-Loop approval gate before any HIGH-risk tool runs.
    - Structured Self-Correction loop with a hard consecutive-error breaker.
    - Automatic incremental context compression when prompt tokens cross threshold.
    - Durable EventJournal persistence for every state transition.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        journal: EventJournal | None = None,
        approval_broker: ApprovalBroker | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.journal = journal or EventJournal(":memory:")
        self.approval_broker = approval_broker or ApprovalBroker()
        self.config = config or HarnessConfig()
        self.compressor = ContextCompressor(
            threshold_tokens=self.config.context_compress_threshold,
            keep_recent_messages=self.config.keep_recent_messages,
        )

    async def _await_with_cancellation(
        self,
        coro: Awaitable[Any],
        cancellation: CancellationToken,
        timeout: float | None = None,
    ) -> Any:
        """Race any awaitable against CancellationToken and an optional timeout."""
        cancellation.raise_if_cancelled()
        work_task = asyncio.create_task(coro)
        cancel_task = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {work_task, cancel_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done or cancellation.is_cancelled:
                work_task.cancel()
                raise TurnCancelledError(cancellation.reason or "cancelled")
            if work_task in done:
                return work_task.result()
            work_task.cancel()
            raise asyncio.TimeoutError(f"Operation timed out after {timeout}s")
        finally:
            cancel_task.cancel()

    async def _execute_single_tool(
        self,
        session_id: str,
        turn_id: str,
        tc: ToolCall,
        semaphore: asyncio.Semaphore,
        cancellation: CancellationToken,
    ) -> ToolResult:
        t0 = time.perf_counter()

        # 1. Handle malformed JSON emitted by LLM -> feed back for Self-Correction
        if tc.parse_error:
            return ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Tool Argument Parse Error] {tc.parse_error}. Please output valid JSON.",
                is_error=True,
                error_type="json_parse_error",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        reg = self.tools.get(tc.name)
        if reg is None:
            available = ", ".join(s.name for s in self.tools.list_specs())
            return ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Unknown Tool] '{tc.name}' does not exist. Available tools: [{available}].",
                is_error=True,
                error_type="unknown_tool",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 2. Schema required-argument validation
        validation_err = self.tools.validate_arguments(reg.spec, tc.arguments)
        if validation_err:
            return ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Validation Error] {validation_err}. Please correct the parameters.",
                is_error=True,
                error_type="validation_error",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        await self.journal.emit(
            session_id,
            turn_id,
            EventType.TOOL_CALL,
            {
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "risk_level": reg.spec.risk_level.value,
            },
        )

        # 3. Human-in-the-Loop governance gate for HIGH-risk tools
        if reg.spec.risk_level == RiskLevel.HIGH:
            req = self.approval_broker.create_request(
                session_id=session_id,
                turn_id=turn_id,
                tool_call=tc,
                reason=f"Tool '{tc.name}' is marked HIGH risk and requires explicit approval.",
            )
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.APPROVAL_REQUIRED,
                {
                    "approval_id": req.approval_id,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "reason": req.reason,
                },
            )
            decision = await self.approval_broker.wait_for_decision(
                req=req,
                cancellation=cancellation,
                timeout_seconds=self.config.approval_timeout_seconds,
            )
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.APPROVAL_RESOLVED,
                {
                    "approval_id": decision.approval_id,
                    "approved": decision.approved,
                    "reviewer": decision.reviewer,
                    "note": decision.note,
                },
            )
            if not decision.approved:
                res = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=(
                        f"[Approval Denied] Execution of '{tc.name}' was rejected by "
                        f"{decision.reviewer}. Reason: {decision.note or 'Not approved'}."
                    ),
                    is_error=True,
                    error_type="approval_denied",
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )
                await self.journal.emit(session_id, turn_id, EventType.TOOL_RESULT, res.model_dump())
                return res

        # 4. Bounded parallel execution with timeout and optional transient retries
        attempts = reg.spec.max_retries + 1
        last_err: Exception | None = None
        async with semaphore:
            for _ in range(attempts):
                cancellation.raise_if_cancelled()
                try:
                    if inspect.iscoroutinefunction(reg.handler):
                        raw_out = await self._await_with_cancellation(
                            reg.handler(**tc.arguments),
                            cancellation=cancellation,
                            timeout=reg.spec.timeout_seconds,
                        )
                    else:
                        raw_out = await self._await_with_cancellation(
                            asyncio.to_thread(reg.handler, **tc.arguments),
                            cancellation=cancellation,
                            timeout=reg.spec.timeout_seconds,
                        )
                    res = ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=str(raw_out),
                        is_error=False,
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    )
                    await self.journal.emit(
                        session_id, turn_id, EventType.TOOL_RESULT, res.model_dump()
                    )
                    return res
                except TurnCancelledError:
                    raise
                except Exception as exc:
                    last_err = exc

        err_type = "timeout" if isinstance(last_err, asyncio.TimeoutError) else type(last_err).__name__
        res = ToolResult(
            tool_call_id=tc.id,
            tool_name=tc.name,
            content=f"[Tool Execution Error: {err_type}] {last_err}. Analyze the error and self-correct.",
            is_error=True,
            error_type=err_type,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
        await self.journal.emit(session_id, turn_id, EventType.TOOL_RESULT, res.model_dump())
        return res

    async def run_turn(
        self,
        session_id: str,
        user_input: str,
        history: list[Message] | None = None,
        system_prompt: str = "You are PyHarness, an enterprise AI assistant with governed tool use.",
        cancellation: CancellationToken | None = None,
    ) -> TurnResult:
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        cancel = cancellation or CancellationToken()
        budget = TokenBudgetGuard(max_tokens_per_turn=self.config.max_tokens_per_turn)
        semaphore = asyncio.Semaphore(self.config.max_parallel_tools)
        t_start = time.perf_counter()

        messages: list[Message] = list(history) if history else [
            Message(role="system", content=system_prompt)
        ]
        messages.append(Message(role="user", content=user_input))

        await self.journal.emit(
            session_id,
            turn_id,
            EventType.TURN_START,
            {"user_input": user_input, "history_len": len(messages)},
        )

        consecutive_errors = 0
        steps = 0

        try:
            for step in range(1, self.config.max_steps + 1):
                cancel.raise_if_cancelled()
                steps = step

                # Step A: Context window guard & incremental compression
                if self.compressor.should_compress(messages):
                    messages, stats = self.compressor.compress(messages)
                    if stats["messages_folded"] > 0:
                        await self.journal.emit(
                            session_id,
                            turn_id,
                            EventType.CONTEXT_COMPRESSED,
                            stats,
                        )

                # Step B: Call LLM (race against cancellation)
                llm_resp = await self._await_with_cancellation(
                    self.llm.complete(messages, tools=self.tools.list_specs()),
                    cancellation=cancel,
                )
                budget.record(llm_resp.usage)

                if llm_resp.reasoning_content:
                    await self.journal.emit(
                        session_id,
                        turn_id,
                        EventType.THINKING,
                        {"step": step, "reasoning": llm_resp.reasoning_content},
                    )

                if budget.is_exceeded:
                    msg = (
                        f"Token budget exceeded: {budget.usage.total_tokens} > "
                        f"{self.config.max_tokens_per_turn}"
                    )
                    await self.journal.emit(
                        session_id,
                        turn_id,
                        EventType.TURN_COMPLETE,
                        {"status": TurnStatus.BUDGET_EXCEEDED.value, "reason": msg},
                    )
                    return TurnResult(
                        session_id=session_id,
                        turn_id=turn_id,
                        status=TurnStatus.BUDGET_EXCEEDED,
                        final_answer=llm_resp.content,
                        steps_executed=steps,
                        usage=budget.usage,
                        total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                        error_message=msg,
                        messages=messages,
                    )

                # Step C: Final answer (no tool calls)
                if not llm_resp.tool_calls:
                    messages.append(Message(role="assistant", content=llm_resp.content))
                    await self.journal.emit(
                        session_id,
                        turn_id,
                        EventType.TOKEN_DELTA,
                        {"content": llm_resp.content},
                    )
                    await self.journal.emit(
                        session_id,
                        turn_id,
                        EventType.TURN_COMPLETE,
                        {
                            "status": TurnStatus.COMPLETED.value,
                            "steps": steps,
                            "usage": budget.usage.model_dump(),
                        },
                    )
                    return TurnResult(
                        session_id=session_id,
                        turn_id=turn_id,
                        status=TurnStatus.COMPLETED,
                        final_answer=llm_resp.content,
                        steps_executed=steps,
                        usage=budget.usage,
                        total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                        messages=messages,
                    )

                # Step D: Execute all tool calls in this turn concurrently (bounded by Semaphore)
                messages.append(
                    Message(
                        role="assistant",
                        content=llm_resp.content,
                        tool_calls=llm_resp.tool_calls,
                    )
                )
                tool_results: list[ToolResult] = await asyncio.gather(
                    *(
                        self._execute_single_tool(
                            session_id=session_id,
                            turn_id=turn_id,
                            tc=tc,
                            semaphore=semaphore,
                            cancellation=cancel,
                        )
                        for tc in llm_resp.tool_calls
                    )
                )

                step_had_error = False
                for tr in tool_results:
                    messages.append(
                        Message(
                            role="tool",
                            name=tr.tool_name,
                            tool_call_id=tr.tool_call_id,
                            content=tr.content,
                        )
                    )
                    if tr.is_error:
                        step_had_error = True

                # Step E: Self-Correction loop accounting & breaker
                if step_had_error:
                    consecutive_errors += 1
                    await self.journal.emit(
                        session_id,
                        turn_id,
                        EventType.SELF_CORRECTION,
                        {
                            "step": step,
                            "consecutive_errors": consecutive_errors,
                            "errors": [tr.error_type for tr in tool_results if tr.is_error],
                        },
                    )
                    if consecutive_errors >= self.config.max_consecutive_errors:
                        err_msg = (
                            f"Stopped after {consecutive_errors} consecutive failed tool steps "
                            "to prevent infinite self-correction loop."
                        )
                        await self.journal.emit(
                            session_id,
                            turn_id,
                            EventType.TURN_COMPLETE,
                            {"status": TurnStatus.ERROR_LIMIT_EXCEEDED.value, "reason": err_msg},
                        )
                        return TurnResult(
                            session_id=session_id,
                            turn_id=turn_id,
                            status=TurnStatus.ERROR_LIMIT_EXCEEDED,
                            steps_executed=steps,
                            usage=budget.usage,
                            total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                            error_message=err_msg,
                            messages=messages,
                        )
                else:
                    consecutive_errors = 0

            # Exceeded max_steps
            msg = f"Reached max_steps={self.config.max_steps} without final answer."
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.TURN_COMPLETE,
                {"status": TurnStatus.MAX_STEPS_EXCEEDED.value, "reason": msg},
            )
            return TurnResult(
                session_id=session_id,
                turn_id=turn_id,
                status=TurnStatus.MAX_STEPS_EXCEEDED,
                steps_executed=steps,
                usage=budget.usage,
                total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                error_message=msg,
                messages=messages,
            )

        except TurnCancelledError as exc:
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.TURN_CANCELLED,
                {"reason": str(exc), "steps": steps},
            )
            return TurnResult(
                session_id=session_id,
                turn_id=turn_id,
                status=TurnStatus.CANCELLED,
                steps_executed=steps,
                usage=budget.usage,
                total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                error_message=str(exc),
                messages=messages,
            )
