from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

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
    HarnessHook,
    Message,
    RiskLevel,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnResult,
    TurnStatus,
)
from pyharness.security.auth import current_principal

if TYPE_CHECKING:
    from pyharness.infra.telemetry import TelemetryRecorder
    from pyharness.security.policy import PolicyEngine

ToolHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Centralized registry with JSON-Schema required-arg & basic type validation plus risk metadata."""

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

        props = spec.parameters.get("properties") or {}
        type_map: dict[str, tuple[type, ...]] = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "object": (dict,),
            "array": (list,),
        }
        for key, val in arguments.items():
            prop_schema = props.get(key)
            if not isinstance(prop_schema, dict) or val is None:
                continue
            expected_type = prop_schema.get("type")
            if expected_type in type_map:
                if expected_type in ("integer", "number") and isinstance(val, bool):
                    return f"Parameter '{key}' for '{spec.name}' must be {expected_type}, got boolean"
                if not isinstance(val, type_map[expected_type]):
                    return f"Parameter '{key}' for '{spec.name}' must be {expected_type}, got {type(val).__name__}"
            enum_vals = prop_schema.get("enum")
            if enum_vals and val not in enum_vals:
                return f"Parameter '{key}' for '{spec.name}' must be one of {enum_vals}, got {val!r}"
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
    enforce_skill_tool_scope: bool = True


class AgentHarnessEngine:
    """
    Deterministic Enterprise Agent Harness Loop (DSH L4 Runtime).

    Key enterprise guarantees:
    - True incremental token & reasoning streaming (`llm.stream`) raced against `CancellationToken`.
    - Clean per-turn transient Skill instruction injection (`turn_system_instructions`) paired
      with `allowed_tools` dual-layer scoping.
    - Synchronous Dynamic Access Control Policy (ACP) evaluation (`DENY` / `ASK` / `ALLOW` + Shadow `AUDIT`).
    - Reversible `HarnessHook` plugin seam (`add_hook() -> dispose`) with DLP redaction applied
      BEFORE persisting tool outputs to the tamper-evident audit journal.
    - Bounded parallel tool execution (`asyncio.gather` + `asyncio.Semaphore`), Four-Eyes
      Human-in-the-Loop approvals, OpenTelemetry GenAI spans, and subagent context binding.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        journal: EventJournal | None = None,
        approval_broker: ApprovalBroker | None = None,
        config: HarnessConfig | None = None,
        *,
        policy_engine: PolicyEngine | None = None,
        telemetry: TelemetryRecorder | None = None,
        subagent_manager: Any | None = None,
        summary_llm: BaseLLMClient | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.journal = journal or EventJournal(":memory:")
        self.approval_broker = approval_broker or ApprovalBroker()
        self.config = config or HarnessConfig()
        self.policy_engine = policy_engine
        self.telemetry = telemetry
        self.subagent_manager = subagent_manager
        self.summary_llm = summary_llm
        self.compressor = ContextCompressor(
            threshold_tokens=self.config.context_compress_threshold,
            keep_recent_messages=self.config.keep_recent_messages,
        )
        self._hooks: list[HarnessHook] = []

    def add_hook(self, hook: HarnessHook) -> Callable[[], None]:
        """Register a lifecycle hook and return a reversible `dispose()` function (Cordis style)."""
        self._hooks.append(hook)

        def _dispose() -> None:
            if hook in self._hooks:
                self._hooks.remove(hook)

        return _dispose

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
        allowed_tools: set[str] | None = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        principal = current_principal.get()

        for hook in list(self._hooks):
            tc = await hook.before_tool(session_id, turn_id, tc)

        async def _finalize_and_emit(result: ToolResult) -> ToolResult:
            # Apply `after_tool` hooks (including DLP redaction & injection tagging) BEFORE
            # emitting into EventJournal so raw secrets never enter the durable audit log!
            for hook in list(self._hooks):
                result = await hook.after_tool(session_id, turn_id, tc, result)
            await self.journal.emit(
                session_id, turn_id, EventType.TOOL_RESULT, result.model_dump()
            )
            if self.telemetry is not None:
                self.telemetry.record_tool_execution(
                    tc.name, result.is_error, result.duration_ms
                )
            return result

        # 1. Handle malformed JSON emitted by LLM -> feed back for Self-Correction
        if tc.parse_error:
            res = ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Tool Argument Parse Error] {tc.parse_error}. Please output valid JSON.",
                is_error=True,
                error_type="json_parse_error",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            return await _finalize_and_emit(res)

        # 2. Enforce active Skill `allowed_tools` least-privilege scope
        if (
            self.config.enforce_skill_tool_scope
            and allowed_tools is not None
            and tc.name not in allowed_tools
        ):
            res = ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=(
                    f"[Tool Scope Denied] Tool '{tc.name}' is outside the active skill's "
                    f"permitted tools: {sorted(allowed_tools)}."
                ),
                is_error=True,
                error_type="tool_scope_denied",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            return await _finalize_and_emit(res)

        reg = self.tools.get(tc.name)
        if reg is None:
            available = ", ".join(s.name for s in self.tools.list_specs())
            res = ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Unknown Tool] '{tc.name}' does not exist. Available tools: [{available}].",
                is_error=True,
                error_type="unknown_tool",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            return await _finalize_and_emit(res)

        # 3. Schema required-argument & type validation
        validation_err = self.tools.validate_arguments(reg.spec, tc.arguments)
        if validation_err:
            res = ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[Validation Error] {validation_err}. Please correct the parameters.",
                is_error=True,
                error_type="validation_error",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            return await _finalize_and_emit(res)

        # 4. Dynamic Agentic Control Plane (ACP) Policy Evaluation (DENY / ASK / ALLOW + Shadow AUDIT)
        policy_requires_approval = False
        policy_required_roles: list[str] | None = None
        policy_rule_id: str | None = None
        policy_reason: str = ""

        if self.policy_engine is not None:
            eval_res = self.policy_engine.evaluate(principal, tc.name, tc.arguments)
            # Emit shadow audit matches without blocking
            for audit_hit in eval_res.audit_matches:
                if self.telemetry is not None:
                    self.telemetry.record_policy_decision(
                        audit_hit.would_action.value, "audit", audit_hit.rule_id
                    )
                await self.journal.emit(
                    session_id,
                    turn_id,
                    EventType.POLICY_EVALUATED,
                    {
                        "tool_name": tc.name,
                        "decision": "audit_shadow",
                        "would_action": audit_hit.would_action.value,
                        "mode": "audit",
                        "rule_id": audit_hit.rule_id,
                        "reason": audit_hit.reason,
                    },
                )

            if eval_res.matched_rule_id is not None:
                if self.telemetry is not None:
                    self.telemetry.record_policy_decision(
                        eval_res.action.value, eval_res.mode.value, eval_res.matched_rule_id
                    )
                await self.journal.emit(
                    session_id,
                    turn_id,
                    EventType.POLICY_EVALUATED,
                    {
                        "tool_name": tc.name,
                        "decision": eval_res.action.value,
                        "mode": eval_res.mode.value,
                        "rule_id": eval_res.matched_rule_id,
                        "reason": eval_res.reason,
                        "required_approver_roles": eval_res.required_approver_roles,
                    },
                )

            if eval_res.action.value == "deny":
                res = ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=(
                        f"[Policy Denied: {eval_res.matched_rule_id or 'default'}] "
                        f"{eval_res.reason or 'Blocked by enterprise access control policy.'}"
                    ),
                    is_error=True,
                    error_type="policy_denied",
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )
                return await _finalize_and_emit(res)

            if eval_res.action.value == "ask":
                policy_requires_approval = True
                policy_required_roles = eval_res.required_approver_roles
                policy_rule_id = eval_res.matched_rule_id
                policy_reason = eval_res.reason

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

        # 5. Human-in-the-Loop governance gate (triggered by static HIGH risk OR dynamic ACP `ASK`)
        if reg.spec.risk_level == RiskLevel.HIGH or policy_requires_approval:
            req = self.approval_broker.create_request(
                session_id=session_id,
                turn_id=turn_id,
                tool_call=tc,
                reason=policy_reason
                or f"Tool '{tc.name}' is marked HIGH risk and requires explicit approval.",
                requester_id=principal.user_id if principal else "anonymous",
                tenant_id=principal.tenant_id if principal else "default",
                required_roles=policy_required_roles,
                policy_rule_id=policy_rule_id,
            )
            if self.telemetry is not None:
                self.telemetry.set_active_approvals(len(self.approval_broker.list_pending()))
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.APPROVAL_REQUIRED,
                {
                    "approval_id": req.approval_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "reason": req.reason,
                    "requester_id": req.requester_id,
                    "tenant_id": req.tenant_id,
                    "required_roles": req.required_roles,
                    "policy_rule_id": req.policy_rule_id,
                },
            )
            try:
                decision = await self.approval_broker.wait_for_decision(
                    req=req,
                    cancellation=cancellation,
                    timeout_seconds=self.config.approval_timeout_seconds,
                )
            finally:
                if self.telemetry is not None:
                    self.telemetry.set_active_approvals(len(self.approval_broker.list_pending()))
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
                return await _finalize_and_emit(res)

        # 6. Bounded parallel execution with timeout, telemetry span, and optional transient retries
        attempts = reg.spec.max_retries + 1
        last_err: Exception | None = None
        span_cm = (
            self.telemetry.start_span(
                "gen_ai.execute_tool",
                {
                    "operation": "execute_tool",
                    "tool_name": tc.name,
                    "tool_call_id": tc.id,
                    "pyharness.tenant_id": principal.tenant_id if principal else "default",
                },
            )
            if self.telemetry is not None
            else nullcontext()
        )

        with span_cm:
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
                        return await _finalize_and_emit(res)
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
        return await _finalize_and_emit(res)

    @staticmethod
    def _build_turn_view_messages(
        messages: list[Message],
        turn_system_instructions: str | None,
    ) -> list[Message]:
        """
        Overlay turn-scoped dynamic Skill instructions onto a transient view of `messages`
        so `messages[0]` remains clean across future dialogue turns.
        """
        if not turn_system_instructions:
            return list(messages)
        view = list(messages)
        if view and view[0].role == "system":
            view[0] = view[0].model_copy(
                update={"content": f"{view[0].content}\n\n{turn_system_instructions}"}
            )
        else:
            view.insert(0, Message(role="system", content=turn_system_instructions))
        return view

    async def _stream_llm_step(
        self,
        session_id: str,
        turn_id: str,
        step: int,
        llm_messages: list[Message],
        available_tools: list[ToolSpec],
        cancellation: CancellationToken,
    ) -> tuple[str, str, list[ToolCall], TokenUsage, bool, str]:
        """Consume `self.llm.stream(...)` live while racing every chunk against `CancellationToken`."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        step_usage = TokenUsage()
        emitted_token_delta = False
        seen_model = ""

        principal = current_principal.get()
        span_cm = (
            self.telemetry.start_span(
                "gen_ai.chat",
                {
                    "operation": "chat",
                    "pyharness.session_id": session_id,
                    "pyharness.step": step,
                    "pyharness.tenant_id": principal.tenant_id if principal else "default",
                },
            )
            if self.telemetry is not None
            else nullcontext()
        )

        with span_cm as span:
            ait = self.llm.stream(llm_messages, tools=available_tools).__aiter__()
            try:
                while True:
                    try:
                        chunk = await self._await_with_cancellation(
                            ait.__anext__(), cancellation=cancellation
                        )
                    except StopAsyncIteration:
                        break

                    if chunk.model:
                        seen_model = chunk.model

                    if chunk.reasoning_delta:
                        reasoning_parts.append(chunk.reasoning_delta)
                        await self.journal.emit(
                            session_id,
                            turn_id,
                            EventType.THINKING,
                            {"step": step, "reasoning": chunk.reasoning_delta, "delta": True},
                        )

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                    if chunk.content_delta:
                        content_parts.append(chunk.content_delta)
                        if not chunk.is_final and not tool_calls:
                            emitted_token_delta = True
                            await self.journal.emit(
                                session_id,
                                turn_id,
                                EventType.TOKEN_DELTA,
                                {"content": chunk.content_delta, "delta": True},
                            )

                    if chunk.usage is not None:
                        step_usage = chunk.usage
            finally:
                aclose_fn = getattr(ait, "aclose", None)
                if callable(aclose_fn):
                    res = aclose_fn()
                    if inspect.isawaitable(res):
                        await res

            if span is not None and hasattr(span, "set_attribute"):
                if seen_model:
                    span.set_attribute("gen_ai.response.model", seen_model)
                span.set_attribute("gen_ai.usage.input_tokens", step_usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", step_usage.completion_tokens)

        if self.telemetry is not None and step_usage.total_tokens > 0:
            self.telemetry.record_tokens(
                tenant_id=principal.tenant_id if principal else "default",
                model=seen_model or "unknown",
                prompt_tokens=step_usage.prompt_tokens,
                completion_tokens=step_usage.completion_tokens,
                cached_tokens=step_usage.prompt_cache_hit_tokens,
                cost_usd=step_usage.estimated_cost_usd,
            )

        return (
            "".join(content_parts),
            "".join(reasoning_parts),
            tool_calls,
            step_usage,
            emitted_token_delta,
            seen_model,
        )

    async def run_turn(
        self,
        session_id: str,
        user_input: str,
        history: list[Message] | None = None,
        system_prompt: str = "You are PyHarness, an enterprise AI assistant with governed tool use.",
        cancellation: CancellationToken | None = None,
        turn_system_instructions: str | None = None,
        allowed_tools: set[str] | None = None,
    ) -> TurnResult:
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        cancel = cancellation or CancellationToken()
        budget = TokenBudgetGuard(max_tokens_per_turn=self.config.max_tokens_per_turn)
        semaphore = asyncio.Semaphore(self.config.max_parallel_tools)
        principal = current_principal.get()
        tenant_id = principal.tenant_id if principal else "default"
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

        sub_ctx = (
            self.subagent_manager.bind_parent_context(
                session_id=session_id,
                turn_id=turn_id,
                cancellation=cancel,
                allowed_tools=allowed_tools,
                parent_engine=self,
            )
            if self.subagent_manager is not None
            else nullcontext()
        )
        turn_span = (
            self.telemetry.start_span(
                "pyharness.turn",
                {
                    "pyharness.session_id": session_id,
                    "pyharness.turn_id": turn_id,
                    "pyharness.tenant_id": tenant_id,
                },
            )
            if self.telemetry is not None
            else nullcontext()
        )

        def _finish(result: TurnResult) -> TurnResult:
            if self.telemetry is not None:
                self.telemetry.record_turn(
                    result.status.value, tenant_id, result.total_duration_ms
                )
            return result

        with turn_span, sub_ctx:
            try:
                for step in range(1, self.config.max_steps + 1):
                    cancel.raise_if_cancelled()
                    steps = step

                    # Step A: Context window guard & incremental (or LLM-assisted) compression
                    if self.compressor.should_compress(messages):
                        messages, stats = await self.compressor.acompress(
                            messages, summary_llm=self.summary_llm
                        )
                        if stats["messages_folded"] > 0:
                            await self.journal.emit(
                                session_id,
                                turn_id,
                                EventType.CONTEXT_COMPRESSED,
                                stats,
                            )

                    # Step B: Build per-turn transient prompt view + tool scope + run hooks
                    llm_messages = self._build_turn_view_messages(
                        messages, turn_system_instructions
                    )
                    available_tools = self.tools.list_specs()
                    if self.config.enforce_skill_tool_scope and allowed_tools is not None:
                        available_tools = [t for t in available_tools if t.name in allowed_tools]

                    for hook in list(self._hooks):
                        llm_messages, available_tools = await hook.before_llm(
                            session_id, turn_id, step, llm_messages, available_tools
                        )

                    # Step C: Stream from LLM (raced against CancellationToken)
                    (
                        full_content,
                        _full_reasoning,
                        tool_calls,
                        step_usage,
                        emitted_token_delta,
                        _seen_model,
                    ) = await self._stream_llm_step(
                        session_id=session_id,
                        turn_id=turn_id,
                        step=step,
                        llm_messages=llm_messages,
                        available_tools=available_tools,
                        cancellation=cancel,
                    )
                    budget.record(step_usage)

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
                        return _finish(
                            TurnResult(
                                session_id=session_id,
                                turn_id=turn_id,
                                status=TurnStatus.BUDGET_EXCEEDED,
                                final_answer=full_content,
                                steps_executed=steps,
                                usage=budget.usage,
                                total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                                error_message=msg,
                                messages=messages,
                            )
                        )

                    # Step D: Final answer (no tool calls)
                    if not tool_calls:
                        messages.append(Message(role="assistant", content=full_content))
                        if not emitted_token_delta:
                            await self.journal.emit(
                                session_id,
                                turn_id,
                                EventType.TOKEN_DELTA,
                                {"content": full_content},
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
                        return _finish(
                            TurnResult(
                                session_id=session_id,
                                turn_id=turn_id,
                                status=TurnStatus.COMPLETED,
                                final_answer=full_content,
                                steps_executed=steps,
                                usage=budget.usage,
                                total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                                messages=messages,
                            )
                        )

                    # Step E: Execute all tool calls in this turn concurrently (bounded by Semaphore)
                    messages.append(
                        Message(
                            role="assistant",
                            content=full_content,
                            tool_calls=tool_calls,
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
                                allowed_tools=allowed_tools,
                            )
                            for tc in tool_calls
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

                    # Step F: Self-Correction loop accounting & breaker
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
                            return _finish(
                                TurnResult(
                                    session_id=session_id,
                                    turn_id=turn_id,
                                    status=TurnStatus.ERROR_LIMIT_EXCEEDED,
                                    steps_executed=steps,
                                    usage=budget.usage,
                                    total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                                    error_message=err_msg,
                                    messages=messages,
                                )
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
                return _finish(
                    TurnResult(
                        session_id=session_id,
                        turn_id=turn_id,
                        status=TurnStatus.MAX_STEPS_EXCEEDED,
                        steps_executed=steps,
                        usage=budget.usage,
                        total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                        error_message=msg,
                        messages=messages,
                    )
                )

            except TurnCancelledError as exc:
                await self.journal.emit(
                    session_id,
                    turn_id,
                    EventType.TURN_CANCELLED,
                    {"reason": str(exc), "steps": steps},
                )
                return _finish(
                    TurnResult(
                        session_id=session_id,
                        turn_id=turn_id,
                        status=TurnStatus.CANCELLED,
                        steps_executed=steps,
                        usage=budget.usage,
                        total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                        error_message=str(exc),
                        messages=messages,
                    )
                )
