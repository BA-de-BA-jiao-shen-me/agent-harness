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
from pyharness.runtime.interaction import (
    ActiveTurnContext,
    TurnSteeringQueue,
    use_turn_context,
)
from pyharness.runtime.llm_client import BaseLLMClient, _supports_kwarg
from pyharness.runtime.memory import (
    ProjectMemoryStore,
    TrustedTemporalContext,
    current_project_id,
    use_project_id,
)
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
        return sorted((item.spec for item in self._tools.values()), key=lambda s: s.name)

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


DEFAULT_META_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user_question",
        "read_artifact",
        "list_artifacts",
        "remember_fact",
        "recall_memory",
    }
)


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
    preserve_prompt_cache: bool = True
    enable_length_recovery: bool = True
    max_length_recoveries: int = 1
    always_allowed_meta_tools: frozenset[str] = DEFAULT_META_TOOLS


class AgentHarnessEngine:
    """
    Deterministic Enterprise Agent Harness Loop (DSH L4 Runtime).

    Key enterprise guarantees:
    - True incremental token & reasoning streaming (`llm.stream`) raced against `CancellationToken`.
    - KV Prefix Cache-optimal transient dynamic context injection (`preserve_prompt_cache=True`),
      deterministic sorted tool prefix, `prompt_cache_key`, and dual-layer `allowed_tools` scoping.
    - Trusted temporal anchors, automatic long-term project memory recall, non-destructive
      compaction anchors, output-length truncation auto-recovery, and mid-turn steering / follow-up queues.
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
        steering_queue: TurnSteeringQueue | None = None,
        memory_store: ProjectMemoryStore | None = None,
        default_timezone: str | None = None,
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
        self.steering_queue = steering_queue
        self.memory_store = memory_store
        self.default_timezone = default_timezone
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

        # 2. Enforce active Skill `allowed_tools` least-privilege scope (always permitting built-in LOW-risk meta-tools)
        if (
            self.config.enforce_skill_tool_scope
            and allowed_tools is not None
            and tc.name not in allowed_tools
            and tc.name not in self.config.always_allowed_meta_tools
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
                    if (
                        self.steering_queue is not None
                        and self.steering_queue.has_steer(session_id)
                    ):
                        res = ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content="[Skipped due to user mid-turn steering]",
                            is_error=False,
                            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                            metadata={"skipped_by_steering": True},
                        )
                        return await _finalize_and_emit(res)
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
    def _extract_compaction_summary(messages: list[Message]) -> str:
        prefix = "[Incremental Context Summary]"
        for msg in messages:
            if msg.role == "system" and msg.content.startswith(prefix):
                body = msg.content[len(prefix) :].lstrip("\r\n").strip()
                return body or msg.content.strip()
        return ""

    @staticmethod
    def _build_turn_view_messages(
        messages: list[Message],
        turn_system_instructions: str | None,
        *,
        preserve_prompt_cache: bool = True,
        tail_context_blocks: list[str] | None = None,
    ) -> list[Message]:
        """
        Overlay turn-scoped dynamic instructions & tail context onto a transient view of `messages`
        without mutating persistent history.

        - `preserve_prompt_cache=True` (v0.4.0 default): keeps `view[0]` (system prompt) 100%
          byte-identical across turns and steps for optimal KV prefix caching, injecting dynamic
          context onto the transient copy of the last `user` message.
        - `preserve_prompt_cache=False` (legacy): concatenates `turn_system_instructions` onto
          `view[0]` and appends any `tail_context_blocks` onto the transient last `user` message.
        """
        view = list(messages)
        clean_tail_blocks = [
            block.strip()
            for block in (tail_context_blocks or [])
            if block and block.strip()
        ]
        clean_turn_inst = (
            turn_system_instructions.strip()
            if turn_system_instructions and turn_system_instructions.strip()
            else ""
        )

        if not preserve_prompt_cache:
            if turn_system_instructions:
                if view and view[0].role == "system":
                    view[0] = view[0].model_copy(
                        update={
                            "content": f"{view[0].content}\n\n{turn_system_instructions}"
                        }
                    )
                else:
                    view.insert(
                        0, Message(role="system", content=turn_system_instructions)
                    )
            if clean_tail_blocks:
                tail_overlay = "\n\n".join(clean_tail_blocks)
                last_user_idx = next(
                    (i for i in range(len(view) - 1, -1, -1) if view[i].role == "user"),
                    None,
                )
                if last_user_idx is not None:
                    orig_user = view[last_user_idx]
                    view[last_user_idx] = orig_user.model_copy(
                        update={"content": f"{orig_user.content}\n\n{tail_overlay}"}
                    )
                else:
                    view.append(Message(role="user", content=tail_overlay))
            return view

        overlay_parts: list[str] = []
        if clean_turn_inst:
            overlay_parts.append(clean_turn_inst)
        overlay_parts.extend(clean_tail_blocks)
        if not overlay_parts:
            return view

        overlay = "\n\n".join(overlay_parts)
        last_user_idx = next(
            (i for i in range(len(view) - 1, -1, -1) if view[i].role == "user"),
            None,
        )
        if last_user_idx is not None:
            orig_user = view[last_user_idx]
            view[last_user_idx] = orig_user.model_copy(
                update={
                    "content": (
                        f"[Turn Dynamic Context]\n{overlay}\n\n"
                        f"[User Request]\n{orig_user.content}"
                    )
                }
            )
        else:
            view.append(
                Message(role="user", content=f"[Turn Dynamic Context]\n{overlay}")
            )
        return view

    async def _stream_llm_step(
        self,
        session_id: str,
        turn_id: str,
        step: int,
        llm_messages: list[Message],
        available_tools: list[ToolSpec],
        cancellation: CancellationToken,
        prompt_cache_key: str | None = None,
    ) -> tuple[str, str, list[ToolCall], TokenUsage, bool, str, str]:
        """Consume `self.llm.stream(...)` live while racing every chunk against `CancellationToken`."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        step_usage = TokenUsage()
        emitted_token_delta = False
        seen_model = ""
        finish_reason = "stop"

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
            if prompt_cache_key and _supports_kwarg(self.llm.stream, "prompt_cache_key"):
                ait = self.llm.stream(
                    llm_messages,
                    tools=available_tools,
                    prompt_cache_key=prompt_cache_key,
                ).__aiter__()
            else:
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

                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason

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
            finish_reason,
        )

    async def _compress_and_anchor(
        self,
        session_id: str,
        turn_id: str,
        messages: list[Message],
    ) -> list[Message]:
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
            summary_body = self._extract_compaction_summary(messages)
            await self.journal.emit(
                session_id,
                turn_id,
                EventType.COMPACTION_ANCHOR,
                {"summary": summary_body, **stats},
            )
        return messages

    async def run_turn(
        self,
        session_id: str,
        user_input: str,
        history: list[Message] | None = None,
        system_prompt: str = "You are PyHarness, an enterprise AI assistant with governed tool use.",
        cancellation: CancellationToken | None = None,
        turn_system_instructions: str | None = None,
        allowed_tools: set[str] | None = None,
        timezone_name: str | None = None,
        project_id: str | None = None,
    ) -> TurnResult:
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        cancel = cancellation or CancellationToken()
        budget = TokenBudgetGuard(max_tokens_per_turn=self.config.max_tokens_per_turn)
        semaphore = asyncio.Semaphore(self.config.max_parallel_tools)
        principal = current_principal.get()
        tenant_id = principal.tenant_id if principal else "default"
        user_id = principal.user_id if principal else "anonymous"
        eff_project_id = (project_id or "").strip() or current_project_id.get() or "default"
        prompt_cache_key = f"pyharness:{tenant_id}:{session_id}"
        t_start = time.perf_counter()

        turn_tail_blocks: list[str] = []
        eff_tz = timezone_name if timezone_name is not None else self.default_timezone
        if eff_tz:
            temporal = TrustedTemporalContext.capture(eff_tz)
            turn_tail_blocks.append(temporal.render_prompt_block())

        if self.memory_store is not None:
            memories = await self.memory_store.recall(
                user_input,
                tenant_id=tenant_id,
                user_id=user_id,
                project_id=eff_project_id,
                top_k=3,
            )
            if memories:
                mem_block = ProjectMemoryStore.render_prompt_block(memories)
                if mem_block:
                    turn_tail_blocks.append(mem_block)

        if (
            self.config.enforce_skill_tool_scope
            and allowed_tools is not None
            and self.config.preserve_prompt_cache
        ):
            registered_meta = {
                t for t in self.config.always_allowed_meta_tools if self.tools.get(t) is not None
            }
            eff_scope = sorted(allowed_tools | registered_meta)
            turn_tail_blocks.append(
                f"[Active Skill Tool Scope] For this turn, only invoke tools in: "
                f"{', '.join(eff_scope)}."
            )

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
        length_recoveries = 0
        length_continuation_parts: list[str] = []

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
        turn_ctx_cm = use_turn_context(
            ActiveTurnContext(
                session_id=session_id,
                turn_id=turn_id,
                cancellation=cancel,
                owner_user_id=user_id,
                tenant_id=tenant_id,
            )
        )
        project_ctx_cm = use_project_id(eff_project_id)

        def _finish(result: TurnResult) -> TurnResult:
            if self.telemetry is not None:
                self.telemetry.record_turn(
                    result.status.value, tenant_id, result.total_duration_ms
                )
            return result

        with turn_span, sub_ctx, turn_ctx_cm, project_ctx_cm:
            try:
                step = 0
                step_limit = self.config.max_steps
                step_ceiling = self.config.max_steps * 4
                while step < step_limit:
                    step += 1
                    cancel.raise_if_cancelled()
                    steps = step

                    # Step A: Context window guard & incremental (or LLM-assisted) compression
                    if self.compressor.should_compress(messages):
                        messages = await self._compress_and_anchor(
                            session_id, turn_id, messages
                        )

                    # Step B: Build per-turn transient prompt view + tool scope + run hooks
                    llm_messages = self._build_turn_view_messages(
                        messages,
                        turn_system_instructions,
                        preserve_prompt_cache=self.config.preserve_prompt_cache,
                        tail_context_blocks=turn_tail_blocks,
                    )
                    available_tools = self.tools.list_specs()
                    if (
                        self.config.enforce_skill_tool_scope
                        and allowed_tools is not None
                        and not self.config.preserve_prompt_cache
                    ):
                        available_tools = [
                            t
                            for t in available_tools
                            if t.name in allowed_tools
                            or t.name in self.config.always_allowed_meta_tools
                        ]

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
                        finish_reason,
                    ) = await self._stream_llm_step(
                        session_id=session_id,
                        turn_id=turn_id,
                        step=step,
                        llm_messages=llm_messages,
                        available_tools=available_tools,
                        cancellation=cancel,
                        prompt_cache_key=prompt_cache_key,
                    )
                    budget.record(step_usage)

                    if budget.is_exceeded:
                        combined_answer = "".join([*length_continuation_parts, full_content])
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
                                final_answer=combined_answer,
                                steps_executed=steps,
                                usage=budget.usage,
                                total_duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                                error_message=msg,
                                messages=messages,
                            )
                        )

                    # Step C2: Output length truncation auto-recovery
                    if (
                        finish_reason == "length"
                        and not tool_calls
                        and self.config.enable_length_recovery
                        and length_recoveries < self.config.max_length_recoveries
                    ):
                        length_recoveries += 1
                        if full_content:
                            length_continuation_parts.append(full_content)
                        messages.append(Message(role="assistant", content=full_content))
                        if not emitted_token_delta and full_content:
                            await self.journal.emit(
                                session_id,
                                turn_id,
                                EventType.TOKEN_DELTA,
                                {"content": full_content},
                            )
                        if (
                            self.compressor.should_compress(messages)
                            or len(messages) > self.config.keep_recent_messages + 2
                        ):
                            messages = await self._compress_and_anchor(
                                session_id, turn_id, messages
                            )
                        messages.append(
                            Message(
                                role="user",
                                content=(
                                    "Your previous response was truncated due to output length limit. "
                                    "Continue concisely from where you stopped without repeating earlier content."
                                ),
                            )
                        )
                        step_limit = min(step_ceiling, max(step_limit, step + 1))
                        continue

                    # Step D: Final answer (no tool calls)
                    if not tool_calls:
                        # 1. Check late mid-turn steer before completing
                        if (
                            self.steering_queue is not None
                            and self.steering_queue.has_steer(session_id)
                        ):
                            messages.append(Message(role="assistant", content=full_content))
                            if not emitted_token_delta and full_content:
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.TOKEN_DELTA,
                                    {"content": full_content},
                                )
                            steers = self.steering_queue.pop_all_steers(session_id)
                            for st in steers:
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.TURN_STEERED,
                                    {
                                        "message_id": st.message_id,
                                        "content": st.content,
                                        "step": step,
                                    },
                                )
                                messages.append(Message(role="user", content=st.content))
                            length_continuation_parts.clear()
                            consecutive_errors = 0
                            step_limit = min(step_ceiling, max(step_limit, step + 2))
                            continue

                        # 2. Check queued post-turn follow-ups before completing
                        if (
                            self.steering_queue is not None
                            and self.steering_queue.has_followup(session_id)
                        ):
                            messages.append(Message(role="assistant", content=full_content))
                            if not emitted_token_delta and full_content:
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.TOKEN_DELTA,
                                    {"content": full_content},
                                )
                            fu = self.steering_queue.pop_followup(session_id)
                            if fu is not None:
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.FOLLOWUP_ENQUEUED,
                                    {"message_id": fu.message_id, "content": fu.content},
                                )
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.TOKEN_DELTA,
                                    {"content": "\n\n---\n\n"},
                                )
                                messages.append(Message(role="user", content=fu.content))
                                length_continuation_parts.clear()
                                length_recoveries = 0
                                consecutive_errors = 0
                                step_limit = min(step_ceiling, step + self.config.max_steps)
                                continue

                        combined_answer = "".join([*length_continuation_parts, full_content])
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
                                final_answer=combined_answer,
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

                    # Drain mid-turn steering directives queued during tool execution
                    if self.steering_queue is not None:
                        steers = self.steering_queue.pop_all_steers(session_id)
                        if steers:
                            for st in steers:
                                await self.journal.emit(
                                    session_id,
                                    turn_id,
                                    EventType.TURN_STEERED,
                                    {
                                        "message_id": st.message_id,
                                        "content": st.content,
                                        "step": step,
                                    },
                                )
                                messages.append(Message(role="user", content=st.content))
                            consecutive_errors = 0
                            step_limit = min(step_ceiling, max(step_limit, step + 2))
                            continue

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
