from __future__ import annotations

import inspect
import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from pyharness.runtime.engine import (
    AgentHarnessEngine,
    HarnessConfig,
    HarnessHook,
    ToolRegistry,
)
from pyharness.runtime.event_journal import EventJournal
from pyharness.runtime.governance import ApprovalBroker, CancellationToken
from pyharness.runtime.llm_client import BaseLLMClient
from pyharness.runtime.schemas import EventType, Message, RiskLevel, ToolSpec

_subagent_depth: ContextVar[int] = ContextVar("pyharness_subagent_depth", default=0)
_parent_session_id: ContextVar[str | None] = ContextVar(
    "pyharness_subagent_parent_session_id", default=None
)
_parent_turn_id: ContextVar[str | None] = ContextVar(
    "pyharness_subagent_parent_turn_id", default=None
)
_parent_cancellation: ContextVar[CancellationToken | None] = ContextVar(
    "pyharness_subagent_parent_cancellation", default=None
)
_parent_allowed_tools: ContextVar[set[str] | frozenset[str] | None] = ContextVar(
    "pyharness_subagent_parent_allowed_tools", default=None
)
_parent_denied_tools: ContextVar[set[str] | frozenset[str] | None] = ContextVar(
    "pyharness_subagent_parent_denied_tools", default=None
)
_parent_engine_ctx: ContextVar[AgentHarnessEngine | None] = ContextVar(
    "pyharness_subagent_parent_engine", default=None
)


@dataclass(frozen=True)
class SubagentProfile:
    """Declarative capability, prompt, budget, and risk contract for a specialized subagent."""

    name: str
    description: str
    system_prompt: str
    allowed_tools: frozenset[str]
    max_steps: int = 5
    model_tier: str = "fast"
    allow_high_risk: bool = False


DEFAULT_SUBAGENT_PROFILES: dict[str, SubagentProfile] = {
    "researcher": SubagentProfile(
        name="researcher",
        description="Read-only enterprise knowledge base and operations incident investigator.",
        system_prompt=(
            "You are the Enterprise Researcher subagent. Investigate safety manuals, SOPs, "
            "and active operational incidents using read-only search tools. Cite concrete facts "
            "and return a concise, structured summary for the parent coordinator."
        ),
        allowed_tools=frozenset({"rag_search", "mcp_list_incidents"}),
        max_steps=5,
        model_tier="fast",
        allow_high_risk=False,
    ),
    "analyst": SubagentProfile(
        name="analyst",
        description="Structured SQL analytics and isolated Python statistical computation specialist.",
        system_prompt=(
            "You are the Enterprise Data Analyst subagent. Inspect database schemas, execute "
            "read-only SQL aggregations, and run isolated Python calculations to answer "
            "quantitative questions accurately and concisely."
        ),
        allowed_tools=frozenset(
            {"inspect_sql_schema", "query_sql_readonly", "run_python_sandbox"}
        ),
        max_steps=5,
        model_tier="fast",
        allow_high_risk=False,
    ),
}


class SubagentManager:
    """
    DSH-inspired `ctx.subagents` delegation seam.

    Features:
    - Isolated child `AgentHarnessEngine` sessions sharing durable `EventJournal`,
      `ToolRegistry`, and `ApprovalBroker`.
    - Parent context propagation via `ContextVar`s (session lineage, turn lineage,
      cooperative `CancellationToken`, tool-scope intersection, and recursion depth).
    - Strict privilege-escalation guards:
      1. Enforces hard `max_depth` recursion limit and strips `delegate_subagent` at max depth.
      2. Blocks `RiskLevel.HIGH` tools inside subagents unless `profile.allow_high_risk=True`.
      3. Intersects parent `allowed_tools` when the parent turn is explicitly scoped without
         delegation wildcarding, and honors explicit `extra_denied_tools` / RBAC denials.
    - Supports both `one_shot` ephemeral subtasks and `continuable` multi-turn sub-sessions.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        journal: EventJournal,
        approval_broker: ApprovalBroker,
        profiles: dict[str, SubagentProfile] | None = None,
        max_depth: int = 2,
        max_tokens_per_subtask: int = 12000,
        hooks: list[HarnessHook] | None = None,
        policy_engine: Any | None = None,
        telemetry: Any | None = None,
        parent_engine_getter: Callable[[], AgentHarnessEngine | None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.journal = journal
        self.approval_broker = approval_broker
        self.profiles: dict[str, SubagentProfile] = (
            dict(profiles) if profiles is not None else dict(DEFAULT_SUBAGENT_PROFILES)
        )
        self.max_depth = max_depth
        self.max_tokens_per_subtask = max_tokens_per_subtask
        self.hooks: list[HarnessHook] = list(hooks) if hooks else []
        self.policy_engine = policy_engine
        self.telemetry = telemetry
        self._parent_engine: AgentHarnessEngine | None = None
        self._parent_engine_getter = parent_engine_getter
        self._continuable_histories: dict[str, list[Message]] = {}

    def attach_parent_engine(self, engine: AgentHarnessEngine) -> None:
        """Attach the parent engine so child subagents inherit PolicyEngine, hooks, and telemetry."""
        self._parent_engine = engine

    def _resolve_parent_engine(self) -> AgentHarnessEngine | None:
        ctx_engine = _parent_engine_ctx.get()
        if ctx_engine is not None:
            return ctx_engine
        if self._parent_engine is not None:
            return self._parent_engine
        if self._parent_engine_getter is not None:
            return self._parent_engine_getter()
        return None

    def _collect_inherited_governance(
        self,
    ) -> tuple[Any | None, Any | None, list[HarnessHook]]:
        parent_engine = self._resolve_parent_engine()
        policy_engine = (
            self.policy_engine
            if self.policy_engine is not None
            else getattr(parent_engine, "policy_engine", None)
        )
        telemetry = (
            self.telemetry
            if self.telemetry is not None
            else getattr(parent_engine, "telemetry", None)
        )
        inherited_hooks: list[HarnessHook] = []
        seen_ids: set[int] = set()
        for h in (*getattr(parent_engine, "_hooks", []), *self.hooks):
            if id(h) not in seen_ids:
                seen_ids.add(id(h))
                inherited_hooks.append(h)
        return policy_engine, telemetry, inherited_hooks

    def register_profile(self, profile: SubagentProfile) -> None:
        self.profiles[profile.name] = profile

    def list_profiles(self) -> list[SubagentProfile]:
        return list(self.profiles.values())

    def get_continuable_history(self, subagent_session_id: str) -> list[Message] | None:
        hist = self._continuable_histories.get(subagent_session_id)
        return list(hist) if hist is not None else None

    @contextmanager
    def bind_parent_context(
        self,
        session_id: str,
        turn_id: str,
        cancellation: CancellationToken | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
        depth: int | None = None,
        denied_tools: set[str] | frozenset[str] | None = None,
        parent_engine: AgentHarnessEngine | None = None,
    ) -> Iterator[None]:
        """
        Bind the active parent turn execution context so tool-invoked `delegate_subagent`
        inherits session lineage, cancellation token, tool scoping, governance hooks/policies,
        and recursion depth.
        """
        t_sid = _parent_session_id.set(session_id)
        t_tid = _parent_turn_id.set(turn_id)
        t_cancel = _parent_cancellation.set(cancellation)
        t_allowed = _parent_allowed_tools.set(allowed_tools)
        t_denied = _parent_denied_tools.set(denied_tools)
        t_depth = _subagent_depth.set(depth) if depth is not None else None
        t_engine = _parent_engine_ctx.set(parent_engine) if parent_engine is not None else None
        try:
            yield
        finally:
            if t_engine is not None:
                _parent_engine_ctx.reset(t_engine)
            if t_depth is not None:
                _subagent_depth.reset(t_depth)
            _parent_denied_tools.reset(t_denied)
            _parent_allowed_tools.reset(t_allowed)
            _parent_cancellation.reset(t_cancel)
            _parent_turn_id.reset(t_tid)
            _parent_session_id.reset(t_sid)

    def _compute_effective_tools(
        self,
        prof: SubagentProfile,
        current_depth: int,
        extra_denied_tools: set[str] | None = None,
    ) -> set[str]:
        effective = set(prof.allowed_tools)

        # Rule 1: Never permit further delegation once child reaches max_depth
        if current_depth + 1 >= self.max_depth:
            effective.discard("delegate_subagent")

        # Rule 2: Strip HIGH-risk tools unless profile explicitly opts in
        if not prof.allow_high_risk:
            filtered: set[str] = set()
            for name in effective:
                reg = self.tools.get(name)
                if reg is not None and reg.spec.risk_level == RiskLevel.HIGH:
                    continue
                filtered.add(name)
            effective = filtered

        # Rule 3: Intersect with parent scope when parent is restricted without delegation escape
        parent_allowed = _parent_allowed_tools.get()
        if (
            parent_allowed is not None
            and "*" not in parent_allowed
            and "delegate_subagent" not in parent_allowed
        ):
            effective &= set(parent_allowed)

        # Rule 4: Honor explicit deny lists (from parent RBAC policy or caller)
        denied: set[str] = set()
        parent_denied = _parent_denied_tools.get()
        if parent_denied:
            denied.update(parent_denied)
        if extra_denied_tools:
            denied.update(extra_denied_tools)
        effective -= denied

        return effective

    async def delegate(
        self,
        task: str,
        profile: str = "researcher",
        mode: str = "one_shot",
        subagent_session_id: str | None = None,
        extra_denied_tools: set[str] | None = None,
    ) -> str:
        """Execute a scoped child AgentHarnessEngine turn and return compact structured JSON."""
        if mode not in ("one_shot", "continuable"):
            raise ValueError(
                f"Unsupported subagent mode '{mode}'. Expected 'one_shot' or 'continuable'."
            )

        current_depth = _subagent_depth.get()
        if current_depth >= self.max_depth:
            raise PermissionError(
                f"Subagent max recursion depth ({self.max_depth}) exceeded."
            )

        prof = self.profiles.get(profile)
        if prof is None:
            available = ", ".join(sorted(self.profiles.keys()))
            raise ValueError(
                f"Unknown subagent profile '{profile}'. Available profiles: [{available}]."
            )

        parent_sid = _parent_session_id.get() or "root"
        parent_tid = _parent_turn_id.get() or "root_turn"
        parent_cancel = _parent_cancellation.get()

        effective_tools = self._compute_effective_tools(
            prof=prof,
            current_depth=current_depth,
            extra_denied_tools=extra_denied_tools,
        )

        if mode == "continuable" and subagent_session_id:
            child_sid = subagent_session_id
        else:
            child_sid = (
                subagent_session_id
                or f"{parent_sid}::sub_{prof.name}_{uuid.uuid4().hex[:8]}"
            )

        child_history: list[Message] | None = None
        if mode == "continuable" and child_sid in self._continuable_histories:
            child_history = list(self._continuable_histories[child_sid])

        sub_llm = (
            self.llm.for_tier(prof.model_tier)
            if hasattr(self.llm, "for_tier")
            else self.llm
        )

        policy_engine, telemetry, inherited_hooks = self._collect_inherited_governance()

        engine_kwargs: dict[str, Any] = {
            "llm": sub_llm,
            "tools": self.tools,
            "journal": self.journal,
            "approval_broker": self.approval_broker,
            "config": HarnessConfig(
                max_steps=prof.max_steps,
                max_tokens_per_turn=self.max_tokens_per_subtask,
                enforce_skill_tool_scope=True,
            ),
        }
        init_params = inspect.signature(AgentHarnessEngine.__init__).parameters
        if "policy_engine" in init_params and policy_engine is not None:
            engine_kwargs["policy_engine"] = policy_engine
        if "telemetry" in init_params and telemetry is not None:
            engine_kwargs["telemetry"] = telemetry
        if "subagent_manager" in init_params:
            engine_kwargs["subagent_manager"] = self

        child_engine = AgentHarnessEngine(**engine_kwargs)
        # Preserve governance attributes even before Stage 2 Engine __init__ kwargs land
        if policy_engine is not None and getattr(child_engine, "policy_engine", None) is None:
            child_engine.policy_engine = policy_engine  # type: ignore[attr-defined]
        if telemetry is not None and getattr(child_engine, "telemetry", None) is None:
            child_engine.telemetry = telemetry  # type: ignore[attr-defined]
        for hook in inherited_hooks:
            child_engine.add_hook(hook)

        start_evt = getattr(EventType, "SUBAGENT_START", None)
        if start_evt is not None:
            await self.journal.emit(
                parent_sid,
                parent_tid,
                start_evt,
                {
                    "parent_session_id": parent_sid,
                    "subagent_session_id": child_sid,
                    "profile": prof.name,
                    "mode": mode,
                    "depth": current_depth + 1,
                    "task": task[:300],
                    "allowed_tools": sorted(effective_tools),
                },
            )

        merged_denied: set[str] = set(_parent_denied_tools.get() or set())
        if extra_denied_tools:
            merged_denied.update(extra_denied_tools)

        with self.bind_parent_context(
            session_id=child_sid,
            turn_id=parent_tid,
            cancellation=parent_cancel,
            allowed_tools=effective_tools,
            depth=current_depth + 1,
            denied_tools=merged_denied or None,
            parent_engine=child_engine,
        ):
            result = await child_engine.run_turn(
                session_id=child_sid,
                user_input=task,
                history=child_history,
                system_prompt=prof.system_prompt,
                cancellation=parent_cancel,
                allowed_tools=effective_tools,
            )

        if mode == "continuable":
            self._continuable_histories[child_sid] = list(result.messages)

        complete_evt = getattr(EventType, "SUBAGENT_COMPLETE", None)
        if complete_evt is not None:
            await self.journal.emit(
                parent_sid,
                parent_tid,
                complete_evt,
                {
                    "parent_session_id": parent_sid,
                    "subagent_session_id": child_sid,
                    "profile": prof.name,
                    "mode": mode,
                    "depth": current_depth + 1,
                    "status": result.status.value,
                    "steps": result.steps_executed,
                    "usage_tokens": result.usage.total_tokens,
                    "summary": result.final_answer[:300],
                },
            )

        response_payload: dict[str, Any] = {
            "parent_session_id": parent_sid,
            "subagent_session_id": child_sid,
            "profile": prof.name,
            "mode": mode,
            "status": result.status.value,
            "steps": result.steps_executed,
            "usage_tokens": result.usage.total_tokens,
            "summary": result.final_answer,
        }
        return json.dumps(response_payload, ensure_ascii=False)

    def register_tools(self, registry: ToolRegistry) -> None:
        """Mount the `delegate_subagent` tool into a `ToolRegistry`."""
        profile_names = sorted(self.profiles.keys())
        profile_summary = "; ".join(
            f"{name} ({self.profiles[name].description})" for name in profile_names
        )

        async def _delegate_handler(
            task: str,
            profile: str = "researcher",
            mode: str = "one_shot",
            subagent_session_id: str | None = None,
        ) -> str:
            return await self.delegate(
                task=task,
                profile=profile,
                mode=mode,
                subagent_session_id=subagent_session_id,
            )

        registry.register(
            ToolSpec(
                name="delegate_subagent",
                description=(
                    "Delegate a focused subtask to an isolated specialized subagent with "
                    f"least-privilege tool scope. Available profiles: {profile_summary}."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Clear, self-contained instruction for the subagent.",
                        },
                        "profile": {
                            "type": "string",
                            "enum": profile_names,
                            "description": f"Subagent profile ({', '.join(profile_names)}).",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["one_shot", "continuable"],
                            "description": (
                                "'one_shot' runs an isolated task; 'continuable' preserves "
                                "child history under subagent_session_id for follow-up calls."
                            ),
                        },
                        "subagent_session_id": {
                            "type": "string",
                            "description": "Optional child session ID to resume in 'continuable' mode.",
                        },
                    },
                    "required": ["task"],
                },
                risk_level=RiskLevel.LOW,
                timeout_seconds=120.0,
            ),
            _delegate_handler,
        )
