from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AdmissionRejectedError(RuntimeError):
    """Raised when a session turn is rejected due to global or per-tenant concurrency limits."""

    def __init__(
        self,
        reason: str,
        retry_after_seconds: int = 5,
        status_code: int = 503,
    ) -> None:
        self.reason = reason
        self.retry_after_seconds = int(retry_after_seconds)
        self.status_code = int(status_code)
        super().__init__(reason)


class TopicBlockedError(PermissionError):
    """Raised when an incoming message is blocked or redirected by `TopicAdmissionGuard`."""

    def __init__(
        self,
        rule_id: str,
        action: str,
        reason: str = "",
        redirect_message: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.action = action
        self.reason = reason
        self.redirect_message = redirect_message
        message = reason or redirect_message or f"Topic {action}ed by rule '{rule_id}'."
        super().__init__(message)


class SessionAdmissionController:
    """
    Fail-fast concurrency admission controller for agent turns (inspired by MOVO session-admission.mjs).

    Enforces global and per-tenant active turn caps without unbounded queueing, returning
    `AdmissionRejectedError` (HTTP 503 + Retry-After semantics) immediately under saturation.
    """

    def __init__(
        self,
        max_global_active_turns: int = 64,
        max_tenant_active_turns: int = 16,
        retry_after_seconds: int = 5,
    ) -> None:
        if max_global_active_turns < 0:
            raise ValueError("max_global_active_turns must be >= 0.")
        if max_tenant_active_turns < 0:
            raise ValueError("max_tenant_active_turns must be >= 0.")
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be >= 0.")

        self.max_global_active_turns = int(max_global_active_turns)
        self.max_tenant_active_turns = int(max_tenant_active_turns)
        self.retry_after_seconds = int(retry_after_seconds)

        self._active_global: int = 0
        self._active_by_tenant: dict[str, int] = {}
        self._active_sessions: set[str] = set()
        self._session_turn_counts: dict[str, int] = {}
        self._slot_counts: dict[tuple[str, str], int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, session_id: str, tenant_id: str = "default") -> None:
        """Reserve an active turn slot for `(session_id, tenant_id)` or fail fast."""
        norm_tenant = tenant_id or "default"
        async with self._lock:
            if self._active_global >= self.max_global_active_turns:
                raise AdmissionRejectedError(
                    reason=(
                        f"Global active turn limit reached "
                        f"({self._active_global}/{self.max_global_active_turns})."
                    ),
                    retry_after_seconds=self.retry_after_seconds,
                    status_code=503,
                )

            tenant_active = self._active_by_tenant.get(norm_tenant, 0)
            if tenant_active >= self.max_tenant_active_turns:
                raise AdmissionRejectedError(
                    reason=(
                        f"Tenant '{norm_tenant}' active turn limit reached "
                        f"({tenant_active}/{self.max_tenant_active_turns})."
                    ),
                    retry_after_seconds=self.retry_after_seconds,
                    status_code=503,
                )

            self._active_global += 1
            self._active_by_tenant[norm_tenant] = tenant_active + 1
            self._session_turn_counts[session_id] = (
                self._session_turn_counts.get(session_id, 0) + 1
            )
            slot = (session_id, norm_tenant)
            self._slot_counts[slot] = self._slot_counts.get(slot, 0) + 1
            self._active_sessions.add(session_id)

    async def release(self, session_id: str, tenant_id: str = "default") -> None:
        """Release a previously acquired turn slot; idempotent if already released."""
        norm_tenant = tenant_id or "default"
        slot = (session_id, norm_tenant)
        async with self._lock:
            slot_count = self._slot_counts.get(slot, 0)
            if slot_count <= 0:
                return

            if slot_count == 1:
                self._slot_counts.pop(slot, None)
            else:
                self._slot_counts[slot] = slot_count - 1

            if self._active_global > 0:
                self._active_global -= 1

            tenant_active = self._active_by_tenant.get(norm_tenant, 0)
            if tenant_active <= 1:
                self._active_by_tenant.pop(norm_tenant, None)
            else:
                self._active_by_tenant[norm_tenant] = tenant_active - 1

            session_turns = self._session_turn_counts.get(session_id, 0)
            if session_turns <= 1:
                self._session_turn_counts.pop(session_id, None)
                self._active_sessions.discard(session_id)
            else:
                self._session_turn_counts[session_id] = session_turns - 1

    @asynccontextmanager
    async def admit(
        self, session_id: str, tenant_id: str = "default"
    ) -> AsyncIterator[None]:
        """Async context manager that acquires a turn slot on entry and always releases on exit."""
        await self.acquire(session_id, tenant_id=tenant_id)
        try:
            yield
        finally:
            await self.release(session_id, tenant_id=tenant_id)

    def active_sessions(self) -> set[str]:
        """Return a snapshot of currently active session IDs."""
        return set(self._active_sessions)

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of admission controller utilization."""
        return {
            "active_global": self._active_global,
            "max_global_active_turns": self.max_global_active_turns,
            "active_by_tenant": dict(self._active_by_tenant),
            "max_tenant_active_turns": self.max_tenant_active_turns,
            "active_sessions": sorted(self._active_sessions),
            "retry_after_seconds": self.retry_after_seconds,
        }


class TopicAdmissionRule(BaseModel):
    """Declarative topic admission rule supporting keyword substrings and regex matching."""

    rule_id: str
    action: Literal["allow", "block", "redirect"] = "block"
    keywords: list[str] = Field(default_factory=list)
    regex: str | None = None
    tenants: list[str] = Field(default_factory=list)
    reason: str = ""
    redirect_message: str = ""
    enabled: bool = True


class TopicAdmissionDecision(BaseModel):
    """Decision returned by `TopicAdmissionGuard.evaluate()`."""

    allowed: bool
    action: str = "allow"
    rule_id: str | None = None
    reason: str = ""
    redirect_message: str = ""


@lru_cache(maxsize=512)
def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


class TopicAdmissionGuard:
    """
    Pre-turn topic admission filter (inspired by MOVO topic_admission).

    Evaluates enabled rules matching the caller's tenant using case-insensitive keyword
    substring or regex matching. Precedence: explicit `allow` rules first, then
    `block`/`redirect` rules.
    """

    def __init__(self, rules: list[TopicAdmissionRule] | None = None) -> None:
        self.rules: list[TopicAdmissionRule] = list(rules or [])

    @classmethod
    def from_yaml(cls, path: str | Path) -> TopicAdmissionGuard:
        """Load topic admission rules from a UTF-8 YAML file."""
        rule_path = Path(path)
        raw_text = rule_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
        if data is None:
            return cls(rules=[])
        if isinstance(data, list):
            raw_rules = data
        elif isinstance(data, dict):
            raw_rules = data.get("rules", [])
            if not isinstance(raw_rules, list):
                raise ValueError(
                    f"Topic admission file {rule_path} 'rules' field must be a list."
                )
        else:
            raise ValueError(
                f"Topic admission file {rule_path} must contain a YAML mapping or list."
            )
        rules = [TopicAdmissionRule.model_validate(item) for item in raw_rules]
        return cls(rules=rules)

    @staticmethod
    def _rule_matches(
        rule: TopicAdmissionRule,
        message: str,
        lowered_message: str,
        tenant_id: str,
    ) -> bool:
        if not rule.enabled:
            return False
        if rule.tenants and tenant_id not in rule.tenants and "*" not in rule.tenants:
            return False

        matched_keyword = any(
            kw and kw.lower() in lowered_message for kw in rule.keywords
        )
        if matched_keyword:
            return True

        if rule.regex:
            if _compile_regex(rule.regex).search(message) is not None:
                return True

        return False

    def evaluate(
        self, message: str, *, tenant_id: str = "default"
    ) -> TopicAdmissionDecision:
        """Evaluate all enabled rules for `tenant_id`; explicit `allow` takes precedence."""
        text = message if isinstance(message, str) else str(message)
        lowered = text.lower()
        norm_tenant = tenant_id or "default"

        first_deny_rule: TopicAdmissionRule | None = None

        for rule in self.rules:
            if not self._rule_matches(rule, text, lowered, norm_tenant):
                continue

            if rule.action == "allow":
                return TopicAdmissionDecision(
                    allowed=True,
                    action="allow",
                    rule_id=rule.rule_id,
                    reason=rule.reason,
                    redirect_message="",
                )

            if first_deny_rule is None:
                first_deny_rule = rule

        if first_deny_rule is not None:
            return TopicAdmissionDecision(
                allowed=False,
                action=first_deny_rule.action,
                rule_id=first_deny_rule.rule_id,
                reason=first_deny_rule.reason,
                redirect_message=first_deny_rule.redirect_message,
            )

        return TopicAdmissionDecision(
            allowed=True,
            action="allow",
            rule_id=None,
            reason="",
            redirect_message="",
        )

    def enforce(
        self, message: str, *, tenant_id: str = "default"
    ) -> TopicAdmissionDecision:
        """Evaluate topic admission and raise `TopicBlockedError` if not allowed."""
        decision = self.evaluate(message, tenant_id=tenant_id)
        if not decision.allowed:
            raise TopicBlockedError(
                rule_id=decision.rule_id or "",
                action=decision.action,
                reason=decision.reason,
                redirect_message=decision.redirect_message,
            )
        return decision


class RuntimeIdleReaper:
    """
    Periodic background idle state reaper (inspired by MOVO runtime-idle-reaper.mjs).

    Evicts inactive session histories, stale steering queues, and expired/revoked
    credential relay leases while never evicting currently active sessions. Safe to
    register `stop()` directly with `PluginContext.effect()`.
    """

    def __init__(
        self,
        session_histories: dict[str, Any],
        *,
        idle_ttl_seconds: float = 1800.0,
        sweep_interval_seconds: float = 60.0,
        steering_queue: Any | None = None,
        credential_relay: Any | None = None,
        active_sessions_getter: Callable[[], set[str]] | None = None,
    ) -> None:
        self._session_histories = session_histories
        self.idle_ttl_seconds = float(idle_ttl_seconds)
        self.sweep_interval_seconds = float(sweep_interval_seconds)
        self._steering_queue = steering_queue
        self._credential_relay = credential_relay
        self._active_sessions_getter = active_sessions_getter

        self._last_touched: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def is_running(self) -> bool:
        """Return True if the background sweep task is currently active."""
        return self._task is not None and not self._task.done()

    def touch(self, session_id: str, now: float | None = None) -> None:
        """Record recent activity for `session_id` to reset its idle expiration clock."""
        self._last_touched[session_id] = (
            time.time() if now is None else float(now)
        )

    def reap_once(self, now: float | None = None) -> dict[str, int]:
        """Run a single synchronous eviction sweep across histories, steering, and leases."""
        current_time = time.time() if now is None else float(now)
        active: set[str] = (
            set(self._active_sessions_getter())
            if self._active_sessions_getter is not None
            else set()
        )

        sessions_evicted = 0
        for sid in list(self._session_histories.keys()):
            if sid not in self._last_touched:
                self._last_touched[sid] = current_time
            if sid in active:
                continue
            if (current_time - self._last_touched[sid]) >= self.idle_ttl_seconds:
                self._session_histories.pop(sid, None)
                self._last_touched.pop(sid, None)
                sessions_evicted += 1

        # Clean up orphaned touch entries no longer present in session_histories
        for sid in list(self._last_touched.keys()):
            if sid not in self._session_histories and sid not in active:
                if (current_time - self._last_touched[sid]) >= self.idle_ttl_seconds:
                    self._last_touched.pop(sid, None)

        steering_evicted = 0
        if self._steering_queue is not None:
            evict_fn = getattr(self._steering_queue, "evict_idle", None)
            if callable(evict_fn):
                result = evict_fn(self.idle_ttl_seconds, active_sessions=active)
                if isinstance(result, int):
                    steering_evicted = result

        leases_evicted = 0
        if self._credential_relay is not None:
            reap_fn = getattr(self._credential_relay, "reap_expired", None)
            if callable(reap_fn):
                result = reap_fn(now=current_time)
                if isinstance(result, int):
                    leases_evicted = result

        return {
            "sessions_evicted": sessions_evicted,
            "steering_evicted": steering_evicted,
            "leases_evicted": leases_evicted,
        }

    async def _sweep_loop(self) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return
        interval = max(0.001, float(self.sweep_interval_seconds))
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                self.reap_once()
            except Exception:
                logger.exception("RuntimeIdleReaper sweep encountered an error.")

    def start(self) -> None:
        """Start the background asyncio sweep loop (idempotent if already running)."""
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._task = loop.create_task(
            self._sweep_loop(),
            name="pyharness-runtime-idle-reaper",
        )

    async def stop(self) -> None:
        """Stop and await the background sweep loop (idempotent; safe for `PluginContext.effect`)."""
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stop_event = None


__all__ = [
    "AdmissionRejectedError",
    "TopicBlockedError",
    "SessionAdmissionController",
    "TopicAdmissionRule",
    "TopicAdmissionDecision",
    "TopicAdmissionGuard",
    "RuntimeIdleReaper",
]
