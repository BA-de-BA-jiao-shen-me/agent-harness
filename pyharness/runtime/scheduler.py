from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from pyharness.runtime.schemas import EventType
from pyharness.security.auth import Principal, use_principal


# ---------------------------------------------------------------------------
# 1. Lightweight 5-Field Cron Expression Matcher
# ---------------------------------------------------------------------------

_CRON_BOUNDS: tuple[tuple[int, int, bool], ...] = (
    (0, 59, False),  # minute
    (0, 23, False),  # hour
    (1, 31, False),  # day of month
    (1, 12, False),  # month
    (0, 7, True),    # day of week (0 or 7 == Sunday, 1 == Monday .. 6 == Saturday)
)


def _parse_cron_field(
    field_expr: str,
    min_val: int,
    max_val: int,
    *,
    is_dow: bool = False,
) -> set[int]:
    token = field_expr.strip()
    if not token:
        raise ValueError("Empty cron field.")

    allowed: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty list element in cron field {field_expr!r}.")

        step: int | None = None
        if "/" in part:
            pieces = part.split("/")
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise ValueError(f"Invalid step syntax in cron field {field_expr!r}.")
            base = pieces[0].strip()
            try:
                step = int(pieces[1].strip())
            except ValueError as exc:
                raise ValueError(f"Non-integer step in cron field {field_expr!r}.") from exc
            if step <= 0:
                raise ValueError(f"Cron step must be positive in {field_expr!r}.")
        else:
            base = part

        if base == "*":
            start, end = min_val, max_val
            eff_step = step if step is not None else 1
            for val in range(start, end + 1, eff_step):
                allowed.add(0 if (is_dow and val == 7) else val)
            continue

        if "-" in base:
            bounds = base.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid range syntax in cron field {field_expr!r}.")
            try:
                lo = int(bounds[0].strip())
                hi = int(bounds[1].strip())
            except ValueError as exc:
                raise ValueError(f"Non-integer range bound in cron field {field_expr!r}.") from exc
            if lo < min_val or hi > max_val or lo > hi:
                raise ValueError(
                    f"Cron range {lo}-{hi} out of bounds [{min_val}, {max_val}]."
                )
            eff_step = step if step is not None else 1
            for val in range(lo, hi + 1, eff_step):
                allowed.add(0 if (is_dow and val == 7) else val)
            continue

        try:
            single = int(base)
        except ValueError as exc:
            raise ValueError(f"Invalid cron token {base!r} in {field_expr!r}.") from exc
        if single < min_val or single > max_val:
            raise ValueError(
                f"Cron value {single} out of bounds [{min_val}, {max_val}]."
            )

        if step is None:
            allowed.add(0 if (is_dow and single == 7) else single)
        else:
            for val in range(single, max_val + 1, step):
                allowed.add(0 if (is_dow and val == 7) else val)

    return allowed


def _parse_cron_expression(cron_expr: str) -> tuple[set[int], ...]:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(
            f"Invalid cron expression {cron_expr!r}: expected exactly 5 fields "
            "(minute hour dom month dow)."
        )
    return tuple(
        _parse_cron_field(raw, min_val, max_val, is_dow=is_dow)
        for raw, (min_val, max_val, is_dow) in zip(fields, _CRON_BOUNDS)
    )


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """
    Evaluate whether `dt` matches a standard 5-field cron expression
    (`minute hour dom month dow`), supporting `*`, `*/N`, comma lists, ranges,
    and integer values (with Sunday as 0 or 7).
    """
    if not cron_expr or not isinstance(cron_expr, str):
        return False
    try:
        minute_set, hour_set, dom_set, month_set, dow_set = _parse_cron_expression(
            cron_expr
        )
    except ValueError:
        return False

    cron_dow = (dt.weekday() + 1) % 7  # Python Monday=0..Sunday=6 -> Cron Sunday=0..Saturday=6
    return (
        dt.minute in minute_set
        and dt.hour in hour_set
        and dt.day in dom_set
        and dt.month in month_set
        and cron_dow in dow_set
    )


# ---------------------------------------------------------------------------
# 2. Scheduled Task Schemas
# ---------------------------------------------------------------------------


class ScheduledTaskSpec(BaseModel):
    """Declarative specification and runtime state for a scheduled autonomous task."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    name: str
    prompt: str
    interval_seconds: float | None = None
    cron_expr: str | None = None
    session_prefix: str = "sched"
    tenant_id: str = "default"
    user_id: str = "scheduler"
    roles: list[str] = Field(default_factory=lambda: ["member"])
    enabled: bool = True
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    last_run_at_ms: int | None = None
    run_count: int = 0
    last_status: str | None = None
    last_artifact_id: str | None = None


class ScheduledTaskRun(BaseModel):
    """Immutable execution record for a single scheduled task run."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    task_id: str
    session_id: str
    turn_id: str
    status: str
    final_answer_preview: str
    artifact_id: str | None = None
    started_at_ms: int
    finished_at_ms: int
    error: str | None = None


# ---------------------------------------------------------------------------
# 3. Autonomous Scheduled Task Manager
# ---------------------------------------------------------------------------


class ScheduledTaskManager:
    """
    Async scheduler that runs autonomous `AgentHarnessEngine` turns on fixed intervals,
    5-field cron schedules, or manual `run_now()` triggers under an isolated scheduler
    `Principal`, persisting optional output artifacts and audit journal events.
    """

    def __init__(
        self,
        engine: Any,
        journal: Any = None,
        artifact_store: Any = None,
        tick_interval_seconds: float = 1.0,
        system_prompt: str = "You are PyHarness autonomous scheduled worker.",
        *,
        max_run_history: int = 200,
    ) -> None:
        self.engine = engine
        self.journal = journal
        self.artifact_store = artifact_store
        self.tick_interval_seconds = float(tick_interval_seconds)
        self.system_prompt = system_prompt

        self._tasks: dict[str, ScheduledTaskSpec] = {}
        self._runs: deque[ScheduledTaskRun] = deque(maxlen=max(1, int(max_run_history)))
        self._bg_task: asyncio.Task[None] | None = None
        self._running_tasks: set[str] = set()
        self._active_run_tasks: set[asyncio.Task[Any]] = set()
        self._last_cron_minute: dict[str, tuple[int, int, int, int, int]] = {}

    @property
    def is_running(self) -> bool:
        return self._bg_task is not None and not self._bg_task.done()

    def register_task(self, spec: ScheduledTaskSpec) -> ScheduledTaskSpec:
        """
        Register a scheduled (or manual-only) task specification.
        Validates `interval_seconds > 0` and 5-field `cron_expr` syntax when provided.
        """
        if spec.interval_seconds is not None and spec.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive when specified.")
        if spec.cron_expr is not None:
            cleaned = spec.cron_expr.strip()
            if cleaned:
                _parse_cron_expression(cleaned)
                spec.cron_expr = cleaned
            else:
                spec.cron_expr = None

        self._tasks[spec.task_id] = spec
        return spec

    def get_task(self, task_id: str) -> ScheduledTaskSpec | None:
        return self._tasks.get(task_id)

    def list_tasks(self, tenant_id: str | None = None) -> list[ScheduledTaskSpec]:
        tasks = list(self._tasks.values())
        if tenant_id is not None:
            tasks = [t for t in tasks if t.tenant_id == tenant_id]
        return tasks

    def list_runs(
        self,
        task_id: str | None = None,
        limit: int = 20,
    ) -> list[ScheduledTaskRun]:
        if limit <= 0:
            return []
        runs = [r for r in self._runs if task_id is None or r.task_id == task_id]
        return runs[-limit:]

    def pause_task(self, task_id: str) -> bool:
        spec = self._tasks.get(task_id)
        if spec is None:
            return False
        spec.enabled = False
        return True

    def resume_task(self, task_id: str) -> bool:
        spec = self._tasks.get(task_id)
        if spec is None:
            return False
        spec.enabled = True
        return True

    def delete_task(self, task_id: str) -> bool:
        self._last_cron_minute.pop(task_id, None)
        return self._tasks.pop(task_id, None) is not None

    async def _save_artifact(
        self,
        *,
        session_id: str,
        turn_id: str,
        tool_name: str,
        content: str,
        tenant_id: str,
        metadata: dict[str, Any],
    ) -> str | None:
        store = self.artifact_store
        if store is None:
            return None

        save_fn = None
        for attr in ("save", "put", "create", "store"):
            candidate = getattr(store, attr, None)
            if callable(candidate):
                save_fn = candidate
                break
        if save_fn is None:
            return None

        all_candidates: dict[str, Any] = {
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_name": tool_name,
            "content": content,
            "tenant_id": tenant_id,
            "content_type": "text/markdown",
            "mime_type": "text/markdown",
            "metadata": metadata,
        }

        try:
            sig = inspect.signature(save_fn)
            params = sig.parameters
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_var_kw:
                kwargs: dict[str, Any] = {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "tool_name": tool_name,
                    "content": content,
                    "tenant_id": tenant_id,
                }
            else:
                kwargs = {k: v for k, v in all_candidates.items() if k in params}
                if "content" not in kwargs:
                    for alt in ("text", "data", "body"):
                        if alt in params:
                            kwargs[alt] = content
                            break
        except (TypeError, ValueError):
            kwargs = {
                "session_id": session_id,
                "turn_id": turn_id,
                "tool_name": tool_name,
                "content": content,
                "tenant_id": tenant_id,
            }

        res = save_fn(**kwargs)
        if inspect.isawaitable(res):
            res = await res

        if res is None:
            return None
        if isinstance(res, str):
            return res
        if hasattr(res, "artifact_id"):
            return str(getattr(res, "artifact_id"))
        if isinstance(res, dict) and "artifact_id" in res:
            return str(res["artifact_id"])
        if hasattr(res, "id"):
            return str(getattr(res, "id"))
        if isinstance(res, dict) and "id" in res:
            return str(res["id"])
        return None

    async def run_now(self, task_id: str) -> ScheduledTaskRun:
        """Execute a registered task immediately and record its run result & audit event."""
        spec = self._tasks.get(task_id)
        if spec is None:
            raise KeyError(f"Scheduled task '{task_id}' not found.")

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        run_session_id = f"{spec.session_prefix}_{spec.task_id}_{run_id}"
        started_at_ms = int(time.time() * 1000)

        principal = Principal(
            user_id=spec.user_id,
            tenant_id=spec.tenant_id,
            roles=frozenset(spec.roles),
            auth_method="scheduler",
        )

        self._running_tasks.add(task_id)
        steps = 0
        try:
            with use_principal(principal):
                try:
                    result = await self.engine.run_turn(
                        session_id=run_session_id,
                        user_input=spec.prompt,
                        system_prompt=self.system_prompt,
                    )
                    finished_at_ms = int(time.time() * 1000)
                    turn_id = str(
                        getattr(result, "turn_id", "") or f"turn_{uuid.uuid4().hex[:10]}"
                    )
                    raw_status = getattr(result, "status", "completed")
                    status_str = (
                        raw_status.value
                        if hasattr(raw_status, "value")
                        else str(raw_status)
                    )
                    final_answer = str(getattr(result, "final_answer", "") or "")
                    steps = int(getattr(result, "steps_executed", 0) or 0)
                    error_msg = getattr(result, "error_message", None)

                    artifact_id: str | None = None
                    if self.artifact_store is not None and final_answer:
                        artifact_id = await self._save_artifact(
                            session_id=run_session_id,
                            turn_id=turn_id,
                            tool_name=f"scheduled_task:{spec.name}",
                            content=final_answer,
                            tenant_id=spec.tenant_id,
                            metadata={
                                "task_id": spec.task_id,
                                "task_name": spec.name,
                                "run_id": run_id,
                            },
                        )

                    run = ScheduledTaskRun(
                        run_id=run_id,
                        task_id=spec.task_id,
                        session_id=run_session_id,
                        turn_id=turn_id,
                        status=status_str,
                        final_answer_preview=final_answer[:300],
                        artifact_id=artifact_id,
                        started_at_ms=started_at_ms,
                        finished_at_ms=finished_at_ms,
                        error=error_msg,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    finished_at_ms = int(time.time() * 1000)
                    turn_id = f"turn_{uuid.uuid4().hex[:10]}"
                    run = ScheduledTaskRun(
                        run_id=run_id,
                        task_id=spec.task_id,
                        session_id=run_session_id,
                        turn_id=turn_id,
                        status="failed",
                        final_answer_preview="",
                        artifact_id=None,
                        started_at_ms=started_at_ms,
                        finished_at_ms=finished_at_ms,
                        error=str(exc),
                    )

                if self.journal is not None:
                    await self.journal.emit(
                        run.session_id,
                        run.turn_id,
                        EventType.SCHEDULED_TASK_RUN,
                        {
                            "task_id": spec.task_id,
                            "name": spec.name,
                            "run_id": run.run_id,
                            "status": run.status,
                            "artifact_id": run.artifact_id,
                            "steps": steps,
                        },
                    )
        finally:
            self._running_tasks.discard(task_id)

        spec.last_run_at_ms = run.finished_at_ms
        spec.run_count += 1
        spec.last_status = run.status
        spec.last_artifact_id = run.artifact_id
        self._runs.append(run)
        return run

    def _is_due(
        self,
        spec: ScheduledTaskSpec,
        now_dt: datetime,
        now_ms: int,
    ) -> bool:
        if not spec.enabled or spec.task_id in self._running_tasks:
            return False

        if spec.interval_seconds is not None:
            base_ms = (
                spec.last_run_at_ms
                if spec.last_run_at_ms is not None
                else spec.created_at_ms
            )
            if (now_ms - base_ms) >= spec.interval_seconds * 1000.0:
                return True

        if spec.cron_expr is not None:
            minute_key = (
                now_dt.year,
                now_dt.month,
                now_dt.day,
                now_dt.hour,
                now_dt.minute,
            )
            if (
                self._last_cron_minute.get(spec.task_id) != minute_key
                and cron_matches(spec.cron_expr, now_dt)
            ):
                self._last_cron_minute[spec.task_id] = minute_key
                return True

        return False

    async def _run_due_task(self, task_id: str) -> None:
        try:
            await self.run_now(task_id)
        except (KeyError, asyncio.CancelledError):
            pass
        finally:
            self._running_tasks.discard(task_id)

    async def _tick_once(self) -> None:
        now_dt = datetime.now()
        now_ms = int(time.time() * 1000)
        for spec in list(self._tasks.values()):
            if self._is_due(spec, now_dt, now_ms):
                self._running_tasks.add(spec.task_id)
                task = asyncio.create_task(
                    self._run_due_task(spec.task_id),
                    name=f"pyharness-sched-{spec.task_id}",
                )
                self._active_run_tasks.add(task)
                task.add_done_callback(self._active_run_tasks.discard)

    async def _tick_loop(self) -> None:
        try:
            while True:
                await self._tick_once()
                await asyncio.sleep(self.tick_interval_seconds)
        except asyncio.CancelledError:
            return

    def start(self) -> None:
        """Start the background scheduler tick loop (idempotent)."""
        if self._bg_task is not None and not self._bg_task.done():
            return
        loop = asyncio.get_running_loop()
        self._bg_task = loop.create_task(self._tick_loop(), name="pyharness-scheduler")

    async def stop(self) -> None:
        """Stop the background tick loop and cancel any in-flight scheduled runs cleanly."""
        bg_task = self._bg_task
        self._bg_task = None
        if bg_task is not None:
            bg_task.cancel()
            try:
                await bg_task
            except asyncio.CancelledError:
                pass

        pending_runs = list(self._active_run_tasks)
        for t in pending_runs:
            t.cancel()
        for t in pending_runs:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._active_run_tasks.clear()


__all__ = [
    "ScheduledTaskManager",
    "ScheduledTaskRun",
    "ScheduledTaskSpec",
    "cron_matches",
]
