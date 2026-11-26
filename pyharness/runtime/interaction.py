from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator, Literal

from pydantic import BaseModel, Field

from pyharness.runtime.governance import CancellationToken, TurnCancelledError
from pyharness.runtime.schemas import RiskLevel, ToolSpec
from pyharness.security.auth import current_principal

if TYPE_CHECKING:
    from pyharness.security.auth import Principal


# ---------------------------------------------------------------------------
# 1. Active Turn Execution Context
# ---------------------------------------------------------------------------


@dataclass
class ActiveTurnContext:
    """Turn-scoped execution metadata propagated to tools via ContextVar."""

    session_id: str
    turn_id: str
    cancellation: CancellationToken | None = None
    owner_user_id: str = "anonymous"
    tenant_id: str = "default"


current_turn_context: ContextVar[ActiveTurnContext | None] = ContextVar(
    "pyharness_turn_context", default=None
)


@contextmanager
def use_turn_context(ctx: ActiveTurnContext | None) -> Iterator[None]:
    """Bind an ActiveTurnContext for the duration of a turn execution block."""
    token = current_turn_context.set(ctx)
    try:
        yield
    finally:
        current_turn_context.reset(token)


# ---------------------------------------------------------------------------
# 2. Dual-Queue Turn Steering & Follow-up Inbox
# ---------------------------------------------------------------------------


class SteeringMessage(BaseModel):
    """Mid-turn steering directive or queued post-turn follow-up message."""

    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    session_id: str
    content: str
    kind: Literal["steer", "followup"] = "steer"
    user_id: str = "anonymous"
    tenant_id: str = "default"
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class TurnSteeringQueue:
    """
    Per-session dual queue separating mid-turn steering interrupts from
    post-turn follow-up prompts, with idle-bucket eviction.
    """

    def __init__(self) -> None:
        self._steers: dict[str, deque[SteeringMessage]] = {}
        self._followups: dict[str, deque[SteeringMessage]] = {}
        self._last_activity: dict[str, float] = {}

    def enqueue_steer(
        self,
        session_id: str,
        content: str,
        *,
        user_id: str = "anonymous",
        tenant_id: str = "default",
    ) -> SteeringMessage:
        msg = SteeringMessage(
            session_id=session_id,
            content=content,
            kind="steer",
            user_id=user_id,
            tenant_id=tenant_id,
        )
        self._steers.setdefault(session_id, deque()).append(msg)
        self._last_activity[session_id] = time.time()
        return msg

    def enqueue_followup(
        self,
        session_id: str,
        content: str,
        *,
        user_id: str = "anonymous",
        tenant_id: str = "default",
    ) -> SteeringMessage:
        msg = SteeringMessage(
            session_id=session_id,
            content=content,
            kind="followup",
            user_id=user_id,
            tenant_id=tenant_id,
        )
        self._followups.setdefault(session_id, deque()).append(msg)
        self._last_activity[session_id] = time.time()
        return msg

    def has_steer(self, session_id: str) -> bool:
        return bool(self._steers.get(session_id))

    def pop_steer(self, session_id: str) -> SteeringMessage | None:
        bucket = self._steers.get(session_id)
        if not bucket:
            return None
        msg = bucket.popleft()
        self._last_activity[session_id] = time.time()
        return msg

    def pop_all_steers(self, session_id: str) -> list[SteeringMessage]:
        bucket = self._steers.get(session_id)
        if not bucket:
            return []
        items = list(bucket)
        bucket.clear()
        self._last_activity[session_id] = time.time()
        return items

    def has_followup(self, session_id: str) -> bool:
        return bool(self._followups.get(session_id))

    def pop_followup(self, session_id: str) -> SteeringMessage | None:
        bucket = self._followups.get(session_id)
        if not bucket:
            return None
        msg = bucket.popleft()
        self._last_activity[session_id] = time.time()
        return msg

    def peek_counts(self, session_id: str) -> dict[str, int]:
        return {
            "steer": len(self._steers.get(session_id, ())),
            "followup": len(self._followups.get(session_id, ())),
        }

    def clear_session(self, session_id: str) -> None:
        self._steers.pop(session_id, None)
        self._followups.pop(session_id, None)
        self._last_activity.pop(session_id, None)

    def evict_idle(
        self,
        max_idle_seconds: float,
        active_sessions: set[str] | None = None,
    ) -> int:
        """Evict empty or stale non-active session buckets older than `max_idle_seconds`."""
        now = time.time()
        active = active_sessions or set()
        all_sessions = (
            set(self._last_activity.keys())
            | set(self._steers.keys())
            | set(self._followups.keys())
        )
        evicted = 0
        for sid in all_sessions:
            if sid in active:
                continue
            last_ts = self._last_activity.get(sid, 0.0)
            if (now - last_ts) >= max_idle_seconds:
                self.clear_session(sid)
                evicted += 1
        return evicted


# ---------------------------------------------------------------------------
# 3. Interactive Clarification Broker (Human-in-the-Loop User Questions)
# ---------------------------------------------------------------------------


class QuestionAuthorizationError(PermissionError):
    """Raised when a caller is not authorized to answer a pending user question."""


class QuestionOption(BaseModel):
    """Selectable option presented in an interactive clarification prompt."""

    id: str
    label: str
    description: str = ""


class UserQuestionAnswer(BaseModel):
    """Structured answer returned by a user (or timeout guard) for a clarification request."""

    question_id: str
    selected_options: list[str] = Field(default_factory=list)
    free_text: str = ""
    answered_by: str = "anonymous"
    timed_out: bool = False


@dataclass
class PendingUserQuestion:
    """In-flight clarification request awaiting user input or cancellation/timeout."""

    question_id: str
    session_id: str
    turn_id: str
    question: str
    options: list[QuestionOption]
    allow_free_text: bool
    multi_select: bool
    owner_user_id: str
    tenant_id: str
    created_at_ms: int
    future: asyncio.Future[UserQuestionAnswer] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "question": self.question,
            "options": [opt.model_dump() for opt in self.options],
            "allow_free_text": self.allow_free_text,
            "multi_select": self.multi_select,
            "owner_user_id": self.owner_user_id,
            "tenant_id": self.tenant_id,
            "created_at_ms": self.created_at_ms,
        }


def _normalize_options(
    options: list[str | dict[str, Any] | QuestionOption] | None,
) -> list[QuestionOption]:
    if not options:
        return []
    normalized: list[QuestionOption] = []
    for idx, item in enumerate(options, start=1):
        if isinstance(item, QuestionOption):
            normalized.append(item)
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            normalized.append(QuestionOption(id=f"opt_{idx}", label=text))
        elif isinstance(item, dict):
            opt_id = str(item.get("id") or f"opt_{idx}")
            label = str(item.get("label") or item.get("text") or item.get("title") or opt_id)
            description = str(item.get("description") or "")
            normalized.append(
                QuestionOption(id=opt_id, label=label, description=description)
            )
        else:
            normalized.append(QuestionOption(id=f"opt_{idx}", label=str(item)))
    return normalized


class UserQuestionBroker:
    """
    Async interactive clarification broker allowing tools (`ask_user_question`) to
    pause a turn, solicit structured/free-text input from the authorized session owner,
    and resume cleanly on answer, cancellation, or timeout.
    """

    def __init__(self, default_timeout_seconds: float = 120.0) -> None:
        self.default_timeout_seconds = float(default_timeout_seconds)
        self._pending: dict[str, PendingUserQuestion] = {}

    def create_question(
        self,
        *,
        session_id: str,
        turn_id: str,
        question: str,
        options: list[str | dict[str, Any] | QuestionOption] | None = None,
        allow_free_text: bool = True,
        multi_select: bool = False,
        owner_user_id: str = "anonymous",
        tenant_id: str = "default",
    ) -> PendingUserQuestion:
        loop = asyncio.get_running_loop()
        question_id = f"q_{uuid.uuid4().hex[:10]}"
        req = PendingUserQuestion(
            question_id=question_id,
            session_id=session_id,
            turn_id=turn_id,
            question=question,
            options=_normalize_options(options),
            allow_free_text=bool(allow_free_text),
            multi_select=bool(multi_select),
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            created_at_ms=int(time.time() * 1000),
            future=loop.create_future(),
        )
        self._pending[question_id] = req
        return req

    async def wait_for_answer(
        self,
        req: PendingUserQuestion,
        *,
        cancellation: CancellationToken | None = None,
        timeout_seconds: float | None = None,
    ) -> UserQuestionAnswer:
        eff_timeout = (
            self.default_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        cancel_task: asyncio.Task[None] | None = None
        try:
            if cancellation is not None and cancellation.is_cancelled:
                raise TurnCancelledError(cancellation.reason or "cancelled_during_question")
            if req.future.done():
                return req.future.result()

            wait_set: set[asyncio.Future[Any]] = {req.future}
            if cancellation is not None:
                cancel_task = asyncio.create_task(cancellation.wait())
                wait_set.add(cancel_task)

            done, _ = await asyncio.wait(
                wait_set,
                timeout=eff_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task is not None and (
                cancel_task in done
                or (cancellation is not None and cancellation.is_cancelled)
            ):
                reason = cancellation.reason if cancellation is not None else ""
                raise TurnCancelledError(reason or "cancelled_during_question")
            if req.future in done:
                return req.future.result()

            return UserQuestionAnswer(
                question_id=req.question_id,
                free_text="[Timed Out] No user answer received.",
                timed_out=True,
            )
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
                try:
                    await cancel_task
                except asyncio.CancelledError:
                    pass
            if not req.future.done():
                req.future.cancel()
            self._pending.pop(req.question_id, None)

    def submit_answer(
        self,
        question_id: str,
        *,
        selected_options: list[str] | None = None,
        free_text: str = "",
        principal: Principal | None = None,
    ) -> bool:
        req = self._pending.get(question_id)
        if req is None or req.future.done():
            return False

        if principal is not None and not principal.is_admin:
            if req.tenant_id != principal.tenant_id:
                raise QuestionAuthorizationError("Cross-tenant question answer denied.")
            if req.owner_user_id != "anonymous" and principal.user_id != req.owner_user_id:
                raise QuestionAuthorizationError(
                    f"Only session owner '{req.owner_user_id}' or admin may answer question '{question_id}'."
                )

        answered_by = principal.user_id if principal is not None else "anonymous"
        answer = UserQuestionAnswer(
            question_id=question_id,
            selected_options=[str(opt) for opt in (selected_options or [])],
            free_text=str(free_text or ""),
            answered_by=answered_by,
            timed_out=False,
        )
        req.future.set_result(answer)
        return True

    def list_pending(
        self,
        *,
        session_id: str | None = None,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for req in self._pending.values():
            if req.future.done():
                continue
            if session_id is not None and req.session_id != session_id:
                continue
            if not is_admin and tenant_id is not None and req.tenant_id != tenant_id:
                continue
            items.append(req.to_dict())
        return items

    def register_tools(
        self,
        registry: Any,
        *,
        on_question_required: Callable[[str, str, dict[str, Any]], Awaitable[None]] | None = None,
        on_question_answered: Callable[[str, str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Mount the `ask_user_question` interactive clarification tool into `registry`."""

        async def _ask_user_question_handler(
            question: str,
            options: list[Any] | None = None,
            allow_free_text: bool = True,
            multi_select: bool = False,
        ) -> str:
            turn_ctx = current_turn_context.get()
            principal = current_principal.get()

            session_id = turn_ctx.session_id if turn_ctx is not None else "default"
            turn_id = turn_ctx.turn_id if turn_ctx is not None else "unknown_turn"
            cancellation = turn_ctx.cancellation if turn_ctx is not None else None

            if principal is not None:
                owner_user_id = (
                    turn_ctx.owner_user_id
                    if turn_ctx is not None and turn_ctx.owner_user_id != "anonymous"
                    else principal.user_id
                )
                tenant_id = (
                    turn_ctx.tenant_id
                    if turn_ctx is not None and turn_ctx.tenant_id != "default"
                    else principal.tenant_id
                )
            elif turn_ctx is not None:
                owner_user_id = turn_ctx.owner_user_id
                tenant_id = turn_ctx.tenant_id
            else:
                owner_user_id = "anonymous"
                tenant_id = "default"

            req = self.create_question(
                session_id=session_id,
                turn_id=turn_id,
                question=question,
                options=options,
                allow_free_text=allow_free_text,
                multi_select=multi_select,
                owner_user_id=owner_user_id,
                tenant_id=tenant_id,
            )

            try:
                if on_question_required is not None:
                    await on_question_required(session_id, turn_id, req.to_dict())
                answer = await self.wait_for_answer(req, cancellation=cancellation)
            except BaseException:
                self._pending.pop(req.question_id, None)
                if not req.future.done():
                    req.future.cancel()
                raise

            payload: dict[str, Any] = {
                "question_id": answer.question_id,
                "question": req.question,
                "selected_options": list(answer.selected_options),
                "free_text": answer.free_text,
                "answered_by": answer.answered_by,
                "timed_out": answer.timed_out,
            }
            if on_question_answered is not None:
                await on_question_answered(session_id, turn_id, payload)
            return json.dumps(payload, ensure_ascii=False)

        registry.register(
            ToolSpec(
                name="ask_user_question",
                description=(
                    "Pause the current turn and ask the user a structured clarification question "
                    "with optional selectable choices and/or free-text input."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Clear clarification question to present to the user.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of selectable choices (strings or "
                                "{id, label, description} objects)."
                            ),
                        },
                        "allow_free_text": {
                            "type": "boolean",
                            "description": "Whether the user may provide a free-text response.",
                        },
                        "multi_select": {
                            "type": "boolean",
                            "description": "Whether multiple options can be selected simultaneously.",
                        },
                    },
                    "required": ["question"],
                },
                risk_level=RiskLevel.LOW,
                timeout_seconds=max(180.0, self.default_timeout_seconds + 30.0),
            ),
            _ask_user_question_handler,
        )


__all__ = [
    "ActiveTurnContext",
    "PendingUserQuestion",
    "QuestionAuthorizationError",
    "QuestionOption",
    "SteeringMessage",
    "TurnSteeringQueue",
    "UserQuestionAnswer",
    "UserQuestionBroker",
    "current_turn_context",
    "use_turn_context",
]
