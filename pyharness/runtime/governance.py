from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from pyharness.runtime.schemas import Message, TokenUsage, ToolCall


class TurnCancelledError(Exception):
    """Raised immediately when a session cancellation signal is triggered."""


class CancellationToken:
    """
    Cooperative session-level cancellation primitive (inspired by DSH session-cancellation.mjs).
    Allows immediate interruption of LLM waits, tool tasks, and pending human approvals.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str = ""

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "user_requested") -> None:
        self.reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TurnCancelledError(self.reason or "cancelled")

    async def wait(self) -> None:
        await self._event.wait()


@dataclass
class ApprovalDecision:
    approval_id: str
    approved: bool
    reviewer: str = "system"
    note: str = ""


@dataclass
class PendingApproval:
    approval_id: str
    session_id: str
    turn_id: str
    tool_call: ToolCall
    reason: str
    future: asyncio.Future[ApprovalDecision] = field(repr=False)


class ApprovalBroker:
    """
    Human-in-the-Loop governance gate (inspired by DSH desktop-approval-broker.mjs).

    Suspends execution before HIGH-risk tool calls using an asyncio.Future, racing against
    both a configurable timeout (fail-closed -> reject) and session cancellation.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def list_pending(self, session_id: str | None = None) -> list[dict[str, Any]]:
        items = []
        for req in self._pending.values():
            if session_id is None or req.session_id == session_id:
                items.append(
                    {
                        "approval_id": req.approval_id,
                        "session_id": req.session_id,
                        "turn_id": req.turn_id,
                        "tool_name": req.tool_call.name,
                        "arguments": req.tool_call.arguments,
                        "reason": req.reason,
                    }
                )
        return items

    def create_request(
        self,
        session_id: str,
        turn_id: str,
        tool_call: ToolCall,
        reason: str,
    ) -> PendingApproval:
        loop = asyncio.get_running_loop()
        approval_id = f"apr_{uuid.uuid4().hex[:10]}"
        req = PendingApproval(
            approval_id=approval_id,
            session_id=session_id,
            turn_id=turn_id,
            tool_call=tool_call,
            reason=reason,
            future=loop.create_future(),
        )
        self._pending[approval_id] = req
        return req

    async def wait_for_decision(
        self,
        req: PendingApproval,
        cancellation: CancellationToken,
        timeout_seconds: float = 120.0,
    ) -> ApprovalDecision:
        cancel_task = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {req.future, cancel_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done or cancellation.is_cancelled:
                raise TurnCancelledError(cancellation.reason or "cancelled_during_approval")
            if req.future in done:
                return req.future.result()
            # Fail-closed on timeout: never auto-approve sensitive actions
            return ApprovalDecision(
                approval_id=req.approval_id,
                approved=False,
                reviewer="timeout_guard",
                note=f"Approval timed out after {timeout_seconds}s (fail-closed).",
            )
        finally:
            cancel_task.cancel()
            self._pending.pop(req.approval_id, None)

    def resolve(
        self,
        approval_id: str,
        approved: bool,
        reviewer: str = "user",
        note: str = "",
    ) -> bool:
        req = self._pending.get(approval_id)
        if req is None or req.future.done():
            return False
        req.future.set_result(
            ApprovalDecision(
                approval_id=approval_id,
                approved=approved,
                reviewer=reviewer,
                note=note,
            )
        )
        return True


class TokenBudgetGuard:
    """Hard per-turn token ceiling preventing runaway Agent loops from burning API quota."""

    def __init__(self, max_tokens_per_turn: int = 32000) -> None:
        self.max_tokens_per_turn = max_tokens_per_turn
        self.usage = TokenUsage()

    def record(self, delta: TokenUsage) -> None:
        self.usage.add(delta)

    @property
    def is_exceeded(self) -> bool:
        return self.usage.total_tokens > self.max_tokens_per_turn


class ContextCompressor:
    """
    Incremental state & dialogue summary compressor.

    Preserves:
    - All leading `system` messages verbatim.
    - The most recent `keep_recent_messages` verbatim (so tool call/result pairs remain intact).
    Compresses middle history into a compact incremental summary block when estimated tokens
    exceed `threshold_tokens`.
    """

    def __init__(
        self,
        threshold_tokens: int = 4000,
        keep_recent_messages: int = 4,
    ) -> None:
        self.threshold_tokens = threshold_tokens
        self.keep_recent_messages = keep_recent_messages

    @staticmethod
    def estimate_total_tokens(messages: list[Message]) -> int:
        return sum(m.estimate_tokens() for m in messages)

    def should_compress(self, messages: list[Message]) -> bool:
        return (
            len(messages) > self.keep_recent_messages + 2
            and self.estimate_total_tokens(messages) > self.threshold_tokens
        )

    def compress(self, messages: list[Message]) -> tuple[list[Message], dict[str, int]]:
        """
        Deterministically fold middle turns into an incremental summary message.
        (In production, the summary builder can optionally delegate to a fast lightweight LLM.)
        """
        tokens_before = self.estimate_total_tokens(messages)

        system_msgs: list[Message] = []
        idx = 0
        while idx < len(messages) and messages[idx].role == "system":
            system_msgs.append(messages[idx])
            idx += 1

        body = messages[idx:]
        if len(body) <= self.keep_recent_messages:
            return messages, {
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "messages_folded": 0,
            }

        split_at = len(body) - self.keep_recent_messages
        # Avoid splitting right between an assistant tool_call and its following tool results
        while split_at > 0 and body[split_at].role == "tool":
            split_at -= 1
        if split_at <= 0:
            return messages, {
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "messages_folded": 0,
            }

        to_fold = body[:split_at]
        recent = body[split_at:]

        bullet_lines: list[str] = []
        for m in to_fold:
            snippet = m.content.strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            if m.role == "tool":
                bullet_lines.append(f"- [tool:{m.name}] {snippet}")
            elif m.tool_calls:
                names = ", ".join(tc.name for tc in m.tool_calls)
                bullet_lines.append(f"- [assistant->tools:{names}] {snippet}")
            else:
                bullet_lines.append(f"- [{m.role}] {snippet}")

        summary_msg = Message(
            role="system",
            content="[Incremental Context Summary]\n" + "\n".join(bullet_lines[-12:]),
        )
        compressed = [*system_msgs, summary_msg, *recent]
        tokens_after = self.estimate_total_tokens(compressed)
        return compressed, {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "messages_folded": len(to_fold),
        }
