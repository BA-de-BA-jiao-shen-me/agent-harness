from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyharness.runtime.schemas import Message, TokenUsage, ToolCall

if TYPE_CHECKING:
    from pyharness.infra.coordination import CoordinationBackend
    from pyharness.runtime.llm_client import BaseLLMClient
    from pyharness.security.auth import Principal


class TurnCancelledError(Exception):
    """Raised immediately when a session cancellation signal is triggered."""


class ApprovalAuthorizationError(PermissionError):
    """Raised when a reviewer violates RBAC, tenant isolation, or Four-Eyes SoD rules."""


class CancellationToken:
    """
    Cooperative session-level cancellation primitive (inspired by DSH session-cancellation.mjs).
    Allows immediate interruption of LLM streams, tool tasks, subagents, and pending human approvals.
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
    requester_id: str = "anonymous"
    tenant_id: str = "default"
    required_roles: list[str] = field(default_factory=lambda: ["approver", "admin"])
    policy_rule_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "tool_name": self.tool_call.name,
            "arguments": self.tool_call.arguments,
            "reason": self.reason,
            "requester_id": self.requester_id,
            "requester": self.requester_id,
            "tenant_id": self.tenant_id,
            "required_roles": list(self.required_roles),
            "policy_rule_id": self.policy_rule_id,
        }


class ApprovalBroker:
    """
    Enterprise Human-in-the-Loop governance gate (inspired by DSH desktop-approval-broker.mjs).

    Enforces:
    - Single-process `asyncio.Future` + cross-worker Redis Pub/Sub wakeups.
    - Fail-closed timeout rejection.
    - Enterprise RBAC role verification, tenant isolation, cryptographic reviewer identity binding,
      and Four-Eyes Separation of Duties (`reviewer_id != requester_id`).
    """

    def __init__(
        self,
        coordination: CoordinationBackend | None = None,
        enforce_four_eyes: bool = False,
    ) -> None:
        self._pending: dict[str, PendingApproval] = {}
        self._coordination = coordination
        self.enforce_four_eyes = enforce_four_eyes

    def list_pending(
        self,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        items = []
        for req in self._pending.values():
            if session_id is not None and req.session_id != session_id:
                continue
            if tenant_id is not None and req.tenant_id != tenant_id:
                continue
            items.append(req.to_dict())
        return items

    async def list_pending_async(
        self,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {
            item["approval_id"]: item
            for item in self.list_pending(session_id=session_id, tenant_id=tenant_id)
        }
        if self._coordination is not None:
            dist_items = await self._coordination.list_pending_approvals(
                session_id=session_id, tenant_id=tenant_id
            )
            for item in dist_items:
                merged.setdefault(item["approval_id"], item)
        return list(merged.values())

    def create_request(
        self,
        session_id: str,
        turn_id: str,
        tool_call: ToolCall,
        reason: str,
        *,
        requester_id: str = "anonymous",
        tenant_id: str = "default",
        required_roles: list[str] | None = None,
        policy_rule_id: str | None = None,
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
            requester_id=requester_id,
            tenant_id=tenant_id,
            required_roles=list(required_roles) if required_roles else ["approver", "admin"],
            policy_rule_id=policy_rule_id,
        )
        self._pending[approval_id] = req
        return req

    async def wait_for_decision(
        self,
        req: PendingApproval,
        cancellation: CancellationToken,
        timeout_seconds: float = 120.0,
    ) -> ApprovalDecision:
        if self._coordination is not None:
            await self._coordination.register_pending_approval(
                req.approval_id,
                req.to_dict(),
                ttl_seconds=max(30, int(timeout_seconds) + 30),
            )

        cancel_task = asyncio.create_task(cancellation.wait())
        dist_task: asyncio.Task[dict[str, Any] | None] | None = None
        wait_set: set[asyncio.Future[Any]] = {req.future, cancel_task}

        if self._coordination is not None:
            dist_task = asyncio.create_task(
                self._coordination.wait_for_approval(
                    approval_id=req.approval_id,
                    timeout_seconds=timeout_seconds,
                    cancellation_wait=cancellation.wait,
                )
            )
            wait_set.add(dist_task)

        try:
            done, _ = await asyncio.wait(
                wait_set,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done or cancellation.is_cancelled:
                raise TurnCancelledError(cancellation.reason or "cancelled_during_approval")
            if req.future in done:
                return req.future.result()
            if dist_task is not None and dist_task in done:
                dist_res = dist_task.result()
                if dist_res is not None:
                    return ApprovalDecision(
                        approval_id=req.approval_id,
                        approved=bool(dist_res.get("approved", False)),
                        reviewer=str(dist_res.get("reviewer", "user")),
                        note=str(dist_res.get("note", "")),
                    )
            # Fail-closed on timeout: never auto-approve sensitive actions
            return ApprovalDecision(
                approval_id=req.approval_id,
                approved=False,
                reviewer="timeout_guard",
                note=f"Approval timed out after {timeout_seconds}s (fail-closed).",
            )
        finally:
            cancel_task.cancel()
            if dist_task is not None:
                dist_task.cancel()
            self._pending.pop(req.approval_id, None)
            if self._coordination is not None:
                await self._coordination.remove_pending_approval(req.approval_id)

    def _validate_reviewer_authority(
        self,
        pending_meta: dict[str, Any],
        reviewer_principal: Principal,
        enforce_sod: bool,
    ) -> None:
        # 1. Tenant isolation (only global admin may cross tenants)
        pending_tenant = str(pending_meta.get("tenant_id") or "default")
        if not reviewer_principal.is_admin and reviewer_principal.tenant_id != pending_tenant:
            raise ApprovalAuthorizationError(
                f"Cross-tenant approval denied: reviewer tenant '{reviewer_principal.tenant_id}' "
                f"!= approval tenant '{pending_tenant}'."
            )

        # 2. RBAC role check
        required_roles = list(pending_meta.get("required_roles") or ["approver", "admin"])
        if not reviewer_principal.has_any_role(required_roles):
            raise ApprovalAuthorizationError(
                f"Reviewer '{reviewer_principal.user_id}' lacks required approval role(s): {required_roles}."
            )

        # 3. Four-Eyes Separation of Duties (SoD): requester cannot self-approve
        requester_id = str(pending_meta.get("requester_id") or "")
        if (
            enforce_sod
            and reviewer_principal.auth_method != "dev_anonymous"
            and requester_id
            and reviewer_principal.user_id == requester_id
        ):
            raise ApprovalAuthorizationError(
                f"Four-Eyes Separation of Duties violation: requester '{requester_id}' "
                "cannot approve their own high-risk action."
            )

    def resolve(
        self,
        approval_id: str,
        approved: bool,
        reviewer: str = "user",
        note: str = "",
    ) -> bool:
        decision = ApprovalDecision(
            approval_id=approval_id,
            approved=approved,
            reviewer=reviewer,
            note=note,
        )
        req = self._pending.get(approval_id)
        local_ok = False
        if req is not None and not req.future.done():
            req.future.set_result(decision)
            local_ok = True

        if self._coordination is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._coordination.resolve_approval(
                        approval_id,
                        {
                            "approval_id": approval_id,
                            "approved": approved,
                            "reviewer": reviewer,
                            "note": note,
                        },
                    )
                )
            except RuntimeError:
                pass
        return local_ok

    async def aresolve(
        self,
        approval_id: str,
        approved: bool,
        reviewer: str = "user",
        note: str = "",
        *,
        reviewer_principal: Principal | None = None,
        enforce_sod: bool | None = None,
    ) -> bool:
        # Look up pending metadata from local map or distributed store before resolving
        req = self._pending.get(approval_id)
        pending_meta: dict[str, Any] | None = req.to_dict() if req is not None else None
        if pending_meta is None and self._coordination is not None:
            pending_meta = await self._coordination.get_pending_approval(approval_id)

        if pending_meta is None:
            return False

        eff_reviewer = reviewer
        if reviewer_principal is not None:
            eff_sod = self.enforce_four_eyes if enforce_sod is None else enforce_sod
            self._validate_reviewer_authority(pending_meta, reviewer_principal, eff_sod)
            # Cryptographically bind reviewer identity from verified token (anti-spoofing!)
            if reviewer_principal.auth_method != "dev_anonymous":
                eff_reviewer = reviewer_principal.user_id

        decision = ApprovalDecision(
            approval_id=approval_id,
            approved=approved,
            reviewer=eff_reviewer,
            note=note,
        )
        local_ok = False
        if req is not None and not req.future.done():
            req.future.set_result(decision)
            local_ok = True

        dist_ok = False
        if self._coordination is not None:
            dist_ok = await self._coordination.resolve_approval(
                approval_id,
                {
                    "approval_id": approval_id,
                    "approved": approved,
                    "reviewer": eff_reviewer,
                    "note": note,
                },
            )
        return local_ok or dist_ok


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
    Incremental state & dialogue summary compressor supporting both deterministic folding
    and optional fast-model structured entity/KPI summarization (`acompress`).
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

    def _split_for_compression(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[str], list[Message], list[Message]] | None:
        system_msgs: list[Message] = []
        prior_summary_lines: list[str] = []
        idx = 0
        while idx < len(messages) and messages[idx].role == "system":
            if messages[idx].content.startswith("[Incremental Context Summary]"):
                for line in messages[idx].content.splitlines()[1:]:
                    if line.strip():
                        prior_summary_lines.append(line.strip())
            else:
                system_msgs.append(messages[idx])
            idx += 1

        body = messages[idx:]
        if len(body) <= self.keep_recent_messages:
            return None

        split_at = len(body) - self.keep_recent_messages
        # Avoid splitting right between an assistant tool_call and its following tool results
        while split_at > 0 and body[split_at].role == "tool":
            split_at -= 1
        if split_at <= 0:
            return None

        return system_msgs, prior_summary_lines, body[:split_at], body[split_at:]

    def compress(self, messages: list[Message]) -> tuple[list[Message], dict[str, int]]:
        tokens_before = self.estimate_total_tokens(messages)
        parts = self._split_for_compression(messages)
        if parts is None:
            return messages, {
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "messages_folded": 0,
            }

        system_msgs, prior_summary_lines, to_fold, recent = parts
        bullet_lines: list[str] = list(prior_summary_lines[-6:])
        for m in to_fold:
            snippet = m.content.strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            if m.role == "tool":
                bullet_lines.append(f"- [tool:{m.name}] {snippet}")
            elif m.tool_calls:
                names = ", ".join(tc.name for tc in m.tool_calls)
                bullet_lines.append(f"- [assistant->tools:{names}] {snippet}")
            else:
                bullet_lines.append(f"- [{m.role}] {snippet}")

        summary_msg = Message(
            role="system",
            content="[Incremental Context Summary]\n" + "\n".join(bullet_lines[-16:]),
        )
        compressed = [*system_msgs, summary_msg, *recent]
        tokens_after = self.estimate_total_tokens(compressed)
        return compressed, {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "messages_folded": len(to_fold),
        }

    async def acompress(
        self,
        messages: list[Message],
        summary_llm: BaseLLMClient | None = None,
    ) -> tuple[list[Message], dict[str, int]]:
        """LLM-assisted structured entity-preserving summary with deterministic fallback."""
        if summary_llm is None:
            return self.compress(messages)

        tokens_before = self.estimate_total_tokens(messages)
        parts = self._split_for_compression(messages)
        if parts is None:
            return messages, {
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "messages_folded": 0,
            }

        system_msgs, prior_summary_lines, to_fold, recent = parts
        transcript = "\n".join(
            f"[{m.role}{':' + m.name if m.name else ''}] {m.content[:400]}"
            for m in to_fold
        )
        prompt = (
            "请将以下历史对话压缩为精简的要点摘要（不超过 8 条），必须严格保留所有业务数值、阈值、"
            "工单/设备 ID、SQL 表名、报错原因及已确认结论：\n"
            + ("\n既往摘要：\n" + "\n".join(prior_summary_lines) if prior_summary_lines else "")
            + f"\n新增历史：\n{transcript}"
        )
        try:
            resp = await summary_llm.complete([Message(role="user", content=prompt)])
            if resp.content.strip():
                summary_msg = Message(
                    role="system",
                    content="[Incremental Context Summary]\n" + resp.content.strip(),
                )
                compressed = [*system_msgs, summary_msg, *recent]
                tokens_after = self.estimate_total_tokens(compressed)
                return compressed, {
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "messages_folded": len(to_fold),
                }
        except Exception:
            pass
        return self.compress(messages)
