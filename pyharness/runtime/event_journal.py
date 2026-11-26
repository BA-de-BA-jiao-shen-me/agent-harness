from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from pyharness.runtime.journal_store import (
    AuditVerificationResult,
    BranchInfo,
    JournalStorageBackend,
    PostgresJournalStore,
    SQLiteJournalStore,
    SessionRecord,
    SessionTreeResult,
    create_journal_store,
)
from pyharness.runtime.schemas import EventType, HarnessEvent, Message
from pyharness.security.auth import current_principal

if TYPE_CHECKING:
    from pyharness.infra.coordination import CoordinationBackend

_DROPPABLE_STREAM_EVENTS = {EventType.TOKEN_DELTA, EventType.THINKING}
_TERMINAL_EVENTS = {EventType.TURN_COMPLETE, EventType.TURN_CANCELLED}


class EventJournal:
    """
    Append-only Event Sourcing Journal inspired by DeepSeek Harness (event-journal.mjs).

    Enterprise capabilities:
    1. Pluggable durable storage Seam (`SQLiteJournalStore` WAL + `PostgresJournalStore` multi-Pod).
    2. Tamper-evident cryptographic SHA-256 hash chain per session (`verify_audit_chain`).
    3. Multi-tenant session ownership & point-in-time session branching (`fork_session`).
    4. Live local + cross-Pod Redis Pub/Sub event streaming with `after_seq` deduplication
       (`stream_live_events`) and cold-restart dialogue reconstruction (`reconstruct_messages`).
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        coordination: CoordinationBackend | None = None,
        store: JournalStorageBackend | None = None,
        database_url: str | None = None,
    ) -> None:
        self.store: JournalStorageBackend = store or create_journal_store(
            db_path=db_path, database_url=database_url
        )
        self._coordination = coordination
        self._subscribers: dict[str, list[asyncio.Queue[HarnessEvent]]] = {}

    @property
    def backend_name(self) -> str:
        return self.store.backend_name

    @staticmethod
    def _dict_to_event(row: dict[str, Any]) -> HarnessEvent:
        return HarnessEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            seq=int(row["seq"]),
            event_type=EventType(row["event_type"]),
            timestamp_ms=int(row["timestamp_ms"]),
            payload=dict(row.get("payload") or {}),
            tenant_id=str(row.get("tenant_id") or "default"),
            user_id=str(row.get("user_id") or "anonymous"),
            prev_hash=str(row.get("prev_hash") or ""),
            event_hash=str(row.get("event_hash") or ""),
            parent_event_id=row.get("parent_event_id"),
            branch_name=str(row.get("branch_name") or "main"),
        )

    @staticmethod
    def _enqueue_safely(q: asyncio.Queue[HarnessEvent], event: HarnessEvent) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Never drop lifecycle-critical events (TURN_COMPLETE / APPROVAL_REQUIRED / etc.)
            if event.event_type in _DROPPABLE_STREAM_EVENTS:
                return
            try:
                q.get_nowait()
                q.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def ensure_session(
        self,
        session_id: str,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
        parent_session_id: str | None = None,
        forked_from_seq: int | None = None,
    ) -> SessionRecord:
        principal = current_principal.get()
        eff_tenant = tenant_id or (principal.tenant_id if principal else "default")
        eff_owner = owner_user_id or (principal.user_id if principal else "anonymous")
        return await self.store.ensure_session(
            session_id=session_id,
            tenant_id=eff_tenant,
            owner_user_id=eff_owner,
            parent_session_id=parent_session_id,
            forked_from_seq=forked_from_seq,
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return await self.store.get_session(session_id)

    async def list_sessions(
        self,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SessionRecord]:
        return await self.store.list_sessions(tenant_id=tenant_id, owner_user_id=owner_user_id)

    async def emit(
        self,
        session_id: str,
        turn_id: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        branch_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> HarnessEvent:
        """Persist hash-chained event and fan out to local + distributed subscribers."""
        principal = current_principal.get()
        eff_tenant = tenant_id or (principal.tenant_id if principal else "default")
        eff_user = user_id or (principal.user_id if principal else "anonymous")

        row = await self.store.append_event(
            session_id=session_id,
            turn_id=turn_id,
            event_type=event_type.value,
            payload=payload or {},
            tenant_id=eff_tenant,
            user_id=eff_user,
            branch_name=branch_name,
            parent_event_id=parent_event_id,
        )
        event = self._dict_to_event(row)

        for q in list(self._subscribers.get(session_id, [])):
            self._enqueue_safely(q, event)

        if self._coordination is not None:
            await self._coordination.publish_event(session_id, event)
        return event

    def subscribe(self, session_id: str, maxsize: int = 512) -> asyncio.Queue[HarnessEvent]:
        q: asyncio.Queue[HarnessEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[HarnessEvent]) -> None:
        subs = self._subscribers.get(session_id, [])
        if queue in subs:
            subs.remove(queue)
        if not subs:
            self._subscribers.pop(session_id, None)

    def replay_session(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[HarnessEvent]:
        """Synchronous replay helper (SQLite WAL) preserving backward compatibility."""
        rows = self.store.load_events_sync(session_id, after_seq=after_seq, up_to_seq=up_to_seq)
        return [self._dict_to_event(r) for r in rows]

    async def areplay_session(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[HarnessEvent]:
        """Async replay supporting both SQLite WAL and PostgreSQL multi-Pod storage."""
        rows = await self.store.load_events(session_id, after_seq=after_seq, up_to_seq=up_to_seq)
        return [self._dict_to_event(r) for r in rows]

    async def verify_audit_chain(self, session_id: str) -> AuditVerificationResult:
        """Verify cryptographic SHA-256 tamper-evident hash chain across all events in a session."""
        return await self.store.verify_chain(session_id)

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        at_seq: int | None = None,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> tuple[SessionRecord, int]:
        """Branch a session at `at_seq` into a new independent session with full lineage."""
        principal = current_principal.get()
        eff_tenant = tenant_id or (principal.tenant_id if principal else "default")
        eff_owner = owner_user_id or (principal.user_id if principal else "anonymous")
        return await self.store.fork_session(
            source_session_id=source_session_id,
            new_session_id=new_session_id,
            at_seq=at_seq,
            tenant_id=eff_tenant,
            owner_user_id=eff_owner,
        )

    async def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_event_id: str | None = None,
        *,
        switch: bool = True,
    ) -> BranchInfo:
        """Create a zero-copy in-place branch inside `session_id`."""
        return await self.store.create_branch(
            session_id=session_id,
            branch_name=branch_name,
            from_event_id=from_event_id,
            switch=switch,
        )

    async def switch_branch(
        self,
        session_id: str,
        branch_name: str,
    ) -> BranchInfo:
        """Switch the active branch pointer for `session_id`."""
        return await self.store.switch_branch(
            session_id=session_id,
            branch_name=branch_name,
        )

    async def get_branch_events(
        self,
        session_id: str,
        branch_name: str | None = None,
    ) -> list[HarnessEvent]:
        """Return chronological root-to-leaf events along a branch lineage."""
        rows = await self.store.get_branch_events(
            session_id=session_id,
            branch_name=branch_name,
        )
        return [self._dict_to_event(r) for r in rows]

    async def get_session_tree(self, session_id: str) -> SessionTreeResult:
        """Return full branch metadata and DAG node topology for `session_id`."""
        return await self.store.get_session_tree(session_id=session_id)

    async def stream_live_events(
        self,
        session_id: str,
        after_seq: int = 0,
        follow_if_active: bool = True,
        idle_timeout_seconds: float = 15.0,
    ) -> AsyncIterator[HarnessEvent]:
        """
        Replay durable events after `after_seq` and seamlessly attach to live local +
        distributed Redis Pub/Sub events with strict `seq` deduplication (`Last-Event-ID`).
        """
        local_q = self.subscribe(session_id)
        merged_q: asyncio.Queue[HarnessEvent] = asyncio.Queue(maxsize=512)
        pump_tasks: list[asyncio.Task[None]] = []

        async def _pump_local() -> None:
            while True:
                ev = await local_q.get()
                self._enqueue_safely(merged_q, ev)

        async def _pump_distributed() -> None:
            if self._coordination is None:
                return
            async for ev in self._coordination.subscribe_events(session_id):
                self._enqueue_safely(merged_q, ev)

        pump_tasks.append(asyncio.create_task(_pump_local()))
        if self._coordination is not None:
            pump_tasks.append(asyncio.create_task(_pump_distributed()))

        try:
            last_seen_seq = after_seq
            historical = await self.areplay_session(session_id, after_seq=after_seq)
            saw_terminal = False
            for ev in historical:
                if ev.seq > last_seen_seq:
                    last_seen_seq = ev.seq
                    if ev.event_type in _TERMINAL_EVENTS:
                        saw_terminal = True
                    yield ev

            if not follow_if_active:
                return

            is_active = (
                await self._coordination.is_session_active(session_id)
                if self._coordination is not None
                else False
            )
            if not is_active and (saw_terminal or historical):
                # Drain any already-buffered events without blocking if no turn is active
                while not merged_q.empty():
                    ev = merged_q.get_nowait()
                    if ev.seq > last_seen_seq:
                        last_seen_seq = ev.seq
                        yield ev
                return

            # Follow live events until terminal event or session goes idle/inactive
            idle_elapsed = 0.0
            poll_step = 0.25
            while True:
                try:
                    ev = await asyncio.wait_for(merged_q.get(), timeout=poll_step)
                    idle_elapsed = 0.0
                    if ev.seq <= last_seen_seq:
                        continue
                    last_seen_seq = ev.seq
                    yield ev
                    if ev.event_type in _TERMINAL_EVENTS:
                        break
                except asyncio.TimeoutError:
                    idle_elapsed += poll_step
                    if self._coordination is not None and not await self._coordination.is_session_active(session_id):
                        # Drain any final events before exiting
                        while not merged_q.empty():
                            ev = merged_q.get_nowait()
                            if ev.seq > last_seen_seq:
                                last_seen_seq = ev.seq
                                yield ev
                        break
                    if idle_elapsed >= idle_timeout_seconds:
                        break
        finally:
            for t in pump_tasks:
                t.cancel()
            for t in pump_tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            self.unsubscribe(session_id, local_q)

    def _build_messages_from_events(
        self,
        events: list[HarnessEvent],
        default_system_prompt: str,
    ) -> list[Message] | None:
        if not events:
            return None

        messages: list[Message] = [Message(role="system", content=default_system_prompt)]

        anchor_idx: int | None = None
        for idx in range(len(events) - 1, -1, -1):
            if events[idx].event_type == EventType.COMPACTION_ANCHOR:
                anchor_idx = idx
                break

        if anchor_idx is not None:
            anchor_summary = str(events[anchor_idx].payload.get("summary") or "").strip()
            if anchor_summary:
                if anchor_summary.startswith("[Incremental Context Summary]"):
                    summary_content = anchor_summary
                else:
                    summary_content = f"[Incremental Context Summary]\n{anchor_summary}"
                messages.append(Message(role="system", content=summary_content))
            effective_events = events[anchor_idx + 1 :]
        else:
            effective_events = events

        turn_buffers: dict[str, list[str]] = {}
        for ev in effective_events:
            if ev.event_type == EventType.TURN_START:
                user_in = str(ev.payload.get("user_input") or "")
                if user_in:
                    messages.append(Message(role="user", content=user_in))
                turn_buffers[ev.turn_id] = []
            elif ev.event_type == EventType.TOKEN_DELTA:
                chunk = str(ev.payload.get("content") or "")
                turn_buffers.setdefault(ev.turn_id, []).append(chunk)
            elif ev.event_type in (EventType.TURN_STEERED, EventType.FOLLOWUP_ENQUEUED):
                steer_text = str(ev.payload.get("content") or "")
                if steer_text.strip():
                    pending = "".join(turn_buffers.pop(ev.turn_id, []))
                    if pending:
                        messages.append(Message(role="assistant", content=pending))
                    messages.append(Message(role="user", content=steer_text))
            elif ev.event_type == EventType.TURN_COMPLETE:
                assembled = "".join(turn_buffers.pop(ev.turn_id, []))
                if assembled:
                    messages.append(Message(role="assistant", content=assembled))
        return messages if len(messages) > 1 else None

    def reconstruct_messages(
        self,
        session_id: str,
        default_system_prompt: str,
    ) -> list[Message] | None:
        """Reconstruct dialogue history synchronously (SQLite WAL)."""
        return self._build_messages_from_events(
            self.replay_session(session_id), default_system_prompt
        )

    async def areconstruct_branch_messages(
        self,
        session_id: str,
        default_system_prompt: str,
        branch_name: str | None = None,
    ) -> list[Message] | None:
        """Reconstruct dialogue history for a specific (or active) branch asynchronously."""
        events = await self.get_branch_events(session_id, branch_name=branch_name)
        return self._build_messages_from_events(events, default_system_prompt)

    async def areconstruct_messages(
        self,
        session_id: str,
        default_system_prompt: str,
    ) -> list[Message] | None:
        """Reconstruct dialogue history asynchronously (SQLite WAL & PostgreSQL)."""
        tree = await self.get_session_tree(session_id)
        if len(tree.branches) > 1 or tree.active_branch != "main":
            events = await self.get_branch_events(session_id)
        else:
            events = await self.areplay_session(session_id)
        return self._build_messages_from_events(events, default_system_prompt)

    def _summarize_events(
        self, session_id: str, turn_id: str, events: list[HarnessEvent]
    ) -> dict[str, Any]:
        turn_events = [e for e in events if e.turn_id == turn_id]
        if not turn_events:
            return {"session_id": session_id, "turn_id": turn_id, "event_count": 0}

        tool_spans: list[dict[str, Any]] = []
        approvals = 0
        compressions = 0
        self_corrections = 0
        policy_events = 0
        security_alerts = 0
        subagents_spawned = 0
        final_status = "running"

        for e in turn_events:
            if e.event_type == EventType.TOOL_RESULT:
                tool_spans.append(
                    {
                        "tool": e.payload.get("tool_name"),
                        "duration_ms": e.payload.get("duration_ms", 0.0),
                        "is_error": e.payload.get("is_error", False),
                    }
                )
            elif e.event_type == EventType.APPROVAL_REQUIRED:
                approvals += 1
            elif e.event_type == EventType.CONTEXT_COMPRESSED:
                compressions += 1
            elif e.event_type == EventType.SELF_CORRECTION:
                self_corrections += 1
            elif e.event_type == EventType.POLICY_EVALUATED:
                policy_events += 1
            elif e.event_type == EventType.SECURITY_ALERT:
                security_alerts += 1
            elif e.event_type == EventType.SUBAGENT_START:
                subagents_spawned += 1
            elif e.event_type == EventType.TURN_COMPLETE:
                final_status = e.payload.get("status", "completed")
            elif e.event_type == EventType.TURN_CANCELLED:
                final_status = "cancelled"

        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "event_count": len(turn_events),
            "status": final_status,
            "wall_time_ms": turn_events[-1].timestamp_ms - turn_events[0].timestamp_ms,
            "tool_spans": tool_spans,
            "approvals_requested": approvals,
            "context_compressions": compressions,
            "self_corrections": self_corrections,
            "policy_evaluations": policy_events,
            "security_alerts": security_alerts,
            "subagents_spawned": subagents_spawned,
        }

    def build_trace_summary(self, session_id: str, turn_id: str) -> dict[str, Any]:
        return self._summarize_events(session_id, turn_id, self.replay_session(session_id))

    async def abuild_trace_summary(self, session_id: str, turn_id: str) -> dict[str, Any]:
        return self._summarize_events(session_id, turn_id, await self.areplay_session(session_id))

    def close(self) -> None:
        self.store.close()

    async def aclose(self) -> None:
        await self.store.aclose()


__all__ = [
    "AuditVerificationResult",
    "BranchInfo",
    "EventJournal",
    "JournalStorageBackend",
    "PostgresJournalStore",
    "SQLiteJournalStore",
    "SessionRecord",
    "SessionTreeResult",
]
