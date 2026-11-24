from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from pyharness.runtime.schemas import EventType, HarnessEvent


class EventJournal:
    """
    Append-only Event Sourcing Journal inspired by DeepSeek Harness (event-journal.mjs).

    Responsibilities:
    1. Persist every runtime event in strict monotonic sequence order per session (SQLite).
    2. Broadcast live events to attached async subscribers (ready for FastAPI SSE streaming).
    3. Reconstruct per-turn trace waterfalls (tool latencies, compression counts, token spend).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._seq_counters: dict[str, int] = {}
        self._subscribers: dict[str, list[asyncio.Queue[HarnessEvent]]] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_session_seq
            ON harness_events (session_id, seq)
            """
        )
        self._conn.commit()

    def _next_seq(self, session_id: str) -> int:
        if session_id not in self._seq_counters:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM harness_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._seq_counters[session_id] = int(row["max_seq"])
        self._seq_counters[session_id] += 1
        return self._seq_counters[session_id]

    async def emit(
        self,
        session_id: str,
        turn_id: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> HarnessEvent:
        """Persist event and notify all live subscribers for this session."""
        async with self._lock:
            seq = self._next_seq(session_id)
            event = HarnessEvent(
                session_id=session_id,
                turn_id=turn_id,
                seq=seq,
                event_type=event_type,
                payload=payload or {},
            )
            self._conn.execute(
                """
                INSERT INTO harness_events
                (event_id, session_id, turn_id, seq, event_type, timestamp_ms, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.session_id,
                    event.turn_id,
                    event.seq,
                    event.event_type.value,
                    event.timestamp_ms,
                    json.dumps(event.payload, ensure_ascii=False),
                ),
            )
            self._conn.commit()

            for q in list(self._subscribers.get(session_id, [])):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass
            return event

    def subscribe(self, session_id: str, maxsize: int = 256) -> asyncio.Queue[HarnessEvent]:
        q: asyncio.Queue[HarnessEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[HarnessEvent]) -> None:
        subs = self._subscribers.get(session_id, [])
        if queue in subs:
            subs.remove(queue)

    def replay_session(
        self, session_id: str, after_seq: int = 0
    ) -> list[HarnessEvent]:
        """Replay durable events for reconnection recovery or audit inspection."""
        rows = self._conn.execute(
            """
            SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms, payload_json
            FROM harness_events
            WHERE session_id = ? AND seq > ?
            ORDER BY seq ASC
            """,
            (session_id, after_seq),
        ).fetchall()
        return [
            HarnessEvent(
                event_id=r["event_id"],
                session_id=r["session_id"],
                turn_id=r["turn_id"],
                seq=r["seq"],
                event_type=EventType(r["event_type"]),
                timestamp_ms=r["timestamp_ms"],
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    def build_trace_summary(self, session_id: str, turn_id: str) -> dict[str, Any]:
        """Compute structured observability metrics for a specific turn."""
        events = [e for e in self.replay_session(session_id) if e.turn_id == turn_id]
        if not events:
            return {"session_id": session_id, "turn_id": turn_id, "event_count": 0}

        tool_spans: list[dict[str, Any]] = []
        approvals = 0
        compressions = 0
        self_corrections = 0
        final_status = "running"

        for e in events:
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
            elif e.event_type == EventType.TURN_COMPLETE:
                final_status = e.payload.get("status", "completed")
            elif e.event_type == EventType.TURN_CANCELLED:
                final_status = "cancelled"

        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "event_count": len(events),
            "status": final_status,
            "wall_time_ms": events[-1].timestamp_ms - events[0].timestamp_ms,
            "tool_spans": tool_spans,
            "approvals_requested": approvals,
            "context_compressions": compressions,
            "self_corrections": self_corrections,
        }

    def close(self) -> None:
        self._conn.close()
