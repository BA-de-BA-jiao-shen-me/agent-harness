from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

GENESIS_HASH = "0" * 64
SESSION_FORKED_EVENT_TYPE = "session_forked"


class SessionRecord(BaseModel):
    """Durable session ownership and lineage metadata for multi-tenant isolation and branching."""

    session_id: str
    tenant_id: str = "default"
    owner_user_id: str = "anonymous"
    parent_session_id: str | None = None
    forked_from_seq: int | None = None
    active_branch: str = "main"
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class BranchInfo(BaseModel):
    """Metadata describing an in-place branch within a session tree."""

    branch_name: str
    base_event_id: str | None = None
    head_event_id: str | None = None
    event_count: int = 0
    is_active: bool = False
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class SessionTreeResult(BaseModel):
    """Complete in-place session DAG / branch tree topology."""

    session_id: str
    active_branch: str = "main"
    branches: list[BranchInfo] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)


class AuditVerificationResult(BaseModel):
    """Cryptographic hash-chain audit verification outcome for a session journal."""

    session_id: str
    event_count: int
    valid: bool
    broken_at_seq: int | None = None
    detail: str = ""


def _normalize_event_type(event_type: str | Enum) -> str:
    if isinstance(event_type, Enum):
        return str(event_type.value)
    return str(event_type)


def _canonical_payload_json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)


def compute_event_hash(
    prev_hash: str,
    session_id: str,
    seq: int,
    event_type: str,
    timestamp_ms: int,
    payload_json: str,
    tenant_id: str,
    user_id: str,
) -> str:
    """
    Compute the canonical SHA-256 tamper-evident hash chain digest for a journal event.
    """
    ev_type = _normalize_event_type(event_type)
    canonical = (
        f"{prev_hash}|{session_id}|{seq}|{ev_type}|"
        f"{timestamp_ms}|{tenant_id}|{user_id}|{payload_json}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_rows(session_id: str, rows: list[dict[str, Any]]) -> AuditVerificationResult:
    if not rows:
        return AuditVerificationResult(
            session_id=session_id,
            event_count=0,
            valid=True,
            broken_at_seq=None,
            detail="No events recorded for session.",
        )

    expected_prev = GENESIS_HASH
    expected_seq = 1
    for row in rows:
        seq = int(row["seq"])
        prev_hash = str(row["prev_hash"] or "")
        event_hash = str(row["event_hash"] or "")
        if prev_hash != expected_prev:
            return AuditVerificationResult(
                session_id=session_id,
                event_count=len(rows),
                valid=False,
                broken_at_seq=seq,
                detail=f"Hash chain discontinuity at seq={seq}: expected prev_hash={expected_prev[:12]}..., got {prev_hash[:12]}...",
            )
        recomputed = compute_event_hash(
            prev_hash=prev_hash,
            session_id=session_id,
            seq=seq,
            event_type=str(row["event_type"]),
            timestamp_ms=int(row["timestamp_ms"]),
            payload_json=str(row["payload_json"]),
            tenant_id=str(row["tenant_id"]),
            user_id=str(row["user_id"]),
        )
        if event_hash != recomputed:
            return AuditVerificationResult(
                session_id=session_id,
                event_count=len(rows),
                valid=False,
                broken_at_seq=seq,
                detail=f"Tampered event digest at seq={seq}: stored hash does not match canonical SHA-256.",
            )
        if seq != expected_seq:
            return AuditVerificationResult(
                session_id=session_id,
                event_count=len(rows),
                valid=False,
                broken_at_seq=seq,
                detail=f"Sequence gap detected: expected seq={expected_seq}, got seq={seq}.",
            )
        expected_prev = event_hash
        expected_seq += 1

    return AuditVerificationResult(
        session_id=session_id,
        event_count=len(rows),
        valid=True,
        broken_at_seq=None,
        detail=f"Verified {len(rows)} events with intact SHA-256 hash chain.",
    )


def _row_field(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _summarize_event_node(ev: dict[str, Any], max_len: int = 120) -> str:
    payload = ev.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    for key in (
        "user_input",
        "summary",
        "tool_name",
        "status",
        "content",
        "reason",
        "target_branch",
        "branch_name",
    ):
        val = payload.get(key)
        if val is not None:
            text = str(val).strip()
            if text:
                if key == "tool_name" and payload.get("is_error"):
                    text = f"{text} (error)"
                elif key == "status":
                    text = f"status={text}"
                if len(text) > max_len:
                    return text[: max_len - 1] + "…"
                return text
    return ""


def _walk_branch_chain(
    events: list[dict[str, Any]],
    head_event_id: str | None,
) -> list[dict[str, Any]]:
    if not head_event_id or not events:
        return []
    by_id = {str(ev["event_id"]): ev for ev in events}
    if head_event_id not in by_id:
        return []
    chain: list[dict[str, Any]] = []
    visited: set[str] = set()
    cur_id: str | None = head_event_id
    while cur_id is not None and cur_id not in visited:
        ev = by_id.get(cur_id)
        if ev is None:
            break
        visited.add(cur_id)
        chain.append(ev)
        parent_id = ev.get("parent_event_id")
        cur_id = str(parent_id) if parent_id else None
    chain.reverse()
    return chain


def _resolve_branch_events(
    all_events: list[dict[str, Any]],
    active_branch: str,
    branches_by_name: dict[str, dict[str, Any]],
    branch_name: str | None = None,
) -> list[dict[str, Any]]:
    target = (branch_name or "").strip() or active_branch or "main"
    branch_row = branches_by_name.get(target)
    if branch_row is not None:
        head_id = branch_row.get("head_event_id")
        if not head_id:
            return []
        return _walk_branch_chain(all_events, str(head_id))
    # Legacy fallback when harness_branches has no entry for target yet (e.g. "main")
    return [ev for ev in all_events if str(ev.get("branch_name") or "main") == target]


def _build_session_tree(
    session_id: str,
    active_branch: str,
    session_created_at_ms: int | None,
    branch_rows: list[dict[str, Any]],
    all_events: list[dict[str, Any]],
) -> SessionTreeResult:
    eff_active = active_branch or "main"
    branches_by_name: dict[str, dict[str, Any]] = {
        str(b["branch_name"]): dict(b) for b in branch_rows
    }

    if "main" not in branches_by_name and all_events:
        main_events = [
            ev for ev in all_events if str(ev.get("branch_name") or "main") == "main"
        ]
        head_id = str(main_events[-1]["event_id"]) if main_events else None
        created_ms = (
            int(session_created_at_ms)
            if session_created_at_ms is not None
            else int(all_events[0]["timestamp_ms"])
        )
        branches_by_name["main"] = {
            "branch_name": "main",
            "base_event_id": None,
            "head_event_id": head_id,
            "created_at_ms": created_ms,
        }

    ordered_names = sorted(
        branches_by_name.keys(),
        key=lambda name: (
            0 if name == "main" else 1,
            int(branches_by_name[name].get("created_at_ms") or 0),
            name,
        ),
    )

    branch_infos: list[BranchInfo] = []
    for name in ordered_names:
        row = branches_by_name[name]
        head_id = row.get("head_event_id")
        chain = _walk_branch_chain(all_events, str(head_id)) if head_id else []
        if not chain and name == "main" and head_id is None and all_events:
            chain = [
                ev for ev in all_events if str(ev.get("branch_name") or "main") == "main"
            ]
        branch_infos.append(
            BranchInfo(
                branch_name=name,
                base_event_id=row.get("base_event_id"),
                head_event_id=head_id,
                event_count=len(chain),
                is_active=(name == eff_active),
                created_at_ms=int(row.get("created_at_ms") or int(time.time() * 1000)),
            )
        )

    nodes: list[dict[str, Any]] = [
        {
            "event_id": ev["event_id"],
            "seq": int(ev["seq"]),
            "turn_id": ev["turn_id"],
            "event_type": ev["event_type"],
            "parent_event_id": ev.get("parent_event_id"),
            "branch_name": str(ev.get("branch_name") or "main"),
            "timestamp_ms": int(ev["timestamp_ms"]),
            "summary": _summarize_event_node(ev),
        }
        for ev in all_events
    ]

    return SessionTreeResult(
        session_id=session_id,
        active_branch=eff_active,
        branches=branch_infos,
        nodes=nodes,
    )


class JournalStorageBackend(ABC):
    """
    Pluggable durable persistence backend for EventJournal supporting:
    - Multi-tenant session ownership & fork lineage (`harness_sessions`).
    - Strict monotonic per-session event sequencing (`harness_events`).
    - In-place zero-copy session branching (`harness_branches`).
    - Tamper-evident cryptographic SHA-256 hash chaining (`verify_chain`).
    - Point-in-time session forking (`fork_session`).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ensure_session(
        self,
        session_id: str,
        tenant_id: str = "default",
        owner_user_id: str = "anonymous",
        parent_session_id: str | None = None,
        forked_from_seq: int | None = None,
    ) -> SessionRecord:
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def list_sessions(
        self,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SessionRecord]:
        raise NotImplementedError

    @abstractmethod
    async def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
        branch_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_events_sync(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def load_events(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def verify_chain(self, session_id: str) -> AuditVerificationResult:
        raise NotImplementedError

    @abstractmethod
    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        at_seq: int | None,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[SessionRecord, int]:
        raise NotImplementedError

    @abstractmethod
    async def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_event_id: str | None = None,
        *,
        switch: bool = True,
    ) -> BranchInfo:
        raise NotImplementedError

    @abstractmethod
    async def switch_branch(
        self,
        session_id: str,
        branch_name: str,
    ) -> BranchInfo:
        raise NotImplementedError

    @abstractmethod
    async def get_branch_events(
        self,
        session_id: str,
        branch_name: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_session_tree(self, session_id: str) -> SessionTreeResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


class SQLiteJournalStore(JournalStorageBackend):
    """
    WAL-hardened SQLite journal store with non-blocking `asyncio.to_thread()` I/O offloading,
    automatic forward-compatible schema migration, and SHA-256 audit hash chaining.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        self._init_schema()

    @property
    def backend_name(self) -> str:
        return "sqlite_wal"

    def _configure_pragmas(self) -> None:
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_user_id TEXT NOT NULL DEFAULT 'anonymous',
                    parent_session_id TEXT,
                    forked_from_seq INTEGER,
                    active_branch TEXT NOT NULL DEFAULT 'main',
                    created_at_ms INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    prev_hash TEXT NOT NULL DEFAULT '',
                    event_hash TEXT NOT NULL DEFAULT '',
                    parent_event_id TEXT,
                    branch_name TEXT NOT NULL DEFAULT 'main'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_branches (
                    session_id TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    base_event_id TEXT,
                    head_event_id TEXT,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (session_id, branch_name)
                )
                """
            )

            # Auto-migrate legacy harness_sessions tables missing active_branch
            session_cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(harness_sessions)").fetchall()
            }
            if "active_branch" not in session_cols:
                self._conn.execute(
                    "ALTER TABLE harness_sessions ADD COLUMN active_branch TEXT NOT NULL DEFAULT 'main'"
                )

            # Auto-migrate legacy harness_events tables missing audit or tree columns
            cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(harness_events)").fetchall()
            }
            added_parent_col = "parent_event_id" not in cols
            migrations = (
                ("tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("user_id", "TEXT NOT NULL DEFAULT 'anonymous'"),
                ("prev_hash", "TEXT NOT NULL DEFAULT ''"),
                ("event_hash", "TEXT NOT NULL DEFAULT ''"),
                ("parent_event_id", "TEXT"),
                ("branch_name", "TEXT NOT NULL DEFAULT 'main'"),
            )
            for col_name, col_ddl in migrations:
                if col_name not in cols:
                    self._conn.execute(
                        f"ALTER TABLE harness_events ADD COLUMN {col_name} {col_ddl}"
                    )

            if added_parent_col:
                legacy_rows = self._conn.execute(
                    "SELECT event_id, session_id FROM harness_events ORDER BY session_id ASC, seq ASC"
                ).fetchall()
                prev_by_sess: dict[str, str] = {}
                for lr in legacy_rows:
                    sid = str(lr["session_id"])
                    eid = str(lr["event_id"])
                    prev_eid = prev_by_sess.get(sid)
                    if prev_eid is not None:
                        self._conn.execute(
                            "UPDATE harness_events SET parent_event_id = ? WHERE event_id = ?",
                            (prev_eid, eid),
                        )
                    prev_by_sess[sid] = eid

            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_seq
                ON harness_events (session_id, seq)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_branch
                ON harness_events (session_id, branch_name, seq)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_tenant_owner
                ON harness_sessions (tenant_id, owner_user_id)
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            owner_user_id=row["owner_user_id"],
            parent_session_id=row["parent_session_id"],
            forked_from_seq=row["forked_from_seq"],
            active_branch=str(_row_field(row, "active_branch") or "main"),
            created_at_ms=int(row["created_at_ms"]),
        )

    @staticmethod
    def _row_to_event_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "seq": int(row["seq"]),
            "event_type": row["event_type"],
            "timestamp_ms": int(row["timestamp_ms"]),
            "payload": json.loads(row["payload_json"]),
            "tenant_id": row["tenant_id"],
            "user_id": row["user_id"],
            "prev_hash": row["prev_hash"],
            "event_hash": row["event_hash"],
            "parent_event_id": _row_field(row, "parent_event_id", None),
            "branch_name": str(_row_field(row, "branch_name") or "main"),
        }

    def _ensure_session_sync(
        self,
        session_id: str,
        tenant_id: str = "default",
        owner_user_id: str = "anonymous",
        parent_session_id: str | None = None,
        forked_from_seq: int | None = None,
    ) -> SessionRecord:
        with self._lock:
            now_ms = int(time.time() * 1000)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO harness_sessions
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                VALUES (?, ?, ?, ?, ?, 'main', ?)
                """,
                (
                    session_id,
                    tenant_id,
                    owner_user_id,
                    parent_session_id,
                    forked_from_seq,
                    now_ms,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            assert row is not None
            return self._row_to_session(row)

    async def ensure_session(
        self,
        session_id: str,
        tenant_id: str = "default",
        owner_user_id: str = "anonymous",
        parent_session_id: str | None = None,
        forked_from_seq: int | None = None,
    ) -> SessionRecord:
        return await asyncio.to_thread(
            self._ensure_session_sync,
            session_id,
            tenant_id,
            owner_user_id,
            parent_session_id,
            forked_from_seq,
        )

    def _get_session_sync(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return self._row_to_session(row) if row is not None else None

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return await asyncio.to_thread(self._get_session_sync, session_id)

    def _list_sessions_sync(
        self,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SessionRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            params.append(owner_user_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                {where_sql}
                ORDER BY created_at_ms DESC, session_id ASC
                """,
                params,
            ).fetchall()
            return [self._row_to_session(r) for r in rows]

    async def list_sessions(
        self,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SessionRecord]:
        return await asyncio.to_thread(self._list_sessions_sync, tenant_id, owner_user_id)

    def _append_event_sync(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
        branch_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        ev_type = _normalize_event_type(event_type)
        payload_dict = dict(payload or {})
        payload_json = _canonical_payload_json(payload_dict)

        with self._lock:
            now_ms = int(time.time() * 1000)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO harness_sessions
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                VALUES (?, ?, ?, NULL, NULL, 'main', ?)
                """,
                (session_id, tenant_id, user_id, now_ms),
            )
            if branch_name is not None and branch_name.strip():
                eff_branch = branch_name.strip()
            else:
                sess_row = self._conn.execute(
                    "SELECT active_branch FROM harness_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                eff_branch = (
                    str(sess_row["active_branch"] or "main")
                    if sess_row is not None
                    else "main"
                )

            branch_row = self._conn.execute(
                """
                SELECT head_event_id
                FROM harness_branches
                WHERE session_id = ? AND branch_name = ?
                """,
                (session_id, eff_branch),
            ).fetchone()
            if branch_row is None:
                last_on_branch = self._conn.execute(
                    """
                    SELECT event_id
                    FROM harness_events
                    WHERE session_id = ? AND branch_name = ?
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    (session_id, eff_branch),
                ).fetchone()
                branch_head: str | None = (
                    str(last_on_branch["event_id"]) if last_on_branch is not None else None
                )
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO harness_branches
                    (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                    VALUES (?, ?, NULL, ?, ?)
                    """,
                    (session_id, eff_branch, branch_head, now_ms),
                )
            else:
                branch_head = (
                    str(branch_row["head_event_id"])
                    if branch_row["head_event_id"] is not None
                    else None
                )
                if branch_head is None:
                    last_on_branch = self._conn.execute(
                        """
                        SELECT event_id
                        FROM harness_events
                        WHERE session_id = ? AND branch_name = ?
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        (session_id, eff_branch),
                    ).fetchone()
                    if last_on_branch is not None:
                        branch_head = str(last_on_branch["event_id"])

            eff_parent_id = parent_event_id if parent_event_id is not None else branch_head

            last = self._conn.execute(
                """
                SELECT seq, event_hash
                FROM harness_events
                WHERE session_id = ?
                ORDER BY seq DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if last is None:
                seq = 1
                prev_hash = GENESIS_HASH
            else:
                seq = int(last["seq"]) + 1
                prev_hash = str(last["event_hash"] or GENESIS_HASH)

            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            event_hash = compute_event_hash(
                prev_hash=prev_hash,
                session_id=session_id,
                seq=seq,
                event_type=ev_type,
                timestamp_ms=now_ms,
                payload_json=payload_json,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            self._conn.execute(
                """
                INSERT INTO harness_events
                (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                 payload_json, tenant_id, user_id, prev_hash, event_hash,
                 parent_event_id, branch_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    turn_id,
                    seq,
                    ev_type,
                    now_ms,
                    payload_json,
                    tenant_id,
                    user_id,
                    prev_hash,
                    event_hash,
                    eff_parent_id,
                    eff_branch,
                ),
            )
            self._conn.execute(
                """
                UPDATE harness_branches
                SET head_event_id = ?
                WHERE session_id = ? AND branch_name = ?
                """,
                (event_id, session_id, eff_branch),
            )
            self._conn.commit()

        return {
            "event_id": event_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "seq": seq,
            "event_type": ev_type,
            "timestamp_ms": now_ms,
            "payload": payload_dict,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "prev_hash": prev_hash,
            "event_hash": event_hash,
            "parent_event_id": eff_parent_id,
            "branch_name": eff_branch,
        }

    async def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
        branch_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._append_event_sync,
            session_id,
            turn_id,
            event_type,
            payload,
            tenant_id,
            user_id,
            branch_name,
            parent_event_id,
        )

    def load_events_sync(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if up_to_seq is None:
                rows = self._conn.execute(
                    """
                    SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                           payload_json, tenant_id, user_id, prev_hash, event_hash,
                           parent_event_id, branch_name
                    FROM harness_events
                    WHERE session_id = ? AND seq > ?
                    ORDER BY seq ASC
                    """,
                    (session_id, after_seq),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                           payload_json, tenant_id, user_id, prev_hash, event_hash,
                           parent_event_id, branch_name
                    FROM harness_events
                    WHERE session_id = ? AND seq > ? AND seq <= ?
                    ORDER BY seq ASC
                    """,
                    (session_id, after_seq, up_to_seq),
                ).fetchall()
            return [self._row_to_event_dict(r) for r in rows]

    async def load_events(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.load_events_sync, session_id, after_seq, up_to_seq)

    def _verify_chain_sync(self, session_id: str) -> AuditVerificationResult:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT seq, event_type, timestamp_ms, payload_json,
                       tenant_id, user_id, prev_hash, event_hash
                FROM harness_events
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            ).fetchall()
            raw_rows = [dict(r) for r in rows]
        return _verify_rows(session_id, raw_rows)

    async def verify_chain(self, session_id: str) -> AuditVerificationResult:
        return await asyncio.to_thread(self._verify_chain_sync, session_id)

    def _fork_session_sync(
        self,
        source_session_id: str,
        new_session_id: str,
        at_seq: int | None,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[SessionRecord, int]:
        with self._lock:
            existing_target = self._conn.execute(
                "SELECT 1 FROM harness_sessions WHERE session_id = ?",
                (new_session_id,),
            ).fetchone()
            existing_target_events = self._conn.execute(
                "SELECT 1 FROM harness_events WHERE session_id = ? LIMIT 1",
                (new_session_id,),
            ).fetchone()
            if existing_target is not None or existing_target_events is not None:
                raise ValueError(f"Target session '{new_session_id}' already exists.")

            source_session = self._conn.execute(
                "SELECT session_id FROM harness_sessions WHERE session_id = ?",
                (source_session_id,),
            ).fetchone()

            if at_seq is None:
                source_rows = self._conn.execute(
                    """
                    SELECT turn_id, seq, event_type, timestamp_ms, payload_json, tenant_id, user_id
                    FROM harness_events
                    WHERE session_id = ?
                    ORDER BY seq ASC
                    """,
                    (source_session_id,),
                ).fetchall()
            else:
                source_rows = self._conn.execute(
                    """
                    SELECT turn_id, seq, event_type, timestamp_ms, payload_json, tenant_id, user_id
                    FROM harness_events
                    WHERE session_id = ? AND seq <= ?
                    ORDER BY seq ASC
                    """,
                    (source_session_id, at_seq),
                ).fetchall()

            if source_session is None and not source_rows:
                raise KeyError(f"Source session '{source_session_id}' not found.")

            effective_seq = int(source_rows[-1]["seq"]) if source_rows else (at_seq or 0)
            now_ms = int(time.time() * 1000)

            self._conn.execute(
                """
                INSERT INTO harness_sessions
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                VALUES (?, ?, ?, ?, ?, 'main', ?)
                """,
                (
                    new_session_id,
                    tenant_id,
                    owner_user_id,
                    source_session_id,
                    effective_seq,
                    now_ms,
                ),
            )

            prev_hash = GENESIS_HASH
            prev_event_id: str | None = None
            new_seq = 0
            for src in source_rows:
                new_seq += 1
                ev_id = f"evt_{uuid.uuid4().hex[:12]}"
                ev_type = str(src["event_type"])
                ts_ms = int(src["timestamp_ms"])
                payload_json = str(src["payload_json"])
                ev_tenant = tenant_id
                ev_user = str(src["user_id"] or owner_user_id)
                ev_hash = compute_event_hash(
                    prev_hash=prev_hash,
                    session_id=new_session_id,
                    seq=new_seq,
                    event_type=ev_type,
                    timestamp_ms=ts_ms,
                    payload_json=payload_json,
                    tenant_id=ev_tenant,
                    user_id=ev_user,
                )
                self._conn.execute(
                    """
                    INSERT INTO harness_events
                    (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                     payload_json, tenant_id, user_id, prev_hash, event_hash,
                     parent_event_id, branch_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'main')
                    """,
                    (
                        ev_id,
                        new_session_id,
                        str(src["turn_id"]),
                        new_seq,
                        ev_type,
                        ts_ms,
                        payload_json,
                        ev_tenant,
                        ev_user,
                        prev_hash,
                        ev_hash,
                        prev_event_id,
                    ),
                )
                prev_hash = ev_hash
                prev_event_id = ev_id

            # Append canonical "session_forked" lineage event chained onto the copied history
            fork_seq = new_seq + 1
            fork_payload_json = _canonical_payload_json(
                {
                    "source_session_id": source_session_id,
                    "forked_from_seq": effective_seq,
                    "copied_events": len(source_rows),
                }
            )
            fork_event_id = f"evt_{uuid.uuid4().hex[:12]}"
            fork_turn_id = f"fork_{uuid.uuid4().hex[:8]}"
            fork_hash = compute_event_hash(
                prev_hash=prev_hash,
                session_id=new_session_id,
                seq=fork_seq,
                event_type=SESSION_FORKED_EVENT_TYPE,
                timestamp_ms=now_ms,
                payload_json=fork_payload_json,
                tenant_id=tenant_id,
                user_id=owner_user_id,
            )
            self._conn.execute(
                """
                INSERT INTO harness_events
                (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                 payload_json, tenant_id, user_id, prev_hash, event_hash,
                 parent_event_id, branch_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'main')
                """,
                (
                    fork_event_id,
                    new_session_id,
                    fork_turn_id,
                    fork_seq,
                    SESSION_FORKED_EVENT_TYPE,
                    now_ms,
                    fork_payload_json,
                    tenant_id,
                    owner_user_id,
                    prev_hash,
                    fork_hash,
                    prev_event_id,
                ),
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO harness_branches
                (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                VALUES (?, 'main', NULL, ?, ?)
                """,
                (new_session_id, fork_event_id, now_ms),
            )
            self._conn.commit()

            record = SessionRecord(
                session_id=new_session_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                parent_session_id=source_session_id,
                forked_from_seq=effective_seq,
                active_branch="main",
                created_at_ms=now_ms,
            )
            return record, len(source_rows)

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        at_seq: int | None,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[SessionRecord, int]:
        return await asyncio.to_thread(
            self._fork_session_sync,
            source_session_id,
            new_session_id,
            at_seq,
            tenant_id,
            owner_user_id,
        )

    def _create_branch_sync(
        self,
        session_id: str,
        branch_name: str,
        from_event_id: str | None = None,
        *,
        switch: bool = True,
    ) -> BranchInfo:
        clean_name = (branch_name or "").strip()
        if not clean_name:
            raise ValueError("Branch name must be a non-empty string.")

        with self._lock:
            existing_branch = self._conn.execute(
                """
                SELECT 1 FROM harness_branches
                WHERE session_id = ? AND branch_name = ?
                """,
                (session_id, clean_name),
            ).fetchone()
            if existing_branch is not None:
                raise ValueError(
                    f"Branch '{clean_name}' already exists on session '{session_id}'."
                )

            sess_row = self._conn.execute(
                "SELECT active_branch, created_at_ms FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if clean_name == "main" and sess_row is not None:
                has_main_events = self._conn.execute(
                    "SELECT 1 FROM harness_events WHERE session_id = ? AND branch_name = 'main' LIMIT 1",
                    (session_id,),
                ).fetchone()
                if has_main_events is not None or str(sess_row["active_branch"] or "main") == "main":
                    raise ValueError(
                        f"Branch 'main' already exists on session '{session_id}'."
                    )

            cur_active = (
                str(sess_row["active_branch"] or "main")
                if sess_row is not None
                else "main"
            )
            now_ms = int(time.time() * 1000)

            if from_event_id is not None:
                ev_row = self._conn.execute(
                    "SELECT event_id FROM harness_events WHERE session_id = ? AND event_id = ?",
                    (session_id, from_event_id),
                ).fetchone()
                if ev_row is None:
                    raise KeyError(
                        f"Event '{from_event_id}' not found in session '{session_id}'."
                    )
                resolved_base: str | None = from_event_id
            else:
                active_b_row = self._conn.execute(
                    """
                    SELECT head_event_id FROM harness_branches
                    WHERE session_id = ? AND branch_name = ?
                    """,
                    (session_id, cur_active),
                ).fetchone()
                if active_b_row is not None and active_b_row["head_event_id"] is not None:
                    resolved_base = str(active_b_row["head_event_id"])
                else:
                    latest_ev = self._conn.execute(
                        """
                        SELECT event_id FROM harness_events
                        WHERE session_id = ?
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                    resolved_base = str(latest_ev["event_id"]) if latest_ev is not None else None

            if sess_row is None:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO harness_sessions
                    (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                    VALUES (?, 'default', 'anonymous', NULL, NULL, 'main', ?)
                    """,
                    (session_id, now_ms),
                )

            if clean_name != "main":
                main_b = self._conn.execute(
                    "SELECT 1 FROM harness_branches WHERE session_id = ? AND branch_name = 'main'",
                    (session_id,),
                ).fetchone()
                if main_b is None:
                    last_main = self._conn.execute(
                        """
                        SELECT event_id FROM harness_events
                        WHERE session_id = ? AND branch_name = 'main'
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if last_main is not None:
                        main_created = (
                            int(sess_row["created_at_ms"])
                            if sess_row is not None
                            else now_ms
                        )
                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO harness_branches
                            (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                            VALUES (?, 'main', NULL, ?, ?)
                            """,
                            (session_id, str(last_main["event_id"]), main_created),
                        )

            self._conn.execute(
                """
                INSERT INTO harness_branches
                (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, clean_name, resolved_base, resolved_base, now_ms),
            )

            if switch:
                self._conn.execute(
                    "UPDATE harness_sessions SET active_branch = ? WHERE session_id = ?",
                    (clean_name, session_id),
                )
                is_active = True
            else:
                is_active = cur_active == clean_name

            event_count = 0
            if resolved_base is not None:
                ev_rows = self._conn.execute(
                    """
                    SELECT event_id, parent_event_id
                    FROM harness_events
                    WHERE session_id = ?
                    ORDER BY seq ASC
                    """,
                    (session_id,),
                ).fetchall()
                light_events = [
                    {"event_id": r["event_id"], "parent_event_id": r["parent_event_id"]}
                    for r in ev_rows
                ]
                event_count = len(_walk_branch_chain(light_events, resolved_base))

            self._conn.commit()

        return BranchInfo(
            branch_name=clean_name,
            base_event_id=resolved_base,
            head_event_id=resolved_base,
            event_count=event_count,
            is_active=is_active,
            created_at_ms=now_ms,
        )

    async def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_event_id: str | None = None,
        *,
        switch: bool = True,
    ) -> BranchInfo:
        return await asyncio.to_thread(
            self._create_branch_sync,
            session_id,
            branch_name,
            from_event_id,
            switch=switch,
        )

    def _switch_branch_sync(self, session_id: str, branch_name: str) -> BranchInfo:
        target = (branch_name or "").strip()
        if not target:
            raise KeyError(f"Branch '{branch_name}' not found in session '{session_id}'.")

        with self._lock:
            sess_row = self._conn.execute(
                "SELECT session_id, created_at_ms FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if sess_row is None:
                raise KeyError(f"Session '{session_id}' not found.")

            branch_row = self._conn.execute(
                """
                SELECT branch_name, base_event_id, head_event_id, created_at_ms
                FROM harness_branches
                WHERE session_id = ? AND branch_name = ?
                """,
                (session_id, target),
            ).fetchone()

            if branch_row is None:
                if target != "main":
                    raise KeyError(
                        f"Branch '{target}' not found in session '{session_id}'."
                    )
                last_main = self._conn.execute(
                    """
                    SELECT event_id FROM harness_events
                    WHERE session_id = ? AND branch_name = 'main'
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                base_id: str | None = None
                head_id: str | None = (
                    str(last_main["event_id"]) if last_main is not None else None
                )
                created_ms = int(sess_row["created_at_ms"])
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO harness_branches
                    (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                    VALUES (?, 'main', NULL, ?, ?)
                    """,
                    (session_id, head_id, created_ms),
                )
            else:
                base_id = (
                    str(branch_row["base_event_id"])
                    if branch_row["base_event_id"] is not None
                    else None
                )
                head_id = (
                    str(branch_row["head_event_id"])
                    if branch_row["head_event_id"] is not None
                    else None
                )
                created_ms = int(branch_row["created_at_ms"])

            self._conn.execute(
                "UPDATE harness_sessions SET active_branch = ? WHERE session_id = ?",
                (target, session_id),
            )

            event_count = 0
            if head_id is not None:
                ev_rows = self._conn.execute(
                    """
                    SELECT event_id, parent_event_id
                    FROM harness_events
                    WHERE session_id = ?
                    ORDER BY seq ASC
                    """,
                    (session_id,),
                ).fetchall()
                light_events = [
                    {"event_id": r["event_id"], "parent_event_id": r["parent_event_id"]}
                    for r in ev_rows
                ]
                event_count = len(_walk_branch_chain(light_events, head_id))

            self._conn.commit()

        return BranchInfo(
            branch_name=target,
            base_event_id=base_id,
            head_event_id=head_id,
            event_count=event_count,
            is_active=True,
            created_at_ms=created_ms,
        )

    async def switch_branch(self, session_id: str, branch_name: str) -> BranchInfo:
        return await asyncio.to_thread(self._switch_branch_sync, session_id, branch_name)

    def _get_branch_events_sync(
        self,
        session_id: str,
        branch_name: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sess_row = self._conn.execute(
                "SELECT active_branch FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            active_branch = (
                str(sess_row["active_branch"] or "main")
                if sess_row is not None
                else "main"
            )
            branch_rows = self._conn.execute(
                """
                SELECT branch_name, base_event_id, head_event_id, created_at_ms
                FROM harness_branches
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            branches_by_name = {
                str(r["branch_name"]): dict(r) for r in branch_rows
            }
            all_events = self.load_events_sync(session_id)
        return _resolve_branch_events(
            all_events, active_branch, branches_by_name, branch_name=branch_name
        )

    async def get_branch_events(
        self,
        session_id: str,
        branch_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_branch_events_sync, session_id, branch_name
        )

    def _get_session_tree_sync(self, session_id: str) -> SessionTreeResult:
        with self._lock:
            sess_row = self._conn.execute(
                "SELECT active_branch, created_at_ms FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            active_branch = (
                str(sess_row["active_branch"] or "main")
                if sess_row is not None
                else "main"
            )
            sess_created = (
                int(sess_row["created_at_ms"]) if sess_row is not None else None
            )
            branch_rows = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT branch_name, base_event_id, head_event_id, created_at_ms
                    FROM harness_branches
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            ]
            all_events = self.load_events_sync(session_id)
        return _build_session_tree(
            session_id=session_id,
            active_branch=active_branch,
            session_created_at_ms=sess_created,
            branch_rows=branch_rows,
            all_events=all_events,
        )

    async def get_session_tree(self, session_id: str) -> SessionTreeResult:
        return await asyncio.to_thread(self._get_session_tree_sync, session_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


class PostgresJournalStore(JournalStorageBackend):
    """
    Enterprise PostgreSQL journal store for horizontally scaled multi-Pod deployments.

    Guarantees strict cross-Pod monotonic `seq` ordering and unbroken SHA-256 hash chains
    via transactional `pg_advisory_xact_lock(hashtext($1))` serialization per session.
    Supports both live `database_url` (lazy `asyncpg` pool) and injectable `pool` doubles.
    """

    def __init__(
        self,
        database_url: str = "",
        *,
        pool: Any | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        if not database_url.strip() and pool is None:
            raise ValueError("PostgresJournalStore requires either 'database_url' or an injectable 'pool'.")
        self.database_url = database_url.strip()
        self._pool = pool
        self._owns_pool = pool is None
        self._min_size = min_size
        self._max_size = max_size
        self._schema_initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        return "postgres"

    async def connect(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "PostgresJournalStore requires 'asyncpg' (`pip install asyncpg>=0.29.0`)."
                ) from exc
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self._min_size,
                max_size=self._max_size,
            )

        if not self._schema_initialized:
            async with self._init_lock:
                if not self._schema_initialized:
                    await self._init_schema()
                    self._schema_initialized = True
        return self._pool

    async def _init_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_user_id TEXT NOT NULL DEFAULT 'anonymous',
                    parent_session_id TEXT,
                    forked_from_seq INTEGER,
                    active_branch TEXT NOT NULL DEFAULT 'main',
                    created_at_ms BIGINT NOT NULL
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp_ms BIGINT NOT NULL,
                    payload_json TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    prev_hash TEXT NOT NULL DEFAULT '',
                    event_hash TEXT NOT NULL DEFAULT '',
                    parent_event_id TEXT,
                    branch_name TEXT NOT NULL DEFAULT 'main'
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_branches (
                    session_id TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    base_event_id TEXT,
                    head_event_id TEXT,
                    created_at_ms BIGINT NOT NULL,
                    PRIMARY KEY (session_id, branch_name)
                )
                """
            )
            # Forward-compatible column migrations for existing PostgreSQL deployments
            await conn.execute(
                "ALTER TABLE harness_sessions ADD COLUMN IF NOT EXISTS active_branch TEXT NOT NULL DEFAULT 'main'"
            )
            await conn.execute(
                "ALTER TABLE harness_events ADD COLUMN IF NOT EXISTS parent_event_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE harness_events ADD COLUMN IF NOT EXISTS branch_name TEXT NOT NULL DEFAULT 'main'"
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_seq
                ON harness_events (session_id, seq)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_branch
                ON harness_events (session_id, branch_name, seq)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_tenant_owner
                ON harness_sessions (tenant_id, owner_user_id)
                """
            )

    @staticmethod
    def _row_to_session(row: Any) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            owner_user_id=row["owner_user_id"],
            parent_session_id=row["parent_session_id"],
            forked_from_seq=row["forked_from_seq"],
            active_branch=str(_row_field(row, "active_branch") or "main"),
            created_at_ms=int(row["created_at_ms"]),
        )

    @staticmethod
    def _row_to_event_dict(row: Any) -> dict[str, Any]:
        payload_raw = row["payload_json"]
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else dict(payload_raw or {})
        return {
            "event_id": row["event_id"],
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "seq": int(row["seq"]),
            "event_type": row["event_type"],
            "timestamp_ms": int(row["timestamp_ms"]),
            "payload": payload,
            "tenant_id": row["tenant_id"],
            "user_id": row["user_id"],
            "prev_hash": row["prev_hash"],
            "event_hash": row["event_hash"],
            "parent_event_id": _row_field(row, "parent_event_id", None),
            "branch_name": str(_row_field(row, "branch_name") or "main"),
        }

    async def ensure_session(
        self,
        session_id: str,
        tenant_id: str = "default",
        owner_user_id: str = "anonymous",
        parent_session_id: str | None = None,
        forked_from_seq: int | None = None,
    ) -> SessionRecord:
        pool = await self.connect()
        now_ms = int(time.time() * 1000)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO harness_sessions
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                VALUES ($1, $2, $3, $4, $5, 'main', $6)
                ON CONFLICT (session_id) DO NOTHING
                """,
                session_id,
                tenant_id,
                owner_user_id,
                parent_session_id,
                forked_from_seq,
                now_ms,
            )
            row = await conn.fetchrow(
                """
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                WHERE session_id = $1
                """,
                session_id,
            )
            assert row is not None
            return self._row_to_session(row)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        pool = await self.connect()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                WHERE session_id = $1
                """,
                session_id,
            )
            return self._row_to_session(row) if row is not None else None

    async def list_sessions(
        self,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SessionRecord]:
        pool = await self.connect()
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            params.append(tenant_id)
            clauses.append(f"tenant_id = ${len(params)}")
        if owner_user_id is not None:
            params.append(owner_user_id)
            clauses.append(f"owner_user_id = ${len(params)}")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms
                FROM harness_sessions
                {where_sql}
                ORDER BY created_at_ms DESC, session_id ASC
                """,
                *params,
            )
            return [self._row_to_session(r) for r in rows]

    async def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
        branch_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        pool = await self.connect()
        ev_type = _normalize_event_type(event_type)
        payload_dict = dict(payload or {})
        payload_json = _canonical_payload_json(payload_dict)

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", session_id)
                now_ms = int(time.time() * 1000)
                await conn.execute(
                    """
                    INSERT INTO harness_sessions
                    (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                    VALUES ($1, $2, $3, NULL, NULL, 'main', $4)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    session_id,
                    tenant_id,
                    user_id,
                    now_ms,
                )
                if branch_name is not None and branch_name.strip():
                    eff_branch = branch_name.strip()
                else:
                    sess_row = await conn.fetchrow(
                        "SELECT active_branch FROM harness_sessions WHERE session_id = $1",
                        session_id,
                    )
                    eff_branch = (
                        str(sess_row["active_branch"] or "main")
                        if sess_row is not None
                        else "main"
                    )

                branch_row = await conn.fetchrow(
                    """
                    SELECT head_event_id
                    FROM harness_branches
                    WHERE session_id = $1 AND branch_name = $2
                    """,
                    session_id,
                    eff_branch,
                )
                if branch_row is None:
                    last_on_branch = await conn.fetchrow(
                        """
                        SELECT event_id
                        FROM harness_events
                        WHERE session_id = $1 AND branch_name = $2
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        session_id,
                        eff_branch,
                    )
                    branch_head: str | None = (
                        str(last_on_branch["event_id"]) if last_on_branch is not None else None
                    )
                    await conn.execute(
                        """
                        INSERT INTO harness_branches
                        (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                        VALUES ($1, $2, NULL, $3, $4)
                        ON CONFLICT (session_id, branch_name) DO NOTHING
                        """,
                        session_id,
                        eff_branch,
                        branch_head,
                        now_ms,
                    )
                else:
                    branch_head = (
                        str(branch_row["head_event_id"])
                        if branch_row["head_event_id"] is not None
                        else None
                    )
                    if branch_head is None:
                        last_on_branch = await conn.fetchrow(
                            """
                            SELECT event_id
                            FROM harness_events
                            WHERE session_id = $1 AND branch_name = $2
                            ORDER BY seq DESC
                            LIMIT 1
                            """,
                            session_id,
                            eff_branch,
                        )
                        if last_on_branch is not None:
                            branch_head = str(last_on_branch["event_id"])

                eff_parent_id = parent_event_id if parent_event_id is not None else branch_head

                last = await conn.fetchrow(
                    """
                    SELECT seq, event_hash
                    FROM harness_events
                    WHERE session_id = $1
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    session_id,
                )
                if last is None:
                    seq = 1
                    prev_hash = GENESIS_HASH
                else:
                    seq = int(last["seq"]) + 1
                    prev_hash = str(last["event_hash"] or GENESIS_HASH)

                event_id = f"evt_{uuid.uuid4().hex[:12]}"
                event_hash = compute_event_hash(
                    prev_hash=prev_hash,
                    session_id=session_id,
                    seq=seq,
                    event_type=ev_type,
                    timestamp_ms=now_ms,
                    payload_json=payload_json,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO harness_events
                    (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                     payload_json, tenant_id, user_id, prev_hash, event_hash,
                     parent_event_id, branch_name)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    event_id,
                    session_id,
                    turn_id,
                    seq,
                    ev_type,
                    now_ms,
                    payload_json,
                    tenant_id,
                    user_id,
                    prev_hash,
                    event_hash,
                    eff_parent_id,
                    eff_branch,
                )
                await conn.execute(
                    """
                    UPDATE harness_branches
                    SET head_event_id = $1
                    WHERE session_id = $2 AND branch_name = $3
                    """,
                    event_id,
                    session_id,
                    eff_branch,
                )

        return {
            "event_id": event_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "seq": seq,
            "event_type": ev_type,
            "timestamp_ms": now_ms,
            "payload": payload_dict,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "prev_hash": prev_hash,
            "event_hash": event_hash,
            "parent_event_id": eff_parent_id,
            "branch_name": eff_branch,
        }

    def load_events_sync(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.load_events(session_id, after_seq=after_seq, up_to_seq=up_to_seq))
        raise RuntimeError(
            "PostgresJournalStore.load_events_sync() cannot be called inside an active event loop; "
            "use `await load_events(...)` instead."
        )

    async def load_events(
        self,
        session_id: str,
        after_seq: int = 0,
        up_to_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self.connect()
        async with pool.acquire() as conn:
            if up_to_seq is None:
                rows = await conn.fetch(
                    """
                    SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                           payload_json, tenant_id, user_id, prev_hash, event_hash,
                           parent_event_id, branch_name
                    FROM harness_events
                    WHERE session_id = $1 AND seq > $2
                    ORDER BY seq ASC
                    """,
                    session_id,
                    after_seq,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                           payload_json, tenant_id, user_id, prev_hash, event_hash,
                           parent_event_id, branch_name
                    FROM harness_events
                    WHERE session_id = $1 AND seq > $2 AND seq <= $3
                    ORDER BY seq ASC
                    """,
                    session_id,
                    after_seq,
                    up_to_seq,
                )
            return [self._row_to_event_dict(r) for r in rows]

    async def verify_chain(self, session_id: str) -> AuditVerificationResult:
        pool = await self.connect()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT seq, event_type, timestamp_ms, payload_json,
                       tenant_id, user_id, prev_hash, event_hash
                FROM harness_events
                WHERE session_id = $1
                ORDER BY seq ASC
                """,
                session_id,
            )
            raw_rows = [dict(r) for r in rows]
        return _verify_rows(session_id, raw_rows)

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        at_seq: int | None,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[SessionRecord, int]:
        pool = await self.connect()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for sid in sorted({source_session_id, new_session_id}):
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", sid)

                existing_target = await conn.fetchrow(
                    "SELECT session_id FROM harness_sessions WHERE session_id = $1",
                    new_session_id,
                )
                existing_events = await conn.fetchrow(
                    "SELECT event_id FROM harness_events WHERE session_id = $1 LIMIT 1",
                    new_session_id,
                )
                if existing_target is not None or existing_events is not None:
                    raise ValueError(f"Target session '{new_session_id}' already exists.")

                source_session = await conn.fetchrow(
                    "SELECT session_id FROM harness_sessions WHERE session_id = $1",
                    source_session_id,
                )
                if at_seq is None:
                    source_rows = await conn.fetch(
                        """
                        SELECT turn_id, seq, event_type, timestamp_ms, payload_json, tenant_id, user_id
                        FROM harness_events
                        WHERE session_id = $1
                        ORDER BY seq ASC
                        """,
                        source_session_id,
                    )
                else:
                    source_rows = await conn.fetch(
                        """
                        SELECT turn_id, seq, event_type, timestamp_ms, payload_json, tenant_id, user_id
                        FROM harness_events
                        WHERE session_id = $1 AND seq <= $2
                        ORDER BY seq ASC
                        """,
                        source_session_id,
                        at_seq,
                    )

                if source_session is None and not source_rows:
                    raise KeyError(f"Source session '{source_session_id}' not found.")

                effective_seq = int(source_rows[-1]["seq"]) if source_rows else (at_seq or 0)
                now_ms = int(time.time() * 1000)

                await conn.execute(
                    """
                    INSERT INTO harness_sessions
                    (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                    VALUES ($1, $2, $3, $4, $5, 'main', $6)
                    """,
                    new_session_id,
                    tenant_id,
                    owner_user_id,
                    source_session_id,
                    effective_seq,
                    now_ms,
                )

                prev_hash = GENESIS_HASH
                prev_event_id: str | None = None
                new_seq = 0
                for src in source_rows:
                    new_seq += 1
                    ev_id = f"evt_{uuid.uuid4().hex[:12]}"
                    ev_type = str(src["event_type"])
                    ts_ms = int(src["timestamp_ms"])
                    payload_json = str(src["payload_json"])
                    ev_tenant = tenant_id
                    ev_user = str(src["user_id"] or owner_user_id)
                    ev_hash = compute_event_hash(
                        prev_hash=prev_hash,
                        session_id=new_session_id,
                        seq=new_seq,
                        event_type=ev_type,
                        timestamp_ms=ts_ms,
                        payload_json=payload_json,
                        tenant_id=ev_tenant,
                        user_id=ev_user,
                    )
                    await conn.execute(
                        """
                        INSERT INTO harness_events
                        (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                         payload_json, tenant_id, user_id, prev_hash, event_hash,
                         parent_event_id, branch_name)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'main')
                        """,
                        ev_id,
                        new_session_id,
                        str(src["turn_id"]),
                        new_seq,
                        ev_type,
                        ts_ms,
                        payload_json,
                        ev_tenant,
                        ev_user,
                        prev_hash,
                        ev_hash,
                        prev_event_id,
                    )
                    prev_hash = ev_hash
                    prev_event_id = ev_id

                fork_seq = new_seq + 1
                fork_payload_json = _canonical_payload_json(
                    {
                        "source_session_id": source_session_id,
                        "forked_from_seq": effective_seq,
                        "copied_events": len(source_rows),
                    }
                )
                fork_event_id = f"evt_{uuid.uuid4().hex[:12]}"
                fork_turn_id = f"fork_{uuid.uuid4().hex[:8]}"
                fork_hash = compute_event_hash(
                    prev_hash=prev_hash,
                    session_id=new_session_id,
                    seq=fork_seq,
                    event_type=SESSION_FORKED_EVENT_TYPE,
                    timestamp_ms=now_ms,
                    payload_json=fork_payload_json,
                    tenant_id=tenant_id,
                    user_id=owner_user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO harness_events
                    (event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                     payload_json, tenant_id, user_id, prev_hash, event_hash,
                     parent_event_id, branch_name)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'main')
                    """,
                    fork_event_id,
                    new_session_id,
                    fork_turn_id,
                    fork_seq,
                    SESSION_FORKED_EVENT_TYPE,
                    now_ms,
                    fork_payload_json,
                    tenant_id,
                    owner_user_id,
                    prev_hash,
                    fork_hash,
                    prev_event_id,
                )
                await conn.execute(
                    """
                    INSERT INTO harness_branches
                    (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                    VALUES ($1, 'main', NULL, $2, $3)
                    ON CONFLICT (session_id, branch_name)
                    DO UPDATE SET head_event_id = EXCLUDED.head_event_id
                    """,
                    new_session_id,
                    fork_event_id,
                    now_ms,
                )

        record = SessionRecord(
            session_id=new_session_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            parent_session_id=source_session_id,
            forked_from_seq=effective_seq,
            active_branch="main",
            created_at_ms=now_ms,
        )
        return record, len(source_rows)

    async def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_event_id: str | None = None,
        *,
        switch: bool = True,
    ) -> BranchInfo:
        clean_name = (branch_name or "").strip()
        if not clean_name:
            raise ValueError("Branch name must be a non-empty string.")

        pool = await self.connect()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", session_id)
                existing_branch = await conn.fetchrow(
                    """
                    SELECT branch_name FROM harness_branches
                    WHERE session_id = $1 AND branch_name = $2
                    """,
                    session_id,
                    clean_name,
                )
                if existing_branch is not None:
                    raise ValueError(
                        f"Branch '{clean_name}' already exists on session '{session_id}'."
                    )

                sess_row = await conn.fetchrow(
                    "SELECT active_branch, created_at_ms FROM harness_sessions WHERE session_id = $1",
                    session_id,
                )
                if clean_name == "main" and sess_row is not None:
                    has_main_events = await conn.fetchrow(
                        "SELECT event_id FROM harness_events WHERE session_id = $1 AND branch_name = 'main' LIMIT 1",
                        session_id,
                    )
                    if has_main_events is not None or str(sess_row["active_branch"] or "main") == "main":
                        raise ValueError(
                            f"Branch 'main' already exists on session '{session_id}'."
                        )

                cur_active = (
                    str(sess_row["active_branch"] or "main")
                    if sess_row is not None
                    else "main"
                )
                now_ms = int(time.time() * 1000)

                if from_event_id is not None:
                    ev_row = await conn.fetchrow(
                        "SELECT event_id FROM harness_events WHERE session_id = $1 AND event_id = $2",
                        session_id,
                        from_event_id,
                    )
                    if ev_row is None:
                        raise KeyError(
                            f"Event '{from_event_id}' not found in session '{session_id}'."
                        )
                    resolved_base: str | None = from_event_id
                else:
                    active_b_row = await conn.fetchrow(
                        """
                        SELECT head_event_id FROM harness_branches
                        WHERE session_id = $1 AND branch_name = $2
                        """,
                        session_id,
                        cur_active,
                    )
                    if active_b_row is not None and active_b_row["head_event_id"] is not None:
                        resolved_base = str(active_b_row["head_event_id"])
                    else:
                        latest_ev = await conn.fetchrow(
                            """
                            SELECT event_id FROM harness_events
                            WHERE session_id = $1
                            ORDER BY seq DESC
                            LIMIT 1
                            """,
                            session_id,
                        )
                        resolved_base = (
                            str(latest_ev["event_id"]) if latest_ev is not None else None
                        )

                if sess_row is None:
                    await conn.execute(
                        """
                        INSERT INTO harness_sessions
                        (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, active_branch, created_at_ms)
                        VALUES ($1, 'default', 'anonymous', NULL, NULL, 'main', $2)
                        ON CONFLICT (session_id) DO NOTHING
                        """,
                        session_id,
                        now_ms,
                    )

                if clean_name != "main":
                    main_b = await conn.fetchrow(
                        "SELECT branch_name FROM harness_branches WHERE session_id = $1 AND branch_name = 'main'",
                        session_id,
                    )
                    if main_b is None:
                        last_main = await conn.fetchrow(
                            """
                            SELECT event_id FROM harness_events
                            WHERE session_id = $1 AND branch_name = 'main'
                            ORDER BY seq DESC
                            LIMIT 1
                            """,
                            session_id,
                        )
                        if last_main is not None:
                            main_created = (
                                int(sess_row["created_at_ms"])
                                if sess_row is not None
                                else now_ms
                            )
                            await conn.execute(
                                """
                                INSERT INTO harness_branches
                                (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                                VALUES ($1, 'main', NULL, $2, $3)
                                ON CONFLICT (session_id, branch_name) DO NOTHING
                                """,
                                session_id,
                                str(last_main["event_id"]),
                                main_created,
                            )

                await conn.execute(
                    """
                    INSERT INTO harness_branches
                    (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    session_id,
                    clean_name,
                    resolved_base,
                    resolved_base,
                    now_ms,
                )

                if switch:
                    await conn.execute(
                        "UPDATE harness_sessions SET active_branch = $1 WHERE session_id = $2",
                        clean_name,
                        session_id,
                    )
                    is_active = True
                else:
                    is_active = cur_active == clean_name

                event_count = 0
                if resolved_base is not None:
                    ev_rows = await conn.fetch(
                        """
                        SELECT event_id, parent_event_id
                        FROM harness_events
                        WHERE session_id = $1
                        ORDER BY seq ASC
                        """,
                        session_id,
                    )
                    light_events = [
                        {"event_id": r["event_id"], "parent_event_id": r["parent_event_id"]}
                        for r in ev_rows
                    ]
                    event_count = len(_walk_branch_chain(light_events, resolved_base))

        return BranchInfo(
            branch_name=clean_name,
            base_event_id=resolved_base,
            head_event_id=resolved_base,
            event_count=event_count,
            is_active=is_active,
            created_at_ms=now_ms,
        )

    async def switch_branch(self, session_id: str, branch_name: str) -> BranchInfo:
        target = (branch_name or "").strip()
        if not target:
            raise KeyError(f"Branch '{branch_name}' not found in session '{session_id}'.")

        pool = await self.connect()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", session_id)
                sess_row = await conn.fetchrow(
                    "SELECT session_id, created_at_ms FROM harness_sessions WHERE session_id = $1",
                    session_id,
                )
                if sess_row is None:
                    raise KeyError(f"Session '{session_id}' not found.")

                branch_row = await conn.fetchrow(
                    """
                    SELECT branch_name, base_event_id, head_event_id, created_at_ms
                    FROM harness_branches
                    WHERE session_id = $1 AND branch_name = $2
                    """,
                    session_id,
                    target,
                )

                if branch_row is None:
                    if target != "main":
                        raise KeyError(
                            f"Branch '{target}' not found in session '{session_id}'."
                        )
                    last_main = await conn.fetchrow(
                        """
                        SELECT event_id FROM harness_events
                        WHERE session_id = $1 AND branch_name = 'main'
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        session_id,
                    )
                    base_id: str | None = None
                    head_id: str | None = (
                        str(last_main["event_id"]) if last_main is not None else None
                    )
                    created_ms = int(sess_row["created_at_ms"])
                    await conn.execute(
                        """
                        INSERT INTO harness_branches
                        (session_id, branch_name, base_event_id, head_event_id, created_at_ms)
                        VALUES ($1, 'main', NULL, $2, $3)
                        ON CONFLICT (session_id, branch_name) DO NOTHING
                        """,
                        session_id,
                        head_id,
                        created_ms,
                    )
                else:
                    base_id = (
                        str(branch_row["base_event_id"])
                        if branch_row["base_event_id"] is not None
                        else None
                    )
                    head_id = (
                        str(branch_row["head_event_id"])
                        if branch_row["head_event_id"] is not None
                        else None
                    )
                    created_ms = int(branch_row["created_at_ms"])

                await conn.execute(
                    "UPDATE harness_sessions SET active_branch = $1 WHERE session_id = $2",
                    target,
                    session_id,
                )

                event_count = 0
                if head_id is not None:
                    ev_rows = await conn.fetch(
                        """
                        SELECT event_id, parent_event_id
                        FROM harness_events
                        WHERE session_id = $1
                        ORDER BY seq ASC
                        """,
                        session_id,
                    )
                    light_events = [
                        {"event_id": r["event_id"], "parent_event_id": r["parent_event_id"]}
                        for r in ev_rows
                    ]
                    event_count = len(_walk_branch_chain(light_events, head_id))

        return BranchInfo(
            branch_name=target,
            base_event_id=base_id,
            head_event_id=head_id,
            event_count=event_count,
            is_active=True,
            created_at_ms=created_ms,
        )

    async def get_branch_events(
        self,
        session_id: str,
        branch_name: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self.connect()
        async with pool.acquire() as conn:
            sess_row = await conn.fetchrow(
                "SELECT active_branch FROM harness_sessions WHERE session_id = $1",
                session_id,
            )
            active_branch = (
                str(sess_row["active_branch"] or "main")
                if sess_row is not None
                else "main"
            )
            branch_rows = await conn.fetch(
                """
                SELECT branch_name, base_event_id, head_event_id, created_at_ms
                FROM harness_branches
                WHERE session_id = $1
                """,
                session_id,
            )
            branches_by_name = {
                str(r["branch_name"]): dict(r) for r in branch_rows
            }
            ev_rows = await conn.fetch(
                """
                SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                       payload_json, tenant_id, user_id, prev_hash, event_hash,
                       parent_event_id, branch_name
                FROM harness_events
                WHERE session_id = $1
                ORDER BY seq ASC
                """,
                session_id,
            )
            all_events = [self._row_to_event_dict(r) for r in ev_rows]
        return _resolve_branch_events(
            all_events, active_branch, branches_by_name, branch_name=branch_name
        )

    async def get_session_tree(self, session_id: str) -> SessionTreeResult:
        pool = await self.connect()
        async with pool.acquire() as conn:
            sess_row = await conn.fetchrow(
                "SELECT active_branch, created_at_ms FROM harness_sessions WHERE session_id = $1",
                session_id,
            )
            active_branch = (
                str(sess_row["active_branch"] or "main")
                if sess_row is not None
                else "main"
            )
            sess_created = (
                int(sess_row["created_at_ms"]) if sess_row is not None else None
            )
            branch_rows = [
                dict(r)
                for r in await conn.fetch(
                    """
                    SELECT branch_name, base_event_id, head_event_id, created_at_ms
                    FROM harness_branches
                    WHERE session_id = $1
                    """,
                    session_id,
                )
            ]
            ev_rows = await conn.fetch(
                """
                SELECT event_id, session_id, turn_id, seq, event_type, timestamp_ms,
                       payload_json, tenant_id, user_id, prev_hash, event_hash,
                       parent_event_id, branch_name
                FROM harness_events
                WHERE session_id = $1
                ORDER BY seq ASC
                """,
                session_id,
            )
            all_events = [self._row_to_event_dict(r) for r in ev_rows]
        return _build_session_tree(
            session_id=session_id,
            active_branch=active_branch,
            session_created_at_ms=sess_created,
            branch_rows=branch_rows,
            all_events=all_events,
        )

    def close(self) -> None:
        if self._owns_pool and self._pool is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aclose())

    async def aclose(self) -> None:
        if self._owns_pool and self._pool is not None:
            close_fn = getattr(self._pool, "close", None)
            if close_fn is not None:
                res = close_fn()
                if asyncio.iscoroutine(res):
                    await res
            self._pool = None


def create_journal_store(
    db_path: str | Path = ":memory:",
    database_url: str | None = None,
) -> JournalStorageBackend:
    """
    Factory selecting `PostgresJournalStore` when `database_url` is configured,
    otherwise falling back to WAL-enabled `SQLiteJournalStore`.
    """
    url = (database_url or "").strip()
    if url:
        return PostgresJournalStore(database_url=url)
    return SQLiteJournalStore(db_path=db_path)
