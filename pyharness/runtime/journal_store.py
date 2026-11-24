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
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


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


class JournalStorageBackend(ABC):
    """
    Pluggable durable persistence backend for EventJournal supporting:
    - Multi-tenant session ownership & fork lineage (`harness_sessions`).
    - Strict monotonic per-session event sequencing (`harness_events`).
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
                    event_hash TEXT NOT NULL DEFAULT ''
                )
                """
            )

            # Auto-migrate legacy harness_events tables missing enterprise audit columns
            cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(harness_events)").fetchall()
            }
            migrations = (
                ("tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("user_id", "TEXT NOT NULL DEFAULT 'anonymous'"),
                ("prev_hash", "TEXT NOT NULL DEFAULT ''"),
                ("event_hash", "TEXT NOT NULL DEFAULT ''"),
            )
            for col_name, col_ddl in migrations:
                if col_name not in cols:
                    self._conn.execute(
                        f"ALTER TABLE harness_events ADD COLUMN {col_name} {col_ddl}"
                    )

            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_seq
                ON harness_events (session_id, seq)
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
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
    ) -> dict[str, Any]:
        ev_type = _normalize_event_type(event_type)
        payload_dict = dict(payload or {})
        payload_json = _canonical_payload_json(payload_dict)

        with self._lock:
            now_ms = int(time.time() * 1000)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO harness_sessions
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                VALUES (?, ?, ?, NULL, NULL, ?)
                """,
                (session_id, tenant_id, user_id, now_ms),
            )
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
                 payload_json, tenant_id, user_id, prev_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
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
        }

    async def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._append_event_sync,
            session_id,
            turn_id,
            event_type,
            payload,
            tenant_id,
            user_id,
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
                           payload_json, tenant_id, user_id, prev_hash, event_hash
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
                           payload_json, tenant_id, user_id, prev_hash, event_hash
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
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
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
                     payload_json, tenant_id, user_id, prev_hash, event_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
                prev_hash = ev_hash

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
                 payload_json, tenant_id, user_id, prev_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            self._conn.commit()

            record = SessionRecord(
                session_id=new_session_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                parent_session_id=source_session_id,
                forked_from_seq=effective_seq,
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
                    event_hash TEXT NOT NULL DEFAULT ''
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_seq
                ON harness_events (session_id, seq)
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
                (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
                SELECT session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms
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
                    (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                    VALUES ($1, $2, $3, NULL, NULL, $4)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    session_id,
                    tenant_id,
                    user_id,
                    now_ms,
                )
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
                     payload_json, tenant_id, user_id, prev_hash, event_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                           payload_json, tenant_id, user_id, prev_hash, event_hash
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
                           payload_json, tenant_id, user_id, prev_hash, event_hash
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
                    (session_id, tenant_id, owner_user_id, parent_session_id, forked_from_seq, created_at_ms)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    new_session_id,
                    tenant_id,
                    owner_user_id,
                    source_session_id,
                    effective_seq,
                    now_ms,
                )

                prev_hash = GENESIS_HASH
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
                         payload_json, tenant_id, user_id, prev_hash, event_hash)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                    )
                    prev_hash = ev_hash

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
                     payload_json, tenant_id, user_id, prev_hash, event_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                )

        record = SessionRecord(
            session_id=new_session_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            parent_session_id=source_session_id,
            forked_from_seq=effective_seq,
            created_at_ms=now_ms,
        )
        return record, len(source_rows)

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
