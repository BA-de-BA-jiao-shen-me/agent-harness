from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec

_FORBIDDEN_READONLY_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

_ALLOWED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_AUTHORIZER_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def _strip_sql_literals_and_comments(sql: str) -> str:
    """Strip single/double quoted string literals and SQL comments before structural checks."""
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\r\n]*", " ", without_block)
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", without_line)


def _readonly_authorizer(
    action_code: int,
    arg1: str | None,
    arg2: str | None,
    _db_name: str | None,
    _trigger_name: str | None,
) -> int:
    if action_code == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    if action_code in _ALLOWED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLSafetyError(ValueError):
    """Raised when a SQL statement violates read-only engine authorizer or anti-injection policy."""


class SQLSandbox:
    """
    Kernel-Governed SQL Analytics Sandbox supporting:
    1. Schema introspection (`inspect_sql_schema`) so the LLM never hallucinates column names.
    2. Read-only SELECT execution (`query_sql_readonly`) enforced by:
       - Per-query isolated read-only SQLite URI connection + `PRAGMA query_only = ON`,
       - C-engine `conn.set_authorizer()` action whitelist (impossible to bypass via CTEs/comments),
       - `conn.set_progress_handler()` VM opcode deadline interrupting infinite recursive CTEs,
       - `cursor.fetchmany(max_rows + 1)` hard row cap preventing memory exhaustion.
    3. Mutating SQL execution (`execute_mutating_sql`) serialized under `threading.RLock`
       and gated at `RiskLevel.HIGH` via `ApprovalBroker`.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        max_rows: int = 100,
        query_timeout_seconds: float = 3.0,
    ) -> None:
        self.max_rows = max_rows
        self.query_timeout_seconds = query_timeout_seconds
        self._write_lock = threading.RLock()

        raw_path = str(db_path)
        if raw_path == ":memory:":
            self._is_memory = True
            self._rw_uri = f"file:pyharness_sql_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._ro_uri = self._rw_uri
            self._anchor_conn = sqlite3.connect(self._rw_uri, uri=True, check_same_thread=False)
        else:
            self._is_memory = False
            resolved = Path(raw_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._rw_uri = resolved.as_uri()
            self._ro_uri = f"{resolved.as_uri()}?mode=ro"
            self._anchor_conn = sqlite3.connect(str(resolved), check_same_thread=False)

        self._anchor_conn.row_factory = sqlite3.Row

    def _open_readonly_connection(self, deadline: float) -> sqlite3.Connection:
        conn = sqlite3.connect(self._ro_uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_readonly_authorizer)

        def _progress_guard() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_progress_guard, 2000)
        return conn

    def seed_demo_enterprise_data(self) -> None:
        """Populate realistic industrial/operations demo tables for testing and benchmarks."""
        with self._write_lock:
            cur = self._anchor_conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS equipment_incidents (
                    incident_id TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    downtime_minutes INTEGER NOT NULL,
                    reported_month TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS monthly_sla_metrics (
                    month TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    uptime_pct REAL NOT NULL,
                    mttr_minutes REAL NOT NULL
                );
                DELETE FROM equipment_incidents;
                DELETE FROM monthly_sla_metrics;
                INSERT INTO equipment_incidents VALUES
                    ('INC-01', '金山A区', '高压断路器', 'high', 95, '2026-08'),
                    ('INC-02', '金山A区', '压力传感器', 'medium', 35, '2026-08'),
                    ('INC-03', '张江片区', '高压断路器', 'high', 120, '2026-09'),
                    ('INC-04', '金山B区', '气体探头', 'high', 60, '2026-09'),
                    ('INC-05', '张江片区', '配电变压器', 'low', 15, '2026-10'),
                    ('INC-06', '金山A区', '高压断路器', 'high', 80, '2026-10');
                INSERT INTO monthly_sla_metrics VALUES
                    ('2026-08', '金山A区', 99.42, 65.0),
                    ('2026-09', '张江片区', 99.18, 90.0),
                    ('2026-10', '金山A区', 99.68, 47.5);
                """
            )
            self._anchor_conn.commit()

    def inspect_schema(self) -> str:
        with self._write_lock:
            tables = self._anchor_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            schema_desc: dict[str, list[dict[str, str | bool]]] = {}
            for row in tables:
                tname = row["name"]
                cols = self._anchor_conn.execute(f"PRAGMA table_info({tname})").fetchall()
                schema_desc[tname] = [
                    {"column": c["name"], "type": c["type"], "pk": bool(c["pk"])}
                    for c in cols
                ]
        return json.dumps(schema_desc, ensure_ascii=False, indent=2)

    @staticmethod
    def _validate_readonly_sql(sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise SQLSafetyError("SQL query cannot be empty.")
        skeleton = _strip_sql_literals_and_comments(cleaned).strip().rstrip(";").strip()
        if ";" in skeleton:
            raise SQLSafetyError("Multi-statement SQL is blocked for security.")
        if not re.match(r"^(SELECT|WITH)\b", skeleton, re.IGNORECASE):
            raise SQLSafetyError("Only SELECT / WITH read-only queries are permitted here.")
        match = _FORBIDDEN_READONLY_PATTERN.search(skeleton)
        if match:
            raise SQLSafetyError(f"Forbidden SQL keyword in read-only sandbox: {match.group(0)}")
        return cleaned

    def query_readonly(self, sql: str) -> str:
        cleaned = self._validate_readonly_sql(sql)
        deadline = time.perf_counter() + self.query_timeout_seconds
        ro_conn = self._open_readonly_connection(deadline)
        try:
            cur = ro_conn.execute(cleaned)
            fetched = cur.fetchmany(self.max_rows + 1)
            truncated = len(fetched) > self.max_rows
            rows = [dict(r) for r in fetched[: self.max_rows]]
            return json.dumps(
                {
                    "row_count": len(rows),
                    "truncated": truncated,
                    "rows": rows,
                    "executed_sql": cleaned,
                },
                ensure_ascii=False,
            )
        except sqlite3.DatabaseError as exc:
            msg = str(exc)
            if "not authorized" in msg.lower() or "attempt to write a readonly database" in msg.lower():
                raise SQLSafetyError(f"SQLite kernel authorizer blocked non-readonly action: {exc}") from exc
            if "interrupted" in msg.lower():
                raise SQLSafetyError(
                    f"SQL query exceeded timeout ({self.query_timeout_seconds}s) and was interrupted."
                ) from exc
            raise
        finally:
            ro_conn.close()

    def execute_mutating(self, sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        skeleton = _strip_sql_literals_and_comments(cleaned).strip().rstrip(";").strip()
        if ";" in skeleton:
            raise SQLSafetyError("Multi-statement SQL is blocked even for approved mutations.")
        with self._write_lock:
            cur = self._anchor_conn.execute(cleaned)
            self._anchor_conn.commit()
            affected = cur.rowcount
        return json.dumps({"affected_rows": affected, "executed_sql": cleaned}, ensure_ascii=False)

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                name="inspect_sql_schema",
                description="查看当前业务数据库的所有表名和字段结构（写 SQL 前必须先调用）",
                parameters={"type": "object", "properties": {}, "required": []},
                risk_level=RiskLevel.LOW,
            ),
            self.inspect_schema,
        )
        registry.register(
            ToolSpec(
                name="query_sql_readonly",
                description="在 SQLite 内核级 Authorizer 只读安全沙箱中执行 SELECT/WITH 查询，自动拦截写操作与慢查询",
                parameters={
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "单条 SELECT SQL 语句"}},
                    "required": ["sql"],
                },
                risk_level=RiskLevel.LOW,
            ),
            self.query_readonly,
        )
        registry.register(
            ToolSpec(
                name="execute_mutating_sql",
                description="执行写库 SQL（UPDATE/DELETE/INSERT），高危操作将自动挂起等待人工审批",
                parameters={
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "单条写库 SQL 语句"}},
                    "required": ["sql"],
                },
                risk_level=RiskLevel.HIGH,
            ),
            self.execute_mutating,
        )

    def close(self) -> None:
        with self._write_lock:
            self._anchor_conn.close()
