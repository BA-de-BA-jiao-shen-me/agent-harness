from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec

_FORBIDDEN_READONLY_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


class SQLSafetyError(ValueError):
    """Raised when a SQL statement violates read-only or anti-injection policy."""


class SQLSandbox:
    """
    Governed SQL Analytics Sandbox supporting:
    1. Schema introspection (`inspect_sql_schema`) so the LLM never hallucinates column names.
    2. Read-only SELECT execution (`query_sql_readonly`) with injection & mutating guards + auto LIMIT.
    3. Mutating SQL execution (`execute_mutating_sql`) gated at `RiskLevel.HIGH` via ApprovalBroker.
    """

    def __init__(self, db_path: str | Path = ":memory:", max_rows: int = 100) -> None:
        self._db_path = str(db_path)
        self.max_rows = max_rows
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def seed_demo_enterprise_data(self) -> None:
        """Populate realistic industrial/operations demo tables for testing and interviews."""
        cur = self._conn.cursor()
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
        self._conn.commit()

    def inspect_schema(self) -> str:
        tables = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema_desc: dict[str, list[dict[str, str]]] = {}
        for row in tables:
            tname = row["name"]
            cols = self._conn.execute(f"PRAGMA table_info({tname})").fetchall()
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
        # Block multi-statement SQL injection
        if ";" in cleaned:
            raise SQLSafetyError("Multi-statement SQL is blocked for security.")
        if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
            raise SQLSafetyError("Only SELECT / WITH read-only queries are permitted here.")
        match = _FORBIDDEN_READONLY_PATTERN.search(cleaned)
        if match:
            raise SQLSafetyError(f"Forbidden SQL keyword in read-only sandbox: {match.group(0)}")
        return cleaned

    def query_readonly(self, sql: str) -> str:
        cleaned = self._validate_readonly_sql(sql)
        if not re.search(r"\bLIMIT\s+\d+\b", cleaned, re.IGNORECASE):
            cleaned = f"{cleaned} LIMIT {self.max_rows}"
        rows = self._conn.execute(cleaned).fetchall()
        result = [dict(r) for r in rows]
        return json.dumps(
            {"row_count": len(result), "rows": result, "executed_sql": cleaned},
            ensure_ascii=False,
        )

    def execute_mutating(self, sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        if ";" in cleaned:
            raise SQLSafetyError("Multi-statement SQL is blocked even for approved mutations.")
        cur = self._conn.execute(cleaned)
        self._conn.commit()
        return json.dumps({"affected_rows": cur.rowcount, "executed_sql": cleaned}, ensure_ascii=False)

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
                description="在只读安全沙箱中执行 SELECT/WITH 查询，自动拦截写操作与注入并返回 JSON 结果",
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
        self._conn.close()
