from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE|COPY|CALL|DO|LOCK|MERGE|VACUUM)\b",
    re.IGNORECASE,
)


def _strip_literals_and_comments(sql: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\r\n]*", " ", without_block)
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", without_line)


def _validate_readonly_query(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL query cannot be empty.")
    skeleton = _strip_literals_and_comments(cleaned).strip().rstrip(";").strip()
    if ";" in skeleton:
        raise PermissionError("Multi-statement SQL is blocked on ExternalReadOnlySQLAdapter.")
    if not re.match(r"^(SELECT|WITH)\b", skeleton, re.IGNORECASE):
        raise PermissionError("Only SELECT / WITH read-only queries are permitted.")
    match = _FORBIDDEN_SQL_KEYWORDS.search(skeleton)
    if match:
        raise PermissionError(
            f"Forbidden SQL keyword in read-only warehouse adapter: {match.group(0)}"
        )
    return cleaned


def _detect_dialect(dsn: str, explicit_dialect: str | None) -> str:
    if explicit_dialect:
        norm = explicit_dialect.strip().lower()
        if norm in ("postgres", "postgresql", "pg"):
            return "postgresql"
        if norm in ("mysql", "mariadb"):
            return "mysql"
        raise ValueError(
            f"Unsupported SQL dialect '{explicit_dialect}'. Supported: postgresql, mysql."
        )

    scheme = urlparse(dsn).scheme.lower().split("+")[0]
    if scheme in ("postgres", "postgresql"):
        return "postgresql"
    if scheme in ("mysql", "mariadb"):
        return "mysql"
    raise ValueError(
        f"Cannot infer SQL dialect from DSN scheme '{scheme}'. "
        "Use a postgresql:// or mysql:// DSN or pass `dialect=` explicitly."
    )


def _redact_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    if not parsed.password:
        return dsn
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    userinfo = f"{parsed.username}:***" if parsed.username else "***"
    return parsed._replace(netloc=f"{userinfo}@{host}").geturl()


class SQLBackendAdapter(ABC):
    """
    Pluggable SQL Engine Seam decoupling `SQLSandbox` tools from the underlying database engine
    (local kernel-authorizer SQLite vs. external enterprise PostgreSQL/MySQL read replicas).
    """

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def inspect_schema(self) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def execute_readonly(
        self,
        sql: str,
        max_rows: int = 100,
        timeout_seconds: float = 3.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        raise NotImplementedError

    @abstractmethod
    def execute_mutating(self, sql: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ExternalReadOnlySQLAdapter(SQLBackendAdapter):
    """
    Production adapter seam for enterprise read-only PostgreSQL / MySQL warehouse replicas.

    Security & reliability controls:
    1. Defense-in-depth read-only query validation (single SELECT/WITH statement only).
    2. Engine-enforced read-only transaction (`SET TRANSACTION READ ONLY`) and query timeout
       (`SET LOCAL statement_timeout` for PostgreSQL; `SET SESSION max_execution_time` for MySQL).
    3. Unconditional rollback in `finally` (never commits).
    4. Hard row cap (`cursor.fetchmany(max_rows + 1)`) preventing warehouse OOM responses.
    5. Unconditional rejection of `execute_mutating()` with `PermissionError`.
    6. Lazy driver loading for `psycopg` / `psycopg2` (PostgreSQL) and `pymysql` (MySQL), plus
       an injectable `connection_factory` seam for zero-dependency unit testing.
    """

    def __init__(
        self,
        dsn: str,
        *,
        dialect: str | None = None,
        schema_name: str | None = None,
        connect_timeout_seconds: float = 5.0,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not dsn.strip():
            raise ValueError("ExternalReadOnlySQLAdapter requires a non-empty DSN.")
        self.dsn = dsn.strip()
        self.dialect = _detect_dialect(self.dsn, dialect)
        self.schema_name = schema_name or ("public" if self.dialect == "postgresql" else None)
        self.connect_timeout_seconds = max(1.0, float(connect_timeout_seconds))
        self._connection_factory = connection_factory

    @property
    def adapter_name(self) -> str:
        return f"external-readonly-{self.dialect}"

    @property
    def redacted_dsn(self) -> str:
        return _redact_dsn(self.dsn)

    def _open_connection(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()

        timeout_sec = int(max(1, round(self.connect_timeout_seconds)))
        if self.dialect == "postgresql":
            try:
                import psycopg  # type: ignore[import-not-found]

                return psycopg.connect(self.dsn, connect_timeout=timeout_sec, autocommit=False)
            except ImportError:
                pass
            try:
                import psycopg2  # type: ignore[import-not-found]

                conn = psycopg2.connect(self.dsn, connect_timeout=timeout_sec)
                conn.autocommit = False
                return conn
            except ImportError as exc:
                raise RuntimeError(
                    "ExternalReadOnlySQLAdapter(postgresql) requires 'psycopg' or 'psycopg2' "
                    "(`pip install psycopg[binary]`)."
                ) from exc

        # MySQL / MariaDB via PyMySQL
        try:
            import pymysql  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ExternalReadOnlySQLAdapter(mysql) requires 'pymysql' (`pip install pymysql`)."
            ) from exc

        parsed = urlparse(self.dsn)
        query = parse_qs(parsed.query)
        charset = query.get("charset", ["utf8mb4"])[0]
        return pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=parsed.path.lstrip("/") or None,
            charset=charset,
            connect_timeout=timeout_sec,
            autocommit=False,
        )

    def _configure_readonly_session(self, cursor: Any, timeout_seconds: float) -> None:
        timeout_ms = max(1, int(timeout_seconds * 1000))
        if self.dialect == "postgresql":
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
        else:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(f"SET SESSION max_execution_time = {timeout_ms}")
            cursor.execute("START TRANSACTION READ ONLY")

    @staticmethod
    def _rows_to_dicts(cursor: Any, raw_rows: list[Any]) -> list[dict[str, Any]]:
        if not raw_rows:
            return []
        if isinstance(raw_rows[0], Mapping):
            return [dict(r) for r in raw_rows]
        col_names = [col[0] for col in (cursor.description or ())]
        return [dict(zip(col_names, row)) for row in raw_rows]

    def inspect_schema(self) -> dict[str, list[dict[str, Any]]]:
        conn = self._open_connection()
        try:
            if hasattr(conn, "autocommit"):
                try:
                    conn.autocommit = False
                except Exception:
                    pass
            cur = conn.cursor()
            try:
                self._configure_readonly_session(cur, timeout_seconds=self.connect_timeout_seconds)
                if self.dialect == "postgresql":
                    target_schema = self.schema_name or "public"
                    cur.execute(
                        """
                        SELECT
                            c.table_name,
                            c.column_name,
                            c.data_type,
                            EXISTS (
                                SELECT 1
                                FROM information_schema.table_constraints tc
                                JOIN information_schema.key_column_usage kcu
                                  ON tc.constraint_name = kcu.constraint_name
                                 AND tc.table_schema = kcu.table_schema
                                WHERE tc.constraint_type = 'PRIMARY KEY'
                                  AND tc.table_schema = c.table_schema
                                  AND tc.table_name = c.table_name
                                  AND kcu.column_name = c.column_name
                            ) AS is_pk
                        FROM information_schema.columns c
                        WHERE c.table_schema = %s
                        ORDER BY c.table_name, c.ordinal_position
                        """,
                        (target_schema,),
                    )
                else:
                    if self.schema_name:
                        cur.execute(
                            """
                            SELECT
                                table_name,
                                column_name,
                                data_type,
                                (column_key = 'PRI') AS is_pk
                            FROM information_schema.columns
                            WHERE table_schema = %s
                            ORDER BY table_name, ordinal_position
                            """,
                            (self.schema_name,),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                table_name,
                                column_name,
                                data_type,
                                (column_key = 'PRI') AS is_pk
                            FROM information_schema.columns
                            WHERE table_schema = DATABASE()
                            ORDER BY table_name, ordinal_position
                            """
                        )
                rows = self._rows_to_dicts(cur, cur.fetchall())
            finally:
                cur.close()
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        schema_desc: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tname = str(row.get("table_name") or row.get("TABLE_NAME") or "")
            cname = str(row.get("column_name") or row.get("COLUMN_NAME") or "")
            dtype = str(row.get("data_type") or row.get("DATA_TYPE") or "")
            is_pk = bool(row.get("is_pk") or row.get("IS_PK") or False)
            if not tname:
                continue
            schema_desc.setdefault(tname, []).append(
                {"column": cname, "type": dtype, "pk": is_pk}
            )
        return schema_desc

    def execute_readonly(
        self,
        sql: str,
        max_rows: int = 100,
        timeout_seconds: float = 3.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        cleaned = _validate_readonly_query(sql)
        row_cap = max(1, int(max_rows))
        conn = self._open_connection()
        try:
            if hasattr(conn, "autocommit"):
                try:
                    conn.autocommit = False
                except Exception:
                    pass
            cur = conn.cursor()
            try:
                self._configure_readonly_session(cur, timeout_seconds=timeout_seconds)
                cur.execute(cleaned)
                fetched = cur.fetchmany(row_cap + 1)
                truncated = len(fetched) > row_cap
                rows = self._rows_to_dicts(cur, fetched[:row_cap])
                return rows, truncated
            finally:
                cur.close()
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def execute_mutating(self, sql: str) -> int:
        raise PermissionError(
            f"{self.adapter_name} forbids mutating SQL statements on read-only replicas."
        )

    def close(self) -> None:
        return None
