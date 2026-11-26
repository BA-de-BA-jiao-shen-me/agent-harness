from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from pyharness.runtime.schemas import RiskLevel, ToolSpec
from pyharness.security.auth import current_principal

current_project_id: ContextVar[str] = ContextVar("pyharness_project_id", default="default")


@contextmanager
def use_project_id(project_id: str) -> Iterator[str]:
    """Bind `project_id` to the current async/thread execution context and reset on exit."""
    normalized = (project_id or "").strip() or "default"
    token = current_project_id.set(normalized)
    try:
        yield normalized
    finally:
        current_project_id.reset(token)


_FALLBACK_IANA_OFFSETS: dict[str, int] = {
    "UTC": 0,
    "Asia/Shanghai": 8,
    "Asia/Tokyo": 9,
    "Asia/Singapore": 8,
    "Europe/London": 0,
    "America/New_York": -5,
    "America/Los_Angeles": -8,
}

_VALID_SCOPES: frozenset[str] = frozenset({"project", "global"})

_EN_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "an",
        "and",
        "or",
        "as",
        "it",
        "this",
        "that",
        "from",
        "what",
        "who",
        "how",
        "when",
        "where",
        "which",
        "we",
        "our",
        "my",
        "your",
        "do",
        "does",
        "did",
        "can",
        "will",
        "should",
        "about",
    }
)

_CN_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没",
        "看",
        "好",
        "这",
        "那",
        "吗",
        "呢",
        "吧",
        "啊",
        "么",
        "什",
        "哪",
        "谁",
    }
)


def _tokenize(text: str) -> set[str]:
    """
    Bilingual EN/CN tokenizer for lexical memory recall.

    Extracts:
    - Lowercase English/numeric tokens (length >= 2, excluding common filler stopwords).
    - CJK character bigrams and meaningful individual CJK characters.
    """
    if not text:
        return set()

    lowered = text.lower()
    tokens: set[str] = set()

    for match in _EN_TOKEN_RE.findall(lowered):
        if match not in _EN_STOPWORDS:
            tokens.add(match)

    for run in _CJK_RUN_RE.findall(lowered):
        run_len = len(run)
        for idx in range(run_len):
            ch = run[idx]
            if ch not in _CN_STOPWORDS:
                tokens.add(ch)
            if idx + 1 < run_len:
                bigram = run[idx : idx + 2]
                if not (bigram[0] in _CN_STOPWORDS and bigram[1] in _CN_STOPWORDS):
                    tokens.add(bigram)

    return tokens


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in tags:
        tag = str(raw).strip()
        if tag and tag not in seen:
            seen.add(tag)
            cleaned.append(tag)
    return cleaned


class TrustedTemporalContext(BaseModel):
    """
    Server-verified temporal anchor injected into LLM context so relative dates
    are never guessed or hallucinated.
    """

    captured_at_utc: str
    user_local_time: str
    user_date: str
    weekday: str
    user_timezone: str
    tz_fallback_used: bool = False

    @classmethod
    def capture(
        cls,
        timezone_name: str | None = None,
        now_utc: datetime | None = None,
    ) -> TrustedTemporalContext:
        if now_utc is None:
            utc_dt = datetime.now(timezone.utc)
        elif now_utc.tzinfo is None:
            utc_dt = now_utc.replace(tzinfo=timezone.utc)
        else:
            utc_dt = now_utc.astimezone(timezone.utc)

        requested_tz = (timezone_name or "").strip() or "UTC"
        tz_fallback_used = False
        resolved_tz_name = requested_tz
        tz_obj: tzinfo

        try:
            tz_obj = ZoneInfo(requested_tz)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            if requested_tz in _FALLBACK_IANA_OFFSETS:
                tz_obj = timezone(timedelta(hours=_FALLBACK_IANA_OFFSETS[requested_tz]))
                resolved_tz_name = requested_tz
            else:
                tz_obj = timezone.utc
                resolved_tz_name = "UTC"
                tz_fallback_used = True

        local_dt = utc_dt.astimezone(tz_obj)
        captured_at_utc = utc_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        user_local_time = local_dt.replace(microsecond=0).isoformat()

        return cls(
            captured_at_utc=captured_at_utc,
            user_local_time=user_local_time,
            user_date=local_dt.strftime("%Y-%m-%d"),
            weekday=local_dt.strftime("%A"),
            user_timezone=resolved_tz_name,
            tz_fallback_used=tz_fallback_used,
        )

    def render_prompt_block(self) -> str:
        lines = [
            "[Trusted Server Temporal Context]",
            f"- Server UTC Time: {self.captured_at_utc}",
            f"- User Local Time: {self.user_local_time}",
            f"- User Local Date: {self.user_date} ({self.weekday})",
            f"- User Timezone: {self.user_timezone}",
        ]
        if self.tz_fallback_used:
            lines.append(
                "- Notice: Requested timezone was invalid or unavailable; defaulted to UTC."
            )
        lines.append(
            "- Authoritative Clock Rule: Never invent or guess the current date or time. "
            "Interpret all relative time expressions ('today', 'yesterday', 'tomorrow', "
            "'this week', 'this month', 'last quarter') strictly against this server clock."
        )
        return "\n".join(lines)


class MemoryEntry(BaseModel):
    """Durable project-scoped or global user memory record."""

    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    tenant_id: str = "default"
    user_id: str = "anonymous"
    project_id: str = "default"
    scope: str = "project"
    category: str = "fact"
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    access_count: int = 0


class ProjectMemoryStore:
    """
    WAL-hardened SQLite long-term memory store with strict tenant/user isolation,
    project + global scoping, and bilingual EN/CN lexical recall.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        self._init_schema()

    def _configure_pragmas(self) -> None:
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    project_id TEXT NOT NULL DEFAULT 'default',
                    scope TEXT NOT NULL DEFAULT 'project',
                    category TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_tenant_user_project
                ON harness_memories (tenant_id, user_id, project_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_tenant_user_scope
                ON harness_memories (tenant_id, user_id, scope)
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        try:
            raw_tags = json.loads(row["tags_json"] or "[]")
            tags = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []
        except (TypeError, ValueError):
            tags = []
        return MemoryEntry(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            scope=row["scope"],
            category=row["category"],
            content=row["content"],
            tags=tags,
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
            access_count=int(row["access_count"]),
        )

    def save_sync(
        self,
        content: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        project_id: str = "default",
        scope: str = "project",
        category: str = "fact",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        cleaned_content = (content or "").strip()
        if not cleaned_content:
            raise ValueError("Memory content must not be empty.")

        normalized_scope = (scope or "").strip().lower()
        if normalized_scope not in _VALID_SCOPES:
            raise ValueError(
                f"Invalid memory scope {scope!r}; expected one of {sorted(_VALID_SCOPES)}."
            )

        norm_tenant = (tenant_id or "").strip() or "default"
        norm_user = (user_id or "").strip() or "anonymous"
        norm_project = (project_id or "").strip() or "default"
        norm_category = (category or "").strip() or "fact"
        norm_tags = _normalize_tags(tags)
        now_ms = int(time.time() * 1000)

        with self._lock:
            existing = self._conn.execute(
                """
                SELECT memory_id, tenant_id, user_id, project_id, scope, category,
                       content, tags_json, created_at_ms, updated_at_ms, access_count
                FROM harness_memories
                WHERE tenant_id = ? AND user_id = ? AND project_id = ?
                  AND scope = ? AND content = ?
                LIMIT 1
                """,
                (norm_tenant, norm_user, norm_project, normalized_scope, cleaned_content),
            ).fetchone()

            if existing is not None:
                prev = self._row_to_entry(existing)
                merged_tags = _normalize_tags([*prev.tags, *norm_tags])
                self._conn.execute(
                    """
                    UPDATE harness_memories
                    SET category = ?, tags_json = ?, updated_at_ms = ?
                    WHERE memory_id = ?
                    """,
                    (
                        norm_category,
                        json.dumps(merged_tags, ensure_ascii=False),
                        now_ms,
                        prev.memory_id,
                    ),
                )
                self._conn.commit()
                return prev.model_copy(
                    update={
                        "category": norm_category,
                        "tags": merged_tags,
                        "updated_at_ms": now_ms,
                    }
                )

            memory_id = f"mem_{uuid.uuid4().hex[:12]}"
            entry = MemoryEntry(
                memory_id=memory_id,
                tenant_id=norm_tenant,
                user_id=norm_user,
                project_id=norm_project,
                scope=normalized_scope,
                category=norm_category,
                content=cleaned_content,
                tags=norm_tags,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
                access_count=0,
            )
            self._conn.execute(
                """
                INSERT INTO harness_memories
                (memory_id, tenant_id, user_id, project_id, scope, category,
                 content, tags_json, created_at_ms, updated_at_ms, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.memory_id,
                    entry.tenant_id,
                    entry.user_id,
                    entry.project_id,
                    entry.scope,
                    entry.category,
                    entry.content,
                    json.dumps(entry.tags, ensure_ascii=False),
                    entry.created_at_ms,
                    entry.updated_at_ms,
                    entry.access_count,
                ),
            )
            self._conn.commit()
            return entry

    async def save(
        self,
        content: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        project_id: str = "default",
        scope: str = "project",
        category: str = "fact",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        return await asyncio.to_thread(
            self.save_sync,
            content,
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            scope=scope,
            category=category,
            tags=tags,
        )

    def recall_sync(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        project_id: str = "default",
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        limit = max(0, int(top_k))
        if limit == 0:
            return []

        norm_tenant = (tenant_id or "").strip() or "default"
        norm_user = (user_id or "").strip() or "anonymous"
        norm_project = (project_id or "").strip() or "default"
        cleaned_query = (query or "").strip()

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT memory_id, tenant_id, user_id, project_id, scope, category,
                       content, tags_json, created_at_ms, updated_at_ms, access_count
                FROM harness_memories
                WHERE tenant_id = ? AND user_id = ?
                  AND (scope = 'global' OR project_id = ?)
                ORDER BY updated_at_ms DESC, created_at_ms DESC, memory_id ASC
                """,
                (norm_tenant, norm_user, norm_project),
            ).fetchall()

            candidates = [self._row_to_entry(row) for row in rows]
            if not candidates:
                return []

            if not cleaned_query:
                selected = candidates[:limit]
            else:
                query_tokens = _tokenize(cleaned_query)
                query_lower = cleaned_query.lower()
                now_ms = int(time.time() * 1000)

                scored: list[tuple[float, MemoryEntry]] = []
                for entry in candidates:
                    content_tokens = _tokenize(entry.content)
                    tag_tokens: set[str] = set()
                    for tag in entry.tags:
                        tag_tokens.update(_tokenize(tag))
                        tag_low = tag.strip().lower()
                        if tag_low:
                            tag_tokens.add(tag_low)

                    doc_tokens = content_tokens | tag_tokens
                    total_overlap = query_tokens & doc_tokens
                    exact_hit = bool(query_lower) and (
                        query_lower in entry.content.lower()
                        or any(query_lower in t.lower() for t in entry.tags)
                    )

                    if not total_overlap and not exact_hit:
                        continue

                    if query_tokens:
                        union_size = len(query_tokens | doc_tokens)
                        jaccard = len(total_overlap) / union_size if union_size else 0.0
                        coverage = len(total_overlap) / len(query_tokens)
                        tag_overlap = query_tokens & tag_tokens
                        tag_bonus = 0.25 * (len(tag_overlap) / len(query_tokens))
                    else:
                        jaccard = 0.0
                        coverage = 0.0
                        tag_bonus = 0.0

                    exact_bonus = 0.15 if exact_hit else 0.0
                    age_days = max(0.0, (now_ms - entry.updated_at_ms) / 86_400_000.0)
                    recency_boost = 0.05 / (1.0 + age_days)

                    score = (0.5 * jaccard) + (0.5 * coverage) + tag_bonus + exact_bonus + recency_boost
                    if score > 0.0:
                        scored.append((score, entry))

                scored.sort(key=lambda item: (-item[0], -item[1].updated_at_ms, item[1].memory_id))
                selected = [entry for _, entry in scored[:limit]]

            if selected:
                updated_results: list[MemoryEntry] = []
                for entry in selected:
                    self._conn.execute(
                        """
                        UPDATE harness_memories
                        SET access_count = access_count + 1
                        WHERE memory_id = ?
                        """,
                        (entry.memory_id,),
                    )
                    updated_results.append(
                        entry.model_copy(update={"access_count": entry.access_count + 1})
                    )
                self._conn.commit()
                return updated_results

            return []

    async def recall(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        project_id: str = "default",
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        return await asyncio.to_thread(
            self.recall_sync,
            query,
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k,
        )

    def list_memories_sync(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[MemoryEntry]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT memory_id, tenant_id, user_id, project_id, scope, category,
                       content, tags_json, created_at_ms, updated_at_ms, access_count
                FROM harness_memories
                WHERE {where_sql}
                ORDER BY updated_at_ms DESC, memory_id ASC
                """,
                params,
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]

    async def list_memories(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[MemoryEntry]:
        return await asyncio.to_thread(
            self.list_memories_sync,
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
        )

    def delete_sync(
        self,
        memory_id: str,
        *,
        tenant_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT memory_id, tenant_id, user_id
                FROM harness_memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                return False

            if not is_admin:
                if row["tenant_id"] != tenant_id:
                    raise PermissionError(
                        f"Cross-tenant delete denied for memory '{memory_id}' "
                        f"(caller tenant '{tenant_id}' != owner tenant '{row['tenant_id']}')."
                    )
                if row["user_id"] != user_id:
                    raise PermissionError(
                        f"Cross-user delete denied for memory '{memory_id}' "
                        f"(caller user '{user_id}' != owner user '{row['user_id']}')."
                    )

            self._conn.execute(
                "DELETE FROM harness_memories WHERE memory_id = ?",
                (memory_id,),
            )
            self._conn.commit()
            return True

    async def delete(
        self,
        memory_id: str,
        *,
        tenant_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> bool:
        return await asyncio.to_thread(
            self.delete_sync,
            memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    @staticmethod
    def render_prompt_block(entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = ["[Recalled Long-Term Memory]"]
        for entry in entries:
            tag_suffix = f" (tags: {', '.join(entry.tags)})" if entry.tags else ""
            lines.append(
                f"- [{entry.scope} | {entry.category}]{tag_suffix} {entry.content}"
            )
        return "\n".join(lines)

    def register_tools(self, registry: Any) -> None:
        """Mount `remember_fact` and `recall_memory` tools into a `ToolRegistry`."""

        async def _remember_fact(
            content: str,
            scope: str = "project",
            category: str = "fact",
            tags: list[str] | None = None,
        ) -> str:
            principal = current_principal.get()
            tenant_id = principal.tenant_id if principal is not None else "default"
            user_id = principal.user_id if principal is not None else "anonymous"
            project_id = current_project_id.get()
            entry = await self.save(
                content,
                tenant_id=tenant_id,
                user_id=user_id,
                project_id=project_id,
                scope=scope,
                category=category,
                tags=tags,
            )
            return json.dumps(entry.model_dump(), ensure_ascii=False)

        async def _recall_memory(
            query: str = "",
            top_k: int = 5,
        ) -> str:
            principal = current_principal.get()
            tenant_id = principal.tenant_id if principal is not None else "default"
            user_id = principal.user_id if principal is not None else "anonymous"
            project_id = current_project_id.get()
            entries = await self.recall(
                query,
                tenant_id=tenant_id,
                user_id=user_id,
                project_id=project_id,
                top_k=top_k,
            )
            return json.dumps(
                {
                    "count": len(entries),
                    "project_id": project_id,
                    "memories": [e.model_dump() for e in entries],
                },
                ensure_ascii=False,
            )

        registry.register(
            ToolSpec(
                name="remember_fact",
                description=(
                    "Persist an important fact, decision, constraint, or user preference into "
                    "long-term memory scoped to the current project or globally for the user."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Concise, self-contained fact or preference to remember.",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["project", "global"],
                            "description": "'project' for current project only; 'global' across all projects for this user.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Memory category such as 'fact', 'preference', 'decision', or 'constraint'.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of tags to improve lexical recall.",
                        },
                    },
                    "required": ["content"],
                },
                risk_level=RiskLevel.LOW,
            ),
            _remember_fact,
        )

        registry.register(
            ToolSpec(
                name="recall_memory",
                description=(
                    "Search and recall relevant long-term memories for the current user and "
                    "project (including global user preferences)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords or topic to match; leave empty to retrieve recent memories.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of memories to return (default 5).",
                        },
                    },
                    "required": [],
                },
                risk_level=RiskLevel.LOW,
            ),
            _recall_memory,
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


__all__ = [
    "MemoryEntry",
    "ProjectMemoryStore",
    "TrustedTemporalContext",
    "_tokenize",
    "current_project_id",
    "use_project_id",
]
