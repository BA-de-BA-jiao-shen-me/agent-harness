from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pyharness.runtime.schemas import (
    HarnessHook,
    RiskLevel,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from pyharness.security.auth import current_principal

_SAFE_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_artifact_id(artifact_id: str) -> str:
    if not isinstance(artifact_id, str) or not _SAFE_ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise ValueError(
            f"Invalid artifact_id {artifact_id!r}: must match ^[A-Za-z0-9_\\-]+$."
        )
    return artifact_id


def _build_preview(content: str, max_chars: int = 240) -> str:
    if len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 3)] + "..."


def _is_s3_not_found(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        err = response.get("Error")
        if isinstance(err, Mapping):
            code = str(err.get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return True
    code = str(getattr(exc, "code", ""))
    return code in {"NoSuchKey", "404", "NotFound"}


async def _extract_s3_body_text(resp: Any) -> str:
    body = resp["Body"] if isinstance(resp, Mapping) else getattr(resp, "Body", resp)
    if hasattr(body, "read") and callable(body.read):
        if inspect.iscoroutinefunction(body.read):
            raw = await body.read()
        else:
            raw = await asyncio.to_thread(body.read)
            if inspect.isawaitable(raw):
                raw = await raw
    else:
        raw = body

    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8")
    return str(raw)


class ArtifactRecord(BaseModel):
    """Durable metadata handle for an offloaded tool output artifact."""

    artifact_id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:12]}")
    session_id: str
    turn_id: str
    tenant_id: str = "default"
    tool_name: str = ""
    content_type: str = "text/plain"
    char_count: int = 0
    preview: str = ""
    storage_uri: str = ""
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactAuthorizationError(PermissionError):
    """Raised when a caller attempts to access an artifact belonging to another tenant."""


class BaseArtifactStore(ABC):
    """
    Pluggable artifact persistence seam supporting tenant-isolated blob storage
    and paginated character-slice inspection.
    """

    @property
    @abstractmethod
    def store_name(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _enforce_tenant_acl(
        record: ArtifactRecord,
        *,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> None:
        if tenant_id is not None and not is_admin and record.tenant_id != tenant_id:
            raise ArtifactAuthorizationError(
                f"Tenant '{tenant_id}' is not authorized to access artifact "
                f"'{record.artifact_id}' (owned by tenant '{record.tenant_id}')."
            )

    @staticmethod
    def _slice_content(content: str, offset: int, limit: int) -> tuple[str, bool]:
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        end = safe_offset + safe_limit
        chunk = content[safe_offset:end]
        has_more = end < len(content)
        return chunk, has_more

    @abstractmethod
    async def save(
        self,
        content: str,
        *,
        session_id: str,
        turn_id: str,
        tenant_id: str = "default",
        tool_name: str = "",
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        raise NotImplementedError

    @abstractmethod
    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def read_slice(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4000,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> tuple[ArtifactRecord, str, bool]:
        raise NotImplementedError

    @abstractmethod
    async def list_for_session(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> list[ArtifactRecord]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, artifact_id: str) -> bool:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class InMemoryArtifactStore(BaseArtifactStore):
    """Process-local in-memory artifact store suitable for tests and ephemeral sessions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, ArtifactRecord] = {}
        self._blobs: dict[str, str] = {}

    @property
    def store_name(self) -> str:
        return "memory"

    async def save(
        self,
        content: str,
        *,
        session_id: str,
        turn_id: str,
        tenant_id: str = "default",
        tool_name: str = "",
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        record = ArtifactRecord(
            artifact_id=artifact_id,
            session_id=session_id,
            turn_id=turn_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            content_type=content_type,
            char_count=len(content),
            preview=_build_preview(content),
            storage_uri=f"memory://{artifact_id}",
            created_at_ms=int(time.time() * 1000),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records[artifact_id] = record
            self._blobs[artifact_id] = content
        return record.model_copy(deep=True)

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            rec = self._records.get(artifact_id)
            return rec.model_copy(deep=True) if rec is not None else None

    async def read_slice(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4000,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> tuple[ArtifactRecord, str, bool]:
        with self._lock:
            rec = self._records.get(artifact_id)
            if rec is None:
                raise KeyError(f"Artifact '{artifact_id}' not found.")
            self._enforce_tenant_acl(rec, tenant_id=tenant_id, is_admin=is_admin)
            content = self._blobs.get(artifact_id)
            if content is None:
                raise KeyError(f"Artifact '{artifact_id}' not found.")
            record_copy = rec.model_copy(deep=True)

        chunk, has_more = self._slice_content(content, offset=offset, limit=limit)
        return record_copy, chunk, has_more

    async def list_for_session(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> list[ArtifactRecord]:
        with self._lock:
            items = [
                rec.model_copy(deep=True)
                for rec in self._records.values()
                if rec.session_id == session_id
                and (tenant_id is None or is_admin or rec.tenant_id == tenant_id)
            ]
        items.sort(key=lambda r: (r.created_at_ms, r.artifact_id))
        return items

    async def delete(self, artifact_id: str) -> bool:
        with self._lock:
            existed = artifact_id in self._records or artifact_id in self._blobs
            self._records.pop(artifact_id, None)
            self._blobs.pop(artifact_id, None)
            return existed

    async def aclose(self) -> None:
        return None


class FileArtifactStore(BaseArtifactStore):
    """
    Local filesystem artifact store storing UTF-8 blobs at `<base_dir>/<artifact_id>.txt`
    and JSON metadata at `<base_dir>/<artifact_id>.meta.json` via `asyncio.to_thread()`.
    """

    def __init__(self, base_dir: str | Path = "./data/artifacts") -> None:
        self._base_dir = Path(base_dir).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def store_name(self) -> str:
        return "file"

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _artifact_paths(self, artifact_id: str) -> tuple[Path, Path]:
        safe_id = _validate_artifact_id(artifact_id)
        blob_path = (self._base_dir / f"{safe_id}.txt").resolve()
        meta_path = (self._base_dir / f"{safe_id}.meta.json").resolve()
        if blob_path.parent != self._base_dir or meta_path.parent != self._base_dir:
            raise ValueError(f"Path traversal detected for artifact_id {artifact_id!r}.")
        return blob_path, meta_path

    def _save_sync(
        self,
        content: str,
        *,
        session_id: str,
        turn_id: str,
        tenant_id: str,
        tool_name: str,
        content_type: str,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        blob_path, meta_path = self._artifact_paths(artifact_id)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            session_id=session_id,
            turn_id=turn_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            content_type=content_type,
            char_count=len(content),
            preview=_build_preview(content),
            storage_uri=blob_path.as_uri(),
            created_at_ms=int(time.time() * 1000),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            blob_path.write_text(content, encoding="utf-8")
            meta_path.write_text(record.model_dump_json(), encoding="utf-8")
        return record

    async def save(
        self,
        content: str,
        *,
        session_id: str,
        turn_id: str,
        tenant_id: str = "default",
        tool_name: str = "",
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        return await asyncio.to_thread(
            self._save_sync,
            content,
            session_id=session_id,
            turn_id=turn_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            content_type=content_type,
            metadata=metadata,
        )

    def _get_record_sync(self, artifact_id: str) -> ArtifactRecord | None:
        _, meta_path = self._artifact_paths(artifact_id)
        with self._lock:
            if not meta_path.is_file():
                return None
            return ArtifactRecord.model_validate_json(meta_path.read_text(encoding="utf-8"))

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await asyncio.to_thread(self._get_record_sync, artifact_id)

    def _read_slice_sync(
        self,
        artifact_id: str,
        *,
        offset: int,
        limit: int,
        tenant_id: str | None,
        is_admin: bool,
    ) -> tuple[ArtifactRecord, str, bool]:
        blob_path, meta_path = self._artifact_paths(artifact_id)
        with self._lock:
            if not meta_path.is_file():
                raise KeyError(f"Artifact '{artifact_id}' not found.")
            record = ArtifactRecord.model_validate_json(meta_path.read_text(encoding="utf-8"))
            self._enforce_tenant_acl(record, tenant_id=tenant_id, is_admin=is_admin)
            if not blob_path.is_file():
                raise KeyError(f"Artifact '{artifact_id}' not found.")
            content = blob_path.read_text(encoding="utf-8")
        chunk, has_more = self._slice_content(content, offset=offset, limit=limit)
        return record, chunk, has_more

    async def read_slice(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4000,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> tuple[ArtifactRecord, str, bool]:
        return await asyncio.to_thread(
            self._read_slice_sync,
            artifact_id,
            offset=offset,
            limit=limit,
            tenant_id=tenant_id,
            is_admin=is_admin,
        )

    def _list_for_session_sync(
        self,
        session_id: str,
        *,
        tenant_id: str | None,
        is_admin: bool,
    ) -> list[ArtifactRecord]:
        items: list[ArtifactRecord] = []
        with self._lock:
            if not self._base_dir.is_dir():
                return []
            for meta_path in self._base_dir.glob("*.meta.json"):
                try:
                    rec = ArtifactRecord.model_validate_json(
                        meta_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                if rec.session_id != session_id:
                    continue
                if tenant_id is not None and not is_admin and rec.tenant_id != tenant_id:
                    continue
                items.append(rec)
        items.sort(key=lambda r: (r.created_at_ms, r.artifact_id))
        return items

    async def list_for_session(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> list[ArtifactRecord]:
        return await asyncio.to_thread(
            self._list_for_session_sync,
            session_id,
            tenant_id=tenant_id,
            is_admin=is_admin,
        )

    def _delete_sync(self, artifact_id: str) -> bool:
        blob_path, meta_path = self._artifact_paths(artifact_id)
        deleted = False
        with self._lock:
            if blob_path.is_file():
                blob_path.unlink()
                deleted = True
            if meta_path.is_file():
                meta_path.unlink()
                deleted = True
        return deleted

    async def delete(self, artifact_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, artifact_id)

    async def aclose(self) -> None:
        return None


class S3ArtifactStore(BaseArtifactStore):
    """
    Object storage seam for AWS S3 / MinIO / S3-compatible services.

    Accepts either an injectable S3 client double (sync boto3-style or async coroutines)
    or lazily constructs a `boto3.client("s3", ...)` on first use.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "pyharness/artifacts/",
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        cleaned_bucket = (bucket or "").strip()
        if not cleaned_bucket:
            raise ValueError("S3 bucket name must not be empty.")
        self.bucket = cleaned_bucket

        cleaned_prefix = (prefix or "").lstrip("/")
        if cleaned_prefix and not cleaned_prefix.endswith("/"):
            cleaned_prefix += "/"
        self.prefix = cleaned_prefix

        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client: Any | None = client
        self._owns_client = client is None

    @property
    def store_name(self) -> str:
        return "s3"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for S3ArtifactStore when no injectable client is provided. "
                "Install it with `pip install boto3`."
            ) from exc

        kwargs: dict[str, Any] = {"region_name": self.region_name}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self._access_key_id is not None:
            kwargs["aws_access_key_id"] = self._access_key_id
        if self._secret_access_key is not None:
            kwargs["aws_secret_access_key"] = self._secret_access_key

        self._client = boto3.client("s3", **kwargs)
        self._owns_client = True
        return self._client

    async def _call_client(self, method_name: str, **kwargs: Any) -> Any:
        client = self._get_client()
        fn = getattr(client, method_name)
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        ):
            return await fn(**kwargs)
        result = await asyncio.to_thread(fn, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _blob_key(self, artifact_id: str) -> str:
        safe_id = _validate_artifact_id(artifact_id)
        return f"{self.prefix}{safe_id}.blob"

    def _meta_key(self, artifact_id: str) -> str:
        safe_id = _validate_artifact_id(artifact_id)
        return f"{self.prefix}{safe_id}.meta.json"

    async def save(
        self,
        content: str,
        *,
        session_id: str,
        turn_id: str,
        tenant_id: str = "default",
        tool_name: str = "",
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        blob_key = self._blob_key(artifact_id)
        meta_key = self._meta_key(artifact_id)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            session_id=session_id,
            turn_id=turn_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            content_type=content_type,
            char_count=len(content),
            preview=_build_preview(content),
            storage_uri=f"s3://{self.bucket}/{blob_key}",
            created_at_ms=int(time.time() * 1000),
            metadata=dict(metadata or {}),
        )
        await self._call_client(
            "put_object",
            Bucket=self.bucket,
            Key=blob_key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )
        await self._call_client(
            "put_object",
            Bucket=self.bucket,
            Key=meta_key,
            Body=record.model_dump_json().encode("utf-8"),
            ContentType="application/json",
        )
        return record

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        meta_key = self._meta_key(artifact_id)
        try:
            resp = await self._call_client("get_object", Bucket=self.bucket, Key=meta_key)
        except Exception as exc:
            if _is_s3_not_found(exc):
                return None
            raise
        raw_meta = await _extract_s3_body_text(resp)
        return ArtifactRecord.model_validate_json(raw_meta)

    async def read_slice(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4000,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> tuple[ArtifactRecord, str, bool]:
        record = await self.get_record(artifact_id)
        if record is None:
            raise KeyError(f"Artifact '{artifact_id}' not found.")
        self._enforce_tenant_acl(record, tenant_id=tenant_id, is_admin=is_admin)

        blob_key = self._blob_key(artifact_id)
        try:
            resp = await self._call_client("get_object", Bucket=self.bucket, Key=blob_key)
        except Exception as exc:
            if _is_s3_not_found(exc):
                raise KeyError(f"Artifact '{artifact_id}' not found.") from exc
            raise

        content = await _extract_s3_body_text(resp)
        chunk, has_more = self._slice_content(content, offset=offset, limit=limit)
        return record, chunk, has_more

    async def list_for_session(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        is_admin: bool = False,
    ) -> list[ArtifactRecord]:
        meta_keys: list[str] = []
        continuation_token: str | None = None

        while True:
            list_kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": self.prefix,
            }
            if continuation_token is not None:
                list_kwargs["ContinuationToken"] = continuation_token
            resp = await self._call_client("list_objects_v2", **list_kwargs)
            contents = resp.get("Contents") or [] if isinstance(resp, Mapping) else []
            for entry in contents:
                key = str(entry.get("Key", "")) if isinstance(entry, Mapping) else ""
                if key.endswith(".meta.json"):
                    meta_keys.append(key)

            is_truncated = bool(resp.get("IsTruncated", False)) if isinstance(resp, Mapping) else False
            next_token = resp.get("NextContinuationToken") if isinstance(resp, Mapping) else None
            if is_truncated and next_token:
                continuation_token = str(next_token)
            else:
                break

        items: list[ArtifactRecord] = []
        for meta_key in meta_keys:
            try:
                obj = await self._call_client("get_object", Bucket=self.bucket, Key=meta_key)
                raw_meta = await _extract_s3_body_text(obj)
                rec = ArtifactRecord.model_validate_json(raw_meta)
            except Exception as exc:
                if _is_s3_not_found(exc) or isinstance(exc, ValueError):
                    continue
                raise
            if rec.session_id != session_id:
                continue
            if tenant_id is not None and not is_admin and rec.tenant_id != tenant_id:
                continue
            items.append(rec)

        items.sort(key=lambda r: (r.created_at_ms, r.artifact_id))
        return items

    async def delete(self, artifact_id: str) -> bool:
        blob_key = self._blob_key(artifact_id)
        meta_key = self._meta_key(artifact_id)
        existing = await self.get_record(artifact_id)
        if existing is None:
            return False

        try:
            await self._call_client("delete_object", Bucket=self.bucket, Key=blob_key)
        except Exception as exc:
            if not _is_s3_not_found(exc):
                raise

        try:
            await self._call_client("delete_object", Bucket=self.bucket, Key=meta_key)
        except Exception as exc:
            if not _is_s3_not_found(exc):
                raise
        return True

    async def aclose(self) -> None:
        if not self._owns_client or self._client is None:
            return
        client = self._client
        self._client = None
        if hasattr(client, "aclose") and callable(client.aclose):
            res = client.aclose()
            if inspect.isawaitable(res):
                await res
        elif hasattr(client, "close") and callable(client.close):
            await asyncio.to_thread(client.close)


def create_artifact_store(
    mode: str = "auto",
    *,
    base_dir: str | Path | None = None,
    s3_bucket: str | None = None,
    s3_endpoint_url: str | None = None,
    s3_prefix: str = "pyharness/artifacts/",
    s3_client: Any | None = None,
) -> BaseArtifactStore:
    """Factory selecting S3, File, or InMemory artifact storage backend."""
    normalized = (mode or "auto").strip().lower()
    if normalized == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when artifact store mode is 's3'.")
        return S3ArtifactStore(
            bucket=s3_bucket,
            prefix=s3_prefix,
            endpoint_url=s3_endpoint_url,
            client=s3_client,
        )
    if normalized == "file":
        return FileArtifactStore(base_dir or "./data/artifacts")
    if normalized == "memory":
        return InMemoryArtifactStore()
    if normalized == "auto":
        if s3_bucket:
            return S3ArtifactStore(
                bucket=s3_bucket,
                prefix=s3_prefix,
                endpoint_url=s3_endpoint_url,
                client=s3_client,
            )
        if base_dir is not None:
            return FileArtifactStore(base_dir)
        return InMemoryArtifactStore()
    raise ValueError(
        f"Unsupported artifact store mode {mode!r}; expected 'auto', 'memory', 'file', or 's3'."
    )


class ToolOutputOffloadHook(HarnessHook):
    """
    Composable post-tool hook that offloads oversized tool outputs into `BaseArtifactStore`
    and replaces the inline payload with a compact head/tail preview plus `read_artifact` handle.
    """

    SKIP_TOOLS: frozenset[str] = frozenset({"read_artifact", "list_artifacts"})

    def __init__(
        self,
        store: BaseArtifactStore,
        *,
        max_inline_chars: int = 4000,
        head_chars: int = 1200,
        tail_chars: int = 600,
        offload_errors: bool = False,
        on_artifact_created: Callable[[str, str, ArtifactRecord], Awaitable[None]] | None = None,
    ) -> None:
        self.store = store
        self.max_inline_chars = max_inline_chars
        self.head_chars = max(0, head_chars)
        self.tail_chars = max(0, tail_chars)
        self.offload_errors = offload_errors
        self.on_artifact_created = on_artifact_created

    def _format_offloaded_summary(self, content: str, record: ArtifactRecord) -> str:
        total = len(content)
        eff_head = self.head_chars
        eff_tail = self.tail_chars
        if eff_head + eff_tail >= total:
            eff_head = min(self.head_chars, max(1, self.max_inline_chars // 2))
            eff_tail = min(self.tail_chars, max(0, self.max_inline_chars // 4))
            if eff_head + eff_tail >= total:
                eff_head = max(0, total // 2)
                eff_tail = max(0, total - eff_head - 1)

        truncated = max(0, total - eff_head - eff_tail)
        head_part = content[:eff_head]
        tail_part = content[total - eff_tail :] if eff_tail > 0 else ""

        if tail_part:
            preview_body = f"{head_part}\n... [truncated {truncated} chars] ...\n{tail_part}"
        else:
            preview_body = f"{head_part}\n... [truncated {truncated} chars] ..."

        handle = (
            f"[Artifact Offloaded: {record.artifact_id} | {record.char_count} chars stored in "
            f'{self.store.store_name}. Call read_artifact(artifact_id="{record.artifact_id}", '
            f"offset=0, limit=4000) to inspect full data.]"
        )
        return f"{preview_body}\n\n{handle}"

    async def after_tool(
        self,
        session_id: str,
        turn_id: str,
        call: ToolCall,
        result: ToolResult,
    ) -> ToolResult:
        if call.name in self.SKIP_TOOLS:
            return result
        if result.is_error and not self.offload_errors:
            return result
        if len(result.content) <= self.max_inline_chars:
            return result

        principal = current_principal.get()
        tenant_id = principal.tenant_id if principal is not None else "default"

        rec = await self.store.save(
            result.content,
            session_id=session_id,
            turn_id=turn_id,
            tenant_id=tenant_id,
            tool_name=call.name,
            metadata={
                "tool_call_id": call.id,
                "is_error": result.is_error,
            },
        )
        if self.on_artifact_created is not None:
            await self.on_artifact_created(session_id, turn_id, rec)

        summary_text = self._format_offloaded_summary(result.content, rec)
        return result.model_copy(
            update={
                "content": summary_text,
                "metadata": {
                    **result.metadata,
                    "artifact_id": rec.artifact_id,
                    "offloaded_chars": rec.char_count,
                },
            }
        )


def register_artifact_tools(
    registry: Any,
    store: BaseArtifactStore,
    *,
    session_id_getter: Callable[[], str | None] | None = None,
) -> None:
    """Register tenant-scoped `read_artifact` and `list_artifacts` inspection tools."""

    async def _read_artifact(
        artifact_id: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> str:
        principal = current_principal.get()
        tenant_id = principal.tenant_id if principal is not None else "default"
        is_admin = principal.is_admin if principal is not None else False
        rec, chunk, has_more = await store.read_slice(
            artifact_id,
            offset=offset,
            limit=limit,
            tenant_id=tenant_id,
            is_admin=is_admin,
        )
        return json.dumps(
            {
                "artifact_id": rec.artifact_id,
                "tool_name": rec.tool_name,
                "offset": max(0, int(offset)),
                "limit": max(0, int(limit)),
                "total_chars": rec.char_count,
                "has_more": has_more,
                "content": chunk,
            },
            ensure_ascii=False,
        )

    async def _list_artifacts(session_id: str = "") -> str:
        target_session = (session_id or "").strip()
        if not target_session and session_id_getter is not None:
            target_session = (session_id_getter() or "").strip()
        if not target_session:
            return json.dumps([], ensure_ascii=False)

        principal = current_principal.get()
        tenant_id = principal.tenant_id if principal is not None else "default"
        is_admin = principal.is_admin if principal is not None else False
        records = await store.list_for_session(
            target_session,
            tenant_id=tenant_id,
            is_admin=is_admin,
        )
        summaries = [
            {
                "artifact_id": rec.artifact_id,
                "session_id": rec.session_id,
                "turn_id": rec.turn_id,
                "tenant_id": rec.tenant_id,
                "tool_name": rec.tool_name,
                "content_type": rec.content_type,
                "char_count": rec.char_count,
                "preview": rec.preview,
                "storage_uri": rec.storage_uri,
                "created_at_ms": rec.created_at_ms,
            }
            for rec in records
        ]
        return json.dumps(summaries, ensure_ascii=False)

    registry.register(
        ToolSpec(
            name="read_artifact",
            description=(
                "Read a character slice from a previously offloaded tool output artifact "
                "using pagination (offset and limit)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "Identifier of the offloaded artifact (e.g. art_a1b2c3d4e5f6).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based starting character offset (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of characters to return (default 4000).",
                    },
                },
                "required": ["artifact_id"],
            },
            risk_level=RiskLevel.LOW,
        ),
        _read_artifact,
    )

    registry.register(
        ToolSpec(
            name="list_artifacts",
            description=(
                "List metadata summaries of offloaded artifacts available for the current "
                "or specified session within the caller's tenant."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID; defaults to the active session when omitted.",
                    },
                },
                "required": [],
            },
            risk_level=RiskLevel.LOW,
        ),
        _list_artifacts,
    )


__all__ = [
    "ArtifactAuthorizationError",
    "ArtifactRecord",
    "BaseArtifactStore",
    "FileArtifactStore",
    "InMemoryArtifactStore",
    "S3ArtifactStore",
    "ToolOutputOffloadHook",
    "create_artifact_store",
    "register_artifact_tools",
]
