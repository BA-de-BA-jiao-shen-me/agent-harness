from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from pyharness.rag.chunker import ChildChunk, ParentChunk


class BaseVectorStore(ABC):
    """Pluggable Dense Vector Store Seam (`DenseVectorIndex` | `QdrantVectorStore`)."""

    @property
    @abstractmethod
    def store_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[ChildChunk],
        parent_map: dict[str, ParentChunk],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    async def load_persisted_chunks(self) -> tuple[dict[str, ParentChunk], dict[str, ChildChunk]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class QdrantVectorStore(BaseVectorStore):
    """
    Production Persistent Vector Store backed by Qdrant v1 REST API (`http://qdrant:6333`).

    Highlights:
    - Deterministic UUIDv5 point IDs (`chunk_id` -> UUID) for 100% idempotent re-ingestion.
    - Persists both dense vectors and Parent/Child chunk payloads so knowledge survives API restarts.
    - Supports enterprise metadata/ACL filtering (`filter.must`) at search time.
    """

    def __init__(
        self,
        embedder: Any,
        url: str = "http://localhost:6333",
        collection_name: str = "pyharness_chunks",
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.embedder = embedder
        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["api-key"] = api_key
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._collection_ready = False

    @property
    def store_name(self) -> str:
        return "qdrant"

    @staticmethod
    def point_id_for_chunk(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pyharness-chunk:{chunk_id}"))

    async def _ensure_collection(self, vector_dim: int | None = None) -> bool:
        if self._collection_ready:
            return True
        resp = await self._client.get(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers,
        )
        if resp.status_code == 200:
            self._collection_ready = True
            return True
        if resp.status_code != 404:
            resp.raise_for_status()
        if vector_dim is None:
            sample = await self.embedder.embed(["dimension_probe"])
            vector_dim = len(sample[0])
        create_resp = await self._client.put(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers,
            json={"vectors": {"size": vector_dim, "distance": "Cosine"}},
        )
        create_resp.raise_for_status()
        self._collection_ready = True
        return True

    async def delete_by_doc_id(self, doc_id: str) -> None:
        resp = await self._client.get(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers,
        )
        if resp.status_code == 404:
            return
        resp.raise_for_status()
        del_resp = await self._client.post(
            f"{self.url}/collections/{self.collection_name}/points/delete",
            params={"wait": "true"},
            headers=self._headers,
            json={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )
        del_resp.raise_for_status()

    async def upsert_chunks(
        self,
        chunks: list[ChildChunk],
        parent_map: dict[str, ParentChunk],
    ) -> None:
        if not chunks:
            return
        vectors = await self.embedder.embed([c.content for c in chunks])
        await self._ensure_collection(vector_dim=len(vectors[0]))

        points = []
        for chunk, vec in zip(chunks, vectors):
            parent = parent_map.get(chunk.parent_id)
            points.append(
                {
                    "id": self.point_id_for_chunk(chunk.chunk_id),
                    "vector": vec,
                    "payload": {
                        "chunk_id": chunk.chunk_id,
                        "parent_id": chunk.parent_id,
                        "doc_id": chunk.doc_id,
                        "source": chunk.source,
                        "section_path": chunk.section_path,
                        "content": chunk.content,
                        "is_table": chunk.is_table,
                        "metadata": chunk.metadata,
                        "parent_context": parent.content if parent else chunk.content,
                    },
                }
            )

        up_resp = await self._client.put(
            f"{self.url}/collections/{self.collection_name}/points",
            params={"wait": "true"},
            headers=self._headers,
            json={"points": points},
        )
        up_resp.raise_for_status()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        check = await self._client.get(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers,
        )
        if check.status_code == 404:
            return []
        check.raise_for_status()

        q_vec = (await self.embedder.embed([query]))[0]
        body: dict[str, Any] = {
            "vector": q_vec,
            "limit": top_k,
            "with_payload": True,
        }
        if metadata_filter:
            must_clauses: list[dict[str, Any]] = []
            for k, v in metadata_filter.items():
                if k == "_acl_tenant":
                    must_clauses.append(
                        {
                            "should": [
                                {"key": "metadata.tenant_id", "match": {"value": v}},
                                {"key": "metadata.visibility", "match": {"value": "public"}},
                            ]
                        }
                    )
                    continue
                key_path = k if k in {"doc_id", "source"} else f"metadata.{k}"
                must_clauses.append({"key": key_path, "match": {"value": v}})
            body["filter"] = {"must": must_clauses}

        resp = await self._client.post(
            f"{self.url}/collections/{self.collection_name}/points/search",
            headers=self._headers,
            json=body,
        )
        resp.raise_for_status()
        results = resp.json().get("result", [])
        return [
            (str(item["payload"]["chunk_id"]), float(item["score"]))
            for item in results
            if item.get("payload") and "chunk_id" in item["payload"]
        ]

    async def load_persisted_chunks(self) -> tuple[dict[str, ParentChunk], dict[str, ChildChunk]]:
        check = await self._client.get(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers,
        )
        if check.status_code == 404:
            return {}, {}
        check.raise_for_status()

        parents: dict[str, ParentChunk] = {}
        children: dict[str, ChildChunk] = {}
        next_offset: Any = None

        while True:
            req_body: dict[str, Any] = {
                "limit": 256,
                "with_payload": True,
                "with_vector": False,
            }
            if next_offset is not None:
                req_body["offset"] = next_offset
            resp = await self._client.post(
                f"{self.url}/collections/{self.collection_name}/points/scroll",
                headers=self._headers,
                json=req_body,
            )
            resp.raise_for_status()
            res_obj = resp.json().get("result", {})
            points = res_obj.get("points", [])
            for pt in points:
                pl = pt.get("payload") or {}
                cid = pl.get("chunk_id")
                pid = pl.get("parent_id")
                if not cid or not pid:
                    continue
                children[cid] = ChildChunk(
                    chunk_id=cid,
                    parent_id=pid,
                    doc_id=pl.get("doc_id", ""),
                    source=pl.get("source", ""),
                    section_path=pl.get("section_path", ""),
                    content=pl.get("content", ""),
                    is_table=bool(pl.get("is_table", False)),
                    metadata=dict(pl.get("metadata") or {}),
                )
                if pid not in parents:
                    parents[pid] = ParentChunk(
                        parent_id=pid,
                        doc_id=pl.get("doc_id", ""),
                        source=pl.get("source", ""),
                        section_path=pl.get("section_path", ""),
                        content=pl.get("parent_context", pl.get("content", "")),
                    )
            next_offset = res_obj.get("next_page_offset")
            if next_offset is None or not points:
                break

        return parents, children

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
