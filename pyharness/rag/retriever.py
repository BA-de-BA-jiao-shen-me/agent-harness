from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

from pyharness.rag.chunker import ChildChunk, HierarchicalChunker, ParentChunk
from pyharness.rag.query_rewriter import QueryRewriter
from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import Message, RiskLevel, ToolSpec


def tokenize_mixed(text: str) -> list[str]:
    """
    Zero-dependency bilingual (Chinese + English/Code) tokenizer:
    - Extracts lowercase alphanumeric/identifier tokens for English & technical codes (e.g. `sf6`, `bge-m3`, `inc-01`).
    - Extracts Chinese unigrams + bigrams so domain phrases (`断路器`, `温升异常`, `精排`) match accurately.
    """
    text_lower = text.lower()
    tokens: list[str] = []
    # English / numeric / hyphenated technical tokens
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]", text_lower))
    # Chinese character spans -> unigrams + bigrams
    for span in re.findall(r"[\u4e00-\u9fff]+", text_lower):
        for ch in span:
            tokens.append(ch)
        for i in range(len(span) - 1):
            tokens.append(span[i : i + 2])
    return tokens


class BM25Index:
    """Standard Okapi BM25 sparse lexical retriever (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add_documents(self, items: list[tuple[str, str]]) -> None:
        for doc_id, text in items:
            toks = tokenize_mixed(text)
            tf = Counter(toks)
            self._doc_ids.append(doc_id)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(toks))
            for term in tf:
                self._df[term] += 1
        self._avgdl = sum(self._doc_lens) / max(1, len(self._doc_lens))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        n_docs = len(self._doc_ids)
        if n_docs == 0:
            return []
        q_tokens = tokenize_mixed(query)
        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self._doc_ids):
            dl = self._doc_lens[idx]
            tf_map = self._doc_freqs[idx]
            score = 0.0
            for term in q_tokens:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self._avgdl)))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingSemanticEmbedder(BaseEmbedder):
    """
    Lightweight deterministic feature-hashing embedder (character 2/3-gram + token projection)
    for zero-dependency local/CI testing. Swap to `OpenAICompatibleEmbedder` (BGE-M3) in production.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = tokenize_mixed(text)
            # Add char trigrams for fuzzy semantic overlap
            clean = re.sub(r"\s+", "", text.lower())
            trigrams = [clean[i : i + 3] for i in range(max(0, len(clean) - 2))]
            for feat in toks + trigrams:
                h = hash(feat)
                idx = abs(h) % self.dim
                sign = 1.0 if (h & 1) == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Production embedder supporting BGE-M3 / vLLM / DashScope / OpenAI `/v1/embeddings`."""

    def __init__(self, api_key: str, base_url: str, model: str = "BAAI/bge-m3") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


class DenseVectorIndex:
    """In-memory cosine similarity vector store (easily swappable with Qdrant/Milvus)."""

    def __init__(self, embedder: BaseEmbedder | None = None) -> None:
        self.embedder = embedder or HashingSemanticEmbedder()
        self._doc_ids: list[str] = []
        self._vectors: list[list[float]] = []

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    async def add_documents(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        vecs = await self.embedder.embed([txt for _, txt in items])
        for (doc_id, _), vec in zip(items, vecs):
            self._doc_ids.append(doc_id)
            self._vectors.append(vec)

    async def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._doc_ids:
            return []
        q_vec = (await self.embedder.embed([query]))[0]
        scored = [
            (doc_id, self._cosine(q_vec, vec))
            for doc_id, vec in zip(self._doc_ids, self._vectors)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        raise NotImplementedError


class LexicalSemanticReranker(BaseReranker):
    """
    Lightweight Cross-Encoder-style pair scorer for offline/CI execution:
    combines exact query bigram coverage, phrase proximity, and section heading match.
    """

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        q_toks = set(tokenize_mixed(query))
        q_bigrams = {t for t in q_toks if len(t) >= 2}
        scored: list[tuple[str, float]] = []
        for cid, text in candidates:
            c_toks = set(tokenize_mixed(text))
            overlap_all = len(q_toks & c_toks) / max(1, len(q_toks))
            overlap_bi = len(q_bigrams & c_toks) / max(1, len(q_bigrams)) if q_bigrams else overlap_all
            exact_bonus = 0.35 if query.lower().strip() in text.lower() else 0.0
            score = 0.45 * overlap_all + 0.55 * overlap_bi + exact_bonus
            scored.append((cid, round(score, 5)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class HTTPCrossEncoderReranker(BaseReranker):
    """Production Cross-Encoder Reranker client (compatible with BGE-Reranker / Jina / SiliconFlow)."""

    def __init__(self, endpoint: str, api_key: str, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [txt for _, txt in candidates],
                    "top_n": top_k,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        out: list[tuple[str, float]] = []
        for item in results:
            idx = int(item["index"])
            out.append((candidates[idx][0], float(item["relevance_score"])))
        out.sort(key=lambda x: -x[1])
        return out[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Scale-invariant Reciprocal Rank Fusion:
        RRF(d) = sum_i 1 / (k + rank_i(d))
    Eliminates fragile manual score normalization between dense cosine [0,1] and unbounded BM25 scores.
    """
    fused: dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, (doc_id, _score) in enumerate(rlist, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    out = list(fused.items())
    out.sort(key=lambda x: -x[1])
    return out


@dataclass
class RetrievalHit:
    citation_id: int
    chunk_id: str
    parent_id: str
    source: str
    section_path: str
    score: float
    child_snippet: str
    parent_context: str


class HybridRAGPipeline:
    """
    End-to-end Enterprise Hybrid RAG Pipeline:
    Ingest (Hierarchical Table-Aware Parent-Child) -> Query Rewrite -> BM25 + Dense -> RRF -> Cross-Encoder Rerank -> Parent Context & Citations.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        rewriter: QueryRewriter | None = None,
        chunker: HierarchicalChunker | None = None,
    ) -> None:
        self.chunker = chunker or HierarchicalChunker()
        self.rewriter = rewriter or QueryRewriter()
        self.bm25 = BM25Index()
        self.dense = DenseVectorIndex(embedder=embedder)
        self.reranker = reranker or LexicalSemanticReranker()
        self._parents: dict[str, ParentChunk] = {}
        self._children: dict[str, ChildChunk] = {}

    async def ingest_markdown(self, text: str, source: str, doc_id: str | None = None) -> dict[str, int]:
        parents, children = self.chunker.chunk_document(text=text, source=source, doc_id=doc_id)
        for p in parents:
            self._parents[p.parent_id] = p
        items: list[tuple[str, str]] = []
        for c in children:
            self._children[c.chunk_id] = c
            items.append((c.chunk_id, c.content))
        self.bm25.add_documents(items)
        await self.dense.add_documents(items)
        return {"parents_indexed": len(parents), "children_indexed": len(children)}

    async def search(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid_rerank",  # "dense" | "bm25" | "hybrid" | "hybrid_rerank"
        recent_history: list[Message] | None = None,
    ) -> list[RetrievalHit]:
        rq = await self.rewriter.rewrite(query, recent_history=recent_history)
        recall_k = max(top_k * 4, 12)

        ranked_lists: list[list[tuple[str, float]]] = []
        for sq in rq.sub_queries:
            if mode in ("bm25", "hybrid", "hybrid_rerank"):
                ranked_lists.append(self.bm25.search(sq, top_k=recall_k))
            if mode in ("dense", "hybrid", "hybrid_rerank"):
                ranked_lists.append(await self.dense.search(sq, top_k=recall_k))

        if len(ranked_lists) == 1:
            merged = ranked_lists[0]
        else:
            merged = reciprocal_rank_fusion(ranked_lists, k=60)

        if mode == "hybrid_rerank" and merged:
            cand_pairs = [(cid, self._children[cid].content) for cid, _ in merged[:recall_k] if cid in self._children]
            final_ranked = await self.reranker.rerank(rq.rewritten, cand_pairs, top_k=top_k)
        else:
            final_ranked = merged[:top_k]

        hits: list[RetrievalHit] = []
        for idx, (cid, score) in enumerate(final_ranked, start=1):
            child = self._children[cid]
            parent = self._parents[child.parent_id]
            hits.append(
                RetrievalHit(
                    citation_id=idx,
                    chunk_id=cid,
                    parent_id=child.parent_id,
                    source=child.source,
                    section_path=child.section_path,
                    score=round(score, 5),
                    child_snippet=child.content,
                    parent_context=parent.content,
                )
            )
        return hits

    async def search_as_tool(self, query: str, top_k: int = 3) -> str:
        hits = await self.search(query=query, top_k=top_k, mode="hybrid_rerank")
        payload = [
            {
                "citation": f"[{h.citation_id}]",
                "source": h.source,
                "section": h.section_path,
                "score": h.score,
                "content": h.parent_context,
            }
            for h in hits
        ]
        return json.dumps(payload, ensure_ascii=False)

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                name="rag_search",
                description="企业多源知识库混合检索工具（BM25 + 稠密向量 RRF 融合 + Cross-Encoder 精排，返回带 [1][2] 编号的父块上下文证据）",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "需要检索的业务问题或技术术语"},
                        "top_k": {"type": "integer", "description": "返回前 K 条最权威证据，默认 3"},
                    },
                    "required": ["query"],
                },
                risk_level=RiskLevel.LOW,
            ),
            self.search_as_tool,
        )
