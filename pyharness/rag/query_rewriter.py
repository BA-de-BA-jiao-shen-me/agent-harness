from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pyharness.runtime.llm_client import BaseLLMClient
from pyharness.runtime.schemas import Message


@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    sub_queries: list[str]


class QueryRewriter:
    """
    Multi-turn Query Understanding Layer:
    - Coreference resolution (resolves '它/该设备/上述问题' using recent dialogue context).
    - Multi-hop decomposition (splits compound questions connected by '以及/并且/分别' into sub-queries).
    - Supports both live LLM rewriting and deterministic rule-based offline fallback.
    """

    def __init__(self, llm: BaseLLMClient | None = None) -> None:
        self.llm = llm

    async def rewrite(
        self,
        query: str,
        recent_history: list[Message] | None = None,
    ) -> RewrittenQuery:
        if self.llm is not None:
            hist_text = "\n".join(
                f"{m.role}: {m.content[:150]}" for m in (recent_history or [])[-4:]
            )
            prompt = (
                "请对用户检索问题进行多轮指代消解与多跳问题拆解，严格输出 JSON："
                '{"rewritten": "...", "sub_queries": ["...", "..."]}\n'
                f"历史上下文：\n{hist_text}\n当前问题：{query}"
            )
            try:
                resp = await self.llm.complete([Message(role="user", content=prompt)])
                match = re.search(r"\{.*\}", resp.content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    rewritten = str(parsed.get("rewritten") or query).strip()
                    subs = [str(s).strip() for s in (parsed.get("sub_queries") or [rewritten]) if str(s).strip()]
                    return RewrittenQuery(original=query, rewritten=rewritten, sub_queries=subs or [rewritten])
            except Exception:
                pass

        # Deterministic fallback: coreference attachment + compound clause split
        rewritten = query.strip()
        if recent_history and any(p in rewritten for p in ("它", "该设备", "这个问题", "上面")):
            last_user = next((m.content for m in reversed(recent_history) if m.role == "user"), "")
            if last_user:
                rewritten = f"{last_user[:40]} {rewritten}"

        raw_parts = re.split(r"[；;]|以及|并且|同时", rewritten)
        sub_queries = [p.strip(" ，,？?。") for p in raw_parts if len(p.strip(" ，,？?。")) >= 4]
        if not sub_queries:
            sub_queries = [rewritten]
        return RewrittenQuery(original=query, rewritten=rewritten, sub_queries=sub_queries)
