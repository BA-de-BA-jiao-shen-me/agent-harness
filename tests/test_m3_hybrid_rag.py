from __future__ import annotations

from pyharness.rag import HierarchicalChunker, HybridRAGPipeline, reciprocal_rank_fusion
from pyharness.runtime import (
    AgentHarnessEngine,
    LLMResponse,
    ScriptedMockLLMClient,
    ToolCall,
    ToolRegistry,
    TurnStatus,
)

SAMPLE_SAFETY_DOC = """
# 高压配电设备安全隐患排查手册

## 1. SF6 高压断路器巡检规范
SF6 断路器运行期间应每 24 小时检查一次气体压力表读数。当气体压力低于 0.52 MPa 时触发一级低压告警，低于 0.48 MPa 时闭锁分合闸并立即上报调度中心。

### 1.1 关键阈值速查表
| 设备类型 | 正常压力范围 | 一级告警阈值 | 紧急闭锁阈值 |
| --- | --- | --- | --- |
| SF6 断路器 | 0.58 - 0.65 MPa | 0.52 MPa | 0.48 MPa |
| 气体监测探头 | 0 - 15 ppm | 25 ppm | 50 ppm |

## 2. 石化装置温升异常应急流程
当金山石化 A 区压力阀红外测温连续 10 分钟超过 85℃ 时，运维人员须在 15 分钟内启动二级降温预案，并同步校验上游泄压阀开度与热电偶零漂。
"""


def test_hierarchical_chunker_preserves_markdown_tables() -> None:
    chunker = HierarchicalChunker(child_max_chars=120, child_overlap=20)
    parents, children = chunker.chunk_document(SAMPLE_SAFETY_DOC, source="safety_manual.md", doc_id="doc1")

    assert len(parents) == 3
    table_chunks = [c for c in children if c.is_table]
    assert len(table_chunks) == 1
    # Entire table including header and both rows must remain intact
    assert "0.48 MPa" in table_chunks[0].content
    assert "50 ppm" in table_chunks[0].content
    assert "关键阈值速查表" in table_chunks[0].section_path


def test_rrf_merges_unnormalized_rankings() -> None:
    bm25_ranks = [("d2", 18.5), ("d1", 12.0), ("d3", 4.1)]
    dense_ranks = [("d1", 0.91), ("d4", 0.85), ("d2", 0.72)]
    fused = reciprocal_rank_fusion([bm25_ranks, dense_ranks], k=60)
    top_ids = [doc_id for doc_id, _ in fused[:2]]
    assert set(top_ids) == {"d1", "d2"}


async def test_hybrid_rag_pipeline_and_harness_integration() -> None:
    rag = HybridRAGPipeline()
    stats = await rag.ingest_markdown(SAMPLE_SAFETY_DOC, source="safety_manual.md", doc_id="manual_v1")
    assert stats["children_indexed"] >= 3

    hits = await rag.search("SF6 断路器紧急闭锁阈值是多少 MPa", top_k=2, mode="hybrid_rerank")
    assert len(hits) >= 1
    assert "0.48 MPa" in hits[0].parent_context
    assert hits[0].citation_id == 1

    # Wire RAG as a governed tool inside AgentHarnessEngine
    registry = ToolRegistry()
    rag.register_tools(registry)
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="r1", name="rag_search", arguments={"query": "压力阀温升超过多少度启动二级预案"})
                ]
            ),
            LLMResponse(content="超过 85℃ 且持续 10 分钟须在 15 分钟内启动二级降温预案 [1]。"),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=registry)
    res = await engine.run_turn("sess_rag", "压力阀温升超过多少度要启动降温预案？")
    assert res.status == TurnStatus.COMPLETED
    assert "85℃" in res.final_answer
    assert any("85℃" in m.content for m in res.messages if m.role == "tool")
