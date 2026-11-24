# Milestone 3 源码精讲与上海一面深挖手记（企业级 Hybrid RAG 引擎）

## 一、 这一阶段的 5 大核心技术点（直接对应你简历上的 RAG 调优亮点！）

1. **表格感知 + 父子层级切片（`pyharness/rag/chunker.py`）**
   * **面试痛点**：普通 `RecursiveCharacterTextSplitter(chunk_size=500)` 经常把 Markdown/PDF 表格的表头和第 3 行数据硬生生劈成两半，导致检索出来的数字不知道属于哪一列！
   * **我们的解法**：`_split_blocks_preserving_tables()` 将连续表格行锁定为一个完整不可分割的块；同时采用 **Parent-Child（父子块）策略**——用短小的 `ChildChunk`（高信噪比）参与 BM25/向量检索，命中后再把所属的完整 `ParentChunk`（带章节路径 `section_path`）喂给大模型，彻底兼顾“召回准度”和“上下文完整度”！
2. **中英文混合分词 + 手写 Okapi BM25（`pyharness/rag/retriever.py`）**
   * 工业与金融文档充满专属代号（如 `SF6`、`0.48 MPa`、`INC-2026-01`），纯稠密向量经常把 `0.48 MPa` 和 `0.58 MPa` 算成相似度 0.98 导致召回错位！BM25 稀疏精确词法匹配天然补足了稠密向量对专有名词和型号编码的盲区。
3. **为什么用 RRF（Reciprocal Rank Fusion）而不是 `0.5 * dense + 0.5 * bm25`？**
   * **高频面试必杀题！** 稠密余弦相似度在 $[0, 1]$，而 BM25 分数无上界（可能 0~30），且不同查询下分布极不稳定，直接加权求和极易被某一路分数碾压。
   * **RRF 公式**：$\text{RRF}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}$（经验常数 $k=60$）只看**排名名次**，完全与分数尺度解耦，零调参且对离群高分极度鲁棒！
4. **双塔召回（Bi-Encoder）vs 交叉编码精排（Cross-Encoder Reranker）**
   * 召回阶段用 Bi-Encoder（如 BGE-M3）离线把百万文档算好向量，在线毫秒级捞出 Top-20，但 Query 与文档之间没有做 Token 级细粒度交互；
   * 精排阶段把 `(Query, Doc_i)` 拼在一起送进 Cross-Encoder（如 `BAAI/bge-reranker-v2-m3`）做全交叉 Attention，精准挑出 Top-3，速度与精度完美兼得！
5. **Query 理解层（`query_rewriter.py`）**
   * 多轮对话里用户问“它的闭锁阈值是多少？”，直接拿“它”去搜向量库必然召回垃圾！先通过 Query 改写做指代消解和多跳问题拆解，分别检索再统一 RRF 融合，大幅拉高复杂问题召回率。
