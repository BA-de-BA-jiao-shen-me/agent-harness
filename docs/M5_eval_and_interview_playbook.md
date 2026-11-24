# Milestone 5 评测闭环精讲与上海 Agent 面试总监级通关手册

> **读完这 5 份 `docs/M1~M5` 并亲手跑完 `pytest` 和 `eval/` 压测后，这个仓库就 100% 变成你自己的硬核代表作！**

---

## 一、 3 分钟面试高分自我项目阐述逐字稿（直接对标你推上 GitHub 的 `PyHarness`）

> “为了把我在实际业务中沉淀的多智能体和 RAG 架构系统化、开源化，我受 DeepSeek Harness（DSH）生产设计启发，从零自研了一套轻量级企业 Agent Harness 平台 **PyHarness**（Python + FastAPI + Vue 3 + Docker），重点解决大模型落地中的**确定性治理、高并发低延迟、多源 RAG 精度与数据评测闭环**四个核心痛点：
>
> 1. **在运行时治理层**，我参考 DSH 实现了基于 SQLite 的 **Event Journal 事件溯源日志**（支持按单调序号 `seq` 断线补发与 Trace 耗时瀑布统计）、基于 `asyncio.Event` 竞态的 **会话级秒断取消**、针对高危操作（写库 SQL / 外发告警）的 **Human-in-the-Loop 审批挂起与 Fail-Closed 超时兜底**，以及**防止劈裂 `tool_call/tool` 消息对的上下文增量摘要压缩**。
> 2. **在工具与安全层**，采用声明式 `SKILL.md` 按轮次动态路由注入技能，实现了兼容标准 **JSON-RPC 2.0 的 MCP Client/Server 工具桥**，并构建了带 SQL 注入/写操作拦截的只读沙箱与基于 **Python AST 静态语法树审计**的受限计算沙箱（彻底封死 `__subclasses__` 逃逸链）；配合 `asyncio.gather + Semaphore` 限流并发与 `Self-Correction` 连续错误熔断器，实测多工具并发将 P50 响应延迟降低了 **60%+**。
> 3. **在企业级 Hybrid RAG 层**，我实现了**表格不跨块劈裂的 Parent-Child 层级切片**、Query 多轮指代消解与多跳拆解、**中英文 Okapi BM25 + 向量检索 + RRF 融合 + Cross-Encoder 精排**，在离线评测集上实现了 **Top-1 / Top-3 召回率与 MRR 的显著提升**，并强制带编号引用溯源。
> 4. **在全栈工程与数据飞轮层**，后端基于 FastAPI 提供 **Streamable HTTP / SSE 实时推流**（配 Nginx `proxy_buffering off` 零缓冲），前端用 Vue 3 + TS 实现思考链折叠、审批弹窗和实时事件流；内置 `eval/` 评测套件可自动归因失败轨迹并导出 JSONL 供后续 Prompt 迭代或 LoRA/DPO 微调。”

---

## 二、 上海技术总监最爱追问的 10 个压轴难题 & 满分答法

1. **Q：你的 `PyHarness` 和直接用 `LangGraph` 相比，核心取舍是什么？**
   * *答*：LangGraph 强在声明式图编排生态；PyHarness 聚焦**轻量、零黑盒的运行时治理内核**（事件溯源、协程取消、审批挂起、信号量并发、Token 熔断总计不足 500 行），排查线上异步 Bug 极快，且随时可以把 LangGraph 节点作为单个工具挂载进来。
2. **Q：你们把 Event Journal 存在 SQLite，线上 QPS 上千怎么扛？**
   * *答*：开源单机 MVP 用 SQLite 实现零依赖一键启动；高并发生产环境按仓库接口抽象平滑切换为 **PostgreSQL（持久化审计事件表）+ Redis Streams / PubSub（实时 SSE 跨 Pod 广播与分布式审批回调）**，上层 Engine 代码一行不用改。
3. **Q：如果用户在审批弹窗一直不点按钮，后端协程一直挂着会内存泄漏吗？**
   * *答*：不会！`ApprovalBroker.wait_for_decision` 用 `asyncio.wait` 设了硬超时（如 120s），超时自动 `Fail-Closed` 拒绝并清理 `_pending` 字典；且若前端断开 SSE 连接，网关 `finally` 块也会立刻触发 `CancellationToken` 释放协程。
4. **Q：Python 沙箱只用 `ast` 静态检查，真上线放给外部用户够安全吗？**
   * *答*：不够！AST 白名单能防常规反射逃逸，但防不了 `10**10**10` 内存/CPU 死循环炸弹（DoS）。本项目的 AST 审计是第一道轻量防线，真实多租户生产环境我会再叠加 **独立容器/微虚拟机沙箱（如 Docker `--network=none --memory=256m --cpus=0.5` 或 gVisor / E2B）** 做操作系统级隔离！
5. **Q：Hybrid RAG 里为什么用 RRF 融合，而不用把 BM25 和余弦相似度归一化后加权？**
   * *答*：BM25 分数无上界且高度依赖文档长度和查询词频，Min-Max 归一化在单次查询候选集里遇到极端值就会失效；RRF 只用倒数排名 $\frac{1}{60+\text{rank}}$，天然尺度无关、无需调参且极为稳健。
6. **Q：上下文增量摘要压缩时，为什么一定要做 `while body[split_at].role == "tool"` 边界检查？**
   * *答*：OpenAI/DeepSeek/Qwen 聊天协议强制要求每条 `tool` 消息必须紧跟在声明该 `tool_call_id` 的 `assistant` 消息之后。如果不检查边界硬切，孤立的 `tool` 消息会导致下游 LLM API 直接报 `400 Bad Request` 崩溃。
7. **Q：你们如何评估一个 Multi-Agent 系统的真实质量，而不只是凭感觉看两句对话？**
   * *答*：分三层评估——① **检索层**：Top-K Recall、MRR、Ragas Context Recall/Faithfulness；② **过程轨迹层**（看 `eval/benchmark_harness.py`）：工具选择准确率、SQL Self-Correction 修复率、平均步数与并发延迟 P50/P95；③ **数据闭环**：把触发熔断或人工点踩的失败轨迹自动打根因标签导出 JSONL，反哺 Prompt 优化或 LoRA/DPO 微调！
8. **Q：大模型流式输出时，为什么 Nginx 默认配置会让前端看起来像“卡死 10 秒再瞬间刷屏”？**
   * *答*：Nginx 默认开启 `proxy_buffering on`，会把上游 FastAPI 吐出的零散 SSE 字节攒满缓冲区才往下游发；必须在 FastAPI 加响应头 `X-Accel-Buffering: no`，并在 Nginx `location /api/` 里配 `proxy_buffering off; proxy_cache off;`。
9. **Q：如何控制 Agent 系统的单次请求 Token 成本？**
   * *答*：四管齐下——① `SKILL.md` 按需动态加载（不把全部 SOP 塞进 System Prompt）；② `ContextCompressor` 增量压缩历史工具大段输出；③ `SQLSandbox` 强制加 `LIMIT 100` 防止查出几万行撑爆窗口；④ `TokenBudgetGuard` 硬预算熔断 + 简单路由/改写走小模型（如 Qwen-7B/DeepSeek-Chat），复杂规划走强模型。
10. **Q：如何把你在工作中的 `Qwen3-VL-32B` 微调经验和这个 Agent Harness 结合？**
    * *答*：本平台的 `eval/benchmark_harness.py` 已经打通了 Badcase 轨迹自动归因导出 JSONL；在实际业务中，把线上积累的高质量成功轨迹和失败轨迹构造成 SFT/DPO 样本，直接用 LoRA 微调垂类 Qwen 模型，再通过 `OpenAICompatibleLLMClient` 无缝挂载回本 Harness，就能形成完整的 **“Agent 运行 $\rightarrow$ 评测归因 $\rightarrow$ 微调升级 $\rightarrow$ 线上提效”数据飞轮**！
