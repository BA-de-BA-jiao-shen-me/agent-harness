# Milestone 1 源码精讲与上海一面深挖手记（Agent Harness 运行时内核）

> **学习目标**：拒绝“代码是 AI 帮我写的，面试一问就露馅”！请逐文件对照源码阅读本手记，确保你能**脱稿画出架构图、讲清每一项工程权衡，并亲手完成文末 2 道小改造练习**。

---

## 一、 架构全景：为什么我们要自研 Agent Harness，而不是直接套壳 LangChain？

你在上海面试 **Agent 工程师 / 大模型落地** 时，80% 的技术主管都会先问你一个灵魂问题：
> **“市面上现成的 LangChain / Dify / CrewAI 这么多，你们为什么还要自研或深度定制 Agent Harness？”**

学完 Milestone 1 的 5 个文件，你现在可以给出一个让面试官眼前一亮的**生产级回答**：
1. **确定性治理（Harness Engineering）**：大模型本身是非确定性的（会输出坏 JSON、死循环调同一个失败工具、误删生产数据），必须用确定性的状态机外壳把不确定性关进笼子里（`max_steps`、`max_consecutive_errors`、`TokenBudgetGuard`、`ApprovalBroker`）。
2. **可中断与可观测（参考 DeepSeek Harness 设计）**：很多通用编排框架在异步流式中断（用户点“停止生成”瞬间杀掉正在跑的慢工具）和事件级回放（Event Sourcing）上抽象太黑盒，线上排错困难。我们在 `event_journal.py` 和 `governance.py` 中仅用 ~300 行清晰代码就实现了 **SQLite 持久化事件流 + 实时 SSE 订阅广播 + 协程级秒断取消**。
3. **无厂商 SDK 锁定**：`llm_client.py` 直接基于 `httpx.AsyncClient` 对接标准 OpenAI Chat Completions 协议，DeepSeek、Qwen DashScope、本地 vLLM/SGLang 零改动切换。

---

## 二、 逐文件源码核心拆解（对照代码精读 20 分钟彻底吃透）

### 1. `pyharness/runtime/event_journal.py`（对应 DSH `event-journal.mjs`）
* **核心思想：事件溯源（Event Sourcing）**。
  * Agent 执行的每一步不是“打印一行 `print` 日志就算了”，而是生成带严格单调递增序号 `seq` 的 `HarnessEvent`，原子写入 SQLite，同时通过 `asyncio.Queue` 广播给所有在线订阅者（Milestone 4 的 FastAPI SSE 接口直接 `subscribe()` 就能把思考链、工具调用和审批卡片推给前端！）。
  * **断线重连怎么实现？** 前端断网重连时把上次收到的 `last_seq` 发过来，后端调用 `replay_session(session_id, after_seq=last_seq)` 就能无遗漏补发漏掉的事件！
  * **为什么要加 `asyncio.Lock()`？** 因为工具是并发执行的（`asyncio.gather`），如果不加锁，多个并发工具同时分配 `seq` 会导致序号冲突乱序。

### 2. `pyharness/runtime/governance.py`（对应 DSH `session-cancellation` + `approval-broker`）
* **`CancellationToken`（协作式取消）**：
  * 底层是一个 `asyncio.Event`。在 `engine.py` 的 `_await_with_cancellation()` 中，我们用 `asyncio.wait({work_task, cancel_task}, return_when=FIRST_COMPLETED)` 让**任何耗时操作（调 LLM、跑工具、等人工审批）与取消信号赛跑**！
  * 一旦用户点停止，`cancel_task` 毫秒级获胜，立即 `work_task.cancel()` 终止协程并落盘 `TURN_CANCELLED`，彻底避免后台僵尸协程继续空烧 Token！
* **`ApprovalBroker`（Human-in-the-Loop 高危审批）**：
  * 当工具声明 `risk_level=RiskLevel.HIGH`（如写库 SQL、发外部请求），引擎调用 `create_request()` 生成一个挂起的 `asyncio.Future`，向事件流发出 `APPROVAL_REQUIRED`，然后原地异步挂起（**不阻塞事件循环，其他用户的会话照常并发运行！**）。
  * 前端用户点“批准/拒绝”调后端接口触发 `broker.resolve(approval_id, approved=...)`，`Future` 立刻被唤醒恢复执行；**如果超时没人理，默认 Fail-Closed（安全拒绝），绝不自动放行高危操作！**
* **`ContextCompressor`（历史状态增量摘要压缩）**：
  * **一个极其细节的生产防坑点（看 `governance.py` 第 198 行！）**：
    ```python
    while split_at > 0 and body[split_at].role == "tool":
        split_at -= 1
    ```
    **为什么要写这行 `while`？** 因为 OpenAI/DeepSeek/Qwen 协议严格要求 `role="tool"` 消息**前面必须紧跟着对应的 `assistant(tool_calls)` 消息**！如果按固定条数硬切上下文，把 `assistant` 切进摘要而把孤零零的 `tool` 留在近期消息里，下一次调大模型 API 会直接 400 报错崩溃！你面试讲出这个细节，面试官立刻知道你真踩过生产坑！

### 3. `pyharness/runtime/engine.py`（确定性编排与 Self-Correction）
* **为什么并发工具调度还要套一个 `asyncio.Semaphore(max_parallel_tools)`？**
  如果模型一次吐出 20 个并发检索/API 调用，裸跑 `asyncio.gather` 会瞬间打爆下游数据库或触发外部 API 限流（429）；套上信号量既拿到了并发低延迟红利，又守住了系统稳定性底线。
* **Self-Correction 闭环 + 熔断器（Circuit Breaker）**：
  当工具出现 JSON 解析错、缺少必填参数或运行时报错（如 SQL 语法错），引擎不直接抛异常崩掉整个会话，而是把结构化错误包装成 `role="tool"` 喂回给大模型触发自省修复，同时递增 `consecutive_errors`——一旦**成功执行一步立刻清零**；连续错满 `max_consecutive_errors` 次立刻熔断退出，彻底杜绝无限死循环烧光 API 余额！

---

## 三、 上海 BOSS直聘 Agent 面试：8 道必问深挖题（基于 M1 源码直接答！）

1. **Q1：为什么要用 Event Sourcing（事件溯源）记录 Agent 执行，而不是只存最终问答对？**
   * *答*：支持前端断线按 `seq` 增量补发、支持 Trace 耗时瀑布图定位慢工具瓶颈、支持把每条轨迹连同 Badcase 导出用于离线评测和后续微调（对应你简历里的细粒度轨迹评测）。
2. **Q2：异步并发调用多个工具时，同步阻塞函数（如老式 `requests` / `sqlite3` / CPU 密集解析）会卡死整个 FastAPI 事件循环，怎么解决？**
   * *答*：看 `engine.py` 第 207 行——用 `inspect.iscoroutinefunction()` 自动探测，异步函数直接 `await`，同步函数自动丢进 `asyncio.to_thread()` 线程池执行，绝不阻塞主 Event Loop。
3. **Q3：Human-in-the-Loop 审批挂起时，后端内存里的 `asyncio.Future` 在多进程/多实例集群（如 K8s 多个 Pod）下会有什么问题？怎么演进？**
   * *答*：单机用内存 `Future` 极简高效；多实例部署时用户审批请求可能打到另一个 Pod，演进方案是把待审批状态存进 **Redis（带 TTL）**，审批接口通过 **Redis Pub/Sub** 广播唤醒持有该会话协程的 Pod，或者直接把 Checkpoint 持久化后彻底释放协程、收到审批回调再从 Checkpoint 重建恢复！
4. **Q4：上下文压缩（Context Compression）怎么防止把关键业务数字（如金额、ID、报错码）压缩丢掉？**
   * *答*：保留首条 `system` 约束与最近 $N$ 轮完整对话；压缩中间历史时采用**结构化槽位摘要模板**（强制保留实体 ID、SQL 表名、已确认结论、失败报错原因），并在截断边界确保不劈开 `assistant(tool_calls)` 与 `tool` 消息对。
5. **Q5：Agent 陷入 Self-Correction 死循环（反复改同一个错 SQL）怎么处理？**
   * *答*：双保险——`max_consecutive_errors` 连续失败熔断器 + `max_steps` 全局步数上限 + `TokenBudgetGuard` 预算封顶；进阶可在重试超过 2 次时自动切换更强推理模型（Fallback）或转人工接管。
6. **Q6：为什么还要写一个 `ScriptedMockLLMClient`（确定性测试桩）？**
   * *答*：真实 LLM 输出随机且耗钱耗时，无法进 CI 回归测试。用确定性 Mock 可以毫秒级覆盖 100% 边界条件（并行提速、审批拒绝、取消中断、熔断器、压缩边界），确保重构 Harness 引擎绝不引入 Regression。
7. **Q7：Token 估算用字符启发式和 `tiktoken` 有什么区别？线上怎么选？**
   * *答*：`tiktoken` 仅精确对应 OpenAI 词表，对 Qwen/DeepSeek 仍有误差且引入额外依赖；生产上用“偏保守字符启发式（中文/英文约 2.5 char/token）触发提前压缩 + 以 LLM 返回的真实 `usage` 做精准成本计费与预算熔断”是性价比最高的组合。
8. **Q8：你这套 Harness 和 LangGraph 的设计异同是什么？**
   * *答*：LangGraph 用显式 DAG/Cyclic StateGraph 声明节点与边，生态丰富适合快速拼装；自研 PyHarness 围绕**企业运行时治理**做极致透明化（事件溯源、协程秒断取消、审批 Broker、并发信号量），核心代码不到 500 行，无版本升级黑盒风险，也极易按需接回 LangGraph 节点作为子图。

---

## 四、 课后动手小练习（强烈建议亲手改 10 分钟，改完代码就真属于你了！）

1. **练习 1（10 分钟）**：打开 `pyharness/runtime/governance.py` 的 `ApprovalBroker.wait_for_decision`，在 `tests/test_m1_runtime.py` 里自己补充第 8 个单元测试 `test_approval_timeout_fails_closed()`——设置 `approval_timeout_seconds=0.1`，不调用 `resolve()`，验证超时后系统是否安全拒绝执行高危工具！
2. **练习 2（5 分钟）**：如果你手头有 DeepSeek 或阿里云通义千问 API Key，写一个 15 行的 `demo_live.py` 用 `OpenAICompatibleLLMClient` 跑通一次真实模型调用并打印 `journal.build_trace_summary(...)`！
