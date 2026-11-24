# Milestone 5（L4 升级版）评测闭环精讲与上海 Agent 面试总监级通关手册

> **全套学习顺序建议**：精读 `M1` → `M2` → `M3` → `M4` → `M6` → `M5`，同时在浏览器里打开 [interactive_architecture_explorer.html](file:///D:/Code/agent-harness/docs/interactive_architecture_explorer.html) 亲手点一遍全部 4 个交互实验室！

---

## 一、 3 分钟面试高分自我项目阐述逐字稿（L4 企业级 `PyHarness` 完整版）

> “为了把我在企业里做 Multi-Agent、Hybrid RAG 和大模型微调的工程沉淀系统化，我深度对标 **DeepSeek Harness (DSH) + Agentic Control Plane (ACP)** 架构标准，从零自研了一套企业内网可推行级 Agent Harness 平台 **PyHarness**（Python 3.11 + FastAPI + Vue 3 + PostgreSQL + Redis + Qdrant + Docker），目前全量单元、集成与红队安全测试共 **37 项零警告全过**，核心攻克了企业 Agent 落地的 5 大工程难题：
>
> 1. **企业安全控制面与合规审计**：实现了零外部依赖的 HS256 JWT / Scoped API Key 三模鉴权并通过 `ContextVar` 全链路传播 `Principal`；落地了**四眼原则（SoD）防自审批与跨租户拦截**；设计了声明式 YAML **动态策略引擎 ACP**（按 `DENY > ASK > ALLOW` 裁决、防 SQL 注释绕过、支持影子 `AUDIT` 灰度模式）；在事件落盘前经 **DLP 脱敏 + 中英文间接 Prompt 注入 XML 隔离护栏** 清洗，并集成滑动窗口 RPM 限流与部门级 FinOps 成本分账报表。
> 2. **微内核运行时、多模型熔断与防提权子代理**：借鉴 Cordis 实现了支持 Profile 切换与 LIFO 原子回滚的 **`PluginContext` 插件微内核**；构建了支持 `reasoning`/`fast` 分层路由与半开探测熔断器的 **`ModelRouterLLMClient`**（流式吐字前无感主备切换）；实现了上下文硬隔离、强制继承父级策略/取消令牌、且防递归防提权的 **`SubagentManager`**；实测 `asyncio.gather + Semaphore` 并发工具调度将 P50 延迟从 **200.9ms 降至 76.8ms（-61.8%）**。
> 3. **防篡改事件溯源与多 Pod 分布式协作**：抽象 `JournalStorageBackend` 同时支持 SQLite WAL 与 **PostgreSQL 事务级建议锁（`pg_advisory_xact_lock`）**，为每条事件构建 **SHA-256 防篡改审计哈希链（`/audit/verify`）** 与时间点 **Session Forking（`/fork`）**；基于 Redis 实现分布式会话锁、跨 Pod 审批唤醒/取消广播与 **`Last-Event-ID` 断线无损重连**，并暴露 OpenTelemetry GenAI Trace 与 Prometheus `/metrics`。
> 4. **标准 MCP 与内核/容器级隔离沙箱**：实现了按轮次瞬态注入的 `SKILL.md` 与双层 `allowed_tools` 强管控；支持净空环境变量的真实 **`StdioMCPTransport` 子进程多路复用** 与 OAuth/Bearer Streamable HTTP MCP；SQL 沙箱叠加 SQLite 内核 `set_authorizer`、`progress_handler` 防递归 CTE 炸弹与外部 PG/MySQL 只读适配器；Python 沙箱完成 0-day 反射逃逸红队加固，抽象了 **Subprocess / Docker+gVisor (`--network=none`) / E2B Firecracker MicroVM** 三态 Fail-Closed 插件边界。
> 5. **多租户 Hybrid RAG 与自动化数据飞轮**：表格保护 Parent-Child 层级切片 + Query 改写 + 中英文 BM25 + **Qdrant 持久化向量库（UUIDv5 幂等、重启自动恢复、对话期强制 `tenant_id` ACL Pre-Filtering）** + RRF 融合 + Cross-Encoder 精排；配合 `eval/` 评测套件自动将不可恢复失败轨迹打根因标签导出 JSONL，直接反哺 Prompt 优化或 LoRA/DPO 后训练。”

---

## 二、 上海技术总监最爱追问的 15 个压轴难题速查索引

> 以下 15 题覆盖了从架构选型、分布式并发到红队攻防的全部核心追问（每题的源码级深讲分别位于对应 `M1`–`M6` 手记中）：

| 编号 | 高频压轴追问 | 核武器级答法核心（1 句话抓重点） | 详讲出处 |
| :---: | :--- | :--- | :---: |
| **1** | 为什么自研 Harness 而不是直接用 LangChain / Dify？ | 确定性治理透明可控（防坏 JSON、防死循环、秒断取消、防篡改审计、ACP 动态策略），零黑盒风险且可挂载 LangGraph 作为子图。 | [M1](M1_runtime_mastery.md) |
| **2** | 多 Pod 下两个实例同时往一个会话写事件怎么防 `seq` 乱序？ | PostgreSQL 事务级建议锁 `pg_advisory_xact_lock(hashtext(session_id))` 按会话粒度串行化，不同会话 100% 并发。 | [M1](M1_runtime_mastery.md) |
| **3** | 主模型流式输出到一半挂了，能无感切备用模型吗？ | 首字未吐（`yielded_any=False`）秒切备模型；已吐部分 Token 禁止静默拼接防语义撕裂，直接报错并计入熔断器。 | [M1](M1_runtime_mastery.md) |
| **4** | 子代理委派（Subagent）怎么防越权提权和无限套娃？ | `ContextVar` 追踪深度到 `max_depth` 剥离委派工具 + 默认剥离 `HIGH` 风险工具 + 有效工具与父级取交集 + 强制继承父策略/DLP/取消令牌。 | [M1](M1_runtime_mastery.md) |
| **5** | 上下文增量摘要压缩最隐蔽的生产 Bug 是什么？ | 切分边界不能劈开 `assistant(tool_calls)` 和 `role="tool"` 配对消息，否则下游 LLM API 必报 400。 | [M1](M1_runtime_mastery.md) |
| **6** | 怎么防第 1 轮加载的 Skill 污染第 2 轮对话？ | 持久化 `history[0]` 永远保持纯净基础 System Prompt，Skill 指令与 `allowed_tools` 仅作单轮瞬态合成并在执行层二次拦截。 | [M2](M2_skills_mcp_sandbox_mastery.md) |
| **7** | Python 沙箱只禁 `__class__` 为什么还会被攻破？怎么修？ | 存在 `collections._sys` 单下划线泄露和 `"{0.__globals__}".format()` 运行时解析逃逸；必须禁一切 `_` 属性 + `format` + OS/Docker `runsc`/MicroVM 物理隔离。 | [M2](M2_skills_mcp_sandbox_mastery.md) |
| **8** | SQL 只读沙箱怎么防 `WITH RECURSIVE` 无限递归 CPU 炸弹？ | SQLite 内核 `set_authorizer` 保只读 + `set_progress_handler` 限时强杀 + `fetchmany(max_rows+1)` 游标截断；外部库用 `SET TRANSACTION READ ONLY` + `statement_timeout`。 | [M2](M2_skills_mcp_sandbox_mastery.md) |
| **9** | Stdio MCP 子进程怎么防宿主机密钥泄露和僵尸进程？ | 最小 OS 环境变量白名单净空 `LLM_API_KEY` + 独立协程抽干 `stderr` 防管道死锁 + 按 JSON-RPC `id` 多路复用 + `stdin->terminate->kill` 兜底。 | [M2](M2_skills_mcp_sandbox_mastery.md) |
| **10** | 多租户 RAG 为什么不能搜完 Top-10 再在 Python 过滤租户？ | Post-Filtering 会导致召回坍塌甚至空结果；必须下推为 Qdrant `filter.must` 与 BM25 检索期 Pre-Filtering！ | [M3](M3_hybrid_rag_mastery.md) |
| **11** | SSE 断线重连怎么做到不丢事件也不重复发事件？ | 带 `Last-Event-ID` 先订阅实时队列、再回放 `seq > after_seq` 历史事件，切回实时队列后按 `seq > last_sent_seq` 去重。 | [M4](M4_fullstack_deployment_mastery.md) |
| **12** | Nginx 代理 SSE 为什么会憋住不流式输出？ | 默认开启了 `proxy_buffering on`；必须配置 Nginx `proxy_buffering off` + 响应头 `X-Accel-Buffering: no`。 | [M4](M4_fullstack_deployment_mastery.md) |
| **13** | 攻击者用 `DELETE FROM tbl -- WHERE 1=1` 绕过正则策略怎么防？ | `PolicyEngine` 在正则匹配前先剥离单引号外部的 `--` 与 `/* */` SQL 注释，同时按严格 `DENY > ASK > ALLOW` 优先级裁决。 | [M6](M6_enterprise_security_acp_and_plugins_mastery.md) |
| **14** | 怎么防用户自提自批高危操作或伪造 `reviewer="admin"`？ | 完全忽略请求体 `reviewer`，从加密鉴权的 `Principal` 绑定真实审批人，强校验 `principal.user_id != requester_id` 且同租户且具 `approver/admin` 角色。 | [M6](M6_enterprise_security_acp_and_plugins_mastery.md) |
| **15** | 怎么把 Agent 线上失败轨迹变成模型能力提升飞轮？ | `eval/benchmark_harness.py` 自动识别触发熔断或失败轨迹，提取 `error_type` 根因标签导出标准 JSONL，直接用于 Prompt 迭代或 LoRA/DPO 后训练微调！ | [M5](M5_eval_and_interview_playbook.md) |
