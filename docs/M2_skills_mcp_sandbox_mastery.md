# Milestone 2（企业升级版）源码精讲：瞬态动态 Skills、Stdio/OAuth MCP 协议与内核级/容器级隔离沙箱

> **对应最新源码**：
> - [loader.py](file:///D:/Code/agent-harness/pyharness/skills/loader.py) | [skills/](file:///D:/Code/agent-harness/skills)（含 `data_analysis`、`knowledge_research`、`incident_ops`）
> - [bridge.py](file:///D:/Code/agent-harness/pyharness/mcp/bridge.py)（`StdioMCPTransport` + `MCPClientBridge` OAuth/Bearer & `Mcp-Session-Id`）
> - [sql_sandbox.py](file:///D:/Code/agent-harness/pyharness/sandbox/sql_sandbox.py) | [sql_adapter.py](file:///D:/Code/agent-harness/pyharness/sandbox/sql_adapter.py)
> - [python_sandbox.py](file:///D:/Code/agent-harness/pyharness/sandbox/python_sandbox.py) | [providers.py](file:///D:/Code/agent-harness/pyharness/sandbox/providers.py)（`Subprocess` / `Docker+gVisor` / `E2B MicroVM`）

---

## 一、 瞬态动态 `SKILL.md` 路由与双层 `allowed_tools` 强管控（`skills/loader.py`）

### 旧版隐蔽大坑 $\rightarrow$ 新版优雅修复！
* **旧版坑点**：如果在第 1 轮把 `knowledge_research` 的 `SKILL.md` 直接拼进持久化 `history[0]`（System Prompt），那么第 2 轮用户改问 SQL 数据分析时，历史里依然死死粘着第 1 轮的知识库规则，导致**多轮技能污染（Skill Bleed）**！
* **新版企业设计（`resolve_turn_context` + `turn_system_instructions` + `allowed_tools`）**：
  1. 持久化 `history[0]` **永远保持纯净的 `BASE_SYSTEM_PROMPT`**；
  2. 每轮调用 `router.resolve_turn_context(query)` 动态产出本轮专属的瞬态指令 `turn_system_instructions` 与允许工具集合 `allowed_tools`（仅在该轮调 LLM 时临时合成，不写回历史！）；
  3. **双层工具作用域强制执行**：
     * **表层过滤**：只把 `allowed_tools` 内的工具 Schema 发给 LLM，既省 Token 又防干扰；
     * **底层兜底拦截**：万一大模型发生幻觉强行输出了未授权的工具名（例如在只读查资料轮次试图调用 `mcp_dispatch_emergency_alert`），`engine.py` 在执行前校验作用域，直接返回 `tool_scope_denied` 并触发 Self-Correction 纠正！

---

## 二、 生产级标准 MCP 工具桥：从内存直连到真实子进程 Stdio & OAuth HTTP（`mcp/bridge.py`）

升级后的 MCP 模块完整支持 **3 种真实工业传输协议**：

### 1. `StdioMCPTransport`（本地 / Sidecar 子进程标准 MCP 传输）
看 `bridge.py` 第 40–160 行，这是极高含金量的并发系统编程范例：
* **安全防泄密（Env Scrubbing）**：拉起第三方开源 MCP Server 子进程时，**绝对不能把宿主机 `os.environ` 原封不动继承进去**（否则第三方恶意 MCP 工具一行 `os.environ["LLM_API_KEY"]` 就能偷走公司所有密钥！）。`_build_scrubbed_env()` 只透传最小 OS 白名单（Windows 保留 `SystemRoot` 防 CPython BCrypt 初始化崩坏），彻底净空所有宿主机秘钥！
* **并发多路复用（JSON-RPC ID Multiplexing）**：单条持久子进程同时支撑并发多工具调用！后台 `_read_loop()` 持续读取 `stdout`，按每行返回的 JSON-RPC `id` 精准唤醒 `_pending[id]` 对应的 `asyncio.Future`！
* **防管道死锁与防僵尸进程**：单独起 `_drain_stderr()` 持续消费 `stderr`（防止 64KB 操作系统管道写满导致子进程永久卡死），关闭时按 `stdin.close() -> terminate() -> kill()` 三步确保零僵尸进程！

### 2. Streamable HTTP MCP 传输（带 OAuth/Bearer 鉴权 + `Mcp-Session-Id` 会话保持）
对接企业远程 MCP 网关时，自动透传 Bearer Token / OAuth 凭证，并遵守 MCP Streamable HTTP 规范持久携带服务端下发的 `Mcp-Session-Id` 请求头，同时把远程工具 `annotations.riskLevel` 自动映射进 PyHarness 风控等级体系。

---

## 三、 纵深防御 SQL 沙箱：SQLite 内核 Authorizer + 外部只读数仓 Adapter

### 1. 为什么只靠正则拦截 SQL 注入远远不够？（面试红队必考！）
* **正则的两大致命漏洞**：
  1. **误杀合法内容**：用户查 `SELECT 'safe; DELETE word inside literal'`（单引号字符串里包含分号和 DELETE），粗糙正则直接误杀！
  2. **容易绕过与资源炸弹**：攻击者写 `WITH dummy AS (SELECT 1 LIMIT 1) SELECT * FROM big_table` 骗过简单 `LIMIT` 检测，或者写**无限递归 CTE 炸弹** `WITH RECURSIVE inf(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM inf) SELECT SUM(x) FROM inf` 把整台服务器 CPU 打满到 100%！

### 2. 新版 `SQLSandbox`（`sql_sandbox.py`）的 4 重内核级防御
1. **词法去注释与字面量保护**：准确识别单引号字符串，放行合法字面量，精准拦截真实语句中的多语句分号与写操作；
2. **SQLite C 内核级 `conn.set_authorizer()`**：即使正则被某种奇技淫巧绕过，SQLite C 引擎在编译字节码阶段会逐动作回调 Authorizer——**非只读 `SELECT / READ / FUNCTION` 操作一律由数据库内核直接返 `SQLITE_DENY` 拒绝！**
3. **强制游标截断 `fetchmany(max_rows + 1)`**：不依赖脆弱的字符串拼接 `LIMIT`，直接在游标层硬截断最大返回行数并返回 `truncated=True`，彻底破解 CTE `LIMIT 1` 绕过！
4. **CPU 炸弹秒杀 `conn.set_progress_handler()`**：每执行一定数量虚拟机指令就检查耗时，一旦超过 `query_timeout_seconds` 返回非零立即由 SQLite 内核中断无限递归 CTE！

### 3. 外部企业数仓只读适配器（`ExternalReadOnlySQLAdapter` in `sql_adapter.py`）
通过 `SQLBackendAdapter` Seam，零代码改动即可把本地 SQLite 切换为**企业 PostgreSQL / MySQL 只读从库**：
* 自动开启数据库引擎级只读事务 `SET TRANSACTION READ ONLY`；
* 下发数据库级超时保护（PostgreSQL `SET LOCAL statement_timeout` / MySQL `SET SESSION max_execution_time`）；
* `finally` 块**无条件 `rollback()` 绝不 `commit()`**；日志自动对 DSN 密码做 `_redact_dsn()` 脱敏！

---

## 四、 0-Day 红队加固与三态可插拔 Python 沙箱（`sandbox/providers.py`）

### 1. 真实 Python 沙箱逃逸红队案例（你面试讲出这 3 个 Payload，面试官绝对惊呆！）
早期版本虽然禁了 `__class__` 双下划线，但安全红队发现 Python 依然存在 3 条绕过链：
1. **单下划线内部模块逃逸**：`import collections; collections._sys.modules['os'].environ`（通过白名单标准库里的单下划线属性 `_sys` 直接捞出 `os`！）；
2. **字符串格式化反射逃逸**：`"{0.__globals__[__builtins__]}".format(statistics.mean)`（AST 看不见字符串内部，但 `.format()` 运行时会自动解析属性路径捞出 `__globals__`！）；
3. **`while True: pass` 死循环卡死进程**。

### 2. 新版三层纵深防护 + `SandboxProvider` 可插拔架构
* **第一层（升级版 AST 审计）**：把禁止属性从双下划线 `__` 收紧为**一切单/双下划线开头属性（`attr.startswith("_")`）**，并彻底封禁 `.format`、`.format_map`、`type`、`dir`！
* **第二层（三态物理隔离 Provider Seam）**：
  | Provider | 隔离级别 | 核心安全配置 | 适用环境 |
  | :--- | :--- | :--- | :--- |
  | **`SubprocessSandboxProvider`** | OS 独立进程隔离 | `python -I -B` 隔离模式 + **净空宿主机环境变量**（不泄露 `LLM_API_KEY`）+ 独立 `stdout` + 超时直接 `proc.kill()` 强杀死循环 | 本地开发 / CI（零 Docker） |
  | **`DockerSandboxProvider`** | 容器 + **gVisor (`runsc`)** 强隔离 | `NetworkDisabled=True` & `NetworkMode="none"`（阻断外网及本机回环攻击）、`ReadonlyRootfs=True`、`/tmp` `noexec,nosuid`、`CapDrop=["ALL"]`、`no-new-privileges`、非特权用户 `65534:65534`、`Memory=256MB`、`NanoCpus=0.5`、`PidsLimit=32`、`finally` 强制 `DELETE ?force=true` | **企业内网生产首选！** |
  | **`E2BSandboxProvider`** | **Firecracker MicroVM** 硬件虚拟化 | 独立微虚拟机内核隔离，按次创建 $\rightarrow$ 执行 $\rightarrow$ `DELETE` 销毁；缺 Key/Daemon 时 **Fail-Closed 拒绝静默降级** | 高安全云端多租户 |

---

## 五、 本阶段上海一面高频深挖 6 题

1. **Q：你们的 `SKILL.md` 怎么防止第 1 轮的技能规则污染第 2 轮对话？**
   * *答*：持久历史只保留干净的基础 System Prompt，每轮动态计算瞬态 `turn_system_instructions` 与 `allowed_tools` 仅在单轮推理有效，并在工具分发层做二次白名单校验。
2. **Q：为什么拉起第三方 Stdio MCP 子进程时必须清洗环境变量？**
   * *答*：避免宿主机进程里的 `LLM_API_KEY`、数据库密码、`HARNESS_JWT_SECRET` 随子进程 `os.environ` 继承被不可信第三方工具读取外泄。
3. **Q：SQL 只读沙箱怎么防御 `WITH RECURSIVE` 无限递归 CPU 炸弹和绕过 `LIMIT`？**
   * *答*：用 SQLite 内核 `set_authorizer` 保只读、`set_progress_handler` 限时中断递归炸弹、`fetchmany(max_rows + 1)` 游标硬截断；外部 PG/MySQL 使用 `SET TRANSACTION READ ONLY` + `statement_timeout` + 强制 `rollback`。
4. **Q：Python 沙箱里为什么光禁双下划线 `__` 还不够？怎么彻底防住？**
   * *答*：存在 `collections._sys` 单下划线泄露和 `"{0.__globals__}".format(...)` 运行时属性解析逃逸。我们采用“AST 封禁一切 `_` 开头属性与 `format/format_map` + 独立隔离进程/Docker(`--network=none`, `runsc`, `CapDrop=ALL`, `PidsLimit=32`)/E2B MicroVM”纵深防御。
5. **Q：显式配置了 `sandbox_provider="docker"` 但宿主机没起 Docker Daemon，应该静默降级到普通子进程吗？**
   * *答*：**绝对不行！必须 Fail-Closed（安全失败）抛出 `SandboxConfigurationError`！** 静默降级会让安全团队误以为运行在容器隔离里，实际裸奔在宿主机上。
6. **Q：Docker Engine REST API 返回的 `/logs` 为什么不能直接 `.decode("utf-8")` 解析 JSON？**
   * *答*：在 `Tty=False` 时，Docker 返回的是多路复用二进制帧（每帧带 8 字节头：1 字节流类型 `1=stdout/2=stderr` + 3 字节保留 + 4 字节大端长度），我们在 `_decode_docker_multiplexed_logs()` 中严格按帧解包只提取 `stdout`！
