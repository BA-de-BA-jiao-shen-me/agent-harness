# Milestone 2 源码精讲与上海一面深挖手记（Skills / 标准 MCP / 安全自省沙箱）

## 一、 本阶段你掌握了哪 3 项上海 Agent 岗“降维打击”硬核技术？

### 1. 声明式 `SKILL.md` 动态按需注入（`pyharness/skills/loader.py`）
* **解决什么痛点？** 如果把 10 个业务领域的 Prompt 规则全部写死进全局 `system_prompt`，会导致：① System Prompt 暴涨几千 Token，白烧钱且引发指令冲突；② 出现“长上下文注意力稀释（Lost in the Middle）”。
* **我们的解法（借鉴 DSH `skill-turn-selection.mjs`）**：每个领域技能封装成独立的 `skills/<name>/SKILL.md`（YAML Frontmatter 声明触发词与工具白名单 + Markdown 正文写 SOP），`SkillRouter` 每轮动态挑选最相关的 1–2 个 Skill 注入 Prompt，极大降低 Token 消耗并提升指令遵循度！

### 2. 标准 MCP（Model Context Protocol, JSON-RPC 2.0）工具桥（`pyharness/mcp/bridge.py`）
* **为什么 2026 年上海 JD 必问 MCP？** 过去每接一个外部系统（Jira、企业数据库、告警系统）都要在 Agent 代码里写一套私有 Tool 胶水代码（$M \times N$ 耦合）；MCP 用标准 **JSON-RPC 2.0 协议（`initialize` / `tools/list` / `tools/call`）** 把工具提供方和 Agent 运行时彻底解耦。
* **亮点设计**：`MCPClientBridge` 同时支持 **In-Process 零延迟直连** 与 **Remote HTTP JSON-RPC 接入**，并在挂载进 `ToolRegistry` 时自动读取 MCP `annotations.riskLevel`——例如 `dispatch_emergency_alert` 自动标为 `RiskLevel.HIGH`，**无缝触发 Milestone 1 的 `ApprovalBroker` 人工审批拦截！**

### 3. SQL + Python AST 双安全沙箱与 Self-Correction（`pyharness/sandbox/`）
* **SQL 沙箱防御 4 重锁** (`sql_sandbox.py`)：
  1. 只允许 `SELECT / WITH` 开头，正则封杀一切 `DROP/DELETE/UPDATE/ALTER/PRAGMA`；
  2. 检测到多语句分号 `;` 直接拒绝，彻底封死 `SELECT 1; DROP TABLE ...` 拼接注入；
  3. 未显式写 `LIMIT` 时自动追加 `LIMIT 100`，防止大表全表扫描打爆内存和上下文窗口；
  4. 将真实写库剥离为独立高危工具 `execute_mutating_sql`（`RiskLevel.HIGH`），强制人工审批！
* **Python 沙箱静态 AST 审计** (`python_sandbox.py`)：
  * 面试官极爱问：“只靠限制 `builtins` 能防住 Python 沙箱逃逸吗？”
  * **高分回答**：不能！经典逃逸链 `().__class__.__base__.__subclasses__()` 不需要 `import` 就能从基类反射捞出 `os._wrap_close` 执行任意命令！所以我们在执行前先跑 `ast.parse()` 遍历整棵语法树，**直接禁止一切 `__` 双下划线魔术属性访问 + 禁止黑名单调用 + 白名单模块导入**；并在更高安全隔离的生产环境下进一步叠加 Docker/gVisor/E2B 容器隔离！

---

## 二、 上海一面高频深挖问答（M2 专属）

1. **Q：MCP 和传统的 Function Calling 有什么本质区别？**
   * *答*：Function Calling 是大模型厂商层面的“输出结构化调用意图”能力；MCP 是应用架构层面的“工具发现与通信标准协议”。大模型通过 Function Calling 决定调用什么，Agent Harness 通过 MCP Client 去标准发现并安全执行远程工具。
2. **Q：你们的 SQL Self-Correction 怎么提升首次命中率、减少无效重试？**
   * *答*：两条组合拳——① 通过 `data_analysis/SKILL.md` 强制要求写 SQL 前先调 `inspect_sql_schema` 获取精准表结构，杜绝幻觉列名；② 报错时把数据库原始报错和报错类型包装回 `role="tool"`，同时配合 Milestone 1 的 `max_consecutive_errors` 熔断器防止无限纠错。
