---
name: data_analysis
description: 面向结构化业务数据库与指标统计的 SQL + Python 自省分析技能
triggers:
  - sql
  - 数据库
  - 统计
  - 报表
  - 指标
  - 故障数
  - 趋势
  - 计算
allowed_tools:
  - inspect_sql_schema
  - query_sql_readonly
  - execute_mutating_sql
  - run_python_sandbox
---

# 数据分析专家执行规范（Skill: data_analysis）

1. **先查结构再写 SQL**：生成 SQL 前必须先调用 `inspect_sql_schema` 确认真实表名与字段名，严禁凭空猜测列名。
2. **错误自省闭环（Self-Correction）**：如果 `query_sql_readonly` 返回语法或列名错误，必须依据报错信息修正 SQL 后重试，最多重试 3 次。
3. **安全原则**：默认只使用只读查询；只有用户明确要求写入/清理数据时，才可调用 `execute_mutating_sql`（将自动触发人工审批拦截）。
4. **数值计算精算**：涉及增长率、方差或复杂聚合时，优先交由 `run_python_sandbox` 完成精确计算，禁止大模型心算。
