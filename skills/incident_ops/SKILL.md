---
name: incident_ops
description: 面向高危安全生产隐患查询、应急预案核查与值班告警分发的受控运维技能
triggers:
  - 告警
  - 紧急
  - 应急
  - 通知
  - alert
allowed_tools:
  - mcp_list_incidents
  - mcp_dispatch_emergency_alert
  - rag_search
---

# 应急告警与工单协同规范（Skill: incident_ops）

1. **先核实工单与规程**：在触发外部高危告警前，优先核实对应工单编号（`mcp_list_incidents`）或应急处置阈值（`rag_search`）。
2. **受控高危外发**：调用 `mcp_dispatch_emergency_alert` 会自动进入 `ApprovalBroker` 人工审批拦截，若审批被拒绝，须向操作员如实汇报拦截原因，严禁反复绕过重试。
