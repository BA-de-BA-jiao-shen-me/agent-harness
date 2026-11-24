---
name: knowledge_research
description: 面向企业技术文档、规章制度与多源知识库的混合检索与带引用研报生成技能
triggers:
  - 文档
  - 规范
  - 排查
  - 知识库
  - 制度
  - 报告
  - 隐患
  - 检索
allowed_tools:
  - rag_search
  - mcp_list_incidents
---

# 企业知识检索与研报规范（Skill: knowledge_research）

1. **必须先检索后作答**：回答业务规范或排查步骤前必须调用 `rag_search` 检索权威证据。
2. **强制引用溯源**：关键论断与参数值末尾必须标注来源编号（如 `[1]`、`[2]`），与检索结果严格对应。
3. **防幻觉底线**：若检索结果中不包含确切答案，必须如实说明“当前知识库未覆盖该细节”，禁止编造数字或规程条款。
