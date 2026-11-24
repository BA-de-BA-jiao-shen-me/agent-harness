from __future__ import annotations

import os
from pathlib import Path

from pyharness.mcp import MCPClientBridge, create_builtin_enterprise_mcp_server
from pyharness.rag import HybridRAGPipeline
from pyharness.runtime import (
    AgentHarnessEngine,
    ApprovalBroker,
    BaseLLMClient,
    CancellationToken,
    EventJournal,
    HarnessConfig,
    LLMResponse,
    Message,
    OpenAICompatibleLLMClient,
    TokenUsage,
    ToolCall,
    ToolRegistry,
)
from pyharness.sandbox import PythonSandbox, SQLSandbox
from pyharness.skills import SkillRouter

BUILTIN_DEMO_DOC = """
# 企业智能运维与安全隐患排查知识库

## 1. SF6 高压断路器巡检与告警阈值
运行中应每 24 小时核查气体压力表。压力低于 0.52 MPa 触发一级低压告警；低于 0.48 MPa 时强制闭锁分合闸并上报调度中心。

| 设备类型 | 正常范围 | 一级告警阈值 | 紧急闭锁阈值 |
| --- | --- | --- | --- |
| SF6 断路器 | 0.58 - 0.65 MPa | 0.52 MPa | 0.48 MPa |
| 气体监测探头 | 0 - 15 ppm | 25 ppm | 50 ppm |

## 2. 石化压力阀温升应急处置规程
金山石化 A 区压力阀红外测温连续 10 分钟高于 85℃ 时，须在 15 分钟内启动二级降温预案，同时核查泄压阀开度与传感器零漂。
"""


class DemoSmartRouterLLM(BaseLLMClient):
    """
    Zero-API-key interactive demo LLM so reviewers/interviewers can run the full platform
    locally and experience live RAG citations, SQL queries, and HIGH-risk approval dialogs.
    """

    async def complete(self, messages: list[Message], tools=None) -> LLMResponse:
        last = messages[-1]
        if last.role == "tool":
            tool_outputs = []
            for m in reversed(messages):
                if m.role == "user":
                    break
                if m.role == "tool":
                    tool_outputs.append(f"- **{m.name}**: `{m.content[:260]}`")
            summary = "\n".join(reversed(tool_outputs))
            return LLMResponse(
                content=f"已通过受控工具链完成分析与核查，核心结果如下：\n\n{summary}\n\n（当前运行于免 Key 演示模式，配置 `.env` 中 `LLM_API_KEY` 即可无缝切换 DeepSeek / Qwen 真实大模型。）",
                usage=TokenUsage(prompt_tokens=180, completion_tokens=85),
                model="pyharness-demo-planner",
            )

        q = last.content.lower()
        calls: list[ToolCall] = []
        reasoning = "分析用户请求意图并路由至对应受控工具。"
        if any(k in q for k in ("告警", "紧急", "alert", "删", "drop", "delete")):
            reasoning = "检测到高危外部告警意图，调用 MCP 高危工具触发 Human-in-the-Loop 审批。"
            calls.append(
                ToolCall(
                    name="mcp_dispatch_emergency_alert",
                    arguments={"incident_id": "INC-2026-01", "channel": "ops-oncall"},
                )
            )
        elif any(k in q for k in ("sql", "统计", "停机", "数据库", "故障")):
            reasoning = "检测到结构化数据分析意图，先并发查表结构与高危工单统计。"
            calls.append(ToolCall(name="inspect_sql_schema", arguments={}))
            calls.append(
                ToolCall(
                    name="query_sql_readonly",
                    arguments={
                        "sql": "SELECT region, COUNT(*) AS cnt, SUM(downtime_minutes) AS total_min FROM equipment_incidents GROUP BY region"
                    },
                )
            )
        else:
            reasoning = "检测到知识规范查询意图，调用企业 Hybrid RAG 混合检索与精排引擎。"
            calls.append(ToolCall(name="rag_search", arguments={"query": last.content, "top_k": 3}))

        return LLMResponse(
            reasoning_content=reasoning,
            tool_calls=calls,
            usage=TokenUsage(prompt_tokens=140, completion_tokens=45),
            model="pyharness-demo-planner",
        )


class RuntimeContainer:
    """Singleton dependency container managing runtime state across API requests."""

    def __init__(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        db_path = os.getenv("HARNESS_JOURNAL_DB", ":memory:")

        self.journal = EventJournal(db_path=db_path)
        self.approval_broker = ApprovalBroker()
        self.tools = ToolRegistry()
        self.rag = HybridRAGPipeline()
        self.sql_sandbox = SQLSandbox(":memory:")
        self.sql_sandbox.seed_demo_enterprise_data()
        self.python_sandbox = PythonSandbox()
        self.skill_router = SkillRouter(root_dir / "skills")
        self.mcp_server = create_builtin_enterprise_mcp_server()
        self.mcp_bridge = MCPClientBridge(server=self.mcp_server)

        self.active_cancellations: dict[str, CancellationToken] = {}
        self.session_histories: dict[str, list[Message]] = {}
        self._initialized = False

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if api_key and api_key != "your_api_key_here":
            self.llm: BaseLLMClient = OpenAICompatibleLLMClient(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
            )
        else:
            self.llm = DemoSmartRouterLLM()

        self.engine = AgentHarnessEngine(
            llm=self.llm,
            tools=self.tools,
            journal=self.journal,
            approval_broker=self.approval_broker,
            config=HarnessConfig(
                max_steps=int(os.getenv("HARNESS_MAX_STEPS", "10")),
                max_parallel_tools=int(os.getenv("HARNESS_MAX_PARALLEL_TOOLS", "4")),
                context_compress_threshold=int(os.getenv("HARNESS_CONTEXT_COMPRESS_THRESHOLD", "6000")),
                max_tokens_per_turn=int(os.getenv("HARNESS_MAX_TOKENS_PER_TURN", "32000")),
            ),
        )

    async def startup(self) -> None:
        if self._initialized:
            return
        self.sql_sandbox.register_tools(self.tools)
        self.python_sandbox.register_tools(self.tools)
        self.rag.register_tools(self.tools)
        await self.mcp_bridge.mount_into_registry(self.tools)
        await self.rag.ingest_markdown(BUILTIN_DEMO_DOC, source="enterprise_safety_manual.md", doc_id="demo_manual")
        self._initialized = True

    def shutdown(self) -> None:
        self.journal.close()
        self.sql_sandbox.close()
