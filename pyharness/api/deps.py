from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyharness.core import HarnessPlugin, PluginContext
from pyharness.infra import (
    CoordinationBackend,
    TelemetryRecorder,
    create_coordination_backend,
)
from pyharness.mcp import MCPClientBridge, create_builtin_enterprise_mcp_server
from pyharness.rag import (
    BaseVectorStore,
    DenseVectorIndex,
    HashingSemanticEmbedder,
    HybridRAGPipeline,
    OpenAICompatibleEmbedder,
    QdrantVectorStore,
)
from pyharness.runtime import (
    AgentHarnessEngine,
    ApprovalBroker,
    BaseLLMClient,
    CancellationToken,
    EventJournal,
    EventType,
    HarnessConfig,
    JournalStorageBackend,
    LLMResponse,
    Message,
    ModelRoute,
    ModelRouterLLMClient,
    OpenAICompatibleLLMClient,
    SubagentManager,
    TokenUsage,
    ToolCall,
    ToolRegistry,
)
from pyharness.sandbox import (
    PythonSandbox,
    SQLSandbox,
    SandboxProvider,
    create_sandbox_provider,
)
from pyharness.security import (
    AuthManager,
    GuardrailHook,
    PolicyEngine,
    QuotaManager,
    current_principal,
)
from pyharness.skills import SkillRouter

DEFAULT_SYSTEM_PROMPT = (
    "You are PyHarness, an enterprise AI assistant with Hybrid RAG, MCP tools, "
    "governed subagent delegation, dynamic access control policies, and isolated SQL/Python sandboxes."
)

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
    Zero-API-key interactive demo LLM so reviewers can run the full platform locally
    and experience live RAG citations, SQL queries, Python sandboxes, subagents, and approvals.
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
                content=(
                    f"已通过受控工具链完成分析与核查，核心结果如下：\n\n{summary}\n\n"
                    "（当前运行于免 Key 演示模式，配置 `.env` 中 `LLM_API_KEY` 即可无缝切换 DeepSeek / Qwen 真实大模型。）"
                ),
                usage=TokenUsage(prompt_tokens=180, completion_tokens=85),
                model="pyharness-demo-planner",
            )

        q = last.content.lower()
        calls: list[ToolCall] = []
        reasoning = "分析用户请求意图并路由至对应受控工具。"
        if any(k in q for k in ("委派", "子代理", "subagent")):
            reasoning = "检测到复合排查意图，通过 ctx.subagents 并行委派隔离上下文子代理执行调查。"
            calls.append(
                ToolCall(
                    name="delegate_subagent",
                    arguments={"task": "核查当前高危运维隐患与安全规程", "profile": "researcher"},
                )
            )
        elif any(k in q for k in ("告警", "紧急", "alert", "删", "drop", "delete")):
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
        elif any(k in q for k in ("python", "方差", "标准差")):
            reasoning = "检测到精确统计计算需求，调度隔离 Python 安全沙箱执行运算。"
            calls.append(
                ToolCall(
                    name="run_python_sandbox",
                    arguments={
                        "code": "import statistics\ndata = [95, 120, 60, 80]\nresult = round(statistics.stdev(data), 2)"
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


class _SecurityControlPlanePlugin(HarnessPlugin):
    """Mounts AuthN/RBAC, ACP PolicyEngine, GuardrailHook (DLP+Injection), and FinOps QuotaManager."""

    name = "security_control_plane"

    def __init__(self, container: RuntimeContainer) -> None:
        self._rt = container

    async def install(self, ctx: PluginContext) -> None:
        rt = self._rt
        ctx.provide("auth", rt.auth_manager)
        ctx.provide("policy", rt.policy_engine)
        ctx.provide("quota", rt.quota_manager)
        ctx.provide("telemetry", rt.telemetry)
        dispose_guardrail = rt.engine.add_hook(rt.guardrail_hook)
        ctx.effect(dispose_guardrail)


class _SandboxAndToolsPlugin(HarnessPlugin):
    """Mounts SQL Authorizer Sandbox, pluggable Python SandboxProvider, Hybrid RAG, and MCP Bridge."""

    name = "sandboxes_rag_mcp"

    def __init__(self, container: RuntimeContainer) -> None:
        self._rt = container

    async def install(self, ctx: PluginContext) -> None:
        rt = self._rt
        rt.sql_sandbox.register_tools(rt.tools)
        rt.python_sandbox.register_tools(rt.tools)
        rt.rag.register_tools(rt.tools)
        await rt.mcp_bridge.mount_into_registry(rt.tools)
        await rt.rag.restore_from_store()
        await rt.rag.ingest_markdown(
            BUILTIN_DEMO_DOC,
            source="enterprise_safety_manual.md",
            doc_id="demo_manual",
            metadata={"visibility": "public"},
        )
        ctx.provide("tools", rt.tools)
        ctx.provide("rag", rt.rag)
        ctx.provide("sandbox_provider", rt.python_sandbox.provider)
        ctx.effect(rt.python_sandbox.aclose)
        ctx.effect(rt.rag.aclose)
        ctx.effect(rt.mcp_bridge.aclose)


class _SubagentsPlugin(HarnessPlugin):
    """Mounts DSH-style `ctx.subagents` isolated delegation manager onto the parent engine."""

    name = "subagents"
    profiles = frozenset({"enterprise", "standalone"})

    def __init__(self, container: RuntimeContainer) -> None:
        self._rt = container

    async def install(self, ctx: PluginContext) -> None:
        rt = self._rt
        rt.subagent_manager.attach_parent_engine(rt.engine)
        rt.subagent_manager.register_tools(rt.tools)
        ctx.provide("subagents", rt.subagent_manager)


class RuntimeContainer:
    """
    Enterprise Dependency Container orchestrated via a Cordis-inspired `PluginContext`.

    Wires pluggable Seams:
    - `AuthManager` (JWT / API Key / Trusted Gateway + RBAC)
    - `PolicyEngine` (Dynamic ACP `allow` / `ask` / `deny` + Shadow `audit`)
    - `GuardrailHook` (DLP secret/PII redaction + indirect prompt injection tagging)
    - `QuotaManager` (Sliding-window RPM + daily departmental FinOps ledger)
    - `JournalStorageBackend` (PostgreSQL multi-Pod / SQLite WAL + SHA-256 hash chain)
    - `CoordinationBackend` (Redis Pub/Sub + locks + approvals or In-Memory)
    - `ModelRouterLLMClient` (Primary -> Fallback circuit breaker + Reasoning/Fast tiers)
    - `SandboxProvider` (Docker+socket-proxy / gVisor / E2B MicroVM / Isolated Subprocess)
    - `BaseVectorStore` (Qdrant persistent ACL-filtered vector store or In-Memory)
    - `TelemetryRecorder` (OpenTelemetry GenAI spans + Prometheus `/metrics`)
    """

    def __init__(
        self,
        coordination: CoordinationBackend | None = None,
        sandbox_provider: SandboxProvider | None = None,
        vector_store: BaseVectorStore | None = None,
        journal_store: JournalStorageBackend | None = None,
        auth_manager: AuthManager | None = None,
        policy_engine: PolicyEngine | None = None,
        llm: BaseLLMClient | None = None,
        profile: str | None = None,
        enforce_four_eyes: bool | None = None,
        rpm_limit: int | None = None,
    ) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.profile = (profile or os.getenv("HARNESS_PROFILE", "enterprise")).strip().lower()
        self.plugin_context = PluginContext(profile=self.profile)

        db_path = os.getenv("HARNESS_JOURNAL_DB", ":memory:")
        database_url = os.getenv("DATABASE_URL", "").strip() or None
        redis_url = os.getenv("REDIS_URL", "").strip()
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "pyharness_chunks").strip()
        sandbox_mode = os.getenv("HARNESS_SANDBOX_PROVIDER", "auto").strip()

        # 1. Identity, Telemetry & Coordination
        self.auth_manager: AuthManager = auth_manager or AuthManager.from_env()
        self.telemetry = TelemetryRecorder(service_name="pyharness")
        self.coordination: CoordinationBackend = coordination or create_coordination_backend(redis_url)

        allow_self_env = os.getenv("HARNESS_APPROVAL_ALLOW_SELF", "").strip().lower()
        if enforce_four_eyes is not None:
            sod_enabled = enforce_four_eyes
        elif allow_self_env in {"false", "0", "no"}:
            sod_enabled = True
        elif allow_self_env in {"true", "1", "yes"}:
            sod_enabled = False
        else:
            sod_enabled = self.auth_manager.mode == "enforce"

        # 2. Durable Hash-Chained EventJournal & Four-Eyes ApprovalBroker
        self.journal = EventJournal(
            db_path=db_path,
            coordination=self.coordination,
            store=journal_store,
            database_url=database_url,
        )
        self.approval_broker = ApprovalBroker(
            coordination=self.coordination,
            enforce_four_eyes=sod_enabled,
        )

        # 3. Dynamic Policy Engine (ACP), Guardrails & FinOps Quota
        if policy_engine is not None:
            self.policy_engine = policy_engine
        else:
            policy_file_env = os.getenv("HARNESS_POLICY_FILE", "").strip()
            policy_path = Path(policy_file_env) if policy_file_env else (root_dir / "policies" / "enterprise_policy.yaml")
            self.policy_engine = (
                PolicyEngine.from_yaml(policy_path) if policy_path.exists() else PolicyEngine()
            )

        async def _emit_security_alert(
            session_id: str, turn_id: str, detail: dict[str, Any]
        ) -> None:
            principal = current_principal.get()
            alert_type = (
                "prompt_injection" if detail.get("injection_detected") else "dlp_redaction"
            )
            self.telemetry.record_security_alert(
                alert_type=alert_type,
                tenant_id=principal.tenant_id if principal else "default",
            )
            await self.journal.emit(session_id, turn_id, EventType.SECURITY_ALERT, detail)

        self.guardrail_hook = GuardrailHook(on_alert=_emit_security_alert)
        eff_rpm = (
            rpm_limit
            if rpm_limit is not None
            else int(os.getenv("HARNESS_RPM_LIMIT", "60"))
        )
        self.quota_manager = QuotaManager(
            coordination=self.coordination,
            rpm_limit=eff_rpm,
            enabled=os.getenv("HARNESS_QUOTA_ENABLED", "true").strip().lower() not in {"false", "0"},
        )

        # 4. Tools, RAG & Isolated Sandboxes
        self.tools = ToolRegistry()

        embed_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        embed_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
        embedder = (
            OpenAICompatibleEmbedder(
                api_key=embed_key,
                base_url=embed_url,
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            )
            if embed_key and embed_url
            else HashingSemanticEmbedder()
        )

        if vector_store is not None:
            resolved_store: BaseVectorStore = vector_store
        elif qdrant_url:
            resolved_store = QdrantVectorStore(
                embedder=embedder,
                url=qdrant_url,
                collection_name=qdrant_collection,
                api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
            )
        else:
            resolved_store = DenseVectorIndex(embedder=embedder)

        self.rag = HybridRAGPipeline(embedder=embedder, vector_store=resolved_store)
        self.sql_sandbox = SQLSandbox(os.getenv("HARNESS_SQL_SANDBOX_DB", ":memory:"))
        self.sql_sandbox.seed_demo_enterprise_data()

        resolved_sb_provider = sandbox_provider or create_sandbox_provider(sandbox_mode)
        self.python_sandbox = PythonSandbox(provider=resolved_sb_provider)
        self.skill_router = SkillRouter(root_dir / "skills")
        self.mcp_server = create_builtin_enterprise_mcp_server()
        self.mcp_bridge = MCPClientBridge(server=self.mcp_server)

        self.active_cancellations: dict[str, CancellationToken] = {}
        self.session_histories: dict[str, list[Message]] = {}
        self._initialized = False

        # 5. Multi-Model Router (Primary -> Fallback + Fast Tier)
        if llm is not None:
            self.llm: BaseLLMClient = llm
        else:
            self.llm = self._build_llm_from_env()

        summary_llm = (
            self.llm.for_tier("fast")
            if isinstance(self.llm, ModelRouterLLMClient)
            else None
        )

        # 6. Subagent Manager & Governed AgentHarnessEngine
        self.subagent_manager = SubagentManager(
            llm=self.llm,
            tools=self.tools,
            journal=self.journal,
            approval_broker=self.approval_broker,
            policy_engine=self.policy_engine,
            telemetry=self.telemetry,
        )

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
            policy_engine=self.policy_engine,
            telemetry=self.telemetry,
            subagent_manager=self.subagent_manager,
            summary_llm=summary_llm,
        )

    @staticmethod
    def _build_llm_from_env() -> BaseLLMClient:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            return DemoSmartRouterLLM()

        primary = OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
        )
        routes: list[ModelRoute] = [
            ModelRoute(name="primary", client=primary, tier="reasoning", priority=0),
        ]

        fb_key = os.getenv("LLM_FALLBACK_API_KEY", "").strip()
        fb_url = os.getenv("LLM_FALLBACK_BASE_URL", "").strip()
        if fb_key and fb_url:
            fallback = OpenAICompatibleLLMClient(
                api_key=fb_key,
                base_url=fb_url,
                model=os.getenv("LLM_FALLBACK_MODEL", "deepseek-chat"),
            )
            routes.append(ModelRoute(name="fallback", client=fallback, tier="reasoning", priority=1))

        fast_key = os.getenv("LLM_FAST_API_KEY", "").strip() or api_key
        fast_url = os.getenv("LLM_FAST_BASE_URL", "").strip() or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        fast_model = os.getenv("LLM_FAST_MODEL", "").strip()
        if fast_model:
            fast_client = OpenAICompatibleLLMClient(
                api_key=fast_key,
                base_url=fast_url,
                model=fast_model,
            )
            routes.append(ModelRoute(name="fast_primary", client=fast_client, tier="fast", priority=0))
        else:
            routes.append(ModelRoute(name="fast_default", client=primary, tier="fast", priority=0))

        return ModelRouterLLMClient(routes=routes, default_tier="reasoning")

    async def startup(self) -> None:
        if self._initialized:
            return
        await self.plugin_context.mount(_SecurityControlPlanePlugin(self))
        await self.plugin_context.mount(_SandboxAndToolsPlugin(self))
        await self.plugin_context.mount(_SubagentsPlugin(self))
        self._initialized = True

    async def ashutdown(self) -> None:
        await self.plugin_context.shutdown()
        await self.llm.aclose()
        await self.coordination.aclose()
        await self.journal.aclose()
        self.sql_sandbox.close()

    def shutdown(self) -> None:
        self.journal.close()
        self.sql_sandbox.close()
