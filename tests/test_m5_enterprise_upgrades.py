from __future__ import annotations

import asyncio
import json
import os
import struct
from typing import Any

import httpx
import pytest

from pyharness.api.deps import RuntimeContainer
from pyharness.api.main import create_app
from pyharness.infra import RedisCoordinationBackend
from pyharness.rag import HashingSemanticEmbedder, HybridRAGPipeline, QdrantVectorStore
from pyharness.runtime import (
    AgentHarnessEngine,
    ApprovalBroker,
    EventJournal,
    EventType,
    HarnessHook,
    LLMResponse,
    Message,
    OpenAICompatibleLLMClient,
    RiskLevel,
    ScriptedMockLLMClient,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    TurnStatus,
)
from pyharness.sandbox import (
    DockerSandboxProvider,
    E2BSandboxProvider,
    PythonSandbox,
    SQLSafetyError,
    SQLSandbox,
    SandboxConfigurationError,
    SandboxSecurityError,
    SandboxTimeoutError,
    SubprocessSandboxProvider,
    create_sandbox_provider,
)
from pyharness.skills import SkillRouter


# ---------------------------------------------------------------------------
# 1. Sandbox Security Hardening & Pluggable Seam Tests (Docker / E2B / Subprocess / SQL)
# ---------------------------------------------------------------------------


async def test_python_sandbox_blocks_rce_escapes_scrubs_env_and_kills_infinite_loop() -> None:
    os.environ["SUPER_SECRET_API_KEY"] = "sk-never-leak-999"
    sb = PythonSandbox(provider=SubprocessSandboxProvider(), timeout_seconds=0.6)

    # Previously confirmed 0-day RCE & reflection escape vectors must now be 100% blocked!
    exploit_vectors = [
        "import collections\nresult = collections._sys.modules['os'].environ['SUPER_SECRET_API_KEY']",
        "import statistics\nresult = '{0.__globals__[__builtins__]}'.format(statistics.mean)",
        "import re\nresult = re._compile",
        "result = '{x}'.format_map({'x': 1})",
    ]
    for payload in exploit_vectors:
        with pytest.raises(SandboxSecurityError):
            await sb.arun(payload)

    # Infinite loop must be hard-killed by OS process termination without wedging host event loop
    with pytest.raises(SandboxTimeoutError):
        await sb.arun("while True:\n    pass")

    # Legitimate math/collections/datetime execution works in isolated subprocess
    out = json.loads(
        await sb.arun(
            "from collections import Counter\nc = Counter(['a', 'b', 'a'])\nresult = dict(c)\nprint('isolated-ok')"
        )
    )
    assert out["result"] == {"a": 2, "b": 1}
    assert out["stdout"] == "isolated-ok"
    assert out["sandbox_provider"] == "subprocess"
    await sb.aclose()


async def test_docker_sandbox_provider_enforces_hardened_container_spec_and_force_cleanup() -> None:
    captured_create: dict[str, Any] = {}
    deleted_containers: list[str] = []

    def _docker_frame(payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload).encode("utf-8")
        return struct.pack(">BxxxI", 1, len(raw)) + raw

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/containers/create":
            captured_create.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(201, json={"Id": "cnt_sec_001"})
        if path == "/containers/cnt_sec_001/start":
            return httpx.Response(204)
        if path == "/containers/cnt_sec_001/wait":
            return httpx.Response(200, json={"StatusCode": 0})
        if path == "/containers/cnt_sec_001/logs":
            return httpx.Response(
                200,
                content=_docker_frame({"ok": True, "stdout": "docker-log", "result": 42}),
            )
        if path == "/containers/cnt_sec_001" and request.method == "DELETE":
            deleted_containers.append(request.url.params.get("force", ""))
            return httpx.Response(204)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DockerSandboxProvider(
        runtime="runsc",
        http_client=mock_client,
    )
    sb = PythonSandbox(provider=provider)
    raw = json.loads(await sb.arun("result = 6 * 7"))

    assert raw["result"] == 42
    assert raw["stdout"] == "docker-log"
    assert raw["sandbox_provider"] == "docker+runsc"

    # Verify hardened container security policy against CVE-2026-82533 & container breakout
    assert captured_create["NetworkDisabled"] is True
    assert captured_create["User"] == "65534:65534"
    hc = captured_create["HostConfig"]
    assert hc["NetworkMode"] == "none"
    assert hc["ReadonlyRootfs"] is True
    assert hc["CapDrop"] == ["ALL"]
    assert hc["PidsLimit"] == 32
    assert hc["Runtime"] == "runsc"
    assert deleted_containers == ["true"]
    await mock_client.aclose()


async def test_e2b_microvm_provider_and_fail_closed_factory() -> None:
    # Explicit docker/e2b without credentials/daemon must fail closed!
    with pytest.raises(SandboxConfigurationError):
        create_sandbox_provider("e2b", e2b_api_key="")
    with pytest.raises(SandboxConfigurationError):
        create_sandbox_provider("docker", docker_host="")

    lifecycle_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sandboxes":
            lifecycle_calls.append("create")
            assert request.headers["X-API-Key"] == "e2b_test_key"
            return httpx.Response(
                201,
                json={
                    "sandboxID": "vm_firecracker_1",
                    "envdAccessToken": "tok_vm",
                    "execUrl": "https://api.e2b.dev/sandboxes/vm_firecracker_1/exec",
                },
            )
        if request.method == "POST" and request.url.path == "/sandboxes/vm_firecracker_1/exec":
            lifecycle_calls.append("exec")
            return httpx.Response(
                200,
                json={"stdout": json.dumps({"ok": True, "stdout": "vm-out", "result": 3.14})},
            )
        if request.method == "DELETE" and request.url.path == "/sandboxes/vm_firecracker_1":
            lifecycle_calls.append("destroy")
            return httpx.Response(204)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    e2b = E2BSandboxProvider(api_key="e2b_test_key", http_client=mock_client)
    res = await e2b.execute("result = 3.14")
    assert res.result == 3.14
    assert res.provider == "e2b-microvm"
    assert lifecycle_calls == ["create", "exec", "destroy"]
    await mock_client.aclose()


def test_sql_sandbox_authorizer_row_cap_and_recursive_cte_timeout() -> None:
    sb = SQLSandbox(":memory:", max_rows=3, query_timeout_seconds=0.25)
    sb.seed_demo_enterprise_data()

    # 1. Legitimate string literal containing semicolon & SQL keyword is NOT falsely blocked
    out_literal = json.loads(
        sb.query_readonly("SELECT 'safe; DELETE word inside literal' AS note")
    )
    assert out_literal["rows"][0]["note"] == "safe; DELETE word inside literal"

    # 2. CTE LIMIT 1 trick cannot bypass outer max_rows=3 cap
    out_capped = json.loads(
        sb.query_readonly("WITH dummy AS (SELECT 1 LIMIT 1) SELECT * FROM equipment_incidents")
    )
    assert out_capped["row_count"] == 3
    assert out_capped["truncated"] is True

    # 3. Infinite recursive CTE CPU bomb is interrupted cleanly by progress_handler
    with pytest.raises(SQLSafetyError, match="interrupted|timeout"):
        sb.query_readonly(
            "WITH RECURSIVE inf(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM inf) SELECT SUM(x) FROM inf"
        )

    sb.close()


# ---------------------------------------------------------------------------
# 2. Multi-Turn Dynamic Skill Switching, `allowed_tools` Scoping & Reversible Hooks
# ---------------------------------------------------------------------------


async def test_multi_turn_skill_switching_and_allowed_tools_enforcement() -> None:
    from pathlib import Path

    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    router = SkillRouter(skills_dir)

    tools = ToolRegistry()
    tools.register(ToolSpec(name="rag_search", description="RAG"), lambda query="": "doc_hit")
    tools.register(ToolSpec(name="inspect_sql_schema", description="Schema"), lambda: "tables")
    tools.register(
        ToolSpec(name="mcp_dispatch_emergency_alert", description="Alert"),
        lambda incident_id="": "sent",
    )

    llm = ScriptedMockLLMClient(
        [
            # Turn 1 (knowledge_research): LLM only sees allowed_tools
            LLMResponse(content="制度要求低于 0.48 MPa 闭锁 [1]。"),
            # Turn 2 (data_analysis): LLM tries to call unauthorized `mcp_dispatch_emergency_alert`,
            # gets blocked by `tool_scope_denied`, and self-corrects to `inspect_sql_schema`!
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="bad_scope",
                        name="mcp_dispatch_emergency_alert",
                        arguments={"incident_id": "INC-1"},
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[ToolCall(id="ok_scope", name="inspect_sql_schema", arguments={})]
            ),
            LLMResponse(content="已完成 SQL 表结构核查。"),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=tools)

    # Turn 1: Knowledge Research query
    t1_inst, t1_allowed, t1_skills = router.resolve_turn_context("查阅高压断路器排查规范制度")
    assert [s.name for s in t1_skills] == ["knowledge_research"]
    res1 = await engine.run_turn(
        "sess_mt",
        "查阅高压断路器排查规范制度",
        system_prompt="BASE_SYS",
        turn_system_instructions=t1_inst,
        allowed_tools=t1_allowed,
    )
    assert res1.status == TurnStatus.COMPLETED
    # Persistent history[0] MUST remain clean `BASE_SYS`!
    assert res1.messages[0].content == "BASE_SYS"
    assert "Active Skill: knowledge_research" in llm.call_history[0][0].content
    assert set(llm.tools_history[0]) == {"rag_search"}

    # Turn 2: Switch to SQL Data Analysis on the SAME session history!
    t2_inst, t2_allowed, t2_skills = router.resolve_turn_context("写 SQL 统计本月数据库故障数")
    assert [s.name for s in t2_skills] == ["data_analysis"]
    res2 = await engine.run_turn(
        "sess_mt",
        "写 SQL 统计本月数据库故障数",
        history=res1.messages,
        system_prompt="BASE_SYS",
        turn_system_instructions=t2_inst,
        allowed_tools=t2_allowed,
    )
    assert res2.status == TurnStatus.COMPLETED
    # Verify Turn 2 received `data_analysis` instructions (NOT stuck on Turn 1's `knowledge_research`!)
    assert "Active Skill: data_analysis" in llm.call_history[1][0].content
    assert "Active Skill: knowledge_research" not in llm.call_history[1][0].content
    assert set(llm.tools_history[1]) == {"inspect_sql_schema"}
    assert any("Tool Scope Denied" in m.content for m in res2.messages if m.role == "tool")


async def test_reversible_cordis_style_harness_hook() -> None:
    hook_calls: list[str] = []

    class AuditRedactionHook(HarnessHook):
        async def after_tool(
            self, session_id: str, turn_id: str, call: ToolCall, result: ToolResult
        ) -> ToolResult:
            hook_calls.append(call.name)
            return result.model_copy(update={"content": result.content.replace("SECRET", "[REDACTED]")})

    tools = ToolRegistry()
    tools.register(ToolSpec(name="read_secret", description="Read"), lambda: "token=SECRET")
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="read_secret", arguments={})]),
            LLMResponse(content="ok1"),
            LLMResponse(tool_calls=[ToolCall(id="c2", name="read_secret", arguments={})]),
            LLMResponse(content="ok2"),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=tools)
    dispose = engine.add_hook(AuditRedactionHook())

    r1 = await engine.run_turn("s_hook", "run 1")
    assert any("[REDACTED]" in m.content for m in r1.messages if m.role == "tool")
    assert len(hook_calls) == 1

    # Unwind effect cleanly (Cordis reversible plugin unload)
    dispose()
    r2 = await engine.run_turn("s_hook2", "run 2")
    assert any("token=SECRET" in m.content for m in r2.messages if m.role == "tool")
    assert len(hook_calls) == 1


# ---------------------------------------------------------------------------
# 3. True Incremental SSE Streaming (`stream=True`) & Tool Call Delta Assembly
# ---------------------------------------------------------------------------


async def test_openai_compatible_llm_client_true_sse_streaming_and_engine_deltas() -> None:
    sse_Step1 = "\n\n".join(
        [
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"先思考"}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"再调工具"}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_stream_1","function":{"name":"echo_","arguments":""}}]}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"tool","arguments":"{\\"msg\\": "}}]}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"hi\\"}"}}]}}]}',
            'data: {"model":"deepseek-reasoner","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":25,"prompt_cache_hit_tokens":60}}',
            "data: [DONE]",
        ]
    ) + "\n\n"

    sse_Step2 = "\n\n".join(
        [
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"流式"}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"回答"}}]}',
            'data: {"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"完毕！"}}]}',
            'data: {"model":"deepseek-reasoner","choices":[],"usage":{"prompt_tokens":130,"completion_tokens":15,"prompt_cache_hit_tokens":100}}',
            "data: [DONE]",
        ]
    ) + "\n\n"

    call_idx = 0

    def sse_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_idx
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True
        call_idx += 1
        body = sse_Step1 if call_idx == 1 else sse_Step2
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(sse_handler))
    llm = OpenAICompatibleLLMClient(api_key="sk-test", http_client=mock_http)
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            name="echo_tool",
            description="Echo",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        ),
        lambda msg: f"ECHO:{msg}",
    )
    journal = EventJournal(":memory:")
    engine = AgentHarnessEngine(llm=llm, tools=tools, journal=journal)

    res = await engine.run_turn("sess_stream", "测试真流式")
    assert res.status == TurnStatus.COMPLETED
    assert res.final_answer == "流式回答完毕！"
    assert res.usage.prompt_cache_hit_tokens == 160

    events = journal.replay_session("sess_stream")
    thinking_chunks = [e.payload["reasoning"] for e in events if e.event_type == EventType.THINKING]
    token_chunks = [e.payload["content"] for e in events if e.event_type == EventType.TOKEN_DELTA]
    assert thinking_chunks == ["先思考", "再调工具"]
    assert token_chunks == ["流式", "回答", "完毕！"]
    await mock_http.aclose()


# ---------------------------------------------------------------------------
# 4. Distributed Redis Coordination (Cross-Worker Approval, Cancel & Session Lock) & Qdrant Store
# ---------------------------------------------------------------------------


class FakeAsyncPubSub:
    def __init__(self, hub: dict[str, list[asyncio.Queue[str]]]) -> None:
        self._hub = hub
        self._queues: dict[str, asyncio.Queue[str]] = {}

    async def subscribe(self, channel: str) -> None:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._queues[channel] = q
        self._hub.setdefault(channel, []).append(q)

    async def unsubscribe(self, channel: str) -> None:
        q = self._queues.pop(channel, None)
        if q and q in self._hub.get(channel, []):
            self._hub[channel].remove(q)

    async def get_message(
        self, ignore_subscribe_messages: bool = True, timeout: float = 0.05
    ) -> dict[str, Any] | None:
        if not self._queues:
            await asyncio.sleep(timeout)
            return None
        q = next(iter(self._queues.values()))
        try:
            data = await asyncio.wait_for(q.get(), timeout=timeout)
            return {"type": "message", "data": data}
        except asyncio.TimeoutError:
            return None

    async def aclose(self) -> None:
        for ch in list(self._queues):
            await self.unsubscribe(ch)


class FakeAsyncRedis:
    """In-memory wire-equivalent async Redis mock verifying `RedisCoordinationBackend` end-to-end."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.hub: dict[str, list[asyncio.Queue[str]]] = {}

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set(
        self, key: str, value: str, ex: int | None = None, nx: bool = False
    ) -> bool:
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.kv.pop(key, None) is not None else 0

    async def sadd(self, key: str, member: str) -> int:
        s = self.sets.setdefault(key, set())
        before = len(s)
        s.add(member)
        return len(s) - before

    async def srem(self, key: str, member: str) -> int:
        s = self.sets.get(key, set())
        if member in s:
            s.remove(member)
            return 1
        return 0

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    async def publish(self, channel: str, message: str) -> int:
        subs = self.hub.get(channel, [])
        for q in subs:
            q.put_nowait(message)
        return len(subs)

    def pubsub(self) -> FakeAsyncPubSub:
        return FakeAsyncPubSub(self.hub)

    async def aclose(self) -> None:
        return None


async def test_redis_coordination_cross_worker_approval_and_session_lock() -> None:
    shared_redis = FakeAsyncRedis()
    worker_a_coord = RedisCoordinationBackend(redis_client=shared_redis, poll_interval_seconds=0.02)
    worker_b_coord = RedisCoordinationBackend(redis_client=shared_redis, poll_interval_seconds=0.02)

    # 1. Distributed mutual-exclusion lock across Worker A and Worker B
    async with worker_a_coord.session_lock("sess_dist") as locked_a:
        assert locked_a is True
        async with worker_b_coord.session_lock("sess_dist") as locked_b:
            assert locked_b is False

    # 2. Cross-worker Approval: Worker A suspends HIGH-risk tool; Worker B lists & approves it!
    broker_a = ApprovalBroker(coordination=worker_a_coord)
    broker_b = ApprovalBroker(coordination=worker_b_coord)

    tools = ToolRegistry()
    tools.register(
        ToolSpec(name="mutating_op", description="Danger", risk_level=RiskLevel.HIGH),
        lambda: "MUTATED_OK",
    )
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="c_dist", name="mutating_op", arguments={})]),
            LLMResponse(content="Approved across pods!"),
        ]
    )
    engine_a = AgentHarnessEngine(llm=llm, tools=tools, approval_broker=broker_a)

    async def resolve_from_worker_b() -> None:
        for _ in range(50):
            pending = await broker_b.list_pending_async(session_id="sess_pod")
            if pending:
                ok = await broker_b.aresolve(
                    pending[0]["approval_id"],
                    approved=True,
                    reviewer="pod_b_supervisor",
                    note="cross-pod ok",
                )
                assert ok is True
                return
            await asyncio.sleep(0.02)
        raise AssertionError("Worker B never saw pending approval from Worker A")

    b_task = asyncio.create_task(resolve_from_worker_b())
    res = await engine_a.run_turn("sess_pod", "Run high risk op")
    await b_task
    assert res.status == TurnStatus.COMPLETED
    assert "Approved across pods!" in res.final_answer

    # 3. Session history persistence across workers
    await worker_a_coord.save_session_history("sess_pod", res.messages)
    restored = await worker_b_coord.get_session_history("sess_pod")
    assert restored is not None
    assert restored[-1].content == "Approved across pods!"


async def test_fastapi_concurrent_same_session_returns_409_conflict() -> None:
    container = RuntimeContainer()
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        conflict_status: list[int] = []

        async def concurrent_attacker() -> None:
            # Wait until Turn 1 is actively suspended inside HIGH-risk approval (holding session_lock)
            for _ in range(50):
                await asyncio.sleep(0.02)
                pending = (await client.get("/api/v1/approvals?session_id=sess_lock_1")).json()["pending"]
                if pending:
                    dup = await client.post(
                        "/api/v1/chat/stream",
                        json={"session_id": "sess_lock_1", "message": "并发冲突请求"},
                    )
                    conflict_status.append(dup.status_code)
                    cancel_res = await client.post("/api/v1/sessions/sess_lock_1/cancel")
                    assert cancel_res.json()["cancelled"] is True
                    return
            raise AssertionError("Turn 1 never entered pending approval state")

        attacker_task = asyncio.create_task(concurrent_attacker())
        resp1 = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": "sess_lock_1", "message": "立即发送紧急告警通知"},
        )
        await attacker_task
        assert resp1.status_code == 200
        assert conflict_status == [409]
        assert "turn_cancelled" in resp1.text
    await container.ashutdown()


async def test_qdrant_vector_store_persistence_restart_recovery_and_acl_filter() -> None:
    # Stateful in-memory Qdrant REST v1 mock server
    collections: dict[str, dict[str, dict[str, Any]]] = {}

    def qdrant_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/collections/test_col" and request.method == "GET":
            if "test_col" not in collections:
                return httpx.Response(404, json={"status": {"error": "Not found"}})
            return httpx.Response(200, json={"result": {"status": "green"}})
        if path == "/collections/test_col" and request.method == "PUT":
            collections.setdefault("test_col", {})
            return httpx.Response(200, json={"result": True})
        if path == "/collections/test_col/points" and request.method == "PUT":
            body = json.loads(request.content.decode("utf-8"))
            col = collections.setdefault("test_col", {})
            for pt in body["points"]:
                col[pt["id"]] = pt
            return httpx.Response(200, json={"result": {"status": "completed"}})
        if path == "/collections/test_col/points/delete" and request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            doc_id = body["filter"]["must"][0]["match"]["value"]
            col = collections.get("test_col", {})
            for pid in [k for k, v in col.items() if v["payload"].get("doc_id") == doc_id]:
                col.pop(pid, None)
            return httpx.Response(200, json={"result": {"status": "completed"}})
        if path == "/collections/test_col/points/search" and request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            q_vec = body["vector"]
            must = (body.get("filter") or {}).get("must") or []
            col = collections.get("test_col", {})
            scored = []
            for pt in col.values():
                pl = pt["payload"]
                ok = True
                for clause in must:
                    key = clause["key"]
                    val = clause["match"]["value"]
                    actual = (
                        pl["metadata"].get(key.removeprefix("metadata."))
                        if key.startswith("metadata.")
                        else pl.get(key)
                    )
                    if actual != val:
                        ok = False
                        break
                if not ok:
                    continue
                score = sum(a * b for a, b in zip(q_vec, pt["vector"]))
                scored.append({"id": pt["id"], "score": score, "payload": pl})
            scored.sort(key=lambda x: -x["score"])
            return httpx.Response(200, json={"result": scored[: body["limit"]]})
        if path == "/collections/test_col/points/scroll" and request.method == "POST":
            col = collections.get("test_col", {})
            return httpx.Response(
                200,
                json={"result": {"points": list(col.values()), "next_page_offset": None}},
            )
        return httpx.Response(404)

    mock_qdrant_http = httpx.AsyncClient(transport=httpx.MockTransport(qdrant_handler))
    embedder = HashingSemanticEmbedder()

    # Instance 1: Ingest two tenant-scoped docs into Qdrant
    store1 = QdrantVectorStore(embedder=embedder, collection_name="test_col", http_client=mock_qdrant_http)
    rag1 = HybridRAGPipeline(embedder=embedder, vector_store=store1)
    await rag1.ingest_markdown(
        "## 金山厂区规程\n反应釜最高工作压力为 4.2 MPa。",
        source="jinshan.md",
        doc_id="d_js",
        metadata={"tenant": "jinshan"},
    )
    await rag1.ingest_markdown(
        "## 张江厂区规程\n反应釜最高工作压力为 1.6 MPa。",
        source="zhangjiang.md",
        doc_id="d_zj",
        metadata={"tenant": "zhangjiang"},
    )
    # Re-ingest d_js idempotently -> point count must NOT duplicate!
    count_before = len(collections["test_col"])
    await rag1.ingest_markdown(
        "## 金山厂区规程\n反应釜最高工作压力为 4.2 MPa。",
        source="jinshan.md",
        doc_id="d_js",
        metadata={"tenant": "jinshan"},
    )
    assert len(collections["test_col"]) == count_before

    # Instance 2: Cold restart -> restore corpus from Qdrant & query with tenant ACL filter!
    store2 = QdrantVectorStore(embedder=embedder, collection_name="test_col", http_client=mock_qdrant_http)
    rag2 = HybridRAGPipeline(embedder=embedder, vector_store=store2)
    restored = await rag2.restore_from_store()
    assert restored["children_restored"] == 2

    hits_zj = await rag2.search(
        "反应釜最高工作压力是多少",
        top_k=2,
        metadata_filter={"tenant": "zhangjiang"},
    )
    assert len(hits_zj) == 1
    assert "1.6 MPa" in hits_zj[0].parent_context
    assert "4.2 MPa" not in hits_zj[0].parent_context
    await mock_qdrant_http.aclose()
