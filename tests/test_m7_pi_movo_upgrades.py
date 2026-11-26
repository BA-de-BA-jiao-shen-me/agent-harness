from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from pyharness.api.deps import RuntimeContainer
from pyharness.api.main import create_app
from pyharness.runtime import (
    AgentHarnessEngine,
    ArtifactAuthorizationError,
    BaseLLMClient,
    EventJournal,
    EventType,
    FileArtifactStore,
    HarnessConfig,
    InMemoryArtifactStore,
    LLMResponse,
    Message,
    ModelRoute,
    ModelRouterLLMClient,
    ProjectMemoryStore,
    RuntimeIdleReaper,
    S3ArtifactStore,
    ScheduledTaskManager,
    ScheduledTaskSpec,
    ScriptedMockLLMClient,
    SessionAdmissionController,
    ToolCall,
    ToolOutputOffloadHook,
    ToolRegistry,
    ToolSpec,
    TopicAdmissionGuard,
    TopicAdmissionRule,
    TrustedTemporalContext,
    TurnStatus,
    TurnSteeringQueue,
    UserQuestionBroker,
    create_artifact_store,
    cron_matches,
    register_artifact_tools,
)
from pyharness.security import (
    AuthManager,
    CredentialRelay,
    CredentialRelayError,
    CredentialTarget,
    GuardrailHook,
    Principal,
    use_principal,
)
from pyharness.skills import SkillBundle, SkillRouter


# ---------------------------------------------------------------------------
# 1. KV Prefix Cache Stability & Length-Truncation Auto-Recovery (Pi + DSH)
# ---------------------------------------------------------------------------


async def test_kv_prefix_cache_stability_across_skill_switches_and_length_recovery() -> None:
    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    router = SkillRouter(skills_dir)

    tools = ToolRegistry()
    # Register intentionally out of alphabetical order to verify deterministic sorting!
    tools.register(
        ToolSpec(name="mcp_dispatch_emergency_alert", description="Alert"),
        lambda incident_id="": "sent",
    )
    tools.register(ToolSpec(name="rag_search", description="RAG"), lambda query="": "doc_hit")
    tools.register(ToolSpec(name="inspect_sql_schema", description="Schema"), lambda: "tables")

    llm = ScriptedMockLLMClient(
        [
            # Turn 1 (knowledge_research)
            LLMResponse(content="制度要求低于 0.48 MPa 闭锁 [1]。"),
            # Turn 2 (data_analysis): attempts out-of-scope tool -> blocked at runtime -> self-corrects -> truncated -> recovered!
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="scope_bad",
                        name="mcp_dispatch_emergency_alert",
                        arguments={"incident_id": "INC-99"},
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[ToolCall(id="scope_ok", name="inspect_sql_schema", arguments={})]
            ),
            LLMResponse(content="第一部分：表结构核查完成，", finish_reason="length"),
            LLMResponse(content="第二部分：共发现 6 条故障记录。", finish_reason="stop"),
        ]
    )
    # Default HarnessConfig uses preserve_prompt_cache=True & enable_length_recovery=True!
    engine = AgentHarnessEngine(
        llm=llm,
        tools=tools,
        config=HarnessConfig(max_steps=3),
    )

    t1_inst, t1_allowed, _ = router.resolve_turn_context("查阅高压断路器排查规范制度")
    res1 = await engine.run_turn(
        "sess_kv",
        "查阅高压断路器排查规范制度",
        system_prompt="STATIC_BASE_SYSTEM_PROMPT",
        turn_system_instructions=t1_inst,
        allowed_tools=t1_allowed,
        timezone_name="Asia/Shanghai",
    )
    assert res1.status == TurnStatus.COMPLETED

    t2_inst, t2_allowed, _ = router.resolve_turn_context("写 SQL 统计本月数据库故障数")
    res2 = await engine.run_turn(
        "sess_kv",
        "写 SQL 统计本月数据库故障数",
        history=res1.messages,
        system_prompt="STATIC_BASE_SYSTEM_PROMPT",
        turn_system_instructions=t2_inst,
        allowed_tools=t2_allowed,
        timezone_name="Asia/Shanghai",
    )
    assert res2.status == TurnStatus.COMPLETED

    # 1. System prompt prefix `messages[0]` MUST stay 100% byte-identical across all LLM calls!
    for sent_msgs in llm.call_history:
        assert sent_msgs[0].role == "system"
        assert sent_msgs[0].content == "STATIC_BASE_SYSTEM_PROMPT"

    # 2. Tools list MUST stay 100% identical & alphabetically sorted across skill switches!
    expected_sorted_tools = ["inspect_sql_schema", "mcp_dispatch_emergency_alert", "rag_search"]
    for sent_tools in llm.tools_history:
        assert sent_tools == expected_sorted_tools

    # 3. `prompt_cache_key` MUST be consistently passed for session prefix affinity!
    assert all(k == "pyharness:default:sess_kv" for k in llm.cache_keys_history)

    # 4. Dynamic Skill + Trusted Clock appear in transient tail user message, NOT in persisted history!
    assert "Active Skill: knowledge_research" in llm.call_history[0][-1].content
    assert "Trusted Server Temporal Context" in llm.call_history[0][-1].content
    assert "Active Skill: data_analysis" in llm.call_history[1][-1].content
    assert "Active Skill: knowledge_research" not in llm.call_history[1][-1].content
    assert res2.messages[0].content == "STATIC_BASE_SYSTEM_PROMPT"
    assert "[Turn Dynamic Context]" not in res2.messages[1].content

    # 5. Unauthorized tool call was STILL hard-blocked by runtime tool_scope_denied!
    assert any("Tool Scope Denied" in m.content for m in res2.messages if m.role == "tool")

    # 6. Length truncation on step 3 auto-recovered (despite max_steps=3!) and stitched full answer!
    assert "第一部分：表结构核查完成，" in res2.final_answer
    assert "第二部分：共发现 6 条故障记录。" in res2.final_answer


# ---------------------------------------------------------------------------
# 2. Tool Output Offloading (DLP-Before-Offload), File & S3/MinIO Seam + ACL
# ---------------------------------------------------------------------------


async def test_tool_output_offloading_dlp_before_store_and_s3_minio_seam(tmp_path: Path) -> None:
    # A. Verify FileArtifactStore + path traversal guard
    file_store = FileArtifactStore(tmp_path / "artifacts")
    rec_f = await file_store.save(
        "A" * 5000,
        session_id="s_file",
        turn_id="t1",
        tenant_id="tenant_alpha",
        tool_name="big_sql",
    )
    _, chunk, has_more = await file_store.read_slice(
        rec_f.artifact_id, offset=0, limit=1000, tenant_id="tenant_alpha"
    )
    assert len(chunk) == 1000 and has_more is True
    with pytest.raises(ArtifactAuthorizationError):
        await file_store.read_slice(rec_f.artifact_id, tenant_id="tenant_beta")
    with pytest.raises(ValueError):
        await file_store.read_slice("../escape", tenant_id="tenant_alpha")

    # B. Verify S3ArtifactStore (MinIO / AWS S3 Seam) with mock S3 client + lazy boto3 error
    s3_objects: dict[str, bytes] = {}

    class FakeS3Client:
        def put_object(self, Bucket: str, Key: str, Body: bytes, **_kw: Any) -> dict[str, Any]:
            s3_objects[f"{Bucket}/{Key}"] = Body
            return {"ETag": "mock-etag"}

        def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
            full = f"{Bucket}/{Key}"
            if full not in s3_objects:
                raise KeyError(full)

            class _Body:
                def read(self) -> bytes:
                    return s3_objects[full]

            return {"Body": _Body()}

        def list_objects_v2(self, Bucket: str, Prefix: str = "", **_kw: Any) -> dict[str, Any]:
            prefix_full = f"{Bucket}/{Prefix}"
            contents = [
                {"Key": k.removeprefix(f"{Bucket}/")}
                for k in sorted(s3_objects.keys())
                if k.startswith(prefix_full)
            ]
            return {"Contents": contents, "IsTruncated": False}

        def delete_object(self, Bucket: str, Key: str) -> dict[str, Any]:
            s3_objects.pop(f"{Bucket}/{Key}", None)
            return {}

    s3_store = create_artifact_store(
        mode="s3",
        s3_bucket="enterprise-artifacts",
        s3_endpoint_url="http://minio.internal:9000",
        s3_client=FakeS3Client(),
    )
    assert isinstance(s3_store, S3ArtifactStore)
    s3_rec = await s3_store.save(
        "S3-PAYLOAD-" + ("Z" * 3000),
        session_id="s_s3",
        turn_id="t1",
        tenant_id="plant_a",
        tool_name="export_report",
    )
    assert s3_rec.storage_uri.startswith("s3://enterprise-artifacts/")
    _, s3_chunk, s3_more = await s3_store.read_slice(
        s3_rec.artifact_id, offset=0, limit=11, tenant_id="plant_a"
    )
    assert s3_chunk == "S3-PAYLOAD-" and s3_more is True
    assert len(await s3_store.list_for_session("s_s3", tenant_id="plant_a")) == 1
    assert len(await s3_store.list_for_session("s_s3", tenant_id="plant_b")) == 0
    with pytest.raises(ArtifactAuthorizationError):
        await s3_store.read_slice(s3_rec.artifact_id, tenant_id="plant_b")
    assert await s3_store.delete(s3_rec.artifact_id) is True

    # C. End-to-end Engine: GuardrailHook (DLP) runs BEFORE ToolOutputOffloadHook!
    mem_store = InMemoryArtifactStore()
    tools = ToolRegistry()
    tools.register(
        ToolSpec(name="dump_logs", description="Big output"),
        lambda: "HEAD_MARKER sk-live-supersecret999999 " + ("X" * 5000) + " TAIL_MARKER",
    )
    register_artifact_tools(tools, mem_store)

    journal = EventJournal(":memory:")
    created_ids: list[str] = []

    async def on_art(sid: str, tid: str, rec: Any) -> None:
        created_ids.append(rec.artifact_id)
        await journal.emit(sid, tid, EventType.ARTIFACT_CREATED, {"artifact_id": rec.artifact_id})

    llm = ScriptedMockLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="big1", name="dump_logs", arguments={})]),
            lambda msgs: LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="read1",
                        name="read_artifact",
                        arguments={"artifact_id": created_ids[0], "offset": 0, "limit": 120},
                    )
                ]
            ),
            LLMResponse(content="已安全卸载并分页读取日志。"),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=tools, journal=journal)
    engine.add_hook(GuardrailHook())
    engine.add_hook(
        ToolOutputOffloadHook(mem_store, max_inline_chars=500, head_chars=120, tail_chars=80, on_artifact_created=on_art)
    )

    alice = Principal(user_id="alice", tenant_id="t_alpha", roles=frozenset({"member"}))
    with use_principal(alice):
        res = await engine.run_turn("sess_art", "导出大日志")

    assert res.status == TurnStatus.COMPLETED
    assert len(created_ids) == 1
    first_tool_msg = next(m for m in res.messages if m.role == "tool" and m.name == "dump_logs")
    assert "Artifact Offloaded:" in first_tool_msg.content
    assert "sk-live-supersecret999999" not in first_tool_msg.content
    # Even the stored artifact blob itself MUST have the raw secret redacted!
    _, raw_stored, _ = await mem_store.read_slice(created_ids[0], offset=0, limit=10000, tenant_id="t_alpha")
    assert "sk-live-supersecret999999" not in raw_stored
    assert "[REDACTED_API_KEY]" in raw_stored
    assert "TAIL_MARKER" in raw_stored


# ---------------------------------------------------------------------------
# 3. Mid-Turn Steering (`steer`), Post-Turn Follow-Up (`followUp`) & Queue Modes
# ---------------------------------------------------------------------------


async def test_mid_turn_steering_skips_queued_tools_and_followup_chains_seamlessly() -> None:
    steering = TurnSteeringQueue()
    journal = EventJournal(":memory:")
    executed: list[str] = []

    async def slow_first_tool() -> str:
        executed.append("tool_1")
        # Simulate user injecting a mid-turn steering directive while tool_1 is running!
        steering.enqueue_steer("sess_steer", "改查上海厂区，不要查北京厂区！")
        await asyncio.sleep(0.02)
        return "tool_1_done"

    async def second_tool() -> str:
        executed.append("tool_2")
        return "tool_2_done"

    tools = ToolRegistry()
    tools.register(ToolSpec(name="step_one", description="Step 1"), slow_first_tool)
    tools.register(ToolSpec(name="step_two", description="Step 2"), second_tool)

    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="step_one", arguments={}),
                    ToolCall(id="c2", name="step_two", arguments={}),
                ]
            ),
            # After steering, LLM produces steered answer; meanwhile a follow-up is queued!
            lambda msgs: (
                steering.enqueue_followup("sess_steer", "顺便总结停机原因")
                and False
                or LLMResponse(content="已立即转向上海厂区核查完成。")
            ),
            LLMResponse(content="追加总结：停机主因为阀门过热。"),
        ]
    )
    # max_parallel_tools=1 ensures step_two waits on semaphore while step_one runs and triggers steer!
    engine = AgentHarnessEngine(
        llm=llm,
        tools=tools,
        journal=journal,
        config=HarnessConfig(max_parallel_tools=1),
        steering_queue=steering,
    )

    res = await engine.run_turn("sess_steer", "先查北京厂区")
    assert res.status == TurnStatus.COMPLETED
    # tool_2 MUST be skipped due to mid-turn steering!
    assert executed == ["tool_1"]
    tool2_msg = next(m for m in res.messages if m.role == "tool" and m.name == "step_two")
    assert "Skipped due to user mid-turn steering" in tool2_msg.content
    assert "停机主因为阀门过热" in res.final_answer

    evts = journal.replay_session("sess_steer")
    assert any(e.event_type == EventType.TURN_STEERED for e in evts)
    assert any(e.event_type == EventType.FOLLOWUP_ENQUEUED for e in evts)


async def test_api_chat_queue_modes_and_steer_followup_endpoints() -> None:
    gate = asyncio.Event()

    class SlowLLM(BaseLLMClient):
        async def complete(self, messages, tools=None, *, prompt_cache_key=None) -> LLMResponse:
            await gate.wait()
            return LLMResponse(content="slow-complete")

    container = RuntimeContainer(llm=SlowLLM())
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_req = asyncio.create_task(
            client.post(
                "/api/v1/chat/stream",
                json={"session_id": "s_qmode", "message": "initial task"},
            )
        )
        await asyncio.sleep(0.05)

        # Default queue_mode="reject" MUST still return 409 Conflict!
        r_409 = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": "s_qmode", "message": "dup"},
        )
        assert r_409.status_code == 409

        # queue_mode="steer" returns 202 Accepted and enqueues mid-turn steer!
        r_202 = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": "s_qmode", "message": "pivot now", "queue_mode": "steer"},
        )
        assert r_202.status_code == 202
        assert r_202.json()["mode"] == "steer"

        # Dedicated /followup endpoint also enqueues cleanly
        r_fu = await client.post(
            "/api/v1/sessions/s_qmode/followup",
            json={"message": "next task"},
        )
        assert r_fu.status_code == 200
        assert r_fu.json()["pending"]["followup"] == 1

        gate.set()
        r_first = await first_req
        assert r_first.status_code == 200
        assert "turn_steered" in r_first.text
        assert "followup_enqueued" in r_first.text

    await container.ashutdown()


# ---------------------------------------------------------------------------
# 4. Interactive Clarification Broker (`ask_user_question`) & Owner RBAC (MOVO)
# ---------------------------------------------------------------------------


async def test_interactive_ask_user_question_suspend_owner_rbac_and_timeout() -> None:
    alice = Principal(user_id="alice", tenant_id="plant_a", roles=frozenset({"member"}))
    bob_same_tenant = Principal(user_id="bob", tenant_id="plant_a", roles=frozenset({"member"}))
    eve_other_tenant = Principal(user_id="eve", tenant_id="plant_b", roles=frozenset({"member"}))

    auth = AuthManager(
        mode="enforce",
        jwt_secret="q-broker-secret",
        api_keys={"k_alice": alice, "k_bob": bob_same_tenant, "k_eve": eve_other_tenant},
    )
    container = RuntimeContainer(auth_manager=auth)
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        statuses: dict[str, int] = {}

        async def answer_flow() -> None:
            for _ in range(50):
                await asyncio.sleep(0.02)
                r_q = await client.get(
                    "/api/v1/sessions/s_clarify/questions",
                    headers={"X-API-Key": "k_alice"},
                )
                pending = r_q.json().get("pending", [])
                if pending:
                    qid = pending[0]["question_id"]
                    # Cross-tenant Eve rejected (403)
                    r_eve = await client.post(
                        f"/api/v1/sessions/s_clarify/questions/{qid}/answer",
                        headers={"X-API-Key": "k_eve"},
                        json={"selected_options": ["opt_1"]},
                    )
                    statuses["eve"] = r_eve.status_code

                    # Non-owner Bob rejected (403)
                    r_bob = await client.post(
                        f"/api/v1/sessions/s_clarify/questions/{qid}/answer",
                        headers={"X-API-Key": "k_bob"},
                        json={"selected_options": ["opt_1"]},
                    )
                    statuses["bob"] = r_bob.status_code

                    # Owner Alice submits structured choice + free text (200)
                    r_ok = await client.post(
                        f"/api/v1/sessions/s_clarify/questions/{qid}/answer",
                        headers={"X-API-Key": "k_alice"},
                        json={"selected_options": ["opt_2"], "free_text": "仅看金山A区本周数据"},
                    )
                    statuses["alice"] = r_ok.status_code
                    return
            raise AssertionError("Pending question never appeared")

        flow_task = asyncio.create_task(answer_flow())
        r_chat = await client.post(
            "/api/v1/chat/stream",
            headers={"X-API-Key": "k_alice"},
            json={"session_id": "s_clarify", "message": "请帮我澄清后再统计设备隐患"},
        )
        await flow_task

        assert r_chat.status_code == 200
        assert statuses == {"eve": 403, "bob": 403, "alice": 200}
        assert "question_required" in r_chat.text
        assert "question_answered" in r_chat.text
        assert "仅看金山A区本周数据" in r_chat.text

    # Also verify graceful timeout without hanging
    broker = UserQuestionBroker(default_timeout_seconds=0.05)
    req = broker.create_question(session_id="s_to", turn_id="t1", question="Timeout?")
    ans = await broker.wait_for_answer(req, timeout_seconds=0.05)
    assert ans.timed_out is True
    assert "Timed Out" in ans.free_text

    await container.ashutdown()


# ---------------------------------------------------------------------------
# 5. In-Place Session Tree Branching & Compaction Anchor Cold-Restart Replay (Pi)
# ---------------------------------------------------------------------------


async def test_in_place_session_tree_branching_and_compaction_anchor_replay() -> None:
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(content="主干回答1：初始方案A。"),
            LLMResponse(content="主干回答2：沿着方案A深化。"),
            LLMResponse(content="分支B回答：改用方案B独立探索。"),
        ]
    )
    container = RuntimeContainer(llm=llm)
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Turn 1 on main
        assert (
            await client.post(
                "/api/v1/chat/stream",
                json={"session_id": "s_tree", "message": "第一轮问题"},
            )
        ).status_code == 200

        # Capture Turn 1 completion event_id as branch point
        evts_after_t1 = (await client.get("/api/v1/sessions/s_tree/events")).json()["events"]
        branch_anchor_id = evts_after_t1[-1]["event_id"]

        # Turn 2 on main
        assert (
            await client.post(
                "/api/v1/chat/stream",
                json={"session_id": "s_tree", "message": "主干第二轮"},
            )
        ).status_code == 200

        # Create zero-copy branch `alt_b` anchored at Turn 1 head and switch to it
        r_br = await client.post(
            "/api/v1/sessions/s_tree/branches",
            json={"branch_name": "alt_b", "from_event_id": branch_anchor_id, "switch": True},
        )
        assert r_br.status_code == 200
        assert r_br.json()["branch_name"] == "alt_b"

        # Turn 3 on branch `alt_b`
        assert (
            await client.post(
                "/api/v1/chat/stream",
                json={"session_id": "s_tree", "message": "分支B新探索"},
            )
        ).status_code == 200

        # Inspect tree topology
        r_tree = await client.get("/api/v1/sessions/s_tree/tree")
        assert r_tree.status_code == 200
        tree = r_tree.json()
        assert tree["active_branch"] == "alt_b"
        assert {b["branch_name"] for b in tree["branches"]} == {"main", "alt_b"}

        # Reconstructed messages on `alt_b` MUST NOT include `主干第二轮`!
        alt_msgs = await container.journal.areconstruct_branch_messages("s_tree", "SYS", "alt_b")
        assert alt_msgs is not None
        alt_text = "\n".join(m.content for m in alt_msgs)
        assert "第一轮问题" in alt_text
        assert "分支B新探索" in alt_text
        assert "主干第二轮" not in alt_text

        # Switch back to `main` -> reconstructed messages MUST NOT include `分支B新探索`!
        r_sw = await client.post("/api/v1/sessions/s_tree/branches/main/switch")
        assert r_sw.status_code == 200
        main_msgs = await container.journal.areconstruct_branch_messages("s_tree", "SYS", "main")
        assert main_msgs is not None
        main_text = "\n".join(m.content for m in main_msgs)
        assert "主干第二轮" in main_text
        assert "分支B新探索" not in main_text

        # Cryptographic SHA-256 audit hash chain MUST remain 100% intact across all branches!
        r_audit = await client.get("/api/v1/sessions/s_tree/audit/verify")
        assert r_audit.json()["valid"] is True

    # Also verify non-destructive COMPACTION_ANCHOR cold-restart reconstruction
    j = EventJournal(":memory:")
    await j.emit("s_cmp", "t1", EventType.TURN_START, {"user_input": "旧历史问题1"})
    await j.emit("s_cmp", "t1", EventType.TOKEN_DELTA, {"content": "旧历史回答1"})
    await j.emit("s_cmp", "t1", EventType.TURN_COMPLETE, {"status": "completed"})
    await j.emit(
        "s_cmp",
        "t2",
        EventType.COMPACTION_ANCHOR,
        {"summary": "- 核心结论：断路器闭锁阈值 0.48 MPa"},
    )
    await j.emit("s_cmp", "t3", EventType.TURN_START, {"user_input": "压缩后的新问题"})
    await j.emit("s_cmp", "t3", EventType.TOKEN_DELTA, {"content": "基于锚点的新回答"})
    await j.emit("s_cmp", "t3", EventType.TURN_COMPLETE, {"status": "completed"})

    reconstructed = await j.areconstruct_messages("s_cmp", "SYS")
    assert reconstructed is not None
    joined = "\n".join(m.content for m in reconstructed)
    assert "断路器闭锁阈值 0.48 MPa" in joined
    assert "压缩后的新问题" in joined
    assert "旧历史问题1" not in joined
    # Raw pre-compaction audit events remain preserved in full journal!
    assert len(j.replay_session("s_cmp")) == 7
    assert (await j.verify_audit_chain("s_cmp")).valid is True

    await container.ashutdown()


# ---------------------------------------------------------------------------
# 6. Cross-Session Project Memory & Trusted Server Temporal Context (MOVO)
# ---------------------------------------------------------------------------


async def test_cross_session_project_memory_and_trusted_temporal_context() -> None:
    # A. TrustedTemporalContext validation & invalid timezone fallback
    fixed_utc = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)
    cn_ctx = TrustedTemporalContext.capture("Asia/Shanghai", now_utc=fixed_utc)
    assert cn_ctx.user_timezone == "Asia/Shanghai"
    assert cn_ctx.user_date == "2026-11-26"
    assert "18:00:00" in cn_ctx.user_local_time
    assert cn_ctx.tz_fallback_used is False

    bad_ctx = TrustedTemporalContext.capture("Mars/Olympus_Mons", now_utc=fixed_utc)
    assert bad_ctx.user_timezone == "UTC"
    assert bad_ctx.tz_fallback_used is True

    # B. Cross-session Project Memory recall & strict tenant/user isolation
    alice = Principal(user_id="alice", tenant_id="corp_a", roles=frozenset({"member"}))
    eve = Principal(user_id="eve", tenant_id="corp_b", roles=frozenset({"member"}))
    auth = AuthManager(mode="enforce", jwt_secret="mem-secret", api_keys={"ka": alice, "ke": eve})

    mock_llm = ScriptedMockLLMClient(
        [
            LLMResponse(content="会话B已读取长期记忆并完成回复。"),
        ]
    )
    container = RuntimeContainer(auth_manager=auth, llm=mock_llm)
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Alice saves a project-scoped memory in REST API
        r_save = await client.post(
            "/api/v1/memories",
            headers={"X-API-Key": "ka"},
            json={
                "content": "金山石化项目统一使用 MPa 单位并保留两位小数",
                "project_id": "jinshan_ops",
                "scope": "project",
                "category": "preference",
                "tags": ["金山石化", "单位"],
            },
        )
        assert r_save.status_code == 200
        mem_id = r_save.json()["memory_id"]

        # Eve (other tenant) CANNOT read or delete Alice's memory!
        r_eve_list = await client.get(
            "/api/v1/memories?project_id=jinshan_ops",
            headers={"X-API-Key": "ke"},
        )
        assert r_eve_list.json()["memories"] == []
        r_eve_del = await client.delete(
            f"/api/v1/memories/{mem_id}",
            headers={"X-API-Key": "ke"},
        )
        assert r_eve_del.status_code == 403

        # Alice starts a BRAND-NEW session B on the same project -> memory is auto-recalled into LLM context!
        r_chat = await client.post(
            "/api/v1/chat/stream",
            headers={"X-API-Key": "ka"},
            json={
                "session_id": "sess_brand_new_b",
                "message": "汇报金山石化压力表读数",
                "project_id": "jinshan_ops",
                "timezone": "Asia/Shanghai",
            },
        )
        assert r_chat.status_code == 200
        sent_tail = mock_llm.call_history[0][-1].content
        assert "Recalled Long-Term Memory" in sent_tail
        assert "MPa 单位并保留两位小数" in sent_tail
        assert "Asia/Shanghai" in sent_tail

    await container.ashutdown()


# ---------------------------------------------------------------------------
# 7. Zero-Trust Credential Relay (SSRF), Session Admission, Topic Guard & Reaper
# ---------------------------------------------------------------------------


async def test_zero_trust_credential_relay_admission_topic_guard_and_idle_reaper() -> None:
    # A. Zero-Trust Credential Relay + SSRF Protection
    seen_auth_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization", ""))
        # Simulate an upstream that carelessly echoes the token back!
        return httpx.Response(
            200,
            text=f"ok echoed={request.headers.get('authorization', '')}",
            headers={"Set-Cookie": "sid=secret", "X-Upstream": "erp"},
        )

    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CredentialRelay(http_client=mock_http)
    relay.register_secret(
        CredentialTarget(
            name="erp_api",
            secret_value="erp-super-secret-token-777",
            allowed_hosts=["api.erp.example.com", "*.trusted.example.com"],
            tenant_id="plant_a",
        )
    )
    assert "erp-super-secret-token-777" not in json.dumps(relay.list_targets())

    # Cross-tenant lease issuance rejected
    with pytest.raises(CredentialRelayError):
        relay.issue_lease("erp_api", tenant_id="plant_b")

    lease = relay.issue_lease("erp_api", ttl_seconds=60.0, max_calls=1, tenant_id="plant_a")
    resp = await relay.proxy_request(
        lease_id=lease.lease_id,
        lease_token=lease.lease_token,
        method="GET",
        url="https://api.erp.example.com/v1/orders",
        headers={"Authorization": "Bearer attacker-spoof", "X-Trace": "t1"},
    )
    assert resp["status_code"] == 200
    assert seen_auth_headers == ["Bearer erp-super-secret-token-777"]
    # Secret echoed by upstream MUST be scrubbed and Set-Cookie stripped!
    assert "erp-super-secret-token-777" not in resp["text"]
    assert "[REDACTED_RELAY_SECRET]" in resp["text"]
    assert "set-cookie" not in {k.lower() for k in resp["headers"]}
    assert resp["lease_calls_remaining"] == 0

    # Budget exhausted -> blocked!
    with pytest.raises(CredentialRelayError, match="budget"):
        await relay.proxy_request(
            lease_id=lease.lease_id,
            lease_token=lease.lease_token,
            method="GET",
            url="https://api.erp.example.com/v1/orders",
        )

    # Fresh lease tests SSRF, HTTP downgrade, non-allowlisted host & wrong token
    lease2 = relay.issue_lease("erp_api", ttl_seconds=60.0, max_calls=10, tenant_id="plant_a")
    for bad_url in (
        "http://api.erp.example.com/insecure",
        "https://evil.attacker.com/steal",
        "https://127.0.0.1/admin",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::ffff:127.0.0.1]/admin",
    ):
        with pytest.raises(CredentialRelayError):
            await relay.proxy_request(
                lease_id=lease2.lease_id,
                lease_token=lease2.lease_token,
                method="GET",
                url=bad_url,
            )

    with pytest.raises(CredentialRelayError, match="token"):
        await relay.proxy_request(
            lease_id=lease2.lease_id,
            lease_token="wrong-token",
            method="GET",
            url="https://api.erp.example.com/v1/orders",
        )
    await mock_http.aclose()

    # B. Topic Admission Guard + Session Admission Control (503) via FastAPI
    topic_guard = TopicAdmissionGuard(
        rules=[
            TopicAdmissionRule(
                rule_id="allow_safety_override",
                action="allow",
                keywords=["断路器"],
                reason="Core safety ops",
            ),
            TopicAdmissionRule(
                rule_id="block_personal_stock",
                action="block",
                keywords=["炒股", "股票推荐"],
                reason="Non-business personal finance request",
                redirect_message="本助手仅支持企业运维与业务分析，不提供个人股票投资建议。",
            ),
        ]
    )
    gate = asyncio.Event()

    class BlockingLLM(BaseLLMClient):
        async def complete(self, messages, tools=None, *, prompt_cache_key=None) -> LLMResponse:
            await gate.wait()
            return LLMResponse(content="ok")

    container = RuntimeContainer(
        llm=BlockingLLM(),
        topic_guard=topic_guard,
        admission_controller=SessionAdmissionController(
            max_global_active_turns=1, max_tenant_active_turns=1, retry_after_seconds=3
        ),
    )
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Out-of-scope prompt blocked with 403 BEFORE LLM!
        r_blocked = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": "s_topic", "message": "给我做一个明天炒股股票推荐"},
        )
        assert r_blocked.status_code == 403
        assert r_blocked.json()["detail"]["rule_id"] == "block_personal_stock"

        # Concurrency saturation -> second distinct session gets 503 + Retry-After!
        t1 = asyncio.create_task(
            client.post("/api/v1/chat/stream", json={"session_id": "s_adm_1", "message": "查断路器"})
        )
        await asyncio.sleep(0.05)
        r_503 = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": "s_adm_2", "message": "查断路器"},
        )
        assert r_503.status_code == 503
        assert r_503.headers["retry-after"] == "3"
        gate.set()
        assert (await t1).status_code == 200

    # C. RuntimeIdleReaper evicts idle histories, steering buckets & expired leases while sparing active ones
    histories: dict[str, Any] = {"s_idle": ["m1"], "s_active": ["m2"]}
    sq = TurnSteeringQueue()
    sq.enqueue_steer("s_idle", "stale")
    sq._last_activity["s_idle"] = time.time() - 500
    exp_lease = relay.issue_lease("erp_api", ttl_seconds=-1.0, tenant_id="plant_a")
    assert exp_lease.lease_id in relay._leases

    reaper = RuntimeIdleReaper(
        histories,
        idle_ttl_seconds=60.0,
        steering_queue=sq,
        credential_relay=relay,
        active_sessions_getter=lambda: {"s_active"},
    )
    now = time.time()
    reaper.touch("s_idle", now=now - 120.0)
    reaper.touch("s_active", now=now - 120.0)
    stats = reaper.reap_once(now=now)
    assert stats["sessions_evicted"] == 1
    assert "s_idle" not in histories and "s_active" in histories
    assert stats["steering_evicted"] >= 1
    assert stats["leases_evicted"] >= 1

    await container.ashutdown()


# ---------------------------------------------------------------------------
# 8. Skill Lifecycle/Readiness, Weighted Canary Router & Scheduled Tasks (MOVO)
# ---------------------------------------------------------------------------


async def test_skill_lifecycle_readiness_weighted_router_and_scheduled_tasks() -> None:
    # A. Skill lifecycle (draft / published / deprecated) + readiness check + stats
    router = SkillRouter()
    router.register_skill(
        SkillBundle(
            name="draft_audit",
            description="Draft skill",
            instructions="Draft instructions",
            triggers=["合规审查"],
            allowed_tools=["existing_tool", "missing_tool_xyz"],
            status="draft",
        )
    )
    router.register_skill(
        SkillBundle(
            name="old_legacy",
            description="Deprecated skill",
            instructions="Old",
            triggers=["合规审查"],
            allowed_tools=["existing_tool"],
            status="deprecated",
        )
    )

    # Default selection MUST skip both draft and deprecated skills
    assert router.select_for_query("执行合规审查") == []
    assert [s.name for s in router.select_for_query("执行合规审查", include_drafts=True)] == [
        "draft_audit"
    ]

    reg = ToolRegistry()
    reg.register(ToolSpec(name="existing_tool", description="Exists"), lambda: "ok")
    reports = {r.skill_name: r for r in router.check_readiness(reg)}
    assert reports["draft_audit"].ready is False
    assert reports["draft_audit"].missing_tools == ["missing_tool_xyz"]
    assert reports["old_legacy"].ready is False

    # Promote draft -> published and register missing tool -> ready & selected!
    router.set_status("draft_audit", "published")
    reg.register(ToolSpec(name="missing_tool_xyz", description="Now exists"), lambda: "ok")
    assert {r.skill_name: r for r in router.check_readiness(reg)}["draft_audit"].ready is True
    _, _, selected = router.resolve_turn_context("执行合规审查")
    assert [s.name for s in selected] == ["draft_audit"]
    assert router.get_invocation_stats()["draft_audit"]["invocations"] == 1

    # B. Weighted Canary Model Router (80:20 split + automatic failover!)
    class NamedLLM(BaseLLMClient):
        def __init__(self, tag: str, should_fail: bool = False) -> None:
            self.tag = tag
            self.should_fail = should_fail

        async def complete(self, messages, tools=None, *, prompt_cache_key=None) -> LLMResponse:
            if self.should_fail:
                raise RuntimeError(f"{self.tag} down")
            return LLMResponse(content=self.tag, model=self.tag)

    llm_a = NamedLLM("deepseek-v3")
    llm_b = NamedLLM("qwen-canary")
    model_router = ModelRouterLLMClient(
        routes=[
            ModelRoute(name="deepseek", client=llm_a, tier="reasoning", priority=0, weight=80),
            ModelRoute(name="qwen", client=llm_b, tier="reasoning", priority=0, weight=20),
        ],
        failure_threshold=1,
        cooldown_seconds=60.0,
        rng=random.Random(42),
    )
    hits = {"deepseek-v3": 0, "qwen-canary": 0}
    for _ in range(200):
        resp = await model_router.complete([Message(role="user", content="ping")])
        hits[resp.model] += 1
    assert 130 <= hits["deepseek-v3"] <= 185
    assert 15 <= hits["qwen-canary"] <= 70
    assert {h["name"]: h["weight"] for h in model_router.route_health()} == {
        "deepseek": 80,
        "qwen": 20,
    }

    # If canary fails, router circuit-breaks it and falls over to healthy peer seamlessly!
    llm_b.should_fail = True
    for _ in range(10):
        resp = await model_router.complete([Message(role="user", content="failover")])
        assert resp.model == "deepseek-v3"

    # C. Scheduled Autonomous Tasks (cron_matches + API run_now + artifact + journal event)
    assert cron_matches("*/15 9 * * 1-5", datetime(2026, 11, 26, 9, 30)) is True  # Thursday
    assert cron_matches("*/15 9 * * 1-5", datetime(2026, 11, 28, 9, 30)) is False  # Saturday

    container = RuntimeContainer(
        llm=ScriptedMockLLMClient([LLMResponse(content="每日巡检汇总：全厂设备状态正常，0 条紧急告警。")])
    )
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_create = await client.post(
            "/api/v1/tasks/scheduled",
            json={
                "name": "daily_safety_digest",
                "prompt": "生成今日全厂巡检摘要",
                "cron_expr": "0 8 * * *",
            },
        )
        assert r_create.status_code == 200
        task_id = r_create.json()["task_id"]

        r_run = await client.post(f"/api/v1/tasks/scheduled/{task_id}/run")
        assert r_run.status_code == 200
        run_data = r_run.json()
        assert run_data["status"] == "completed"
        assert run_data["artifact_id"] is not None

        # Verify persisted artifact is readable via REST API
        r_art = await client.get(f"/api/v1/artifacts/{run_data['artifact_id']}")
        assert r_art.status_code == 200
        assert "全厂设备状态正常" in r_art.json()["content"]

        # Verify SCHEDULED_TASK_RUN audit event persisted in journal
        evts = await container.journal.areplay_session(run_data["session_id"])
        assert any(e.event_type == EventType.SCHEDULED_TASK_RUN for e in evts)

        # Verify /health exposes v0.4.0 metadata & /api/v1/skills/readiness works
        r_health = await client.get("/health")
        assert r_health.json()["version"] == "0.4.0"
        assert "context_and_resilience" in r_health.json()["mounted_plugins"]
        r_ready = await client.get("/api/v1/skills/readiness")
        assert r_ready.status_code == 200
        assert all(s["ready"] is True for s in r_ready.json()["skills"])

    await container.ashutdown()
