from __future__ import annotations

import asyncio
import json

import httpx

from pyharness.api.deps import RuntimeContainer
from pyharness.api.main import create_app


async def test_fastapi_health_rag_and_sse_stream_end_to_end() -> None:
    container = RuntimeContainer()
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health & loaded tools/skills
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "rag_search" in body["tools_loaded"]
        assert "query_sql_readonly" in body["tools_loaded"]
        assert "mcp_dispatch_emergency_alert" in body["tools_loaded"]
        assert "data_analysis" in body["skills_loaded"]

        # 2. RAG ingest + search REST endpoints
        r_ing = await client.post(
            "/api/v1/rag/ingest",
            json={"source": "extra.md", "content": "## 3. 冷却泵维护\n冷却泵轴承温度上限为 72℃。"},
        )
        assert r_ing.status_code == 200
        r_search = await client.post(
            "/api/v1/rag/search",
            json={"query": "冷却泵轴承温度上限", "top_k": 2},
        )
        assert r_search.status_code == 200
        assert "72℃" in r_search.json()["hits"][0]["parent_context"]

        # 3. Streamable HTTP / SSE chat turn (SQL + schema concurrent tool execution)
        events_seen: list[str] = []
        turn_id = ""
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"session_id": "web_s1", "message": "用 SQL 统计各区域故障数和总停机时长"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    events_seen.append(line.removeprefix("event: ").strip())
                elif line.startswith("data: ") and not turn_id:
                    data = json.loads(line.removeprefix("data: "))
                    if data.get("turn_id"):
                        turn_id = data["turn_id"]

        assert "skills_activated" in events_seen
        assert "tool_call" in events_seen
        assert "tool_result" in events_seen
        assert "turn_complete" in events_seen

        # 4. Event replay + Trace waterfall API
        r_replay = await client.get("/api/v1/sessions/web_s1/events?after_seq=0")
        assert r_replay.status_code == 200
        assert len(r_replay.json()["events"]) >= 4

        r_trace = await client.get(f"/api/v1/sessions/web_s1/trace/{turn_id}")
        assert r_trace.status_code == 200
        assert r_trace.json()["status"] == "completed"
        assert len(r_trace.json()["tool_spans"]) == 2

    container.shutdown()


async def test_fastapi_sse_human_in_the_loop_approval_flow() -> None:
    container = RuntimeContainer()
    await container.startup()
    app = create_app(container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        events_seen: list[str] = []

        async def approve_when_pending() -> None:
            for _ in range(40):
                await asyncio.sleep(0.03)
                r = await client.get("/api/v1/approvals?session_id=web_apr")
                pending = r.json()["pending"]
                if pending:
                    apr_id = pending[0]["approval_id"]
                    res = await client.post(
                        f"/api/v1/approvals/{apr_id}/resolve",
                        json={"approved": True, "reviewer": "shanghai_ops", "note": "LGTM"},
                    )
                    assert res.status_code == 200
                    return

        approver = asyncio.create_task(approve_when_pending())
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"session_id": "web_apr", "message": "立即发送紧急告警通知"},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    events_seen.append(line.removeprefix("event: ").strip())
        await approver

        assert "approval_required" in events_seen
        assert "approval_resolved" in events_seen
        assert "turn_complete" in events_seen

    container.shutdown()
