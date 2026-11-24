from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pyharness.api.deps import RuntimeContainer
from pyharness.runtime.governance import CancellationToken
from pyharness.runtime.schemas import EventType


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ApprovalResolveRequest(BaseModel):
    approved: bool
    reviewer: str = "web_operator"
    note: str = ""


class RAGIngestRequest(BaseModel):
    source: str
    content: str
    doc_id: str | None = None


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 3
    mode: str = "hybrid_rerank"


def create_app(container: RuntimeContainer | None = None) -> FastAPI:
    rt = container or RuntimeContainer()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await rt.startup()
        try:
            yield
        finally:
            rt.shutdown()

    app = FastAPI(
        title="PyHarness Enterprise Agent API",
        version="0.1.0",
        description="Lightweight Enterprise Agent Harness Runtime (DSH-inspired) with Hybrid RAG, MCP & SSE.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = rt

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tools_loaded": [t.name for t in rt.tools.list_specs()],
            "skills_loaded": [s.name for s in rt.skill_router.list_skills()],
        }

    @app.post("/api/v1/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """
        Streamable HTTP / SSE endpoint emitting real-time Harness events:
        thinking -> tool_call -> approval_required -> tool_result -> token_delta -> turn_complete.
        """
        await rt.startup()
        sub_queue = rt.journal.subscribe(req.session_id)
        cancel_token = CancellationToken()
        rt.active_cancellations[req.session_id] = cancel_token

        base_prompt = "You are PyHarness, an enterprise AI assistant with Hybrid RAG, MCP tools, and SQL/Python sandboxes."
        augmented_prompt, active_skills = rt.skill_router.build_augmented_prompt(base_prompt, req.message)
        history = rt.session_histories.get(req.session_id)

        async def _run_agent():
            try:
                result = await rt.engine.run_turn(
                    session_id=req.session_id,
                    user_input=req.message,
                    history=history,
                    system_prompt=augmented_prompt,
                    cancellation=cancel_token,
                )
                rt.session_histories[req.session_id] = result.messages
                return result
            finally:
                rt.active_cancellations.pop(req.session_id, None)

        agent_task = asyncio.create_task(_run_agent())

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # Immediately notify client which dynamic skills activated
                if active_skills:
                    meta_evt = {
                        "event_type": "skills_activated",
                        "payload": {"skills": [s.name for s in active_skills]},
                    }
                    yield f"event: skills_activated\ndata: {json.dumps(meta_evt, ensure_ascii=False)}\n\n"

                while True:
                    try:
                        evt = await asyncio.wait_for(sub_queue.get(), timeout=0.25)
                        data_str = evt.model_dump_json()
                        yield f"id: {evt.seq}\nevent: {evt.event_type.value}\ndata: {data_str}\n\n"
                        if evt.event_type in (EventType.TURN_COMPLETE, EventType.TURN_CANCELLED):
                            break
                    except asyncio.TimeoutError:
                        if agent_task.done():
                            # Drain any remaining queued events before exiting
                            while not sub_queue.empty():
                                evt = sub_queue.get_nowait()
                                yield f"id: {evt.seq}\nevent: {evt.event_type.value}\ndata: {evt.model_dump_json()}\n\n"
                            break
            finally:
                rt.journal.unsubscribe(req.session_id, sub_queue)
                if not agent_task.done():
                    cancel_token.cancel("client_disconnected")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering for zero-latency SSE!
            },
        )

    @app.post("/api/v1/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> dict[str, Any]:
        token = rt.active_cancellations.get(session_id)
        if token is None:
            return {"session_id": session_id, "cancelled": False, "detail": "No active turn running."}
        token.cancel("user_cancelled_via_api")
        return {"session_id": session_id, "cancelled": True}

    @app.get("/api/v1/approvals")
    async def list_approvals(session_id: str | None = None) -> dict[str, Any]:
        return {"pending": rt.approval_broker.list_pending(session_id=session_id)}

    @app.post("/api/v1/approvals/{approval_id}/resolve")
    async def resolve_approval(approval_id: str, body: ApprovalResolveRequest) -> dict[str, Any]:
        ok = rt.approval_broker.resolve(
            approval_id=approval_id,
            approved=body.approved,
            reviewer=body.reviewer,
            note=body.note,
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Pending approval '{approval_id}' not found.")
        return {"approval_id": approval_id, "approved": body.approved, "resolved": True}

    @app.get("/api/v1/sessions/{session_id}/events")
    async def replay_events(session_id: str, after_seq: int = 0) -> dict[str, Any]:
        events = rt.journal.replay_session(session_id, after_seq=after_seq)
        return {"session_id": session_id, "events": [e.model_dump() for e in events]}

    @app.get("/api/v1/sessions/{session_id}/trace/{turn_id}")
    async def get_trace(session_id: str, turn_id: str) -> dict[str, Any]:
        return rt.journal.build_trace_summary(session_id, turn_id)

    @app.post("/api/v1/rag/ingest")
    async def rag_ingest(body: RAGIngestRequest) -> dict[str, Any]:
        await rt.startup()
        stats = await rt.rag.ingest_markdown(body.content, source=body.source, doc_id=body.doc_id)
        return {"source": body.source, **stats}

    @app.post("/api/v1/rag/search")
    async def rag_search(body: RAGSearchRequest) -> dict[str, Any]:
        await rt.startup()
        hits = await rt.rag.search(body.query, top_k=body.top_k, mode=body.mode)
        return {
            "query": body.query,
            "mode": body.mode,
            "hits": [
                {
                    "citation_id": h.citation_id,
                    "source": h.source,
                    "section_path": h.section_path,
                    "score": h.score,
                    "snippet": h.child_snippet,
                    "parent_context": h.parent_context,
                }
                for h in hits
            ],
        }

    return app


app = create_app()
