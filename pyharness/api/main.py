from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from pyharness.api.deps import DEFAULT_SYSTEM_PROMPT, RuntimeContainer
from pyharness.runtime.governance import ApprovalAuthorizationError, CancellationToken
from pyharness.runtime.schemas import EventType
from pyharness.security import (
    AuthenticationError,
    Principal,
    QuotaExceededError,
    use_principal,
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ApprovalResolveRequest(BaseModel):
    approved: bool
    reviewer: str = "web_operator"
    note: str = ""


class SessionForkRequest(BaseModel):
    new_session_id: str | None = None
    at_seq: int | None = None


class RAGIngestRequest(BaseModel):
    source: str
    content: str
    doc_id: str | None = None
    metadata: dict[str, str] | None = None


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 3
    mode: str = "hybrid_rerank"
    metadata_filter: dict[str, Any] | None = None


def _parse_cors_origins(auth_mode: str) -> list[str]:
    raw = os.getenv("HARNESS_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if auth_mode == "enforce":
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return ["*"]


def create_app(container: RuntimeContainer | None = None) -> FastAPI:
    rt = container or RuntimeContainer()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await rt.startup()
        try:
            yield
        finally:
            await rt.ashutdown()

    app = FastAPI(
        title="PyHarness Enterprise Agent API",
        version="0.3.0",
        description=(
            "Enterprise DSH-Grade Agent Harness Control Plane with JWT/RBAC, Four-Eyes SoD Approvals, "
            "Dynamic ACP Policy Engine, DLP/Injection Guardrails, Cordis Plugin Kernel, Subagents, "
            "Tamper-Evident Audit Hash Chain, Pluggable Sandboxes, Redis & Qdrant."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(rt.auth_manager.mode),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = rt

    async def require_principal(request: Request) -> Principal:
        try:
            return rt.auth_manager.authenticate(request.headers)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    async def assert_session_access(
        session_id: str,
        principal: Principal,
        *,
        allow_missing: bool = False,
    ) -> None:
        record = await rt.journal.get_session(session_id)
        if record is None:
            if allow_missing:
                return
            return
        if principal.is_admin:
            return
        if record.tenant_id != principal.tenant_id:
            raise HTTPException(
                status_code=403,
                detail=f"Cross-tenant access denied for session '{session_id}'.",
            )
        if (
            record.owner_user_id not in (principal.user_id, "anonymous")
            and not principal.has_any_role({"auditor", "approver", "admin"})
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: session '{session_id}' belongs to another user.",
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "profile": rt.profile,
            "auth_mode": rt.auth_manager.mode,
            "enforce_four_eyes": rt.approval_broker.enforce_four_eyes,
            "mounted_plugins": rt.plugin_context.mounted_plugins(),
            "journal_backend": rt.journal.backend_name,
            "sandbox_provider": rt.python_sandbox.provider.provider_name,
            "coordination_backend": rt.coordination.backend_name,
            "vector_store": rt.rag.dense.store_name,
            "policy_rules_loaded": len(rt.policy_engine.rules),
            "subagent_profiles": [p.name for p in rt.subagent_manager.list_profiles()],
            "tools_loaded": [t.name for t in rt.tools.list_specs()],
            "skills_loaded": [s.name for s in rt.skill_router.list_skills()],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            content=rt.telemetry.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/api/v1/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        principal: Principal = Depends(require_principal),
    ) -> StreamingResponse:
        """
        Streamable HTTP / SSE endpoint emitting real-time Harness events with AuthN/RBAC,
        FinOps RPM/Budget guard, ACP policy enforcement, DLP guardrails, and session locking.
        """
        await rt.startup()
        await assert_session_access(req.session_id, principal, allow_missing=True)

        # Enterprise FinOps rate-limit & daily token/USD budget check
        try:
            await rt.quota_manager.check_before_turn(principal)
        except QuotaExceededError as exc:
            raise HTTPException(
                status_code=429,
                detail=exc.reason,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc

        lock_cm = rt.coordination.session_lock(req.session_id)
        acquired = await lock_cm.__aenter__()
        if not acquired:
            await lock_cm.__aexit__(None, None, None)
            raise HTTPException(
                status_code=409,
                detail=f"Session '{req.session_id}' already has an active turn running.",
            )

        await rt.journal.ensure_session(
            req.session_id,
            tenant_id=principal.tenant_id,
            owner_user_id=principal.user_id,
        )

        sub_queue = rt.journal.subscribe(req.session_id)
        cancel_token = CancellationToken()
        rt.active_cancellations[req.session_id] = cancel_token
        await rt.coordination.register_active_session(req.session_id)

        async def _watch_remote_cancel() -> None:
            try:
                reason = await rt.coordination.wait_for_cancel(req.session_id)
                if not cancel_token.is_cancelled:
                    cancel_token.cancel(reason)
            except asyncio.CancelledError:
                pass

        remote_cancel_task = asyncio.create_task(_watch_remote_cancel())

        turn_instructions, allowed_tools, active_skills = rt.skill_router.resolve_turn_context(
            req.message
        )

        # Load durable session history from Redis/Coordination -> fallback to EventJournal
        history = await rt.coordination.get_session_history(req.session_id)
        if history is None:
            history = rt.session_histories.get(req.session_id)
        if history is None:
            history = await rt.journal.areconstruct_messages(
                req.session_id, DEFAULT_SYSTEM_PROMPT
            )

        async def _run_agent():
            try:
                with use_principal(principal):
                    result = await rt.engine.run_turn(
                        session_id=req.session_id,
                        user_input=req.message,
                        history=history,
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        cancellation=cancel_token,
                        turn_system_instructions=turn_instructions,
                        allowed_tools=allowed_tools,
                    )
                rt.session_histories[req.session_id] = result.messages
                await rt.coordination.save_session_history(req.session_id, result.messages)
                if result.usage.total_tokens > 0 or result.usage.estimated_cost_usd > 0:
                    await rt.quota_manager.record_usage(
                        principal=principal,
                        prompt_tokens=result.usage.prompt_tokens,
                        completion_tokens=result.usage.completion_tokens,
                        cost_usd=result.usage.estimated_cost_usd,
                    )
                return result
            finally:
                remote_cancel_task.cancel()
                rt.active_cancellations.pop(req.session_id, None)
                await rt.coordination.unregister_active_session(req.session_id)
                await lock_cm.__aexit__(None, None, None)

        agent_task = asyncio.create_task(_run_agent())

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                if active_skills:
                    meta_evt = {
                        "event_type": "skills_activated",
                        "payload": {
                            "skills": [s.name for s in active_skills],
                            "allowed_tools": sorted(allowed_tools) if allowed_tools else [],
                        },
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
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/sessions/{session_id}/stream")
    async def reconnect_session_stream(
        session_id: str,
        request: Request,
        after_seq: int = 0,
        follow: bool = True,
        principal: Principal = Depends(require_principal),
    ) -> StreamingResponse:
        """
        Resumable SSE endpoint supporting `after_seq` query parameter or standard `Last-Event-ID`
        header to replay missed events and re-attach seamlessly across Pods via Redis Pub/Sub.
        """
        await rt.startup()
        await assert_session_access(session_id, principal)

        last_evt_hdr = request.headers.get("last-event-id", "").strip()
        effective_after = after_seq
        if last_evt_hdr.isdigit():
            effective_after = max(effective_after, int(last_evt_hdr))

        async def _resume_gen() -> AsyncGenerator[str, None]:
            async for evt in rt.journal.stream_live_events(
                session_id=session_id,
                after_seq=effective_after,
                follow_if_active=follow,
            ):
                yield f"id: {evt.seq}\nevent: {evt.event_type.value}\ndata: {evt.model_dump_json()}\n\n"

        return StreamingResponse(
            _resume_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/v1/sessions/{session_id}/fork")
    async def fork_session(
        session_id: str,
        body: SessionForkRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Branch an existing session at `at_seq` into a new independent trajectory (DSH Session Fork)."""
        await rt.startup()
        await assert_session_access(session_id, principal)

        new_sid = body.new_session_id or f"{session_id}_fork_{uuid.uuid4().hex[:6]}"
        try:
            record, copied = await rt.journal.fork_session(
                source_session_id=session_id,
                new_session_id=new_sid,
                at_seq=body.at_seq,
                tenant_id=principal.tenant_id,
                owner_user_id=principal.user_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # Pre-warm forked dialogue history in coordination cache
        reconstructed = await rt.journal.areconstruct_messages(new_sid, DEFAULT_SYSTEM_PROMPT)
        if reconstructed:
            rt.session_histories[new_sid] = reconstructed
            await rt.coordination.save_session_history(new_sid, reconstructed)

        forked_events = await rt.journal.areplay_session(new_sid)
        last_seq = forked_events[-1].seq if forked_events else 0
        return {
            "source_session_id": session_id,
            "new_session_id": record.session_id,
            "forked_from_seq": record.forked_from_seq,
            "copied_events": copied,
            "last_seq": last_seq,
            "tenant_id": record.tenant_id,
        }

    @app.get("/api/v1/sessions/{session_id}/audit/verify")
    async def verify_session_audit(
        session_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Cryptographically verify the SHA-256 tamper-evident hash chain for a session."""
        await rt.startup()
        await assert_session_access(session_id, principal)
        res = await rt.journal.verify_audit_chain(session_id)
        return res.model_dump()

    @app.post("/api/v1/sessions/{session_id}/cancel")
    async def cancel_session(
        session_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await assert_session_access(session_id, principal)
        token = rt.active_cancellations.get(session_id)
        local_cancelled = False
        if token is not None:
            token.cancel("user_cancelled_via_api")
            local_cancelled = True
        dist_cancelled = await rt.coordination.publish_cancel(session_id, "user_cancelled_via_api")
        if not local_cancelled and not dist_cancelled:
            return {"session_id": session_id, "cancelled": False, "detail": "No active turn running."}
        return {"session_id": session_id, "cancelled": True}

    @app.get("/api/v1/approvals")
    async def list_approvals(
        session_id: str | None = None,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        tenant_scope = None if principal.is_admin else principal.tenant_id
        pending = await rt.approval_broker.list_pending_async(
            session_id=session_id,
            tenant_id=tenant_scope,
        )
        return {"pending": pending}

    @app.post("/api/v1/approvals/{approval_id}/resolve")
    async def resolve_approval(
        approval_id: str,
        body: ApprovalResolveRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        try:
            ok = await rt.approval_broker.aresolve(
                approval_id=approval_id,
                approved=body.approved,
                reviewer=body.reviewer,
                note=body.note,
                reviewer_principal=principal,
            )
        except ApprovalAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not ok:
            raise HTTPException(status_code=404, detail=f"Pending approval '{approval_id}' not found.")
        eff_reviewer = (
            body.reviewer
            if principal.auth_method == "dev_anonymous"
            else principal.user_id
        )
        return {
            "approval_id": approval_id,
            "approved": body.approved,
            "reviewer": eff_reviewer,
            "resolved": True,
        }

    @app.get("/api/v1/sessions/{session_id}/events")
    async def replay_events(
        session_id: str,
        after_seq: int = 0,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await assert_session_access(session_id, principal)
        events = await rt.journal.areplay_session(session_id, after_seq=after_seq)
        return {"session_id": session_id, "events": [e.model_dump() for e in events]}

    @app.get("/api/v1/sessions/{session_id}/trace/{turn_id}")
    async def get_trace(
        session_id: str,
        turn_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await assert_session_access(session_id, principal)
        return await rt.journal.abuild_trace_summary(session_id, turn_id)

    @app.post("/api/v1/rag/ingest")
    async def rag_ingest(
        body: RAGIngestRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        meta = dict(body.metadata or {})
        if not principal.is_admin:
            meta["tenant_id"] = principal.tenant_id
            meta.pop("visibility", None)
        elif "tenant_id" not in meta and meta.get("visibility") != "public":
            meta["tenant_id"] = principal.tenant_id
        stats = await rt.rag.ingest_markdown(
            body.content,
            source=body.source,
            doc_id=body.doc_id,
            metadata=meta,
        )
        return {"source": body.source, "tenant_id": meta.get("tenant_id"), **stats}

    @app.post("/api/v1/rag/search")
    async def rag_search(
        body: RAGSearchRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        eff_filter = dict(body.metadata_filter or {})
        if not principal.is_admin:
            eff_filter["_acl_tenant"] = principal.tenant_id
        hits = await rt.rag.search(
            body.query,
            top_k=body.top_k,
            mode=body.mode,
            metadata_filter=eff_filter or None,
        )
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

    @app.get("/api/v1/admin/usage")
    async def admin_usage_report(
        tenant_id: str | None = None,
        date: str | None = None,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Departmental & tenant FinOps token/USD cost attribution report (`admin` or `auditor`)."""
        if not principal.has_any_role({"admin", "auditor"}):
            raise HTTPException(
                status_code=403,
                detail="Admin or auditor role is required to view usage reports.",
            )
        scoped_tenant = tenant_id if principal.is_admin else principal.tenant_id
        return await rt.quota_manager.get_usage_report(tenant_id=scoped_tenant, date_str=date)

    return app


app = create_app()
