from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncGenerator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from pyharness.api.deps import DEFAULT_SYSTEM_PROMPT, RuntimeContainer
from pyharness.runtime import (
    AdmissionRejectedError,
    ApprovalAuthorizationError,
    ArtifactAuthorizationError,
    CancellationToken,
    EventType,
    ModelRouterLLMClient,
    QuestionAuthorizationError,
    ScheduledTaskSpec,
    TopicBlockedError,
)
from pyharness.security import (
    AuthenticationError,
    Principal,
    QuotaExceededError,
    use_principal,
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    queue_mode: Literal["reject", "steer", "followup"] = "reject"
    timezone: str | None = None
    project_id: str | None = None
    branch_name: str | None = None


class SteeringInputRequest(BaseModel):
    message: str = Field(..., min_length=1)


class QuestionAnswerRequest(BaseModel):
    selected_options: list[str] = Field(default_factory=list)
    free_text: str = ""


class BranchCreateRequest(BaseModel):
    branch_name: str = Field(..., min_length=1)
    from_event_id: str | None = None
    switch: bool = True


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    project_id: str = "default"
    scope: Literal["project", "global"] = "project"
    category: str = "fact"
    tags: list[str] = Field(default_factory=list)


class ScheduledTaskCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    interval_seconds: float | None = None
    cron_expr: str | None = None
    enabled: bool = True


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
        version="0.4.0",
        description=(
            "Enterprise DSH-Grade Agent Harness Control Plane (v0.4.0) with Prefix-Cache-Optimal "
            "Context Engine, Tool Output Offloading (Memory/File/S3), Mid-Turn Steer/Follow-up, "
            "Interactive Clarification, In-Place Session Tree, Project Memory, Trusted Clock, "
            "Zero-Trust Credential Relay, Admission & Idle Reaper, Weighted Model Routing, "
            "JWT/RBAC, Four-Eyes SoD Approvals, Dynamic ACP, and SHA-256 Audit Hash Chain."
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
        readiness = [asdict(r) for r in rt.skill_router.check_readiness(rt.tools)]
        resp: dict[str, Any] = {
            "status": "ok",
            "version": "0.4.0",
            "profile": rt.profile,
            "auth_mode": rt.auth_manager.mode,
            "enforce_four_eyes": rt.approval_broker.enforce_four_eyes,
            "mounted_plugins": rt.plugin_context.mounted_plugins(),
            "journal_backend": rt.journal.backend_name,
            "artifact_store": rt.artifact_store.store_name,
            "sandbox_provider": rt.python_sandbox.provider.provider_name,
            "coordination_backend": rt.coordination.backend_name,
            "vector_store": rt.rag.dense.store_name,
            "policy_rules_loaded": len(rt.policy_engine.rules),
            "subagent_profiles": [p.name for p in rt.subagent_manager.list_profiles()],
            "tools_loaded": [t.name for t in rt.tools.list_specs()],
            "skills_loaded": [s.name for s in rt.skill_router.list_skills()],
            "skill_readiness": readiness,
            "admission_stats": rt.admission_controller.stats(),
            "scheduled_tasks_count": len(rt.scheduler.list_tasks()),
            "credential_targets_count": len(rt.credential_relay.list_targets()),
        }
        if isinstance(rt.llm, ModelRouterLLMClient):
            resp["model_routes"] = rt.llm.route_health()
        return resp

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
    ):
        """
        Streamable HTTP / SSE endpoint emitting real-time Harness events with AuthN/RBAC,
        Topic Guard, Concurrency Admission, FinOps RPM/Budget guard, ACP policy enforcement,
        DLP guardrails, Prefix-Cache-Optimal context, and session locking/steering.
        """
        await rt.startup()
        await assert_session_access(req.session_id, principal, allow_missing=True)

        # 1. Pre-turn Topic Admission Guard (blocks/redirects out-of-scope prompts before spending tokens)
        try:
            rt.topic_guard.enforce(req.message, tenant_id=principal.tenant_id)
        except TopicBlockedError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "topic_blocked",
                    "rule_id": exc.rule_id,
                    "action": exc.action,
                    "reason": exc.reason,
                    "redirect_message": exc.redirect_message,
                },
            ) from exc

        # 2. Enterprise FinOps rate-limit & daily token/USD budget check
        try:
            await rt.quota_manager.check_before_turn(principal)
        except QuotaExceededError as exc:
            raise HTTPException(
                status_code=429,
                detail=exc.reason,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc

        # 3. Per-session exclusivity lock with optional Pi-style steer/followup queue modes
        lock_cm = rt.coordination.session_lock(req.session_id)
        acquired = await lock_cm.__aenter__()
        if not acquired:
            await lock_cm.__aexit__(None, None, None)
            if req.queue_mode == "steer":
                queued = rt.steering_queue.enqueue_steer(
                    req.session_id,
                    req.message,
                    user_id=principal.user_id,
                    tenant_id=principal.tenant_id,
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "queued": True,
                        "mode": "steer",
                        "message_id": queued.message_id,
                        "session_id": req.session_id,
                    },
                )
            if req.queue_mode == "followup":
                queued = rt.steering_queue.enqueue_followup(
                    req.session_id,
                    req.message,
                    user_id=principal.user_id,
                    tenant_id=principal.tenant_id,
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "queued": True,
                        "mode": "followup",
                        "message_id": queued.message_id,
                        "session_id": req.session_id,
                    },
                )
            raise HTTPException(
                status_code=409,
                detail=f"Session '{req.session_id}' already has an active turn running.",
            )

        # 4. Global & per-tenant concurrency admission control (fail-fast HTTP 503)
        try:
            await rt.admission_controller.acquire(req.session_id, tenant_id=principal.tenant_id)
        except AdmissionRejectedError as exc:
            await lock_cm.__aexit__(None, None, None)
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.reason,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc

        await rt.journal.ensure_session(
            req.session_id,
            tenant_id=principal.tenant_id,
            owner_user_id=principal.user_id,
        )
        if req.branch_name:
            try:
                await rt.journal.switch_branch(req.session_id, req.branch_name)
                rt.session_histories.pop(req.session_id, None)
            except KeyError:
                pass

        rt.idle_reaper.touch(req.session_id)
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
        history = None
        if not req.branch_name:
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
                        timezone_name=req.timezone or "UTC",
                        project_id=req.project_id,
                    )
                rt.session_histories[req.session_id] = result.messages
                rt.idle_reaper.touch(req.session_id)
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
                await rt.admission_controller.release(req.session_id, tenant_id=principal.tenant_id)
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

    @app.post("/api/v1/sessions/{session_id}/steer")
    async def steer_session(
        session_id: str,
        body: SteeringInputRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Inject a high-priority mid-turn steering directive into an active session (Pi pattern)."""
        await rt.startup()
        await assert_session_access(session_id, principal, allow_missing=True)
        msg = rt.steering_queue.enqueue_steer(
            session_id,
            body.message,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        return {
            "session_id": session_id,
            "queued": True,
            "kind": "steer",
            "message_id": msg.message_id,
            "pending": rt.steering_queue.peek_counts(session_id),
        }

    @app.post("/api/v1/sessions/{session_id}/followup")
    async def followup_session(
        session_id: str,
        body: SteeringInputRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Enqueue a post-turn follow-up message to auto-execute after the current turn settles."""
        await rt.startup()
        await assert_session_access(session_id, principal, allow_missing=True)
        msg = rt.steering_queue.enqueue_followup(
            session_id,
            body.message,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        return {
            "session_id": session_id,
            "queued": True,
            "kind": "followup",
            "message_id": msg.message_id,
            "pending": rt.steering_queue.peek_counts(session_id),
        }

    @app.get("/api/v1/sessions/{session_id}/questions")
    async def list_session_questions(
        session_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        await assert_session_access(session_id, principal)
        pending = rt.question_broker.list_pending(
            session_id=session_id,
            tenant_id=principal.tenant_id,
            is_admin=principal.is_admin,
        )
        return {"session_id": session_id, "pending": pending}

    @app.post("/api/v1/sessions/{session_id}/questions/{question_id}/answer")
    async def answer_session_question(
        session_id: str,
        question_id: str,
        body: QuestionAnswerRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Submit the session owner's answer to resume a paused `ask_user_question` tool call."""
        await rt.startup()
        await assert_session_access(session_id, principal)
        try:
            ok = rt.question_broker.submit_answer(
                question_id,
                selected_options=body.selected_options,
                free_text=body.free_text,
                principal=principal,
            )
        except QuestionAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Pending question '{question_id}' not found or already resolved.",
            )
        return {
            "session_id": session_id,
            "question_id": question_id,
            "answered": True,
            "answered_by": principal.user_id,
        }

    @app.get("/api/v1/sessions/{session_id}/artifacts")
    async def list_session_artifacts(
        session_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        await assert_session_access(session_id, principal)
        records = await rt.artifact_store.list_for_session(
            session_id,
            tenant_id=principal.tenant_id,
            is_admin=principal.is_admin,
        )
        return {
            "session_id": session_id,
            "artifacts": [r.model_dump() for r in records],
        }

    @app.get("/api/v1/artifacts/{artifact_id}")
    async def read_artifact_endpoint(
        artifact_id: str,
        offset: int = 0,
        limit: int = 4000,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        try:
            rec, chunk, has_more = await rt.artifact_store.read_slice(
                artifact_id,
                offset=offset,
                limit=limit,
                tenant_id=principal.tenant_id,
                is_admin=principal.is_admin,
            )
        except ArtifactAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "artifact_id": rec.artifact_id,
            "session_id": rec.session_id,
            "tool_name": rec.tool_name,
            "offset": max(0, int(offset)),
            "limit": max(0, int(limit)),
            "total_chars": rec.char_count,
            "has_more": has_more,
            "storage_uri": rec.storage_uri,
            "content": chunk,
        }

    @app.get("/api/v1/sessions/{session_id}/tree")
    async def get_session_tree_endpoint(
        session_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Return the in-place tree topology and all branches for `session_id` (Pi Tree Session)."""
        await rt.startup()
        await assert_session_access(session_id, principal)
        tree = await rt.journal.get_session_tree(session_id)
        return tree.model_dump()

    @app.post("/api/v1/sessions/{session_id}/branches")
    async def create_session_branch_endpoint(
        session_id: str,
        body: BranchCreateRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Create a zero-copy in-place branch anchored at `from_event_id` (or active head)."""
        await rt.startup()
        await assert_session_access(session_id, principal)
        try:
            branch = await rt.journal.create_branch(
                session_id,
                body.branch_name,
                from_event_id=body.from_event_id,
                switch=body.switch,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if body.switch:
            reconstructed = await rt.journal.areconstruct_branch_messages(
                session_id, DEFAULT_SYSTEM_PROMPT, branch_name=branch.branch_name
            )
            if reconstructed is not None:
                rt.session_histories[session_id] = reconstructed
                await rt.coordination.save_session_history(session_id, reconstructed)
            else:
                rt.session_histories.pop(session_id, None)
        return branch.model_dump()

    @app.post("/api/v1/sessions/{session_id}/branches/{branch_name}/switch")
    async def switch_session_branch_endpoint(
        session_id: str,
        branch_name: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        """Switch active in-place branch and rebuild branch dialogue history projection."""
        await rt.startup()
        await assert_session_access(session_id, principal)
        try:
            branch = await rt.journal.switch_branch(session_id, branch_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        reconstructed = await rt.journal.areconstruct_branch_messages(
            session_id, DEFAULT_SYSTEM_PROMPT, branch_name=branch.branch_name
        )
        if reconstructed is not None:
            rt.session_histories[session_id] = reconstructed
            await rt.coordination.save_session_history(session_id, reconstructed)
        else:
            rt.session_histories.pop(session_id, None)
        return branch.model_dump()

    @app.get("/api/v1/memories")
    async def list_memories_endpoint(
        project_id: str | None = None,
        query: str | None = None,
        top_k: int = 10,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        if query:
            items = await rt.memory_store.recall(
                query,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                project_id=project_id or "default",
                top_k=top_k,
            )
        else:
            items = await rt.memory_store.list_memories(
                tenant_id=principal.tenant_id,
                user_id=None if principal.is_admin else principal.user_id,
                project_id=project_id,
            )
        return {"memories": [m.model_dump() for m in items]}

    @app.post("/api/v1/memories")
    async def create_memory_endpoint(
        body: MemoryCreateRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        entry = await rt.memory_store.save(
            body.content,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            project_id=body.project_id,
            scope=body.scope,
            category=body.category,
            tags=body.tags,
        )
        return entry.model_dump()

    @app.delete("/api/v1/memories/{memory_id}")
    async def delete_memory_endpoint(
        memory_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        try:
            deleted = await rt.memory_store.delete(
                memory_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                is_admin=principal.is_admin,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
        return {"memory_id": memory_id, "deleted": True}

    @app.get("/api/v1/skills/readiness")
    async def skills_readiness_endpoint(
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        reports = [asdict(r) for r in rt.skill_router.check_readiness(rt.tools)]
        return {
            "skills": reports,
            "invocation_stats": rt.skill_router.get_invocation_stats(),
        }

    @app.get("/api/v1/tasks/scheduled")
    async def list_scheduled_tasks(
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        tenant_scope = None if principal.is_admin else principal.tenant_id
        tasks = [t.model_dump() for t in rt.scheduler.list_tasks(tenant_id=tenant_scope)]
        runs = [r.model_dump() for r in rt.scheduler.list_runs(limit=20)]
        return {"tasks": tasks, "recent_runs": runs}

    @app.post("/api/v1/tasks/scheduled")
    async def create_scheduled_task(
        body: ScheduledTaskCreateRequest,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        spec = ScheduledTaskSpec(
            name=body.name,
            prompt=body.prompt,
            interval_seconds=body.interval_seconds,
            cron_expr=body.cron_expr,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            roles=sorted(principal.roles),
            enabled=body.enabled,
        )
        try:
            registered = rt.scheduler.register_task(spec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return registered.model_dump()

    @app.post("/api/v1/tasks/scheduled/{task_id}/run")
    async def run_scheduled_task_now(
        task_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        spec = rt.scheduler.get_task(task_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Scheduled task '{task_id}' not found.")
        if not principal.is_admin and spec.tenant_id != principal.tenant_id:
            raise HTTPException(status_code=403, detail="Cross-tenant task execution denied.")
        run = await rt.scheduler.run_now(task_id)
        return run.model_dump()

    @app.delete("/api/v1/tasks/scheduled/{task_id}")
    async def delete_scheduled_task(
        task_id: str,
        principal: Principal = Depends(require_principal),
    ) -> dict[str, Any]:
        await rt.startup()
        spec = rt.scheduler.get_task(task_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Scheduled task '{task_id}' not found.")
        if not principal.is_admin and spec.tenant_id != principal.tenant_id:
            raise HTTPException(status_code=403, detail="Cross-tenant task deletion denied.")
        rt.scheduler.delete_task(task_id)
        return {"task_id": task_id, "deleted": True}

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
