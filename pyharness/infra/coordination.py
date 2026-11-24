from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from pyharness.runtime.schemas import HarnessEvent, Message


class CoordinationBackend(ABC):
    """
    Pluggable Distributed State & Pub/Sub Seam (inspired by DSH sessionPersistence & control-plane bus).

    Responsibilities:
    1. Persist multi-turn `Message` histories across API workers and restarts (with TTL).
    2. Enforce distributed per-session turn mutual exclusion (`session_lock`).
    3. Coordinate cross-worker Human-in-the-Loop approvals (`ApprovalBroker`) with SoD metadata.
    4. Broadcast cross-worker session cancellation and live SSE events (`publish_event`/`subscribe_events`).
    5. Track per-user RPM sliding windows and daily departmental token/cost FinOps quotas.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_session_history(self, session_id: str) -> list[Message] | None:
        raise NotImplementedError

    @abstractmethod
    async def save_session_history(
        self,
        session_id: str,
        messages: list[Message],
        ttl_seconds: int = 86400,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def session_lock(self, session_id: str, ttl_seconds: int = 300) -> AsyncIterator[bool]:
        yield False

    @abstractmethod
    async def register_pending_approval(
        self,
        approval_id: str,
        payload: dict[str, Any],
        ttl_seconds: int = 180,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_pending_approval(self, approval_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def list_pending_approvals(
        self,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def resolve_approval(self, approval_id: str, decision: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def wait_for_approval(
        self,
        approval_id: str,
        timeout_seconds: float,
        cancellation_wait: Callable[[], Awaitable[None]],
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def remove_pending_approval(self, approval_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def register_active_session(self, session_id: str, ttl_seconds: int = 300) -> None:
        raise NotImplementedError

    @abstractmethod
    async def unregister_active_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def is_session_active(self, session_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def publish_cancel(self, session_id: str, reason: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def wait_for_cancel(self, session_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def publish_event(self, session_id: str, event: HarnessEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    async def subscribe_events(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        if False:  # pragma: no cover - abstract async generator stub
            yield  # type: ignore[misc]
        raise NotImplementedError

    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        raise NotImplementedError

    @abstractmethod
    async def get_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def record_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        department: str,
        user_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_usage_summary(
        self,
        date_str: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


class InMemoryCoordinationBackend(CoordinationBackend):
    """
    Bounded, TTL-aware in-process coordination backend for zero-daemon local execution and CI tests.
    """

    def __init__(self, max_sessions: int = 1024) -> None:
        self.max_sessions = max_sessions
        self._histories: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._active_locks: set[str] = set()
        self._lock_guard = asyncio.Lock()

        self._pending_approvals: dict[str, tuple[float, dict[str, Any]]] = {}
        self._approval_waiters: dict[str, list[asyncio.Future[dict[str, Any]]]] = {}
        self._resolved_decisions: dict[str, tuple[float, dict[str, Any]]] = {}

        self._active_sessions: dict[str, float] = {}
        self._cancel_waiters: dict[str, list[asyncio.Future[str]]] = {}
        self._cancel_reasons: dict[str, tuple[float, str]] = {}

        self._event_subs: dict[str, list[asyncio.Queue[HarnessEvent]]] = {}
        self._rate_windows: dict[str, deque[float]] = {}
        self._daily_usage: dict[tuple[str, str, str], dict[str, Any]] = {}

    @property
    def backend_name(self) -> str:
        return "memory"

    def _prune_histories(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._histories.items() if exp <= now]
        for k in expired:
            self._histories.pop(k, None)
        while len(self._histories) > self.max_sessions:
            self._histories.popitem(last=False)

    async def get_session_history(self, session_id: str) -> list[Message] | None:
        self._prune_histories()
        entry = self._histories.get(session_id)
        if entry is None:
            return None
        exp, raw = entry
        if exp <= time.monotonic():
            self._histories.pop(session_id, None)
            return None
        self._histories.move_to_end(session_id)
        return [Message.model_validate(item) for item in raw]

    async def save_session_history(
        self,
        session_id: str,
        messages: list[Message],
        ttl_seconds: int = 86400,
    ) -> None:
        self._prune_histories()
        serialized = [m.model_dump() for m in messages]
        self._histories[session_id] = (time.monotonic() + ttl_seconds, serialized)
        self._histories.move_to_end(session_id)

    @asynccontextmanager
    async def session_lock(self, session_id: str, ttl_seconds: int = 300) -> AsyncIterator[bool]:
        async with self._lock_guard:
            if session_id in self._active_locks:
                acquired = False
            else:
                self._active_locks.add(session_id)
                acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                async with self._lock_guard:
                    self._active_locks.discard(session_id)

    async def register_pending_approval(
        self,
        approval_id: str,
        payload: dict[str, Any],
        ttl_seconds: int = 180,
    ) -> None:
        self._pending_approvals[approval_id] = (time.monotonic() + ttl_seconds, dict(payload))

    async def get_pending_approval(self, approval_id: str) -> dict[str, Any] | None:
        entry = self._pending_approvals.get(approval_id)
        if entry is None:
            return None
        exp, data = entry
        if exp <= time.monotonic():
            self._pending_approvals.pop(approval_id, None)
            return None
        return dict(data)

    async def list_pending_approvals(
        self,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        now = time.monotonic()
        out: list[dict[str, Any]] = []
        for aid, (exp, data) in list(self._pending_approvals.items()):
            if exp <= now:
                self._pending_approvals.pop(aid, None)
                continue
            if session_id is not None and data.get("session_id") != session_id:
                continue
            if tenant_id is not None and data.get("tenant_id", "default") != tenant_id:
                continue
            out.append(dict(data))
        return out

    async def resolve_approval(self, approval_id: str, decision: dict[str, Any]) -> bool:
        existed = approval_id in self._pending_approvals
        self._pending_approvals.pop(approval_id, None)
        if not existed and approval_id not in self._approval_waiters:
            return False
        self._resolved_decisions[approval_id] = (time.monotonic() + 60.0, dict(decision))
        for fut in self._approval_waiters.pop(approval_id, []):
            if not fut.done():
                fut.set_result(dict(decision))
        return True

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout_seconds: float,
        cancellation_wait: Callable[[], Awaitable[None]],
    ) -> dict[str, Any] | None:
        if approval_id in self._resolved_decisions:
            _, dec = self._resolved_decisions.pop(approval_id)
            return dec

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._approval_waiters.setdefault(approval_id, []).append(fut)
        cancel_task = asyncio.create_task(cancellation_wait())
        try:
            done, _ = await asyncio.wait(
                {fut, cancel_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                return None
            if fut in done:
                return fut.result()
            return None
        finally:
            cancel_task.cancel()
            waiters = self._approval_waiters.get(approval_id, [])
            if fut in waiters:
                waiters.remove(fut)

    async def remove_pending_approval(self, approval_id: str) -> None:
        self._pending_approvals.pop(approval_id, None)
        self._approval_waiters.pop(approval_id, None)

    async def register_active_session(self, session_id: str, ttl_seconds: int = 300) -> None:
        self._cancel_reasons.pop(session_id, None)
        self._active_sessions[session_id] = time.monotonic() + ttl_seconds

    async def unregister_active_session(self, session_id: str) -> None:
        self._active_sessions.pop(session_id, None)
        self._cancel_reasons.pop(session_id, None)
        for fut in self._cancel_waiters.pop(session_id, []):
            if not fut.done():
                fut.cancel()

    async def is_session_active(self, session_id: str) -> bool:
        exp = self._active_sessions.get(session_id)
        if exp is None or exp <= time.monotonic():
            self._active_sessions.pop(session_id, None)
            return False
        return True

    async def publish_cancel(self, session_id: str, reason: str) -> bool:
        if not await self.is_session_active(session_id):
            return False
        self._cancel_reasons[session_id] = (time.monotonic() + 60.0, reason)
        for fut in self._cancel_waiters.pop(session_id, []):
            if not fut.done():
                fut.set_result(reason)
        return True

    async def wait_for_cancel(self, session_id: str) -> str:
        if session_id in self._cancel_reasons:
            return self._cancel_reasons[session_id][1]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._cancel_waiters.setdefault(session_id, []).append(fut)
        return await fut

    async def publish_event(self, session_id: str, event: HarnessEvent) -> None:
        for q in list(self._event_subs.get(session_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe_events(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        q: asyncio.Queue[HarnessEvent] = asyncio.Queue(maxsize=512)
        self._event_subs.setdefault(session_id, []).append(q)
        try:
            while True:
                evt = await q.get()
                yield evt
        finally:
            subs = self._event_subs.get(session_id, [])
            if q in subs:
                subs.remove(q)
            if not subs:
                self._event_subs.pop(session_id, None)

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - float(window_seconds)
        window = self._rate_windows.setdefault(key, deque())
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, math.ceil(window[0] + window_seconds - now))
            return False, retry_after
        window.append(now)
        return True, 0

    async def get_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        entry = self._daily_usage.get((date_str, tenant_id, user_id))
        if not entry:
            return {"prompt_tokens": 0, "completion_tokens": 0, "tokens": 0, "cost_usd": 0.0}
        return dict(entry)

    async def record_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        department: str,
        user_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        key = (date_str, tenant_id, user_id)
        bucket = self._daily_usage.setdefault(
            key,
            {
                "date": date_str,
                "tenant_id": tenant_id,
                "department": department,
                "user_id": user_id,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens": 0,
                "cost_usd": 0.0,
            },
        )
        bucket["department"] = department
        bucket["prompt_tokens"] += max(0, int(prompt_tokens))
        bucket["completion_tokens"] += max(0, int(completion_tokens))
        bucket["tokens"] = bucket["prompt_tokens"] + bucket["completion_tokens"]
        bucket["cost_usd"] = round(bucket["cost_usd"] + max(0.0, float(cost_usd)), 6)

    async def list_usage_summary(
        self,
        date_str: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (d, tid, _uid), entry in sorted(self._daily_usage.items()):
            if d != date_str:
                continue
            if tenant_id is not None and tid != tenant_id:
                continue
            out.append(dict(entry))
        return out

    async def aclose(self) -> None:
        self._histories.clear()
        self._pending_approvals.clear()
        self._active_locks.clear()
        self._event_subs.clear()


_LUA_RELEASE_LOCK = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class RedisCoordinationBackend(CoordinationBackend):
    """
    Production Distributed Redis State & Coordination Backend (`redis://...`).

    Supports multi-worker / multi-Pod deployments:
    - Session history caching with sliding TTL (`pyharness:sess:{id}:history`).
    - Atomic `SET NX EX` + Lua compare-and-delete session turn locks.
    - Cross-worker `ApprovalBroker` state & Pub/Sub resolution wakeups with SoD metadata.
    - Cross-worker `CancellationToken` propagation & live SSE `subscribe_events` Pub/Sub bridge.
    - Per-user RPM rate limiting and daily FinOps token/cost ledger.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        redis_client: Any | None = None,
        key_prefix: str = "pyharness",
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.redis_url = redis_url
        self.prefix = key_prefix
        self.poll_interval = poll_interval_seconds
        self._owns_client = redis_client is None
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis.asyncio as aioredis  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "RedisCoordinationBackend requires the 'redis' package (`pip install redis>=5.0.0`)."
                ) from exc
            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    @property
    def backend_name(self) -> str:
        return "redis"

    def _k(self, suffix: str) -> str:
        return f"{self.prefix}:{suffix}"

    async def get_session_history(self, session_id: str) -> list[Message] | None:
        raw = await self._redis.get(self._k(f"sess:{session_id}:history"))
        if not raw:
            return None
        items = json.loads(raw)
        return [Message.model_validate(item) for item in items]

    async def save_session_history(
        self,
        session_id: str,
        messages: list[Message],
        ttl_seconds: int = 86400,
    ) -> None:
        payload = json.dumps([m.model_dump() for m in messages], ensure_ascii=False)
        await self._redis.set(self._k(f"sess:{session_id}:history"), payload, ex=ttl_seconds)

    @asynccontextmanager
    async def session_lock(self, session_id: str, ttl_seconds: int = 300) -> AsyncIterator[bool]:
        lock_key = self._k(f"sess:{session_id}:lock")
        token = uuid.uuid4().hex
        acquired = bool(await self._redis.set(lock_key, token, nx=True, ex=ttl_seconds))
        try:
            yield acquired
        finally:
            if acquired:
                if hasattr(self._redis, "eval"):
                    await self._redis.eval(_LUA_RELEASE_LOCK, 1, lock_key, token)
                else:
                    cur = await self._redis.get(lock_key)
                    if cur == token:
                        await self._redis.delete(lock_key)

    async def register_pending_approval(
        self,
        approval_id: str,
        payload: dict[str, Any],
        ttl_seconds: int = 180,
    ) -> None:
        key = self._k(f"approval:pending:{approval_id}")
        await self._redis.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
        await self._redis.sadd(self._k("approval:index"), approval_id)

    async def get_pending_approval(self, approval_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._k(f"approval:pending:{approval_id}"))
        if not raw:
            return None
        return json.loads(raw)

    async def list_pending_approvals(
        self,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = await self._redis.smembers(self._k("approval:index"))
        out: list[dict[str, Any]] = []
        for aid in sorted(ids):
            raw = await self._redis.get(self._k(f"approval:pending:{aid}"))
            if not raw:
                await self._redis.srem(self._k("approval:index"), aid)
                continue
            item = json.loads(raw)
            if session_id is not None and item.get("session_id") != session_id:
                continue
            if tenant_id is not None and item.get("tenant_id", "default") != tenant_id:
                continue
            out.append(item)
        return out

    async def resolve_approval(self, approval_id: str, decision: dict[str, Any]) -> bool:
        pending_key = self._k(f"approval:pending:{approval_id}")
        existed = await self._redis.delete(pending_key)
        await self._redis.srem(self._k("approval:index"), approval_id)
        if not existed:
            return False
        dec_json = json.dumps(decision, ensure_ascii=False)
        await self._redis.set(self._k(f"approval:decision:{approval_id}"), dec_json, ex=120)
        await self._redis.publish(self._k(f"pubsub:approval:{approval_id}"), dec_json)
        return True

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout_seconds: float,
        cancellation_wait: Callable[[], Awaitable[None]],
    ) -> dict[str, Any] | None:
        dec_key = self._k(f"approval:decision:{approval_id}")
        deadline = time.monotonic() + timeout_seconds
        cancel_task = asyncio.create_task(cancellation_wait())

        pubsub = self._redis.pubsub() if hasattr(self._redis, "pubsub") else None
        channel = self._k(f"pubsub:approval:{approval_id}")
        if pubsub is not None:
            await pubsub.subscribe(channel)

        try:
            while True:
                if cancel_task.done():
                    return None
                cached = await self._redis.get(dec_key)
                if cached:
                    await self._redis.delete(dec_key)
                    return json.loads(cached)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None

                wait_step = min(self.poll_interval, remaining)
                if pubsub is not None:
                    msg_task = asyncio.create_task(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=wait_step)
                    )
                    done, _ = await asyncio.wait(
                        {msg_task, cancel_task},
                        timeout=wait_step + 0.02,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if cancel_task in done:
                        msg_task.cancel()
                        return None
                    if msg_task in done:
                        msg = msg_task.result()
                        if msg and msg.get("data"):
                            raw_data = msg["data"]
                            await self._redis.delete(dec_key)
                            return json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                else:
                    done, _ = await asyncio.wait({cancel_task}, timeout=wait_step)
                    if cancel_task in done:
                        return None
        finally:
            cancel_task.cancel()
            if pubsub is not None:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

    async def remove_pending_approval(self, approval_id: str) -> None:
        await self._redis.delete(self._k(f"approval:pending:{approval_id}"))
        await self._redis.srem(self._k("approval:index"), approval_id)

    async def register_active_session(self, session_id: str, ttl_seconds: int = 300) -> None:
        await self._redis.delete(self._k(f"cancel:signal:{session_id}"))
        await self._redis.set(self._k(f"sess:{session_id}:active"), "1", ex=ttl_seconds)

    async def unregister_active_session(self, session_id: str) -> None:
        await self._redis.delete(self._k(f"sess:{session_id}:active"))
        await self._redis.delete(self._k(f"cancel:signal:{session_id}"))

    async def is_session_active(self, session_id: str) -> bool:
        val = await self._redis.get(self._k(f"sess:{session_id}:active"))
        return bool(val)

    async def publish_cancel(self, session_id: str, reason: str) -> bool:
        if not await self.is_session_active(session_id):
            return False
        await self._redis.set(self._k(f"cancel:signal:{session_id}"), reason, ex=60)
        await self._redis.publish(self._k(f"pubsub:cancel:{session_id}"), reason)
        return True

    async def wait_for_cancel(self, session_id: str) -> str:
        sig_key = self._k(f"cancel:signal:{session_id}")
        channel = self._k(f"pubsub:cancel:{session_id}")
        pubsub = self._redis.pubsub() if hasattr(self._redis, "pubsub") else None
        if pubsub is not None:
            await pubsub.subscribe(channel)
        try:
            while True:
                sig = await self._redis.get(sig_key)
                if sig:
                    return str(sig)
                if pubsub is not None:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=self.poll_interval
                    )
                    if msg and msg.get("data"):
                        return str(msg["data"])
                else:
                    await asyncio.sleep(self.poll_interval)
        finally:
            if pubsub is not None:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

    async def publish_event(self, session_id: str, event: HarnessEvent) -> None:
        await self._redis.publish(
            self._k(f"pubsub:events:{session_id}"),
            event.model_dump_json(),
        )

    async def subscribe_events(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        channel = self._k(f"pubsub:events:{session_id}")
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=self.poll_interval
                )
                if msg and msg.get("data"):
                    raw_data = msg["data"]
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode("utf-8", errors="replace")
                    if isinstance(raw_data, str):
                        yield HarnessEvent.model_validate_json(raw_data)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        bucket = int(time.time() // max(1, window_seconds))
        rkey = self._k(f"rl:{key}:{bucket}")
        if hasattr(self._redis, "incr"):
            count = int(await self._redis.incr(rkey))
            if count == 1 and hasattr(self._redis, "expire"):
                await self._redis.expire(rkey, window_seconds + 5)
        else:
            raw = await self._redis.get(rkey)
            count = int(raw or 0) + 1
            await self._redis.set(rkey, str(count), ex=window_seconds + 5)
        if count > limit:
            remaining = max(1, int((bucket + 1) * window_seconds - time.time()))
            return False, remaining
        return True, 0

    async def get_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        raw = await self._redis.get(self._k(f"usage:{date_str}:{tenant_id}:{user_id}"))
        if not raw:
            return {"prompt_tokens": 0, "completion_tokens": 0, "tokens": 0, "cost_usd": 0.0}
        return json.loads(raw)

    async def record_daily_usage(
        self,
        date_str: str,
        tenant_id: str,
        department: str,
        user_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        ukey = self._k(f"usage:{date_str}:{tenant_id}:{user_id}")
        cur = await self.get_daily_usage(date_str, tenant_id, user_id)
        p_tot = int(cur.get("prompt_tokens", 0)) + max(0, int(prompt_tokens))
        c_tot = int(cur.get("completion_tokens", 0)) + max(0, int(completion_tokens))
        cost_tot = round(float(cur.get("cost_usd", 0.0)) + max(0.0, float(cost_usd)), 6)
        record = {
            "date": date_str,
            "tenant_id": tenant_id,
            "department": department,
            "user_id": user_id,
            "prompt_tokens": p_tot,
            "completion_tokens": c_tot,
            "tokens": p_tot + c_tot,
            "cost_usd": cost_tot,
        }
        await self._redis.set(ukey, json.dumps(record, ensure_ascii=False), ex=86400 * 35)
        await self._redis.sadd(self._k(f"usage:index:{date_str}"), f"{tenant_id}:{user_id}")

    async def list_usage_summary(
        self,
        date_str: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        members = await self._redis.smembers(self._k(f"usage:index:{date_str}"))
        out: list[dict[str, Any]] = []
        for member in sorted(members):
            raw = await self._redis.get(self._k(f"usage:{date_str}:{member}"))
            if not raw:
                continue
            item = json.loads(raw)
            if tenant_id is not None and item.get("tenant_id") != tenant_id:
                continue
            out.append(item)
        return out

    async def aclose(self) -> None:
        if self._owns_client and hasattr(self._redis, "aclose"):
            await self._redis.aclose()


def create_coordination_backend(redis_url: str | None = None) -> CoordinationBackend:
    url = (redis_url or "").strip()
    if url:
        return RedisCoordinationBackend(redis_url=url)
    return InMemoryCoordinationBackend()
