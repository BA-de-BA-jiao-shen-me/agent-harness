from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from pyharness.runtime.llm_client import BaseLLMClient
from pyharness.runtime.schemas import LLMResponse, LLMStreamChunk, Message, ToolSpec


@dataclass
class ModelRoute:
    """Declarative route entry binding a named LLM client to a routing tier and priority."""

    name: str
    client: BaseLLMClient
    tier: str = "reasoning"
    priority: int = 0


@dataclass
class _CircuitBreakerState:
    failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    open_until: float = 0.0
    last_error: str | None = None


class _TierScopedLLMProxy(BaseLLMClient):
    """
    Lightweight `BaseLLMClient` proxy pinned to a specific routing tier (e.g. `tier="fast"`),
    allowing `ContextCompressor`, `QueryRewriter`, and `SubagentManager` to route transparently.
    """

    def __init__(self, router: ModelRouterLLMClient, tier: str) -> None:
        self._router = router
        self._tier = tier

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        return await self._router.complete(messages, tools=tools, tier=self._tier)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        async for chunk in self._router.stream(messages, tools=tools, tier=self._tier):
            yield chunk

    async def aclose(self) -> None:
        # Shared router lifecycle is owned by ModelRouterLLMClient, not tier proxies.
        return None


class ModelRouterLLMClient(BaseLLMClient):
    """
    Multi-Provider Model Router with tiered routing (`reasoning` / `fast` / custom),
    automatic primary-to-fallback failover, and per-route circuit breakers with
    half-open recovery probes.
    """

    def __init__(
        self,
        routes: list[ModelRoute],
        default_tier: str = "reasoning",
        failure_threshold: int = 2,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if not routes:
            raise ValueError("ModelRouterLLMClient requires at least one ModelRoute.")
        # Preserve relative order for routes sharing the same priority
        self._routes: list[ModelRoute] = [
            route
            for _, route in sorted(
                enumerate(routes),
                key=lambda item: (item[1].priority, item[0]),
            )
        ]
        self.default_tier = default_tier
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._breakers: dict[str, _CircuitBreakerState] = {
            r.name: _CircuitBreakerState() for r in self._routes
        }

    def _is_open(self, route_name: str, now: float) -> bool:
        state = self._breakers[route_name]
        return state.open_until > now

    def _record_success(self, route_name: str) -> None:
        state = self._breakers[route_name]
        state.failures = 0
        state.open_until = 0.0
        state.last_error = None
        state.total_successes += 1

    def _record_failure(self, route_name: str, exc: Exception) -> None:
        state = self._breakers[route_name]
        state.failures += 1
        state.total_failures += 1
        state.last_error = f"{type(exc).__name__}: {exc}"
        if state.failures >= self.failure_threshold:
            state.open_until = time.monotonic() + self.cooldown_seconds

    def _select_candidates(self, tier: str | None) -> list[ModelRoute]:
        target_tier = tier or self.default_tier
        tier_routes = [r for r in self._routes if r.tier == target_tier]
        if not tier_routes and target_tier != self.default_tier:
            tier_routes = [r for r in self._routes if r.tier == self.default_tier]
        if not tier_routes:
            tier_routes = list(self._routes)

        now = time.monotonic()
        healthy = [r for r in tier_routes if not self._is_open(r.name, now)]
        if healthy:
            return healthy

        # All routes for this tier are open -> half-open probe fallback ordered by earliest expiry
        return sorted(
            tier_routes,
            key=lambda r: (self._breakers[r.name].open_until, r.priority),
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tier: str | None = None,
    ) -> LLMResponse:
        candidates = self._select_candidates(tier)
        last_exc: Exception | None = None
        for route in candidates:
            try:
                response = await route.client.complete(messages, tools=tools)
                self._record_success(route.name)
                return response
            except Exception as exc:
                self._record_failure(route.name, exc)
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tier: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        candidates = self._select_candidates(tier)
        last_exc: Exception | None = None
        for route in candidates:
            yielded_any = False
            try:
                async for chunk in route.client.stream(messages, tools=tools):
                    yielded_any = True
                    yield chunk
                self._record_success(route.name)
                return
            except Exception as exc:
                self._record_failure(route.name, exc)
                if yielded_any:
                    # Cannot transparently switch routes after partial tokens were already emitted
                    raise
                last_exc = exc
        if last_exc is not None:
            raise last_exc

    def for_tier(self, tier: str) -> BaseLLMClient:
        """Return a `BaseLLMClient` proxy pinned to `tier` (e.g. `'fast'` or `'reasoning'`)."""
        return _TierScopedLLMProxy(self, tier)

    def route_health(self) -> list[dict[str, Any]]:
        """Return per-route circuit breaker and traffic telemetry for `/health`."""
        now = time.monotonic()
        out: list[dict[str, Any]] = []
        for route in self._routes:
            state = self._breakers[route.name]
            is_open = state.open_until > now
            out.append(
                {
                    "name": route.name,
                    "tier": route.tier,
                    "priority": route.priority,
                    "state": "open" if is_open else "closed",
                    "failures": state.failures,
                    "total_failures": state.total_failures,
                    "total_successes": state.total_successes,
                    "cooldown_remaining_seconds": round(max(0.0, state.open_until - now), 2)
                    if is_open
                    else 0.0,
                    "last_error": state.last_error,
                }
            )
        return out

    async def aclose(self) -> None:
        seen_ids: set[int] = set()
        for route in self._routes:
            cid = id(route.client)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            await route.client.aclose()
