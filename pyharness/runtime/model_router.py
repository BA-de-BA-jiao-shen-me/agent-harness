from __future__ import annotations

import inspect
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator

from pyharness.runtime.llm_client import BaseLLMClient
from pyharness.runtime.schemas import LLMResponse, LLMStreamChunk, Message, ToolSpec


@lru_cache(maxsize=256)
def _func_supports_kwarg(func: Any, name: str, is_bound_method: bool) -> bool:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if is_bound_method and params:
        params = params[1:]
    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _supports_kwarg(fn: Any, name: str) -> bool:
    """Return True if `fn` accepts keyword argument `name` (explicitly or via `**kwargs`)."""
    if inspect.ismethod(fn):
        return _func_supports_kwarg(fn.__func__, name, True)
    try:
        return _func_supports_kwarg(fn, name, False)
    except TypeError:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
            if param.name == name and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False


@dataclass
class ModelRoute:
    """Declarative route entry binding a named LLM client to a routing tier, priority, and weight."""

    name: str
    client: BaseLLMClient
    tier: str = "reasoning"
    priority: int = 0
    weight: int = 100

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"ModelRoute.weight must be >= 0, got {self.weight!r} for route {self.name!r}."
            )


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
        prompt_cache_key: str | None = None,
    ) -> LLMResponse:
        return await self._router.complete(
            messages,
            tools=tools,
            tier=self._tier,
            prompt_cache_key=prompt_cache_key,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        prompt_cache_key: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        async for chunk in self._router.stream(
            messages,
            tools=tools,
            tier=self._tier,
            prompt_cache_key=prompt_cache_key,
        ):
            yield chunk

    async def aclose(self) -> None:
        # Shared router lifecycle is owned by ModelRouterLLMClient, not tier proxies.
        return None


class ModelRouterLLMClient(BaseLLMClient):
    """
    Multi-Provider Model Router with tiered routing (`reasoning` / `fast` / custom),
    weighted traffic allocation within equal-priority route groups, automatic
    primary-to-fallback failover, and per-route circuit breakers with half-open
    recovery probes.
    """

    def __init__(
        self,
        routes: list[ModelRoute],
        default_tier: str = "reasoning",
        failure_threshold: int = 2,
        cooldown_seconds: float = 30.0,
        rng: random.Random | None = None,
    ) -> None:
        if not routes:
            raise ValueError("ModelRouterLLMClient requires at least one ModelRoute.")
        for route in routes:
            if route.weight < 0:
                raise ValueError(
                    f"ModelRoute.weight must be >= 0, got {route.weight!r} for route {route.name!r}."
                )
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
        self._rng: random.Random = rng if rng is not None else random.Random()
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

    def _order_priority_group(self, group: list[ModelRoute]) -> list[ModelRoute]:
        if len(group) <= 1:
            return list(group)

        positive = [r for r in group if r.weight > 0]
        standby = [r for r in group if r.weight <= 0]
        if not positive:
            return list(group)

        pool = list(positive)
        ordered: list[ModelRoute] = []
        while len(pool) > 1:
            total_weight = sum(r.weight for r in pool)
            pick = self._rng.uniform(0.0, float(total_weight))
            acc = 0.0
            chosen_idx = len(pool) - 1
            for idx, candidate in enumerate(pool):
                acc += candidate.weight
                if pick < acc:
                    chosen_idx = idx
                    break
            ordered.append(pool.pop(chosen_idx))
        if pool:
            ordered.append(pool[0])
        ordered.extend(standby)
        return ordered

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
            groups: dict[int, list[ModelRoute]] = {}
            for route in healthy:
                groups.setdefault(route.priority, []).append(route)
            ordered_candidates: list[ModelRoute] = []
            for priority in sorted(groups.keys()):
                ordered_candidates.extend(self._order_priority_group(groups[priority]))
            return ordered_candidates

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
        prompt_cache_key: str | None = None,
    ) -> LLMResponse:
        candidates = self._select_candidates(tier)
        last_exc: Exception | None = None
        for route in candidates:
            try:
                kwargs: dict[str, Any] = {}
                if prompt_cache_key is not None and _supports_kwarg(
                    route.client.complete, "prompt_cache_key"
                ):
                    kwargs["prompt_cache_key"] = prompt_cache_key
                response = await route.client.complete(messages, tools=tools, **kwargs)
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
        prompt_cache_key: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        candidates = self._select_candidates(tier)
        last_exc: Exception | None = None
        for route in candidates:
            yielded_any = False
            try:
                kwargs: dict[str, Any] = {}
                if prompt_cache_key is not None and _supports_kwarg(
                    route.client.stream, "prompt_cache_key"
                ):
                    kwargs["prompt_cache_key"] = prompt_cache_key
                async for chunk in route.client.stream(messages, tools=tools, **kwargs):
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
                    "weight": route.weight,
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
