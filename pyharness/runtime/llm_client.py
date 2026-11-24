from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable

import httpx

from pyharness.runtime.schemas import (
    LLMResponse,
    LLMStreamChunk,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class BaseLLMClient(ABC):
    """Unified async LLM interface supporting both single-shot completion and live SSE streaming."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Default streaming adapter delegating to `complete()` so custom/mock subclasses work
        seamlessly without boilerplate, while `OpenAICompatibleLLMClient` overrides with live SSE.
        """
        resp = await self.complete(messages, tools=tools)
        if resp.reasoning_content:
            yield LLMStreamChunk(reasoning_delta=resp.reasoning_content, model=resp.model)
        if resp.content and not resp.tool_calls:
            yield LLMStreamChunk(content_delta=resp.content, model=resp.model)
        yield LLMStreamChunk(
            content_delta=resp.content if resp.tool_calls else "",
            tool_calls=resp.tool_calls,
            usage=resp.usage,
            model=resp.model,
            is_final=True,
        )

    async def aclose(self) -> None:
        return None


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    Production async httpx client for any OpenAI-compatible Chat Completions endpoint
    (DeepSeek, Qwen DashScope, local vLLM/SGLang, OpenAI).

    Enterprise upgrades:
    - Persistent `httpx.AsyncClient` TCP/TLS connection pool reuse.
    - True `stream=True` SSE chunk parsing (real-time `reasoning_content`, `content`,
      and indexed `tool_calls` fragment assembly).
    - Exponential backoff retries on HTTP 429 / 502 / 503 / 504 & transient transport errors.
    - DeepSeek prompt-cache hit accounting (`prompt_cache_hit_tokens`).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout_seconds: float = 60.0,
        input_cost_per_1m: float = 0.27,
        cached_input_cost_per_1m: float = 0.07,
        output_cost_per_1m: float = 1.10,
        max_retries: int = 2,
        retry_base_delay: float = 0.25,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.input_cost_per_1m = input_cost_per_1m
        self.cached_input_cost_per_1m = cached_input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=self.timeout_seconds)

    @staticmethod
    def _serialize_message(msg: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id:
            payload["tool_call_id"] = msg.tool_call_id
        if msg.name:
            payload["name"] = msg.name
        if msg.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments
                        if tc.raw_arguments is not None
                        else json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_usage(self, raw_usage: dict[str, Any] | None) -> TokenUsage:
        u = raw_usage or {}
        p_tok = int(u.get("prompt_tokens", 0))
        c_tok = int(u.get("completion_tokens", 0))
        cache_hit = int(
            u.get("prompt_cache_hit_tokens")
            or (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            or 0
        )
        uncached_p = max(0, p_tok - cache_hit)
        cost = (
            uncached_p * self.input_cost_per_1m
            + cache_hit * self.cached_input_cost_per_1m
            + c_tok * self.output_cost_per_1m
        ) / 1_000_000.0
        return TokenUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            prompt_cache_hit_tokens=cache_hit,
            estimated_cost_usd=round(cost, 6),
        )

    @staticmethod
    def _parse_raw_tool_call(call_id: str, fn_name: str, raw_args: Any) -> ToolCall:
        args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {})
        try:
            parsed_args = json.loads(args_str) if args_str.strip() else {}
            parse_err = None
        except json.JSONDecodeError as exc:
            parsed_args = {}
            parse_err = f"Invalid JSON arguments: {exc}"
        return ToolCall(
            id=call_id or f"call_{int(time.time() * 1000)}",
            name=fn_name,
            arguments=parsed_args if isinstance(parsed_args, dict) else {},
            raw_arguments=args_str if isinstance(raw_args, str) else None,
            parse_error=parse_err,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._serialize_message(m) for m in messages],
        }
        if tools:
            body["tools"] = [t.to_openai_schema() for t in tools]
            body["tool_choice"] = "auto"

        t0 = time.perf_counter()
        last_exc: Exception | None = None
        data: dict[str, Any] = {}
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=body,
                )
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                last_exc = None
                break
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))
                    continue
                raise
        if last_exc is not None:
            raise last_exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = data["choices"][0]["message"]
        parsed_calls = [
            self._parse_raw_tool_call(
                raw_tc.get("id", ""),
                raw_tc["function"]["name"],
                raw_tc["function"].get("arguments", "{}"),
            )
            for raw_tc in (choice.get("tool_calls") or [])
        ]

        return LLMResponse(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content") or "",
            tool_calls=parsed_calls,
            usage=self._build_usage(data.get("usage")),
            model=data.get("model", self.model),
            latency_ms=round(latency_ms, 2),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._serialize_message(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [t.to_openai_schema() for t in tools]
            body["tool_choice"] = "auto"

        tc_buffers: dict[int, dict[str, str]] = {}
        final_usage: TokenUsage | None = None
        seen_model = self.model

        for attempt in range(self.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=body,
                ) as resp:
                    if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        await resp.aread()
                        await asyncio.sleep(self.retry_base_delay * (2**attempt))
                        continue
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        frame = json.loads(data_str)
                        seen_model = frame.get("model") or seen_model
                        if frame.get("usage"):
                            final_usage = self._build_usage(frame["usage"])

                        choices = frame.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}

                        r_delta = delta.get("reasoning_content") or ""
                        c_delta = delta.get("content") or ""
                        if r_delta or c_delta:
                            yield LLMStreamChunk(
                                content_delta=c_delta,
                                reasoning_delta=r_delta,
                                model=seen_model,
                            )

                        for tc_part in delta.get("tool_calls") or []:
                            idx = int(tc_part.get("index", 0))
                            slot = tc_buffers.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if tc_part.get("id"):
                                slot["id"] = tc_part["id"]
                            fn_part = tc_part.get("function") or {}
                            if fn_part.get("name"):
                                slot["name"] += fn_part["name"]
                            if fn_part.get("arguments"):
                                slot["arguments"] += fn_part["arguments"]
                    break
            except httpx.TransportError:
                if attempt < self.max_retries and not tc_buffers:
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))
                    continue
                raise

        assembled_calls = [
            self._parse_raw_tool_call(
                slot["id"],
                slot["name"],
                slot["arguments"],
            )
            for _, slot in sorted(tc_buffers.items(), key=lambda x: x[0])
            if slot["name"]
        ]

        yield LLMStreamChunk(
            tool_calls=assembled_calls,
            usage=final_usage or TokenUsage(),
            model=seen_model,
            is_final=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class ScriptedMockLLMClient(BaseLLMClient):
    """
    Deterministic LLM test double (inspired by DSH deterministic-model-plugin.mjs).
    Supports both single-shot responses and multi-chunk simulated token streaming (`chunk_size`).
    """

    def __init__(
        self,
        responses: list[LLMResponse | Callable[[list[Message]], LLMResponse]],
        chunk_size: int | None = None,
    ) -> None:
        self._responses = list(responses)
        self.chunk_size = chunk_size
        self.call_history: list[list[Message]] = []
        self.tools_history: list[list[str]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.call_history.append([m.model_copy(deep=True) for m in messages])
        self.tools_history.append([t.name for t in (tools or [])])
        if not self._responses:
            return LLMResponse(content="(mock exhausted)", model="scripted-mock")
        nxt = self._responses.pop(0)
        if callable(nxt):
            return nxt(messages)
        return nxt

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        if self.chunk_size is None or self.chunk_size <= 0:
            async for ch in super().stream(messages, tools=tools):
                yield ch
            return

        resp = await self.complete(messages, tools=tools)
        step = self.chunk_size
        if resp.reasoning_content:
            for i in range(0, len(resp.reasoning_content), step):
                yield LLMStreamChunk(
                    reasoning_delta=resp.reasoning_content[i : i + step],
                    model=resp.model,
                )
        if resp.content and not resp.tool_calls:
            for i in range(0, len(resp.content), step):
                yield LLMStreamChunk(
                    content_delta=resp.content[i : i + step],
                    model=resp.model,
                )
        yield LLMStreamChunk(
            content_delta=resp.content if resp.tool_calls else "",
            tool_calls=resp.tool_calls,
            usage=resp.usage,
            model=resp.model,
            is_final=True,
        )
