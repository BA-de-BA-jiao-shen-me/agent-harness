from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import httpx

from pyharness.runtime.schemas import LLMResponse, Message, TokenUsage, ToolCall, ToolSpec


class BaseLLMClient(ABC):
    """Unified async LLM interface decoupling the Agent Harness from any single vendor SDK."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    Direct async httpx client for any OpenAI-compatible Chat Completions endpoint
    (DeepSeek, Qwen DashScope, local vLLM/SGLang, OpenAI).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout_seconds: float = 60.0,
        input_cost_per_1m: float = 0.27,
        output_cost_per_1m: float = 1.10,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m

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
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = data["choices"][0]["message"]
        parsed_calls: list[ToolCall] = []
        for raw_tc in choice.get("tool_calls") or []:
            fn = raw_tc["function"]
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                parse_err = None
            except json.JSONDecodeError as exc:
                parsed_args = {}
                parse_err = f"Invalid JSON arguments: {exc}"
            parsed_calls.append(
                ToolCall(
                    id=raw_tc.get("id", ""),
                    name=fn["name"],
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    raw_arguments=raw_args if isinstance(raw_args, str) else None,
                    parse_error=parse_err,
                )
            )

        raw_usage = data.get("usage") or {}
        p_tok = int(raw_usage.get("prompt_tokens", 0))
        c_tok = int(raw_usage.get("completion_tokens", 0))
        cost = (p_tok * self.input_cost_per_1m + c_tok * self.output_cost_per_1m) / 1_000_000.0

        return LLMResponse(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content") or "",
            tool_calls=parsed_calls,
            usage=TokenUsage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                estimated_cost_usd=round(cost, 6),
            ),
            model=data.get("model", self.model),
            latency_ms=round(latency_ms, 2),
        )


class ScriptedMockLLMClient(BaseLLMClient):
    """
    Deterministic LLM test double (inspired by DSH deterministic-model-plugin.mjs).
    Enables fast, zero-cost, 100% reproducible unit tests for cancellation, approvals,
    parallel tool execution, self-correction, and context compression.
    """

    def __init__(
        self,
        responses: list[LLMResponse | Callable[[list[Message]], LLMResponse]],
    ) -> None:
        self._responses = list(responses)
        self.call_history: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        # Snapshot message list for assertions
        self.call_history.append([m.model_copy(deep=True) for m in messages])
        if not self._responses:
            return LLMResponse(content="(mock exhausted)", model="scripted-mock")
        nxt = self._responses.pop(0)
        if callable(nxt):
            return nxt(messages)
        return nxt
