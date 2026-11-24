from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    """Tool execution risk tier used by the governance layer."""

    LOW = "low"        # Pure read-only / deterministic compute
    MEDIUM = "medium"  # Network / external search / bounded sandbox
    HIGH = "high"      # Mutating SQL, external side-effects -> triggers Human-in-the-Loop approval


class EventType(StrEnum):
    """Canonical event types persisted in EventJournal and streamed over SSE."""

    TURN_START = "turn_start"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"
    CONTEXT_COMPRESSED = "context_compressed"
    SELF_CORRECTION = "self_correction"
    TOKEN_DELTA = "token_delta"
    TURN_COMPLETE = "turn_complete"
    TURN_CANCELLED = "turn_cancelled"
    ERROR = "error"


class TurnStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR_LIMIT_EXCEEDED = "error_limit_exceeded"
    FAILED = "failed"


class ToolSpec(BaseModel):
    """Declarative contract for a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    risk_level: RiskLevel = RiskLevel.LOW
    timeout_seconds: float = 30.0
    max_retries: int = 0

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    parse_error: str | None = None


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    error_type: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def estimate_tokens(self) -> int:
        """Fast char-based heuristic (~2.5 chars/token for mixed CN/EN text)."""
        text_len = len(self.content)
        for tc in self.tool_calls:
            text_len += len(tc.name) + len(str(tc.arguments))
        return max(1, int(text_len / 2.5) + 4)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.estimated_cost_usd = round(
            self.estimated_cost_usd + other.estimated_cost_usd, 6
        )


class LLMResponse(BaseModel):
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: float = 0.0


class HarnessEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    session_id: str
    turn_id: str
    seq: int
    event_type: EventType
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    payload: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    session_id: str
    turn_id: str
    status: TurnStatus
    final_answer: str = ""
    steps_executed: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    total_duration_ms: float = 0.0
    error_message: str | None = None
    messages: list[Message] = Field(default_factory=list)
