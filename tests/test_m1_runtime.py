from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from pyharness.runtime import (
    AgentHarnessEngine,
    ApprovalBroker,
    CancellationToken,
    EventJournal,
    EventType,
    HarnessConfig,
    LLMResponse,
    Message,
    RiskLevel,
    ScriptedMockLLMClient,
    TokenUsage,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    TurnStatus,
)


async def test_parallel_tool_execution_reduces_latency() -> None:
    """Verify asyncio.gather executes multiple independent tool calls concurrently."""
    tools = ToolRegistry()

    async def slow_lookup(key: str) -> str:
        await asyncio.sleep(0.15)
        return f"value_for_{key}"

    tools.register(
        ToolSpec(
            name="slow_lookup",
            description="Simulated network/DB lookup",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        ),
        slow_lookup,
    )

    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                reasoning_content="Fetch both A and B concurrently.",
                tool_calls=[
                    ToolCall(id="c1", name="slow_lookup", arguments={"key": "A"}),
                    ToolCall(id="c2", name="slow_lookup", arguments={"key": "B"}),
                ],
                usage=TokenUsage(prompt_tokens=100, completion_tokens=30),
            ),
            LLMResponse(
                content="Combined result: value_for_A & value_for_B",
                usage=TokenUsage(prompt_tokens=150, completion_tokens=20),
            ),
        ]
    )

    engine = AgentHarnessEngine(llm=llm, tools=tools)
    t0 = time.perf_counter()
    res = await engine.run_turn("sess_parallel", "Lookup A and B")
    elapsed = time.perf_counter() - t0

    assert res.status == TurnStatus.COMPLETED
    assert "value_for_A" in res.final_answer
    # Serial would take >= 0.30s; parallel should finish comfortably under 0.26s
    assert elapsed < 0.26, f"Expected parallel speedup (<0.26s), got {elapsed:.3f}s"
    assert res.usage.total_tokens == 300


async def test_high_risk_tool_approval_approve_and_reject() -> None:
    """Verify Human-in-the-Loop ApprovalBroker gates HIGH-risk tool execution."""
    executed_queries: list[str] = []
    tools = ToolRegistry()

    async def execute_sql(sql: str) -> str:
        executed_queries.append(sql)
        return f"EXECUTED:{sql}"

    tools.register(
        ToolSpec(
            name="execute_sql",
            description="Run mutating SQL",
            parameters={"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
            risk_level=RiskLevel.HIGH,
        ),
        execute_sql,
    )

    broker = ApprovalBroker()
    journal = EventJournal(":memory:")
    sub = journal.subscribe("sess_approval")

    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c_sql", name="execute_sql", arguments={"sql": "DELETE FROM tmp"})]
            ),
            LLMResponse(content="Done after approval."),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=tools, journal=journal, approval_broker=broker)

    async def auto_reviewer(approve: bool) -> None:
        while True:
            evt = await sub.get()
            if evt.event_type == EventType.APPROVAL_REQUIRED:
                broker.resolve(
                    evt.payload["approval_id"],
                    approved=approve,
                    reviewer="qa_admin",
                    note="verified" if approve else "too risky",
                )
                return

    # 1. Approve path
    reviewer_task = asyncio.create_task(auto_reviewer(approve=True))
    res_ok = await engine.run_turn("sess_approval", "Clean tmp table")
    await reviewer_task
    assert res_ok.status == TurnStatus.COMPLETED
    assert executed_queries == ["DELETE FROM tmp"]

    # 2. Reject path -> tool must NOT run
    executed_queries.clear()
    llm_reject = ScriptedMockLLMClient(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c_sql2", name="execute_sql", arguments={"sql": "DROP TABLE prod"})]
            ),
            LLMResponse(content="Understood, I will not run DROP TABLE."),
        ]
    )
    engine_reject = AgentHarnessEngine(llm=llm_reject, tools=tools, journal=journal, approval_broker=broker)
    reviewer_task2 = asyncio.create_task(auto_reviewer(approve=False))
    res_denied = await engine_reject.run_turn("sess_approval", "Drop prod table")
    await reviewer_task2

    assert res_denied.status == TurnStatus.COMPLETED
    assert executed_queries == []
    assert any(m.role == "tool" and "Approval Denied" in m.content for m in res_denied.messages)


async def test_session_cancellation_aborts_pending_tool() -> None:
    """Verify CancellationToken immediately aborts a slow running tool and logs TURN_CANCELLED."""
    tools = ToolRegistry()

    async def very_slow_tool() -> str:
        await asyncio.sleep(5.0)
        return "should_never_finish"

    tools.register(ToolSpec(name="very_slow_tool", description="Slow task"), very_slow_tool)
    llm = ScriptedMockLLMClient(
        [LLMResponse(tool_calls=[ToolCall(id="c_slow", name="very_slow_tool", arguments={})])]
    )
    journal = EventJournal(":memory:")
    engine = AgentHarnessEngine(llm=llm, tools=tools, journal=journal)
    cancel = CancellationToken()

    async def trigger_cancel() -> None:
        await asyncio.sleep(0.08)
        cancel.cancel("user_clicked_stop")

    cancel_task = asyncio.create_task(trigger_cancel())
    t0 = time.perf_counter()
    res = await engine.run_turn("sess_cancel", "Start slow job", cancellation=cancel)
    elapsed = time.perf_counter() - t0
    await cancel_task

    assert res.status == TurnStatus.CANCELLED
    assert elapsed < 0.5
    events = journal.replay_session("sess_cancel")
    assert events[-1].event_type == EventType.TURN_CANCELLED


async def test_self_correction_recovery_and_loop_breaker() -> None:
    """Verify bad arguments trigger Self-Correction and repeated failures trip the breaker."""
    tools = ToolRegistry()

    def divide(a: float, b: float) -> str:
        if b == 0:
            raise ValueError("division by zero")
        return str(a / b)

    tools.register(
        ToolSpec(
            name="divide",
            description="Divide a by b",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        ),
        divide,
    )

    # Case 1: LLM makes an error on step 1 (missing arg / div by 0) -> repairs on step 2
    llm_recover = ScriptedMockLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="divide", arguments={"a": 10, "b": 0})]),
            LLMResponse(tool_calls=[ToolCall(id="c2", name="divide", arguments={"a": 10, "b": 2})]),
            LLMResponse(content="Fixed! 10 / 2 = 5.0"),
        ]
    )
    journal = EventJournal(":memory:")
    engine = AgentHarnessEngine(llm=llm_recover, tools=tools, journal=journal)
    res = await engine.run_turn("sess_sc", "Compute division")
    assert res.status == TurnStatus.COMPLETED
    assert "5.0" in res.final_answer
    trace = journal.build_trace_summary("sess_sc", res.turn_id)
    assert trace["self_corrections"] == 1

    # Case 2: LLM keeps failing -> breaker trips at max_consecutive_errors=2
    llm_stuck = ScriptedMockLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="e1", name="divide", arguments={"a": 1, "b": 0})]),
            LLMResponse(tool_calls=[ToolCall(id="e2", name="divide", arguments={"a": 1, "b": 0})]),
            LLMResponse(content="Should not reach here"),
        ]
    )
    engine_breaker = AgentHarnessEngine(
        llm=llm_stuck,
        tools=tools,
        config=HarnessConfig(max_consecutive_errors=2),
    )
    res_stuck = await engine_breaker.run_turn("sess_stuck", "Broken loop")
    assert res_stuck.status == TurnStatus.ERROR_LIMIT_EXCEEDED
    assert res_stuck.steps_executed == 2


async def test_incremental_context_compression_triggers_cleanly() -> None:
    """Verify long history is automatically compressed while keeping system + recent turns intact."""
    tools = ToolRegistry()
    journal = EventJournal(":memory:")

    # Create bloated historical messages exceeding threshold_tokens=200
    long_history = [
        Message(role="system", content="CRITICAL_SYSTEM_POLICY"),
        *[
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Turn {i}: " + ("A" * 200))
            for i in range(8)
        ],
    ]
    llm = ScriptedMockLLMClient([LLMResponse(content="Answered with compressed context.")])
    engine = AgentHarnessEngine(
        llm=llm,
        tools=tools,
        journal=journal,
        config=HarnessConfig(context_compress_threshold=200, keep_recent_messages=3),
    )
    res = await engine.run_turn("sess_comp", "Latest question", history=long_history)

    assert res.status == TurnStatus.COMPLETED
    sent_messages = llm.call_history[0]
    assert sent_messages[0].content == "CRITICAL_SYSTEM_POLICY"
    assert any("[Incremental Context Summary]" in m.content for m in sent_messages)
    assert len(sent_messages) < len(long_history) + 1
    trace = journal.build_trace_summary("sess_comp", res.turn_id)
    assert trace["context_compressions"] == 1


async def test_event_journal_sqlite_persistence_and_replay(tmp_path: Path) -> None:
    """Verify durable SQLite persistence and strict monotonic sequence replay across instances."""
    db_file = tmp_path / "journal.sqlite3"
    journal1 = EventJournal(db_file)
    llm = ScriptedMockLLMClient([LLMResponse(content="Hello persisted world!")])
    engine = AgentHarnessEngine(llm=llm, tools=ToolRegistry(), journal=journal1)

    res = await engine.run_turn("sess_persist", "Ping")
    journal1.close()

    # Re-open fresh journal pointing at the same SQLite file
    journal2 = EventJournal(db_file)
    replayed = journal2.replay_session("sess_persist")
    seqs = [e.seq for e in replayed]
    assert seqs == list(range(1, len(replayed) + 1))
    assert replayed[0].event_type == EventType.TURN_START
    assert replayed[-1].event_type == EventType.TURN_COMPLETE
    summary = journal2.build_trace_summary("sess_persist", res.turn_id)
    assert summary["status"] == "completed"
    journal2.close()


async def test_token_budget_guard_halts_runaway_turn() -> None:
    """Verify TokenBudgetGuard stops execution immediately if token budget is exceeded."""
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                content="Partial",
                tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                usage=TokenUsage(prompt_tokens=800, completion_tokens=300),
            )
        ]
    )
    engine = AgentHarnessEngine(
        llm=llm,
        tools=ToolRegistry(),
        config=HarnessConfig(max_tokens_per_turn=1000),
    )
    res = await engine.run_turn("sess_budget", "Expensive prompt")
    assert res.status == TurnStatus.BUDGET_EXCEEDED
    assert res.usage.total_tokens == 1100
