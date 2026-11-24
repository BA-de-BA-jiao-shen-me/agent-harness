from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pyharness.mcp import MCPClientBridge, create_builtin_enterprise_mcp_server
from pyharness.runtime import (
    AgentHarnessEngine,
    ApprovalBroker,
    EventJournal,
    EventType,
    LLMResponse,
    ScriptedMockLLMClient,
    ToolCall,
    ToolRegistry,
    TurnStatus,
)
from pyharness.sandbox import PythonSandbox, SQLSafetyError, SQLSandbox, SandboxSecurityError
from pyharness.skills import SkillRouter


def test_skill_router_loads_and_selects_dynamically() -> None:
    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    router = SkillRouter(skills_dir)
    assert len(router.list_skills()) >= 2

    prompt, selected = router.build_augmented_prompt(
        "Base prompt", "帮我写一条 SQL 统计本月高压断路器故障数"
    )
    assert [s.name for s in selected] == ["data_analysis"]
    assert "inspect_sql_schema" in prompt


async def test_mcp_bridge_discovers_and_enforces_approval_on_high_risk_tool() -> None:
    mcp_server = create_builtin_enterprise_mcp_server()
    bridge = MCPClientBridge(server=mcp_server)
    registry = ToolRegistry()
    mounted = await bridge.mount_into_registry(registry)

    assert "mcp_list_incidents" in mounted
    assert "mcp_dispatch_emergency_alert" in mounted

    # Run low-risk MCP tool + high-risk MCP tool through full Harness governance
    broker = ApprovalBroker()
    journal = EventJournal(":memory:")
    sub = journal.subscribe("sess_mcp")
    llm = ScriptedMockLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="mcp_list_incidents", arguments={"severity": "high"})
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="mcp_dispatch_emergency_alert",
                        arguments={"incident_id": "INC-2026-01"},
                    )
                ]
            ),
            LLMResponse(content="已查询高危工单并经审批发送应急告警。"),
        ]
    )
    engine = AgentHarnessEngine(llm=llm, tools=registry, journal=journal, approval_broker=broker)

    async def approve_once() -> None:
        while True:
            evt = await sub.get()
            if evt.event_type == EventType.APPROVAL_REQUIRED:
                broker.resolve(evt.payload["approval_id"], approved=True, reviewer="ops_lead")
                return

    approver = asyncio.create_task(approve_once())
    res = await engine.run_turn("sess_mcp", "查高危隐患并告警")
    await approver

    assert res.status == TurnStatus.COMPLETED
    trace = journal.build_trace_summary("sess_mcp", res.turn_id)
    assert trace["approvals_requested"] == 1
    assert len(trace["tool_spans"]) == 2


async def test_sql_sandbox_security_and_self_correction_loop() -> None:
    sandbox = SQLSandbox(":memory:")
    sandbox.seed_demo_enterprise_data()

    # 1. Direct security checks: block mutation & multi-statement injection
    with pytest.raises(SQLSafetyError):
        sandbox.query_readonly("DELETE FROM equipment_incidents")
    with pytest.raises(SQLSafetyError):
        sandbox.query_readonly("SELECT * FROM equipment_incidents; DROP TABLE equipment_incidents")

    # 2. End-to-end Self-Correction: LLM hallucinates column name -> SQLite error -> LLM fixes it
    registry = ToolRegistry()
    sandbox.register_tools(registry)
    journal = EventJournal(":memory:")

    llm = ScriptedMockLLMClient(
        [
            # Step 1: bad column name `fault_minutes`
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="query_sql_readonly",
                        arguments={"sql": "SELECT SUM(fault_minutes) FROM equipment_incidents"},
                    )
                ]
            ),
            # Step 2: self-correct by inspecting schema first
            LLMResponse(
                tool_calls=[ToolCall(id="s2", name="inspect_sql_schema", arguments={})]
            ),
            # Step 3: corrected query using true column `downtime_minutes`
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="s3",
                        name="query_sql_readonly",
                        arguments={
                            "sql": "SELECT SUM(downtime_minutes) AS total_down FROM equipment_incidents WHERE severity='high'"
                        },
                    )
                ]
            ),
            LLMResponse(content="高危故障总停机时长为 355 分钟。"),
        ]
    )

    engine = AgentHarnessEngine(llm=llm, tools=registry, journal=journal)
    res = await engine.run_turn("sess_sql_sc", "统计所有高危故障总停机时间")

    assert res.status == TurnStatus.COMPLETED
    assert "355" in res.final_answer
    assert any("355" in m.content for m in res.messages if m.role == "tool")
    trace = journal.build_trace_summary("sess_sql_sc", res.turn_id)
    assert trace["self_corrections"] == 1
    sandbox.close()


def test_python_sandbox_executes_math_and_blocks_escapes() -> None:
    sb = PythonSandbox()
    out = sb.run("import statistics\nvals = [95, 120, 60, 80]\nresult = statistics.mean(vals)\nprint(f'avg={result}')")
    assert '"result": 88.75' in out
    assert "avg=88.75" in out

    # Security breakout tests
    for malicious in [
        "import os\nos.system('whoami')",
        "open('/etc/passwd').read()",
        "().__class__.__base__.__subclasses__()",
        "eval('1+1')",
    ]:
        with pytest.raises(SandboxSecurityError):
            sb.run(malicious)
