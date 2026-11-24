"""
Automated Agent Harness Latency, Self-Correction & Badcase Attribution Suite.

Measures:
1. Serial (`max_parallel_tools=1`) vs Parallel (`max_parallel_tools=4`) tool scheduling latency (P50/P95).
2. SQL Self-Correction recovery rate across realistic malformed-query tasks.
3. Exports failed trajectories with root-cause tags into JSONL (`eval/out/badcases.jsonl`)
   ready for Prompt iteration or LoRA / DPO post-training data flywheel.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from pyharness.runtime import (
    AgentHarnessEngine,
    EventJournal,
    EventType,
    HarnessConfig,
    LLMResponse,
    ScriptedMockLLMClient,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    TurnStatus,
)
from pyharness.sandbox import SQLSandbox


async def bench_parallel_vs_serial(trials: int = 8) -> dict[str, float]:
    async def io_tool(slot: str) -> str:
        await asyncio.sleep(0.06)
        return f"ok:{slot}"

    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            name="io_tool",
            description="Simulated 60ms external RPC/DB lookup",
            parameters={"type": "object", "properties": {"slot": {"type": "string"}}, "required": ["slot"]},
        ),
        io_tool,
    )

    async def measure(max_parallel: int) -> list[float]:
        latencies: list[float] = []
        for i in range(trials):
            llm = ScriptedMockLLMClient(
                [
                    LLMResponse(
                        tool_calls=[
                            ToolCall(id=f"c1_{i}", name="io_tool", arguments={"slot": "A"}),
                            ToolCall(id=f"c2_{i}", name="io_tool", arguments={"slot": "B"}),
                            ToolCall(id=f"c3_{i}", name="io_tool", arguments={"slot": "C"}),
                        ]
                    ),
                    LLMResponse(content="done"),
                ]
            )
            eng = AgentHarnessEngine(
                llm=llm,
                tools=tools,
                config=HarnessConfig(max_parallel_tools=max_parallel),
            )
            t0 = time.perf_counter()
            res = await eng.run_turn(f"s_{max_parallel}_{i}", "Run 3 lookups")
            assert res.status == TurnStatus.COMPLETED
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return latencies

    serial_ms = await measure(max_parallel=1)
    parallel_ms = await measure(max_parallel=4)
    p50_serial = statistics.median(serial_ms)
    p50_parallel = statistics.median(parallel_ms)
    reduction_pct = (1.0 - p50_parallel / p50_serial) * 100.0

    return {
        "serial_p50_ms": round(p50_serial, 2),
        "parallel_p50_ms": round(p50_parallel, 2),
        "latency_reduction_pct": round(reduction_pct, 1),
    }


async def bench_self_correction_and_export_badcases(out_jsonl: Path) -> dict[str, object]:
    sb = SQLSandbox(":memory:")
    sb.seed_demo_enterprise_data()
    registry = ToolRegistry()
    sb.register_tools(registry)
    journal = EventJournal(":memory:")

    scenarios = [
        # Scenario 1: Bad column -> schema inspect -> repaired SELECT (Recovered!)
        (
            "sc_recover_1",
            "统计高危故障数量",
            [
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT COUNT(*) FROM incidents_wrong"})]),
                LLMResponse(tool_calls=[ToolCall(name="inspect_sql_schema", arguments={})]),
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT COUNT(*) FROM equipment_incidents WHERE severity='high'"})]),
                LLMResponse(content="共 4 条高危故障。"),
            ],
        ),
        # Scenario 2: Injection attempt blocked -> repaired SELECT (Recovered!)
        (
            "sc_recover_2",
            "查询张江片区停机时长",
            [
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT * FROM equipment_incidents; DROP TABLE equipment_incidents"})]),
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT SUM(downtime_minutes) FROM equipment_incidents WHERE region='张江片区'"})]),
                LLMResponse(content="张江片区总停机 135 分钟。"),
            ],
        ),
        # Scenario 3: Unrecoverable repeated syntax errors -> breaker trips -> exported as Badcase!
        (
            "sc_badcase_1",
            "无法修复的复杂查询",
            [
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT bad_col FROM bad_tbl"})]),
                LLMResponse(tool_calls=[ToolCall(name="query_sql_readonly", arguments={"sql": "SELECT still_bad FROM bad_tbl"})]),
            ],
        ),
    ]

    recovered = 0
    badcases: list[dict[str, object]] = []

    for sid, prompt, scripted in scenarios:
        eng = AgentHarnessEngine(
            llm=ScriptedMockLLMClient(scripted),
            tools=registry,
            journal=journal,
            config=HarnessConfig(max_consecutive_errors=2),
        )
        res = await eng.run_turn(sid, prompt)
        trace = journal.build_trace_summary(sid, res.turn_id)
        if res.status == TurnStatus.COMPLETED and trace["self_corrections"] > 0:
            recovered += 1
        elif res.status != TurnStatus.COMPLETED:
            events = journal.replay_session(sid)
            err_types = [
                e.payload.get("error_type")
                for e in events
                if e.event_type == EventType.TOOL_RESULT and e.payload.get("is_error")
            ]
            badcases.append(
                {
                    "session_id": sid,
                    "prompt": prompt,
                    "final_status": res.status.value,
                    "root_cause_tags": err_types,
                    "rejected_trajectory": [m.model_dump() for m in res.messages],
                }
            )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in badcases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    sb.close()
    return {
        "total_scenarios": len(scenarios),
        "self_corrected_success": recovered,
        "badcases_exported": len(badcases),
        "badcase_file": str(out_jsonl),
    }


async def main() -> None:
    lat = await bench_parallel_vs_serial()
    sc = await bench_self_correction_and_export_badcases(Path("eval/out/badcases.jsonl"))
    print("=" * 72)
    print("PyHarness Automated Benchmark Report")
    print("-" * 72)
    print(f"Serial Tool P50 Latency   : {lat['serial_p50_ms']} ms")
    print(f"Parallel Tool P50 Latency : {lat['parallel_p50_ms']} ms (Latency Reduced by {lat['latency_reduction_pct']}%)")
    print(f"Self-Correction Recovered : {sc['self_corrected_success']} / {sc['total_scenarios']} scenarios")
    print(f"Badcases Exported (JSONL) : {sc['badcases_exported']} -> {sc['badcase_file']}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
