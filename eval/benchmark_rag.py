"""
Automated Hybrid RAG Offline Evaluation Suite (with realistic Hard-Negative Distractors).

Compares 4 retrieval modes on an enterprise/industrial QA benchmark containing confusingly
similar equipment specs, model codes, and overlapping domain terminology:
  1. dense          (Dense vector similarity only)
  2. bm25           (Sparse lexical BM25 only)
  3. hybrid         (BM25 + Dense via RRF)
  4. hybrid_rerank  (BM25 + Dense via RRF + Cross-Encoder style Rerank)

Reports Top-1 Recall, Top-3 Recall, MRR@3, and P50 retrieval latency (ms).
"""
from __future__ import annotations

import asyncio
import statistics
import time

from pyharness.rag import HybridRAGPipeline

CORPUS_DOCS: list[tuple[str, str]] = [
    (
        "power_grid_manual.md",
        """
# 配电网一次设备运维与检修规程

## 1. LW30-252 型 SF6 断路器气体压力监控
LW30-252 高压断路器额定气压为 0.60 MPa。运行中气压降至 0.52 MPa 触发一级低压告警；降至 0.48 MPa 时闭锁分合闸操作回路并上报调度中心。

## 2. LW36-126 型 SF6 断路器气体压力监控（干扰项）
LW36-126 型号断路器额定气压为 0.50 MPa，低压补气告警为 0.45 MPa，闭锁压力为 0.40 MPa，主要用于 110kV 变电站间隔。

## 3. ZN63A-12 真空断路器机械特性巡检（干扰项）
真空断路器无 SF6 气体介质，重点检查触头开距 9-11 mm、分闸速度 0.9-1.2 m/s 以及辅助触点氧化状态。

## 4. 10kV 油浸式变压器绕组温升标准
油浸式配电变压器顶层油温正常运行不应超过 85℃，温升限值为 55K；连续超温 30 分钟须减载 20% 并排查冷却风机故障。

## 5. 35kV 干式变压器绕组温升标准（干扰项）
SCB14 干式变压器采用环氧树脂绝缘，绕组超温报警设定 130℃，超温跳闸设定 150℃，日常巡检重点检查温控器风机启停。

## 6. 氧化锌避雷器泄露电流巡检
氧化锌避雷器阻性泄露电流大于 0.5 mA 或全电流同比增加 30% 以上时，判定为受潮隐患，须在 48 小时内停电试验。
""",
    ),
    (
        "petrochem_safety.md",
        """
# 金山石化装置安全隐患排查指南

## 1. 金山石化 A 区高压反应釜压力阀温升处置
金山石化 A 区压力阀红外测温超过 85℃ 且持续 10 分钟，须在 15 分钟内启动二级降温预案，并检查热电偶零漂与泄压阀开度。

## 2. 金山石化 B 区低温储槽压力阀巡检（干扰项）
金山石化 B 区低温压力阀正常运行温度为 -20℃ 至 15℃，巡检重点关注阀体结霜与伴热带供电是否正常。

## 3. 催化燃烧式可燃气体探头校准周期
催化燃烧式可燃气体探头正常本底读数 0-15 ppm，一级声光告警阈值 25 ppm，联锁紧急切断阀阈值 50 ppm；每 90 天强制标准气校准一次。

## 4. 电化学有毒气体探头巡检（干扰项）
硫化氢电化学探头一级告警为 10 ppm，二级紧急撤离阈值为 20 ppm，探头传感器寿命一般为 24 个月。

## 5. 内浮顶储罐氮封系统微正压控制
内浮顶储罐氮封阀设定压力为 0.5 kPa，呼吸阀起跳压力 1.5 kPa；若氮封压力跌破 0.2 kPa，立即核查氮气管网减压阀堵塞情况。
""",
    ),
    (
        "agent_ops_handbook.md",
        """
# 企业 AI Agent 平台上线与治理规范

## 1. Nginx 代理 SSE 流式输出配置规范
Nginx 转发 Streamable HTTP 或 SSE 接口时，必须显式配置 proxy_buffering off 与响应头 X-Accel-Buffering: no，避免首字延迟飙升与前端卡顿。

## 2. Nginx 普通静态文件缓存规范（干扰项）
前端构建产物 dist 目录中的带哈希 JS/CSS 静态资源可开启长期强缓存 Cache-Control: public, max-age=31536000。

## 3. 高危工具 Human-in-the-Loop 审批标准
标记为 RiskLevel.HIGH 的工具（写库 SQL、外发应急告警）必须经 ApprovalBroker 挂起并由管理员确认，超时 120 秒默认 Fail-Closed 拒绝。
""",
    ),
]

EVAL_SET: list[tuple[str, str]] = [
    ("LW30-252 断路器气压低于多少会触发闭锁分合闸？", "0.48 MPa"),
    ("LW36-126 变电站断路器闭锁压力是多少？", "0.40 MPa"),
    ("SCB14 干式变压器超温跳闸设定是多少度？", "150℃"),
    ("10kV 油浸式配电变压器顶层油温上限和减载比例是多少？", "减载 20%"),
    ("氧化锌避雷器阻性泄露电流多少判定受潮隐患？", "0.5 mA"),
    ("金山石化 A 区压力阀超温启动什么预案？", "二级降温预案"),
    ("可燃气体探头触发联锁紧急切断阀是多少 ppm？", "50 ppm"),
    ("硫化氢电化学探头二级紧急撤离阈值是多少？", "20 ppm"),
    ("内浮顶储罐氮封压力跌破多少 kPa 需核查减压阀？", "0.2 kPa"),
    ("Nginx 转发 SSE 流式接口为什么卡顿，如何关闭缓冲？", "proxy_buffering off"),
    ("高危工具审批超时默认放行还是拒绝？", "Fail-Closed"),
]


async def run_benchmark() -> dict[str, dict[str, float]]:
    rag = HybridRAGPipeline()
    for src, text in CORPUS_DOCS:
        await rag.ingest_markdown(text, source=src)

    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    summary: dict[str, dict[str, float]] = {}

    print("=" * 78)
    print(f"{'Mode':<16} | {'Top-1 Recall':>12} | {'Top-3 Recall':>12} | {'MRR@3':>8} | {'P50 (ms)':>9}")
    print("-" * 78)

    for mode in modes:
        top1_hits = 0
        top3_hits = 0
        rr_sum = 0.0
        latencies: list[float] = []

        for query, expected_evidence in EVAL_SET:
            t0 = time.perf_counter()
            hits = await rag.search(query, top_k=3, mode=mode)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            rank_found = None
            for idx, h in enumerate(hits, start=1):
                if expected_evidence in h.parent_context:
                    rank_found = idx
                    break
            if rank_found == 1:
                top1_hits += 1
            if rank_found is not None and rank_found <= 3:
                top3_hits += 1
                rr_sum += 1.0 / rank_found

        n = len(EVAL_SET)
        metrics = {
            "top1_recall": round(top1_hits / n * 100, 1),
            "top3_recall": round(top3_hits / n * 100, 1),
            "mrr_at_3": round(rr_sum / n, 4),
            "p50_ms": round(statistics.median(latencies), 2),
        }
        summary[mode] = metrics
        print(
            f"{mode:<16} | {metrics['top1_recall']:>11.1f}% | "
            f"{metrics['top3_recall']:>11.1f}% | {metrics['mrr_at_3']:>8.4f} | {metrics['p50_ms']:>9.2f}"
        )

    print("=" * 78)
    return summary


if __name__ == "__main__":
    asyncio.run(run_benchmark())
