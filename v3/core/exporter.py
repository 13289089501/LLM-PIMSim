"""
LLM-PIMSim v3 core.exporter — 【输出系统】

职责：把仿真的 AnalysisResult（SimulationResult）转成可读 / 可落盘的形态。
  1. result_to_dict —— SimulationResult → 纯 dict（JSON 友好）
  2. save_json      —— dict 落盘为 JSON（异常容错、目录自动创建）
  3. report         —— 控制台摘要报告（含完成度 / 搬运量诊断）

依赖：core.common（SimulationResult 等数据结构）。独立于精度/算子/调度。
由顶层 `output.py` 转发（若需要）保持接口统一。
"""

import json
import os


def result_to_dict(result) -> dict:
    """把 SimulationResult 转成 JSON 友好的 dict。"""
    bd = result.breakdown
    return {
        "metadata": result.metadata,
        "total_latency_ns": result.total_latency_ns,
        "total_latency_ms": round(result.total_latency_ns / 1_000_000, 3),
        "breakdown": {
            "compute_ns": bd.compute_ns,
            "transfer_ns": bd.transfer_ns,
            "sync_ns": bd.sync_ns,
            "local_read_ns": bd.local_read_ns,
            "local_write_ns": bd.local_write_ns,
            "local_rw_ns": bd.local_read_ns + bd.local_write_ns,
        },
        "bottleneck": result.bottleneck.name,
        "bottleneck_rationale": result.bottleneck_rationale,
        "operator_timings": [
            {"op_id": t.op_id, "op_type": t.op_type,
             "hardware": t.hardware, "start_ns": t.start_ns,
             "end_ns": t.end_ns, "duration_ns": t.duration_ns,
             "compute_ns": t.compute_ns,
             "local_read_ns": t.local_read_ns, "local_write_ns": t.local_write_ns,
             "transfer_ns": t.transfer_ns, "sync_ns": t.sync_ns}
            for t in result.operator_timings
        ],
        "event_trace": [
            {"id": e.id, "type": e.event_type.name,
             "start_ns": e.start_time_ns, "end_ns": e.end_time_ns,
             "operator": e.operator_id, "resource": e.resource_id,
             "component": e.component}
            for e in result.event_trace
        ],
        "movement_bytes": result.movement_bytes,
        "data_source_notes": result.data_source_notes,
        "diagnostics": result.diagnostics,
        "analysis": {"critical_path": build_critical_path(result)},
    }


def save_json(result, output_dir: str, name: str, module_dir: str = None) -> str:
    """把结果保存为 <output_dir>/<name>.json。

    module_dir: 可选的锚定目录（默认取调用方项目根）。若 output_dir 是相对路径，
                则相对 module_dir 解析；否则用绝对路径。
    返回保存后的完整路径。
    """
    base = module_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = output_dir if os.path.isabs(output_dir) else os.path.join(base, output_dir)
    os.makedirs(out_path, exist_ok=True)
    path = os.path.join(out_path, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_dict(result), f, indent=2, ensure_ascii=False)
    return path


def report(result) -> str:
    """生成控制台/日志用的多行摘要文本（含完成度与搬运诊断）。"""
    bd = result.breakdown
    lines = []
    lines.append("=" * 60)
    lines.append(f"  实验: {result.metadata.get('experiment','?')}   模型: {result.metadata.get('model','?')}")
    lines.append("=" * 60)
    lines.append(f"  总延迟:     {result.total_latency_ns/1e6:>10.2f} ms")
    lines.append(f"  计算:       {bd.compute_ns/1e6:>10.2f} ms")
    lines.append(f"  搬运:       {bd.transfer_ns/1e6:>10.2f} ms")
    lines.append(f"  同步等待:   {bd.sync_ns/1e6:>10.2f} ms")
    lines.append(f"  本地读写:   {(bd.local_read_ns+bd.local_write_ns)/1e6:>10.2f} ms")
    lines.append(f"  瓶颈类型:   {result.bottleneck.name}")
    lines.append(f"  理由:       {result.bottleneck_rationale}")
    mb = (result.movement_bytes or {}).get("total_bytes", 0)
    lines.append(f"  搬运字节:   {mb/1e6:>10.2f} MB")
    if result.data_source_notes:
        lines.append(f"  数据源决策记录: 共 {len(result.data_source_notes)} 条"
                     f"（已存 JSON，此处仅显示前 3 条）")
        pinned = sum(1 for n in result.data_source_notes if "[参考]" not in n and "指定" in n)
        auto = sum(1 for n in result.data_source_notes if "[参考]" in n)
        warn = len(result.data_source_notes) - pinned - auto
        lines.append(f"    用户固定源: {pinned} | 就近参考: {auto} | 告警/异常: {warn}")
        for n in result.data_source_notes[:3]:
            lines.append(f"    {n}")
    return "\n".join(lines)


# =================================================================
# 关键路径归因（初学者友好）
# =================================================================
def build_critical_path(result) -> dict:
    """由调度结果估算"决定总延时的关键路径"。

    方法（wall-clock 主链回溯）：
      - 从"结束最晚"的算子 M 出发；
      - 每次向前找"最晚前驱"：在 start 早于当前节点的算子中，end 最大的那个（因为它完工最晚，
        决定了下一个能开始的最晚前置时刻；这是无依赖信息时可解释的近似）；
      - 一路回溯到 start≈0 的最早算子，得出一条时间主链。
    输出 {ops:[...], explanation, bottleneck_proxy} 便于前端展示与文字说明。

    说明：真正的关键路径需要算子间的数据依赖；这里用"时间上相互拖延"近似，
    能在不引入依赖图的情况下给出可解释的主耗时链。严格版待调度器记录"牵制源"后增强。
    """
    tims = [t for t in (result.operator_timings or []) if t.duration_ns > 0]
    if not tims:
        return {"ops": [], "explanation": "无已执行算子，无法计算关键路径。", "bottleneck_proxy": None}
    total = result.total_latency_ns or max((t.end_ns for t in tims), default=0)

    # 按 start 递增排序后建"前驱=end 最大"的查找
    cur = max(tims, key=lambda t: t.end_ns)
    chain = []
    seen = set()
    # 限制最大回溯长度，避免环/过长
    max_steps = max(20, len(tims))
    while cur is not None and cur.op_id not in seen and len(chain) < max_steps:
        seen.add(cur.op_id)
        chain.append(cur)
        # 找 start 早于 cur.start 的算子中 end 最大者
        candidates = [t for t in tims if t.start_ns < cur.start_ns and t.op_id not in seen]
        if not candidates:
            break
        cur = max(candidates, key=lambda t: t.end_ns)
        if cur.end_ns <= 0:
            break
    chain.reverse()   # 时间正序

    def node(t):
        d = t.duration_ns or 1
        return {
            "op_id": t.op_id, "op_type": t.op_type, "hardware": t.hardware,
            "start_ns": t.start_ns, "end_ns": t.end_ns,
            "duration_pct": round(t.duration_ns / total * 100
                                  if t.duration_ns and t.duration_ns != 0 else 0, 1),
            "compute_ms": round(t.compute_ns / 1e6, 3),
            "local_rw_ms": round((t.local_read_ns + t.local_write_ns) / 1e6, 3),
            "transfer_ms": round(t.transfer_ns / 1e6, 3),
            "sync_ms": round(t.sync_ns / 1e6, 3),
        }

    big = max(chain, key=lambda t: t.duration_ns) if chain else None
    explanation = (
        f"总延迟 {total/1e6:.2f} ms。红色主链：{len(chain)} 个在时间上前后相拖、最终撑到最晚结束"
        f"时刻的算子（近似）。若再往下追为什么慢，耗时最大的节点是 {big.op_id if big else '?'} "
        f"({(big.duration_ns/total*100 if big and total else 0):.1f}%)——它通常是瓶颈所在。"
    )
    return {"ops": [node(t) for t in chain], "explanation": explanation,
            "bottleneck_proxy": result.bottleneck.name}
