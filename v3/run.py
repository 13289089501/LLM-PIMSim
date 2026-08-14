"""
LLM-PIMSim v3 — 配置驱动入口（薄壳）
用户不需要改这个文件。只需要改 configs/ 下的 YAML，然后:
    python run.py                     # 跑默认实验(第一个)
    python run.py configs/experiments/01_gpu_only.yaml   # 指定实验
    python run.py --all               # 跑 experiments/ 下全部实验

v3 说明：`--workload` 开关保留以兼容旧命令，但 v3 已统一以 workload/kernel
路径为唯一标准，不传该开关同样走 workload 路径。
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 统一 stdout/stderr 为 UTF-8，避免 Windows 控制台 GBK 造成中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiment_runner import run_workload_experiment
import json
from pathlib import Path

EXPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiments")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _fmt_row(name, r):
    bd = r.breakdown
    diag = r.diagnostics or {}
    ok = diag.get("finished_operators", 0) == diag.get("total_operators", 0) \
        if diag.get("total_operators") else True
    print(f"  {name:<26s} 延迟 {r.total_latency_ns/1e6:>9.2f} ms "
          f"计算 {bd.compute_ns/1e6:>9.2f} | 搬运 {bd.transfer_ns/1e6:>7.2f} "
          f"| 本地读写 {(bd.local_read_ns+bd.local_write_ns)/1e6:>7.2f} "
          f"| 完成 {diag.get('finished_operators','?')}/{diag.get('total_operators','?')} "
          f"| {'✔' if ok else '✘'}")


def main(argv):
    # v3：统一走 workload/kernel 路径（新标准）。--workload 开关保留兼容，不再区分。
    argv_clean = [a for a in argv if a != "--workload"]

    # --compare name1 name2 ... ：读取保存好的结果 JSON 做对比
    if "--compare" in argv:
        idx = argv.index("--compare")
        names = [a for a in argv[idx+1:] if not a.startswith("--")]
        if names:
            print("\n" + "=" * 70)
            print("  结果对比（来自 results/*.json）")
            print("=" * 70)
            for nm in names:
                p = Path(RESULTS_DIR) / (nm if nm.endswith(".json") else nm + ".json")
                if not p.exists():
                    print(f"  ⚠ 找不到结果文件: {p.name}")
                    continue
                data = json.loads(p.read_text(encoding="utf-8"))
                bd = data.get("breakdown", {})
                diag = data.get("diagnostics", {})
                ok = (diag.get("finished_operators") == diag.get("total_operators")
                      if diag.get("total_operators") else True)
                print(f"  {p.name:<26s} 延迟 {data.get('total_latency_ms',0):>9.2f} ms "
                      f"计算 {bd.get('compute_ns',0)/1e6:>9.2f} | 搬运 {bd.get('transfer_ns',0)/1e6:>7.2f} "
                      f"| 本地读写 {bd.get('local_rw_ns',0)/1e6:>7.2f} "
                      f"| 完成 {diag.get('finished_operators','?')}/{diag.get('total_operators','?')} "
                      f"| {'✔' if ok else '✘'}")
            return

    if "--all" in argv_clean:
        exp_files = sorted(glob.glob(os.path.join(EXPS_DIR, "??_*.yaml")))
        # 排除子配置文件（NN_name_hardware/interconnect/mapping/placement.yaml）
        exp_files = [f for f in exp_files
                     if not any(k in os.path.basename(f) for k in
                                ("_hardware", "_interconnect", "_mapping", "_placement"))]
        if not exp_files:
            print("没有找到实验配置 (需 NN_*.yaml 命名):", EXPS_DIR)
            return
        results = []
        for f in exp_files:
            print(f">>> 运行 {os.path.basename(f)}  [workload]")
            r = run_workload_experiment(f, verbose=False)
            results.append((r["result"].metadata.get("experiment", os.path.basename(f)), r["result"]))
        print("\n" + "=" * 70)
        print("  对比总结")
        print("=" * 70)
        for name, r in results:
            _fmt_row(name, r)
        return

    # 指定 or 默认第一个
    if argv_clean and os.path.exists(argv_clean[0]):
        exp_file = argv_clean[0]
    else:
        exp_file = os.path.join(EXPS_DIR, "04_ic_reference.yaml")
        if not os.path.exists(exp_file):
            files = sorted(glob.glob(os.path.join(EXPS_DIR, "*.yaml")))
            exp_file = files[0] if files else None
        if not exp_file:
            print("找不到任何实验配置，请先创建 configs/experiments/*.yaml")
            return

    print(f"运行: {exp_file}  [workload]")
    run_workload_experiment(exp_file)


if __name__ == "__main__":
    main(sys.argv[1:])
