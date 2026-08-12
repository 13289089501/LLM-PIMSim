"""
LLM-PIMSim v2 — 配置驱动入口（薄壳）
用户不需要改这个文件。只需要改 configs/ 下的 YAML，然后:
    python run.py                     # 跑默认实验(第一个)
    python run.py configs/experiments/01_gpu_only.yaml   # 指定实验
    python run.py --all               # 跑 experiments/ 下全部实验
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 统一 stdout/stderr 为 UTF-8，避免 Windows 控制台 GBK 造成中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiment_runner import run_experiment, run_workload_experiment

EXPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiments")


def main(argv):
    # --workload 开关: 用 kernel 粒度 workload 驱动（替代 model_lib 路径）
    use_workload = "--workload" in argv
    argv_clean = [a for a in argv if a != "--workload"]

    if "--all" in argv_clean:
        exp_files = sorted(glob.glob(os.path.join(EXPS_DIR, "??_*.yaml")))
        # 排除子配置文件（NN_name_hardware/interconnect/mapping/placement.yaml），
        # 只保留实验入口（与 GUI api_experiments 的过滤规则一致）
        exp_files = [f for f in exp_files
                     if not any(k in os.path.basename(f) for k in
                                ("_hardware", "_interconnect", "_mapping", "_placement"))]
        if not exp_files:
            print("没有找到实验配置 (需 NN_*.yaml 命名):", EXPS_DIR)
            return
        results = []
        for f in exp_files:
            print(f">>> 运行 {os.path.basename(f)}  [{('workload' if use_workload else 'model_lib')}]")
            if use_workload:
                r = run_workload_experiment(f, verbose=False)
            else:
                r = run_experiment(f, verbose=False)
            results.append(r["result"])
        print("\n" + "=" * 60)
        print("  对比总结")
        print("=" * 60)
        for r in results:
            print(f"  {r.metadata.get('experiment','?'):<24s} "
                  f"延迟 {r.total_latency_ns/1e6:>8.2f} ms "
                  f"瓶颈 {r.bottleneck.name}")
        return

    # 指定 or 默认第一个
    if argv_clean and os.path.exists(argv_clean[0]):
        exp_file = argv_clean[0]
    else:
        exp_file = os.path.join(EXPS_DIR, "01_gpu_only.yaml")
        if not os.path.exists(exp_file):
            files = sorted(glob.glob(os.path.join(EXPS_DIR, "*.yaml")))
            exp_file = files[0] if files else None
        if not exp_file:
            print("找不到任何实验配置，请先创建 configs/experiments/*.yaml")
            return

    print(f"运行: {exp_file}  [{('workload' if use_workload else 'model_lib')}]")
    if use_workload:
        run_workload_experiment(exp_file)
    else:
        run_experiment(exp_file)


if __name__ == "__main__":
    main(sys.argv[1:])
