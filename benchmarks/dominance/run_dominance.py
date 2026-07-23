"""Platform-dominance campaign runner.

Executes each dominance front and prints a compact verdict table plus the
mandatory cross-engine ``correctness_ok`` gate. A front that reports
correctness_ok=False fails the campaign regardless of throughput.

Usage:
    MOONLAB_LIB_DIR=$(pwd)/build-f3 PYTHONPATH=$(pwd)/bindings/python \
        python3 benchmarks/dominance/run_dominance.py
"""
import os
import sys

# Allow `from fronts import ...` when run from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fronts import f3_batch_sampling_vs_stim, f3_clifford_vs_stim


def _print_front(name, result):
    print(f"\n=== front {name} ===")
    print(f"correctness_ok = {result['correctness_ok']}")
    per = result.get("correctness_per_size", {})
    if per:
        print("  agreement: " + ", ".join(f"n{n}={v}" for n, v in per.items()))
    print(f"{'n':>6} {'moonlab M/s':>12} {'stim M/s':>10} {'ratio':>7} {'verdict':>8}")
    for r in result["throughput"]:
        print(f"{r['n']:>6} {r['moonlab_gps']/1e6:>12.3f} "
              f"{r['stim_gps']/1e6:>10.3f} {r['ratio']:>7.2f} {r['verdict']:>8}")


def _print_batch_front(name, result):
    print(f"\n=== front {name} ===")
    print(f"correctness_ok = {result['correctness_ok']}  "
          f"(SIMD {result['simd_backend']} x{result['simd_lanes']}, "
          f"{result['phys_cores']} physical cores)")
    for k, v in result.get("correctness_detail", {}).items():
        print(f"  gate {k:26} marginals={v['marg_sigma']}s parity={v['parity_sigma']}s "
              f"corr={v['corr_sigma']}s")
    print(f"{'workload':22} {'shots':>8} {'ML 1T M/s':>10} {'ML MT M/s':>10} "
          f"{'stim M/s':>9} {'1T':>6} {'MT':>8} {'verdict':>8}")
    for r in result["rows"]:
        print(f"{r['workload']:22} {r['shots']:>8} {r['ml_st_sps']/1e6:>10.0f} "
              f"{r['ml_mt_sps']/1e6:>10.0f} {r['stim_best_sps']/1e6:>9.0f} "
              f"{r['st_ratio']:>5.2f}x {r['mt_ratio']:>7.2f}x {r['verdict']:>8}")
    print("  stim is single-threaded per sample() call; MT ratios use all cores.")
    print("  ratios are against stim's best of {unpacked, bit_packed}.")


def main():
    ok = True
    res = f3_clifford_vs_stim.run()
    _print_front("F3 clifford_vs_stim", res)
    ok = ok and res["correctness_ok"]

    bres = f3_batch_sampling_vs_stim.run()
    _print_batch_front("F3 batch_sampling_vs_stim", bres)
    ok = ok and bres["correctness_ok"]

    print(f"\ndominance campaign correctness_ok = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
