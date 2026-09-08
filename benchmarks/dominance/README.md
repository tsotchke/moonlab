# Dominance benchmark harness

Phase 1 of the platform dominance campaign
(`docs/roadmap/platform-dominance-campaign.md`). This directory holds the
head-to-head benchmarks that DEFINE dominance: each one runs MoonLab against a
named incumbent on a front's defining metric and emits a gate-bindable result.

A benchmark is not a MoonLab-only microbenchmark. It must:

1. Run the SAME workload through MoonLab and the incumbent (Stim, quimb,
   cuQuantum, QuTiP, PennyLane, PySCF/OpenFermion, tket, ...).
2. Verify the two produce the SAME answer within the front's tolerance
   (correctness gate -- a faster wrong answer is not a win).
3. Measure the defining metric for both (throughput, wall-clock, max scale,
   accuracy) with warmup and repeats.
4. Emit one JSON record matching `harness.py::BenchmarkResult` with a verdict
   in {lead, parity, behind} and the clean source fingerprint from
   `scripts/moonlab_source_identity.py`, so the ICC gate can bind the result
   to the exact source state that produced it.

## Layout

- `harness.py` -- the shared contract: result schema, timing (warmup+repeats),
  verdict rule, fingerprint binding, JSONL emit.
- `fronts/` -- one module per front benchmark (`f3_clifford_vs_stim.py`, ...).
  Each imports the harness, defines the shared workload, runs both engines,
  and returns a `BenchmarkResult`.
- `run_dominance.py` -- the runner: discovers front benchmarks, runs the ones
  whose incumbent is installed, writes `benchmarks/dominance/results/*.jsonl`.

## Rules (inherited from the charter)

- No self-referential wins: the incumbent must actually run, same workload.
- No manufactured wins: no loosened tolerance, no stub incumbent, no
  cherry-picked size. A benchmark that can't run the incumbent SKIPS loudly.
- Every result carries the source fingerprint; a result without it cannot
  certify anything.
- verdict = lead only when MoonLab is decisively past the incumbent on the
  metric AND passed the correctness check; parity within noise; behind otherwise.

## Sampling accuracy and issue21

The batch front now accounts for both independent sample variances. Its
six-sigma cutoff is unchanged; `legacy_marg_sigma` retains the old score that
treated Stim's empirical marginal as an exact probability. Detector checks
also compare **each** engine to analytic probabilities from an exact Stim
detector-error model, without disjoint-error approximation or gauge relaxation.
Matching two biased samplers therefore does not pass. These are componentwise
checks, not a claim of a global six-sigma confidence level.

Run the reproducible accuracy checks with a pinned Stim environment and a
built library (the local investigation used Stim1.16.0 and NumPy2.4.6):

```sh
export MOONLAB_LIB_DIR="$PWD/build"
python -m unittest tests.release.test_sampling_statistics \
  benchmarks.dominance.tests.test_batch_sampling_gate \
  benchmarks.dominance.tests.test_detector_joint_accuracy
python benchmarks/dominance/check_sampling_reference.py --shots 200000
python benchmarks/dominance/check_detector_joint_accuracy.py \
  --shots 200000 --seeds 1234 7 47 12345 987654321 --threads 1 4
```

The joint audit covers neighbor and temporal-stride pairs spanning all120
detectors of the noisy d5/r8 workload. It preserves four-cell counts and uses
binomial KL-tail Chernoff bounds for rare events. Its per-cell alpha is
`erfc(6/sqrt(2))`; the reported union bound accounts for the number of cells,
engines and configurations under the IID-shot null. It is not an exact p-value
or proof about every higher-order correlation. Impossible events and injected
wrong correlations must fail.

Accuracy claims must distinguish same-model correctness from finite-sample
estimation error at a fixed compute budget and from decoder logical failure
rates. Bit-identical packing optimizations improve neither per-shot fidelity
nor intrinsic correctness. A valid claim of lower estimation error needs
matched budgets and replicated error measurements; faster sampling alone is
not a measured accuracy win.

## Latency and memory probes

`benchmark_detector_sampling.py` runs one engine per process and reports
preparation time, cold sampling, warm median/p95, output bytes and peak process
RSS. Compare both equivalent byte outputs and Stim's packed output explicitly:

```sh
python benchmarks/dominance/benchmark_detector_sampling.py \
  --library "$MOONLAB_LIB_DIR/libquantumsim.dylib" --engine moonlab --threads 1
python benchmarks/dominance/benchmark_detector_sampling.py \
  --library "$MOONLAB_LIB_DIR/libquantumsim.dylib" --engine stim-unpacked
python benchmarks/dominance/benchmark_detector_sampling.py \
  --library "$MOONLAB_LIB_DIR/libquantumsim.dylib" --engine stim-packed
```

Use `.so` on Linux. Peak RSS includes the common Python/NumPy/Stim/Moonlab
harness and is not just backend workspace. These bounded probes are diagnostic,
not a dominance certificate. Same circuit/shot count, declared thread budgets,
output layout, library hash and unloaded repeatable host conditions matter.
