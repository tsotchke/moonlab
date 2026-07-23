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
