# Detector estimation accuracy at fixed checkpoints

Exploratory observation on one co-resident macOS ARM64 host, source
`625a65a2166585f10dd6d022f63eec5b6706e61d`, Stim 1.16.0 and NumPy 2.4.6.
The exact d5/r8 detector-error model has 120 detector marginals and
`p1=p2=pm=0.001`. Every engine uses one thread and 100,000-shot batches.
Five distinct seeds (1234, 2345, 3456, 4567, 5678) were used at each horizon,
with fresh processes and rotating engine order. The two horizons reuse seed
streams; they are not ten independent replications.

| Checkpoint | Engine | Median completed shots | Median marginal RMSE |
|---|---|---:|---:|
| 250 ms | Moonlab | 1,900,000 | 8.5862e-5 |
| 250 ms | Stim unpacked | 600,000 | 1.4833e-4 |
| 250 ms | Stim packed | 800,000 | 1.2671e-4 |
| 1 s | Moonlab | 7,900,000 | 4.1173e-5 |
| 1 s | Stim unpacked | 2,500,000 | 7.3230e-5 |
| 1 s | Stim packed | 3,400,000 | 6.3486e-5 |

Moonlab had lower observed error than both Stim variants for each of the five
seeds at each horizon. Relative to packed Stim, the median RMSE was about 32%
lower at 250 ms and 35% lower at one second. This is an estimation-precision
observation at a deadline; it does not establish better intrinsic simulation
accuracy, a universal advantage, or release readiness.

Only batches whose sampling, output normalization, validation, summation and
count aggregation all finished by the deadline contribute. Stim's required
unpacking is included. Preparation is outside the budget and reported. A
native call already in flight cannot be interrupted: the final late batch is
discarded, and actual time/overshoot remain in the data. Maximum observed
overrun was 24.9 ms. Thus this is a completed-checkpoint comparison, not a hard
real-time return guarantee or proof of a strict process-lifetime budget.

The fixed-N IID expected MSE is descriptive under deadline stopping, not a
conditional guarantee. The finite seed cohort, common workload and co-resident
host limit generalization. Correctness is checked separately against analytic
marginals and joint distributions; impossible-event and wrong-correlation
negative controls remain required.

All raw counts, reference probabilities, preparation/actual/checkpoint times,
late batches and identities are retained in
[the JSON observation](estimation_accuracy_20260908.json). Repeated common
metadata and reference vectors are stored once; local filesystem paths are
omitted. Reproduce with `benchmark_estimation_accuracy.py` at both `--budget-s
0.25` and `--budget-s 1`, using the recorded seed set and library source.
