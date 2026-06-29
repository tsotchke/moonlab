# Archived Moonlab Documentation: MPI scaling harness

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# MPI scaling harness

Measures moonlab's distributed-state-vector wall-clock at a grid of
`(N qubits, MPI ranks)` points.  Drives
`examples/distributed/large_state_random_circuit` -- a sharded
RZ + CNOT-chain circuit that exercises both
`dist_rz` and `dist_cnot` across ranks.

The harness produces a JSON report with one row per `(N, ranks)`
combination; you compute speedup curves yourself from that.

## Build

[archived fence delimiter: ```]
cmake -B build -DCMAKE_BUILD_TYPE=Release -DQSIM_ENABLE_MPI=ON
cmake --build build --target large_state_random_circuit -j
[archived fence delimiter: ```]

## Run

[archived fence delimiter: ```]
./benchmarks/mpi_scaling/run_scaling.sh \
  --binary  build/examples/distributed/large_state_random_circuit \
  --depth   8 \
  --seed    42 \
  --output  bench/mpi_scaling.json
[archived fence delimiter: ```]

Default grid: N in {32, 34, 36}; ranks in {1, 4, 16}.  Override with
`--qubits 30,32,34` / `--ranks 1,2,4,8`.

## Output

[archived fence delimiter: ```json]
{
  "harness": "moonlab/mpi-scaling/v1.0",
  "host": "node-01",
  "date": "2026-05-20T...",
  "depth": 8,
  "seed": 42,
  "binary": "...",
  "points": [
    {"N": 32, "ranks":  1, "wall_s":  ..., "sim_s": ..., "norm": 1.0},
    {"N": 32, "ranks":  4, "wall_s":  ..., "sim_s": ..., "norm": 1.0},
    {"N": 32, "ranks": 16, "wall_s":  ..., "sim_s": ..., "norm": 1.0},
    ...
  ]
}
[archived fence delimiter: ```]

`sim_s` is the rank-0 wall-clock of just the simulation loop (no
MPI init / teardown), as reported by the example binary.  `wall_s`
is the end-to-end `mpirun` invocation.

## Memory ceiling

N=36 needs `2^36 * 16 bytes = 1.1 TiB` total amplitudes.  Split
across 16 ranks that's 68 GiB per node -- plan accordingly.

| N  | Total amps | Total memory |
|----|-----------:|-------------:|
| 30 | 2^30       | 17 GiB       |
| 32 | 2^32       | 68 GiB       |
| 34 | 2^34       | 275 GiB      |
| 36 | 2^36       | 1.1 TiB      |

For a single-node sanity run, restrict to N ≤ 32 ranks * total_RAM/16.

## Verification

Every point reports the global L2 norm post-simulation.  For unitary
circuits this must be 1.0 to ~1e-10; deviations indicate a numerical
problem or a bug in the dist_* gate code.
```
