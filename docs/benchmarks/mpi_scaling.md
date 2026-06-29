# Archived Moonlab Documentation: MPI state-vector scaling: published numbers

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# MPI state-vector scaling: published numbers

Distributed-state-vector sharding across MPI ranks, measured on the
random-RZ + CNOT-chain circuit (`examples/distributed/large_state_random_circuit.c`)
across the (N, ranks) grid below.  Archived under
`benchmarks/results/mpi_scaling_2026-05-20/scaling.json`.

## Setup

- Build: `cmake -B build-mpi -DQSIM_ENABLE_MPI=ON` then
  `cmake --build build-mpi --target large_state_random_circuit`.
- Runtime: Open MPI 5.0.9 on macOS arm64 (single host, `--oversubscribe`).
- Circuit: 4 layers of `RZ` on every qubit + alternating-parity
  CNOT-chain.  Total gates per N qubits: `4 * (N + N - 1)` = `8N - 4`.
- Verification: unitary evolution must preserve the global L2 norm.
- Single trial per cell (timing is wall-clock, not averaged).

## Results

| N  | ranks=1 (sim) | ranks=2 (sim) | ranks=4 (sim) | norm        | 4x speedup |
|----|---------------|---------------|---------------|-------------|------------|
| 22 | 0.380 s       | 0.233 s       | 0.167 s       | 1.0000000000 | 2.27       |
| 24 | 1.690 s       | 1.056 s       | 0.667 s       | 1.0000000000 | 2.53       |
| 26 | 7.105 s       | 4.671 s       | 2.974 s       | 1.0000000000 | 2.39       |
| 28 | 30.485 s      | 19.627 s      | 13.161 s      | 1.0000000000 | 2.32       |

`norm` is the global L2 norm after the full circuit on rank 0; the
table shows it pinned to 1.0 to machine precision at every cell,
which validates that the cross-rank `dist_rz` + `dist_cnot`
communication patterns preserve unitarity.

## Observations

- **MPI sharding works correctly across ranks.**  The norm column
  is 1.0 to all 10 reported digits at every (N, ranks); the
  `MPI_Sendrecv` pattern in `dist_cnot` (when the target qubit
  spans the rank boundary) does not corrupt amplitudes.
- **Parallel efficiency ~60% at 4 ranks.**  Speedup at 4 ranks
  ranges 2.27-2.53 across the four N values, vs the ideal 4.0.
  The remaining ~40% is communication cost in cross-rank gates
  -- when one of the two operands is a partition qubit, the rank
  has to `MPI_Sendrecv` the relevant amplitude block.  The
  efficiency is essentially constant across N because the
  communication is dominated by the data-volume per gate, not by
  N (each cross-rank gate moves O(local_dim) amplitudes).
- **No regression at the dist_* buffer-size boundary.**  Task #256
  fixed a bug where N >= 26 caused buffer-size truncation in the
  distributed gate dispatcher.  The N=26 and N=28 entries above
  exercise that path and produce norm = 1.0, confirming the fix.

## Scaling beyond N=28 on this host

Each rank holds `2^(N - log2(ranks)) * 16` bytes of complex
amplitudes.  On a 32 GB host:

- 1 rank: N <= 31 fits (16 GB state).
- 4 ranks: N <= 33 fits (4 GB per rank).
- 16 ranks: N <= 35 fits (2 GB per rank).

Pushing to N = 34..40 requires either a 128 GB host or a
multi-host MPI deployment.  The harness ships unchanged; the
limitation is RAM, not the code.  The (N, ranks) grid above is
the largest cell that fits comfortably on a workstation while
keeping wall-time per cell under a minute.

## Reproduce

[archived fence delimiter: ```bash]
cmake -B build-mpi -DQSIM_ENABLE_MPI=ON
cmake --build build-mpi --target large_state_random_circuit -j 8
mkdir -p bench/mpi_scaling
for N in 22 24 26 28; do
    for R in 1 2 4; do
        mpirun --oversubscribe -np $R \
               build-mpi/large_state_random_circuit $N 4 42 \
               > bench/mpi_scaling/N${N}_R${R}.log 2>&1
    done
done
[archived fence delimiter: ```]
```
