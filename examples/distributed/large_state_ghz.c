/**
 * @file  large_state_ghz.c
 * @brief Sharded GHZ state across MPI ranks (v0.7.9 demo).
 *
 * Demonstrates the v0.7.6 partitioned-state path scaling.  Builds
 * an N-qubit GHZ state with `dist_hadamard(0)` + `dist_cnot(0,k)`
 * for k = 1..N-1 across the MPI communicator, then verifies the
 * (|0...0> + |1...1>) / sqrt(2) signature on rank 0.
 *
 * At N=24 the state fits on a single rank (256 MB total, 64 MB per
 * rank with 4 ranks) but exercises the same `partition_plan_*`
 * MPI cross-rank gate path that scales to >32 qubits on adequate
 * cluster RAM.
 *
 * Run:
 *   mpirun -n 4 large_state_ghz [N]      (default N=24)
 */

#include "../../src/distributed/state_partition.h"
#include "../../src/distributed/distributed_gates.h"
#include "../../src/distributed/mpi_bridge.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    distributed_ctx_t *ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) return 1;

    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);
    const uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 24;

    if (rank == 0) {
        fprintf(stdout, "=== Sharded GHZ demo: N=%u across %d ranks ===\n", N, size);
        fprintf(stdout, "    total amplitudes: 2^%u = %llu\n", N, (1ULL << N));
        fprintf(stdout, "    total memory:     %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) / (1024.0 * 1024.0));
        fprintf(stdout, "    per-rank memory:  %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) /
                ((double)size * 1024.0 * 1024.0));
    }

    partitioned_state_t *state = partition_state_create(ctx, N, NULL);
    if (!state) {
        if (rank == 0) fprintf(stderr, "partition_state_create failed\n");
        mpi_bridge_free(ctx); mpi_bridge_finalize();
        return 1;
    }
    partition_init_zero(state);

    const clock_t t0 = clock();

    /* GHZ: H_0 then CNOT(0, k) for k = 1..N-1. */
    if (dist_hadamard(state, 0) != DIST_GATE_SUCCESS) goto fail;
    for (uint32_t k = 1; k < N; k++) {
        if (dist_cnot(state, 0, k) != DIST_GATE_SUCCESS) goto fail;
    }

    const clock_t t1 = clock();
    const double sim_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    /* Verify per-rank: each rank owns one half of the state.  GHZ
     * has support only at indices 0 (rank 0 owns this) and 2^N-1
     * (rank N-1 owns this on standard layouts).  Each rank prints
     * the squared magnitude of *its* local extremal amplitude.
     * Rank 0: |amp(0)|^2 = 0.5.  Last rank: |amp(2^N-1)|^2 = 0.5. */
    const uint64_t last_idx = (1ULL << N) - 1;
    const int owner_last = partition_get_owner(state, last_idx);
    double mN1_local = 0.0;
    if (rank == owner_last) {
        const double complex aN1 = partition_get_amplitude(state, last_idx);
        mN1_local = creal(aN1) * creal(aN1) + cimag(aN1) * cimag(aN1);
    }
    double mN1_global = 0.0;
    mpi_allreduce_sum_double(ctx, &mN1_local, &mN1_global, 1);

    if (rank == 0) {
        const double complex a0 = partition_get_amplitude(state, 0);
        const double m0 = creal(a0) * creal(a0) + cimag(a0) * cimag(a0);
        fprintf(stdout, "\n    simulation time:  %.4f s (wall, rank 0)\n", sim_s);
        fprintf(stdout, "    P(|0...0>)   = %.6f  (owned by rank 0)\n", m0);
        fprintf(stdout, "    P(|1...1>)   = %.6f  (owned by rank %d, reduced)\n",
                mN1_global, owner_last);
        fprintf(stdout, "    expected     = 0.500000 each\n");

        if (fabs(m0 - 0.5) < 1e-9 && fabs(mN1_global - 0.5) < 1e-9) {
            fprintf(stdout, "\n    GHZ verified across %d MPI ranks at N=%u\n",
                    size, N);
            fprintf(stdout, "    This is the path to >32 qubits: same code,\n");
            fprintf(stdout, "    more ranks, more RAM.  At N=40, 16 TB total\n");
            fprintf(stdout, "    across 256 ranks = 64 GB/rank (cluster-class).\n");
        } else {
            fprintf(stderr, "\n    FAIL: GHZ amplitudes off\n");
        }
    }

    partition_state_free(state);
    mpi_barrier(ctx);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return 0;

fail:
    if (rank == 0) fprintf(stderr, "dist_* gate failed\n");
    partition_state_free(state);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return 1;
}
