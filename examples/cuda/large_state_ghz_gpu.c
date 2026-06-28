/**
 * @file  large_state_ghz_gpu.c
 * @brief v1.1 follow-up #6 demo: sharded MPI + CUDA GHZ state.
 *
 * Same workload as examples/distributed/large_state_ghz.c, but each
 * rank's local shard lives on a CUDA GPU.  Demonstrates the new
 * partition_state_create_gpu / partition_sync_* path:
 *   - dist_hadamard(0) is a "local" gate (target = local qubit on
 *     every rank) and dispatches entirely on GPU with NO MPI.
 *   - dist_cnot(0, k) for k in the partition region requires an
 *     MPI exchange; the new path syncs GPU->host before send and
 *     host->GPU after recv.
 *
 * Run on a CUDA-equipped MPI host:
 *   mpirun -n 2 large_state_ghz_gpu [N]   (default N=10, must fit
 *                                          on a single GPU per rank)
 *
 * For multi-host setups CUDA must be installed on every rank.
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
    const uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 10;

    if (rank == 0) {
        fprintf(stdout,
            "=== Sharded MPI+CUDA GHZ demo: N=%u across %d ranks ===\n", N, size);
        fprintf(stdout, "    total amplitudes: 2^%u = %llu\n", N, (1ULL << N));
        fprintf(stdout, "    per-rank GPU mem: %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) /
                ((double)size * 1024.0 * 1024.0));
    }

    partitioned_state_t *state = partition_state_create_gpu(ctx, N, NULL);
    if (!state) {
        if (rank == 0) {
            fprintf(stderr,
                "partition_state_create_gpu failed (no CUDA, no GPU, or "
                "non-CUDA libquantumsim build)\n");
        }
        mpi_bridge_free(ctx); mpi_bridge_finalize();
        return 2;
    }

    if (rank == 0) {
        fprintf(stdout,
            "    local shard: 2^%u amps on GPU (rank %d holds [%llu, %llu))\n",
            state->local_qubits, rank,
            (unsigned long long)state->local_start,
            (unsigned long long)state->local_end);
    }

    const clock_t t0 = clock();

    /* GHZ: H_0 then CNOT(0, k) for k=1..N-1.  When k >= local_qubits
     * the CNOT spans the partition boundary and triggers the MPI
     * exchange path with GPU<->host sync wrapped around it. */
    if (dist_hadamard(state, 0) != DIST_GATE_SUCCESS) goto fail;
    for (uint32_t k = 1; k < N; k++) {
        if (dist_cnot(state, 0, k) != DIST_GATE_SUCCESS) goto fail;
    }

    /* Pull final GPU shard back to host for inspection. */
    if (partition_sync_to_host(state) != PARTITION_SUCCESS) goto fail;

    const clock_t t1 = clock();
    const double sim_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    /* Same verification as the CPU-only demo. */
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
        fprintf(stdout,
            "\n    simulation time:  %.4f s (wall, rank 0)\n", sim_s);
        fprintf(stdout, "    P(|0...0>) = %.6f  (rank 0)\n", m0);
        fprintf(stdout, "    P(|1...1>) = %.6f  (rank %d, reduced)\n",
                mN1_global, owner_last);
        fprintf(stdout, "    expected   = 0.500000 each\n");

        if (fabs(m0 - 0.5) < 1e-9 && fabs(mN1_global - 0.5) < 1e-9) {
            fprintf(stdout,
                "\n    GHZ verified: sharded MPI+CUDA at N=%u, %d ranks.\n",
                N, size);
            fprintf(stdout,
                "    Local gates ran on GPU; cross-partition CNOTs used\n"
                "    the GPU->host->MPI->host->GPU sync path.\n");
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
    if (rank == 0) fprintf(stderr, "FAIL during dist_* call\n");
    partition_state_free(state);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return 3;
}
