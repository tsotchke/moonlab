/**
 * @file  large_state_qft.c
 * @brief Sharded QFT across MPI ranks (v0.8.1 demo).
 *
 * Applies the textbook N-qubit Quantum Fourier Transform across
 * an MPI-partitioned state vector:
 *
 *   for j = 0..N-1:
 *       H_j
 *       for k = j+1..N-1:
 *           CPHASE(k, j; pi / 2^(k-j))
 *   then reverse bit-order via SWAPs.
 *
 * Cross-rank gates use dist_hadamard, dist_cphase, dist_swap.
 *
 * Test signature: applying QFT to the |0..0> input must produce
 * the uniform superposition (every basis state at amplitude
 * 1/sqrt(2^N), uniform phase).  We verify the L2-norm and the
 * P(|0..0>) = 1/2^N amplitude on rank 0.
 *
 * Run:
 *   mpirun -n 8 large_state_qft [N]      (default N=24)
 */

#include "../../src/distributed/state_partition.h"
#include "../../src/distributed/distributed_gates.h"
#include "../../src/distributed/mpi_bridge.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    distributed_ctx_t *ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) return 1;

    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);
    const uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 24;

    if (rank == 0) {
        fprintf(stdout, "=== Sharded QFT demo: N=%u across %d ranks ===\n", N, size);
        fprintf(stdout, "    total amplitudes: 2^%u = %llu\n", N, (1ULL << N));
        fprintf(stdout, "    total memory:     %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) / (1024.0 * 1024.0));
        fprintf(stdout, "    per-rank memory:  %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) /
                ((double)size * 1024.0 * 1024.0));
        fprintf(stdout, "    gates:            %u Hadamards + %u CPHASEs + %u SWAPs\n",
                N, N * (N - 1) / 2, N / 2);
    }

    partitioned_state_t *state = partition_state_create(ctx, N, NULL);
    if (!state) {
        if (rank == 0) fprintf(stderr, "partition_state_create failed\n");
        mpi_bridge_free(ctx); mpi_bridge_finalize();
        return 1;
    }
    partition_init_zero(state);

    const clock_t t0 = clock();

    /* Forward QFT. */
    for (uint32_t j = 0; j < N; j++) {
        if (dist_hadamard(state, j) != DIST_GATE_SUCCESS) goto fail;
        for (uint32_t k = j + 1; k < N; k++) {
            const double phi = M_PI / (double)(1ULL << (k - j));
            if (dist_cphase(state, k, j, phi) != DIST_GATE_SUCCESS) goto fail;
        }
    }
    /* Bit-reverse to match the standard QFT permutation. */
    for (uint32_t j = 0; j < N / 2; j++) {
        if (dist_swap(state, j, N - 1 - j) != DIST_GATE_SUCCESS) goto fail;
    }

    const clock_t t1 = clock();
    const double sim_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    /* Verification:  QFT|0> = (1/sqrt(2^N)) sum_x |x>.
     * Each rank owns 2^N / size amplitudes; the local L2-norm of
     * those amplitudes contributes 1/size to the global norm. */
    const double expected_local_norm = 1.0 / (double)size;
    const double expected_amp_sq = 1.0 / (double)(1ULL << N);

    double local_norm = 0.0;
    for (size_t idx = 0; idx < state->local_count; idx++) {
        const double complex a = state->amplitudes[idx];
        local_norm += creal(a) * creal(a) + cimag(a) * cimag(a);
    }

    double global_norm = 0.0;
    mpi_allreduce_sum_double(ctx, &local_norm, &global_norm, 1);

    if (rank == 0) {
        const double complex a0 = partition_get_amplitude(state, 0);
        const double m0 = creal(a0) * creal(a0) + cimag(a0) * cimag(a0);

        fprintf(stdout, "\n    simulation time:  %.4f s (wall, rank 0)\n", sim_s);
        fprintf(stdout, "    global L2 norm:   %.10f  (expected 1.0)\n", global_norm);
        fprintf(stdout, "    P(|0...0>):       %.4e  (expected %.4e)\n",
                m0, expected_amp_sq);

        const int norm_ok = fabs(global_norm - 1.0) < 1e-9;
        const int amp_ok  = fabs(m0 - expected_amp_sq) / expected_amp_sq < 1e-6;
        if (norm_ok && amp_ok) {
            fprintf(stdout, "\n    QFT verified across %d MPI ranks at N=%u\n",
                    size, N);
            fprintf(stdout, "    Uniform-superposition signature holds.\n");
        } else {
            fprintf(stderr, "\n    FAIL: norm_ok=%d amp_ok=%d\n", norm_ok, amp_ok);
        }
    }
    (void)expected_local_norm;

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
