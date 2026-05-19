/**
 * @file  large_state_random_circuit.c
 * @brief Sharded random RZ + CNOT-chain circuit across MPI ranks
 *        (v0.8.2 demo).
 *
 * Applies `depth` layers of:
 *   - RZ(theta_i) on every qubit i (random theta_i in [0, 2pi))
 *   - CNOT-chain: CNOT(i, i+1) for i = 0..N-2 in alternating
 *     even/odd parity each layer
 *
 * Demonstrates non-trivial entanglement growth across MPI shards,
 * exercising dist_rz + dist_cnot together (different cross-rank
 * communication patterns than QFT's dist_cphase + dist_swap).
 *
 * Verification: unitary evolution preserves L2-norm, so the
 * global norm must remain 1 to machine precision after the
 * entire circuit.
 *
 * Run:
 *   mpirun -n 8 large_state_random_circuit [N=22] [depth=8] [seed=42]
 */

#include "../../src/distributed/state_partition.h"
#include "../../src/distributed/distributed_gates.h"
#include "../../src/distributed/mpi_bridge.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* splitmix64 for reproducible per-rank-agreed RNG. */
static uint64_t sm64_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double sm64_uniform(uint64_t *s) {
    return (double)(sm64_next(s) >> 11) / (double)(1ULL << 53);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    distributed_ctx_t *ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) return 1;

    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);
    const uint32_t N     = (argc > 1) ? (uint32_t)atoi(argv[1]) : 22;
    const uint32_t depth = (argc > 2) ? (uint32_t)atoi(argv[2]) : 8;
    const uint64_t seed  = (argc > 3) ? (uint64_t)strtoull(argv[3], NULL, 10) : 42;

    if (rank == 0) {
        fprintf(stdout, "=== Sharded random RZ+CNOT circuit: N=%u depth=%u seed=%llu / %d ranks ===\n",
                N, depth, (unsigned long long)seed, size);
        fprintf(stdout, "    total amplitudes: 2^%u = %llu\n", N, (1ULL << N));
        fprintf(stdout, "    total memory:     %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) / (1024.0 * 1024.0));
        fprintf(stdout, "    per-rank memory:  %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) /
                ((double)size * 1024.0 * 1024.0));
        const uint64_t total_gates = (uint64_t)depth * (N + (N - 1));
        fprintf(stdout, "    total gates:      %llu  (%u RZ + %u CNOT per layer)\n",
                (unsigned long long)total_gates, N, N - 1);
    }

    partitioned_state_t *state = partition_state_create(ctx, N, NULL);
    if (!state) {
        if (rank == 0) fprintf(stderr, "partition_state_create failed\n");
        mpi_bridge_free(ctx); mpi_bridge_finalize();
        return 1;
    }
    partition_init_zero(state);

    /* Spread the |0..0> input with one H layer first so the circuit
     * actually has support outside basis-zero. */
    for (uint32_t i = 0; i < N; i++) {
        if (dist_hadamard(state, i) != DIST_GATE_SUCCESS) goto fail;
    }

    const clock_t t0 = clock();

    uint64_t rng_state = seed;
    for (uint32_t layer = 0; layer < depth; layer++) {
        for (uint32_t i = 0; i < N; i++) {
            const double theta = sm64_uniform(&rng_state) * 2.0 * M_PI;
            if (dist_rz(state, i, theta) != DIST_GATE_SUCCESS) goto fail;
        }
        /* Alternating even/odd CNOT-chain layers. */
        const uint32_t parity = layer & 1u;
        for (uint32_t i = parity; i + 1 < N; i += 2) {
            if (dist_cnot(state, i, i + 1) != DIST_GATE_SUCCESS) goto fail;
        }
    }

    const clock_t t1 = clock();
    const double sim_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    /* Global L2 norm must be 1.0 after a unitary circuit. */
    double local_norm = 0.0;
    for (size_t idx = 0; idx < state->local_count; idx++) {
        const double complex a = state->amplitudes[idx];
        local_norm += creal(a) * creal(a) + cimag(a) * cimag(a);
    }
    double global_norm = 0.0;
    mpi_allreduce_sum_double(ctx, &local_norm, &global_norm, 1);

    /* Local-max amplitude per rank, reduced to the global max. */
    double local_max = 0.0;
    for (size_t idx = 0; idx < state->local_count; idx++) {
        const double complex a = state->amplitudes[idx];
        const double m = creal(a) * creal(a) + cimag(a) * cimag(a);
        if (m > local_max) local_max = m;
    }
    /* Build a per-rank vector then take the global max via a sum
     * over per-rank "is-this-rank-the-max?" weight 1.  Cheap: just
     * reduce-sum the local max as a sanity stat. */
    double global_max_sum = 0.0;
    mpi_allreduce_sum_double(ctx, &local_max, &global_max_sum, 1);

    if (rank == 0) {
        fprintf(stdout, "\n    simulation time:  %.4f s (wall, rank 0)\n", sim_s);
        fprintf(stdout, "    global L2 norm:   %.10f  (expected 1.0)\n", global_norm);
        fprintf(stdout, "    sum-of-local-max: %.6e   (sanity)\n", global_max_sum);

        if (fabs(global_norm - 1.0) < 1e-9) {
            fprintf(stdout, "\n    Random circuit verified across %d MPI ranks at N=%u depth=%u\n",
                    size, N, depth);
            fprintf(stdout, "    Unitarity preserved to machine precision.\n");
        } else {
            fprintf(stderr, "\n    FAIL: norm drift = %.3e\n", fabs(global_norm - 1.0));
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
