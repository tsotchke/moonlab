/**
 * @file test_distributed_gates.c
 * @brief MPI end-to-end smoke test.
 *
 * Meant to be launched under `mpirun -np <N>`. Exercises:
 *  - MPI bridge init / finalize
 *  - Rank / size / local-range queries
 *  - An allreduce_sum_double so inter-rank comm is verified
 *  - mpi_sendrecv wrapper round-trip between rank 0 and rank 1 (np>=2)
 *  - mpi_barrier
 *
 * Exits 0 on every rank when everything passes. Returns non-zero on
 * any rank that observed a failure, so `mpirun` aggregates to non-zero.
 */

#include "../../src/distributed/mpi_bridge.h"
#include "../../src/distributed/state_partition.h"
#include "../../src/distributed/distributed_gates.h"
#include <mpi.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    }                                                           \
} while (0)

int main(int argc, char** argv) {
    distributed_ctx_t* ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) {
        fprintf(stderr, "mpi_bridge_init failed\n");
        return 1;
    }

    int rank = mpi_get_rank(ctx);
    int size = mpi_get_size(ctx);
    CHECK(rank >= 0 && rank < size,
          "rank=%d is in [0, size=%d)", rank, size);
    CHECK(size >= 1, "size=%d >= 1", size);

    /* allreduce_sum_double: each rank contributes its own rank+1, the
     * sum should be size*(size+1)/2. */
    double local = (double)(rank + 1);
    double global = 0.0;
    mpi_bridge_error_t er = mpi_allreduce_sum_double(ctx, &local, &global, 1);
    CHECK(er == MPI_BRIDGE_SUCCESS,
          "allreduce_sum_double returns success (got %d)", er);
    double expected = 0.5 * (double)size * (double)(size + 1);
    CHECK(fabs(global - expected) < 1e-12,
          "allreduce sum == size*(size+1)/2: got %.2f vs expected %.2f",
          global, expected);

    /* mpi_sendrecv: if size >= 2, rank 0 <-> rank 1 exchange payload. */
    if (size >= 2) {
        const int partner = (rank == 0) ? 1 : (rank == 1 ? 0 : -1);
        if (partner != -1) {
            uint64_t send_val = (uint64_t)rank * 0xDEADBEEFULL + 0x1234ULL;
            uint64_t recv_val = 0;
            er = mpi_sendrecv(ctx,
                              &send_val, sizeof send_val, partner,
                              &recv_val, sizeof recv_val, partner);
            CHECK(er == MPI_BRIDGE_SUCCESS,
                  "mpi_sendrecv rank %d <-> %d returned success (got %d)",
                  rank, partner, er);
            uint64_t expect_from =
                (uint64_t)partner * 0xDEADBEEFULL + 0x1234ULL;
            CHECK(recv_val == expect_from,
                  "rank %d got 0x%llx from rank %d, expected 0x%llx",
                  rank, (unsigned long long)recv_val, partner,
                  (unsigned long long)expect_from);
        }
    }

    er = mpi_barrier(ctx);
    CHECK(er == MPI_BRIDGE_SUCCESS,
          "mpi_barrier returns success (got %d)", er);

    /* H(0) + CNOT(0, n-1) where qubit n-1 is a partition qubit should
     * produce (|0...0> + |1*...*1>) / sqrt(2). Verify norm=1, P(0)=0.5. */
    int size_is_pow2 = size > 0 && (size & (size - 1)) == 0;
    if (size_is_pow2) {
        uint32_t partition_bits = 0;
        uint64_t s = (uint64_t)size;
        while (s > 1) { s >>= 1; partition_bits++; }
        uint32_t num_qubits = partition_bits + 2;

        partitioned_state_t *pstate =
            partition_state_create(ctx, num_qubits, NULL);
        CHECK(pstate != NULL,
              "partition_state_create(%u qubits, np=%d) succeeds",
              num_qubits, size);

        if (pstate) {
            CHECK(partition_init_zero(pstate) == PARTITION_SUCCESS,
                  "partition_init_zero(|0...0>)");

            dist_gate_error_t g = dist_hadamard(pstate, 0);
            CHECK(g == DIST_GATE_SUCCESS,
                  "dist_hadamard on local qubit 0 (err=%d)", g);

            g = dist_cnot(pstate, 0, num_qubits - 1);
            CHECK(g == DIST_GATE_SUCCESS,
                  "dist_cnot(0, %u) across partition boundary (err=%d)",
                  num_qubits - 1, g);

            /* Local squared-norm should allreduce-sum to 1.0. */
            double local_sq = 0.0;
            for (uint64_t i = 0; i < pstate->local_count; i++) {
                double m = cabs(pstate->amplitudes[i]);
                local_sq += m * m;
            }
            double global_sq = 0.0;
            mpi_allreduce_sum_double(ctx, &local_sq, &global_sq, 1);
            CHECK(fabs(global_sq - 1.0) < 1e-10,
                  "norm^2 after H + distributed CNOT = %.10f "
                  "(expected 1.0)", global_sq);

            /* P(|0...0>) should be 1/2. Rank 0 owns the local index 0,
             * which corresponds to global index 0 = |0...0>. */
            if (rank == 0) {
                double p00 = cabs(pstate->amplitudes[0]);
                p00 = p00 * p00;
                CHECK(fabs(p00 - 0.5) < 1e-10,
                      "P(|0...0>) after H+CNOT = %.10f (expected 0.5)",
                      p00);
            }

            partition_state_free(pstate);
        }

        /* SWAP test: put |1> on partition qubit via X, SWAP across to
         * local qubit 0, verify the excitation moved. */
        pstate = partition_state_create(ctx, num_qubits, NULL);
        if (pstate) {
            partition_init_zero(pstate);
            dist_gate_error_t g =
                dist_pauli_x(pstate, num_qubits - 1);
            CHECK(g == DIST_GATE_SUCCESS,
                  "dist_pauli_x on partition qubit");
            g = dist_swap(pstate, 0, num_qubits - 1);
            CHECK(g == DIST_GATE_SUCCESS,
                  "dist_swap(0, %u) across partition (err=%d)",
                  num_qubits - 1, g);

            /* After X then SWAP, the |1> should sit on qubit 0, so the
             * global amplitude that is non-zero is at index 1 (bit 0
             * set). That index lives on rank 0. */
            double local_sq = 0.0;
            for (uint64_t i = 0; i < pstate->local_count; i++) {
                double m = cabs(pstate->amplitudes[i]);
                local_sq += m * m;
            }
            double global_sq = 0.0;
            mpi_allreduce_sum_double(ctx, &local_sq, &global_sq, 1);
            CHECK(fabs(global_sq - 1.0) < 1e-10,
                  "norm^2 after X+SWAP = %.10f", global_sq);

            if (rank == 0) {
                /* amp[1] should be 1.0 (|0...01>). */
                double a = cabs(pstate->amplitudes[1]);
                CHECK(fabs(a * a - 1.0) < 1e-10,
                      "P(|0...01>) = %.10f (expected 1.0)", a * a);
            }
            partition_state_free(pstate);
        }

        /* Toffoli test: prepare |1>|1>|0> via two X gates then apply
         * Toffoli across the partition; target bit should flip. Only
         * exercised for size >= 4 so we get 3 qubits + partition bits. */
        if (size >= 4 && num_qubits >= 3) {
            pstate = partition_state_create(ctx, num_qubits, NULL);
            if (pstate) {
                partition_init_zero(pstate);
                dist_pauli_x(pstate, 0);
                dist_pauli_x(pstate, 1);
                dist_gate_error_t g =
                    dist_toffoli(pstate, 0, 1, num_qubits - 1);
                CHECK(g == DIST_GATE_SUCCESS,
                      "dist_toffoli(0,1,%u) across partition (err=%d)",
                      num_qubits - 1, g);

                double local_sq = 0.0;
                for (uint64_t i = 0; i < pstate->local_count; i++) {
                    double m = cabs(pstate->amplitudes[i]);
                    local_sq += m * m;
                }
                double global_sq = 0.0;
                mpi_allreduce_sum_double(ctx, &local_sq, &global_sq, 1);
                CHECK(fabs(global_sq - 1.0) < 1e-10,
                      "norm^2 after Toffoli = %.10f", global_sq);
                partition_state_free(pstate);
            }
        }
    }

    /* Rank 0 aggregates failure count across all ranks. */
    int my_failures = failures;
    int total_failures = 0;
    MPI_Reduce(&my_failures, &total_failures, 1, MPI_INT, MPI_SUM, 0,
               *(MPI_Comm*)ctx->mpi_comm);

    if (rank == 0) {
        fprintf(stdout, "=== MPI distributed smoke (np=%d): %d failure%s ===\n",
                size, total_failures, total_failures == 1 ? "" : "s");
    }

    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return failures == 0 ? 0 : 1;
}
