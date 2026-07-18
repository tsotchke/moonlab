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
#include "../../src/distributed/collective_ops.h"
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

        /* GHZ preparation across every qubit.  H on qubit 0 then a
         * CNOT chain (0 -> 1 -> 2 -> ...) produces
         *   |GHZ_n> = (|0...0> + |1...1>) / sqrt(2).
         * With partition_bits partition qubits, the top qubit sits
         * across the partition; the CNOT chain therefore exercises
         * both local and cross-partition two-qubit dispatch.  Verify
         * the amplitude of |0...0> (index 0) and |1...1>
         * (index (1 << num_qubits) - 1) both round to 1/sqrt(2). */
        {
            partitioned_state_t *pstate2 =
                partition_state_create(ctx, num_qubits, NULL);
            if (pstate2) {
                CHECK(partition_init_zero(pstate2) == PARTITION_SUCCESS,
                      "GHZ: partition_init_zero");
                dist_gate_error_t g = dist_hadamard(pstate2, 0);
                CHECK(g == DIST_GATE_SUCCESS,
                      "GHZ: H(0) (err=%d)", g);
                for (uint32_t q = 0; q + 1 < num_qubits; q++) {
                    g = dist_cnot(pstate2, q, q + 1);
                    CHECK(g == DIST_GATE_SUCCESS,
                          "GHZ: CNOT(%u,%u) (err=%d)", q, q + 1, g);
                }

                double local_sq = 0.0;
                for (uint64_t i = 0; i < pstate2->local_count; i++) {
                    double m = cabs(pstate2->amplitudes[i]);
                    local_sq += m * m;
                }
                double global_sq = 0.0;
                mpi_allreduce_sum_double(ctx, &local_sq, &global_sq, 1);
                CHECK(fabs(global_sq - 1.0) < 1e-10,
                      "GHZ: norm^2 = %.10f", global_sq);

                /* Rank 0 owns global index 0 (|0...0>).  The last
                 * rank owns the top of the distribution so it holds
                 * global index (1 << num_qubits) - 1 = |1...1>. */
                double local_p_zero = 0.0, local_p_ones = 0.0;
                const uint64_t global_ones =
                    ((uint64_t)1 << num_qubits) - 1;
                if (pstate2->local_start == 0 && pstate2->local_count > 0) {
                    double a = cabs(pstate2->amplitudes[0]);
                    local_p_zero = a * a;
                }
                if (global_ones >= pstate2->local_start &&
                    global_ones <  pstate2->local_start + pstate2->local_count) {
                    uint64_t idx = global_ones - pstate2->local_start;
                    double a = cabs(pstate2->amplitudes[idx]);
                    local_p_ones = a * a;
                }
                double p_zero_global = 0.0, p_ones_global = 0.0;
                mpi_allreduce_sum_double(ctx, &local_p_zero,
                                         &p_zero_global, 1);
                mpi_allreduce_sum_double(ctx, &local_p_ones,
                                         &p_ones_global, 1);
                CHECK(fabs(p_zero_global - 0.5) < 1e-10,
                      "GHZ: P(|0...0>) = %.10f (expected 0.5)",
                      p_zero_global);
                CHECK(fabs(p_ones_global - 0.5) < 1e-10,
                      "GHZ: P(|1...1>) = %.10f (expected 0.5)",
                      p_ones_global);

                /* Coverage smoke: exercise additional public symbols
                 * in src/distributed/ that lack in-tree callers.
                 * Each call drives the API path; numerical assertions
                 * are loose -- the point is to surface API breakage,
                 * not pin specific physics.  Calls limited to those
                 * with simple collective semantics so all ranks
                 * uniformly enter the MPI ops below.  More
                 * sophisticated patterns (collective_top_k_states,
                 * partition_fetch_remote / scatter_updates) need
                 * carefully-matched per-rank descriptors -- queued
                 * for a follow-up MPI-only test under v0.3. */

                /* distributed_gates.h: multi-controlled X.  GHZ has
                 * 50/50 weight on |0...0> and |1...1>; flipping with
                 * controls = (q0, q1) on the |1...1> branch routes
                 * through the cross-partition multi-control path. */
                if (num_qubits >= 3) {
                    uint32_t controls[2] = { 0, 1 };
                    dist_gate_error_t dg = dist_mcx(
                        pstate2, controls, 2, num_qubits - 1);
                    CHECK(dg == DIST_GATE_SUCCESS,
                          "dist_mcx({0,1}, %u) = %d",
                          num_qubits - 1, dg);
                }

                /* collective_ops.h: per-qubit X/Y/Z expectations and
                 * Pauli-string expectation.  Each is a pure allreduce
                 * with the same descriptor on every rank, so calling
                 * with identical scalar args from all ranks is safe. */
                {
                    double ev_x = 0.0, ev_y = 0.0, ev_z = 0.0;
                    collective_error_t ce =
                        collective_expectation_x(pstate2, 0, &ev_x);
                    CHECK(ce == COLLECTIVE_SUCCESS,
                          "collective_expectation_x(0) = %d", ce);
                    ce = collective_expectation_y(pstate2, 0, &ev_y);
                    CHECK(ce == COLLECTIVE_SUCCESS,
                          "collective_expectation_y(0) = %d", ce);
                    ce = collective_expectation_z(pstate2, 0, &ev_z);
                    CHECK(ce == COLLECTIVE_SUCCESS,
                          "collective_expectation_z(0) = %d", ce);

                    char* pauli_string = malloc(num_qubits + 1);
                    if (pauli_string) {
                        for (uint32_t q = 0; q < num_qubits; q++) {
                            pauli_string[q] = 'Z';
                        }
                        pauli_string[num_qubits] = '\0';
                        double pauli_ev = 0.0;
                        ce = collective_expectation_pauli(
                            pstate2, pauli_string, &pauli_ev);
                        CHECK(ce == COLLECTIVE_SUCCESS,
                              "collective_expectation_pauli('%s') = %d",
                              pauli_string, ce);
                        free(pauli_string);
                    }
                }

                partition_state_free(pstate2);
            }
        }

        /* Bounded-buffer exchange regression: force a tiny communication
         * buffer and exercise bounded chunking on:
         *   - CNOT where target is a partition qubit
         *   - 2Q gate with one partition + one local qubit
         *   - 2Q gate with both qubits in partition space */
        if (size >= 4) {
            const uint32_t bounded_n = num_qubits + 2;
            partition_config_t tiny_cfg = {0};
            tiny_cfg.use_aligned_memory = 1;
            tiny_cfg.comm_buffer_size = 2 * sizeof(double complex);

            partitioned_state_t* bounded = partition_state_create(
                ctx, bounded_n, &tiny_cfg);
            CHECK(bounded != NULL,
                  "bounded-buffer state create(%u qubits) succeeds",
                  bounded_n);

            if (bounded) {
                CHECK(bounded->buffer_size == 2,
                      "bounded buffer resolves to 2 amplitudes (got %zu)",
                      bounded->buffer_size);

                CHECK(partition_init_zero(bounded) == PARTITION_SUCCESS,
                      "bounded state initialized to |0...0>");

                dist_gate_error_t g;
                const uint32_t lp = bounded->local_qubits;
                const uint32_t p0 = lp;          // first partition qubit
                const uint64_t idx_base = 0;
                const uint64_t idx_h = (1ULL << p0) | 1ULL;
                const uint64_t idx_cnot_src = 1ULL;
                const uint64_t idx_cnot_dst = (1ULL << p0) | idx_cnot_src;
                const uint64_t idx_pp_src = (1ULL << p0);
                const uint64_t idx_pp_dst = (1ULL << (p0 + 1)) | idx_pp_src;

                g = dist_hadamard(bounded, p0);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer remote dist_hadamard(%u) succeeds (err=%d)",
                      p0, g);
                double bounded_x = 0.0;
                collective_error_t ce =
                    collective_expectation_x(bounded, p0, &bounded_x);
                CHECK(ce == COLLECTIVE_SUCCESS && fabs(bounded_x - 1.0) < 1e-12,
                      "bounded-buffer remote <X_%u> = %.12f (err=%d)",
                      p0, bounded_x, ce);
                g = dist_s_gate(bounded, p0);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer remote dist_s_gate(%u) succeeds (err=%d)",
                      p0, g);
                double bounded_y = 0.0;
                ce = collective_expectation_y(bounded, p0, &bounded_y);
                CHECK(ce == COLLECTIVE_SUCCESS && fabs(bounded_y - 1.0) < 1e-12,
                      "bounded-buffer remote <Y_%u> = %.12f (err=%d)",
                      p0, bounded_y, ce);
                CHECK(partition_init_zero(bounded) == PARTITION_SUCCESS,
                      "bounded state reset after remote expectation");

                g = dist_hadamard(bounded, 0);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer dist_hadamard(0) succeeds (err=%d)", g);
                g = dist_cnot(bounded, 0, p0);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer dist_cnot(0, %u) succeeds (err=%d)",
                      p0, g);

                double p0_local = 0.0, p1_local = 0.0;
                {
                    double complex a0 = partition_get_amplitude(bounded, idx_base);
                    double complex a1 = partition_get_amplitude(bounded, idx_h);
                    p0_local = cabs(a0) * cabs(a0);
                    p1_local = cabs(a1) * cabs(a1);
                }
                double p0_global = 0.0, p1_global = 0.0;
                mpi_allreduce_sum_double(ctx, &p0_local, &p0_global, 1);
                mpi_allreduce_sum_double(ctx, &p1_local, &p1_global, 1);
                CHECK(fabs(p0_global - 0.5) < 1e-12 &&
                      fabs(p1_global - 0.5) < 1e-12,
                      "bounded-buffer CNOT preserves |01> and |11> probs: "
                      "P0=%f P1=%f", p0_global, p1_global);

                CHECK(partition_init_zero(bounded) == PARTITION_SUCCESS,
                      "bounded-buffer state reset to |0...0>");
                if (partition_get_owner(bounded, idx_base) == rank) {
                    CHECK(partition_set_amplitude(bounded, idx_base, 0.0)
                          == PARTITION_SUCCESS,
                          "bounded-buffer clear |0...0> before basis init");
                }
                if (partition_get_owner(bounded, idx_cnot_src) == rank) {
                    CHECK(partition_set_amplitude(bounded, idx_cnot_src, 1.0)
                          == PARTITION_SUCCESS,
                          "bounded-buffer init basis %llu",
                          (unsigned long long)idx_cnot_src);
                }
                g = dist_gate_2q(bounded, 0, p0, &GATE_CNOT);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer dist_gate_2q(0,%u,&CNOT) succeeds (err=%d)",
                      p0, g);

                {
                    double p_src_local = 0.0, p_dst_local = 0.0;
                    double complex src = partition_get_amplitude(bounded, idx_cnot_src);
                    double complex dst = partition_get_amplitude(bounded, idx_cnot_dst);
                    p_src_local = cabs(src) * cabs(src);
                    p_dst_local = cabs(dst) * cabs(dst);
                    double p_src_global = 0.0, p_dst_global = 0.0;
                    mpi_allreduce_sum_double(ctx, &p_src_local, &p_src_global, 1);
                    mpi_allreduce_sum_double(ctx, &p_dst_local, &p_dst_global, 1);
                    CHECK(fabs(p_src_global) < 1e-12 && fabs(p_dst_global - 1.0) < 1e-12,
                          "bounded-buffer single-partition 2Q remaps amplitude: "
                          "Psrc=%f Pdst=%f", p_src_global, p_dst_global);
                }

                CHECK(partition_init_zero(bounded) == PARTITION_SUCCESS,
                      "bounded-buffer state reset before both-partition 2Q");
                if (partition_get_owner(bounded, idx_base) == rank) {
                    CHECK(partition_set_amplitude(bounded, idx_base, 0.0)
                          == PARTITION_SUCCESS,
                          "bounded-buffer clear |0...0> before partition basis init");
                }
                if (partition_get_owner(bounded, idx_pp_src) == rank) {
                    CHECK(partition_set_amplitude(bounded, idx_pp_src, 1.0)
                          == PARTITION_SUCCESS,
                          "bounded-buffer init basis %llu",
                          (unsigned long long)idx_pp_src);
                }
                g = dist_gate_2q(bounded, p0, p0 + 1, &GATE_CNOT);
                CHECK(g == DIST_GATE_SUCCESS,
                      "bounded-buffer dist_gate_2q(%u,%u,&CNOT) succeeds (err=%d)",
                      p0, p0 + 1, g);

                {
                    double p_src_local = 0.0, p_dst_local = 0.0;
                    double complex src = partition_get_amplitude(bounded, idx_pp_src);
                    double complex dst = partition_get_amplitude(bounded, idx_pp_dst);
                    p_src_local = cabs(src) * cabs(src);
                    p_dst_local = cabs(dst) * cabs(dst);
                    double p_src_global = 0.0, p_dst_global = 0.0;
                    mpi_allreduce_sum_double(ctx, &p_src_local, &p_src_global, 1);
                    mpi_allreduce_sum_double(ctx, &p_dst_local, &p_dst_global, 1);
                    CHECK(fabs(p_src_global) < 1e-12 && fabs(p_dst_global - 1.0) < 1e-12,
                          "bounded-buffer both-partition 2Q remaps amplitude: "
                          "Psrc=%f Pdst=%f", p_src_global, p_dst_global);
                }

                partition_state_free(bounded);
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
