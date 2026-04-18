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
#include <mpi.h>
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
