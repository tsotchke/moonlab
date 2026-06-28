/**
 * @file  test_scheduler_mpi.c
 * @brief MPI scheduler transport validation (v0.7.4).
 *
 * Standalone MPI test program.  Initialises MPI via the bridge,
 * runs a Bell circuit on 1024 shots through `moonlab_scheduler_run_mpi`,
 * gathers outcomes to rank 0, asserts Bell-correlation across the
 * full merged buffer.
 *
 * Built only when QSIM_ENABLE_MPI=ON; ctest runs it under
 * `mpirun -n 4`.
 */

#include "../../src/distributed/scheduler.h"
#include "../../src/distributed/mpi_bridge.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK_RANK0(rank, cond, fmt, ...) do {                  \
    if (rank == 0) {                                            \
        if (!(cond)) {                                          \
            fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);\
            failures++;                                         \
        } else {                                                \
            fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);\
        }                                                       \
    }                                                           \
} while (0)

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    distributed_ctx_t *ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) {
        fprintf(stderr, "mpi_bridge_init returned NULL; aborting\n");
        return 1;
    }
    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);

    if (rank == 0) {
        fprintf(stdout, "=== MPI scheduler transport on %d rank%s ===\n\n",
                size, size == 1 ? "" : "s");
    }

    /* Rank 0 builds the job; non-root passes NULL. */
    moonlab_job_t *job = NULL;
    if (rank == 0) {
        job = moonlab_job_create(2);
        moonlab_job_add_gate(job, MOONLAB_QGTL_GATE_H,    0, -1, NULL);
        moonlab_job_add_gate(job, MOONLAB_QGTL_GATE_CNOT, 1,  0, NULL);
        moonlab_job_set_num_shots(job, 1024);
        moonlab_job_set_rng_seed(job, 0xdeadbeefULL);
    }

    moonlab_job_results_t res = {0};
    const int rc = moonlab_scheduler_run_mpi(job, &res, ctx);
    CHECK_RANK0(rank, rc == 0, "scheduler_run_mpi rc=%d", rc);

    if (rank == 0 && rc == 0) {
        CHECK_RANK0(rank, res.total_shots == 1024,
                    "merged buffer has 1024 outcomes (got %d)", res.total_shots);
        CHECK_RANK0(rank, res.num_workers_used == size,
                    "num_workers_used = MPI size (%d)", res.num_workers_used);
        int n00 = 0, n11 = 0, nother = 0;
        for (int s = 0; s < res.total_shots; s++) {
            if (res.outcomes[s] == 0) n00++;
            else if (res.outcomes[s] == 3) n11++;
            else nother++;
        }
        fprintf(stdout, "    counts: |00>=%d  |11>=%d  other=%d (across %d ranks)\n",
                n00, n11, nother, size);
        CHECK_RANK0(rank, nother == 0,
                    "no off-Bell outcomes across MPI ranks");
        CHECK_RANK0(rank, n00 + n11 == 1024,
                    "all 1024 outcomes Bell-correlated");
        CHECK_RANK0(rank, n00 > 1024 / 2 - 100 && n00 < 1024 / 2 + 100,
                    "n00 = %d within statistical bounds of 512", n00);
    }

    moonlab_job_results_free(&res);
    if (job) moonlab_job_free(job);

    if (rank == 0) {
        fprintf(stdout, "\n=== %d failure%s ===\n",
                failures, failures == 1 ? "" : "s");
    }

    mpi_barrier(ctx);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();

    /* Only rank 0's exit code matters for ctest. */
    return (rank == 0 && failures > 0) ? 1 : 0;
}
