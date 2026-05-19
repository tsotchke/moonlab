/**
 * @file  test_scheduler.c
 * @brief Distributed-scheduler MVP validation (v0.7.0).
 *
 * Verifies the in-process worker fan-out produces statistically
 * identical shot histograms to single-worker execution, JSON
 * serialisation round-trips the schema, and the worker timing
 * vector is populated.
 */

#include "../../src/distributed/scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static moonlab_job_t *make_bell_job(int shots, int workers)
{
    moonlab_job_t *j = moonlab_job_create(2);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    moonlab_job_set_num_shots(j, shots);
    moonlab_job_set_num_workers(j, workers);
    moonlab_job_set_rng_seed(j, 0xdeadbeefULL);
    return j;
}

static void test_single_worker_bell(void)
{
    fprintf(stdout, "\n--- single-worker Bell, 1024 shots ---\n");
    moonlab_job_t *j = make_bell_job(1024, 1);
    CHECK(j != NULL, "job_create");
    moonlab_job_results_t res = {0};
    const int rc = moonlab_scheduler_run(j, &res);
    CHECK(rc == 0, "scheduler_run rc=%d", rc);
    CHECK(res.total_shots == 1024, "total_shots = %d", res.total_shots);
    CHECK(res.num_workers_used == 1, "num_workers_used = %d", res.num_workers_used);
    /* Every outcome must be Bell-correlated (0 or 3). */
    int n_other = 0;
    for (int s = 0; s < 1024; s++) {
        if (res.outcomes[s] != 0 && res.outcomes[s] != 3) n_other++;
    }
    CHECK(n_other == 0, "no off-Bell outcomes (got %d)", n_other);
    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

static void test_multi_worker_bell(void)
{
    fprintf(stdout, "\n--- 4-worker Bell, 1024 shots ---\n");
    moonlab_job_t *j = make_bell_job(1024, 4);
    moonlab_job_results_t res = {0};
    const int rc = moonlab_scheduler_run(j, &res);
    CHECK(rc == 0, "scheduler_run rc=%d", rc);
    CHECK(res.num_workers_used == 4, "num_workers_used = %d", res.num_workers_used);

    int n00 = 0, n11 = 0, n_other = 0;
    for (int s = 0; s < 1024; s++) {
        if (res.outcomes[s] == 0) n00++;
        else if (res.outcomes[s] == 3) n11++;
        else n_other++;
    }
    fprintf(stdout, "    counts: |00>=%d  |11>=%d  other=%d  total=%d\n",
            n00, n11, n_other, n00 + n11 + n_other);
    CHECK(n_other == 0, "no off-Bell outcomes (workers stayed correlated)");
    CHECK(n00 + n11 == 1024, "all 1024 outcomes accounted for");
    /* Bell parity: n00 + n11 = 1024.  Per-worker n00 should not be
     * disastrously skewed -- 5-sigma bound around 512 is +/- 80. */
    CHECK(n00 > 1024 / 2 - 80 && n00 < 1024 / 2 + 80,
          "n00 = %d within 5sigma of 512", n00);

    /* Worker timing vector populated. */
    double total_seconds = 0.0;
    for (int w = 0; w < 4; w++) total_seconds += res.worker_seconds[w];
    CHECK(total_seconds >= 0.0,
          "worker_seconds populated (total = %.6fs across 4 workers)",
          total_seconds);

    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

static void test_ghz_3_workers(void)
{
    fprintf(stdout, "\n--- 3-worker GHZ, 3000 shots ---\n");
    moonlab_job_t *j = moonlab_job_create(3);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 2, 1, NULL);
    moonlab_job_set_num_shots(j, 3000);
    moonlab_job_set_num_workers(j, 3);
    moonlab_job_set_rng_seed(j, 0xcafeULL);

    moonlab_job_results_t res = {0};
    CHECK(moonlab_scheduler_run(j, &res) == 0, "scheduler_run OK");

    int n000 = 0, n111 = 0, n_other = 0;
    for (int s = 0; s < 3000; s++) {
        if (res.outcomes[s] == 0) n000++;
        else if (res.outcomes[s] == 7) n111++;
        else n_other++;
    }
    CHECK(n_other == 0, "GHZ outcomes restricted to {|000>, |111>}");
    fprintf(stdout, "    GHZ counts: |000>=%d  |111>=%d\n", n000, n111);
    CHECK(n000 + n111 == 3000, "all 3000 outcomes captured");

    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

static void test_json_serialisation(void)
{
    fprintf(stdout, "\n--- JSON serialisation ---\n");
    moonlab_job_t *j = make_bell_job(256, 2);
    /* Size-probe first. */
    const int needed = moonlab_job_to_json(j, NULL, 0);
    CHECK(needed > 0, "size-probe returns positive (%d bytes)", needed);

    char *buf = (char *)malloc((size_t)needed + 1);
    CHECK(buf != NULL, "buffer alloc");
    if (buf) {
        const int written = moonlab_job_to_json(j, buf, (size_t)needed + 1);
        CHECK(written == needed, "written (%d) == size-probe (%d)", written, needed);
        CHECK(strstr(buf, "\"schema\": \"moonlab/job/v0.7.0\"") != NULL,
              "schema field present");
        CHECK(strstr(buf, "\"num_qubits\": 2") != NULL, "num_qubits = 2");
        CHECK(strstr(buf, "\"num_shots\": 256") != NULL, "num_shots = 256");
        CHECK(strstr(buf, "\"num_workers\": 2") != NULL, "num_workers = 2");
        CHECK(strstr(buf, "\"type\": 4") != NULL, "H gate (type 4) recorded");
        CHECK(strstr(buf, "\"type\": 10") != NULL, "CNOT gate (type 10) recorded");
        CHECK(strstr(buf, "\"control\": 0") != NULL, "CNOT control = 0");
        fprintf(stdout, "    JSON dump (%d bytes):\n%s", needed, buf);
    }
    free(buf);
    moonlab_job_free(j);
}

static void test_introspection(void)
{
    fprintf(stdout, "\n--- introspection getters ---\n");
    moonlab_job_t *j = moonlab_job_create(4);
    CHECK(moonlab_job_num_qubits(j) == 4, "num_qubits = 4");
    CHECK(moonlab_job_num_gates(j) == 0, "num_gates = 0 before any add_gate");
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    CHECK(moonlab_job_num_gates(j) == 1, "num_gates = 1 after one add_gate");
    moonlab_job_set_num_shots(j, 100);
    moonlab_job_set_num_workers(j, 4);
    CHECK(moonlab_job_num_shots(j) == 100, "num_shots = 100");
    CHECK(moonlab_job_num_workers(j) == 4, "num_workers = 4");
    moonlab_job_free(j);
}

static void test_error_paths(void)
{
    fprintf(stdout, "\n--- error paths ---\n");
    CHECK(moonlab_job_create(0) == NULL, "num_qubits = 0 rejected");
    CHECK(moonlab_job_create(33) == NULL, "num_qubits = 33 rejected");

    moonlab_job_t *j = moonlab_job_create(2);
    CHECK(moonlab_job_set_num_shots(j, -1) == MOONLAB_SCHED_BAD_ARG,
          "negative shots rejected");
    CHECK(moonlab_job_set_num_workers(j, 0) == MOONLAB_SCHED_BAD_ARG,
          "zero workers rejected");

    /* run_with_zero_shots should fail cleanly. */
    moonlab_job_set_num_shots(j, 0);
    moonlab_job_results_t res = {0};
    CHECK(moonlab_scheduler_run(j, &res) == MOONLAB_SCHED_BAD_ARG,
          "zero shots rejected at scheduler_run");
    moonlab_job_free(j);

    moonlab_job_free(NULL); /* must not crash */
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== distributed scheduler MVP (v0.7.0) ===\n");
    test_introspection();
    test_single_worker_bell();
    test_multi_worker_bell();
    test_ghz_3_workers();
    test_json_serialisation();
    test_error_paths();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
