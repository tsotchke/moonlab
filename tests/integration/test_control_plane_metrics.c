/**
 * @file  test_control_plane_metrics.c
 * @brief Prometheus METRICS endpoint test for the v0.8.23 control plane.
 *
 * Sequence:
 *   - submit a CIRCUIT (Bell)
 *   - submit two SHOTS requests
 *   - submit two HEALTH probes
 *   - scrape METRICS
 *   - verify the counters reflect the activity
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

typedef struct {
    moonlab_control_server_t *server;
    int max_iters;
    int rc;
} ra_t;

static void *run_thread(void *arg)
{
    ra_t *a = (ra_t *)arg;
    a->rc = moonlab_control_server_run(a->server, a->max_iters);
    return NULL;
}

/* Return the counter value for a given verb name from the
 * Prometheus exposition body. */
static long parse_count(const char *body, const char *line_prefix)
{
    const char *p = strstr(body, line_prefix);
    if (!p) return -1;
    p += strlen(line_prefix);
    long v = -1;
    if (sscanf(p, "%ld", &v) != 1) return -1;
    return v;
}

static char *serialize_bell(void)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *buf = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, buf, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);
    return buf;
}

int main(void)
{
    fprintf(stdout, "=== test_control_plane_metrics (v0.8.23) ===\n\n");

    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);

    /* 6 cycles: 1 CIRCUIT + 2 SHOTS + 2 HEALTH + 1 METRICS scrape. */
    ra_t ra = { server, 6, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    char *text = serialize_bell();

    /* 1 CIRCUIT */
    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit("127.0.0.1", port, text, 0, &probs, &num);
    CHECK(rc == 0, "CIRCUIT submit rc=%d", rc);
    free(probs);

    /* 2 SHOTS */
    for (int i = 0; i < 2; i++) {
        uint64_t *outcomes = NULL; size_t nout = 0;
        rc = moonlab_control_submit_circuit_shots(
            "127.0.0.1", port, text, 0, 16, &outcomes, &nout);
        CHECK(rc == 0, "SHOTS #%d rc=%d", i + 1, rc);
        free(outcomes);
    }

    /* 2 HEALTH */
    for (int i = 0; i < 2; i++) {
        rc = moonlab_control_submit_health("127.0.0.1", port);
        CHECK(rc == 0, "HEALTH #%d rc=%d", i + 1, rc);
    }

    /* 1 METRICS scrape */
    char *metrics = NULL;
    rc = moonlab_control_submit_metrics("127.0.0.1", port, &metrics);
    CHECK(rc == 0, "METRICS scrape rc=%d", rc);
    CHECK(metrics != NULL, "metrics body non-NULL");

    if (metrics) {
        fprintf(stdout, "\n--- metrics body ---\n%s\n", metrics);

        const long c_circuit = parse_count(metrics,
            "moonlab_control_requests_total{verb=\"CIRCUIT\"} ");
        const long c_shots   = parse_count(metrics,
            "moonlab_control_requests_total{verb=\"SHOTS\"} ");
        const long c_health  = parse_count(metrics,
            "moonlab_control_requests_total{verb=\"HEALTH\"} ");
        const long c_metrics = parse_count(metrics,
            "moonlab_control_requests_total{verb=\"METRICS\"} ");

        /* Counters are process-wide, so just check >= our increment. */
        CHECK(c_circuit >= 1, "CIRCUIT counter >= 1 (got %ld)", c_circuit);
        CHECK(c_shots   >= 2, "SHOTS counter   >= 2 (got %ld)", c_shots);
        CHECK(c_health  >= 2, "HEALTH counter  >= 2 (got %ld)", c_health);
        CHECK(c_metrics >= 1, "METRICS counter >= 1 (got %ld)", c_metrics);

        /* v1.0.3: admission_refused_total and completion_hook_fires_total
         * should be present in the exposition.  Their values are >= 0;
         * this test does not install an admission hook so refused
         * stays at 0, but the LINE must be present so SREs can scrape
         * a fresh server without seeing an "absent" gap. */
        CHECK(strstr(metrics,
            "moonlab_control_admission_refused_total") != NULL,
            "admission_refused_total counter present in METRICS body");
        CHECK(strstr(metrics,
            "moonlab_control_completion_hook_fires_total") != NULL,
            "completion_hook_fires_total counter present in METRICS body");
    }
    free(metrics);

    pthread_join(tid, NULL);
    moonlab_control_server_close(server);
    free(text);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
