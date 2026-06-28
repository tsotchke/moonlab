/**
 * @file  test_control_plane_ipv6.c
 * @brief IPv6 (since v0.10.0): bind ::1, accept v6 client, scrape
 *        METRICS, drive a CIRCUIT round-trip, and exercise the
 *        rate-limit key path under v6.
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    fflush(stdout);                                             \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
        fflush(stdout);                                         \
    }                                                           \
} while (0)

typedef struct {
    moonlab_control_server_t *server;
    int                       max_iters;
    int                       rc;
} ra_t;

static void *run_thread(void *arg) {
    ra_t *a = (ra_t *)arg;
    a->rc = moonlab_control_server_run(a->server, a->max_iters);
    return NULL;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stdout, "=== test_control_plane_ipv6 ===\n\n");

    /* Bind on ::1 (loopback v6 only).  Validates the v6-only path. */
    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("::1", 0, &server, &port);
    CHECK(rc == 0, "server_open(::1) rc=%d", rc);
    if (rc != 0) { fprintf(stdout, "=== %d failure(s) ===\n", failures); return 1; }

    /* Configure rate limit to exercise the v6 key path. */
    rc = moonlab_control_server_set_rate_limit(server, 100, 100);
    CHECK(rc == 0, "set_rate_limit rc=%d", rc);

    ra_t ra = { server, 16, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    /* HEALTH against ::1. */
    rc = moonlab_control_submit_health("::1", port);
    CHECK(rc == 0, "submit_health(::1) rc=%d", rc);

    /* METRICS against ::1; v6 client must be accepted and rate-limit
     * bucket must allocate under the v6 key. */
    char *body = NULL;
    rc = moonlab_control_submit_metrics("::1", port, &body);
    CHECK(rc == 0 && body != NULL, "submit_metrics(::1) rc=%d", rc);
    if (body) {
        CHECK(strstr(body, "moonlab_control_requests_total") != NULL,
              "METRICS body contains Prometheus counter");
        free(body);
    }

    /* CIRCUIT round-trip: build a Bell pair via QGTL, submit, verify. */
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *text = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, text, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);

    double *probs = NULL;
    size_t  num   = 0;
    rc = moonlab_control_submit_circuit("::1", port, text, 0, &probs, &num);
    CHECK(rc == 0, "submit_circuit(::1) rc=%d", rc);
    CHECK(num == 4, "4 probabilities returned (got %zu)", num);
    if (probs && num == 4) {
        CHECK(probs[0] > 0.49 && probs[0] < 0.51, "P(00) ~= 0.5 (got %.3f)", probs[0]);
        CHECK(probs[3] > 0.49 && probs[3] < 0.51, "P(11) ~= 0.5 (got %.3f)", probs[3]);
    }
    free(probs);
    free(text);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
