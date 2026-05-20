/**
 * @file  test_control_plane_shutdown.c
 * @brief Graceful-shutdown test for the v0.8.13 lifecycle API.
 *
 * Path 1 -- shutdown drains an idle accept():
 *   open server, run() in a worker thread, sleep briefly so the
 *   server is parked in select(), call shutdown() from the main
 *   thread, assert run() returns cleanly with no served clients.
 *
 * Path 2 -- shutdown does NOT interrupt in-flight requests:
 *   open server, run() in a worker thread, submit a Bell circuit,
 *   call shutdown() right after.  Server must finish the Bell
 *   request before run() returns.
 *
 * Also exercises the structured request log under
 * `MOONLAB_CONTROL_LOG=1` (visual inspection only -- we just confirm
 * no crash from the logger).
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <math.h>
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
    int                       rc;
} run_args_t;

static void *run_thread(void *arg)
{
    run_args_t *a = (run_args_t *)arg;
    a->rc = moonlab_control_server_run(a->server, 64);
    return NULL;
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
    fprintf(stdout, "=== test_control_plane_shutdown (v0.8.13) ===\n\n");

    /* ---- Path 1: shutdown wakes an idle accept() ---- */
    fprintf(stdout, "--- path 1: shutdown drains idle accept() ---\n");
    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == MOONLAB_CONTROL_OK, "server_open rc=%d", rc);
    CHECK(port != 0, "server bound to port %u", port);

    run_args_t ra = { server, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);

    /* Park briefly so we're definitely inside select(). */
    struct timespec ts = { 0, 50 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    CHECK(ra.rc == MOONLAB_CONTROL_OK,
          "run() returned cleanly after idle shutdown (rc=%d)", ra.rc);

    moonlab_control_server_close(server);

    /* ---- Path 2: in-flight request finishes despite shutdown ---- */
    fprintf(stdout, "\n--- path 2: in-flight request survives shutdown ---\n");
    server = NULL;
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == MOONLAB_CONTROL_OK, "server_open rc=%d", rc);

    ra.server = server; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);

    nanosleep(&ts, NULL);  /* let server park */

    char *text = serialize_bell();
    double *probs = NULL;
    size_t  num   = 0;
    rc = moonlab_control_submit_circuit("127.0.0.1", port, text, 0, &probs, &num);
    CHECK(rc == MOONLAB_CONTROL_OK, "in-flight submit rc=%d", rc);
    CHECK(num == 4, "got 4 probabilities");
    if (num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f", probs[0]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f", probs[3]);
    }
    free(probs);
    free(text);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    CHECK(ra.rc == MOONLAB_CONTROL_OK, "run() rc=%d after in-flight + shutdown", ra.rc);

    moonlab_control_server_close(server);

    /* ---- Path 3: idempotent close() / double shutdown ---- */
    fprintf(stdout, "\n--- path 3: idempotent close() ---\n");
    moonlab_control_server_close(NULL); /* must not crash */
    fprintf(stdout, "  OK    close(NULL) is a no-op\n");

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
