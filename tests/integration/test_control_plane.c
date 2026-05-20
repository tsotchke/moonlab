/**
 * @file  test_control_plane.c
 * @brief Round-trip test for the v0.8.7 TCP control plane.
 *
 * Spins up the server in a worker thread, submits a Bell circuit
 * over loopback, confirms the returned probability vector matches
 * the expected (0.5, 0, 0, 0.5) signature.
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
    uint16_t port;
    int      max_iters;
    int      rc;
    /* The control-plane writes the OS-chosen port into `bind_port`
     * after bind() but before the blocking accept() returns, so the
     * test thread can read it while the server is still in accept().
     * `volatile` is sufficient since aligned 16-bit stores are atomic
     * on the architectures we target (x86_64 / aarch64). */
    volatile uint16_t bind_port;
} server_args_t;

static void *server_thread(void *arg)
{
    server_args_t *sa = (server_args_t *)arg;
    sa->rc = moonlab_control_serve(
        "127.0.0.1", sa->port, sa->max_iters,
        (uint16_t *)&sa->bind_port);
    return NULL;
}

static char *serialize_bell(void)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    if (!c) return NULL;
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
    fprintf(stdout, "=== test_control_plane (v0.8.7) ===\n\n");
    fprintf(stdout, "--- spinning up server on 127.0.0.1:0 ---\n");

    server_args_t sa;
    memset(&sa, 0, sizeof(sa));
    sa.port = 0;             /* OS picks the port. */
    sa.max_iters = 2;        /* Two cycles: one OK, one ERR. */

    pthread_t tid;
    int prc = pthread_create(&tid, NULL, server_thread, &sa);
    CHECK(prc == 0, "pthread_create rc=%d", prc);

    /* Wait for the server to bind() (the OS-chosen port becomes
     * visible in `sa.bind_port` before the server enters accept()). */
    for (int i = 0; i < 200 && sa.bind_port == 0; i++) {
        struct timespec ts = { 0, 5 * 1000 * 1000 }; /* 5 ms */
        nanosleep(&ts, NULL);
    }
    uint16_t port = sa.bind_port;
    CHECK(port != 0, "server bound to port %u", port);
    if (port == 0) {
        pthread_join(tid, NULL);
        return 1;
    }

    fprintf(stdout, "\n--- submit Bell circuit ---\n");
    char *text = serialize_bell();
    CHECK(text != NULL, "serialize_bell -> non-NULL");

    double *probs = NULL;
    size_t  num   = 0;
    int rc = moonlab_control_submit_circuit("127.0.0.1", port, text, 0,
                                            &probs, &num);
    CHECK(rc == MOONLAB_CONTROL_OK, "submit_circuit rc=%d", rc);
    CHECK(num == 4, "got 4 probabilities (2 qubits) -- num=%zu", num);
    if (probs && num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f (expected 0.5)", probs[0]);
        CHECK(fabs(probs[1])       < 1e-9, "P[01] = %.6f (expected 0.0)", probs[1]);
        CHECK(fabs(probs[2])       < 1e-9, "P[10] = %.6f (expected 0.0)", probs[2]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f (expected 0.5)", probs[3]);
    }
    free(probs);
    free(text);

    fprintf(stdout, "\n--- submit garbage circuit (expect ERR) ---\n");
    const char *garbage = "this is not a valid moonlab-circuit v1\n";
    double *garbage_probs = NULL;
    size_t  garbage_num   = 0;
    rc = moonlab_control_submit_circuit("127.0.0.1", port, garbage, 0,
                                        &garbage_probs, &garbage_num);
    CHECK(rc == MOONLAB_CONTROL_REJECTED,
          "submit_circuit on garbage -> REJECTED (rc=%d)", rc);
    CHECK(garbage_probs == NULL,
          "garbage path leaves out_probs NULL");
    CHECK(garbage_num == 0,
          "garbage path leaves out_num zero");

    pthread_join(tid, NULL);
    CHECK(sa.rc == MOONLAB_CONTROL_OK, "server thread exit rc=%d", sa.rc);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
