/**
 * @file  test_control_plane_shots.c
 * @brief Shots-mode test for the v0.8.11 control plane.
 *
 * Submits a Bell circuit with N=2048 shots, verifies that outcomes
 * are only {0b00, 0b11} (the GHZ-pair signature), with roughly 50/50
 * split.  Confirms the SHOTS / SAMPLES wire path end-to-end.
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

typedef struct { volatile uint16_t bind_port; int rc; } server_args_t;

static void *server_thread(void *arg)
{
    server_args_t *sa = (server_args_t *)arg;
    sa->rc = moonlab_control_serve(
        "127.0.0.1", 0, 2, (uint16_t *)&sa->bind_port);
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
    fprintf(stdout, "=== test_control_plane_shots (v0.8.11) ===\n\n");

    server_args_t sa = {0};
    pthread_t srv_tid;
    pthread_create(&srv_tid, NULL, server_thread, &sa);
    for (int i = 0; i < 200 && sa.bind_port == 0; i++) {
        struct timespec ts = { 0, 5 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    uint16_t port = sa.bind_port;
    CHECK(port != 0, "server bound to port %u", port);

    char *text = serialize_bell();
    const int N_SHOTS = 2048;

    fprintf(stdout, "\n--- shots-mode: 2048 Bell samples ---\n");
    uint64_t *outcomes = NULL;
    size_t    nout     = 0;
    int rc = moonlab_control_submit_circuit_shots(
        "127.0.0.1", port, text, 0, N_SHOTS, &outcomes, &nout);
    CHECK(rc == MOONLAB_CONTROL_OK, "submit_circuit_shots rc=%d", rc);
    CHECK(nout == (size_t)N_SHOTS, "got %zu shots", nout);

    int count_00 = 0, count_11 = 0, count_other = 0;
    for (size_t i = 0; i < nout; i++) {
        if      (outcomes[i] == 0u) count_00++;
        else if (outcomes[i] == 3u) count_11++;
        else                        count_other++;
    }
    fprintf(stdout, "    |00>: %d   |11>: %d   other: %d\n",
            count_00, count_11, count_other);
    CHECK(count_other == 0,
          "no off-Bell outcomes (got %d)", count_other);
    /* 50/50 with N=2048 -> std dev ~22.6 counts; 3 sigma window ~68. */
    const int diff = count_00 - count_11;
    CHECK(diff > -120 && diff < 120,
          "Bell split |00>-|11| = %d within 3-sigma", diff);
    free(outcomes);

    fprintf(stdout, "\n--- shots-mode: reject num_shots = 0 ---\n");
    outcomes = NULL; nout = 0;
    rc = moonlab_control_submit_circuit_shots(
        "127.0.0.1", port, text, 0, 0, &outcomes, &nout);
    CHECK(rc == MOONLAB_CONTROL_BAD_ARG,
          "client rejects num_shots=0 locally (rc=%d)", rc);

    /* Drive a real submission so the server max_iters=2 budget is met. */
    fprintf(stdout, "\n--- shots-mode: 2nd request to drain server ---\n");
    rc = moonlab_control_submit_circuit_shots(
        "127.0.0.1", port, text, 0, 16, &outcomes, &nout);
    CHECK(rc == MOONLAB_CONTROL_OK, "second submission rc=%d", rc);
    CHECK(nout == 16, "second submission got %zu shots", nout);
    free(outcomes);

    free(text);
    pthread_join(srv_tid, NULL);
    CHECK(sa.rc == MOONLAB_CONTROL_OK, "server thread exit rc=%d", sa.rc);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
