/**
 * @file  test_control_plane_concurrent.c
 * @brief Stress test for the v0.8.10 thread-pool control plane.
 *
 * Spins up the server with max_iters=8, fires 8 client pthreads in
 * parallel, each submitting a Bell circuit and verifying the
 * probability vector.  All 8 must succeed concurrently.
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

#define N_CLIENTS 8

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
    volatile uint16_t bind_port;
    int               rc;
} server_args_t;

static void *server_thread(void *arg)
{
    server_args_t *sa = (server_args_t *)arg;
    sa->rc = moonlab_control_serve(
        "127.0.0.1", 0, N_CLIENTS, (uint16_t *)&sa->bind_port);
    return NULL;
}

typedef struct {
    int      id;
    uint16_t port;
    int      ok;
} client_args_t;

static void *client_thread(void *arg)
{
    client_args_t *ca = (client_args_t *)arg;

    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    if (!c) return NULL;
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *text = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, text, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);

    double *probs = NULL;
    size_t  num   = 0;
    int rc = moonlab_control_submit_circuit("127.0.0.1", ca->port, text, 0,
                                            &probs, &num);
    free(text);

    if (rc == MOONLAB_CONTROL_OK && num == 4 &&
        fabs(probs[0] - 0.5) < 1e-9 &&
        fabs(probs[1])       < 1e-9 &&
        fabs(probs[2])       < 1e-9 &&
        fabs(probs[3] - 0.5) < 1e-9) {
        ca->ok = 1;
    }
    free(probs);
    return NULL;
}

int main(void)
{
    fprintf(stdout, "=== test_control_plane_concurrent (v0.8.10) ===\n\n");
    fprintf(stdout, "--- spinning up threaded server for %d concurrent clients ---\n",
            N_CLIENTS);

    server_args_t sa;
    memset(&sa, 0, sizeof(sa));

    pthread_t srv_tid;
    int prc = pthread_create(&srv_tid, NULL, server_thread, &sa);
    CHECK(prc == 0, "server pthread_create rc=%d", prc);

    for (int i = 0; i < 200 && sa.bind_port == 0; i++) {
        struct timespec ts = { 0, 5 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    uint16_t port = sa.bind_port;
    CHECK(port != 0, "server bound to port %u", port);
    if (port == 0) {
        pthread_join(srv_tid, NULL);
        return 1;
    }

    fprintf(stdout, "\n--- launching %d parallel client pthreads ---\n", N_CLIENTS);
    pthread_t clients[N_CLIENTS];
    client_args_t cargs[N_CLIENTS];

    const clock_t t0 = clock();
    for (int i = 0; i < N_CLIENTS; i++) {
        cargs[i].id   = i;
        cargs[i].port = port;
        cargs[i].ok   = 0;
        pthread_create(&clients[i], NULL, client_thread, &cargs[i]);
    }
    for (int i = 0; i < N_CLIENTS; i++) {
        pthread_join(clients[i], NULL);
    }
    const clock_t t1 = clock();
    const double total_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    int total_ok = 0;
    for (int i = 0; i < N_CLIENTS; i++) {
        if (cargs[i].ok) total_ok++;
    }
    CHECK(total_ok == N_CLIENTS,
          "all %d clients succeeded (got %d)", N_CLIENTS, total_ok);
    fprintf(stdout, "    wall time: %.4f s for %d concurrent submissions\n",
            total_s, N_CLIENTS);

    pthread_join(srv_tid, NULL);
    CHECK(sa.rc == MOONLAB_CONTROL_OK, "server thread exit rc=%d", sa.rc);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
