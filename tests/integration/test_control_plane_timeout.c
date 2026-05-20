/**
 * @file  test_control_plane_timeout.c
 * @brief Per-request socket timeout test for the v0.8.26 control plane.
 *
 * Path 1 -- silent client: open a connection, send nothing, the server
 *           must close the fd within ~timeout seconds (we use 1).
 * Path 2 -- a normal Bell circuit still succeeds inside the timeout.
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
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

static double monotonic_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void)
{
    fprintf(stdout, "=== test_control_plane_timeout (v0.8.26) ===\n\n");

    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_set_request_timeout(server, 1);
    CHECK(rc == 0, "set_request_timeout(1) rc=%d", rc);

    /* 2 cycles: 1 silent connect + 1 Bell. */
    ra_t ra = { server, 2, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    /* ---- Path 1: silent client ---- */
    fprintf(stdout, "--- path 1: silent client times out ~1s ---\n");
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    CHECK(connect(sfd, (struct sockaddr *)&addr, sizeof(addr)) == 0,
          "connect to silent-test port");

    const double t0 = monotonic_s();
    /* Block on recv() -- the server should close the connection after
     * its SO_RCVTIMEO fires; recv() then returns 0. */
    char buf[16];
    ssize_t got = recv(sfd, buf, sizeof(buf), 0);
    const double dt = monotonic_s() - t0;
    fprintf(stdout, "    recv() returned %zd after %.3f s\n", got, dt);
    CHECK(got <= 0, "server closed silent connection");
    CHECK(dt < 3.0, "closed within 3s (got %.3f)", dt);
    CHECK(dt > 0.5, "took >=0.5s to close (got %.3f)", dt);
    close(sfd);

    /* ---- Path 2: real Bell circuit still works ---- */
    fprintf(stdout, "\n--- path 2: normal Bell still works inside timeout ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *text = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, text, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);

    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit("127.0.0.1", port, text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4, "Bell submit rc=%d num=%zu", rc, num);
    if (num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f", probs[0]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f", probs[3]);
    }
    free(probs);
    free(text);

    pthread_join(tid, NULL);
    moonlab_control_server_close(server);
    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
