/**
 * @file  test_control_plane_auth.c
 * @brief HMAC-SHA3-256 authentication tests for the v0.8.15 control plane.
 *
 * Four paths:
 *   1. Server with secret, client with correct secret  -> OK Bell.
 *   2. Server with secret, client without secret       -> REJECTED.
 *   3. Server with secret, client with WRONG secret    -> REJECTED.
 *   4. Server without secret, client with secret       -> OK (graceful).
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
    fprintf(stdout, "=== test_control_plane_auth (v0.8.15) ===\n\n");

    const uint8_t correct_secret[] = "moonlab-shared-2026";
    const uint8_t wrong_secret[]   = "wrong-secret-xyz";
    char *text = serialize_bell();

    /* ---- Path 1: matching secret -> OK ---- */
    fprintf(stdout, "--- path 1: matching secret -> OK Bell ---\n");
    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_set_secret(server, correct_secret,
                                           sizeof(correct_secret) - 1);
    CHECK(rc == 0, "set_secret rc=%d", rc);

    run_args_t ra = { server, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);

    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit_auth(
        "127.0.0.1", port,
        correct_secret, sizeof(correct_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4, "OK Bell with matching secret (rc=%d, num=%zu)", rc, num);
    if (num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f", probs[0]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f", probs[3]);
    }
    free(probs);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 2: client with NO secret against authed server ---- */
    fprintf(stdout, "\n--- path 2: missing AUTH -> REJECTED ---\n");
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    moonlab_control_server_set_secret(server, correct_secret,
                                      sizeof(correct_secret) - 1);
    ra.server = server; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);

    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit("127.0.0.1", port, text, 0, &probs, &num);
    CHECK(rc == MOONLAB_CONTROL_REJECTED, "unauth client -> REJECTED (rc=%d)", rc);
    CHECK(probs == NULL, "no probs returned");

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 3: client with WRONG secret ---- */
    fprintf(stdout, "\n--- path 3: wrong secret -> REJECTED ---\n");
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    moonlab_control_server_set_secret(server, correct_secret,
                                      sizeof(correct_secret) - 1);
    ra.server = server; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);

    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth(
        "127.0.0.1", port,
        wrong_secret, sizeof(wrong_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == MOONLAB_CONTROL_REJECTED, "wrong-secret -> REJECTED (rc=%d)", rc);
    CHECK(probs == NULL, "no probs returned for wrong secret");

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 4: unauthenticated server, AUTH-carrying client (graceful) ---- */
    fprintf(stdout, "\n--- path 4: unauthed server tolerates client AUTH ---\n");
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    /* no set_secret(); server is unauthenticated */
    ra.server = server; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);

    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth(
        "127.0.0.1", port,
        correct_secret, sizeof(correct_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4, "unauthed server still accepts AUTH-carrying client (rc=%d)", rc);
    free(probs);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 5: tenant-form AUTH (v1.0.3) ---- */
    fprintf(stdout, "\n--- path 5: AUTH tenant_id:hex form -> OK Bell ---\n");
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    moonlab_control_server_set_secret(server, correct_secret,
                                      sizeof(correct_secret) - 1);
    ra.server = server; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);

    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth_tenant(
        "127.0.0.1", port,
        "acme-corp",
        correct_secret, sizeof(correct_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4,
          "tenant-form AUTH accepted (rc=%d, num=%zu)", rc, num);
    free(probs);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 6: tenant_id with illegal chars rejected client-side ---- */
    fprintf(stdout, "\n--- path 6: bad tenant_id rejected before send ---\n");
    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth_tenant(
        "127.0.0.1", 1,            /* port doesn't matter, won't connect */
        "acme;rm -rf /",           /* shell-metachar -> client rejects */
        correct_secret, sizeof(correct_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == MOONLAB_CONTROL_BAD_ARG,
          "illegal-char tenant_id rejected client-side (rc=%d)", rc);
    CHECK(probs == NULL, "no probs returned for rejected tenant_id");

    free(text);
    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
