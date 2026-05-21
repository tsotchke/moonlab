/**
 * @file  test_control_plane_tenant.c
 * @brief End-to-end smoke for the v1.0.3 tenant-identity pipeline.
 *
 * Proves the whole loop works:
 *
 *   client submits   AUTH <tenant>:<hmac>\n CIRCUIT <n>\n<body>
 *        |
 *        v
 *   control plane parses tenant_id, validates HMAC
 *        |
 *        v
 *   control plane sets scheduler thread-local request context
 *        |
 *        v
 *   scheduler runs the job, fires the completion hook
 *        |
 *        v
 *   hook reads moonlab_scheduler_current_tenant_id() and records it
 *        |
 *        v
 *   test asserts the captured tenant_id matches what the client sent.
 *
 * Without this end-to-end test the multi-tenant pipeline could
 * silently lose the tenant_id at any layer; the smoke is the
 * canonical premium-tier demonstration that a customer-submitted
 * job actually reaches a billing/audit overlay attributed to the
 * right account.
 */

#include "../../src/control/control_plane.h"
#include "../../src/distributed/scheduler.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <pthread.h>
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

/* Captures every hook fire so we can replay the sequence in the
 * asserts.  64 fires is well past anything this test generates. */
typedef struct {
    int  n_fires;
    char tenants[64][64];
    int  shot_counts[64];
} hook_record_t;

static void recording_hook(const moonlab_job_t          *job,
                           const moonlab_job_results_t  *out,
                           const char                   *backend_name,
                           void                         *ctx)
{
    (void)job; (void)backend_name;
    hook_record_t *r = (hook_record_t *)ctx;
    if (r->n_fires >= (int)(sizeof(r->tenants) / sizeof(r->tenants[0]))) return;
    const char *tid = moonlab_scheduler_current_tenant_id();
    snprintf(r->tenants[r->n_fires], sizeof(r->tenants[0]),
             "%s", tid ? tid : "");
    r->shot_counts[r->n_fires] = out ? out->total_shots : 0;
    r->n_fires++;
}

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
    fprintf(stdout, "=== test_control_plane_tenant: end-to-end smoke ===\n\n");

    const uint8_t shared_secret[] = "moonlab-shared-2026";
    char *text = serialize_bell();
    hook_record_t hook = {0};
    moonlab_scheduler_set_completion_hook(recording_hook, &hook);

    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_set_secret(server, shared_secret,
                                           sizeof(shared_secret) - 1);
    CHECK(rc == 0, "set_secret rc=%d", rc);

    run_args_t ra = { server, 0 };
    pthread_t  tid;
    pthread_create(&tid, NULL, run_thread, &ra);

    /* ---- Submit as acme-corp ---- */
    fprintf(stdout, "--- submit as acme-corp ---\n");
    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit_auth_tenant(
        "127.0.0.1", port,
        "acme-corp",
        shared_secret, sizeof(shared_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4,
          "submit_circuit_auth_tenant(acme-corp) rc=%d num=%zu", rc, num);
    free(probs);

    /* ---- Submit as beta-startup ---- */
    fprintf(stdout, "\n--- submit as beta-startup ---\n");
    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth_tenant(
        "127.0.0.1", port,
        "beta-startup",
        shared_secret, sizeof(shared_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4,
          "submit_circuit_auth_tenant(beta-startup) rc=%d num=%zu", rc, num);
    free(probs);

    /* ---- Submit with no tenant (legacy AUTH form) ---- */
    fprintf(stdout, "\n--- submit with no tenant (legacy form) ---\n");
    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_auth(
        "127.0.0.1", port,
        shared_secret, sizeof(shared_secret) - 1,
        text, 0, &probs, &num);
    CHECK(rc == 0 && num == 4,
          "submit_circuit_auth (legacy) rc=%d num=%zu", rc, num);
    free(probs);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Assert the hook saw the right tenants in the right order ---- */
    fprintf(stdout, "\n--- hook observations ---\n");
    CHECK(hook.n_fires == 3, "hook fired 3 times (got %d)", hook.n_fires);
    if (hook.n_fires >= 3) {
        CHECK(strcmp(hook.tenants[0], "acme-corp") == 0,
              "fire #0 tenant = %s (expected acme-corp)", hook.tenants[0]);
        CHECK(strcmp(hook.tenants[1], "beta-startup") == 0,
              "fire #1 tenant = %s (expected beta-startup)", hook.tenants[1]);
        CHECK(hook.tenants[2][0] == '\0',
              "fire #2 tenant = empty for legacy AUTH (got %s)",
              hook.tenants[2]);
    }

    moonlab_scheduler_set_completion_hook(NULL, NULL);
    free(text);
    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
