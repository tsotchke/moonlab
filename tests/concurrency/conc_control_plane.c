/**
 * @file  conc_control_plane.c
 * @brief ThreadSanitizer harness for the Moonlab control plane.
 *
 * Drives one moonlab_control_server on 127.0.0.1:0 with many concurrent
 * client threads mixing HEALTH / METRICS / CIRCUIT / SHOTS / AUTH+tenant
 * requests, exercising:
 *   - the process-wide atomic metric counters,
 *   - the stack-local Prometheus metrics text buffer,
 *   - the per-IP token bucket (rl_lock),
 *   - the admission hook + completion hook fan-out,
 *   - the bounded live-worker registry reap/drain path (v1.1.0).
 *
 * Two modes (argv[1]):
 *   "steady"      (default) -- server config is set ONCE before run().
 *                 This is the documented, supported usage; a clean TSan
 *                 pass here means the request path itself is race-free.
 *   "adversarial" -- a background toggler thread mutates rate-limit /
 *                 max-concurrent / admission-hook / request-timeout WHILE
 *                 the server is serving.  Surfaces the unsynchronised
 *                 config-field reads in the accept loop.  set_admission_hook
 *                 is documented "Thread-safe" so any race there is a true
 *                 bug, not misuse.
 *
 * Exit code 0 iff every client transaction succeeded (steady mode).  TSan
 * aborts the process on a race when run with halt_on_error=1; run_tsan.sh
 * parses the report either way.
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef N_CLIENTS
#define N_CLIENTS 12
#endif
#ifndef REQ_PER_CLIENT
#define REQ_PER_CLIENT 24
#endif

static moonlab_control_server_t *g_server = NULL;
static uint16_t                  g_port    = 0;
static _Atomic int               g_client_failures = 0;
static _Atomic int               g_admission_calls = 0;
static _Atomic int               g_run_toggler     = 1;

/* Build a Bell-pair circuit text (heap; caller frees). */
static char *make_bell_text(void)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    if (!c) return NULL;
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *text = (char *)malloc(needed + 1);
    if (text) moonlab_qgtl_circuit_serialize(c, text, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);
    return text;
}

/* Thread-safe admission hook: increments an atomic, refuses one tenant so
 * the refuse/counter path is exercised concurrently. */
static int admission_hook(const char *tenant_id, const char *verb,
                          int num_qubits, int num_shots, void *ctx)
{
    (void)verb; (void)num_qubits; (void)num_shots; (void)ctx;
    atomic_fetch_add(&g_admission_calls, 1);
    if (tenant_id && strcmp(tenant_id, "blocked") == 0) {
        return MOONLAB_CONTROL_REJECTED;
    }
    return 0;
}

static void *server_thread(void *arg)
{
    (void)arg;
    /* Large cap; we stop it explicitly via shutdown() once clients drain. */
    (void)moonlab_control_server_run(g_server, 1 << 30);
    return NULL;
}

/* Adversarial: mutate server config while it serves. */
static void *toggler_thread(void *arg)
{
    (void)arg;
    int i = 0;
    while (atomic_load(&g_run_toggler)) {
        moonlab_control_server_set_rate_limit(g_server, (i & 3) ? 100000 : 0, 100000);
        moonlab_control_server_set_max_concurrent(g_server, (i & 1) ? 0 : 64);
        moonlab_control_server_set_request_timeout(g_server, (i & 1) ? 0 : 5);
        moonlab_control_server_set_admission_hook(
            g_server, (i & 1) ? admission_hook : NULL, NULL);
        i++;
        struct timespec ts = { 0, 200 * 1000 }; /* 200 us */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

typedef struct { int id; } client_args_t;

static void *client_thread(void *arg)
{
    client_args_t *ca = (client_args_t *)arg;
    char *text = make_bell_text();
    if (!text) { atomic_fetch_add(&g_client_failures, 1); return NULL; }

    for (int k = 0; k < REQ_PER_CLIENT; k++) {
        const int pick = (ca->id + k) % 5;
        int rc = MOONLAB_CONTROL_OK;
        if (pick == 0) {
            rc = moonlab_control_submit_health("127.0.0.1", g_port);
        } else if (pick == 1) {
            char *mtext = NULL;
            rc = moonlab_control_submit_metrics("127.0.0.1", g_port, &mtext);
            free(mtext);
        } else if (pick == 2) {
            double *probs = NULL; size_t num = 0;
            rc = moonlab_control_submit_circuit("127.0.0.1", g_port,
                                                text, 0, &probs, &num);
            free(probs);
        } else if (pick == 3) {
            uint64_t *out = NULL; size_t num = 0;
            rc = moonlab_control_submit_circuit_shots("127.0.0.1", g_port,
                                                      text, 0, 64, &out, &num);
            free(out);
        } else {
            /* AUTH + tenant path (server has no secret configured, so the
             * token is accepted gracefully; tenant is plumbed through). */
            double *probs = NULL; size_t num = 0;
            const char *tenant = (k % 7 == 0) ? "blocked" : "tenant_a";
            rc = moonlab_control_submit_circuit_auth_tenant(
                "127.0.0.1", g_port, tenant, NULL, 0, text, 0, &probs, &num);
            free(probs);
        }
        /* Only a hard transport failure (negative IO) is tallied; policy
         * refusals (rate-limit / admission) are legitimate outcomes. */
        if (rc == MOONLAB_CONTROL_IO_ERROR) {
            atomic_fetch_add(&g_client_failures, 1);
        }
    }
    free(text);
    return NULL;
}

int main(int argc, char **argv)
{
    const int adversarial = (argc > 1 && strcmp(argv[1], "adversarial") == 0);
    fprintf(stdout, "=== conc_control_plane (%s) ===\n",
            adversarial ? "adversarial" : "steady");

    /* Warm up the shared-core lazy singletons (config.c g_config,
     * simd_ops.c g_simd_vtable) single-threaded so their unsynchronised
     * first-touch init -- a SEPARATE finding, see conc_core_init -- does
     * not mask control-plane-specific races.  Executing one circuit here
     * resolves those globals before any server thread runs. */
    {
        moonlab_qgtl_circuit_t *w = moonlab_qgtl_circuit_create(2);
        if (w) {
            moonlab_qgtl_add_gate(w, MOONLAB_QGTL_GATE_H, 0, 0, NULL);
            moonlab_qgtl_add_gate(w, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
            moonlab_qgtl_exec_options_t o; memset(&o, 0, sizeof(o));
            o.return_probabilities = 1;
            moonlab_qgtl_results_t r; memset(&r, 0, sizeof(r));
            moonlab_qgtl_execute(w, &o, &r);
            moonlab_qgtl_results_free(&r);
            moonlab_qgtl_circuit_free(w);
        }
    }

    if (moonlab_control_server_open("127.0.0.1", 0, &g_server, &g_port)
        != MOONLAB_CONTROL_OK) {
        fprintf(stderr, "server open failed\n");
        return 2;
    }

    /* Steady config: set ONCE before run(). */
    moonlab_control_server_set_rate_limit(g_server, 1000000, 1000000);
    moonlab_control_server_set_max_concurrent(g_server, 64);
    moonlab_control_server_set_admission_hook(g_server, admission_hook, NULL);

    pthread_t srv;
    pthread_create(&srv, NULL, server_thread, NULL);

    /* One synchronous request warms handle_one_request's env-cached log
     * flag (control_plane.c log_enabled() static) on a single worker before
     * the concurrent storm, so its lazy-init race does not mask others. */
    (void)moonlab_control_submit_health("127.0.0.1", g_port);

    pthread_t tog;
    int have_tog = 0;
    if (adversarial) {
        have_tog = (pthread_create(&tog, NULL, toggler_thread, NULL) == 0);
    }

    pthread_t clients[N_CLIENTS];
    client_args_t cargs[N_CLIENTS];
    for (int i = 0; i < N_CLIENTS; i++) {
        cargs[i].id = i;
        pthread_create(&clients[i], NULL, client_thread, &cargs[i]);
    }
    for (int i = 0; i < N_CLIENTS; i++) pthread_join(clients[i], NULL);

    if (have_tog) {
        atomic_store(&g_run_toggler, 0);
        pthread_join(tog, NULL);
    }

    moonlab_control_server_shutdown(g_server);
    pthread_join(srv, NULL);
    moonlab_control_server_close(g_server);

    const int fails = atomic_load(&g_client_failures);
    fprintf(stdout, "admission_calls=%d client_io_failures=%d\n",
            atomic_load(&g_admission_calls), fails);
    /* In adversarial mode transient IO failures are expected (rate-limit
     * flapping to 0-tokens etc.); only steady mode enforces zero. */
    return (!adversarial && fails > 0) ? 1 : 0;
}
