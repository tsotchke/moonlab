/**
 * @file  control_plane_loadtest.c
 * @brief Concurrent load test against a running moonlab-control-server.
 *
 * Spawns N worker threads, each submitting CIRCUIT or SHOTS jobs
 * against the server for D seconds; reports total throughput plus
 * P50 / P90 / P99 wall-clock latency.  Validates the per-replica
 * capacity numbers in docs/operations/RUNBOOK.md and the fleet
 * arithmetic in docs/operations/FLEET_DEPLOYMENT.md.
 *
 * Usage:
 *   control_plane_loadtest [flags]
 *     --host HOST              target server (default 127.0.0.1)
 *     --port PORT              target port   (default 17070)
 *     --tenant TENANT          AUTH <tenant>:<hmac> if --secret set
 *     --secret HEX             HMAC shared secret (raw bytes, not hex)
 *     --workers N              concurrent clients (default 8)
 *     --duration SECS          run length (default 5)
 *     --qubits N               circuit size (default 2 -- Bell)
 *     --shots N                shots per job; 0 = probabilities mode (default)
 *     --quiet                  emit one summary line, suitable for CI
 *
 * Output (default):
 *     control_plane_loadtest: 8 workers x 5s on 127.0.0.1:17070 Bell-2q
 *       requests       4923
 *       throughput     984.6 req/sec
 *       latency P50    8.12 ms
 *       latency P90    14.30 ms
 *       latency P99    22.65 ms
 *       errors         0
 *
 * Quiet output:
 *     LOADTEST workers=8 duration=5.00 qubits=2 reqs=4923 rps=984.6 \
 *              p50_ms=8.12 p90_ms=14.30 p99_ms=22.65 errors=0
 */

#include "../src/control/control_plane.h"
#include "../src/applications/moonlab_qgtl_backend.h"

#include <getopt.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- shared state ---------- */

typedef struct {
    const char *host;
    int         port;
    const char *tenant;
    const char *secret;
    size_t      secret_len;
    int         qubits;
    int         shots;          /* 0 = probabilities mode */
    double      deadline_sec;   /* CLOCK_MONOTONIC wall-clock cutoff */
} worker_cfg_t;

typedef struct {
    /* Per-worker latency histogram: round to microseconds, cap at
     * 60-second bins.  60M buckets at 4B each = 240 MB; we use
     * coarser binning to stay reasonable. */
    int      n_samples;
    int      cap;
    double  *samples_ms;     /* malloc'd; grows */
    int      n_errors;
} worker_stats_t;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static void push_sample(worker_stats_t *s, double ms)
{
    if (s->n_samples >= s->cap) {
        int new_cap = s->cap > 0 ? s->cap * 2 : 1024;
        s->samples_ms = realloc(s->samples_ms, (size_t)new_cap * sizeof(double));
        s->cap = new_cap;
    }
    s->samples_ms[s->n_samples++] = ms;
}

static char *serialize_circuit(int n_qubits)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(n_qubits);
    /* Bell-style entangler: H on q0, CNOT chain through the rest. */
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, 0, NULL);
    for (int q = 1; q < n_qubits; q++) {
        moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, q, q - 1, NULL);
    }
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *buf = malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, buf, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);
    return buf;
}

/* ---------- worker ---------- */

typedef struct {
    const worker_cfg_t *cfg;
    worker_stats_t     *stats;
    const char         *circuit_body;
} worker_arg_t;

static void *worker_fn(void *vp)
{
    worker_arg_t *w = vp;
    const worker_cfg_t *cfg = w->cfg;
    const uint8_t *secret = (const uint8_t *)cfg->secret;
    while (now_sec() < cfg->deadline_sec) {
        double t0 = now_sec();
        int rc;
        if (cfg->shots > 0) {
            uint64_t *outcomes = NULL; size_t num = 0;
            /* No tenant variant for SHOTS yet; falls back to legacy. */
            rc = moonlab_control_submit_circuit_shots(
                cfg->host, (uint16_t)cfg->port,
                w->circuit_body, 0,
                cfg->shots, &outcomes, &num);
            free(outcomes);
        } else {
            double *probs = NULL; size_t num = 0;
            if (cfg->secret_len > 0 && cfg->tenant) {
                rc = moonlab_control_submit_circuit_auth_tenant(
                    cfg->host, (uint16_t)cfg->port,
                    cfg->tenant, secret, cfg->secret_len,
                    w->circuit_body, 0, &probs, &num);
            } else if (cfg->secret_len > 0) {
                rc = moonlab_control_submit_circuit_auth(
                    cfg->host, (uint16_t)cfg->port,
                    secret, cfg->secret_len,
                    w->circuit_body, 0, &probs, &num);
            } else {
                rc = moonlab_control_submit_circuit(
                    cfg->host, (uint16_t)cfg->port,
                    w->circuit_body, 0, &probs, &num);
            }
            free(probs);
        }
        double t1 = now_sec();
        if (rc == 0) push_sample(w->stats, (t1 - t0) * 1000.0);
        else         w->stats->n_errors++;
    }
    return NULL;
}

/* ---------- percentile ---------- */

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double percentile(double *sorted, int n, double p)
{
    if (n == 0) return 0.0;
    int idx = (int)((double)(n - 1) * p);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    worker_cfg_t cfg = {
        .host = "127.0.0.1", .port = 17070,
        .tenant = NULL, .secret = NULL, .secret_len = 0,
        .qubits = 2, .shots = 0, .deadline_sec = 0.0,
    };
    int workers = 8;
    int duration = 5;
    int quiet = 0;

    static struct option longopts[] = {
        {"host",     required_argument, 0, 'H'},
        {"port",     required_argument, 0, 'P'},
        {"tenant",   required_argument, 0, 'T'},
        {"secret",   required_argument, 0, 'S'},
        {"workers",  required_argument, 0, 'w'},
        {"duration", required_argument, 0, 'd'},
        {"qubits",   required_argument, 0, 'q'},
        {"shots",    required_argument, 0, 's'},
        {"quiet",    no_argument,       0, 'Q'},
        {0,0,0,0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "H:P:T:S:w:d:q:s:Q", longopts, NULL)) != -1) {
        switch (opt) {
        case 'H': cfg.host    = optarg; break;
        case 'P': cfg.port    = atoi(optarg); break;
        case 'T': cfg.tenant  = optarg; break;
        case 'S': cfg.secret  = optarg; cfg.secret_len = strlen(optarg); break;
        case 'w': workers     = atoi(optarg); break;
        case 'd': duration    = atoi(optarg); break;
        case 'q': cfg.qubits  = atoi(optarg); break;
        case 's': cfg.shots   = atoi(optarg); break;
        case 'Q': quiet       = 1; break;
        default: fprintf(stderr, "see source for flags\n"); return 2;
        }
    }
    if (workers < 1 || workers > 4096) {
        fprintf(stderr, "workers out of range\n"); return 2;
    }
    if (duration < 1 || duration > 3600) {
        fprintf(stderr, "duration out of range\n"); return 2;
    }

    char *circuit = serialize_circuit(cfg.qubits);
    cfg.deadline_sec = now_sec() + (double)duration;

    pthread_t      *tids  = calloc((size_t)workers, sizeof(pthread_t));
    worker_arg_t   *wargs = calloc((size_t)workers, sizeof(worker_arg_t));
    worker_stats_t *stats = calloc((size_t)workers, sizeof(worker_stats_t));
    double t_start = now_sec();
    for (int i = 0; i < workers; i++) {
        wargs[i].cfg          = &cfg;
        wargs[i].stats        = &stats[i];
        wargs[i].circuit_body = circuit;
        pthread_create(&tids[i], NULL, worker_fn, &wargs[i]);
    }
    for (int i = 0; i < workers; i++) pthread_join(tids[i], NULL);
    double t_end = now_sec();
    free(circuit);

    /* Merge all latency samples for fleet-wide percentiles. */
    int total = 0, errors = 0;
    for (int i = 0; i < workers; i++) { total += stats[i].n_samples; errors += stats[i].n_errors; }
    double *all = malloc((size_t)total * sizeof(double));
    int k = 0;
    for (int i = 0; i < workers; i++) {
        memcpy(all + k, stats[i].samples_ms,
               (size_t)stats[i].n_samples * sizeof(double));
        k += stats[i].n_samples;
        free(stats[i].samples_ms);
    }
    free(stats); free(wargs); free(tids);
    qsort(all, (size_t)total, sizeof(double), cmp_double);

    const double elapsed = t_end - t_start;
    const double rps = (double)total / elapsed;
    const double p50 = percentile(all, total, 0.50);
    const double p90 = percentile(all, total, 0.90);
    const double p99 = percentile(all, total, 0.99);
    free(all);

    if (quiet) {
        printf("LOADTEST workers=%d duration=%.2f qubits=%d "
               "reqs=%d rps=%.1f p50_ms=%.2f p90_ms=%.2f p99_ms=%.2f "
               "errors=%d\n",
               workers, elapsed, cfg.qubits, total, rps,
               p50, p90, p99, errors);
    } else {
        printf("control_plane_loadtest: %d workers x %ds on %s:%d %s-%dq\n",
               workers, duration, cfg.host, cfg.port,
               cfg.shots > 0 ? "shots" : "Bell", cfg.qubits);
        printf("  requests       %d\n", total);
        printf("  throughput     %.1f req/sec\n", rps);
        printf("  latency P50    %.2f ms\n", p50);
        printf("  latency P90    %.2f ms\n", p90);
        printf("  latency P99    %.2f ms\n", p99);
        printf("  errors         %d\n", errors);
    }
    return errors == 0 ? 0 : 1;
}
