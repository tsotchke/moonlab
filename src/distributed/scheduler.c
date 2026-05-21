/**
 * @file    scheduler.c
 * @brief   Distributed-execution scheduler MVP -- in-process workers.
 *
 * The job record duplicates the gate-record format from
 * `moonlab_qgtl_backend.c` so we don't have to expose the latter's
 * internals.  Each worker reconstructs a fresh `moonlab_qgtl_circuit_t`
 * from the recorded gates, executes its shot slice, and writes
 * outcomes into the shared output buffer at its offset.
 *
 * OpenMP is the transport in v0.7.0.  The contract -- gate list +
 * shot slice + per-worker outcome write -- is identical to what an
 * MPI / gRPC worker would do, so v0.7.1+ can swap the loop without
 * touching the API.
 */

#include "scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(HAS_MPI)
#include "mpi_bridge.h"
#endif

typedef struct {
    moonlab_qgtl_gate_t type;
    int     target;
    int     control;
    double  theta;
    int     has_param;
} sched_gate_record_t;

struct moonlab_job {
    int                  num_qubits;
    int                  num_shots;
    int                  num_workers;
    uint64_t             rng_seed;
    int                  num_gates;
    int                  cap;
    sched_gate_record_t *gates;
    /* Backend plug-in (since v1.1): NULL => default "simulator".
     * Heap-owned copy of the caller's string. */
    char                *backend_name;
};

moonlab_job_t *moonlab_job_create(int num_qubits)
{
    if (num_qubits < 1 || num_qubits > 32) return NULL;
    moonlab_job_t *j = (moonlab_job_t *)calloc(1, sizeof(*j));
    if (!j) return NULL;
    j->num_qubits  = num_qubits;
    j->num_workers = 1;
    return j;
}

void moonlab_job_free(moonlab_job_t *j)
{
    if (!j) return;
    free(j->gates);
    free(j->backend_name);
    free(j);
}

int moonlab_job_num_qubits(const moonlab_job_t *j) {
    return j ? j->num_qubits : MOONLAB_SCHED_BAD_ARG;
}
int moonlab_job_num_gates(const moonlab_job_t *j) {
    return j ? j->num_gates : MOONLAB_SCHED_BAD_ARG;
}
int moonlab_job_num_shots(const moonlab_job_t *j) {
    return j ? j->num_shots : MOONLAB_SCHED_BAD_ARG;
}
int moonlab_job_num_workers(const moonlab_job_t *j) {
    return j ? j->num_workers : MOONLAB_SCHED_BAD_ARG;
}

static int gate_takes_param_local(moonlab_qgtl_gate_t t)
{
    return t == MOONLAB_QGTL_GATE_RX ||
           t == MOONLAB_QGTL_GATE_RY ||
           t == MOONLAB_QGTL_GATE_RZ;
}

int moonlab_job_add_gate(moonlab_job_t *j,
                         moonlab_qgtl_gate_t type,
                         int target, int control,
                         const double *params)
{
    if (!j) return MOONLAB_SCHED_BAD_ARG;
    if (gate_takes_param_local(type) && params == NULL) return MOONLAB_SCHED_BAD_ARG;
    if (j->num_gates >= j->cap) {
        const int new_cap = j->cap ? j->cap * 2 : 16;
        sched_gate_record_t *n = (sched_gate_record_t *)
            realloc(j->gates, (size_t)new_cap * sizeof(*n));
        if (!n) return MOONLAB_SCHED_OOM;
        j->gates = n;
        j->cap   = new_cap;
    }
    sched_gate_record_t *g = &j->gates[j->num_gates++];
    g->type      = type;
    g->target    = target;
    g->control   = control;
    g->theta     = gate_takes_param_local(type) ? params[0] : 0.0;
    g->has_param = gate_takes_param_local(type) ? 1 : 0;
    return MOONLAB_SCHED_OK;
}

int moonlab_job_set_num_shots(moonlab_job_t *j, int n) {
    if (!j || n < 0) return MOONLAB_SCHED_BAD_ARG;
    j->num_shots = n;
    return MOONLAB_SCHED_OK;
}

int moonlab_job_set_num_workers(moonlab_job_t *j, int n) {
    if (!j || n < 1) return MOONLAB_SCHED_BAD_ARG;
    j->num_workers = n;
    return MOONLAB_SCHED_OK;
}

int moonlab_job_set_rng_seed(moonlab_job_t *j, uint64_t seed) {
    if (!j) return MOONLAB_SCHED_BAD_ARG;
    j->rng_seed = seed;
    return MOONLAB_SCHED_OK;
}

/* splitmix64 step -- worker derives its seed as
 * splitmix64(base XOR worker_id_hi). */
static inline uint64_t splitmix64(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d49bb133111ebbULL;
    return x ^ (x >> 31);
}

/* Build a fresh moonlab_qgtl_circuit from the job, run a shot
 * slice, write outcomes into [out_offset, out_offset+slice). */
static int run_worker_slice(const moonlab_job_t *j,
                             int slice_start, int slice_end,
                             int worker_id,
                             uint64_t *outcomes_out)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(j->num_qubits);
    if (!c) return MOONLAB_SCHED_OOM;
    for (int g = 0; g < j->num_gates; g++) {
        const sched_gate_record_t *gr = &j->gates[g];
        const double param = gr->theta;
        const int rc = moonlab_qgtl_add_gate(
            c, gr->type, gr->target, gr->control,
            gr->has_param ? &param : NULL);
        if (rc != 0) { moonlab_qgtl_circuit_free(c); return MOONLAB_SCHED_INTERNAL; }
    }
    const int slice_size = slice_end - slice_start;
    if (slice_size <= 0) {
        moonlab_qgtl_circuit_free(c);
        return MOONLAB_SCHED_OK;
    }
    const uint64_t seed = j->rng_seed
        ? splitmix64(j->rng_seed ^ ((uint64_t)worker_id << 32))
        : (uint64_t)(time(NULL) ^ ((uint64_t)worker_id << 32) ^ 0x9e3779b97f4a7c15ULL);
    const moonlab_qgtl_exec_options_t opts = {
        .num_shots = slice_size,
        .rng_seed  = seed,
        .return_probabilities = 0,
    };
    moonlab_qgtl_results_t res = {0};
    const int rc = moonlab_qgtl_execute(c, &opts, &res);
    moonlab_qgtl_circuit_free(c);
    if (rc != 0) return MOONLAB_SCHED_INTERNAL;
    /* Copy slice outcomes into the merged buffer. */
    memcpy(&outcomes_out[slice_start], res.outcomes,
           (size_t)slice_size * sizeof(uint64_t));
    moonlab_qgtl_results_free(&res);
    return MOONLAB_SCHED_OK;
}

/* Default in-process simulator backend.  Splits shots across OpenMP
 * workers; each worker reconstructs the QGTL circuit and runs its
 * slice via moonlab_qgtl_execute. */
static int simulator_backend_execute(const moonlab_job_t       *j,
                                     moonlab_job_results_t     *out,
                                     void                      *ctx)
{
    (void)ctx;
    if (!j || !out) return MOONLAB_SCHED_BAD_ARG;
    if (j->num_shots <= 0) return MOONLAB_SCHED_BAD_ARG;

    const int total_shots = j->num_shots;
    int num_workers = j->num_workers;
    if (num_workers > total_shots) num_workers = total_shots;
    if (num_workers < 1) num_workers = 1;

    out->num_qubits       = j->num_qubits;
    out->total_shots      = total_shots;
    out->num_workers_used = num_workers;
    out->outcomes = (uint64_t *)malloc((size_t)total_shots * sizeof(uint64_t));
    out->worker_seconds = (double *)calloc((size_t)num_workers, sizeof(double));
    if (!out->outcomes || !out->worker_seconds) {
        return MOONLAB_SCHED_OOM;
    }

    int rc_first_err = MOONLAB_SCHED_OK;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int w = 0; w < num_workers; w++) {
        const int start = (int)((long long)w * total_shots / num_workers);
        const int end   = (int)((long long)(w + 1) * total_shots / num_workers);

        const clock_t t0 = clock();
        const int rc = run_worker_slice(j, start, end, w, out->outcomes);
        const clock_t t1 = clock();
        out->worker_seconds[w] = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

        if (rc != MOONLAB_SCHED_OK) {
#if defined(_OPENMP)
            #pragma omp critical
#endif
            if (rc_first_err == MOONLAB_SCHED_OK) rc_first_err = rc;
        }
    }

    return rc_first_err;
}

/* ============================================================ */
/* Backend registry (since v1.1.0).                              */
/*                                                                */
/* Small static array; 16 slots is plenty for the foreseeable    */
/* set ("simulator" + ~3 vendor-noise emulators + a few user     */
/* slots).  Mutex-protected so register/unregister/find are      */
/* thread-safe.                                                   */
/* ============================================================ */

#include <pthread.h>

#define MOONLAB_SCHED_MAX_BACKENDS 16

static moonlab_backend_t g_backends[MOONLAB_SCHED_MAX_BACKENDS];
static int g_num_backends = 0;
static pthread_mutex_t g_backend_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_backend_init_once = PTHREAD_ONCE_INIT;

/* Completion hook (since v1.1.0).  Protected by the same lock as
 * the backend registry -- install/clear/fire are serialised so
 * the private overlay can swap hooks at runtime without racing the
 * scheduler. */
static moonlab_completion_hook_fn g_completion_hook = NULL;
static void                      *g_completion_hook_ctx = NULL;

static void register_default_simulator_backend(void)
{
    /* Direct-store; we hold no lock during the once init because
     * pthread_once serialises us against other registrars. */
    g_backends[g_num_backends].name        = "simulator";
    g_backends[g_num_backends].execute     = simulator_backend_execute;
    g_backends[g_num_backends].ctx         = NULL;
    g_backends[g_num_backends].description =
        "In-process moonlab simulator (OpenMP shot fan-out via moonlab_qgtl_execute).";
    g_num_backends = 1;
}

static void ensure_default_backend(void)
{
    pthread_once(&g_backend_init_once, register_default_simulator_backend);
}

int moonlab_register_backend(const moonlab_backend_t *backend)
{
    if (!backend || !backend->name || !backend->execute) {
        return MOONLAB_SCHED_BAD_ARG;
    }
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    /* Replace if name already exists. */
    for (int i = 0; i < g_num_backends; i++) {
        if (strcmp(g_backends[i].name, backend->name) == 0) {
            g_backends[i] = *backend;
            pthread_mutex_unlock(&g_backend_lock);
            return MOONLAB_SCHED_OK;
        }
    }
    if (g_num_backends >= MOONLAB_SCHED_MAX_BACKENDS) {
        pthread_mutex_unlock(&g_backend_lock);
        return MOONLAB_SCHED_OOM;
    }
    g_backends[g_num_backends++] = *backend;
    pthread_mutex_unlock(&g_backend_lock);
    return MOONLAB_SCHED_OK;
}

int moonlab_unregister_backend(const char *name)
{
    if (!name) return MOONLAB_SCHED_BAD_ARG;
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    for (int i = 0; i < g_num_backends; i++) {
        if (strcmp(g_backends[i].name, name) == 0) {
            /* Shift the tail down. */
            for (int j = i; j < g_num_backends - 1; j++) {
                g_backends[j] = g_backends[j + 1];
            }
            g_num_backends--;
            pthread_mutex_unlock(&g_backend_lock);
            return MOONLAB_SCHED_OK;
        }
    }
    pthread_mutex_unlock(&g_backend_lock);
    return MOONLAB_SCHED_BACKEND_NOT_FOUND;
}

const moonlab_backend_t *moonlab_find_backend(const char *name)
{
    if (!name) return NULL;
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    const moonlab_backend_t *hit = NULL;
    for (int i = 0; i < g_num_backends; i++) {
        if (strcmp(g_backends[i].name, name) == 0) {
            hit = &g_backends[i];
            break;
        }
    }
    pthread_mutex_unlock(&g_backend_lock);
    return hit;
}

int moonlab_num_backends(void)
{
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    const int n = g_num_backends;
    pthread_mutex_unlock(&g_backend_lock);
    return n;
}

int moonlab_list_backends(const char **out_names, int max)
{
    if (!out_names || max <= 0) return 0;
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    const int n = g_num_backends < max ? g_num_backends : max;
    for (int i = 0; i < n; i++) {
        out_names[i] = g_backends[i].name;
    }
    pthread_mutex_unlock(&g_backend_lock);
    return n;
}

int moonlab_job_set_backend(moonlab_job_t *j, const char *backend_name)
{
    if (!j) return MOONLAB_SCHED_BAD_ARG;
    free(j->backend_name);
    j->backend_name = NULL;
    if (!backend_name) return MOONLAB_SCHED_OK;
    const size_t len = strlen(backend_name);
    j->backend_name = (char *)malloc(len + 1);
    if (!j->backend_name) return MOONLAB_SCHED_OOM;
    memcpy(j->backend_name, backend_name, len + 1);
    return MOONLAB_SCHED_OK;
}

const char *moonlab_job_backend(const moonlab_job_t *j)
{
    return j ? j->backend_name : NULL;
}

int moonlab_scheduler_set_completion_hook(moonlab_completion_hook_fn hook, void *ctx)
{
    ensure_default_backend();
    pthread_mutex_lock(&g_backend_lock);
    g_completion_hook     = hook;
    g_completion_hook_ctx = ctx;
    pthread_mutex_unlock(&g_backend_lock);
    return MOONLAB_SCHED_OK;
}

/* Thread-local per-request context.  No locking needed -- one
 * connection / one request handler / one scheduler_run / one hook
 * fire all live on the same thread. */
static __thread const char *t_request_tenant_id = NULL;
static __thread const char *t_request_id        = NULL;

void moonlab_scheduler_set_request_context(const char *tenant_id,
                                           const char *request_id)
{
    t_request_tenant_id = tenant_id;
    t_request_id        = request_id;
}

const char *moonlab_scheduler_current_tenant_id(void)
{
    return t_request_tenant_id;
}

const char *moonlab_scheduler_current_request_id(void)
{
    return t_request_id;
}

void moonlab_scheduler_fire_completion_hook(
    const moonlab_job_t          *job,
    const moonlab_job_results_t  *results,
    const char                   *backend_name)
{
    pthread_mutex_lock(&g_backend_lock);
    moonlab_completion_hook_fn hook = g_completion_hook;
    void *hook_ctx = g_completion_hook_ctx;
    pthread_mutex_unlock(&g_backend_lock);
    if (hook) {
        hook(job, results, backend_name, hook_ctx);
    }
}

int moonlab_scheduler_run(moonlab_job_t         *j,
                          moonlab_job_results_t *out)
{
    if (!j || !out) return MOONLAB_SCHED_BAD_ARG;
    if (j->num_shots <= 0) return MOONLAB_SCHED_BAD_ARG;

    memset(out, 0, sizeof(*out));

    /* Backend dispatch (since v1.1.0): the job may pin a non-default
     * backend ("ibm-noise", "rigetti-noise", QPU drivers, ...).  An
     * unset backend_name routes to "simulator" (auto-registered).
     * An unknown name is a hard error -- silently falling back to
     * the simulator would hide config bugs in production deploys. */
    const char *bname = j->backend_name ? j->backend_name : "simulator";
    const moonlab_backend_t *be = moonlab_find_backend(bname);
    if (!be) {
        return MOONLAB_SCHED_BACKEND_NOT_FOUND;
    }
    const int rc = be->execute(j, out, be->ctx);
    if (rc != MOONLAB_SCHED_OK) {
        moonlab_job_results_free(out);
        return rc;
    }

    /* Completion hook (since v1.1.0): private overlay uses this for
     * billing / audit / customer-dashboard event push.  Snapshot
     * under the lock so a concurrent set_completion_hook() can't
     * race us.  Hook fires synchronously on the caller's thread; if
     * it's long-running, the overlay is expected to hand off to a
     * worker pool itself. */
    pthread_mutex_lock(&g_backend_lock);
    moonlab_completion_hook_fn hook = g_completion_hook;
    void *hook_ctx = g_completion_hook_ctx;
    pthread_mutex_unlock(&g_backend_lock);
    if (hook) {
        hook(j, out, bname, hook_ctx);
    }

    return MOONLAB_SCHED_OK;
}

void moonlab_job_results_free(moonlab_job_results_t *r)
{
    if (!r) return;
    free(r->outcomes);
    free(r->worker_seconds);
    r->outcomes = NULL;
    r->worker_seconds = NULL;
}

/* -------- JSON serialisation -------- */

int moonlab_job_to_json(const moonlab_job_t *j,
                        char *buf, size_t bufsize)
{
    if (!j) return MOONLAB_SCHED_BAD_ARG;

    /* Render to a local growable string, then snprintf into buf at
     * the end.  For an MVP we render directly into buf and use
     * snprintf's tail-pointer trick. */
    int total = 0;
    #define APPEND(...) do { \
        int needed = snprintf(buf ? buf + total : NULL, \
                              buf ? (bufsize > (size_t)total ? bufsize - total : 0) : 0, \
                              __VA_ARGS__); \
        if (needed < 0) return MOONLAB_SCHED_INTERNAL; \
        total += needed; \
    } while (0)

    APPEND("{\n");
    APPEND("  \"schema\": \"moonlab/job/v0.7.0\",\n");
    APPEND("  \"num_qubits\": %d,\n", j->num_qubits);
    APPEND("  \"num_shots\": %d,\n", j->num_shots);
    APPEND("  \"num_workers\": %d,\n", j->num_workers);
    APPEND("  \"rng_seed\": \"0x%016llx\",\n",
           (unsigned long long)j->rng_seed);
    APPEND("  \"gates\": [");
    for (int g = 0; g < j->num_gates; g++) {
        const sched_gate_record_t *gr = &j->gates[g];
        APPEND("\n    { \"type\": %d, \"target\": %d",
               (int)gr->type, gr->target);
        if (gr->type == MOONLAB_QGTL_GATE_CNOT ||
            gr->type == MOONLAB_QGTL_GATE_CY   ||
            gr->type == MOONLAB_QGTL_GATE_CZ   ||
            gr->type == MOONLAB_QGTL_GATE_SWAP) {
            APPEND(", \"control\": %d", gr->control);
        }
        if (gr->has_param) {
            APPEND(", \"theta\": %.17g", gr->theta);
        }
        APPEND(" }%s", g + 1 < j->num_gates ? "," : "");
    }
    APPEND("\n  ]\n");
    APPEND("}\n");
    #undef APPEND

    return total;
}

/* ============================================================================
 * MPI transport (v0.7.4) -- collective entry point.
 * ========================================================================= */

int moonlab_scheduler_run_mpi(moonlab_job_t         *job,
                              moonlab_job_results_t *out,
                              void                  *ctx_opaque)
{
#if defined(HAS_MPI)
    distributed_ctx_t *ctx = (distributed_ctx_t *)ctx_opaque;
    if (!out) return MOONLAB_SCHED_BAD_ARG;
    if (!ctx) {
        /* No MPI context -- fall back to the in-process worker fan-out. */
        if (!job) return MOONLAB_SCHED_BAD_ARG;
        return moonlab_scheduler_run(job, out);
    }

    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);

    /* Broadcast the job header: { num_qubits, num_shots, num_gates,
     * rng_seed_lo, rng_seed_hi, reserved }. */
    int header[6] = {0};
    if (rank == 0) {
        if (!job || job->num_shots <= 0) return MOONLAB_SCHED_BAD_ARG;
        header[0] = job->num_qubits;
        header[1] = job->num_shots;
        header[2] = job->num_gates;
        header[3] = (int)(uint32_t)(job->rng_seed & 0xFFFFFFFFu);
        header[4] = (int)(uint32_t)((job->rng_seed >> 32) & 0xFFFFFFFFu);
    }
    if (mpi_broadcast(ctx, header, sizeof(header), 0) != 0) {
        return MOONLAB_SCHED_INTERNAL;
    }

    const int num_qubits  = header[0];
    const int total_shots = header[1];
    const int num_gates   = header[2];
    const uint64_t rng_seed =
        ((uint64_t)(uint32_t)header[3]) |
        ((uint64_t)(uint32_t)header[4] << 32);

    if (num_qubits < 1 || total_shots < 1) {
        return MOONLAB_SCHED_BAD_ARG;
    }

    /* Broadcast the gate list. */
    sched_gate_record_t *gates = NULL;
    if (num_gates > 0) {
        gates = (sched_gate_record_t *)
            malloc((size_t)num_gates * sizeof(sched_gate_record_t));
        if (!gates) return MOONLAB_SCHED_OOM;
        if (rank == 0) memcpy(gates, job->gates,
                              (size_t)num_gates * sizeof(sched_gate_record_t));
        if (mpi_broadcast(ctx, gates,
                          (size_t)num_gates * sizeof(sched_gate_record_t),
                          0) != 0) {
            free(gates);
            return MOONLAB_SCHED_INTERNAL;
        }
    }

    /* Per-rank slice [start, end). */
    const int slice_start = (int)((long long)rank * total_shots / size);
    const int slice_end   = (int)((long long)(rank + 1) * total_shots / size);
    const int slice_size  = slice_end - slice_start;

    /* Build a working job from the broadcast gate list. */
    moonlab_job_t local = {
        .num_qubits  = num_qubits,
        .num_shots   = slice_size,
        .num_workers = 1,
        .rng_seed    = rng_seed,
        .num_gates   = num_gates,
        .cap         = num_gates,
        .gates       = gates,
    };

    uint64_t *slice_outcomes = NULL;
    int rc = MOONLAB_SCHED_OK;
    if (slice_size > 0) {
        slice_outcomes = (uint64_t *)malloc((size_t)slice_size * sizeof(uint64_t));
        if (!slice_outcomes) { free(gates); return MOONLAB_SCHED_OOM; }
        rc = run_worker_slice(&local, 0, slice_size, rank, slice_outcomes);
    }

    /* Allgather per-rank slice sizes; mpi_gather contract is uniform-
     * size so we pad to max_slice when gathering outcomes. */
    int *slice_sizes = (int *)malloc((size_t)size * sizeof(int));
    if (!slice_sizes) {
        free(gates); free(slice_outcomes);
        return MOONLAB_SCHED_OOM;
    }
    int my_slice = slice_size;
    if (mpi_allgather(ctx, &my_slice, sizeof(int), slice_sizes) != 0) {
        free(gates); free(slice_outcomes); free(slice_sizes);
        return MOONLAB_SCHED_INTERNAL;
    }

    int max_slice = 0;
    for (int r = 0; r < size; r++) if (slice_sizes[r] > max_slice) max_slice = slice_sizes[r];

    uint64_t *padded = (uint64_t *)calloc((size_t)max_slice, sizeof(uint64_t));
    if (!padded) {
        free(gates); free(slice_outcomes); free(slice_sizes);
        return MOONLAB_SCHED_OOM;
    }
    if (slice_outcomes) memcpy(padded, slice_outcomes,
                                (size_t)slice_size * sizeof(uint64_t));

    uint64_t *gathered = NULL;
    if (rank == 0) {
        gathered = (uint64_t *)malloc((size_t)max_slice * (size_t)size * sizeof(uint64_t));
        if (!gathered) {
            free(gates); free(slice_outcomes); free(slice_sizes); free(padded);
            return MOONLAB_SCHED_OOM;
        }
    }
    if (mpi_gather(ctx, padded, (size_t)max_slice * sizeof(uint64_t),
                   gathered, 0) != 0) {
        free(gates); free(slice_outcomes); free(slice_sizes);
        free(padded); free(gathered);
        return MOONLAB_SCHED_INTERNAL;
    }

    memset(out, 0, sizeof(*out));
    out->num_qubits       = num_qubits;
    out->num_workers_used = size;

    if (rank == 0) {
        out->total_shots = total_shots;
        out->outcomes = (uint64_t *)malloc((size_t)total_shots * sizeof(uint64_t));
        out->worker_seconds = (double *)calloc((size_t)size, sizeof(double));
        if (!out->outcomes || !out->worker_seconds) {
            moonlab_job_results_free(out);
            free(gates); free(slice_outcomes); free(slice_sizes);
            free(padded); free(gathered);
            return MOONLAB_SCHED_OOM;
        }
        /* Trim per-rank padding while merging. */
        int off = 0;
        for (int r = 0; r < size; r++) {
            memcpy(&out->outcomes[off],
                   &gathered[(size_t)r * (size_t)max_slice],
                   (size_t)slice_sizes[r] * sizeof(uint64_t));
            off += slice_sizes[r];
        }
    } else {
        out->total_shots = slice_size;
        out->outcomes = slice_outcomes;  /* transfer ownership */
        slice_outcomes = NULL;
    }

    free(gates);
    free(slice_outcomes);
    free(slice_sizes);
    free(padded);
    free(gathered);
    return rc;
#else
    (void)job; (void)out; (void)ctx_opaque;
    return MOONLAB_SCHED_NOT_BUILT;
#endif
}
