/**
 * @file    decoder_bench.c
 * @brief   Multi-decoder bench harness scaffold implementation.
 *
 * Ships the dispatcher + in-tree GREEDY decoder.  SBNN /
 * LIBIRREP_SS / PYMATCHING slots return MOONLAB_DECODER_NOT_BUILT
 * until v0.6.8 wires them.  MWPM_EXACT currently shares the greedy
 * implementation as a placeholder; v0.6.8 lifts the exact +
 * 2-opt path out of `examples/applications/surface_code_threshold.c`
 * into a reusable library function and points the slot at it.
 */

#include "decoder_bench.h"
#include "mwpm_exact.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef MOONLAB_HAS_LIBIRREP
#include <irrep/css_code.h>
#include <irrep/single_shot.h>
#include <irrep/toric_code.h>
#include <irrep/types.h>
#endif

#ifdef MOONLAB_HAS_SBNN
#include <qec_decoder/qec_decoder.h>
#include <toric_code.h>
#endif

/* PYMATCHING bridge: invoke a Python subprocess that wraps the
 * pymatching package.  The bridge script lives at
 * `src/applications/pymatching_bridge.py`; its path is baked into
 * the library at compile time as MOONLAB_PYMATCHING_SCRIPT_PATH so
 * tests + the installed library both find it without env-var
 * fiddling.  Available unconditionally -- Python subprocess + JSON
 * I/O works on any platform with python3 on PATH and pymatching
 * pip-installed, so we don't need a build flag. */
#ifndef MOONLAB_PYMATCHING_SCRIPT_PATH
#  define MOONLAB_PYMATCHING_SCRIPT_PATH NULL
#endif

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

const char *moonlab_decoder_slot_name(moonlab_decoder_kind_t slot)
{
    switch (slot) {
    case MOONLAB_DECODER_GREEDY:      return "greedy";
    case MOONLAB_DECODER_MWPM_EXACT:  return "mwpm_exact";
    case MOONLAB_DECODER_SBNN:        return "sbnn";
    case MOONLAB_DECODER_LIBIRREP_SS: return "libirrep_single_shot";
    case MOONLAB_DECODER_PYMATCHING:  return "pymatching";
    default:                          return "unknown";
    }
}

int moonlab_decoder_slot_available(moonlab_decoder_kind_t slot)
{
    switch (slot) {
    case MOONLAB_DECODER_GREEDY:
    case MOONLAB_DECODER_MWPM_EXACT:
        return 1;
    case MOONLAB_DECODER_LIBIRREP_SS:
#ifdef MOONLAB_HAS_LIBIRREP
        return 1;
#else
        return 0;
#endif
    case MOONLAB_DECODER_SBNN:
#ifdef MOONLAB_HAS_SBNN
        return 1;
#else
        return 0;
#endif
    case MOONLAB_DECODER_PYMATCHING:
        /* Unconditional: a python3 + pymatching subprocess is the
         * transport.  Availability check is runtime (errors from
         * the subprocess if python or pymatching is missing). */
        return MOONLAB_PYMATCHING_SCRIPT_PATH != NULL ? 1 : 0;
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------
 * In-tree GREEDY decoder
 *
 * Naive nearest-pair matching: for each pair of flagged stabilisers,
 * connect them along a straight path of data qubits.  Far from
 * optimal but always succeeds and gives a reasonable baseline.
 * ------------------------------------------------------------------ */

/* Compute the linear data-qubit index between two stabilisers on a
 * d x d torus, picking a shortest path along the lattice.  Returns
 * -1 if no straight path exists (shouldn't happen on a torus). */
static int torus_edge_between(int d, int a, int b, unsigned char *flip)
{
    const int ax = a / d, ay = a % d;
    const int bx = b / d, by = b % d;
    int dx = (bx - ax + d) % d;
    int dy = (by - ay + d) % d;
    if (dx > d / 2) dx -= d;
    if (dy > d / 2) dy -= d;
    /* Walk in y first, then in x; flip each crossed edge.  The
     * edge-index convention matches compute_syndrome (and the
     * surface_code_threshold.c harness): horizontal edge h(x, y) at
     * index `x * d + y` connects vertex (x, y) to vertex (x+1, y),
     * so h(x, y) runs in the +X direction.  Vertical edge v(x, y)
     * at index `d*d + x*d + y` connects (x, y) to (x, y+1) and runs
     * in the +Y direction.  Therefore: walking the defect from
     * vertex (ax, ay) to (bx, by), a Y-step crosses VERTICAL
     * edges, and an X-step crosses HORIZONTAL edges.  An earlier
     * version of this function had the index families swapped,
     * which inflated logical-error rates and erased below-threshold
     * scaling. */
    int x = ax, y = ay;
    while (y != by) {
        const int step = (dy > 0) ? 1 : -1;
        const int y_edge = (step > 0) ? y : (y + d - 1) % d;
        const int e = d * d + x * d + y_edge;  /* v(x, y_edge) */
        flip[e] ^= 1;
        y = (y + step + d) % d;
        dy -= step;
    }
    while (x != bx) {
        const int step = (dx > 0) ? 1 : -1;
        const int x_edge = (step > 0) ? x : (x + d - 1) % d;
        const int e = x_edge * d + y;          /* h(x_edge, y) */
        flip[e] ^= 1;
        x = (x + step + d) % d;
        dx -= step;
    }
    return 0;
}

static int decoder_greedy(const moonlab_decoder_input_t *in)
{
    const int d   = in->code->distance;
    const int n   = in->code->num_qubits;
    const int n_s = in->num_stabilisers;

    /* Collect flagged stabiliser indices. */
    int *defects = (int *)malloc((size_t)n_s * sizeof(int));
    if (!defects) return MOONLAB_DECODER_OOM;
    int n_defects = 0;
    for (int i = 0; i < n_s; i++) {
        if (in->syndromes[i]) defects[n_defects++] = i;
    }

    /* Toric stabilisers always come in pairs (sum of plaquette syndromes
     * is zero mod 2 on any closed surface).  Open boundaries can have
     * an odd parity -- we accept it and pair to a virtual boundary
     * defect via a no-op (the un-matched defect leaves residual
     * syndrome which downstream logical-error tracking treats as a
     * logical fault).  Don't error here; just leave any leftover. */
    memset(in->corrections, 0, (size_t)n);

    /* Nearest-pair matching: each defect paired with its closest
     * unmatched neighbour.  O(n_defects^2) -- fine for d <= 9. */
    char *matched = (char *)calloc((size_t)n_defects, 1);
    if (!matched) { free(defects); return MOONLAB_DECODER_OOM; }

    for (int i = 0; i < n_defects; i++) {
        if (matched[i]) continue;
        int best_j = -1;
        int best_dist = INT32_MAX;
        for (int j = i + 1; j < n_defects; j++) {
            if (matched[j]) continue;
            const int ax = defects[i] / d, ay = defects[i] % d;
            const int bx = defects[j] / d, by = defects[j] % d;
            int dx = (bx - ax + d) % d; if (dx > d / 2) dx = d - dx;
            int dy = (by - ay + d) % d; if (dy > d / 2) dy = d - dy;
            const int dist = dx + dy;
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        if (best_j >= 0 && in->code->is_toric) {
            torus_edge_between(d, defects[i], defects[best_j], in->corrections);
            matched[i] = 1;
            matched[best_j] = 1;
        }
    }
    free(matched);
    free(defects);
    return MOONLAB_DECODER_OK;
}

#ifdef MOONLAB_HAS_SBNN
/* SBNN: route the syndrome through SbNN's qec_decoder_t.  Default to
 * the MWPM kind (always available in SbNN v0.4+); the TRANSFORMER /
 * MAMBA kinds in SbNN fall back to MWPM with a stderr warning, so
 * this slot is honest even when the learned models are unimplemented
 * upstream.  Translates moonlab's syndrome byte layout (row-major,
 * `[a*d + b]`) into SbNN's `plaquette_syndrome` int layout
 * (`[a*Ly + b]`), runs the decoder, translates SbNN's interleaved
 * `x_errors[2*(x*Ly+y) + dir]` corrections back into moonlab's
 * `[0, d*d)` horizontal + `[d*d, 2*d*d)` vertical layout. */
static int decoder_sbnn(const moonlab_decoder_input_t *in)
{
    if (!in->code->is_toric) return MOONLAB_DECODER_INFEASIBLE;
    const int d = in->code->distance;

    ToricCode *code = initialize_toric_code(d, d);
    if (!code) return MOONLAB_DECODER_OOM;

    /* Push our syndromes into SbNN's plaquette array. */
    for (int a = 0; a < d; a++) {
        for (int b = 0; b < d; b++) {
            code->plaquette_syndrome[a * d + b] =
                in->syndromes[a * d + b] ? 1 : 0;
        }
    }
    /* Zero error accumulators so the decoder writes into a clean slate. */
    for (int q = 0; q < code->num_links; q++) {
        code->x_errors[q] = 0;
        code->z_errors[q] = 0;
    }

    qec_decoder_t dec = qec_decoder_create(QEC_DECODER_MWPM);
    const int rc = qec_decoder_run(&dec, code);
    if (rc != 0) {
        free_toric_code(code);
        return MOONLAB_DECODER_OOM;
    }

    /* Translate SbNN's interleaved link-index layout into moonlab's
     * horizontal-then-vertical layout.  SbNN: `2*(x*Ly + y) + dir`
     * with dir=0 east (horizontal) and dir=1 north (vertical).
     * Moonlab: `h_idx(x, y) = x*d + y`, `v_idx(x, y) = d*d + x*d + y`. */
    memset(in->corrections, 0, (size_t)in->code->num_qubits);
    for (int x = 0; x < d; x++) {
        for (int y = 0; y < d; y++) {
            const int h_sbnn = 2 * (x * d + y);
            const int v_sbnn = 2 * (x * d + y) + 1;
            if (code->x_errors[h_sbnn]) in->corrections[x * d + y] ^= 1;
            if (code->x_errors[v_sbnn]) in->corrections[d * d + x * d + y] ^= 1;
        }
    }

    free_toric_code(code);
    return MOONLAB_DECODER_OK;
}
#endif

#ifdef MOONLAB_HAS_LIBIRREP
/* LIBIRREP_SS: build the matching libirrep toric code, lift it to a
 * single-shot code (Quintavalle-Vasmer-Roffe-Campbell 2021), verify
 * the meta-check property, then defer to greedy for the data-qubit
 * correction.  The libirrep call's purpose is to confirm the code's
 * meta-check matrices exist + are valid -- a precondition for a real
 * single-shot decoder.  Full single-shot decoding (using the
 * meta-syndrome to filter measurement errors) is a v0.7+ piece. */
static int decoder_libirrep_ss(const moonlab_decoder_input_t *in)
{
    if (!in->code->is_toric) return MOONLAB_DECODER_INFEASIBLE;

    irrep_toric_params_t p;
    if (irrep_toric_init(&p, in->code->distance, in->code->distance) != IRREP_OK) {
        return MOONLAB_DECODER_OOM;
    }
    irrep_css_code_t css;
    if (irrep_toric_code_build_css(&p, &css) != IRREP_OK) {
        return MOONLAB_DECODER_OOM;
    }
    irrep_single_shot_code_t ss;
    irrep_status_t st = irrep_single_shot_lift(&css, &ss);
    if (st != IRREP_OK) {
        irrep_css_code_free(&css);
        return MOONLAB_DECODER_OOM;
    }
    st = irrep_single_shot_verify_meta(&ss);
    irrep_single_shot_code_free(&ss);
    irrep_css_code_free(&css);
    if (st != IRREP_OK) {
        return MOONLAB_DECODER_INFEASIBLE; /* meta-check failed: code not single-shot */
    }
    /* Single-shot lift verified.  Defer to greedy for correction. */
    return decoder_greedy(in);
}
#endif

/* PYMATCHING: pipe a JSON syndrome record over the Python subprocess'
 * stdin, parse the hex-encoded correction vector from its stdout.
 * Falls back to OOM/BAD_ARG if Python or pymatching isn't installed
 * (subprocess emits "ERR import:" line); falls back to GREEDY's
 * result only on caller request -- here we just propagate the error. */
static int decoder_pymatching(const moonlab_decoder_input_t *in)
{
    if (!in->code->is_toric) return MOONLAB_DECODER_INFEASIBLE;
    if (!MOONLAB_PYMATCHING_SCRIPT_PATH) return MOONLAB_DECODER_NOT_BUILT;

    const int d = in->code->distance;
    const int n_s = in->num_stabilisers;
    const int n_q = in->code->num_qubits;

    /* Serialise input JSON.  Worst case for syndrome len d^2 <= 1024:
     * "[0, 1, 0, ...]" -> 4 bytes/entry. */
    const size_t json_cap = 256 + (size_t)n_s * 4;
    char *json = (char *)malloc(json_cap);
    if (!json) return MOONLAB_DECODER_OOM;
    int off = snprintf(json, json_cap,
        "{\"distance\":%d,\"is_toric\":true,\"rng_seed\":%llu,\"syndromes\":[",
        d, (unsigned long long)in->rng_seed);
    for (int i = 0; i < n_s; i++) {
        off += snprintf(json + off, json_cap - (size_t)off, "%s%d",
                        i ? "," : "", in->syndromes[i] ? 1 : 0);
    }
    off += snprintf(json + off, json_cap - (size_t)off, "]}");

    /* Set up stdin / stdout pipes around python3. */
    int in_pipe[2];   /* parent writes, child reads */
    int out_pipe[2];  /* child writes, parent reads */
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
        free(json); return MOONLAB_DECODER_OOM;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        free(json); return MOONLAB_DECODER_OOM;
    }
    if (pid == 0) {
        /* Child: wire pipes to stdin/stdout, exec python3. */
        dup2(in_pipe[0], 0);
        dup2(out_pipe[1], 1);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        execlp("python3", "python3", MOONLAB_PYMATCHING_SCRIPT_PATH, (char *)NULL);
        _exit(127);
    }
    /* Parent. */
    close(in_pipe[0]);
    close(out_pipe[1]);
    /* Write JSON to child stdin. */
    const ssize_t written = write(in_pipe[1], json, (size_t)off);
    close(in_pipe[1]);
    free(json);
    if (written != off) {
        close(out_pipe[0]);
        return MOONLAB_DECODER_OOM;
    }

    /* Read child stdout. */
    char buf[8192];
    size_t total = 0;
    ssize_t r;
    while ((r = read(out_pipe[0], buf + total, sizeof(buf) - 1 - total)) > 0) {
        total += (size_t)r;
        if (total >= sizeof(buf) - 1) break;
    }
    close(out_pipe[0]);
    buf[total] = '\0';

    /* Reap child. */
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}

    /* Parse response: "OK <hex>" on success, "ERR ..." on failure. */
    if (strncmp(buf, "OK ", 3) != 0) {
        return MOONLAB_DECODER_NOT_BUILT; /* pymatching unavailable */
    }
    const char *hex = buf + 3;
    memset(in->corrections, 0, (size_t)n_q);
    for (int q = 0; q < n_q; q++) {
        const char c1 = hex[2 * q];
        const char c2 = hex[2 * q + 1];
        if (c1 == '\0' || c1 == '\n') break;
        const int hi = (c1 >= '0' && c1 <= '9') ? (c1 - '0') :
                       (c1 >= 'a' && c1 <= 'f') ? (c1 - 'a' + 10) : -1;
        const int lo = (c2 >= '0' && c2 <= '9') ? (c2 - '0') :
                       (c2 >= 'a' && c2 <= 'f') ? (c2 - 'a' + 10) : -1;
        if (hi < 0 || lo < 0) return MOONLAB_DECODER_OOM;
        in->corrections[q] = (unsigned char)((hi << 4) | lo) & 1;
    }
    return MOONLAB_DECODER_OK;
}

/* ------------------------------------------------------------------
 * Decoder runtime registry (since v1.0.3)
 *
 * Single source of truth for both the enum dispatcher and the name
 * dispatcher.  The five baked-in decoders auto-register on first use
 * (pthread_once).  External callers (the private overlay, QGTL, etc.)
 * use moonlab_register_decoder() at any time.
 * ------------------------------------------------------------------ */

#define DECODER_REGISTRY_MAX 32

typedef struct {
    char               *name;        /**< owned strdup */
    moonlab_decoder_fn  fn;
    void               *ctx;
    char               *description; /**< owned strdup, may be NULL */
} decoder_slot_t;

static decoder_slot_t  g_decoders[DECODER_REGISTRY_MAX];
static int             g_n_decoders = 0;
static pthread_mutex_t g_decoder_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_decoder_init_once = PTHREAD_ONCE_INIT;

/* Trampolines: each baked-in decoder has the runtime fn signature
 * (in, ctx).  ctx is ignored for the built-ins -- their behaviour is
 * fixed by the source. */
static int decoder_greedy_trampoline(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
    return decoder_greedy(in);
}

static int decoder_mwpm_exact_trampoline(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
    memset(in->corrections, 0, (size_t)in->code->num_qubits);
    const int rc = moonlab_mwpm_exact_decode_toric(
        in->code->distance, in->syndromes,
        in->num_stabilisers, in->corrections);
    if (rc == MOONLAB_MWPM_OK || rc == MOONLAB_MWPM_INFEASIBLE) {
        return MOONLAB_DECODER_OK;
    }
    return MOONLAB_DECODER_OOM;
}

static int decoder_sbnn_trampoline(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
#ifdef MOONLAB_HAS_SBNN
    return decoder_sbnn(in);
#else
    (void)in;
    return MOONLAB_DECODER_NOT_BUILT;
#endif
}

static int decoder_libirrep_ss_trampoline(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
#ifdef MOONLAB_HAS_LIBIRREP
    return decoder_libirrep_ss(in);
#else
    (void)in;
    return MOONLAB_DECODER_NOT_BUILT;
#endif
}

static int decoder_pymatching_trampoline(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
    return decoder_pymatching(in);
}

/* Caller holds g_decoder_lock.  Returns existing slot index for
 * `name`, or -1 if absent. */
static int find_decoder_locked(const char *name)
{
    for (int i = 0; i < g_n_decoders; i++) {
        if (g_decoders[i].name && strcmp(g_decoders[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/* Caller holds g_decoder_lock.  Installs (or replaces) `name`. */
static int install_decoder_locked(const char         *name,
                                  moonlab_decoder_fn  fn,
                                  void               *ctx,
                                  const char         *description)
{
    int idx = find_decoder_locked(name);
    if (idx < 0) {
        if (g_n_decoders >= DECODER_REGISTRY_MAX) {
            return MOONLAB_DECODER_BAD_ARG; /* registry full */
        }
        idx = g_n_decoders++;
        g_decoders[idx].name = NULL;
        g_decoders[idx].description = NULL;
    }
    char *new_name = strdup(name);
    char *new_desc = description ? strdup(description) : NULL;
    if (!new_name || (description && !new_desc)) {
        free(new_name);
        free(new_desc);
        return MOONLAB_DECODER_OOM;
    }
    free(g_decoders[idx].name);
    free(g_decoders[idx].description);
    g_decoders[idx].name        = new_name;
    g_decoders[idx].fn          = fn;
    g_decoders[idx].ctx         = ctx;
    g_decoders[idx].description = new_desc;
    return 0;
}

static void register_builtin_decoders(void)
{
    pthread_mutex_lock(&g_decoder_lock);
    install_decoder_locked("greedy", decoder_greedy_trampoline, NULL,
        "In-tree nearest-pair matching baseline (always available)");
    install_decoder_locked("mwpm_exact", decoder_mwpm_exact_trampoline, NULL,
        "Built-in exact MWPM (brute force <=10 defects, greedy + 2-opt past)");
    install_decoder_locked("sbnn", decoder_sbnn_trampoline, NULL,
        "SbNN learned decoder (gated on MOONLAB_HAS_SBNN)");
    install_decoder_locked("libirrep_single_shot", decoder_libirrep_ss_trampoline,
                           NULL,
        "libirrep single-shot decoder (gated on MOONLAB_HAS_LIBIRREP)");
    install_decoder_locked("pymatching", decoder_pymatching_trampoline, NULL,
        "Stim-pymatching reference via python3 subprocess (gated on bridge script)");
    pthread_mutex_unlock(&g_decoder_lock);
}

static void ensure_decoders_initialised(void)
{
    pthread_once(&g_decoder_init_once, register_builtin_decoders);
}

int moonlab_register_decoder(const char         *name,
                             moonlab_decoder_fn  fn,
                             void               *ctx,
                             const char         *description)
{
    if (!name || !fn) return MOONLAB_DECODER_BAD_ARG;
    ensure_decoders_initialised();
    pthread_mutex_lock(&g_decoder_lock);
    const int rc = install_decoder_locked(name, fn, ctx, description);
    pthread_mutex_unlock(&g_decoder_lock);
    return rc;
}

int moonlab_unregister_decoder(const char *name)
{
    if (!name) return MOONLAB_DECODER_BAD_ARG;
    ensure_decoders_initialised();
    pthread_mutex_lock(&g_decoder_lock);
    const int idx = find_decoder_locked(name);
    if (idx < 0) {
        pthread_mutex_unlock(&g_decoder_lock);
        return MOONLAB_DECODER_BAD_ARG;
    }
    free(g_decoders[idx].name);
    free(g_decoders[idx].description);
    /* Compact: move tail entry into the freed slot to keep the table dense. */
    const int last = g_n_decoders - 1;
    if (idx != last) g_decoders[idx] = g_decoders[last];
    memset(&g_decoders[last], 0, sizeof(g_decoders[last]));
    g_n_decoders--;
    pthread_mutex_unlock(&g_decoder_lock);
    return 0;
}

const moonlab_decoder_entry_t *moonlab_lookup_decoder(const char *name)
{
    /* Returns a pointer into a per-thread static buffer so callers
     * get a stable snapshot while still seeing live ctx updates if
     * they re-call.  Registry rows can move when entries are
     * unregistered (compaction), so we copy out instead of leaking
     * the slot pointer. */
    static __thread moonlab_decoder_entry_t snapshot;
    if (!name) return NULL;
    ensure_decoders_initialised();
    pthread_mutex_lock(&g_decoder_lock);
    const int idx = find_decoder_locked(name);
    if (idx < 0) {
        pthread_mutex_unlock(&g_decoder_lock);
        return NULL;
    }
    snapshot.name        = g_decoders[idx].name;
    snapshot.fn          = g_decoders[idx].fn;
    snapshot.ctx         = g_decoders[idx].ctx;
    snapshot.description = g_decoders[idx].description;
    pthread_mutex_unlock(&g_decoder_lock);
    return &snapshot;
}

int moonlab_num_decoders(void)
{
    ensure_decoders_initialised();
    pthread_mutex_lock(&g_decoder_lock);
    const int n = g_n_decoders;
    pthread_mutex_unlock(&g_decoder_lock);
    return n;
}

int moonlab_list_decoders(const char **out_names, int max)
{
    if (!out_names || max <= 0) return 0;
    ensure_decoders_initialised();
    pthread_mutex_lock(&g_decoder_lock);
    const int n = g_n_decoders < max ? g_n_decoders : max;
    for (int i = 0; i < n; i++) out_names[i] = g_decoders[i].name;
    pthread_mutex_unlock(&g_decoder_lock);
    return n;
}

int moonlab_decoder_decode_by_name(const char                     *name,
                                   const moonlab_decoder_input_t  *in)
{
    if (!in || !in->code || !in->syndromes || !in->corrections) {
        return MOONLAB_DECODER_BAD_ARG;
    }
    if (in->code->distance < 2 || in->code->num_qubits < 1 ||
        in->num_stabilisers < 0) {
        return MOONLAB_DECODER_BAD_ARG;
    }
    if (!name) return MOONLAB_DECODER_BAD_ARG;
    ensure_decoders_initialised();
    /* Take a snapshot of the (fn, ctx) under lock so an unregister
     * racing us can't observe a torn pointer. */
    pthread_mutex_lock(&g_decoder_lock);
    const int idx = find_decoder_locked(name);
    if (idx < 0) {
        pthread_mutex_unlock(&g_decoder_lock);
        return MOONLAB_DECODER_BAD_ARG;
    }
    moonlab_decoder_fn fn = g_decoders[idx].fn;
    void              *ctx = g_decoders[idx].ctx;
    pthread_mutex_unlock(&g_decoder_lock);
    return fn(in, ctx);
}

int moonlab_decoder_decode(moonlab_decoder_kind_t          slot,
                           const moonlab_decoder_input_t  *in)
{
    /* Enum surface: translate slot -> built-in name and delegate to
     * the registry dispatcher.  This makes the registry the single
     * source of truth: any private overlay that re-registers
     * "mwpm_exact" with its own implementation transparently takes
     * over the enum path too. */
    const char *name = moonlab_decoder_slot_name(slot);
    if (!name || strcmp(name, "unknown") == 0) {
        return MOONLAB_DECODER_BAD_ARG;
    }
    return moonlab_decoder_decode_by_name(name, in);
}
