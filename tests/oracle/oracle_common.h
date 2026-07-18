/**
 * @file oracle_common.h
 * @brief Shared machinery for the Moonlab adversarial oracle pillars.
 *
 * Provides the corpus data structures, a deterministic splitmix64 entropy
 * source (so every sampling probe is a pure function of its seed), the
 * per-backend gate dispatchers, the KNOWN_FAILURES allowlist reader, and the
 * probe accounting framework used by every oracle binary.
 *
 * Every helper is `static inline` so a translation unit that includes this
 * header but does not use a given helper produces no unused-symbol warning.
 *
 * See .swarm/ADVERSARIAL_TESTING_CAMPAIGN.md for the doctrine.
 */
#ifndef MOONLAB_ORACLE_COMMON_H
#define MOONLAB_ORACLE_COMMON_H

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"
#include "../../src/algorithms/tensor_network/tn_measurement.h"
#include "../../src/backends/clifford/clifford.h"

/* ------------------------------------------------------------------ */
/* Corpus data structures (populated by the generated circuit_corpus.h) */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *g;   /* gate mnemonic (h,x,y,z,s,sdg,t,tdg,rx,ry,rz,cnot,cz,swap) */
    int q0;          /* first qubit (control for cnot) */
    int q1;          /* second qubit, or -1 for a 1q gate */
    double p;        /* rotation angle for rx/ry/rz, else 0 */
} oracle_gate_t;

typedef struct {
    const char *id;         /* stable probe base id, e.g. clifford_n4_d16_s0 */
    const char *cls;        /* circuit class */
    int num_qubits;
    int depth;
    int num_gates;
    const oracle_gate_t *gates;
} oracle_circuit_t;

/* ------------------------------------------------------------------ */
/* Deterministic entropy (splitmix64) -- no OS RNG, no wall clock.      */
/* ------------------------------------------------------------------ */

typedef struct { uint64_t state; } oracle_rng_t;

static inline uint64_t oracle_splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void oracle_rng_seed(oracle_rng_t *r, uint64_t seed) {
    r->state = seed;
}

/* Stable FNV-1a hash for deriving per-probe streams.  Never mix pointer
 * values into deterministic seeds: ASLR makes those streams process-local. */
static inline uint64_t oracle_stable_id_hash(const char *id) {
    uint64_t hash = 14695981039346656037ULL;
    const unsigned char *p = (const unsigned char *)id;
    while (*p) {
        hash ^= (uint64_t)*p++;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static inline double oracle_rng_unit(oracle_rng_t *r) {
    /* 53-bit uniform in [0, 1). */
    return (double)(oracle_splitmix64(&r->state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline uint32_t oracle_rng_below(oracle_rng_t *r, uint32_t bound) {
    return (uint32_t)(oracle_splitmix64(&r->state) % bound);
}

/* quantum_entropy_fn-compatible callback backed by an oracle_rng_t. */
static inline int oracle_entropy_get_bytes(void *ud, uint8_t *buf, size_t n) {
    oracle_rng_t *r = (oracle_rng_t *)ud;
    size_t i = 0;
    while (i < n) {
        uint64_t word = oracle_splitmix64(&r->state);
        for (int b = 0; b < 8 && i < n; b++, i++) {
            buf[i] = (uint8_t)(word >> (8 * b));
        }
    }
    return 0;
}

static inline void oracle_entropy_bind(quantum_entropy_ctx_t *ctx,
                                       oracle_rng_t *rng, uint64_t seed) {
    oracle_rng_seed(rng, seed);
    quantum_entropy_init(ctx, oracle_entropy_get_bytes, rng);
}

/* ------------------------------------------------------------------ */
/* Per-backend gate dispatch.                                          */
/* Returns 0 on success, nonzero on an unknown / unsupported gate.      */
/* ------------------------------------------------------------------ */

static inline int oracle_apply_dense(quantum_state_t *s, const oracle_gate_t *g) {
    const char *n = g->g;
    if (!strcmp(n, "h"))    return gate_hadamard(s, g->q0);
    if (!strcmp(n, "x"))    return gate_pauli_x(s, g->q0);
    if (!strcmp(n, "y"))    return gate_pauli_y(s, g->q0);
    if (!strcmp(n, "z"))    return gate_pauli_z(s, g->q0);
    if (!strcmp(n, "s"))    return gate_s(s, g->q0);
    if (!strcmp(n, "sdg"))  return gate_s_dagger(s, g->q0);
    if (!strcmp(n, "t"))    return gate_t(s, g->q0);
    if (!strcmp(n, "tdg"))  return gate_t_dagger(s, g->q0);
    if (!strcmp(n, "rx"))   return gate_rx(s, g->q0, g->p);
    if (!strcmp(n, "ry"))   return gate_ry(s, g->q0, g->p);
    if (!strcmp(n, "rz"))   return gate_rz(s, g->q0, g->p);
    if (!strcmp(n, "cnot")) return gate_cnot(s, g->q0, g->q1);
    if (!strcmp(n, "cz"))   return gate_cz(s, g->q0, g->q1);
    if (!strcmp(n, "swap")) return gate_swap(s, g->q0, g->q1);
    return -1;
}

static inline int oracle_apply_mps(tn_mps_state_t *s, const oracle_gate_t *g) {
    const char *n = g->g;
    if (!strcmp(n, "h"))    return tn_apply_h(s, (uint32_t)g->q0);
    if (!strcmp(n, "x"))    return tn_apply_x(s, (uint32_t)g->q0);
    if (!strcmp(n, "y"))    return tn_apply_y(s, (uint32_t)g->q0);
    if (!strcmp(n, "z"))    return tn_apply_z(s, (uint32_t)g->q0);
    if (!strcmp(n, "s"))    return tn_apply_s(s, (uint32_t)g->q0);
    if (!strcmp(n, "sdg")) {  /* S^dagger = S^3 (exact, correct global phase). */
        int rc = 0;
        for (int k = 0; k < 3 && rc == 0; k++) rc = tn_apply_s(s, (uint32_t)g->q0);
        return rc;
    }
    if (!strcmp(n, "t"))    return tn_apply_t(s, (uint32_t)g->q0);
    if (!strcmp(n, "tdg")) {  /* T^dagger = T^7 (exact). */
        int rc = 0;
        for (int k = 0; k < 7 && rc == 0; k++) rc = tn_apply_t(s, (uint32_t)g->q0);
        return rc;
    }
    if (!strcmp(n, "rx"))   return tn_apply_rx(s, (uint32_t)g->q0, g->p);
    if (!strcmp(n, "ry"))   return tn_apply_ry(s, (uint32_t)g->q0, g->p);
    if (!strcmp(n, "rz"))   return tn_apply_rz(s, (uint32_t)g->q0, g->p);
    if (!strcmp(n, "cnot")) return tn_apply_cnot(s, (uint32_t)g->q0, (uint32_t)g->q1);
    if (!strcmp(n, "cz"))   return tn_apply_cz(s, (uint32_t)g->q0, (uint32_t)g->q1);
    if (!strcmp(n, "swap")) return tn_apply_swap(s, (uint32_t)g->q0, (uint32_t)g->q1);
    return -1;
}

static inline int oracle_apply_clifford(clifford_tableau_t *t, const oracle_gate_t *g) {
    const char *n = g->g;
    if (!strcmp(n, "h"))    return clifford_h(t, (size_t)g->q0);
    if (!strcmp(n, "x"))    return clifford_x(t, (size_t)g->q0);
    if (!strcmp(n, "y"))    return clifford_y(t, (size_t)g->q0);
    if (!strcmp(n, "z"))    return clifford_z(t, (size_t)g->q0);
    if (!strcmp(n, "s"))    return clifford_s(t, (size_t)g->q0);
    if (!strcmp(n, "sdg"))  return clifford_s_dag(t, (size_t)g->q0);
    if (!strcmp(n, "cnot")) return clifford_cnot(t, (size_t)g->q0, (size_t)g->q1);
    if (!strcmp(n, "cz"))   return clifford_cz(t, (size_t)g->q0, (size_t)g->q1);
    if (!strcmp(n, "swap")) return clifford_swap(t, (size_t)g->q0, (size_t)g->q1);
    return -1;   /* non-Clifford gate: caller must not run this backend. */
}

static inline int oracle_class_is_clifford(const oracle_circuit_t *c) {
    return strcmp(c->cls, "clifford") == 0;
}

/* ------------------------------------------------------------------ */
/* KNOWN_FAILURES allowlist.                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    char **ids;
    int count;
    int cap;
} oracle_allowlist_t;

static inline const char *oracle_known_failures_path(void) {
    const char *env = getenv("MOONLAB_ORACLE_KNOWN_FAILURES");
    if (env && env[0]) return env;
#ifdef ORACLE_KNOWN_FAILURES_PATH
    return ORACLE_KNOWN_FAILURES_PATH;
#else
    return "tests/oracle/KNOWN_FAILURES.txt";
#endif
}

static inline oracle_allowlist_t oracle_load_allowlist(void) {
    oracle_allowlist_t a = {0};
    const char *path = oracle_known_failures_path();
    FILE *f = fopen(path, "r");
    if (!f) return a;  /* No allowlist file: every failure is a real failure. */
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
        /* First whitespace-delimited token is the probe id. */
        char *end = p;
        while (*end && *end != ' ' && *end != '\t' && *end != '\n' && *end != '\r') end++;
        size_t len = (size_t)(end - p);
        if (len == 0) continue;
        if (a.count == a.cap) {
            a.cap = a.cap ? a.cap * 2 : 16;
            a.ids = (char **)realloc(a.ids, (size_t)a.cap * sizeof(char *));
        }
        char *id = (char *)malloc(len + 1);
        memcpy(id, p, len);
        id[len] = '\0';
        a.ids[a.count++] = id;
    }
    fclose(f);
    return a;
}

/* An allowlist entry matches a probe id exactly, or -- when it ends in '*' --
 * as a prefix. A trailing '*' lets one entry quarantine a whole known-bug
 * family (e.g. reversed_2q_*__diff_mps) instead of enumerating every probe. */
static inline int oracle_is_allowlisted(const oracle_allowlist_t *a, const char *id) {
    for (int i = 0; i < a->count; i++) {
        const char *pat = a->ids[i];
        size_t len = strlen(pat);
        if (len > 0 && pat[len - 1] == '*') {
            if (strncmp(pat, id, len - 1) == 0) return 1;
        } else if (strcmp(pat, id) == 0) {
            return 1;
        }
    }
    return 0;
}

static inline void oracle_free_allowlist(oracle_allowlist_t *a) {
    for (int i = 0; i < a->count; i++) free(a->ids[i]);
    free(a->ids);
    a->ids = NULL;
    a->count = a->cap = 0;
}

/* ------------------------------------------------------------------ */
/* Probe accounting.                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *pillar;
    int total;
    int failed;   /* non-allowlisted failures */
    int xfail;    /* allowlisted (expected) failures */
    int xpass;    /* allowlisted probes that passed -> stale allowlist entry */
    oracle_allowlist_t allow;
} oracle_ctx_t;

static inline void oracle_ctx_init(oracle_ctx_t *ctx, const char *pillar) {
    ctx->pillar = pillar;
    ctx->total = ctx->failed = ctx->xfail = ctx->xpass = 0;
    ctx->allow = oracle_load_allowlist();
}

static inline void oracle_probe_pass(oracle_ctx_t *ctx, const char *id) {
    ctx->total++;
    if (oracle_is_allowlisted(&ctx->allow, id)) {
        ctx->xpass++;
        fprintf(stdout, "XPASS %s %s  (stale allowlist entry -- remove it)\n",
                ctx->pillar, id);
    }
}

static inline void oracle_probe_fail(oracle_ctx_t *ctx, const char *id,
                                     const char *fmt, ...) {
    ctx->total++;
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    if (oracle_is_allowlisted(&ctx->allow, id)) {
        ctx->xfail++;
        fprintf(stdout, "XFAIL %s %s  %s\n", ctx->pillar, id, detail);
    } else {
        ctx->failed++;
        fprintf(stderr, "ORACLE_FAIL %s %s  %s\n", ctx->pillar, id, detail);
    }
}

/* Emit the machine-readable summary the runner parses, free the allowlist,
 * and return the process exit status (nonzero iff a real failure occurred). */
static inline int oracle_finish(oracle_ctx_t *ctx) {
    fprintf(stdout,
            "ORACLE_SUMMARY pillar=%s total=%d failed=%d xfail=%d xpass=%d\n",
            ctx->pillar, ctx->total, ctx->failed, ctx->xfail, ctx->xpass);
    oracle_free_allowlist(&ctx->allow);
    return ctx->failed > 0 ? 1 : 0;
}

#endif /* MOONLAB_ORACLE_COMMON_H */
