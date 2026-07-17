/**
 * @file diff_backends.c
 * @brief Cross-backend differential driver for the Moonlab adversarial campaign.
 *
 * Runs every circuit in a generated corpus on multiple backends and proves they
 * all produce the SAME result AND match an INDEPENDENT numpy reference oracle
 * (pinned in the corpus by scripts/gen_diff_corpus.py):
 *
 *   - dense statevector  (src/quantum, little-endian qubit0=LSB)
 *   - tn_mps             (plain MPS; big-endian site0=MSB, bit-reversed here)
 *   - Clifford tableau   (for clifford-only cases; exact stabilizer observables)
 *   - GPU statevector    (when quantum_state_create_gpu succeeds at runtime)
 *
 * Comparisons (never loosened):
 *   - full probability vectors: dense/tn_mps/gpu vs reference and vs each other
 *     within 1e-10;
 *   - <Z_i> and <Z_iZ_j> expectations vs reference within 1e-10;
 *   - Clifford exact on stabilizer observables (<=1e-9 for fp noise).
 *
 * A mismatch prints the case id + seed for replay and fails the gate, UNLESS the
 * (case, backend) pair is quarantined in KNOWN_DIVERGENCES.txt because a sibling
 * lane is mid-fix (e.g. the tn 2q-gate transpose for reversed CNOT). Quarantined
 * mismatches are reported but excluded from the exit status.
 *
 * The corpus is read from the flat corpus.txt mirror (whitespace-delimited) so no
 * JSON parser is needed in C; that file is byte-derived from the same in-memory
 * structure as corpus.json, so both are the same shared reference.
 *
 * Usage:
 *   diff_backends <corpus.txt> [--quarantine FILE] [--gpu] [--tn-max-n N] [--verbose]
 *   diff_backends --selftest
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"
#include "../../src/backends/clifford/clifford.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL_FP      1e-10   /* dense/tn_mps/gpu vs reference and cross-backend */
#define TOL_CLIFF   1e-9    /* Clifford stabilizer observables (exact + noise)  */
#define MAX_ZZ      64

/* ---- gate record ---- */
typedef struct { char name[16]; int q0, q1, q2; double angle; } dgate_t;

/* ---- one corpus case ---- */
typedef struct {
    char     id[96];
    char     cls[32];
    int      num_qubits;
    int      depth;
    int      clifford_only;
    uint64_t seed;
    int      ngates;
    dgate_t  *gates;
    /* reference */
    size_t   dim;
    double  *ref_prob;    /* dim */
    double  *ref_z;       /* num_qubits */
    int      nzz;
    int      zz_a[MAX_ZZ];
    int      zz_b[MAX_ZZ];
    double   zz_v[MAX_ZZ];
} corpus_case_t;

/* ---- quarantine set: (case_id, backend) pairs ---- */
typedef struct { char id[96]; char backend[16]; char lane[32]; } quar_t;
static quar_t *g_quar = NULL;
static int     g_nquar = 0;

static int is_quarantined(const char *id, const char *backend) {
    for (int i = 0; i < g_nquar; i++)
        if (strcmp(g_quar[i].id, id) == 0 && strcmp(g_quar[i].backend, backend) == 0)
            return 1;
    return 0;
}

/* ---- counters ---- */
typedef struct { long passed, failed, quarantined, skipped; } counter_t;
static counter_t c_cross;      /* cross-backend agreement checks */
static counter_t c_reference;  /* backend-vs-reference agreement checks */
static long ex_dense = 0, ex_tn = 0, ex_cliff = 0, ex_gpu = 0, ex_tn_skipped = 0;
static int  g_tn_max_n = 1000000;   /* --tn-max-n cap on the exact tn_mps leg */
static int  g_verbose = 0;
static char gpu_reason[128] = "not attempted";

/* ================================================================= */
/*  Helpers                                                          */
/* ================================================================= */

static uint64_t bitrev(uint64_t x, int n) {
    uint64_t r = 0;
    for (int i = 0; i < n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

static double max_abs_dev(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

static void exp_from_prob(const double *prob, int n, size_t dim,
                          double *out_z, const int *za, const int *zb, int nzz,
                          double *out_zz) {
    for (int q = 0; q < n; q++) {
        double e = 0.0;
        for (size_t i = 0; i < dim; i++)
            e += ((i >> q) & 1u) ? -prob[i] : prob[i];
        out_z[q] = e;
    }
    for (int k = 0; k < nzz; k++) {
        double e = 0.0;
        for (size_t i = 0; i < dim; i++) {
            int sa = ((i >> za[k]) & 1u) ? -1 : 1;
            int sb = ((i >> zb[k]) & 1u) ? -1 : 1;
            e += sa * sb * prob[i];
        }
        out_zz[k] = e;
    }
}

/* Record a comparison outcome into a counter, honoring quarantine. */
static int record(counter_t *ctr, int ok, double dev, double tol,
                  const corpus_case_t *cc, const char *backend,
                  const char *what, const char *against) {
    if (ok) { ctr->passed++; return 1; }
    if (is_quarantined(cc->id, backend)) {
        ctr->quarantined++;
        if (g_verbose)
            fprintf(stderr, "  QUARANTINE %s [%s] %s vs %s dev=%.3e (>%.0e)\n",
                    cc->id, backend, what, against, dev, tol);
        return 1;
    }
    ctr->failed++;
    fprintf(stderr, "  FAIL  case=%s seed=%llu class=%s n=%d %s [%s] vs %s "
                    "dev=%.3e (>%.0e)  -- replay: gen_diff_corpus seed reproduces\n",
            cc->id, (unsigned long long)cc->seed, cc->cls, cc->num_qubits,
            what, backend, against, dev, tol);
    return 0;
}

/* ================================================================= */
/*  Dense backend                                                    */
/* ================================================================= */

static int dense_run(const corpus_case_t *cc, double *prob_out) {
    quantum_state_t *sv = quantum_state_create(cc->num_qubits);
    if (!sv) return -1;
    for (int g = 0; g < cc->ngates; g++) {
        const dgate_t *G = &cc->gates[g];
        const char *nm = G->name;
        int rc = 0;
        if      (!strcmp(nm, "h"))   rc = gate_hadamard(sv, G->q0);
        else if (!strcmp(nm, "x"))   rc = gate_pauli_x(sv, G->q0);
        else if (!strcmp(nm, "y"))   rc = gate_pauli_y(sv, G->q0);
        else if (!strcmp(nm, "z"))   rc = gate_pauli_z(sv, G->q0);
        else if (!strcmp(nm, "s"))   rc = gate_s(sv, G->q0);
        else if (!strcmp(nm, "sdg")) rc = gate_s_dagger(sv, G->q0);
        else if (!strcmp(nm, "t"))   rc = gate_t(sv, G->q0);
        else if (!strcmp(nm, "tdg")) rc = gate_t_dagger(sv, G->q0);
        else if (!strcmp(nm, "rx"))  rc = gate_rx(sv, G->q0, G->angle);
        else if (!strcmp(nm, "ry"))  rc = gate_ry(sv, G->q0, G->angle);
        else if (!strcmp(nm, "rz"))  rc = gate_rz(sv, G->q0, G->angle);
        else if (!strcmp(nm, "p"))   rc = gate_phase(sv, G->q0, G->angle);
        else if (!strcmp(nm, "cx"))  rc = gate_cnot(sv, G->q0, G->q1);
        else if (!strcmp(nm, "cz"))  rc = gate_cz(sv, G->q0, G->q1);
        else if (!strcmp(nm, "swap"))rc = gate_swap(sv, G->q0, G->q1);
        else if (!strcmp(nm, "cp"))  rc = gate_cphase(sv, G->q0, G->q1, G->angle);
        else if (!strcmp(nm, "ccx")) rc = gate_toffoli(sv, G->q0, G->q1, G->q2);
        else { fprintf(stderr, "dense: unknown gate %s\n", nm); quantum_state_free(sv); return -1; }
        if (rc != 0) { fprintf(stderr, "dense: gate %s rc=%d\n", nm, rc); quantum_state_free(sv); return -1; }
    }
    for (size_t i = 0; i < cc->dim; i++)
        prob_out[i] = quantum_state_get_probability(sv, i);
    quantum_state_free(sv);
    return 0;
}

/* ================================================================= */
/*  tn_mps backend (big-endian; bit-reversed to little-endian here)   */
/* ================================================================= */

static int tn_run(const corpus_case_t *cc, double *prob_le_out) {
    int n = cc->num_qubits;
    /* 2^floor(n/2) is the max Schmidt rank across any cut of an n-qubit pure
     * state, so it is exact with no truncation. A larger cap can be forced via
     * MOONLAB_DIFF_CHI for diagnostics / stricter margins. */
    uint32_t chi = 1u << ((n + 1) / 2);
    const char *chi_env = getenv("MOONLAB_DIFF_CHI");
    if (chi_env) { long v = strtol(chi_env, NULL, 10); if (v > 0) chi = (uint32_t)v; }
    if (chi > TN_MAX_BOND_DIM) chi = TN_MAX_BOND_DIM;
    tn_state_config_t cfg = tn_state_config_create(chi, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)n, &cfg);
    if (!mps) return -1;
    for (int g = 0; g < cc->ngates; g++) {
        const dgate_t *G = &cc->gates[g];
        const char *nm = G->name;
        int rc = 0;
        if      (!strcmp(nm, "h"))   rc = tn_apply_h(mps, G->q0);
        else if (!strcmp(nm, "x"))   rc = tn_apply_x(mps, G->q0);
        else if (!strcmp(nm, "y"))   rc = tn_apply_y(mps, G->q0);
        else if (!strcmp(nm, "z"))   rc = tn_apply_z(mps, G->q0);
        else if (!strcmp(nm, "s"))   rc = tn_apply_s(mps, G->q0);
        else if (!strcmp(nm, "sdg")) rc = tn_apply_gate_1q(mps, G->q0, &TN_GATE_SDG);
        else if (!strcmp(nm, "t"))   rc = tn_apply_t(mps, G->q0);
        else if (!strcmp(nm, "tdg")) rc = tn_apply_gate_1q(mps, G->q0, &TN_GATE_TDG);
        else if (!strcmp(nm, "rx"))  rc = tn_apply_rx(mps, G->q0, G->angle);
        else if (!strcmp(nm, "ry"))  rc = tn_apply_ry(mps, G->q0, G->angle);
        else if (!strcmp(nm, "rz"))  rc = tn_apply_rz(mps, G->q0, G->angle);
        else if (!strcmp(nm, "p"))   { tn_gate_1q_t gp = tn_gate_phase(G->angle);
                                       rc = tn_apply_gate_1q(mps, G->q0, &gp); }
        else if (!strcmp(nm, "cx"))  rc = tn_apply_cnot(mps, G->q0, G->q1);
        else if (!strcmp(nm, "cz"))  rc = tn_apply_cz(mps, G->q0, G->q1);
        else if (!strcmp(nm, "swap"))rc = tn_apply_swap(mps, G->q0, G->q1);
        else if (!strcmp(nm, "cp"))  { tn_gate_2q_t gc = tn_gate_cphase(G->angle);
                                       rc = tn_apply_gate_2q(mps, G->q0, G->q1, &gc, NULL); }
        else if (!strcmp(nm, "ccx")) rc = tn_apply_toffoli(mps, G->q0, G->q1, G->q2);
        else { fprintf(stderr, "tn: unknown gate %s\n", nm); tn_mps_free(mps); return -1; }
        if (rc != 0) { fprintf(stderr, "tn: gate %s rc=%d\n", nm, rc); tn_mps_free(mps); return -1; }
    }
    double _Complex *amps = malloc(cc->dim * sizeof(double _Complex));
    if (!amps) { tn_mps_free(mps); return -1; }
    tn_state_error_t e = tn_mps_to_statevector(mps, amps);
    if (e != TN_STATE_SUCCESS) { free(amps); tn_mps_free(mps); return -1; }
    for (size_t i = 0; i < cc->dim; i++) {
        double re = creal(amps[i]), im = cimag(amps[i]);
        prob_le_out[bitrev(i, n)] = re * re + im * im;   /* big-endian -> LE */
    }
    free(amps);
    tn_mps_free(mps);
    return 0;
}

/* ================================================================= */
/*  Clifford backend (clifford-only cases; exact stabilizer obs)      */
/* ================================================================= */

/* <psi|P|psi> for stabilizer state |psi>=C|0^n>, P diagonal (Z on `zmask`),
 * via <0^n| C^dagger P C |0^n>: conjugate P through C^-1 and read the
 * expectation off |0^n>. Exact: +-1 if the conjugate is diagonal, else 0. */
static double cliff_expect_diag(const clifford_tableau_t *t, int n, const uint8_t *zmask) {
    uint8_t in[64], out[64];
    int ph = 0;
    memcpy(in, zmask, (size_t)n);
    clifford_conjugate_pauli_inverse(t, in, 0, out, &ph);
    for (int k = 0; k < n; k++)
        if (out[k] == 1 || out[k] == 2) return 0.0;   /* X or Y -> flips |0^n> */
    if (ph == 0) return 1.0;
    if (ph == 2) return -1.0;
    return 0.0;
}

static int cliff_run(const corpus_case_t *cc, double *z_out, double *zz_out) {
    int n = cc->num_qubits;
    clifford_tableau_t *t = clifford_tableau_create((size_t)n);
    if (!t) return -1;
    for (int g = 0; g < cc->ngates; g++) {
        const dgate_t *G = &cc->gates[g];
        const char *nm = G->name;
        clifford_error_t rc = CLIFFORD_SUCCESS;
        if      (!strcmp(nm, "h"))   rc = clifford_h(t, (size_t)G->q0);
        else if (!strcmp(nm, "x"))   rc = clifford_x(t, (size_t)G->q0);
        else if (!strcmp(nm, "y"))   rc = clifford_y(t, (size_t)G->q0);
        else if (!strcmp(nm, "z"))   rc = clifford_z(t, (size_t)G->q0);
        else if (!strcmp(nm, "s"))   rc = clifford_s(t, (size_t)G->q0);
        else if (!strcmp(nm, "sdg")) rc = clifford_s_dag(t, (size_t)G->q0);
        else if (!strcmp(nm, "cx"))  rc = clifford_cnot(t, (size_t)G->q0, (size_t)G->q1);
        else if (!strcmp(nm, "cz"))  rc = clifford_cz(t, (size_t)G->q0, (size_t)G->q1);
        else if (!strcmp(nm, "swap"))rc = clifford_swap(t, (size_t)G->q0, (size_t)G->q1);
        else { fprintf(stderr, "cliff: non-Clifford gate %s in clifford-only case %s\n",
                       nm, cc->id); clifford_tableau_free(t); return -1; }
        if (rc != CLIFFORD_SUCCESS) { fprintf(stderr, "cliff: gate %s rc=%d\n", nm, rc);
                                      clifford_tableau_free(t); return -1; }
    }
    uint8_t zmask[64];
    for (int q = 0; q < n; q++) {
        memset(zmask, 0, (size_t)n);
        zmask[q] = 3;   /* Z */
        z_out[q] = cliff_expect_diag(t, n, zmask);
    }
    for (int k = 0; k < cc->nzz; k++) {
        memset(zmask, 0, (size_t)n);
        zmask[cc->zz_a[k]] = 3;
        zmask[cc->zz_b[k]] = 3;
        zz_out[k] = cliff_expect_diag(t, n, zmask);
    }
    clifford_tableau_free(t);
    return 0;
}

/* ================================================================= */
/*  GPU backend (skip cleanly when unavailable)                       */
/* ================================================================= */

static int g_gpu_enabled = 0;
static int g_gpu_available = -1;   /* -1 unknown, 0 no, 1 yes */

static int gpu_probe(void) {
    if (g_gpu_available >= 0) return g_gpu_available;
    quantum_state_t *g = NULL;
    qs_error_t rc = quantum_state_create_gpu(1, &g);
    if (rc == QS_SUCCESS && g) {
        quantum_state_destroy(g);
        g_gpu_available = 1;
        snprintf(gpu_reason, sizeof gpu_reason, "available");
    } else {
        g_gpu_available = 0;
        snprintf(gpu_reason, sizeof gpu_reason,
                 "quantum_state_create_gpu rc=%d (QS_ERROR_NOT_SUPPORTED=-7)", (int)rc);
    }
    return g_gpu_available;
}

static int gpu_run(const corpus_case_t *cc, double *prob_out) {
    quantum_state_t *g = NULL;
    if (quantum_state_create_gpu((size_t)cc->num_qubits, &g) != QS_SUCCESS || !g) return -1;
    for (int i = 0; i < cc->ngates; i++) {
        const dgate_t *G = &cc->gates[i];
        const char *nm = G->name;
        int rc = 0;
        if      (!strcmp(nm, "h"))   rc = gate_hadamard(g, G->q0);
        else if (!strcmp(nm, "x"))   rc = gate_pauli_x(g, G->q0);
        else if (!strcmp(nm, "y"))   rc = gate_pauli_y(g, G->q0);
        else if (!strcmp(nm, "z"))   rc = gate_pauli_z(g, G->q0);
        else if (!strcmp(nm, "s"))   rc = gate_s(g, G->q0);
        else if (!strcmp(nm, "sdg")) rc = gate_s_dagger(g, G->q0);
        else if (!strcmp(nm, "t"))   rc = gate_t(g, G->q0);
        else if (!strcmp(nm, "tdg")) rc = gate_t_dagger(g, G->q0);
        else if (!strcmp(nm, "rx"))  rc = gate_rx(g, G->q0, G->angle);
        else if (!strcmp(nm, "ry"))  rc = gate_ry(g, G->q0, G->angle);
        else if (!strcmp(nm, "rz"))  rc = gate_rz(g, G->q0, G->angle);
        else if (!strcmp(nm, "p"))   rc = gate_phase(g, G->q0, G->angle);
        else if (!strcmp(nm, "cx"))  rc = gate_cnot(g, G->q0, G->q1);
        else if (!strcmp(nm, "cz"))  rc = gate_cz(g, G->q0, G->q1);
        else if (!strcmp(nm, "swap"))rc = gate_swap(g, G->q0, G->q1);
        else if (!strcmp(nm, "cp"))  rc = gate_cphase(g, G->q0, G->q1, G->angle);
        else if (!strcmp(nm, "ccx")) rc = gate_toffoli(g, G->q0, G->q1, G->q2);
        else { quantum_state_destroy(g); return -1; }
        if (rc != 0) { quantum_state_destroy(g); return -1; }
    }
    quantum_state_sync_to_host(g);
    for (size_t i = 0; i < cc->dim; i++)
        prob_out[i] = quantum_state_get_probability(g, i);
    quantum_state_destroy(g);
    return 0;
}

/* ================================================================= */
/*  Per-case runner                                                  */
/* ================================================================= */

static void run_case(const corpus_case_t *cc) {
    int n = cc->num_qubits;
    size_t dim = cc->dim;
    if (g_verbose) { fprintf(stderr, "[start %s ngates=%d]\n", cc->id, cc->ngates); fflush(stderr); }
    double *pd = malloc(dim * sizeof(double));   /* dense probs */
    double *pt = malloc(dim * sizeof(double));   /* tn probs (LE) */
    double *pg = malloc(dim * sizeof(double));   /* gpu probs */
    double zbuf[64], zzbuf[MAX_ZZ];

    /* --- dense (the primary absolute anchor) --- */
    if (dense_run(cc, pd) != 0) {
        fprintf(stderr, "  FAIL  case=%s dense backend errored\n", cc->id);
        c_reference.failed++;
        free(pd); free(pt); free(pg);
        return;
    }
    ex_dense++;
    /* dense vs reference: full prob vector */
    double dev = max_abs_dev(pd, cc->ref_prob, dim);
    record(&c_reference, dev <= TOL_FP, dev, TOL_FP, cc, "dense", "prob", "reference");
    /* dense vs reference: expectations */
    exp_from_prob(pd, n, dim, zbuf, cc->zz_a, cc->zz_b, cc->nzz, zzbuf);
    dev = max_abs_dev(zbuf, cc->ref_z, (size_t)n);
    record(&c_reference, dev <= TOL_FP, dev, TOL_FP, cc, "dense", "expZ", "reference");
    dev = max_abs_dev(zzbuf, cc->zz_v, (size_t)cc->nzz);
    record(&c_reference, dev <= TOL_FP, dev, TOL_FP, cc, "dense", "expZZ", "reference");

    /* --- tn_mps --- */
    if (is_quarantined(cc->id, "tn_mps")) {
        /* Known tn-lane divergence (reversed-CNOT transpose / Toffoli
         * decomposition). The buggy 2q path writes out of bounds and can poison
         * later cases' heap state, so we DO NOT execute it -- skipping the leg
         * keeps the rest of the run deterministic. Counted as quarantined, never
         * as a pass. Mirrors the 1 cross + 3 reference checks a live tn leg makes. */
        c_cross.quarantined     += 1;
        c_reference.quarantined += 3;
        if (g_verbose) fprintf(stderr, "  QUARANTINE-SKIP %s [tn_mps] (tn-lane)\n", cc->id);
    } else if (n > g_tn_max_n) {
        /* The exact tn_mps leg is O(chi^3) per gate with chi = 2^floor(n/2);
         * it is bounded by --tn-max-n for wall-clock (dense-vs-reference remains
         * the absolute anchor at every n). Recorded as a skip, never silent. */
        ex_tn_skipped++;
        c_cross.skipped++;
        c_reference.skipped++;
    } else if (tn_run(cc, pt) == 0) {
        ex_tn++;
        double dref = max_abs_dev(pt, cc->ref_prob, dim);
        record(&c_reference, dref <= TOL_FP, dref, TOL_FP, cc, "tn_mps", "prob", "reference");
        double dden = max_abs_dev(pt, pd, dim);
        record(&c_cross, dden <= TOL_FP, dden, TOL_FP, cc, "tn_mps", "prob", "dense");
        exp_from_prob(pt, n, dim, zbuf, cc->zz_a, cc->zz_b, cc->nzz, zzbuf);
        double dz = max_abs_dev(zbuf, cc->ref_z, (size_t)n);
        record(&c_reference, dz <= TOL_FP, dz, TOL_FP, cc, "tn_mps", "expZ", "reference");
        double dzz = max_abs_dev(zzbuf, cc->zz_v, (size_t)cc->nzz);
        record(&c_reference, dzz <= TOL_FP, dzz, TOL_FP, cc, "tn_mps", "expZZ", "reference");
    } else {
        fprintf(stderr, "  FAIL  case=%s tn_mps backend errored\n", cc->id);
        if (is_quarantined(cc->id, "tn_mps")) c_cross.quarantined++; else c_cross.failed++;
    }

    /* --- Clifford (clifford-only cases): exact stabilizer observables --- */
    if (cc->clifford_only) {
        double czb[64], czzb[MAX_ZZ];
        if (cliff_run(cc, czb, czzb) == 0) {
            ex_cliff++;
            double dz = max_abs_dev(czb, cc->ref_z, (size_t)n);
            record(&c_reference, dz <= TOL_CLIFF, dz, TOL_CLIFF, cc, "clifford", "expZ", "reference");
            double dzz = max_abs_dev(czzb, cc->zz_v, (size_t)cc->nzz);
            record(&c_reference, dzz <= TOL_CLIFF, dzz, TOL_CLIFF, cc, "clifford", "expZZ", "reference");
            /* cross-check against dense's expectations too */
            exp_from_prob(pd, n, dim, zbuf, cc->zz_a, cc->zz_b, cc->nzz, zzbuf);
            double cdz = max_abs_dev(czb, zbuf, (size_t)n);
            record(&c_cross, cdz <= TOL_CLIFF, cdz, TOL_CLIFF, cc, "clifford", "expZ", "dense");
        } else {
            fprintf(stderr, "  FAIL  case=%s clifford backend errored\n", cc->id);
            if (is_quarantined(cc->id, "clifford")) c_reference.quarantined++; else c_reference.failed++;
        }
    }

    /* --- GPU (optional) --- */
    if (g_gpu_enabled && gpu_probe() == 1) {
        if (gpu_run(cc, pg) == 0) {
            ex_gpu++;
            double dg = max_abs_dev(pg, pd, dim);
            record(&c_cross, dg <= TOL_FP, dg, TOL_FP, cc, "gpu", "prob", "dense");
            double dgr = max_abs_dev(pg, cc->ref_prob, dim);
            record(&c_reference, dgr <= TOL_FP, dgr, TOL_FP, cc, "gpu", "prob", "reference");
        } else {
            fprintf(stderr, "  FAIL  case=%s gpu backend errored\n", cc->id);
            if (is_quarantined(cc->id, "gpu")) c_cross.quarantined++; else c_cross.failed++;
        }
    }

    if (g_verbose)
        fprintf(stdout, "  ok    %s (n=%d, %s)\n", cc->id, n, cc->cls);
    free(pd); free(pt); free(pg);
}

/* ================================================================= */
/*  Corpus + quarantine parsing                                      */
/* ================================================================= */

static int load_quarantine(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;   /* absent quarantine file = nothing quarantined */
    char line[512];
    int cap = 16;
    g_quar = malloc(sizeof(quar_t) * cap);
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        char id[96], backend[16], lane[32];
        int got = sscanf(p, "%95s %15s %31s", id, backend, lane);
        if (got < 2) continue;
        if (got < 3) strcpy(lane, "unknown");
        if (g_nquar == cap) { cap *= 2; g_quar = realloc(g_quar, sizeof(quar_t) * cap); }
        strncpy(g_quar[g_nquar].id, id, sizeof g_quar[g_nquar].id - 1);
        g_quar[g_nquar].id[sizeof g_quar[g_nquar].id - 1] = '\0';
        strncpy(g_quar[g_nquar].backend, backend, sizeof g_quar[g_nquar].backend - 1);
        g_quar[g_nquar].backend[sizeof g_quar[g_nquar].backend - 1] = '\0';
        strncpy(g_quar[g_nquar].lane, lane, sizeof g_quar[g_nquar].lane - 1);
        g_quar[g_nquar].lane[sizeof g_quar[g_nquar].lane - 1] = '\0';
        g_nquar++;
    }
    fclose(f);
    return g_nquar;
}

static int expect_tok(FILE *f, const char *want) {
    char tok[32];
    if (fscanf(f, "%31s", tok) != 1) return 0;
    return strcmp(tok, want) == 0;
}

static void free_case(corpus_case_t *cc) {
    free(cc->gates); free(cc->ref_prob); free(cc->ref_z);
    cc->gates = NULL; cc->ref_prob = NULL; cc->ref_z = NULL;
}

/* Parse one CASE ... ENDCASE block (CASE token already consumed). */
static int parse_case(FILE *f, corpus_case_t *cc) {
    memset(cc, 0, sizeof *cc);
    int cliff;
    if (fscanf(f, "%95s %31s %d %d %d %llu %d",
               cc->id, cc->cls, &cc->num_qubits, &cc->depth, &cliff,
               (unsigned long long *)&cc->seed, &cc->ngates) != 7) {
        fprintf(stderr, "corpus: bad CASE header\n"); return -1;
    }
    cc->clifford_only = cliff;
    cc->dim = (size_t)1 << cc->num_qubits;
    cc->gates = malloc(sizeof(dgate_t) * (cc->ngates > 0 ? cc->ngates : 1));
    for (int g = 0; g < cc->ngates; g++) {
        if (!expect_tok(f, "G")) { fprintf(stderr, "corpus: expected G\n"); return -1; }
        if (fscanf(f, "%15s %d %d %d %lf", cc->gates[g].name,
                   &cc->gates[g].q0, &cc->gates[g].q1, &cc->gates[g].q2,
                   &cc->gates[g].angle) != 5) {
            fprintf(stderr, "corpus: bad gate line\n"); return -1;
        }
    }
    /* PROB */
    if (!expect_tok(f, "PROB")) { fprintf(stderr, "corpus: expected PROB\n"); return -1; }
    int np; if (fscanf(f, "%d", &np) != 1 || (size_t)np != cc->dim) {
        fprintf(stderr, "corpus: PROB count %d != dim %zu\n", np, cc->dim); return -1; }
    cc->ref_prob = malloc(sizeof(double) * cc->dim);
    for (size_t i = 0; i < cc->dim; i++)
        if (fscanf(f, "%lf", &cc->ref_prob[i]) != 1) { fprintf(stderr, "corpus: bad prob\n"); return -1; }
    /* EXPZ */
    if (!expect_tok(f, "EXPZ")) { fprintf(stderr, "corpus: expected EXPZ\n"); return -1; }
    int nz; if (fscanf(f, "%d", &nz) != 1 || nz != cc->num_qubits) {
        fprintf(stderr, "corpus: EXPZ count mismatch\n"); return -1; }
    cc->ref_z = malloc(sizeof(double) * cc->num_qubits);
    for (int i = 0; i < nz; i++)
        if (fscanf(f, "%lf", &cc->ref_z[i]) != 1) { fprintf(stderr, "corpus: bad expz\n"); return -1; }
    /* EXPZZ */
    if (!expect_tok(f, "EXPZZ")) { fprintf(stderr, "corpus: expected EXPZZ\n"); return -1; }
    if (fscanf(f, "%d", &cc->nzz) != 1 || cc->nzz > MAX_ZZ) {
        fprintf(stderr, "corpus: EXPZZ count %d too large\n", cc->nzz); return -1; }
    for (int k = 0; k < cc->nzz; k++)
        if (fscanf(f, "%d %d %lf", &cc->zz_a[k], &cc->zz_b[k], &cc->zz_v[k]) != 3) {
            fprintf(stderr, "corpus: bad expzz\n"); return -1; }
    if (!expect_tok(f, "ENDCASE")) { fprintf(stderr, "corpus: expected ENDCASE\n"); return -1; }
    return 0;
}

static int run_corpus(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open corpus %s\n", path); return 2; }
    char tok[32];
    int ver; unsigned long long seed; int ncases;
    if (fscanf(f, "%31s %d %llu %d", tok, &ver, &seed, &ncases) != 4 || strcmp(tok, "CORPUS")) {
        fprintf(stderr, "corpus: bad header\n"); fclose(f); return 2;
    }
    fprintf(stdout, "=== cross-backend differential: %d cases (corpus seed=%llu) ===\n",
            ncases, seed);
    if (g_gpu_enabled) { gpu_probe(); fprintf(stdout, "GPU: %s\n", gpu_reason); }
    else                 snprintf(gpu_reason, sizeof gpu_reason, "not enabled (--gpu to attempt)");

    int parsed = 0;
    while (fscanf(f, "%31s", tok) == 1) {
        if (!strcmp(tok, "END")) break;
        if (strcmp(tok, "CASE")) { fprintf(stderr, "corpus: unexpected token %s\n", tok); fclose(f); return 2; }
        corpus_case_t cc;
        if (parse_case(f, &cc) != 0) { fclose(f); return 2; }
        run_case(&cc);
        free_case(&cc);
        parsed++;
    }
    fclose(f);

    /* Machine-readable summary for run_cross_diff.sh */
    long cross_fail = c_cross.failed, ref_fail = c_reference.failed;
    const char *cross_status = cross_fail ? "FAIL" : "PASS";
    const char *ref_status   = ref_fail   ? "FAIL" : "PASS";
    printf("\nDIFF_RESULT name=cross_backend_differential status=%s checks=%ld failed=%ld quarantined=%ld skipped=%ld\n",
           cross_status, c_cross.passed, c_cross.failed, c_cross.quarantined, c_cross.skipped);
    printf("DIFF_RESULT name=reference_oracle_agreement status=%s checks=%ld failed=%ld quarantined=%ld skipped=%ld\n",
           ref_status, c_reference.passed, c_reference.failed, c_reference.quarantined, c_reference.skipped);
    printf("DIFF_EXERCISED backend=dense cases=%ld\n", ex_dense);
    printf("DIFF_EXERCISED backend=tn_mps cases=%ld skipped=%ld tn_max_n=%d\n",
           ex_tn, ex_tn_skipped, g_tn_max_n);
    printf("DIFF_EXERCISED backend=clifford cases=%ld\n", ex_cliff);
    printf("DIFF_EXERCISED backend=gpu cases=%ld reason=\"%s\"\n", ex_gpu, gpu_reason);
    printf("DIFF_CASES parsed=%d\n", parsed);

    fprintf(stdout, "\n=== cross-backend: %s (%ld checks, %ld failed, %ld quarantined) | "
                    "reference: %s (%ld checks, %ld failed, %ld quarantined) ===\n",
            cross_status, c_cross.passed + c_cross.failed + c_cross.quarantined, c_cross.failed, c_cross.quarantined,
            ref_status, c_reference.passed + c_reference.failed + c_reference.quarantined, c_reference.failed, c_reference.quarantined);

    return (cross_fail || ref_fail) ? 1 : 0;
}

/* ================================================================= */
/*  Self-test: prove a corrupted reference probability is caught      */
/* ================================================================= */

static int self_test(void) {
    /* Build a Bell state on the dense backend; its exact probs are
     * [0.5, 0, 0, 0.5] (little-endian). Confirm the comparator (a) accepts the
     * correct reference within 1e-10 and (b) REJECTS a reference with one
     * probability corrupted by 1e-3. This proves the oracle catches a wrong
     * reference rather than rubber-stamping the backend. */
    quantum_state_t *sv = quantum_state_create(2);
    if (!sv) { fprintf(stderr, "selftest: alloc failed\n"); return 1; }
    gate_hadamard(sv, 0);
    gate_cnot(sv, 0, 1);
    double got[4];
    for (int i = 0; i < 4; i++) got[i] = quantum_state_get_probability(sv, i);
    quantum_state_free(sv);

    double good[4] = {0.5, 0.0, 0.0, 0.5};
    double dev_good = max_abs_dev(got, good, 4);

    double bad[4] = {0.5, 0.0, 0.0, 0.5};
    bad[0] += 1e-3;   /* deliberate corruption */
    double dev_bad = max_abs_dev(got, bad, 4);

    int clean_ok  = dev_good <= TOL_FP;    /* correct ref accepted */
    int caught    = dev_bad  >  TOL_FP;    /* corrupted ref rejected */

    printf("selftest: clean_dev=%.3e (accepted=%d), corrupted_dev=%.3e (caught=%d)\n",
           dev_good, clean_ok, dev_bad, caught);
    if (clean_ok && caught) {
        printf("DIFF_SELFTEST status=PASS "
               "(reference oracle accepts correct probs, catches a 1e-3 corruption at tol=%.0e)\n",
               TOL_FP);
        return 0;
    }
    printf("DIFF_SELFTEST status=FAIL\n");
    return 1;
}

/* ================================================================= */

int main(int argc, char **argv) {
    const char *corpus = NULL;
    const char *quar = NULL;
    int selftest = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--selftest")) selftest = 1;
        else if (!strcmp(argv[i], "--gpu"))      g_gpu_enabled = 1;
        else if (!strcmp(argv[i], "--verbose"))  g_verbose = 1;
        else if (!strcmp(argv[i], "--tn-max-n") && i + 1 < argc) g_tn_max_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quarantine") && i + 1 < argc) quar = argv[++i];
        else if (argv[i][0] != '-') corpus = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (selftest) return self_test();
    if (!corpus) {
        fprintf(stderr, "usage: %s <corpus.txt> [--quarantine FILE] [--gpu] [--tn-max-n N] [--verbose]\n", argv[0]);
        fprintf(stderr, "       %s --selftest\n", argv[0]);
        return 2;
    }
    if (quar) {
        int nq = load_quarantine(quar);
        fprintf(stdout, "quarantine: %d (case,backend) entr%s from %s\n",
                nq, nq == 1 ? "y" : "ies", quar);
    }
    int rc = run_corpus(corpus);
    free(g_quar);
    return rc;
}
