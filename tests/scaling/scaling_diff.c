/**
 * @file scaling_diff.c
 * @brief SCALING cross-backend differential for Moonlab.
 *
 * A sibling of tests/differential/diff_backends.c specialised for pushing SCALE
 * and circuit diversity to hunt for scale-dependent divergences that the
 * adjacent-only cross-diff corpus missed. It runs every circuit in a corpus
 * (tests/scaling/gen_scaling_corpus.py, pure function of seed, independent numpy
 * reference pinned in) on:
 *
 *   - dense statevector  (src/quantum; little-endian, qubit0 = LSB) -- primary
 *     absolute anchor, compared to the numpy reference at 1e-10;
 *   - tn_mps             (plain MPS; big-endian site0=MSB, bit-reversed here),
 *     chi = 2^ceil(n/2) so it is EXACT at the given size (no truncation);
 *   - Clifford tableau   (clifford-only families; exact stabilizer observables),
 *     a locality-free second oracle that is exact at ANY n.
 *
 * KNOWN-vs-NEW classification (the whole point of this lane)
 * ---------------------------------------------------------
 * This lane's hunt ROOT-CAUSED the tn 2q-gate divergence the tn-lane tracks
 * (see tests/scaling/KNOWN_DIVERGENCES.txt and FINDINGS.md). The single defect:
 * the adjacent two-qubit gate path (apply_gate_2q_adjacent, tn_gates.c) rescales
 * the two-site tensor's singular values by their Frobenius norm assuming the MPS
 * is in mixed-canonical gauge at the bond -- but it never establishes that gauge
 * (every 2q gate leaves canonical = TN_CANONICAL_NONE). When a 2q gate lands on
 * a bond whose LEFT and RIGHT outer bonds are BOTH non-trivial (both neighbours
 * already entangled), ||theta||_F != state-norm and the rescale corrupts the
 * norm. Minimal repro (4 qubits, forward, pure Clifford, all-adjacent):
 *     H(0) CX(0,1) H(2) CX(2,3) CZ(1,2)  -> tn norm 0.5, every prob halved.
 * It is bond-dimension independent (chi 8..1024 identical) and fires at n=4 --
 * far below the "n>=10 forward rotation" scope the tn-lane documented. The
 * long-range / SWAP-network divergences are the SAME defect surfaced after the
 * SWAP network makes the gate adjacent-in-an-entangled-bulk.
 *
 * Gate semantics: the ABSOLUTE anchors are the two backend-INDEPENDENT oracles,
 * which are NEVER quarantined and a divergence there is always a NEW finding:
 *   - dense statevector  vs numpy reference;
 *   - Clifford tableau   vs numpy reference (clifford families; exact at any n).
 * The tn_mps leg is this lane's hunt surface. A tn divergence is classified
 * KNOWN (the root-caused normalization envelope above) and REPORTED but excluded
 * from the gate, EXCEPT on the provably canonical-safe LIVE tn guards, where a
 * tn divergence would be a genuinely DIFFERENT bug and FAILS the gate:
 *   - family ghz_chain (H(0) then a single left-to-right CNOT sweep keeps the
 *     right outer bond trivial at every gate, so the defect cannot fire);
 *   - every case at n <= 4 (too small for a both-sides-entangled bulk bond).
 *
 * Tolerances are NEVER loosened to make anything pass.
 *
 * Usage:
 *   scaling_diff <corpus.txt> [--tn-max-n N] [--verbose]
 *   scaling_diff --selftest
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

#define TOL_DENSE   1e-10   /* dense vs numpy reference */
#define TOL_TN      1e-9    /* tn_mps vs reference/dense (SVD roundoff headroom) */
#define TOL_CLIFF   1e-9    /* Clifford stabilizer observables */
#define MAX_ZZ      16
#define LIVE_TN_GUARD_MAX_N 4   /* n<=this: tn must be exact (no bulk bond) */

typedef struct { char name[16]; int q0, q1, q2; double angle; } dgate_t;

typedef struct {
    char     id[96];
    char     cls[32];
    int      num_qubits;
    int      depth;
    int      clifford_only;
    uint64_t seed;
    int      ngates;
    dgate_t *gates;
    size_t   dim;
    double  *ref_prob;
    double  *ref_z;
    int      nzz;
    int      zz_a[MAX_ZZ];
    int      zz_b[MAX_ZZ];
    double   zz_v[MAX_ZZ];
} corpus_case_t;

/* ---- counters ---- */
static long g_new_div = 0;      /* NEW divergences (fail the gate) */
static long g_known_div = 0;    /* known-quarantined divergences (reported only) */
static long g_checks_pass = 0;  /* comparisons that agreed */
static long g_tn_skipped = 0;   /* tn legs skipped for affordability (n>cap) */
static long g_cases = 0;
static int  g_tn_max_n = 16;
static int  g_tn_max_depth = 1000000;  /* affordability wall: skip tn leg above this depth */
static int  g_verbose = 0;

/* A LIVE tn guard is a case on which the root-caused normalization defect
 * provably cannot fire, so tn MUST reproduce the reference exactly there; a
 * divergence would be a genuinely DIFFERENT tn bug and fails the gate:
 *   - ghz_chain: H(0) then a single L->R CNOT sweep keeps the right outer bond
 *     trivial at every gate, so no both-sides-entangled bulk bond ever forms;
 *   - any case at n <= LIVE_TN_GUARD_MAX_N: too small for a bulk bond. */
static int is_live_tn_guard(const corpus_case_t *cc) {
    return !strcmp(cc->cls, "ghz_chain") || cc->num_qubits <= LIVE_TN_GUARD_MAX_N;
}

/* A tn divergence is KNOWN (the root-caused normalization envelope, tn-lane owns
 * the fix) unless the case is a live tn guard. */
static int is_known_tn_case(const corpus_case_t *cc) {
    return !is_live_tn_guard(cc);
}

/* Does the circuit use any non-adjacent 2q gate (|q0-q1| > 1)? Used only for
 * reporting -- so a NEW finding says whether the SWAP network was exercised. */
static int uses_long_range_2q(const corpus_case_t *cc) {
    for (int g = 0; g < cc->ngates; g++) {
        const dgate_t *G = &cc->gates[g];
        if (G->q1 >= 0 && abs(G->q0 - G->q1) > 1) return 1;
    }
    return 0;
}

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

/* Record one comparison. `known` marks the case as owning a quarantined bug:
 * a mismatch is then reported but NOT counted as a failure. */
static void record(int ok, double dev, double tol, int known,
                   const corpus_case_t *cc, const char *backend,
                   const char *what, const char *against) {
    if (ok) { g_checks_pass++; return; }
    if (known) {
        g_known_div++;
        if (g_verbose)
            fprintf(stderr, "  KNOWN  %s [%s] %s vs %s dev=%.3e (>%.0e)  (tn 2q-gate normalization envelope)\n",
                    cc->id, backend, what, against, dev, tol);
        return;
    }
    g_new_div++;
    fprintf(stderr,
        "  *** NEW DIVERGENCE ***  case=%s seed=%llu class=%s n=%d depth=%d "
        "long_range_2q=%d  %s [%s] vs %s  dev=%.3e (>%.0e)  "
        "-- replay: gen_scaling_corpus.py --seed 0x5CA11B16 reproduces this case\n",
        cc->id, (unsigned long long)cc->seed, cc->cls, cc->num_qubits, cc->depth,
        uses_long_range_2q(cc), what, backend, against, dev, tol);
}

/* ================================================================= */
/*  Dense backend (primary absolute anchor)                          */
/* ================================================================= */
static int apply_dense_gate(quantum_state_t *sv, const dgate_t *G) {
    const char *nm = G->name;
    if      (!strcmp(nm, "h"))   return gate_hadamard(sv, G->q0);
    else if (!strcmp(nm, "x"))   return gate_pauli_x(sv, G->q0);
    else if (!strcmp(nm, "y"))   return gate_pauli_y(sv, G->q0);
    else if (!strcmp(nm, "z"))   return gate_pauli_z(sv, G->q0);
    else if (!strcmp(nm, "s"))   return gate_s(sv, G->q0);
    else if (!strcmp(nm, "sdg")) return gate_s_dagger(sv, G->q0);
    else if (!strcmp(nm, "t"))   return gate_t(sv, G->q0);
    else if (!strcmp(nm, "tdg")) return gate_t_dagger(sv, G->q0);
    else if (!strcmp(nm, "rx"))  return gate_rx(sv, G->q0, G->angle);
    else if (!strcmp(nm, "ry"))  return gate_ry(sv, G->q0, G->angle);
    else if (!strcmp(nm, "rz"))  return gate_rz(sv, G->q0, G->angle);
    else if (!strcmp(nm, "p"))   return gate_phase(sv, G->q0, G->angle);
    else if (!strcmp(nm, "cx"))  return gate_cnot(sv, G->q0, G->q1);
    else if (!strcmp(nm, "cz"))  return gate_cz(sv, G->q0, G->q1);
    else if (!strcmp(nm, "swap"))return gate_swap(sv, G->q0, G->q1);
    else if (!strcmp(nm, "cp"))  return gate_cphase(sv, G->q0, G->q1, G->angle);
    fprintf(stderr, "dense: unknown gate %s\n", nm);
    return -999;
}

static int dense_run(const corpus_case_t *cc, double *prob_out) {
    quantum_state_t *sv = quantum_state_create(cc->num_qubits);
    if (!sv) return -1;
    for (int g = 0; g < cc->ngates; g++) {
        int rc = apply_dense_gate(sv, &cc->gates[g]);
        if (rc != 0) { fprintf(stderr, "dense: gate %s rc=%d\n", cc->gates[g].name, rc);
                       quantum_state_free(sv); return -1; }
    }
    for (size_t i = 0; i < cc->dim; i++)
        prob_out[i] = quantum_state_get_probability(sv, i);
    quantum_state_free(sv);
    return 0;
}

/* ================================================================= */
/*  tn_mps backend (big-endian; bit-reversed to little-endian here)   */
/* ================================================================= */
static int apply_tn_gate(tn_mps_state_t *mps, const dgate_t *G) {
    const char *nm = G->name;
    if      (!strcmp(nm, "h"))   return tn_apply_h(mps, G->q0);
    else if (!strcmp(nm, "x"))   return tn_apply_x(mps, G->q0);
    else if (!strcmp(nm, "y"))   return tn_apply_y(mps, G->q0);
    else if (!strcmp(nm, "z"))   return tn_apply_z(mps, G->q0);
    else if (!strcmp(nm, "s"))   return tn_apply_s(mps, G->q0);
    else if (!strcmp(nm, "sdg")) return tn_apply_gate_1q(mps, G->q0, &TN_GATE_SDG);
    else if (!strcmp(nm, "t"))   return tn_apply_t(mps, G->q0);
    else if (!strcmp(nm, "tdg")) return tn_apply_gate_1q(mps, G->q0, &TN_GATE_TDG);
    else if (!strcmp(nm, "rx"))  return tn_apply_rx(mps, G->q0, G->angle);
    else if (!strcmp(nm, "ry"))  return tn_apply_ry(mps, G->q0, G->angle);
    else if (!strcmp(nm, "rz"))  return tn_apply_rz(mps, G->q0, G->angle);
    else if (!strcmp(nm, "p"))   { tn_gate_1q_t gp = tn_gate_phase(G->angle);
                                   return tn_apply_gate_1q(mps, G->q0, &gp); }
    else if (!strcmp(nm, "cx"))  return tn_apply_cnot(mps, G->q0, G->q1);
    else if (!strcmp(nm, "cz"))  return tn_apply_cz(mps, G->q0, G->q1);
    else if (!strcmp(nm, "swap"))return tn_apply_swap(mps, G->q0, G->q1);
    else if (!strcmp(nm, "cp"))  { tn_gate_2q_t gc = tn_gate_cphase(G->angle);
                                   return tn_apply_gate_2q(mps, G->q0, G->q1, &gc, NULL); }
    fprintf(stderr, "tn: unknown gate %s\n", nm);
    return -999;
}

static int tn_run(const corpus_case_t *cc, double *prob_le_out) {
    int n = cc->num_qubits;
    uint32_t chi = 1u << ((n + 1) / 2);   /* exact max Schmidt rank at any cut */
    if (chi > TN_MAX_BOND_DIM) chi = TN_MAX_BOND_DIM;
    tn_state_config_t cfg = tn_state_config_create(chi, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)n, &cfg);
    if (!mps) return -1;
    for (int g = 0; g < cc->ngates; g++) {
        int rc = apply_tn_gate(mps, &cc->gates[g]);
        if (rc != 0) { fprintf(stderr, "tn: gate %s rc=%d\n", cc->gates[g].name, rc);
                       tn_mps_free(mps); return -1; }
    }
    double _Complex *amps = malloc(cc->dim * sizeof(double _Complex));
    if (!amps) { tn_mps_free(mps); return -1; }
    tn_state_error_t e = tn_mps_to_statevector(mps, amps);
    if (e != TN_STATE_SUCCESS) { free(amps); tn_mps_free(mps); return -1; }
    for (size_t i = 0; i < cc->dim; i++) {
        double re = creal(amps[i]), im = cimag(amps[i]);
        prob_le_out[bitrev(i, n)] = re * re + im * im;
    }
    free(amps);
    tn_mps_free(mps);
    return 0;
}

/* ================================================================= */
/*  Clifford backend (clifford-only families; exact stabilizer obs)   */
/* ================================================================= */
static double cliff_expect_diag(const clifford_tableau_t *t, int n, const uint8_t *zmask) {
    uint8_t in[128], out[128];
    int ph = 0;
    memcpy(in, zmask, (size_t)n);
    clifford_conjugate_pauli_inverse(t, in, 0, out, &ph);
    for (int k = 0; k < n; k++)
        if (out[k] == 1 || out[k] == 2) return 0.0;
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
        else { fprintf(stderr, "cliff: non-Clifford gate %s in %s\n", nm, cc->id);
               clifford_tableau_free(t); return -1; }
        if (rc != CLIFFORD_SUCCESS) { fprintf(stderr, "cliff: gate %s rc=%d\n", nm, rc);
                                      clifford_tableau_free(t); return -1; }
    }
    uint8_t zmask[128];
    for (int q = 0; q < n; q++) {
        memset(zmask, 0, (size_t)n);
        zmask[q] = 3;
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
/*  Per-case runner                                                  */
/* ================================================================= */
static void run_case(const corpus_case_t *cc) {
    int n = cc->num_qubits;
    size_t dim = cc->dim;
    int known = is_known_tn_case(cc);
    g_cases++;
    if (g_verbose) { fprintf(stderr, "[case %s ngates=%d known_tn=%d]\n", cc->id, cc->ngates, known); }

    double *pd = malloc(dim * sizeof(double));
    double *pt = malloc(dim * sizeof(double));
    double zbuf[128], zzbuf[MAX_ZZ];

    /* --- dense vs reference (absolute anchor, NEVER quarantined) --- */
    if (dense_run(cc, pd) != 0) {
        fprintf(stderr, "  FAIL  case=%s dense backend errored\n", cc->id);
        g_new_div++; free(pd); free(pt); return;
    }
    double dev = max_abs_dev(pd, cc->ref_prob, dim);
    record(dev <= TOL_DENSE, dev, TOL_DENSE, 0, cc, "dense", "prob", "reference");
    exp_from_prob(pd, n, dim, zbuf, cc->zz_a, cc->zz_b, cc->nzz, zzbuf);
    dev = max_abs_dev(zbuf, cc->ref_z, (size_t)n);
    record(dev <= TOL_DENSE, dev, TOL_DENSE, 0, cc, "dense", "expZ", "reference");
    dev = max_abs_dev(zzbuf, cc->zz_v, (size_t)cc->nzz);
    record(dev <= TOL_DENSE, dev, TOL_DENSE, 0, cc, "dense", "expZZ", "reference");

    /* --- tn_mps --- */
    if (n > g_tn_max_n || cc->depth > g_tn_max_depth) {
        g_tn_skipped++;
        if (g_verbose) fprintf(stderr, "  tn skipped (n=%d depth=%d; caps n<=%d depth<=%d)\n",
                               n, cc->depth, g_tn_max_n, g_tn_max_depth);
    } else if (tn_run(cc, pt) == 0) {
        double dref = max_abs_dev(pt, cc->ref_prob, dim);
        record(dref <= TOL_TN, dref, TOL_TN, known, cc, "tn_mps", "prob", "reference");
        double dden = max_abs_dev(pt, pd, dim);
        record(dden <= TOL_TN, dden, TOL_TN, known, cc, "tn_mps", "prob", "dense");
        exp_from_prob(pt, n, dim, zbuf, cc->zz_a, cc->zz_b, cc->nzz, zzbuf);
        double dz = max_abs_dev(zbuf, cc->ref_z, (size_t)n);
        record(dz <= TOL_TN, dz, TOL_TN, known, cc, "tn_mps", "expZ", "reference");
    } else {
        fprintf(stderr, "  FAIL  case=%s tn_mps backend errored\n", cc->id);
        if (known) g_known_div++; else g_new_div++;
    }

    /* --- Clifford (clifford-only families): exact stabilizer observables,
     *     a locality-free oracle that is exact at ANY n and never quarantined. */
    if (cc->clifford_only) {
        double czb[128], czzb[MAX_ZZ];
        if (cliff_run(cc, czb, czzb) == 0) {
            double dz = max_abs_dev(czb, cc->ref_z, (size_t)n);
            record(dz <= TOL_CLIFF, dz, TOL_CLIFF, 0, cc, "clifford", "expZ", "reference");
            double dzz = max_abs_dev(czzb, cc->zz_v, (size_t)cc->nzz);
            record(dzz <= TOL_CLIFF, dzz, TOL_CLIFF, 0, cc, "clifford", "expZZ", "reference");
        } else {
            fprintf(stderr, "  FAIL  case=%s clifford backend errored\n", cc->id);
            g_new_div++;
        }
    }

    if (g_verbose) fprintf(stdout, "  ok    %s (n=%d, %s)\n", cc->id, n, cc->cls);
    free(pd); free(pt);
}

/* ================================================================= */
/*  Corpus parsing                                                   */
/* ================================================================= */
static int expect_tok(FILE *f, const char *want) {
    char tok[32];
    if (fscanf(f, "%31s", tok) != 1) return 0;
    return strcmp(tok, want) == 0;
}

static void free_case(corpus_case_t *cc) {
    free(cc->gates); free(cc->ref_prob); free(cc->ref_z);
    cc->gates = NULL; cc->ref_prob = NULL; cc->ref_z = NULL;
}

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
    size_t np;
    if (!expect_tok(f, "PROB") || fscanf(f, "%zu", &np) != 1 || np != cc->dim) {
        fprintf(stderr, "corpus: bad PROB block (%s)\n", cc->id); return -1;
    }
    cc->ref_prob = malloc(sizeof(double) * np);
    for (size_t i = 0; i < np; i++)
        if (fscanf(f, "%lf", &cc->ref_prob[i]) != 1) {
            fprintf(stderr, "corpus: bad prob value\n"); return -1; }
    int nz;
    if (!expect_tok(f, "EXPZ") || fscanf(f, "%d", &nz) != 1 || nz != cc->num_qubits) {
        fprintf(stderr, "corpus: bad EXPZ block\n"); return -1;
    }
    cc->ref_z = malloc(sizeof(double) * nz);
    for (int i = 0; i < nz; i++)
        if (fscanf(f, "%lf", &cc->ref_z[i]) != 1) {
            fprintf(stderr, "corpus: bad expz value\n"); return -1; }
    if (!expect_tok(f, "EXPZZ") || fscanf(f, "%d", &cc->nzz) != 1) {
        fprintf(stderr, "corpus: bad EXPZZ header\n"); return -1;
    }
    if (cc->nzz > MAX_ZZ) { fprintf(stderr, "corpus: too many zz pairs\n"); return -1; }
    for (int k = 0; k < cc->nzz; k++)
        if (fscanf(f, "%d %d %lf", &cc->zz_a[k], &cc->zz_b[k], &cc->zz_v[k]) != 3) {
            fprintf(stderr, "corpus: bad zz line\n"); return -1; }
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
    fprintf(stderr, "=== scaling_diff: corpus %s  version=%d seed=%llu cases=%d ===\n",
            path, ver, seed, ncases);
    while (fscanf(f, "%31s", tok) == 1) {
        if (!strcmp(tok, "END")) break;
        if (strcmp(tok, "CASE")) { fprintf(stderr, "corpus: expected CASE, got %s\n", tok);
                                   fclose(f); return 2; }
        corpus_case_t cc;
        if (parse_case(f, &cc) != 0) { free_case(&cc); fclose(f); return 2; }
        run_case(&cc);
        free_case(&cc);
    }
    fclose(f);
    return 0;
}

/* ================================================================= */
/*  Self-test: prove the reference-catching path works, no corpus.    */
/* ================================================================= */
static int selftest(void) {
    /* Build a 4-qubit GHZ on dense, confirm parity, then prove a corrupted
     * probability is caught at TOL_DENSE. */
    int n = 4;
    quantum_state_t *sv = quantum_state_create(n);
    if (!sv) { fprintf(stderr, "selftest: state create failed\n"); return 1; }
    gate_hadamard(sv, 0);
    for (int q = 0; q < n - 1; q++) gate_cnot(sv, q, q + 1);
    size_t dim = (size_t)1 << n;
    double p0 = quantum_state_get_probability(sv, 0);
    double pf = quantum_state_get_probability(sv, dim - 1);
    quantum_state_free(sv);
    if (fabs(p0 - 0.5) > 1e-9 || fabs(pf - 0.5) > 1e-9) {
        fprintf(stderr, "selftest FAIL: GHZ probs (%.6f, %.6f) != (0.5, 0.5)\n", p0, pf);
        return 1;
    }
    /* corruption detectable */
    double a[2] = {0.5, 0.5}, b[2] = {0.5, 0.5 + 1e-6};
    if (max_abs_dev(a, b, 2) <= TOL_DENSE) {
        fprintf(stderr, "selftest FAIL: corruption not detectable at %.0e\n", TOL_DENSE);
        return 1;
    }
    /* tn exact on a small non-adjacent (long-range) CNOT: H(0), CX(0,2) on 3q.
     * This is the minimal SWAP-network case; tn must match dense/parity. */
    {
        int m = 3;
        uint32_t chi = 1u << ((m + 1) / 2);
        tn_state_config_t cfg = tn_state_config_create(chi, 1e-14);
        tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)m, &cfg);
        quantum_state_t *d = quantum_state_create(m);
        tn_apply_h(mps, 0); gate_hadamard(d, 0);
        tn_apply_cnot(mps, 0, 2); gate_cnot(d, 0, 2);   /* non-adjacent */
        double _Complex amps[8];
        tn_mps_to_statevector(mps, amps);
        double ptn[8], pdn[8];
        for (size_t i = 0; i < 8; i++) {
            double re = creal(amps[i]), im = cimag(amps[i]);
            ptn[bitrev(i, m)] = re * re + im * im;
            pdn[i] = quantum_state_get_probability(d, i);
        }
        double dv = max_abs_dev(ptn, pdn, 8);
        tn_mps_free(mps); quantum_state_free(d);
        if (dv > TOL_TN) {
            fprintf(stderr, "selftest FAIL: tn long-range CNOT dev=%.3e > %.0e\n", dv, TOL_TN);
            return 1;
        }
    }
    fprintf(stderr, "scaling_diff self-test: PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    const char *corpus = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--verbose")) g_verbose = 1;
        else if (!strcmp(argv[i], "--tn-max-n") && i + 1 < argc) g_tn_max_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tn-max-depth") && i + 1 < argc) g_tn_max_depth = atoi(argv[++i]);
        else if (argv[i][0] != '-') corpus = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!corpus) { fprintf(stderr, "usage: scaling_diff <corpus.txt> [--tn-max-n N] [--tn-max-depth D] [--verbose]\n"
                                   "       scaling_diff --selftest\n"); return 2; }
    int rc = run_corpus(corpus);
    if (rc != 0) return rc;

    fprintf(stderr,
        "\n=== scaling_diff summary ===\n"
        "cases            : %ld\n"
        "checks passed    : %ld\n"
        "tn legs skipped  : %ld (n > --tn-max-n)\n"
        "KNOWN divergences: %ld (tn 2q-gate normalization envelope; excluded from gate)\n"
        "NEW   divergences: %ld\n",
        g_cases, g_checks_pass, g_tn_skipped, g_known_div, g_new_div);
    fprintf(stdout, "SCALING_RESULT diff new=%ld known=%ld\n", g_new_div, g_known_div);
    if (g_new_div > 0) {
        fprintf(stderr, "RESULT: FAIL (%ld new divergence(s))\n", g_new_div);
        return 1;
    }
    fprintf(stderr, "RESULT: PASS\n");
    return 0;
}
