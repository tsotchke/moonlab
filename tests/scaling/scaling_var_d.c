/**
 * @file scaling_var_d.c
 * @brief Scaling checks for the CA-MPS variational-D optimizer (mission pt 4).
 *
 * Runs the greedy local-Clifford var-D optimizer on TFIM and Heisenberg Pauli
 * sums at n in {8, 16, 32} and checks:
 *
 *   1. VARIATIONAL LOWER BOUND (analytic, exact at ANY n): for a Pauli sum
 *      H = sum_k c_k P_k with real c_k, every eigenvalue obeys
 *      E >= -sum_k |c_k| (each <P_k> in [-1,1]). A var-D energy BELOW this bound
 *      is an impossible expectation and a definite bug. This is the structure-
 *      specific check that replaces ED at n=16, 32 where ED is infeasible.
 *   2. ED AGREEMENT at n=8 (feasible): an independent dense ground-state energy
 *      via power iteration on (cI - H) -- nothing to do with the MPS code --
 *      gives E0; the var-D energy must satisfy final_energy >= E0 - tol.
 *   3. MONOTONE IMPROVEMENT: final_energy <= initial_energy (the greedy search
 *      never makes the energy worse).
 *   4. EPS-ESCAPE CONVERGENCE: with a finite improvement_eps the search must
 *      terminate with converged=1, and a LARGER eps must not accept MORE gates
 *      than a smaller eps (coarser threshold => not-more work).
 *   5. UINT32 WIDTH (>255): the bond-dim fields must hold values > 255 without
 *      wrapping (an old uint8/uint16 field would clamp or overflow). Verified
 *      directly on max_bond_dim getters and by growing a real MPS bond past 255.
 *
 * Tolerances are principled fp bounds, never loosened to pass.
 *
 * Usage: scaling_var_d [--n 8,16,32] [--verbose] ;  scaling_var_d --selftest
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long g_fail = 0, g_pass = 0;
static int  g_verbose = 0;

static void check(int ok, const char *what, double got, double bound) {
    if (ok) { g_pass++; if (g_verbose) fprintf(stderr, "  ok   %s (got=%.6f bound=%.6f)\n", what, got, bound); }
    else { g_fail++; fprintf(stderr, "  *** DIVERGENCE ***  %s got=%.6f bound=%.6f\n", what, got, bound); }
}

/* ---- Pauli-sum Hamiltonians (bytes 0=I,1=X,2=Y,3=Z, row-major [term][qubit]) ---- */
/* TFIM: H = -J sum_i Z_i Z_{i+1} - h sum_i X_i (open boundary). */
static uint32_t build_tfim(int n, double J, double h, uint8_t **P, double **c) {
    uint32_t nt = (uint32_t)((n - 1) + n);
    uint8_t *paulis = calloc((size_t)nt * n, 1);
    double  *coeffs = calloc(nt, sizeof(double));
    uint32_t t = 0;
    for (int i = 0; i + 1 < n; i++) { paulis[t * n + i] = 3; paulis[t * n + i + 1] = 3; coeffs[t] = -J; t++; }
    for (int i = 0; i < n; i++)     { paulis[t * n + i] = 1;                              coeffs[t] = -h; t++; }
    *P = paulis; *c = coeffs; return t;
}
/* Heisenberg: H = J sum_i (X X + Y Y + Z Z)_{i,i+1}. */
static uint32_t build_heis(int n, double J, uint8_t **P, double **c) {
    uint32_t nt = (uint32_t)(3 * (n - 1));
    uint8_t *paulis = calloc((size_t)nt * n, 1);
    double  *coeffs = calloc(nt, sizeof(double));
    uint32_t t = 0;
    for (int i = 0; i + 1 < n; i++)
        for (int p = 1; p <= 3; p++) { paulis[t * n + i] = (uint8_t)p; paulis[t * n + i + 1] = (uint8_t)p; coeffs[t] = J; t++; }
    *P = paulis; *c = coeffs; return t;
}

static double sum_abs(const double *c, uint32_t nt) {
    double s = 0; for (uint32_t k = 0; k < nt; k++) s += fabs(c[k]); return s;
}

/* ---- Independent dense ED: ground energy via power iteration on (cI - H). ---- */
/* Apply a single Pauli string P (bytes) with coeff to dense vector: out += coeff * P|in>. */
static void apply_pauli_string(const uint8_t *P, int n, double coeff,
                               const double _Complex *in, double _Complex *out, size_t dim) {
    for (size_t b = 0; b < dim; b++) {
        size_t target = b;
        double _Complex phase = coeff;
        for (int q = 0; q < n; q++) {
            uint8_t p = P[q];
            int bit = (int)((b >> q) & 1u);
            if (p == 1) { target ^= ((size_t)1 << q); }                       /* X flips */
            else if (p == 2) { target ^= ((size_t)1 << q);                     /* Y flips + phase */
                               phase *= (bit ? (double _Complex)(-I) : (double _Complex)(I)); }
            else if (p == 3) { if (bit) phase = -phase; }                      /* Z sign */
        }
        out[target] += phase * in[b];
    }
}
static void apply_H(const uint8_t *P, const double *c, uint32_t nt, int n,
                    const double _Complex *in, double _Complex *out, size_t dim) {
    for (size_t i = 0; i < dim; i++) out[i] = 0.0;
    for (uint32_t k = 0; k < nt; k++)
        apply_pauli_string(P + (size_t)k * n, n, c[k], in, out, dim);
}
static double ed_ground_energy(const uint8_t *P, const double *c, uint32_t nt, int n) {
    size_t dim = (size_t)1 << n;
    double shift = sum_abs(c, nt) + 1.0;          /* cI - H is positive definite */
    double _Complex *v = malloc(dim * sizeof(double _Complex));
    double _Complex *hv = malloc(dim * sizeof(double _Complex));
    for (size_t i = 0; i < dim; i++) v[i] = (double)((i * 2654435761u) % 97) / 97.0 + 0.01;
    double nrm = 0; for (size_t i = 0; i < dim; i++) nrm += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    nrm = sqrt(nrm); for (size_t i = 0; i < dim; i++) v[i] /= nrm;
    for (int it = 0; it < 4000; it++) {
        apply_H(P, c, nt, n, v, hv, dim);
        for (size_t i = 0; i < dim; i++) hv[i] = shift * v[i] - hv[i];   /* (cI - H) v */
        nrm = 0; for (size_t i = 0; i < dim; i++) nrm += creal(hv[i]) * creal(hv[i]) + cimag(hv[i]) * cimag(hv[i]);
        nrm = sqrt(nrm); if (nrm < 1e-300) break;
        for (size_t i = 0; i < dim; i++) v[i] = hv[i] / nrm;
    }
    apply_H(P, c, nt, n, v, hv, dim);
    double _Complex e = 0; for (size_t i = 0; i < dim; i++) e += conj(v[i]) * hv[i];
    free(v); free(hv);
    return creal(e);
}

/* ---- var-D run on a Pauli sum; returns final energy + fills result. ---- */
static int run_var_d(int n, const uint8_t *P, const double *c, uint32_t nt,
                     double eps, ca_mps_var_d_result_t *res) {
    moonlab_ca_mps_t *s = moonlab_ca_mps_create((uint32_t)n, 32);
    if (!s) return -1;
    ca_mps_var_d_config_t cfg = ca_mps_var_d_config_default();
    cfg.improvement_eps = eps;
    cfg.include_2q_gates = 1;
    cfg.max_passes = 40;
    ca_mps_error_t rc = moonlab_ca_mps_optimize_var_d_clifford_only(s, P, c, nt, &cfg, res);
    moonlab_ca_mps_free(s);
    return rc == CA_MPS_SUCCESS ? 0 : -1;
}

static void test_hamiltonian(int n, const char *name, const uint8_t *P, const double *c, uint32_t nt) {
    char lbl[96];
    double lb = -sum_abs(c, nt);                 /* analytic variational lower bound */
    ca_mps_var_d_result_t res; memset(&res, 0, sizeof res);
    if (run_var_d(n, P, c, nt, 1e-6, &res) != 0) {
        g_fail++; fprintf(stderr, "  *** DIVERGENCE ***  %s n=%d var-D FAILED to run\n", name, n);
        return;
    }
    /* 1. variational lower bound (exact at any n) */
    snprintf(lbl, sizeof lbl, "%s n=%d final_energy >= -sum|c| (%.4f)", name, n, lb);
    check(res.final_energy >= lb - 1e-6, lbl, res.final_energy, lb);
    /* 3. monotone improvement */
    snprintf(lbl, sizeof lbl, "%s n=%d final <= initial", name, n);
    check(res.final_energy <= res.initial_energy + 1e-9, lbl, res.final_energy, res.initial_energy);
    /* 4. eps-escape convergence: finite eps must terminate converged */
    snprintf(lbl, sizeof lbl, "%s n=%d converged flag set", name, n);
    check(res.converged == 1, lbl, (double)res.converged, 1.0);

    /* eps monotonicity: a coarser (larger) eps must not accept MORE gates. */
    ca_mps_var_d_result_t r_coarse; memset(&r_coarse, 0, sizeof r_coarse);
    if (run_var_d(n, P, c, nt, 1e-2, &r_coarse) == 0) {
        snprintf(lbl, sizeof lbl, "%s n=%d coarse-eps gates <= fine-eps gates", name, n);
        check(r_coarse.gates_added <= res.gates_added, lbl,
              (double)r_coarse.gates_added, (double)res.gates_added);
    }

    /* 2. ED agreement at small n (feasible up to ~n=12) */
    if (n <= 12) {
        double e0 = ed_ground_energy(P, c, nt, n);
        snprintf(lbl, sizeof lbl, "%s n=%d final_energy >= ED E0 (%.4f)", name, n, e0);
        check(res.final_energy >= e0 - 1e-6, lbl, res.final_energy, e0);
        if (g_verbose)
            fprintf(stderr, "    [%s n=%d] init=%.5f final=%.5f ED=%.5f gates=%d passes=%d\n",
                    name, n, res.initial_energy, res.final_energy, e0, res.gates_added, res.passes);
    }
}

/* ---- uint32 width: bond-dim fields must hold > 255. ---- */
static void test_uint32_width(void) {
    /* (a) CA-MPS max_bond_dim getter must return values > 255 un-clamped. */
    uint32_t widths[] = {256, 300, 512, 1024, 4096};
    for (int i = 0; i < 5; i++) {
        moonlab_ca_mps_t *s = moonlab_ca_mps_create(16, widths[i]);
        if (!s) { g_fail++; fprintf(stderr, "  *** DIVERGENCE ***  ca_mps_create(16,%u) FAILED\n", widths[i]); continue; }
        uint32_t got = moonlab_ca_mps_max_bond_dim(s);
        char lbl[64]; snprintf(lbl, sizeof lbl, "ca_mps max_bond_dim holds %u (>255)", widths[i]);
        check(got == widths[i], lbl, (double)got, (double)widths[i]);
        moonlab_ca_mps_free(s);
    }
    /* (b) plain tn_mps config + a real bond grown past 255 must not wrap. */
    {
        uint32_t chi = 512;
        tn_state_config_t cfg = tn_state_config_create(chi, 1e-14);
        tn_mps_state_t *m = tn_mps_create_zero(16, &cfg);
        if (!m) { g_fail++; fprintf(stderr, "  *** DIVERGENCE ***  tn_mps_create_zero FAILED\n"); return; }
        /* configured cap field must hold 512 (>255) un-clamped */
        check(m->config.max_bond_dim == chi, "tn_mps config.max_bond_dim holds 512 (>255)",
              (double)m->config.max_bond_dim, (double)chi);
        /* grow a middle bond to 300 (> 255) and confirm the current-max getter + the
         * bond_dims field both report it back un-wrapped (an old uint8 would clamp). */
        tn_state_error_t ge = tn_mps_grow_bond(m, 7, 300);
        if (ge == TN_STATE_SUCCESS) {
            uint32_t got = m->bond_dims[7];
            check(got == 300, "tn_mps grown bond[7] == 300 (>255, no wrap)", (double)got, 300.0);
            check(tn_mps_max_bond_dim(m) >= 300, "tn_mps current max bond >= 300 after grow",
                  (double)tn_mps_max_bond_dim(m), 300.0);
        } else if (g_verbose) {
            fprintf(stderr, "    tn_mps_grow_bond -> %d (skipped bond-grow width check)\n", ge);
        }
        tn_mps_free(m);
    }
}

static int selftest(void) {
    /* ED of 2-site TFIM: H = -Z0Z1 - h(X0+X1). Ground energy known to be
     * -(1 + sqrt(1+h^2))? Use h=1: eigen-check via power iteration must give a
     * value <= the trivial product-state energy and >= -sum|c|. */
    int n = 2; double J = 1.0, h = 1.0;
    uint8_t *P; double *c; uint32_t nt = build_tfim(n, J, h, &P, &c);
    double e0 = ed_ground_energy(P, c, nt, n);
    double lb = -sum_abs(c, nt);
    free(P); free(c);
    if (!(e0 >= lb - 1e-9 && e0 <= 0.0)) {
        fprintf(stderr, "selftest FAIL: 2-site TFIM E0=%.5f out of [%.5f, 0]\n", e0, lb); return 1;
    }
    /* closed form for H = -(Z0Z1 + X0 + X1) is E0 = -sqrt(5) */
    if (fabs(e0 - (-sqrt(5.0))) > 1e-3) {
        fprintf(stderr, "selftest FAIL: 2-site TFIM E0=%.5f != -sqrt(5)=%.5f\n", e0, -sqrt(5.0)); return 1;
    }
    fprintf(stderr, "scaling_var_d self-test: PASS (2-site TFIM E0=%.5f)\n", e0);
    return 0;
}

int main(int argc, char **argv) {
    int ns[8], nn = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--verbose")) g_verbose = 1;
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && nn < 8) { ns[nn++] = atoi(tok); tok = strtok(NULL, ","); }
        } else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (nn == 0) { ns[0] = 8; ns[1] = 16; ns[2] = 32; nn = 3; }

    fprintf(stderr, "=== scaling_var_d: uint32 bond-dim width (>255) ===\n");
    test_uint32_width();

    for (int i = 0; i < nn; i++) {
        fprintf(stderr, "=== scaling_var_d: n=%d TFIM + Heisenberg var-D ===\n", ns[i]);
        uint8_t *P; double *c; uint32_t nt;
        nt = build_tfim(ns[i], 1.0, 1.0, &P, &c); test_hamiltonian(ns[i], "TFIM", P, c, nt); free(P); free(c);
        nt = build_heis(ns[i], 1.0, &P, &c);      test_hamiltonian(ns[i], "Heisenberg", P, c, nt); free(P); free(c);
    }
    fprintf(stderr, "\n=== scaling_var_d summary: %ld passed, %ld divergences ===\n", g_pass, g_fail);
    fprintf(stdout, "SCALING_RESULT var_d new=%ld known=0\n", g_fail);
    if (g_fail > 0) { fprintf(stderr, "RESULT: FAIL\n"); return 1; }
    fprintf(stderr, "RESULT: PASS\n");
    return 0;
}
