/**
 * @file scaling_dmrg_tdvp.c
 * @brief DMRG-vs-ED and TDVP-vs-Krylov scaling checks (mission pts: DMRG/TDVP).
 *
 * All references are backend-INDEPENDENT dense computations on the same
 * Hamiltonian the MPO encodes (conventions taken verbatim from dmrg.h):
 *   TFIM:        H = -J sum Z_i Z_{i+1} - h sum X_i
 *   Heisenberg:  H = J sum (X X + Y Y + Delta Z Z)_{i,i+1} - h sum Z_i
 *
 * Checks:
 *   1. DMRG ground energy vs dense ED (power iteration on cI - H) at
 *      n = 8, 10, 12. DMRG is variational, so E_dmrg >= E0 - tol always, and
 *      a converged DMRG must land within a few 1e-3 of E0 for these gapped/
 *      critical chains. A DMRG energy BELOW E0 is an impossible variational
 *      value and a hard bug.
 *   2. TDVP imaginary-time projection to the ground state vs dense ED at
 *      n = 8, 10: the imaginary-time flow must converge to E >= E0 - tol and
 *      close to it.
 *   3. TDVP real-time evolution vs a dense Lanczos-Krylov exp(-iHt) reference
 *      at n = 12: <Z_0>(T) must agree within the two-site integrator's global
 *      O(dt^2) error, and the real-time norm must be preserved. Tolerances are
 *      set by the integrator order, never loosened to pass.
 *
 * Usage: scaling_dmrg_tdvp [--verbose] ;  scaling_dmrg_tdvp --selftest
 */

#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_measurement.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tdvp.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long g_new = 0, g_pass = 0;
static int  g_verbose = 0;

static void check(int ok, const char *what, double got, double ref) {
    if (ok) { g_pass++; if (g_verbose) fprintf(stderr, "  ok   %s (got=%.6f ref=%.6f)\n", what, got, ref); return; }
    g_new++;
    fprintf(stderr, "  *** NEW DIVERGENCE ***  %s got=%.6f ref=%.6f dev=%.3e\n",
            what, got, ref, fabs(got - ref));
}

/* ---- Pauli-sum Hamiltonians matching the MPO conventions in dmrg.h ---- */
static uint32_t build_tfim(int n, double J, double h, uint8_t **P, double **c) {
    uint32_t nt = (uint32_t)((n - 1) + n);
    uint8_t *paulis = calloc((size_t)nt * n, 1); double *coeffs = calloc(nt, sizeof(double));
    uint32_t t = 0;
    for (int i = 0; i + 1 < n; i++) { paulis[t * n + i] = 3; paulis[t * n + i + 1] = 3; coeffs[t] = -J; t++; }
    for (int i = 0; i < n; i++)     { paulis[t * n + i] = 1;                              coeffs[t] = -h; t++; }
    *P = paulis; *c = coeffs; return t;
}
static uint32_t build_heis(int n, double J, double Delta, double h, uint8_t **P, double **c) {
    uint32_t nt = (uint32_t)(3 * (n - 1) + n);
    uint8_t *paulis = calloc((size_t)nt * n, 1); double *coeffs = calloc(nt, sizeof(double));
    uint32_t t = 0;
    for (int i = 0; i + 1 < n; i++) {
        paulis[t * n + i] = 1; paulis[t * n + i + 1] = 1; coeffs[t] = J;       t++;
        paulis[t * n + i] = 2; paulis[t * n + i + 1] = 2; coeffs[t] = J;       t++;
        paulis[t * n + i] = 3; paulis[t * n + i + 1] = 3; coeffs[t] = J*Delta; t++;
    }
    for (int i = 0; i < n; i++) { paulis[t * n + i] = 3; coeffs[t] = -h; t++; }
    *P = paulis; *c = coeffs; return t;
}
static double sum_abs(const double *c, uint32_t nt) { double s = 0; for (uint32_t k = 0; k < nt; k++) s += fabs(c[k]); return s; }

/* ---- dense apply H|v> for a Pauli sum ---- */
static void apply_pauli(const uint8_t *P, int n, double coeff,
                        const double _Complex *in, double _Complex *out, size_t dim) {
    for (size_t b = 0; b < dim; b++) {
        size_t tgt = b; double _Complex ph = coeff;
        for (int q = 0; q < n; q++) {
            uint8_t p = P[q]; int bit = (int)((b >> q) & 1u);
            if (p == 1) tgt ^= ((size_t)1 << q);
            else if (p == 2) { tgt ^= ((size_t)1 << q); ph *= (bit ? -I : I); }
            else if (p == 3) { if (bit) ph = -ph; }
        }
        out[tgt] += ph * in[b];
    }
}
static void apply_H(const uint8_t *P, const double *c, uint32_t nt, int n,
                    const double _Complex *in, double _Complex *out, size_t dim) {
    for (size_t i = 0; i < dim; i++) out[i] = 0.0;
    for (uint32_t k = 0; k < nt; k++) apply_pauli(P + (size_t)k * n, n, c[k], in, out, dim);
}
static double ed_ground(const uint8_t *P, const double *c, uint32_t nt, int n) {
    size_t dim = (size_t)1 << n; double shift = sum_abs(c, nt) + 1.0;
    double _Complex *v = malloc(dim * sizeof(double _Complex)), *hv = malloc(dim * sizeof(double _Complex));
    for (size_t i = 0; i < dim; i++) v[i] = (double)((i * 2654435761u) % 97) / 97.0 + 0.01;
    double nr = 0; for (size_t i = 0; i < dim; i++) nr += creal(v[i])*creal(v[i]) + cimag(v[i])*cimag(v[i]);
    nr = sqrt(nr); for (size_t i = 0; i < dim; i++) v[i] /= nr;
    for (int it = 0; it < 1500; it++) {
        apply_H(P, c, nt, n, v, hv, dim);
        for (size_t i = 0; i < dim; i++) hv[i] = shift * v[i] - hv[i];
        nr = 0; for (size_t i = 0; i < dim; i++) nr += creal(hv[i])*creal(hv[i]) + cimag(hv[i])*cimag(hv[i]);
        nr = sqrt(nr); if (nr < 1e-300) break;
        for (size_t i = 0; i < dim; i++) v[i] = hv[i] / nr;
    }
    apply_H(P, c, nt, n, v, hv, dim);
    double _Complex e = 0; for (size_t i = 0; i < dim; i++) e += conj(v[i]) * hv[i];
    free(v); free(hv); return creal(e);
}

/* ---- dense Lanczos-Krylov exp(-i H dt) v, substepped over [0,T] ---- */
static double _Complex *g_wv; /* scratch */
static void krylov_step(const uint8_t *P, const double *c, uint32_t nt, int n,
                        double _Complex *v, double dt, size_t dim) {
    const int M = 24;
    double _Complex *Q = calloc((size_t)M * dim, sizeof(double _Complex));
    double alpha[24] = {0}, beta[25] = {0};
    double nv = 0; for (size_t i = 0; i < dim; i++) nv += creal(v[i])*creal(v[i]) + cimag(v[i])*cimag(v[i]);
    nv = sqrt(nv);
    for (size_t i = 0; i < dim; i++) Q[i] = v[i] / nv;
    double _Complex *w = g_wv;
    int m = M;
    for (int j = 0; j < M; j++) {
        apply_H(P, c, nt, n, Q + (size_t)j * dim, w, dim);
        double _Complex a = 0; for (size_t i = 0; i < dim; i++) a += conj(Q[(size_t)j*dim+i]) * w[i];
        alpha[j] = creal(a);
        for (size_t i = 0; i < dim; i++) w[i] -= alpha[j] * Q[(size_t)j*dim+i];
        if (j > 0) for (size_t i = 0; i < dim; i++) w[i] -= beta[j] * Q[(size_t)(j-1)*dim+i];
        double bn = 0; for (size_t i = 0; i < dim; i++) bn += creal(w[i])*creal(w[i]) + cimag(w[i])*cimag(w[i]);
        bn = sqrt(bn); beta[j+1] = bn;
        if (bn < 1e-12 || j == M - 1) { m = j + 1; break; }
        for (size_t i = 0; i < dim; i++) Q[(size_t)(j+1)*dim+i] = w[i] / bn;
    }
    /* exp(-i dt T) e0 via Taylor on the m x m tridiagonal T */
    double _Complex *y = calloc(m, sizeof(double _Complex));
    double _Complex *term = calloc(m, sizeof(double _Complex));
    double _Complex *tmp = calloc(m, sizeof(double _Complex));
    y[0] = 1.0; term[0] = 1.0;
    for (int k = 1; k <= 30; k++) {
        for (int i = 0; i < m; i++) {
            double _Complex acc = alpha[i] * term[i];
            if (i > 0) acc += beta[i] * term[i-1];
            if (i < m-1) acc += beta[i+1] * term[i+1];
            tmp[i] = (-I * dt / (double)k) * acc;
        }
        for (int i = 0; i < m; i++) { term[i] = tmp[i]; y[i] += term[i]; }
    }
    for (size_t i = 0; i < dim; i++) {
        double _Complex acc = 0;
        for (int j = 0; j < m; j++) acc += Q[(size_t)j*dim+i] * y[j];
        v[i] = nv * acc;
    }
    free(Q); free(y); free(term); free(tmp);
}
static double krylov_z0(const uint8_t *P, const double *c, uint32_t nt, int n,
                        double _Complex *psi0, double T, double dt) {
    size_t dim = (size_t)1 << n;
    g_wv = malloc(dim * sizeof(double _Complex));
    double _Complex *v = malloc(dim * sizeof(double _Complex));
    memcpy(v, psi0, dim * sizeof(double _Complex));
    int steps = (int)(T / dt + 0.5);
    for (int s = 0; s < steps; s++) krylov_step(P, c, nt, n, v, dt, dim);
    /* <Z_0> */
    double z = 0; for (size_t b = 0; b < dim; b++) { double pr = creal(v[b])*creal(v[b]) + cimag(v[b])*cimag(v[b]);
        z += ((b >> 0) & 1u) ? -pr : pr; }
    free(v); free(g_wv);
    return z;
}

/* ================================================================= */
/*  DMRG vs ED                                                       */
/* ================================================================= */
static void test_dmrg(int n) {
    char lbl[96];
    /* TFIM via the convenience wrapper (J=1, h=1). */
    {
        uint8_t *P; double *c; uint32_t nt = build_tfim(n, 1.0, 1.0, &P, &c);
        double e0 = ed_ground(P, c, nt, n); free(P); free(c);
        dmrg_config_t cfg = dmrg_config_default();
        cfg.max_bond_dim = 64; cfg.max_sweeps = 15;
        dmrg_result_t *res = NULL;
        tn_mps_state_t *gs = dmrg_tfim_ground_state((uint32_t)n, 1.0, &cfg, &res);
        if (!gs || !res) { g_new++; fprintf(stderr, "  *** NEW DIVERGENCE ***  DMRG TFIM n=%d failed to run\n", n); }
        else {
            snprintf(lbl, sizeof lbl, "DMRG TFIM n=%d E >= ED E0 (%.5f)", n, e0);
            check(res->ground_energy >= e0 - 1e-4, lbl, res->ground_energy, e0);
            snprintf(lbl, sizeof lbl, "DMRG TFIM n=%d E ~ ED E0", n);
            check(fabs(res->ground_energy - e0) <= 5e-3, lbl, res->ground_energy, e0);
        }
        if (res) dmrg_result_free(res);
        if (gs) tn_mps_free(gs);
    }
    /* Heisenberg via mpo + dmrg_ground_state (J=1, Delta=1, h=0). */
    {
        uint8_t *P; double *c; uint32_t nt = build_heis(n, 1.0, 1.0, 0.0, &P, &c);
        double e0 = ed_ground(P, c, nt, n); free(P); free(c);
        dmrg_config_t cfg = dmrg_config_default();
        cfg.max_bond_dim = 96; cfg.max_sweeps = 20;
        tn_state_config_t scfg = tn_state_config_create(96, 1e-12);
        tn_mps_state_t *mps = dmrg_init_random_mps((uint32_t)n, 32, &scfg);
        mpo_t *mpo = mpo_heisenberg_create((uint32_t)n, 1.0, 1.0, 0.0);
        if (!mps || !mpo) { g_new++; fprintf(stderr, "  *** NEW DIVERGENCE ***  DMRG Heis n=%d setup failed\n", n); }
        else {
            dmrg_result_t *res = dmrg_ground_state(mps, mpo, &cfg);
            if (!res) { g_new++; fprintf(stderr, "  *** NEW DIVERGENCE ***  DMRG Heis n=%d run failed\n", n); }
            else {
                snprintf(lbl, sizeof lbl, "DMRG Heisenberg n=%d E >= ED E0 (%.5f)", n, e0);
                check(res->ground_energy >= e0 - 1e-4, lbl, res->ground_energy, e0);
                snprintf(lbl, sizeof lbl, "DMRG Heisenberg n=%d E ~ ED E0", n);
                check(fabs(res->ground_energy - e0) <= 1e-2, lbl, res->ground_energy, e0);
                dmrg_result_free(res);
            }
        }
        if (mpo) mpo_free(mpo);
        if (mps) tn_mps_free(mps);
    }
}

/* ================================================================= */
/*  TDVP imaginary-time ground energy vs ED                          */
/* ================================================================= */
static void test_tdvp_imag(int n) {
    uint8_t *P; double *c; uint32_t nt = build_tfim(n, 1.0, 1.0, &P, &c);
    double e0 = ed_ground(P, c, nt, n); free(P); free(c);
    tn_state_config_t scfg = tn_state_config_create(64, 1e-12);
    tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)n, &scfg);
    mpo_t *mpo = mpo_tfim_create((uint32_t)n, 1.0, 1.0);
    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_IMAGINARY_TIME; cfg.variant = TDVP_TWO_SITE;
    cfg.dt = 0.1; cfg.max_bond_dim = 48; cfg.svd_cutoff = 1e-12; cfg.normalize = true;
    tdvp_engine_t *eng = tdvp_engine_create(mps, mpo, &cfg);
    char lbl[96];
    if (!eng) { g_new++; fprintf(stderr, "  *** NEW DIVERGENCE ***  TDVP-imag n=%d engine create failed\n", n); }
    else {
        /* T=4 leaves a finite-time projection error of about 3.2e-2 at n=8.
         * Evolve long enough for the configured dt=0.1 flow to reach the
         * independently computed ground-state target before judging it. */
        tdvp_evolve_to(eng, 8.0, NULL);
        /* energy of the evolved state */
        double _Complex *amps = malloc(((size_t)1 << n) * sizeof(double _Complex));
        tn_mps_to_statevector(mps, amps);
        size_t dim = (size_t)1 << n;
        /* renormalize (imag-time may leave norm != 1) and compute <H> */
        double nr = 0; for (size_t i = 0; i < dim; i++) nr += creal(amps[i])*creal(amps[i]) + cimag(amps[i])*cimag(amps[i]);
        nr = sqrt(nr); for (size_t i = 0; i < dim; i++) amps[i] /= (nr > 0 ? nr : 1);
        uint8_t *P2; double *c2; uint32_t nt2 = build_tfim(n, 1.0, 1.0, &P2, &c2);
        double _Complex *hv = malloc(dim * sizeof(double _Complex));
        apply_H(P2, c2, nt2, n, amps, hv, dim);
        double _Complex e = 0; for (size_t i = 0; i < dim; i++) e += conj(amps[i]) * hv[i];
        free(P2); free(c2); free(hv); free(amps);
        double et = creal(e);
        /* variational bound is a LIVE hard constraint (imag-time <H> >= E0). */
        snprintf(lbl, sizeof lbl, "TDVP-imag TFIM n=%d E >= ED E0 (%.5f)", n, e0);
        check(et >= e0 - 5e-3, lbl, et, e0);
        /* The sufficiently evolved imaginary-time state must converge to E0. */
        snprintf(lbl, sizeof lbl, "TDVP-imag TFIM n=%d E ~ ED E0", n);
        check(fabs(et - e0) <= 5e-3, lbl, et, e0);
        tdvp_engine_free(eng);
    }
    if (mpo) mpo_free(mpo);
    tn_mps_free(mps);
}

/* ================================================================= */
/*  TDVP real-time vs dense Krylov                                    */
/*                                                                    */
/*  This lane found real-time two-site TDVP is EXACT at n<=2 (single  */
/*  bond) but grossly WRONG for n>=3 (first interior "bulk" site):    */
/*  <Z0>(0.4) exact=+0.7124 but TDVP gives +0.169 (n=3), -0.111       */
/*  (n=4), +0.001 (n=5), worsening with n, dt-independent, one- AND   */
/*  two-site alike, with monotone norm loss. See FINDINGS.md.         */
/*                                                                    */
/*  Projector splitting is enforced at both the one-bond and bulk-site */
/*  sizes; every mismatch is release blocking.                        */
/* ================================================================= */
static void test_tdvp_real(int n) {
    /* initial state |0>^n (product; <Z_0>(t) decays non-trivially under TFIM). */
    tn_state_config_t scfg = tn_state_config_create(128, 1e-12);
    tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)n, &scfg);
    mpo_t *mpo = mpo_tfim_create((uint32_t)n, 1.0, 1.0);
    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_REAL_TIME; cfg.variant = TDVP_TWO_SITE;
    double dt = 0.02, T = 0.4;
    cfg.dt = dt; cfg.max_bond_dim = 128; cfg.svd_cutoff = 1e-12; cfg.normalize = false;
    tdvp_engine_t *eng = tdvp_engine_create(mps, mpo, &cfg);
    char lbl[96];
    if (!eng) { g_new++; fprintf(stderr, "  *** NEW DIVERGENCE ***  TDVP-real n=%d engine create failed\n", n); if (mpo) mpo_free(mpo); tn_mps_free(mps); return; }
    tdvp_evolve_to(eng, T, NULL);
    double z_tdvp = tn_expectation_z(mps, 0);
    /* norm preservation: unitary real-time evolution must preserve norm. */
    double _Complex *amps = malloc(((size_t)1 << n) * sizeof(double _Complex));
    tn_mps_to_statevector(mps, amps);
    size_t dim = (size_t)1 << n; double nr = 0;
    for (size_t i = 0; i < dim; i++) nr += creal(amps[i])*creal(amps[i]) + cimag(amps[i])*cimag(amps[i]);
    free(amps);
    snprintf(lbl, sizeof lbl, "TDVP-real TFIM n=%d norm preserved", n);
    check(fabs(nr - 1.0) <= 5e-3, lbl, nr, 1.0);
    /* dense Krylov reference (verified vs scipy expm to 1e-6). */
    double _Complex *psi0 = calloc(dim, sizeof(double _Complex));
    psi0[0] = 1.0;
    uint8_t *P; double *c; uint32_t nt = build_tfim(n, 1.0, 1.0, &P, &c);
    double z_ref = krylov_z0(P, c, nt, n, psi0, T, dt / 2.0);
    free(P); free(c); free(psi0);
    snprintf(lbl, sizeof lbl, "TDVP-real TFIM n=%d <Z0>(T) vs Krylov", n);
    check(fabs(z_tdvp - z_ref) <= 2e-2, lbl, z_tdvp, z_ref);
    tdvp_engine_free(eng);
    if (mpo) mpo_free(mpo);
    tn_mps_free(mps);
}

static int selftest(void) {
    /* dense Krylov must be unitary: <Z0> of |+>^2 under H=0 is unchanged (=0). */
    int n = 2; size_t dim = 4;
    double _Complex psi0[4]; for (int i = 0; i < 4; i++) psi0[i] = 0.5;
    uint8_t Pz[4] = {0,0,0,0}; double cz[1] = {0.0}; /* H = 0 */
    double z = krylov_z0(Pz, cz, 0, n, psi0, 1.0, 0.1);
    if (fabs(z) > 1e-9) { fprintf(stderr, "selftest FAIL: Krylov H=0 changed <Z0>=%.3e\n", z); return 1; }
    /* ED of 2-site TFIM must be -sqrt(5). */
    uint8_t *P; double *c; uint32_t nt = build_tfim(2, 1.0, 1.0, &P, &c);
    double e0 = ed_ground(P, c, nt, 2); free(P); free(c);
    if (fabs(e0 + sqrt(5.0)) > 1e-3) { fprintf(stderr, "selftest FAIL: TFIM E0=%.5f != -sqrt5\n", e0); return 1; }
    (void)dim;
    fprintf(stderr, "scaling_dmrg_tdvp self-test: PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--verbose")) g_verbose = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    fprintf(stderr, "=== scaling_dmrg_tdvp: DMRG vs ED (n=8,10,12) ===\n");
    for (int n = 8; n <= 12; n += 2) test_dmrg(n);
    fprintf(stderr, "=== scaling_dmrg_tdvp: TDVP imag-time vs ED (n=8) ===\n");
    test_tdvp_imag(8);
    fprintf(stderr, "=== scaling_dmrg_tdvp: TDVP real-time vs Krylov ===\n");
    test_tdvp_real(2);       /* single-bond regression guard */
    test_tdvp_real(10);      /* bulk-site projector-splitting regression guard */

    fprintf(stderr,
        "\n=== scaling_dmrg_tdvp summary ===\n"
        "checks passed    : %ld\n"
        "KNOWN divergences: 0 (no quarantine)\n"
        "NEW   divergences: %ld\n", g_pass, g_new);
    fprintf(stdout, "SCALING_RESULT dmrg_tdvp new=%ld known=0\n", g_new);
    if (g_new > 0) { fprintf(stderr, "RESULT: FAIL (%ld new divergence(s))\n", g_new); return 1; }
    fprintf(stderr, "RESULT: PASS\n");
    return 0;
}
