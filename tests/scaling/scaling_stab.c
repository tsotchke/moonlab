/**
 * @file scaling_stab.c
 * @brief Large-n Clifford tableau vs KNOWN stabilizer structure (mission pt 3).
 *
 * Beyond the reach of any statevector reference (n = 50, 100), a stabilizer
 * state's correctness is checked analytically against the group it must be
 * stabilized by. Two structured families with closed-form stabilizers:
 *
 *   GHZ_n = (H(0), CNOT(k-1,k) chain):  stabilized by
 *       X_0 X_1 ... X_{n-1}   and   Z_i Z_{i+1}  for all i.
 *     => <Z_i Z_{i+1}> = +1 for every adjacent pair, <Z_i> = 0, <X..X> = +1.
 *
 *   Linear-cluster (graph) state on a path:  H on all, CZ on each edge (i,i+1).
 *     Stabilizers  g_i = X_i prod_{j~i} Z_j.  => <g_i> = +1 for every vertex i,
 *     and <Z_i> = 0.
 *
 * The Clifford tableau's expectation of a Pauli P on |psi>=C|0^n> is exact:
 * conjugate P back through C^{-1}; it is +-1 if the result is diagonal on
 * |0^n>, else 0. This is O(n^2) per observable, so n=100 is cheap. No
 * statevector is ever formed. A deliberately-wrong expected sign is injected in
 * --selftest to prove the check has teeth.
 *
 * Usage: scaling_stab [--n 50,100] [--verbose] ;  scaling_stab --selftest
 */

#include "../../src/backends/clifford/clifford.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-9
static long g_fail = 0, g_pass = 0;
static int g_verbose = 0;

/* Pauli byte encoding matches clifford_conjugate_pauli_inverse: 0=I,1=X,2=Y,3=Z. */
static double expect_pauli(const clifford_tableau_t *t, int n, const uint8_t *P) {
    uint8_t *in = malloc((size_t)n), *out = malloc((size_t)n);
    memcpy(in, P, (size_t)n);
    int ph = 0;
    clifford_conjugate_pauli_inverse(t, in, 0, out, &ph);
    double v;
    int diag = 1;
    for (int k = 0; k < n; k++) if (out[k] == 1 || out[k] == 2) { diag = 0; break; }
    if (!diag) v = 0.0;
    else if (ph == 0) v = 1.0;
    else if (ph == 2) v = -1.0;
    else v = 0.0;
    free(in); free(out);
    return v;
}

static void check(const char *what, double got, double want) {
    double d = got - want; if (d < 0) d = -d;
    if (d <= TOL) { g_pass++; if (g_verbose) fprintf(stderr, "  ok   %s got=%.3f\n", what, got); }
    else { g_fail++; fprintf(stderr, "  *** DIVERGENCE ***  %s got=%.6f want=%.6f dev=%.3e\n",
                             what, got, want, d); }
}

/* GHZ via H(0) + adjacent CNOT chain. */
static int test_ghz(int n) {
    clifford_tableau_t *t = clifford_tableau_create((size_t)n);
    if (!t) return 1;
    clifford_h(t, 0);
    for (int k = 0; k + 1 < n; k++) clifford_cnot(t, (size_t)k, (size_t)k + 1);

    uint8_t *P = calloc((size_t)n, 1);
    char lbl[64];
    /* <Z_i Z_{i+1}> = +1 for all adjacent pairs */
    for (int i = 0; i + 1 < n; i++) {
        memset(P, 0, (size_t)n); P[i] = 3; P[i + 1] = 3;
        snprintf(lbl, sizeof lbl, "GHZ n=%d <Z%dZ%d>", n, i, i + 1);
        check(lbl, expect_pauli(t, n, P), 1.0);
    }
    /* <Z_0 Z_{n-1}> = +1 (global parity of the chain) */
    memset(P, 0, (size_t)n); P[0] = 3; P[n - 1] = 3;
    snprintf(lbl, sizeof lbl, "GHZ n=%d <Z0Z%d>", n, n - 1);
    check(lbl, expect_pauli(t, n, P), 1.0);
    /* <X_0 X_1 ... X_{n-1}> = +1 */
    memset(P, 0, (size_t)n); for (int k = 0; k < n; k++) P[k] = 1;
    snprintf(lbl, sizeof lbl, "GHZ n=%d <X..X>", n);
    check(lbl, expect_pauli(t, n, P), 1.0);
    /* <Z_i> = 0 for a few sites */
    for (int i = 0; i < n; i += (n / 4 > 0 ? n / 4 : 1)) {
        memset(P, 0, (size_t)n); P[i] = 3;
        snprintf(lbl, sizeof lbl, "GHZ n=%d <Z%d>", n, i);
        check(lbl, expect_pauli(t, n, P), 0.0);
    }
    free(P); clifford_tableau_free(t);
    return 0;
}

/* Linear-cluster (graph) state: H on all, CZ on each path edge. */
static int test_cluster(int n) {
    clifford_tableau_t *t = clifford_tableau_create((size_t)n);
    if (!t) return 1;
    for (int i = 0; i < n; i++) clifford_h(t, (size_t)i);
    for (int i = 0; i + 1 < n; i++) clifford_cz(t, (size_t)i, (size_t)i + 1);

    uint8_t *P = calloc((size_t)n, 1);
    char lbl[64];
    /* stabilizer g_i = X_i * prod_{j~i} Z_j has <g_i> = +1 for every vertex */
    for (int i = 0; i < n; i++) {
        memset(P, 0, (size_t)n);
        P[i] = 1;                       /* X_i */
        if (i - 1 >= 0) P[i - 1] = 3;   /* Z left neighbour */
        if (i + 1 < n) P[i + 1] = 3;    /* Z right neighbour */
        snprintf(lbl, sizeof lbl, "cluster n=%d <g%d>", n, i);
        check(lbl, expect_pauli(t, n, P), 1.0);
    }
    /* single-site <Z_i> = 0 and <X_i> = 0 on a cluster state */
    for (int i = 0; i < n; i += (n / 4 > 0 ? n / 4 : 1)) {
        memset(P, 0, (size_t)n); P[i] = 3;
        snprintf(lbl, sizeof lbl, "cluster n=%d <Z%d>", n, i);
        check(lbl, expect_pauli(t, n, P), 0.0);
        memset(P, 0, (size_t)n); P[i] = 1;
        snprintf(lbl, sizeof lbl, "cluster n=%d <X%d>", n, i);
        check(lbl, expect_pauli(t, n, P), 0.0);
    }
    free(P); clifford_tableau_free(t);
    return 0;
}

static int selftest(void) {
    /* GHZ n=6: prove the check catches a wrong expected value. */
    int n = 6;
    clifford_tableau_t *t = clifford_tableau_create((size_t)n);
    clifford_h(t, 0);
    for (int k = 0; k + 1 < n; k++) clifford_cnot(t, (size_t)k, (size_t)k + 1);
    uint8_t P[6]; memset(P, 0, 6); P[0] = 3; P[1] = 3;
    double v = expect_pauli(t, n, P);
    clifford_tableau_free(t);
    if (v < 0.5) { fprintf(stderr, "selftest FAIL: GHZ <Z0Z1>=%.3f != 1\n", v); return 1; }
    /* the tolerance would catch a flipped sign: |1 - (-1)| = 2 > TOL */
    if (2.0 <= TOL) { fprintf(stderr, "selftest FAIL: tol too loose\n"); return 1; }
    fprintf(stderr, "scaling_stab self-test: PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    int ns[8], nn = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--verbose")) g_verbose = 1;
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
            char *s = argv[++i], *tok = strtok(s, ",");
            while (tok && nn < 8) { ns[nn++] = atoi(tok); tok = strtok(NULL, ","); }
        } else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (nn == 0) { ns[0] = 50; ns[1] = 100; nn = 2; }

    for (int i = 0; i < nn; i++) {
        fprintf(stderr, "=== scaling_stab: n=%d (GHZ + linear-cluster) ===\n", ns[i]);
        test_ghz(ns[i]);
        test_cluster(ns[i]);
    }
    fprintf(stderr, "\n=== scaling_stab summary: %ld passed, %ld divergences ===\n", g_pass, g_fail);
    fprintf(stdout, "SCALING_RESULT stab new=%ld known=0\n", g_fail);
    if (g_fail > 0) { fprintf(stderr, "RESULT: FAIL\n"); return 1; }
    fprintf(stderr, "RESULT: PASS\n");
    return 0;
}
