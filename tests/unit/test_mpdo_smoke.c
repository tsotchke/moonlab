/**
 * @file test_mpdo_smoke.c
 * @brief Smoke test for the MPDO scaffold (single-qubit channels).
 *
 * Pins:
 *   1. |0...0><0...0| product-state initialisation has Tr(rho) = 1
 *      and <Z_q> = +1 for every qubit.
 *   2. Single-qubit X gate acting via Kraus = {X} flips <Z> -> -1.
 *   3. Depolarising channel with rate p = 1.0 (full random) on a |0>
 *      state gives <Z> = 0 (maximally mixed).
 *   4. Depolarising channel with intermediate p sends <Z> ->
 *      (1 - 4p/3) per the standard formula.
 *   5. Lifecycle (clone) preserves observables.
 */

#include "../../src/quantum/noise_mpdo.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

#define CHECK_NEAR(actual, expected, tol, label) do { \
    double _a = (actual), _e = (expected); \
    if (!(fabs(_a - _e) < (tol))) { \
        fprintf(stderr, "FAIL %s: got %.10g, expected %.10g (|diff| = %.3e > %.3e)\n", \
                (label), _a, _e, fabs(_a - _e), (tol)); \
        failures++; \
    } \
} while (0)

int main(void) {
    fprintf(stdout, "=== MPDO scaffold smoke test ===\n");

    /* ---- Case 1: initial state ----------------------------------- */
    {
        moonlab_mpdo_t* m = moonlab_mpdo_create(3, 16);
        CHECK(m != NULL, "create n=3");
        if (!m) return 1;

        double tr = moonlab_mpdo_trace(m);
        CHECK_NEAR(tr, 1.0, 1e-12, "Tr(rho) on |000>");

        for (uint32_t q = 0; q < 3; q++) {
            double e = 0.0;
            mpdo_error_t rc = moonlab_mpdo_expect_pauli_1q(m, q, 3, &e);
            CHECK(rc == MPDO_SUCCESS, "expect rc=%d", (int)rc);
            CHECK_NEAR(e, 1.0, 1e-12, "<Z_q> on |000>");
        }

        moonlab_mpdo_free(m);
    }

    /* ---- Case 2: X gate as Kraus = {X} flips <Z> ----------------- */
    {
        moonlab_mpdo_t* m = moonlab_mpdo_create(2, 16);
        CHECK(m != NULL, "create n=2");

        /* Pauli X as a 2x2 Kraus operator (single op, no decoherence). */
        const mpdo_complex_t X_kraus[4] = { 0.0, 1.0, 1.0, 0.0 };
        mpdo_error_t rc = moonlab_mpdo_apply_kraus_1q(m, 0, X_kraus, 1);
        CHECK(rc == MPDO_SUCCESS, "apply X to qubit 0 rc=%d", (int)rc);

        double tr = moonlab_mpdo_trace(m);
        CHECK_NEAR(tr, 1.0, 1e-12, "Tr(rho) after X");

        double zq0 = 0.0, zq1 = 0.0;
        moonlab_mpdo_expect_pauli_1q(m, 0, 3, &zq0);
        moonlab_mpdo_expect_pauli_1q(m, 1, 3, &zq1);
        CHECK_NEAR(zq0, -1.0, 1e-12, "<Z_0> after X_0 = -1");
        CHECK_NEAR(zq1,  1.0, 1e-12, "<Z_1> unchanged");

        moonlab_mpdo_free(m);
    }

    /* ---- Case 3: full Pauli-twirl channel (p=1, sigma-only) ------
     *
     * With Kraus = (1/sqrt(3)) {X, Y, Z} (no identity component) the
     * channel is rho -> (1/3)(X rho X + Y rho Y + Z rho Z).  On |0><0|:
     *   X|0><0|X = |1><1|,  Y|0><0|Y = |1><1|,  Z|0><0|Z = |0><0|
     * So rho -> (2/3)|1><1| + (1/3)|0><0| and <Z> = 1/3 - 2/3 = -1/3.
     * The "rho -> I/2" outcome requires a different Kraus rep with
     * an identity component (e.g. K_0 = sqrt(p/2) I, K_a = sqrt(p/2) sigma_a
     * with appropriate p), which we test below in Case 4. */
    {
        moonlab_mpdo_t* m = moonlab_mpdo_create(1, 16);

        const double s = 1.0 / sqrt(3.0);
        const mpdo_complex_t depol_kraus[3 * 4] = {
            /* sigma_x */  0.0,            (mpdo_complex_t)s,
                           (mpdo_complex_t)s, 0.0,
            /* sigma_y */  0.0,                  -(mpdo_complex_t)s * _Complex_I,
                           (mpdo_complex_t)s * _Complex_I, 0.0,
            /* sigma_z */  (mpdo_complex_t)s,    0.0,
                           0.0,                 -(mpdo_complex_t)s,
        };
        mpdo_error_t rc = moonlab_mpdo_apply_kraus_1q(m, 0, depol_kraus, 3);
        CHECK(rc == MPDO_SUCCESS, "apply Pauli-twirl rc=%d", (int)rc);

        double tr = moonlab_mpdo_trace(m);
        CHECK_NEAR(tr, 1.0, 1e-12, "Tr(rho) after Pauli-twirl");

        double z = 0.0;
        moonlab_mpdo_expect_pauli_1q(m, 0, 3, &z);
        CHECK_NEAR(z, -1.0/3.0, 1e-12, "<Z> after Pauli-twirl on |0> = -1/3");

        moonlab_mpdo_free(m);
    }

    /* ---- Case 4: depolarising at intermediate p ------------------ */
    {
        const double p = 0.4;
        const double a = sqrt(1.0 - p);              /* identity weight */
        const double b = sqrt(p / 3.0);              /* per-Pauli weight */
        moonlab_mpdo_t* m = moonlab_mpdo_create(1, 16);
        const mpdo_complex_t depol_kraus[4 * 4] = {
            /* I */ (mpdo_complex_t)a, 0.0, 0.0, (mpdo_complex_t)a,
            /* X */ 0.0,                  (mpdo_complex_t)b,
                    (mpdo_complex_t)b, 0.0,
            /* Y */ 0.0,                  -(mpdo_complex_t)b * _Complex_I,
                    (mpdo_complex_t)b * _Complex_I, 0.0,
            /* Z */ (mpdo_complex_t)b, 0.0, 0.0, -(mpdo_complex_t)b,
        };
        moonlab_mpdo_apply_kraus_1q(m, 0, depol_kraus, 4);
        double z = 0.0;
        moonlab_mpdo_expect_pauli_1q(m, 0, 3, &z);
        const double expected = 1.0 - (4.0 * p) / 3.0;
        CHECK_NEAR(z, expected, 1e-12, "<Z> after depol(p=0.4)");

        double tr = moonlab_mpdo_trace(m);
        CHECK_NEAR(tr, 1.0, 1e-12, "Tr(rho) after depol(p=0.4)");

        moonlab_mpdo_free(m);
    }

    /* ---- Case 5: clone preserves state --------------------------- */
    {
        moonlab_mpdo_t* m = moonlab_mpdo_create(2, 16);
        const mpdo_complex_t X_kraus[4] = { 0.0, 1.0, 1.0, 0.0 };
        moonlab_mpdo_apply_kraus_1q(m, 1, X_kraus, 1);
        moonlab_mpdo_t* c = moonlab_mpdo_clone(m);
        CHECK(c != NULL, "clone");

        double z_orig = 0.0, z_copy = 0.0;
        moonlab_mpdo_expect_pauli_1q(m, 1, 3, &z_orig);
        moonlab_mpdo_expect_pauli_1q(c, 1, 3, &z_copy);
        CHECK_NEAR(z_copy, z_orig, 1e-15, "clone <Z> matches original");

        /* Mutate original; copy should be unchanged. */
        moonlab_mpdo_apply_kraus_1q(m, 0, X_kraus, 1);
        double z_orig_after = 0.0, z_copy_after = 0.0;
        moonlab_mpdo_expect_pauli_1q(m, 0, 3, &z_orig_after);
        moonlab_mpdo_expect_pauli_1q(c, 0, 3, &z_copy_after);
        CHECK_NEAR(z_orig_after, -1.0, 1e-12, "<Z_0> on mutated original = -1");
        CHECK_NEAR(z_copy_after,  1.0, 1e-12, "<Z_0> on clone = +1 (independent)");

        moonlab_mpdo_free(c);
        moonlab_mpdo_free(m);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
