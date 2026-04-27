/**
 * @file test_hermitian_eigen.c
 * @brief Pin the complex-Hermitian Jacobi behaviour of
 *        hermitian_eigen_decomposition.
 *
 * The pre-2026-04-26 implementation used real-valued Givens rotations,
 * which left the **eigenvectors** of a complex-Hermitian matrix only
 * diagonalising the real part of A.  Eigenvalues converged but
 * eigenvectors were wrong, so any caller that built projectors from
 * the returned vectors got a silently-incorrect answer.  The Chern-
 * marker workaround was a sign-function iteration that bypassed the
 * primitive entirely.
 *
 * This test pins three things so a regression cannot re-introduce the
 * bug:
 *
 *   1. A 2x2 complex-Hermitian matrix with known closed-form eigenvalues
 *      ((1 +/- sqrt(29))/2 for [[2, 1+2i],[1-2i, -1]]) returns the right
 *      eigenvalues to 1e-12.
 *   2. For the same matrix, ||H v - lambda v|| < 1e-12 for each
 *      returned eigenvector.
 *   3. A 4x4 random complex-Hermitian H built as M + M^dagger satisfies
 *      ||H v - lambda v|| < 1e-10 for all four eigenpairs.
 */

#include "../../src/utils/matrix_math.h"
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static double residual(const complex_t *H, size_t n,
                        const complex_t *v, double lambda) {
    double r = 0.0;
    for (size_t i = 0; i < n; i++) {
        complex_t Hv = 0.0;
        for (size_t k = 0; k < n; k++) Hv += H[i * n + k] * v[k];
        r += creal((Hv - lambda * v[i]) * conj(Hv - lambda * v[i]));
    }
    return sqrt(r);
}

static void test_2x2_closed_form(void) {
    fprintf(stdout, "\n-- 2x2 complex-Hermitian, closed-form eigenvalues --\n");
    /* H = [[2, 1+2i], [1-2i, -1]] -- trace 1, det -7,
     *      eigenvalues (1 +/- sqrt(29))/2. */
    const size_t n = 2;
    complex_t H[4] = {
        2.0,         1.0 + 2.0 * I,
        1.0 - 2.0 * I, -1.0,
    };
    double evals[2] = { 0 };
    complex_t evecs[4] = { 0 };

    int rc = hermitian_eigen_decomposition(H, n, evals, evecs, 0, 1e-12);
    CHECK(rc == 0, "rc = %d (expected 0)", rc);

    const double exp_pos = (1.0 + sqrt(29.0)) / 2.0;
    const double exp_neg = (1.0 - sqrt(29.0)) / 2.0;
    int matched_pos = 0, matched_neg = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs(evals[i] - exp_pos) < 1e-12) matched_pos = 1;
        if (fabs(evals[i] - exp_neg) < 1e-12) matched_neg = 1;
    }
    CHECK(matched_pos, "missing positive eigenvalue %.6f (got [%.6f, %.6f])",
          exp_pos, evals[0], evals[1]);
    CHECK(matched_neg, "missing negative eigenvalue %.6f (got [%.6f, %.6f])",
          exp_neg, evals[0], evals[1]);

    for (size_t j = 0; j < n; j++) {
        complex_t v[2] = { evecs[0 * n + j], evecs[1 * n + j] };
        double r = residual(H, n, v, evals[j]);
        CHECK(r < 1e-12, "||H v_%zu - lambda_%zu v_%zu|| = %.3e", j, j, j, r);
        fprintf(stdout, "  evec %zu: lambda = %+.6f, residual = %.3e\n",
                j, evals[j], r);
    }
}

static void test_4x4_random(void) {
    fprintf(stdout, "\n-- 4x4 random complex-Hermitian (M + M^H, fixed seed) --\n");
    const size_t n = 4;
    complex_t H[16] = { 0 };
    /* Build a random complex M, then symmetrise to M + M^H. */
    srand(0xBEEFu);
    complex_t M[16];
    for (size_t i = 0; i < 16; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        M[i] = re + im * I;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            H[i * n + j] = M[i * n + j] + conj(M[j * n + i]);
        }
    }

    double evals[4] = { 0 };
    complex_t evecs[16] = { 0 };
    int rc = hermitian_eigen_decomposition(H, n, evals, evecs, 0, 1e-12);
    CHECK(rc == 0, "rc = %d (expected 0)", rc);

    double max_resid = 0.0;
    for (size_t j = 0; j < n; j++) {
        complex_t v[4] = {
            evecs[0 * n + j], evecs[1 * n + j],
            evecs[2 * n + j], evecs[3 * n + j],
        };
        double r = residual(H, n, v, evals[j]);
        if (r > max_resid) max_resid = r;
    }
    CHECK(max_resid < 1e-10, "max residual = %.3e (expected < 1e-10)", max_resid);
    fprintf(stdout, "  max residual = %.3e\n", max_resid);
}

int main(void) {
    fprintf(stdout, "=== hermitian_eigen_decomposition tests ===\n");
    test_2x2_closed_form();
    test_4x4_random();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
