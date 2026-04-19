/**
 * @file test_matrix_math.c
 * @brief Ground-truth tests for src/utils/matrix_math.c.
 *
 * Covers the parts of the linear-algebra toolkit that other subsystems
 * rely on:
 *  - matrix_multiply on complex matrices vs hand-computed reference
 *  - matrix_trace, matrix_frobenius_norm, matrix_is_hermitian
 *  - matrix_conjugate_transpose
 *  - hermitian_eigen_decomposition: EIGENVALUES are correct for a 2x2
 *    Pauli-Z-like and a 3x3 real-symmetric matrix; EIGENVECTORS are
 *    only verified on a real-symmetric case (the complex-Hermitian
 *    eigenvector branch is known unsound -- see README + the warning
 *    in matrix_math.h -- so we deliberately do NOT assert on them).
 */

#include "../../src/utils/matrix_math.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_matmul_2x2(void) {
    fprintf(stdout, "\n-- matrix_multiply: 2x2 complex --\n");
    complex_t A[4] = { 1, _Complex_I, 2, -_Complex_I };
    complex_t B[4] = { 3, 0, 0, 4 * _Complex_I };
    complex_t C[4] = { 0 };
    matrix_multiply(A, B, C, 2, 2, 2);
    /* Hand-check: C[0,0] = 1*3 + i*0 = 3;
     *             C[0,1] = 1*0 + i*4i = -4;
     *             C[1,0] = 2*3 + (-i)*0 = 6;
     *             C[1,1] = 2*0 + (-i)*4i = 4. */
    CHECK(fabs(creal(C[0]) - 3.0) < 1e-12 && fabs(cimag(C[0])) < 1e-12,
          "C[0,0] = 3 + 0i (got %.3f + %.3fi)",
          creal(C[0]), cimag(C[0]));
    CHECK(fabs(creal(C[1]) + 4.0) < 1e-12 && fabs(cimag(C[1])) < 1e-12,
          "C[0,1] = -4 + 0i (got %.3f + %.3fi)",
          creal(C[1]), cimag(C[1]));
    CHECK(fabs(creal(C[2]) - 6.0) < 1e-12 && fabs(cimag(C[2])) < 1e-12,
          "C[1,0] = 6 + 0i (got %.3f + %.3fi)",
          creal(C[2]), cimag(C[2]));
    CHECK(fabs(creal(C[3]) - 4.0) < 1e-12 && fabs(cimag(C[3])) < 1e-12,
          "C[1,1] = 4 + 0i (got %.3f + %.3fi)",
          creal(C[3]), cimag(C[3]));
}

static void test_trace_and_norm(void) {
    fprintf(stdout, "\n-- matrix_trace + matrix_frobenius_norm --\n");
    complex_t A[4] = { 1, _Complex_I, -_Complex_I, 2 };
    complex_t t = matrix_trace(A, 2);
    CHECK(fabs(creal(t) - 3.0) < 1e-12 && fabs(cimag(t)) < 1e-12,
          "trace = 3 (got %.3f + %.3fi)", creal(t), cimag(t));

    /* Frobenius norm of A: sqrt(|1|^2 + |i|^2 + |-i|^2 + |2|^2)
     *                    = sqrt(1 + 1 + 1 + 4) = sqrt(7). */
    double f = matrix_frobenius_norm(A, 2, 2);
    CHECK(fabs(f - sqrt(7.0)) < 1e-12,
          "frobenius norm = sqrt(7) (got %.6f)", f);
}

static void test_hermitian_check(void) {
    fprintf(stdout, "\n-- matrix_is_hermitian --\n");
    /* Pauli-Y is Hermitian: [[0, -i], [i, 0]]. */
    complex_t Y[4] = { 0, -_Complex_I, _Complex_I, 0 };
    CHECK(matrix_is_hermitian(Y, 2, 1e-12) == 1,
          "Pauli-Y reports Hermitian");

    /* A non-Hermitian matrix. */
    complex_t B[4] = { 1, _Complex_I, 1, 0 };
    CHECK(matrix_is_hermitian(B, 2, 1e-12) == 0,
          "non-Hermitian rejected");
}

static void test_conjugate_transpose(void) {
    fprintf(stdout, "\n-- matrix_conjugate_transpose --\n");
    complex_t A[6] = {
        1 + _Complex_I, 2, 3,
        -_Complex_I,    4, 5 * _Complex_I,
    };
    complex_t R[6] = { 0 };
    matrix_conjugate_transpose(A, R, 2, 3);
    /* (R is 3 x 2 row-major): R[i,j] = conj(A[j,i]). */
    CHECK(fabs(creal(R[0]) - 1.0) < 1e-12 && fabs(cimag(R[0]) + 1.0) < 1e-12,
          "R[0,0] = conj(A[0,0]) = 1 - i");
    CHECK(fabs(creal(R[1])) < 1e-12 && fabs(cimag(R[1]) - 1.0) < 1e-12,
          "R[0,1] = conj(A[1,0]) = +i (A[1,0] was -i)");
    CHECK(fabs(creal(R[5])) < 1e-12 && fabs(cimag(R[5]) + 5.0) < 1e-12,
          "R[2,1] = conj(A[1,2]) = -5i");
}

static void test_eig_eigenvalues_pauli_z(void) {
    fprintf(stdout, "\n-- hermitian_eigen_decomposition: Pauli-Z eigenvalues --\n");
    complex_t Z[4] = { 1, 0, 0, -1 };
    double eig[2];
    complex_t vec[4];
    int rc = hermitian_eigen_decomposition(Z, 2, eig, vec, 0, 1e-12);
    CHECK(rc == 0, "rc == 0");
    /* Sorted descending: +1, -1. */
    CHECK(fabs(eig[0] - 1.0) < 1e-10, "eig[0] = +1 (got %.6f)", eig[0]);
    CHECK(fabs(eig[1] + 1.0) < 1e-10, "eig[1] = -1 (got %.6f)", eig[1]);
}

static void test_eig_real_symmetric_3x3(void) {
    fprintf(stdout, "\n-- hermitian_eigen_decomposition: real-symmetric 3x3 --\n");
    /* Matrix M = [[2, 1, 0],
     *             [1, 2, 1],
     *             [0, 1, 2]]
     * Analytical eigenvalues: 2 + sqrt(2), 2, 2 - sqrt(2). */
    complex_t M[9] = {
        2, 1, 0,
        1, 2, 1,
        0, 1, 2,
    };
    double eig[3];
    complex_t vec[9];
    int rc = hermitian_eigen_decomposition(M, 3, eig, vec, 0, 1e-12);
    CHECK(rc == 0, "rc == 0");
    const double lo = 2.0 - sqrt(2.0);
    const double mid = 2.0;
    const double hi = 2.0 + sqrt(2.0);
    CHECK(fabs(eig[0] - hi) < 1e-8, "eig[0] = 2 + sqrt(2)");
    CHECK(fabs(eig[1] - mid) < 1e-8, "eig[1] = 2");
    CHECK(fabs(eig[2] - lo) < 1e-8, "eig[2] = 2 - sqrt(2)");

    /* Eigenvector sanity: M v_0 ~ hi * v_0 (real-symmetric path is
     * reliable; complex-Hermitian eigenvectors are NOT -- see header
     * warning and README). */
    complex_t v0[3] = { vec[0 * 3 + 0], vec[1 * 3 + 0], vec[2 * 3 + 0] };
    complex_t Mv0[3] = {
        M[0] * v0[0] + M[1] * v0[1] + M[2] * v0[2],
        M[3] * v0[0] + M[4] * v0[1] + M[5] * v0[2],
        M[6] * v0[0] + M[7] * v0[1] + M[8] * v0[2],
    };
    double err = 0.0;
    for (int i = 0; i < 3; i++) {
        complex_t d = Mv0[i] - hi * v0[i];
        err += creal(d) * creal(d) + cimag(d) * cimag(d);
    }
    CHECK(sqrt(err) < 1e-8, "||M v_0 - lambda_0 v_0|| < 1e-8 (got %.2e)",
          sqrt(err));
}

static void test_eig_null_guards(void) {
    fprintf(stdout, "\n-- hermitian_eigen_decomposition: NULL guards --\n");
    double eig[2];
    complex_t vec[4];
    complex_t M[4] = { 1, 0, 0, 1 };
    CHECK(hermitian_eigen_decomposition(NULL, 2, eig, vec, 0, 0) != 0,
          "NULL matrix rejected");
    CHECK(hermitian_eigen_decomposition(M, 2, NULL, vec, 0, 0) != 0,
          "NULL eigenvalues rejected");
    CHECK(hermitian_eigen_decomposition(M, 2, eig, NULL, 0, 0) != 0,
          "NULL eigenvectors rejected");
    CHECK(hermitian_eigen_decomposition(M, 0, eig, vec, 0, 0) != 0,
          "n=0 rejected");
}

int main(void) {
    fprintf(stdout, "=== matrix_math ===\n");
    test_matmul_2x2();
    test_trace_and_norm();
    test_hermitian_check();
    test_conjugate_transpose();
    test_eig_eigenvalues_pauli_z();
    test_eig_real_symmetric_3x3();
    test_eig_null_guards();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
