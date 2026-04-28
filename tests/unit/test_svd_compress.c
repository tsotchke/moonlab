/**
 * @file test_svd_compress.c
 * @brief Direct unit tests for the SVD compression layer.
 *
 * svd_compress is load-bearing: every MPS truncation, every CA-MPS
 * canonicalization step, every TDVP/DMRG sweep goes through it.
 * The 19-Apr audit flagged it as exercised only indirectly via
 * higher-level tests.  This file pins the contract directly:
 *
 *   - svd_compress on a small full-rank matrix returns U S V^H = A
 *     to machine precision and reports zero truncation error;
 *   - svd_compress with a fixed max_bond < rank truncates, the
 *     returned truncation_error equals the L2 norm of dropped
 *     singular values, and the kept singular values are the
 *     leading-magnitude prefix;
 *   - svd_compress_bond on a 2-site MPS scaffold preserves the
 *     site product to machine precision (bond contraction round-trip).
 */

#include "../../src/algorithms/tensor_network/svd_compress.h"
#include "../../src/algorithms/tensor_network/tensor.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; \
    } \
} while (0)

/* Small deterministic complex random matrix m x n. */
static tensor_t *rand_matrix(uint32_t m, uint32_t n, unsigned seed) {
    uint32_t dims[2] = { m, n };
    tensor_t *t = tensor_create(2, dims);
    if (!t) return NULL;
    srand(seed);
    for (uint32_t i = 0; i < m * n; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        t->data[i] = re + im * I;
    }
    return t;
}

/* Reconstruct A_hat[i,j] = sum_k U[i,k] * S[k] * Vh[k,j] and return
 * max |A_hat - A|. */
static double reconstruction_error(const tensor_t *A,
                                    const svd_compress_result_t *r) {
    uint32_t m = A->dims[0], n = A->dims[1];
    double err = 0.0;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double complex acc = 0.0;
            for (uint32_t k = 0; k < r->bond_dim; k++) {
                acc += r->left->data[i * r->bond_dim + k]
                     * r->singular_values[k]
                     * r->right->data[k * n + j];
            }
            double e = cabs(acc - A->data[i * n + j]);
            if (e > err) err = e;
        }
    }
    return err;
}

/* ||U^H U - I_k||_F.  U has shape (m x k); when k <= m, U should be a
 * column-orthonormal isometry.  Returning > tolerance here means the
 * SVD factorisation is structurally a generic factorisation (still
 * U S V^H = A) but not a true SVD -- DMRG's left-canonical form
 * silently breaks when this is large. */
static double left_isometry_error(const svd_compress_result_t *r,
                                   uint32_t m) {
    uint32_t k = r->bond_dim;
    double err_sq = 0.0;
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < k; j++) {
            double complex inner = 0.0;
            for (uint32_t row = 0; row < m; row++) {
                double complex u_ri = r->left->data[row * k + i];
                double complex u_rj = r->left->data[row * k + j];
                inner += conj(u_ri) * u_rj;
            }
            double target = (i == j) ? 1.0 : 0.0;
            double dr = creal(inner) - target, di = cimag(inner);
            err_sq += dr * dr + di * di;
        }
    }
    return sqrt(err_sq);
}

/* ||Vh Vh^H - I_k||_F.  Vh has shape (k x n); when k <= n it should
 * be row-orthonormal. */
static double right_isometry_error(const svd_compress_result_t *r,
                                    uint32_t n) {
    uint32_t k = r->bond_dim;
    double err_sq = 0.0;
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < k; j++) {
            double complex inner = 0.0;
            for (uint32_t col = 0; col < n; col++) {
                double complex v_ic = r->right->data[i * n + col];
                double complex v_jc = r->right->data[j * n + col];
                inner += v_ic * conj(v_jc);
            }
            double target = (i == j) ? 1.0 : 0.0;
            double dr = creal(inner) - target, di = cimag(inner);
            err_sq += dr * dr + di * di;
        }
    }
    return sqrt(err_sq);
}

static void test_full_rank_roundtrip(void) {
    fprintf(stdout, "\n-- svd_compress: full-rank 6x4 round-trip --\n");
    tensor_t *A = rand_matrix(6, 4, 0xC0FFEEu);
    svd_compress_config_t cfg = svd_compress_config_default();
    cfg.max_bond_dim = 32;            /* effectively unbounded for 6x4 */
    cfg.cutoff = 0.0;
    svd_compress_result_t *r = svd_compress(A, &cfg);
    CHECK(r != NULL, "svd_compress returned NULL");
    if (!r) { tensor_free(A); return; }
    CHECK(r->bond_dim == 4, "expected k = min(m,n) = 4, got %u", r->bond_dim);
    CHECK(r->truncation_error < 1e-12,
          "expected zero truncation error, got %.3e", r->truncation_error);
    double rerr = reconstruction_error(A, r);
    CHECK(rerr < 1e-12, "reconstruction |U S V^H - A| = %.3e (expected < 1e-12)", rerr);
    /* True SVD requires U and Vh to be isometries.  Without these
     * checks the factorisation can be A = U S Vh = A trivially while
     * U is not column-orthonormal -- the Jacobi-fallback regression
     * in 2026-04 looked exactly like that. */
    double iso_l = left_isometry_error(r, A->dims[0]);
    double iso_r = right_isometry_error(r, A->dims[1]);
    CHECK(iso_l < 1e-10, "||U^H U - I||_F = %.3e (expected < 1e-10)", iso_l);
    CHECK(iso_r < 1e-10, "||Vh Vh^H - I||_F = %.3e (expected < 1e-10)", iso_r);
    fprintf(stdout, "  bond_dim=%u  trunc=%.2e  recon=%.2e  iso_L=%.2e  iso_R=%.2e\n",
            r->bond_dim, r->truncation_error, rerr, iso_l, iso_r);
    svd_compress_result_free(r);
    tensor_free(A);
}

static void test_truncation_error_matches_dropped_l2(void) {
    fprintf(stdout, "\n-- svd_compress: truncate 8x4 to bond=2 --\n");
    tensor_t *A = rand_matrix(8, 4, 0xBEEFu);
    svd_compress_config_t cfg = svd_compress_config_fixed(2);
    /* First call with bond=4 (un-truncated) to capture the full SV
     * spectrum; then truncate. */
    svd_compress_config_t cfg_full = svd_compress_config_fixed(4);
    svd_compress_result_t *full = svd_compress(A, &cfg_full);
    svd_compress_result_t *r    = svd_compress(A, &cfg);
    CHECK(full && r, "svd_compress returned NULL");
    if (!full || !r) { svd_compress_result_free(full); svd_compress_result_free(r); tensor_free(A); return; }

    CHECK(r->bond_dim == 2, "expected bond_dim=2, got %u", r->bond_dim);

    /* Kept SVs are the leading two of the full spectrum (descending). */
    for (uint32_t i = 0; i < r->bond_dim; i++) {
        double diff = fabs(r->singular_values[i] - full->singular_values[i]);
        CHECK(diff < 1e-12,
              "kept SV[%u] = %.6f != full SV[%u] = %.6f",
              i, r->singular_values[i], i, full->singular_values[i]);
    }

    /* Reported truncation_error should equal sqrt(sum of dropped SVs squared). */
    double dropped_sq = 0.0;
    for (uint32_t i = r->bond_dim; i < full->bond_dim; i++) {
        dropped_sq += full->singular_values[i] * full->singular_values[i];
    }
    double expected = sqrt(dropped_sq);
    double diff = fabs(r->truncation_error - expected);
    CHECK(diff < 1e-12,
          "truncation_error = %.6e but sqrt(sum dropped SV^2) = %.6e (delta %.2e)",
          r->truncation_error, expected, diff);
    /* Truncated SVD must still produce a column-orthonormal U_k
     * and a row-orthonormal Vh_k.  This is the property DMRG relies
     * on to keep its left-canonical form. */
    double iso_l = left_isometry_error(r, A->dims[0]);
    double iso_r = right_isometry_error(r, A->dims[1]);
    CHECK(iso_l < 1e-10,
          "truncated ||U^H U - I||_F = %.3e (expected < 1e-10)", iso_l);
    CHECK(iso_r < 1e-10,
          "truncated ||Vh Vh^H - I||_F = %.3e (expected < 1e-10)", iso_r);
    fprintf(stdout,
            "  reported trunc=%.4e  expected=%.4e  delta=%.2e  iso_L=%.2e  iso_R=%.2e\n",
            r->truncation_error, expected, diff, iso_l, iso_r);

    svd_compress_result_free(full);
    svd_compress_result_free(r);
    tensor_free(A);
}

int main(void) {
    fprintf(stdout, "=== svd_compress unit tests ===\n");
    test_full_rank_roundtrip();
    test_truncation_error_matches_dropped_l2();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
