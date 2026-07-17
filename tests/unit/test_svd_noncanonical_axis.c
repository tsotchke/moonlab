/**
 * @file test_svd_noncanonical_axis.c
 * @brief Regression for svd_left_canonicalize with bond_axis != rank-1.
 *
 * The truncation branch used to re-reshape the *untransposed* input tensor
 * with the transposed row/col dimensions, scrambling the data whenever the
 * bond axis was not the last axis.  This test left-canonicalizes a rank-3
 * tensor across a middle axis with a truncating config and asserts:
 *   (a) the returned left factor is a left-isometry (columns orthonormal);
 *   (b) left * right reconstructs the transposed input to within the
 *       discarded-weight truncation error.
 */

#include "../../src/algorithms/tensor_network/svd_compress.h"
#include "../../src/algorithms/tensor_network/tensor.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

int main(void) {
    fprintf(stdout, "=== svd_left_canonicalize non-last-axis regression ===\n");

    /* Rank-3 tensor T[a, b, c] with the BOND axis in the middle (axis 1).
     * Choose dims so the row index (a*c) exceeds the requested cap, forcing
     * the truncation (SVD) branch. */
    const uint32_t da = 4, db = 5, dc = 3;   /* bond axis = 1 (size db) */
    uint32_t dims[3] = { da, db, dc };
    tensor_t *T = tensor_create(3, dims);
    CHECK(T != NULL, "tensor_create");
    if (!T) return 1;

    srand(0x51D);
    for (uint32_t a = 0; a < da; a++)
        for (uint32_t b = 0; b < db; b++)
            for (uint32_t c = 0; c < dc; c++) {
                uint32_t idx[3] = { a, b, c };
                double re = (double)rand() / RAND_MAX - 0.5;
                double im = (double)rand() / RAND_MAX - 0.5;
                tensor_set(T, idx, re + im * I);
            }

    /* Row index is a*c = 12, bond is db = 5 -> min side is 5.  Cap at 3 to
     * force truncation. */
    svd_compress_config_t cfg = svd_compress_config_default();
    cfg.max_bond_dim = 3;
    cfg.cutoff = 0.0;

    svd_compress_result_t *res = svd_left_canonicalize(T, /*bond_axis=*/1, &cfg);
    CHECK(res != NULL, "svd_left_canonicalize returned result");
    if (!res) { tensor_free(T); return 1; }

    /* left factor has shape [row-axes..., new_bond]; for bond_axis=1 the row
     * axes are (a, c) in order, so left is [da, dc, k]. */
    uint32_t k = res->bond_dim;
    CHECK(res->left != NULL && res->right != NULL, "left/right non-NULL");
    CHECK(k > 0 && k <= 3, "bond_dim %u within cap", k);
    fprintf(stdout, "  new bond dim k = %u (cap 3), trunc_err = %.3e\n",
            k, res->truncation_error);

    uint32_t rows = da * dc;

    /* (a) Isometry: sum_row conj(U[row,i]) U[row,j] = delta_ij.
     * left is [da, dc, k] row-major, so U[(a*dc+c), i] = left[a,c,i]. */
    double iso_err = 0.0;
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < k; j++) {
            double complex acc = 0.0;
            for (uint32_t row = 0; row < rows; row++) {
                double complex u_i = res->left->data[row * k + i];
                double complex u_j = res->left->data[row * k + j];
                acc += conj(u_i) * u_j;
            }
            double expect = (i == j) ? 1.0 : 0.0;
            double e = cabs(acc - expect);
            if (e > iso_err) iso_err = e;
        }
    }
    fprintf(stdout, "  max |U^dag U - I| = %.3e\n", iso_err);
    CHECK(iso_err < 1e-9, "left factor not an isometry: %.3e", iso_err);

    /* (b) Reconstruction: (left * right) must reproduce the TRANSPOSED input
     * T'[a,c,b] to within the truncation error.  left is [rows, k], right is
     * [k, db]; product M[row, b] compared against T[a, b, c] with
     * row = a*dc + c. */
    double recon_err = 0.0;
    for (uint32_t a = 0; a < da; a++) {
        for (uint32_t c = 0; c < dc; c++) {
            uint32_t row = a * dc + c;
            for (uint32_t b = 0; b < db; b++) {
                double complex acc = 0.0;
                for (uint32_t i = 0; i < k; i++) {
                    acc += res->left->data[row * k + i] * res->right->data[i * db + b];
                }
                uint32_t idx[3] = { a, b, c };
                double complex orig = tensor_get(T, idx);
                double e = cabs(acc - orig);
                if (e > recon_err) recon_err = e;
            }
        }
    }
    fprintf(stdout, "  max |left*right - T| = %.3e (trunc_err %.3e)\n",
            recon_err, res->truncation_error);
    /* Reconstruction error is bounded by the discarded weight; allow a small
     * slack over the reported Frobenius truncation error. */
    CHECK(recon_err < res->truncation_error + 1e-9,
          "reconstruction error %.3e exceeds truncation error %.3e",
          recon_err, res->truncation_error);

    svd_compress_result_free(res);
    tensor_free(T);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
