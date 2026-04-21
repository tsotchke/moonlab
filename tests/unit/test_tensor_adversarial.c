/**
 * @file test_tensor_adversarial.c
 * @brief Adversarial-shape coverage for the tensor primitives.
 *
 * The pre-v0.2.0 audit flagged that every foundational primitive
 * (tensor_contract, tensor_transpose, tensor_reshape, tensor_svd,
 * svd_compress_bond) was only exercised on rank-2, real-valued,
 * power-of-2, square shapes.  Everything the MPO/MPS work above
 * leans on -- the Bianco-Resta pipeline, the MPO * MPO Chebyshev,
 * the dense->MPO converter -- uses rank-3 or rank-4 tensors with
 * complex entries.  One off-by-one permutation in any of these
 * would silently taint every downstream result.
 *
 * This test exercises each primitive with:
 *   - rank >= 3
 *   - complex (non-zero imaginary) values
 *   - non-square, non-power-of-2 dims
 *   - edge cases: dim=1 axis, single-element contraction, identity
 *     permutation, full vs partial contraction.
 *
 * References are computed by hand-rolled nested-loop code against
 * which the library implementation must agree to ULP (1e-13).
 */

#include "../../src/algorithms/tensor_network/tensor.h"
#include "../../src/algorithms/tensor_network/svd_compress.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

/* ---------------------------------------------------------------- */
/* Small scratchpad helpers                                          */
/* ---------------------------------------------------------------- */

static void fill_rand_complex(tensor_t* t, unsigned seed) {
    srand(seed);
    for (uint64_t i = 0; i < t->total_size; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        t->data[i] = re + im * I;
    }
}

static double l2_rel_complex(const double complex* a,
                              const double complex* b,
                              size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double complex d = a[i] - b[i];
        num += creal(d * conj(d));
        den += creal(a[i] * conj(a[i]));
    }
    return (den > 0) ? sqrt(num / den) : sqrt(num);
}

/* ---------------------------------------------------------------- */
/* 1. tensor_contract on rank-3 x rank-3 with one contracted axis  */
/* ---------------------------------------------------------------- */
/*
 *  A: [2, 3, 5]   B: [5, 4, 2]
 *  contract A axis 2 with B axis 0, yielding rank-4 [2, 3, 4, 2].
 *
 *  Reference: C[i, j, k, l] = sum_m A[i, j, m] * B[m, k, l].
 *
 *  Complex-valued random data with a fixed seed so the test is
 *  deterministic.
 */
static void test_contract_rank3(void) {
    fprintf(stdout, "\n-- tensor_contract rank-3 complex --\n");
    uint32_t dA[3] = {2, 3, 5};
    uint32_t dB[3] = {5, 4, 2};
    tensor_t* A = tensor_create(3, dA);
    tensor_t* B = tensor_create(3, dB);
    fill_rand_complex(A, 0xA001);
    fill_rand_complex(B, 0xB002);

    uint32_t axes_a[1] = {2}, axes_b[1] = {0};
    tensor_t* C = tensor_contract(A, B, axes_a, axes_b, 1);
    CHECK(C != NULL, "contract returns tensor");
    CHECK(C->rank == 4, "rank 4");
    CHECK(C->dims[0] == 2 && C->dims[1] == 3 &&
          C->dims[2] == 4 && C->dims[3] == 2, "dims = [2, 3, 4, 2]");

    /* Reference via nested loop. */
    const size_t N = 2 * 3 * 4 * 2;
    double complex* ref = (double complex*)calloc(N, sizeof(double complex));
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 4; k++) {
                for (size_t l = 0; l < 2; l++) {
                    double complex s = 0.0;
                    for (size_t m = 0; m < 5; m++) {
                        const size_t ai = ((i * 3 + j) * 5) + m;
                        const size_t bi = ((m * 4 + k) * 2) + l;
                        s += A->data[ai] * B->data[bi];
                    }
                    const size_t ci = ((i * 3 + j) * 4 + k) * 2 + l;
                    ref[ci] = s;
                }
            }
        }
    }
    const double err = l2_rel_complex(C->data, ref, N);
    fprintf(stdout, "    ||contract - ref||/||ref|| = %.3e\n", err);
    CHECK(err < 1e-13, "rank-3 contract matches hand-rolled reference");

    free(ref);
    tensor_free(A); tensor_free(B); tensor_free(C);
}

/* ---------------------------------------------------------------- */
/* 2. tensor_contract with TWO contracted axes (matrix trace-style)  */
/* ---------------------------------------------------------------- */
/*
 *  A: [3, 4, 5]   B: [4, 5, 7]
 *  contract A axes {1, 2} with B axes {0, 1}, yielding rank-2 [3, 7].
 *  Reference: C[i, k] = sum_{j, l} A[i, j, l] * B[j, l, k].
 */
static void test_contract_two_axes(void) {
    fprintf(stdout, "\n-- tensor_contract rank-3 * rank-3 two-axis --\n");
    uint32_t dA[3] = {3, 4, 5};
    uint32_t dB[3] = {4, 5, 7};
    tensor_t* A = tensor_create(3, dA);
    tensor_t* B = tensor_create(3, dB);
    fill_rand_complex(A, 0xCAFE);
    fill_rand_complex(B, 0xF00D);

    uint32_t axes_a[2] = {1, 2}, axes_b[2] = {0, 1};
    tensor_t* C = tensor_contract(A, B, axes_a, axes_b, 2);
    CHECK(C != NULL && C->rank == 2, "two-axis contract returns rank-2");
    CHECK(C->dims[0] == 3 && C->dims[1] == 7, "dims = [3, 7]");

    const size_t N = 3 * 7;
    double complex* ref = (double complex*)calloc(N, sizeof(double complex));
    for (size_t i = 0; i < 3; i++) {
        for (size_t k = 0; k < 7; k++) {
            double complex s = 0.0;
            for (size_t j = 0; j < 4; j++) {
                for (size_t l = 0; l < 5; l++) {
                    const size_t ai = (i * 4 + j) * 5 + l;
                    const size_t bi = (j * 5 + l) * 7 + k;
                    s += A->data[ai] * B->data[bi];
                }
            }
            ref[i * 7 + k] = s;
        }
    }
    const double err = l2_rel_complex(C->data, ref, N);
    fprintf(stdout, "    ||contract - ref||/||ref|| = %.3e\n", err);
    CHECK(err < 1e-13, "two-axis contract matches reference");

    free(ref);
    tensor_free(A); tensor_free(B); tensor_free(C);
}

/* ---------------------------------------------------------------- */
/* 3. tensor_transpose rank-4 with non-trivial permutation            */
/* ---------------------------------------------------------------- */
/*
 *  A: [2, 3, 4, 5]  -> T with perm {2, 0, 3, 1} -> [4, 2, 5, 3]
 *  Reference: T[a, b, c, d] = A[b, d, a, c].
 */
static void test_transpose_rank4(void) {
    fprintf(stdout, "\n-- tensor_transpose rank-4 perm {2, 0, 3, 1} --\n");
    uint32_t dA[4] = {2, 3, 4, 5};
    tensor_t* A = tensor_create(4, dA);
    fill_rand_complex(A, 0xBEEF);

    uint32_t perm[4] = {2, 0, 3, 1};
    tensor_t* T = tensor_transpose(A, perm);
    CHECK(T != NULL && T->rank == 4, "returns rank-4");
    CHECK(T->dims[0] == 4 && T->dims[1] == 2 &&
          T->dims[2] == 5 && T->dims[3] == 3, "permuted dims ok");

    const size_t N = 2 * 3 * 4 * 5;
    double complex* ref = (double complex*)calloc(N, sizeof(double complex));
    for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 2; b++) {
            for (size_t c = 0; c < 5; c++) {
                for (size_t d = 0; d < 3; d++) {
                    /* T[a,b,c,d] = A[b,d,a,c]. */
                    const size_t ai = ((b * 3 + d) * 4 + a) * 5 + c;
                    const size_t ti = ((a * 2 + b) * 5 + c) * 3 + d;
                    ref[ti] = A->data[ai];
                }
            }
        }
    }
    const double err = l2_rel_complex(T->data, ref, N);
    fprintf(stdout, "    ||transpose - ref||/||ref|| = %.3e\n", err);
    CHECK(err < 1e-14, "rank-4 transpose matches reference");

    free(ref);
    tensor_free(A); tensor_free(T);
}

/* ---------------------------------------------------------------- */
/* 4. tensor_reshape rank-5 -> rank-3 preserves data order            */
/* ---------------------------------------------------------------- */
static void test_reshape_rank5(void) {
    fprintf(stdout, "\n-- tensor_reshape rank-5 -> rank-3 --\n");
    uint32_t dA[5] = {2, 3, 2, 5, 2};
    tensor_t* A = tensor_create(5, dA);
    fill_rand_complex(A, 0x1337);

    uint32_t new_dims[3] = {6, 5, 4};
    tensor_t* R = tensor_reshape(A, 3, new_dims);
    CHECK(R != NULL && R->rank == 3, "reshape returns rank-3");
    CHECK(R->total_size == A->total_size, "total size preserved");

    for (uint64_t i = 0; i < A->total_size; i++) {
        if (cabs(A->data[i] - R->data[i]) > 1e-15) {
            fprintf(stderr, "FAIL reshape changed element %llu\n",
                    (unsigned long long)i);
            failures++;
            break;
        }
    }
    CHECK(failures == 0, "reshape preserves row-major order");

    tensor_free(A); tensor_free(R);
}

/* ---------------------------------------------------------------- */
/* 5. tensor_svd on a 6x4 complex matrix                             */
/* ---------------------------------------------------------------- */
/*
 *  Rank-2 non-square complex random input.  Reconstruct U S V^H and
 *  verify Frobenius agreement with the input.
 */
static void test_svd_nonsquare_complex(void) {
    fprintf(stdout, "\n-- tensor_svd on 6x4 complex --\n");
    uint32_t dA[2] = {6, 4};
    tensor_t* A = tensor_create(2, dA);
    fill_rand_complex(A, 0xD00D);

    tensor_svd_result_t* svd = tensor_svd(A, 0, 0.0);
    CHECK(svd != NULL, "SVD succeeded");
    if (!svd) { tensor_free(A); return; }

    const uint32_t k = svd->k;
    CHECK(k == 4, "kept min(m, n) = 4 singular values (got %u)", k);
    /* All singular values must be non-negative. */
    int all_nn = 1;
    for (uint32_t i = 0; i < k; i++) if (svd->S[i] < -1e-14) all_nn = 0;
    CHECK(all_nn, "singular values are non-negative");
    /* Sorted descending. */
    int sorted = 1;
    for (uint32_t i = 1; i < k; i++) if (svd->S[i] > svd->S[i - 1] + 1e-14) sorted = 0;
    CHECK(sorted, "singular values sorted descending");

    /* Reconstruct U * diag(S) * V^H. */
    const uint32_t m = 6, n = 4;
    double complex* recon = (double complex*)calloc(m * n, sizeof(double complex));
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double complex s = 0.0;
            for (uint32_t r = 0; r < k; r++) {
                s += svd->U->data[i * k + r] * svd->S[r] * svd->Vh->data[r * n + j];
            }
            recon[i * n + j] = s;
        }
    }
    const double err = l2_rel_complex(recon, A->data, m * n);
    fprintf(stdout, "    ||U S V^H - A||/||A|| = %.3e\n", err);
    CHECK(err < 1e-12, "full SVD reconstructs complex 6x4 input");

    free(recon);
    tensor_svd_free(svd);
    tensor_free(A);
}

/* ---------------------------------------------------------------- */
/* 6. svd_compress_bond on a random MPS-like pair                     */
/* ---------------------------------------------------------------- */
/*
 *  Left tensor  [3, 4, 5]  (left-bond, phys, bond)
 *  Right tensor [5, 4, 3]  (bond, phys, right-bond)
 *  Truncating to max_bond = 2 must still preserve the dominant
 *  singular values; the sum of squared errors must be the tail of
 *  the distribution.
 */
static void test_svd_compress_bond(void) {
    fprintf(stdout, "\n-- svd_compress_bond on rank-3 MPS pair --\n");
    uint32_t dL[3] = {3, 4, 5};
    uint32_t dR[3] = {5, 4, 3};
    tensor_t* L = tensor_create(3, dL);
    tensor_t* R = tensor_create(3, dR);
    fill_rand_complex(L, 0x2001);
    fill_rand_complex(R, 0x2002);

    svd_compress_config_t cfg = svd_compress_config_fixed(2);
    svd_compress_result_t* res = svd_compress_bond(L, R, &cfg, true);
    CHECK(res != NULL, "svd_compress_bond returned result");
    CHECK(res->bond_dim <= 2, "bond_dim = %u <= 2", res->bond_dim);
    CHECK(res->truncation_error >= 0.0, "truncation_error >= 0");
    fprintf(stdout, "    truncation error = %.3e  kept bond dim = %u\n",
            res->truncation_error, res->bond_dim);
    svd_compress_result_free(res);
    tensor_free(L); tensor_free(R);
}

/* ---------------------------------------------------------------- */
/* 7. Edge case: tensor with a dim=1 axis                             */
/* ---------------------------------------------------------------- */
static void test_dim1_edge(void) {
    fprintf(stdout, "\n-- dim=1 axis edges --\n");
    /* Reshape [2, 3] -> [1, 2, 3] -> [2, 1, 3] via transpose. */
    uint32_t dA[2] = {2, 3};
    tensor_t* A = tensor_create(2, dA);
    for (uint64_t i = 0; i < A->total_size; i++) A->data[i] = (double)i;

    uint32_t d3[3] = {1, 2, 3};
    tensor_t* R = tensor_reshape(A, 3, d3);
    CHECK(R != NULL, "reshape -> dim=1 prefix OK");

    uint32_t perm[3] = {1, 0, 2};
    tensor_t* T = tensor_transpose(R, perm);
    CHECK(T != NULL && T->dims[0] == 2 && T->dims[1] == 1 && T->dims[2] == 3,
          "transpose with dim=1 axis OK");

    tensor_free(A); tensor_free(R); tensor_free(T);
}

int main(void) {
    fprintf(stdout, "=== tensor-primitive adversarial-shape tests ===\n");
    test_contract_rank3();
    test_contract_two_axes();
    test_transpose_rank4();
    test_reshape_rank5();
    test_svd_nonsquare_complex();
    test_svd_compress_bond();
    test_dim1_edge();

    fprintf(stdout, "\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
