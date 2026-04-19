/**
 * @file test_mpo_kpm.c
 * @brief Validate the MPO-level Chebyshev / Jackson KPM module
 *        against a dense eigendecomposition reference.
 *
 * We run three classes of check:
 *
 *   1. Kernel / coefficient arithmetic:
 *        - Jackson weights are monotone, non-negative, and g_0 ~ 1.
 *        - Sign-function Chebyshev coefficients have the right parity
 *          (c_0 = 0, c_{even} = 0 for n > 0, c_1 = 4/pi).
 *
 *   2. MPS linear-combination helper:
 *        Build |A>, |B> as random product states, form
 *        |C> = alpha |A> + beta |B> via mpo_kpm_mps_combine, and
 *        verify <C|C> = |alpha|^2 <A|A> + alpha* beta <A|B>
 *                    + alpha beta* <B|A> + |beta|^2 <B|B>.
 *
 *   3. End-to-end Chebyshev-KPM sign matrix elements:
 *        On a small transverse-field Ising chain (L = 4, J = 1, h = 0.8),
 *        compute <phi|sign(H_tilde)|psi> via the MPO-KPM pipeline and
 *        via dense LAPACK zheev eigendecomposition of H_tilde, and
 *        compare.  N_c = 160 gets us into the 1e-3 band (the Jackson
 *        regularisation has finite resolution near the gap; our
 *        random bras/kets have weight on all eigenstates).
 */

#include "../../src/algorithms/topology_realspace/mpo_kpm.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/mpo_2d.h"  /* mpo_to_matrix */
#include "../../src/algorithms/tensor_network/tensor.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void zheev_(char *jobz, char *uplo, int *n, double complex *a, int *lda,
                   double *w, double complex *work, int *lwork,
                   double *rwork, int *info);
#endif

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

/* ---------------------------------------------------------------- */
/* Random product-state MPS helper                                  */
/* ---------------------------------------------------------------- */
static tn_mps_state_t* random_product_mps(uint32_t L, unsigned seed) {
    srand(seed);
    double complex (*states)[2] =
        (double complex (*)[2])malloc(L * sizeof(double complex[2]));
    if (!states) return NULL;
    for (uint32_t i = 0; i < L; i++) {
        const double a = (double)rand() / RAND_MAX - 0.5;
        const double b = (double)rand() / RAND_MAX - 0.5;
        const double c = (double)rand() / RAND_MAX - 0.5;
        const double d = (double)rand() / RAND_MAX - 0.5;
        double complex v0 = a + b * I;
        double complex v1 = c + d * I;
        const double nrm = sqrt(creal(v0 * conj(v0) + v1 * conj(v1)));
        states[i][0] = v0 / nrm;
        states[i][1] = v1 / nrm;
    }
    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = 64;
    tn_mps_state_t* mps = tn_mps_create_product(L, states, &cfg);
    free(states);
    return mps;
}

/* ---------------------------------------------------------------- */
/* 1. Kernel / coefficient arithmetic                                */
/* ---------------------------------------------------------------- */
static void test_coefficients(void) {
    fprintf(stdout, "\n-- coefficients --\n");

    /* Jackson kernel: g_0 close to 1, weights positive, monotone
     * non-increasing for small n. */
    size_t N = 64;
    double* g = (double*)malloc(N * sizeof(double));
    mpo_kpm_jackson_weights(N, g);
    CHECK(fabs(g[0] - 1.0) < 1e-12, "g_0 = 1 (got %.12f)", g[0]);
    int all_pos = 1;
    int monotone = 1;
    for (size_t n = 0; n < N; n++) if (g[n] < -1e-15) all_pos = 0;
    for (size_t n = 1; n < N; n++) if (g[n] > g[n - 1] + 1e-12) monotone = 0;
    CHECK(all_pos, "Jackson weights non-negative");
    CHECK(monotone, "Jackson weights monotone non-increasing");
    free(g);

    /* Sign coefficients: c_0 = 0, c_1 = 4/pi, c_even = 0, c_3 = -4/(3 pi). */
    double* c = (double*)malloc(N * sizeof(double));
    mpo_kpm_sign_coefficients(N, c);
    CHECK(fabs(c[0]) < 1e-15, "c_0 = 0");
    CHECK(fabs(c[1] - 4.0 / M_PI) < 1e-12, "c_1 = 4/pi");
    CHECK(fabs(c[2]) < 1e-15, "c_2 = 0");
    CHECK(fabs(c[3] + 4.0 / (3.0 * M_PI)) < 1e-12, "c_3 = -4/(3 pi)");
    CHECK(fabs(c[4]) < 1e-15, "c_4 = 0");
    CHECK(fabs(c[5] - 4.0 / (5.0 * M_PI)) < 1e-12, "c_5 = +4/(5 pi)");
    free(c);
}

/* ---------------------------------------------------------------- */
/* 2. MPS linear combination                                         */
/* ---------------------------------------------------------------- */
static void test_mps_combine_norm(uint32_t L, unsigned seedA, unsigned seedB) {
    fprintf(stdout, "\n-- mps combine norm identity: L=%u --\n", L);
    tn_mps_state_t* A = random_product_mps(L, seedA);
    tn_mps_state_t* B = random_product_mps(L, seedB);
    CHECK(A && B, "random product states created");

    const double complex alpha = 0.7 + 0.3 * I;
    const double complex beta  = -0.2 + 1.1 * I;

    const double complex AA = tn_mps_overlap(A, A);
    const double complex BB = tn_mps_overlap(B, B);
    const double complex AB = tn_mps_overlap(A, B);
    const double complex BA = tn_mps_overlap(B, A);

    const double complex pred = conj(alpha) * alpha * AA
                              + conj(alpha) * beta  * AB
                              + conj(beta)  * alpha * BA
                              + conj(beta)  * beta  * BB;

    tn_mps_state_t* C = mpo_kpm_mps_combine(A, alpha, B, beta);
    CHECK(C != NULL, "combine returns non-null");
    const double complex CC = tn_mps_overlap(C, C);

    const double err = cabs(CC - pred);
    fprintf(stdout, "    <C|C> = %.6f (expected %.6f, diff = %.3e)\n",
            creal(CC), creal(pred), err);
    CHECK(err < 1e-10, "<C|C> matches analytic bilinear form");

    tn_mps_free(A);
    tn_mps_free(B);
    tn_mps_free(C);
}

/* ---------------------------------------------------------------- */
/* 3. Dense reference pipeline                                       */
/* ---------------------------------------------------------------- */

/* Compute dense H matrix from mpo_t via mpo_to_matrix. */
static double complex* build_dense_H(const mpo_t* mpo, size_t* n_out) {
    tensor_t* H = mpo_to_matrix(mpo);
    if (!H) return NULL;
    const size_t N = H->dims[0];
    *n_out = N;
    double complex* dense = (double complex*)malloc(N * N * sizeof(double complex));
    if (!dense) { tensor_free(H); return NULL; }
    memcpy(dense, H->data, N * N * sizeof(double complex));
    tensor_free(H);
    return dense;
}

/* Diagonalise a column-major Hermitian matrix in place; on return
 * @p A holds eigenvectors as columns and @p w holds eigenvalues. */
static int diagonalise_hermitian(double complex* A, int n, double* w) {
    char jobz = 'V', uplo = 'U';
    int lda = n, lwork = -1, info = 0;
    double* rwork = (double*)malloc((size_t)(3 * n) * sizeof(double));
    double complex work_query;
#ifdef __APPLE__
    __CLPK_integer n_c = n, lda_c = lda, lwork_c = lwork, info_c = 0;
    __CLPK_doublecomplex wq;
    zheev_(&jobz, &uplo, &n_c, (__CLPK_doublecomplex*)A, &lda_c,
           w, &wq, &lwork_c, rwork, &info_c);
    memcpy(&work_query, &wq, sizeof(work_query));
    lwork = (int)creal(work_query);
    info = (int)info_c;
#else
    zheev_(&jobz, &uplo, &n, A, &lda, w, &work_query, &lwork, rwork, &info);
    lwork = (int)creal(work_query);
#endif
    if (info != 0) { free(rwork); return info; }
    double complex* work = (double complex*)malloc((size_t)lwork * sizeof(double complex));
    if (!work) { free(rwork); return -1; }
#ifdef __APPLE__
    n_c = n; lda_c = lda; lwork_c = lwork;
    zheev_(&jobz, &uplo, &n_c, (__CLPK_doublecomplex*)A, &lda_c,
           w, (__CLPK_doublecomplex*)work, &lwork_c, rwork, &info_c);
    info = (int)info_c;
#else
    zheev_(&jobz, &uplo, &n, A, &lda, w, work, &lwork, rwork, &info);
#endif
    free(work);
    free(rwork);
    return info;
}

/* Dense <phi|sign(H_tilde)|psi> with H_tilde = (H - a I)/b. */
static double complex dense_sign_matrix_element(
    const double complex* H_rowmajor, size_t N,
    double shift, double scale,
    const double complex* phi,
    const double complex* psi)
{
    /* LAPACK expects column-major Hermitian matrices, but since
     * zheev only looks at the upper triangle (jobz='V', uplo='U')
     * and H is Hermitian, row- and column-major views of the upper
     * triangle give the same upper-triangle entries. Copy the matrix
     * and rescale by 1/scale, with -shift/scale on the diagonal. */
    double complex* M = (double complex*)malloc(N * N * sizeof(double complex));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            M[i * N + j] = H_rowmajor[i * N + j] / scale;
        }
        M[i * N + i] -= shift / scale;
    }
    double* w = (double*)malloc(N * sizeof(double));
    int info = diagonalise_hermitian(M, (int)N, w);
    if (info != 0) {
        fprintf(stderr, "    zheev info=%d\n", info);
        free(M); free(w);
        return 0.0;
    }

    /* M holds eigenvectors column-major: V[i, k] = M[k * N + i].
     * <phi|v_k> = sum_i conj(phi_i) V[i, k],
     * <v_k|psi> = sum_i conj(V[i, k]) psi_i. */
    double complex acc = 0.0;
    for (size_t k = 0; k < N; k++) {
        double complex phi_vk = 0.0;
        double complex vk_psi = 0.0;
        for (size_t i = 0; i < N; i++) {
            phi_vk += conj(phi[i]) * M[k * N + i];
            vk_psi += conj(M[k * N + i]) * psi[i];
        }
        const double s = (w[k] > 0.0) ? 1.0 : (w[k] < 0.0 ? -1.0 : 0.0);
        acc += s * phi_vk * vk_psi;
    }
    free(M);
    free(w);
    return acc;
}

static void test_sign_matrix_element_small_tfim(void) {
    fprintf(stdout, "\n-- sign MTX element vs dense: L=4 TFIM --\n");
    const uint32_t L = 4;
    const double J = 1.0;
    const double h = 0.8;

    mpo_t* H_src = mpo_tfim_create(L, J, h);
    CHECK(H_src != NULL, "TFIM MPO built");

    tn_mpo_t* H = mpo_kpm_mpo_to_tn_mpo(H_src);
    CHECK(H != NULL, "MPO adapted to tn_mpo_t");

    /* Dense reference. */
    size_t N = 0;
    double complex* Hd = build_dense_H(H_src, &N);
    CHECK(Hd != NULL, "dense H built");
    CHECK(N == ((size_t)1 << L), "dense H has dimension 2^L = %zu", N);

    /* Spectrum bounds: diagonalise a scratch copy. */
    double complex* scratch = (double complex*)malloc(N * N * sizeof(double complex));
    memcpy(scratch, Hd, N * N * sizeof(double complex));
    double* evals = (double*)malloc(N * sizeof(double));
    int info = diagonalise_hermitian(scratch, (int)N, evals);
    CHECK(info == 0, "dense diagonalisation ok");
    const double E_min = evals[0];
    const double E_max = evals[N - 1];
    fprintf(stdout, "    E_min = %+.4f  E_max = %+.4f\n", E_min, E_max);
    free(scratch);
    free(evals);

    /* Rescale so that H_tilde = (H - a)/b has spectrum in [-0.95, 0.95]. */
    const double a = 0.5 * (E_max + E_min);
    const double b = 0.55 * (E_max - E_min); /* slight padding from 0.5 */
    fprintf(stdout, "    shift a = %+.4f  scale b = %+.4f\n", a, b);

    /* Build random bra / ket as product states on L qubits. */
    tn_mps_state_t* bra = random_product_mps(L, 0xA53F);
    tn_mps_state_t* ket = random_product_mps(L, 0x17E1);
    CHECK(bra && ket, "random bra/ket built");

    /* Full state vectors for the dense reference. */
    double complex* phi = (double complex*)malloc(N * sizeof(double complex));
    double complex* psi = (double complex*)malloc(N * sizeof(double complex));
    CHECK(tn_mps_to_statevector(bra, phi) == TN_STATE_SUCCESS, "bra -> SV");
    CHECK(tn_mps_to_statevector(ket, psi) == TN_STATE_SUCCESS, "ket -> SV");

    /* Sanity check: MPS overlap must match dense <phi|psi>. */
    {
        double complex o_mps = tn_mps_overlap(bra, ket);
        double complex o_dense = 0.0;
        for (size_t i = 0; i < N; i++) o_dense += conj(phi[i]) * psi[i];
        fprintf(stdout, "    <bra|ket> MPS  = %+.6f%+.6fi\n",
                creal(o_mps), cimag(o_mps));
        fprintf(stdout, "    <bra|ket> dense= %+.6f%+.6fi\n",
                creal(o_dense), cimag(o_dense));
        CHECK(cabs(o_mps - o_dense) < 1e-10,
              "MPS and dense agree on <bra|ket>");
    }

    /* Sanity check: <phi|H|psi> dense vs MPS <bra|H|ket>. */
    {
        /* Dense: Hp = H|psi>, me = sum_i conj(phi_i) Hp_i */
        double complex* Hp = (double complex*)calloc(N, sizeof(double complex));
        for (size_t i = 0; i < N; i++) {
            double complex acc = 0.0;
            for (size_t j = 0; j < N; j++) acc += Hd[i * N + j] * psi[j];
            Hp[i] = acc;
        }
        double complex me_H_dense = 0.0;
        for (size_t i = 0; i < N; i++) me_H_dense += conj(phi[i]) * Hp[i];
        free(Hp);

        /* MPS: apply H to ket, then <bra|Hket>. */
        mpo_kpm_params_t p1 = mpo_kpm_params_default();
        p1.n_cheby = 2; p1.E_shift = 0; p1.E_scale = 1.0; p1.max_bond_dim = 64;
        p1.use_jackson = 0;
        double complex* m2 = (double complex*)malloc(2 * sizeof(double complex));
        mpo_kpm_chebyshev_moments(H, bra, ket, &p1, m2);
        double complex me_H_mps = m2[1]; /* T_1(H) = H when shift=0, scale=1 */
        free(m2);

        fprintf(stdout, "    <bra|H|ket> MPS  = %+.6f%+.6fi\n",
                creal(me_H_mps), cimag(me_H_mps));
        fprintf(stdout, "    <bra|H|ket> dense= %+.6f%+.6fi\n",
                creal(me_H_dense), cimag(me_H_dense));
        CHECK(cabs(me_H_mps - me_H_dense) < 1e-9,
              "MPS and dense agree on <bra|H|ket>");
    }

    /* Sanity check: <bra|T_2(H_tilde)|ket> where T_2(x) = 2x^2 - 1 and
     * H_tilde = H/b.  Dense: me = 2/b^2 <phi|H^2|psi> - <phi|psi>. */
    {
        double complex* Hp  = (double complex*)calloc(N, sizeof(double complex));
        double complex* HHp = (double complex*)calloc(N, sizeof(double complex));
        for (size_t i = 0; i < N; i++) {
            double complex acc = 0.0;
            for (size_t j = 0; j < N; j++) acc += Hd[i * N + j] * psi[j];
            Hp[i] = acc;
        }
        for (size_t i = 0; i < N; i++) {
            double complex acc = 0.0;
            for (size_t j = 0; j < N; j++) acc += Hd[i * N + j] * Hp[j];
            HHp[i] = acc;
        }
        double complex ov = 0.0;
        for (size_t i = 0; i < N; i++) ov += conj(phi[i]) * psi[i];
        double complex phi_HH_psi = 0.0;
        for (size_t i = 0; i < N; i++) phi_HH_psi += conj(phi[i]) * HHp[i];
        free(Hp); free(HHp);

        const double complex me_T2_dense = 2.0 / (b * b) * phi_HH_psi - ov;

        mpo_kpm_params_t p2 = mpo_kpm_params_default();
        p2.n_cheby = 3; p2.E_shift = 0; p2.E_scale = b; p2.max_bond_dim = 64;
        p2.use_jackson = 0;
        double complex* m3 = (double complex*)malloc(3 * sizeof(double complex));
        mpo_kpm_chebyshev_moments(H, bra, ket, &p2, m3);
        const double complex me_T2_mps = m3[2];
        free(m3);

        fprintf(stdout, "    <bra|T_2(H_tilde)|ket> MPS  = %+.6f%+.6fi\n",
                creal(me_T2_mps), cimag(me_T2_mps));
        fprintf(stdout, "    <bra|T_2(H_tilde)|ket> dense= %+.6f%+.6fi\n",
                creal(me_T2_dense), cimag(me_T2_dense));
        CHECK(cabs(me_T2_mps - me_T2_dense) < 1e-9,
              "MPS and dense agree on mu_2 = <bra|T_2(H_tilde)|ket>");
    }

    /* Dense answer. */
    const double complex me_dense =
        dense_sign_matrix_element(Hd, N, a, b, phi, psi);

    /* KPM answer. */
    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby       = 1000;
    params.E_shift       = a;
    params.E_scale       = b;
    params.max_bond_dim  = 64;
    params.svd_cutoff    = 1e-14;
    const double complex me_kpm =
        mpo_kpm_sign_matrix_element(H, bra, ket, &params);

    fprintf(stdout,
            "    dense = %+.6f%+.6fi   kpm = %+.6f%+.6fi\n",
            creal(me_dense), cimag(me_dense),
            creal(me_kpm),   cimag(me_kpm));
    const double err = cabs(me_kpm - me_dense);
    fprintf(stdout, "    |kpm - dense| = %.3e\n", err);
    CHECK(err < 1e-4, "|kpm - dense| < 1e-4 at N_c=1000");

    /* Projector consistency: <phi|P|psi> = 1/2(<phi|psi> - me_dense).
     * Check that the API convenience agrees with a direct assembly. */
    const double complex overlap = tn_mps_overlap(bra, ket);
    const double complex proj_direct = 0.5 * (overlap - me_kpm);
    const double complex proj_api =
        mpo_kpm_projector_matrix_element(H, bra, ket, &params);
    CHECK(cabs(proj_direct - proj_api) < 1e-10,
          "projector matrix element API agrees with direct assembly");

    free(phi);
    free(psi);
    free(Hd);
    tn_mps_free(bra);
    tn_mps_free(ket);
    tn_mpo_free(H);
    mpo_free(H_src);
}

int main(void) {
    fprintf(stdout, "=== mpo_kpm unit tests ===\n");
    test_coefficients();
    test_mps_combine_norm(4, 0xC0DE, 0xBEEF);
    test_mps_combine_norm(6, 0x1234, 0x5678);
    test_sign_matrix_element_small_tfim();

    fprintf(stdout, "\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
