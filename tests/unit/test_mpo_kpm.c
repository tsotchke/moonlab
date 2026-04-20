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

/* ---------------------------------------------------------------- */
/* 4. Full MPS sign / projector application                          */
/* ---------------------------------------------------------------- */
static void test_apply_sign_small_tfim(void) {
    fprintf(stdout, "\n-- apply_sign vs dense: L=4 TFIM --\n");
    const uint32_t L = 4;
    mpo_t* H_src = mpo_tfim_create(L, 1.0, 0.8);
    tn_mpo_t* H = mpo_kpm_mpo_to_tn_mpo(H_src);
    CHECK(H_src && H, "MPOs built");

    size_t N = 0;
    double complex* Hd = build_dense_H(H_src, &N);
    CHECK(N == ((size_t)1 << L), "dense dim = 2^L");

    /* Spectrum bounds via eigendecomp. */
    double complex* scratch = (double complex*)malloc(N * N * sizeof(double complex));
    memcpy(scratch, Hd, N * N * sizeof(double complex));
    double* evals = (double*)malloc(N * sizeof(double));
    int info = diagonalise_hermitian(scratch, (int)N, evals);
    CHECK(info == 0, "diag ok");
    const double E_min = evals[0], E_max = evals[N - 1];
    const double a = 0.5 * (E_max + E_min);
    const double b = 0.55 * (E_max - E_min);
    free(scratch);
    free(evals);

    tn_mps_state_t* ket = random_product_mps(L, 0xAB12);
    CHECK(ket != NULL, "ket built");

    /* Dense reference: |sign(H_tilde)|ket> via eigendecomp. */
    double complex* psi = (double complex*)malloc(N * sizeof(double complex));
    CHECK(tn_mps_to_statevector(ket, psi) == TN_STATE_SUCCESS, "ket -> SV");

    /* Build sign(M) dense via eigendecomp of M = (H - a)/b and
     * apply to psi to get sign_ket_dense. */
    double complex* M = (double complex*)malloc(N * N * sizeof(double complex));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) M[i * N + j] = Hd[i * N + j] / b;
        M[i * N + i] -= a / b;
    }
    double* w = (double*)malloc(N * sizeof(double));
    info = diagonalise_hermitian(M, (int)N, w);
    CHECK(info == 0, "eigen ok");

    /* sign_psi = V diag(sign(w)) V^H psi. */
    double complex* Vhpsi = (double complex*)calloc(N, sizeof(double complex));
    for (size_t k = 0; k < N; k++) {
        double complex acc = 0.0;
        for (size_t i = 0; i < N; i++) acc += conj(M[k * N + i]) * psi[i];
        Vhpsi[k] = acc;
    }
    for (size_t k = 0; k < N; k++) {
        const double s = (w[k] > 0) ? 1.0 : (w[k] < 0 ? -1.0 : 0.0);
        Vhpsi[k] *= s;
    }
    double complex* sign_psi_dense = (double complex*)calloc(N, sizeof(double complex));
    for (size_t i = 0; i < N; i++) {
        double complex acc = 0.0;
        for (size_t k = 0; k < N; k++) acc += M[k * N + i] * Vhpsi[k];
        sign_psi_dense[i] = acc;
    }

    /* MPS path. */
    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby = 1000;
    params.E_shift = a;
    params.E_scale = b;
    params.max_bond_dim = 64;
    tn_mps_state_t* sign_ket_mps = mpo_kpm_apply_sign(H, ket, &params);
    CHECK(sign_ket_mps != NULL, "apply_sign returned MPS");

    double complex* sign_psi_mps = (double complex*)malloc(N * sizeof(double complex));
    CHECK(tn_mps_to_statevector(sign_ket_mps, sign_psi_mps) == TN_STATE_SUCCESS,
          "MPS -> SV");

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < N; i++) {
        double complex d = sign_psi_mps[i] - sign_psi_dense[i];
        num += creal(d * conj(d));
        den += creal(sign_psi_dense[i] * conj(sign_psi_dense[i]));
    }
    const double rel = (den > 0) ? sqrt(num / den) : sqrt(num);
    fprintf(stdout, "    ||sign|ket>_mps - sign|ket>_dense|| / ||dense|| = %.3e\n", rel);
    CHECK(rel < 1e-3, "apply_sign agrees with dense to 1e-3 rel");

    /* Projector path. */
    tn_mps_state_t* proj_ket_mps = mpo_kpm_apply_projector(H, ket, &params);
    CHECK(proj_ket_mps != NULL, "apply_projector returned MPS");
    double complex* proj_mps = (double complex*)malloc(N * sizeof(double complex));
    tn_mps_to_statevector(proj_ket_mps, proj_mps);

    double pnum = 0.0, pden = 0.0;
    for (size_t i = 0; i < N; i++) {
        const double complex expected = 0.5 * (psi[i] - sign_psi_dense[i]);
        double complex d = proj_mps[i] - expected;
        pnum += creal(d * conj(d));
        pden += creal(expected * conj(expected));
    }
    const double prel = (pden > 0) ? sqrt(pnum / pden) : sqrt(pnum);
    fprintf(stdout, "    ||P|ket>_mps - P|ket>_dense|| / ||dense|| = %.3e\n", prel);
    CHECK(prel < 1e-3, "apply_projector agrees with dense to 1e-3 rel");

    free(psi); free(M); free(w); free(Vhpsi);
    free(sign_psi_dense); free(sign_psi_mps); free(proj_mps);
    free(Hd);
    tn_mps_free(ket);
    tn_mps_free(sign_ket_mps);
    tn_mps_free(proj_ket_mps);
    tn_mpo_free(H);
    mpo_free(H_src);
}

/* ---------------------------------------------------------------- */
/* 5. Diagonal single-site-sum MPO                                    */
/* ---------------------------------------------------------------- */
static void test_diagonal_sum_mpo(void) {
    fprintf(stdout, "\n-- diagonal-sum MPO Sum_i f_i sigma^z_i --\n");

    const uint32_t L = 4;
    double f[4] = { 1.0, 2.0, 3.0, 4.0 };
    /* Pauli Z, row-major: [Z00, Z01, Z10, Z11] = [1, 0, 0, -1]. */
    double complex Z[4] = { 1.0, 0.0, 0.0, -1.0 };

    tn_mpo_t* D = mpo_kpm_diagonal_sum_mpo(L, Z, f);
    CHECK(D != NULL, "MPO built");

    /* Build a random product-state ket and its reference state vector. */
    tn_mps_state_t* ket = random_product_mps(L, 0xF001);
    const size_t N = (size_t)1 << L;
    double complex* psi = (double complex*)malloc(N * sizeof(double complex));
    CHECK(tn_mps_to_statevector(ket, psi) == TN_STATE_SUCCESS, "ket -> SV");

    /* Dense: D|psi> and <psi|D|psi>. For Z operator, |basis = s_0 s_1 ... s_{L-1}>
     * has D eigenvalue sum_i f_i * (1 - 2*s_i) (since Z|0>=+1, Z|1>=-1). */
    double complex* Dpsi = (double complex*)calloc(N, sizeof(double complex));
    for (size_t k = 0; k < N; k++) {
        double eig = 0.0;
        for (uint32_t i = 0; i < L; i++) {
            int bit = (int)((k >> (L - 1 - i)) & 1U);
            eig += f[i] * (bit == 0 ? 1.0 : -1.0);
        }
        Dpsi[k] = eig * psi[k];
    }
    double complex expt_dense = 0.0;
    for (size_t k = 0; k < N; k++) expt_dense += conj(psi[k]) * Dpsi[k];

    /* MPS: D|psi> via mpo_kpm_apply_mpo, then <psi|D|psi>. */
    tn_mps_state_t* Dket_mps = mpo_kpm_apply_mpo(D, ket, 64);
    CHECK(Dket_mps != NULL, "apply_mpo returned MPS");
    double complex expt_mps = tn_mps_overlap(ket, Dket_mps);
    double complex* Dpsi_mps_sv = (double complex*)malloc(N * sizeof(double complex));
    tn_mps_to_statevector(Dket_mps, Dpsi_mps_sv);

    fprintf(stdout, "    <psi|D|psi> dense = %+.6f%+.6fi\n",
            creal(expt_dense), cimag(expt_dense));
    fprintf(stdout, "    <psi|D|psi> MPS   = %+.6f%+.6fi\n",
            creal(expt_mps), cimag(expt_mps));
    CHECK(cabs(expt_dense - expt_mps) < 1e-10,
          "<psi|D|psi> MPS matches dense");

    /* Element-wise vector agreement between D|psi>_dense and D|psi>_mps. */
    double num = 0.0, den = 0.0;
    for (size_t k = 0; k < N; k++) {
        double complex d = Dpsi[k] - Dpsi_mps_sv[k];
        num += creal(d * conj(d));
        den += creal(Dpsi[k] * conj(Dpsi[k]));
    }
    const double rel = (den > 0) ? sqrt(num / den) : sqrt(num);
    fprintf(stdout, "    ||D|psi>_mps - D|psi>_dense|| / ||dense|| = %.3e\n", rel);
    CHECK(rel < 1e-12, "D|psi> MPS matches dense vector");

    free(psi); free(Dpsi); free(Dpsi_mps_sv);
    tn_mps_free(ket); tn_mps_free(Dket_mps);
    tn_mpo_free(D);
}

/* ---------------------------------------------------------------- */
/* 6. End-to-end Q X P pipeline validation                           */
/* ---------------------------------------------------------------- */
/*
 * Compose the full Bianco-Resta-shape matrix element on a small TFIM:
 *   <alpha| Q X P |alpha>
 * where
 *   P = (I - sign((H - E_f I)/b)) / 2   (filled-band projector)
 *   Q = I - P
 *   X = sum_i i * sigma^z_i             ("position" diagonal sum MPO)
 * and |alpha> is a random product state.
 *
 * MPS path: apply_projector(H, alpha) -> P|alpha>
 *           apply_mpo(X, P|alpha>)    -> X P|alpha>
 *           form Q|ket> = |ket> - P|ket> via mps_combine, apply to
 *           X P|alpha>
 *           hmm more cleanly: <alpha| Q X P |alpha>
 *                           = <alpha|X P|alpha> - <alpha|P X P|alpha>
 *           so we only need two MPS products + overlaps.
 *
 * Dense path: diagonalise H_tilde, build sign, P, then work in the
 *             2^L-dim state vector space.  Compare.
 */
static void test_qxp_pipeline(void) {
    fprintf(stdout, "\n-- end-to-end <alpha|Q X P|alpha> --\n");
    const uint32_t L = 4;
    mpo_t* H_src = mpo_tfim_create(L, 1.0, 0.8);
    tn_mpo_t* H = mpo_kpm_mpo_to_tn_mpo(H_src);
    CHECK(H && H_src, "TFIM MPO + adapter built");

    size_t N = 0;
    double complex* Hd = build_dense_H(H_src, &N);

    /* Spectrum bounds. */
    double complex* scratch = (double complex*)malloc(N * N * sizeof(double complex));
    memcpy(scratch, Hd, N * N * sizeof(double complex));
    double* evals = (double*)malloc(N * sizeof(double));
    diagonalise_hermitian(scratch, (int)N, evals);
    const double E_min = evals[0], E_max = evals[N - 1];
    free(scratch); free(evals);
    const double a = 0.5 * (E_max + E_min);
    const double b = 0.55 * (E_max - E_min);

    /* Build X = sum_i i * sigma^z_i as an MPO. */
    double f[16] = {0};
    for (uint32_t i = 0; i < L; i++) f[i] = (double)i;
    double complex Zop[4] = { 1.0, 0.0, 0.0, -1.0 };
    tn_mpo_t* X = mpo_kpm_diagonal_sum_mpo(L, Zop, f);
    CHECK(X != NULL, "X-hat MPO built");

    /* Random bra = ket (same state; matrix element becomes an
     * expectation value, which is easier to interpret). */
    tn_mps_state_t* alpha = random_product_mps(L, 0x91A0);
    CHECK(alpha != NULL, "alpha built");

    /* Dense: psi = alpha as statevector; sign(H_tilde), P = (I-sign)/2,
     * Q = I - P; X is diagonal in basis with eigenvalue sum_i i*(1-2*s_i). */
    double complex* psi = (double complex*)malloc(N * sizeof(double complex));
    tn_mps_to_statevector(alpha, psi);

    /* Build M = H_tilde dense, diagonalise into M (eigvecs as cols). */
    double complex* M = (double complex*)malloc(N * N * sizeof(double complex));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) M[i * N + j] = Hd[i * N + j] / b;
        M[i * N + i] -= a / b;
    }
    double* w = (double*)malloc(N * sizeof(double));
    diagonalise_hermitian(M, (int)N, w);

    /* P|psi> dense: project onto eigenvectors with w < 0 (sign=-1
     * means eigenvalue is in the filled band under the (I-sign)/2
     * convention). */
    double complex* Ppsi = (double complex*)calloc(N, sizeof(double complex));
    {
        double complex* Vhpsi = (double complex*)calloc(N, sizeof(double complex));
        for (size_t k = 0; k < N; k++) {
            double complex acc = 0.0;
            for (size_t i = 0; i < N; i++) acc += conj(M[k * N + i]) * psi[i];
            Vhpsi[k] = acc;
        }
        /* Zero out positive-eigenvalue components; keep negative. */
        for (size_t k = 0; k < N; k++) {
            if (w[k] >= 0.0) Vhpsi[k] = 0.0;
        }
        for (size_t i = 0; i < N; i++) {
            double complex acc = 0.0;
            for (size_t k = 0; k < N; k++) acc += M[k * N + i] * Vhpsi[k];
            Ppsi[i] = acc;
        }
        free(Vhpsi);
    }

    /* X|P psi> = diagonal · P psi; each entry scaled by its basis eig. */
    double complex* XPpsi = (double complex*)malloc(N * sizeof(double complex));
    for (size_t k = 0; k < N; k++) {
        double eig = 0.0;
        for (uint32_t i = 0; i < L; i++) {
            int bit = (int)((k >> (L - 1 - i)) & 1U);
            eig += f[i] * (bit == 0 ? 1.0 : -1.0);
        }
        XPpsi[k] = eig * Ppsi[k];
    }

    /* <alpha|X P|alpha> and <alpha|P X P|alpha>. */
    double complex alpha_XP_alpha_dense = 0.0;
    for (size_t k = 0; k < N; k++) alpha_XP_alpha_dense += conj(psi[k]) * XPpsi[k];
    /* P |X P psi>: repeat the project. */
    double complex* PXPpsi = (double complex*)calloc(N, sizeof(double complex));
    {
        double complex* Vhx = (double complex*)calloc(N, sizeof(double complex));
        for (size_t k = 0; k < N; k++) {
            double complex acc = 0.0;
            for (size_t i = 0; i < N; i++) acc += conj(M[k * N + i]) * XPpsi[i];
            Vhx[k] = acc;
        }
        for (size_t k = 0; k < N; k++) {
            if (w[k] >= 0.0) Vhx[k] = 0.0;
        }
        for (size_t i = 0; i < N; i++) {
            double complex acc = 0.0;
            for (size_t k = 0; k < N; k++) acc += M[k * N + i] * Vhx[k];
            PXPpsi[i] = acc;
        }
        free(Vhx);
    }
    double complex alpha_PXP_alpha_dense = 0.0;
    for (size_t k = 0; k < N; k++) alpha_PXP_alpha_dense += conj(psi[k]) * PXPpsi[k];

    const double complex qxp_dense = alpha_XP_alpha_dense - alpha_PXP_alpha_dense;
    fprintf(stdout, "    dense <alpha|X P|alpha>   = %+.6f%+.6fi\n",
            creal(alpha_XP_alpha_dense), cimag(alpha_XP_alpha_dense));
    fprintf(stdout, "    dense <alpha|P X P|alpha> = %+.6f%+.6fi\n",
            creal(alpha_PXP_alpha_dense), cimag(alpha_PXP_alpha_dense));
    fprintf(stdout, "    dense <alpha|Q X P|alpha> = %+.6f%+.6fi\n",
            creal(qxp_dense), cimag(qxp_dense));

    /* MPS path: P|alpha> via apply_projector, X P|alpha> via apply_mpo,
     * then the two overlaps. */
    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby = 1000;
    params.E_shift = a;
    params.E_scale = b;
    params.max_bond_dim = 64;
    params.svd_cutoff = 1e-14;

    tn_mps_state_t* P_alpha   = mpo_kpm_apply_projector(H, alpha, &params);
    CHECK(P_alpha != NULL, "P|alpha> built");
    tn_mps_state_t* XP_alpha  = mpo_kpm_apply_mpo(X, P_alpha, 64);
    CHECK(XP_alpha != NULL, "X P|alpha> built");
    tn_mps_state_t* PXP_alpha = mpo_kpm_apply_projector(H, XP_alpha, &params);
    CHECK(PXP_alpha != NULL, "P X P|alpha> built");

    const double complex alpha_XP_alpha_mps  = tn_mps_overlap(alpha, XP_alpha);
    const double complex alpha_PXP_alpha_mps = tn_mps_overlap(alpha, PXP_alpha);
    const double complex qxp_mps = alpha_XP_alpha_mps - alpha_PXP_alpha_mps;

    fprintf(stdout, "    MPS   <alpha|X P|alpha>   = %+.6f%+.6fi\n",
            creal(alpha_XP_alpha_mps), cimag(alpha_XP_alpha_mps));
    fprintf(stdout, "    MPS   <alpha|P X P|alpha> = %+.6f%+.6fi\n",
            creal(alpha_PXP_alpha_mps), cimag(alpha_PXP_alpha_mps));
    fprintf(stdout, "    MPS   <alpha|Q X P|alpha> = %+.6f%+.6fi\n",
            creal(qxp_mps), cimag(qxp_mps));
    fprintf(stdout, "    |mps - dense| = %.3e\n", cabs(qxp_mps - qxp_dense));

    CHECK(cabs(alpha_XP_alpha_mps - alpha_XP_alpha_dense) < 1e-3,
          "<alpha|X P|alpha> mps matches dense");
    CHECK(cabs(alpha_PXP_alpha_mps - alpha_PXP_alpha_dense) < 1e-3,
          "<alpha|P X P|alpha> mps matches dense");
    CHECK(cabs(qxp_mps - qxp_dense) < 1e-3,
          "composed <alpha|Q X P|alpha> mps matches dense");

    free(psi); free(M); free(w); free(Ppsi); free(XPpsi); free(PXPpsi);
    free(Hd);
    tn_mps_free(alpha);
    tn_mps_free(P_alpha); tn_mps_free(XP_alpha); tn_mps_free(PXP_alpha);
    tn_mpo_free(X); tn_mpo_free(H); mpo_free(H_src);
}

/* ---------------------------------------------------------------- */
/* 7. MPO-level sign(H) validation                                   */
/* ---------------------------------------------------------------- */

/* Contract a tn_mpo_t into its dense 2^L x 2^L matrix representation
 * (row-major, row = input basis state, col = output basis state,
 * same convention as mpo_to_matrix).  O(2^(2L) * chi^L); only usable
 * for small L.  Caller owns the returned buffer. */
static double complex* tn_mpo_to_dense(const tn_mpo_t* mpo, size_t* out_N) {
    if (!mpo || mpo->num_sites == 0) return NULL;
    const uint32_t L = mpo->num_sites;
    const uint32_t d = mpo->tensors[0]->dims[1];
    const size_t N = 1;  /* placeholder */
    (void)N;
    size_t dim = 1;
    for (uint32_t i = 0; i < L; i++) dim *= d;
    double complex* out = (double complex*)calloc(dim * dim,
                                                   sizeof(double complex));
    if (!out) return NULL;
    /* For each (input s1..sL, output s1'..sL'), walk the product. */
    uint32_t* s_in  = (uint32_t*)calloc(L, sizeof(uint32_t));
    uint32_t* s_out = (uint32_t*)calloc(L, sizeof(uint32_t));
    for (size_t ii = 0; ii < dim; ii++) {
        /* Decode ii into s_in[]. */
        size_t tmp = ii;
        for (int k = (int)L - 1; k >= 0; k--) {
            s_in[k] = (uint32_t)(tmp % d);
            tmp /= d;
        }
        for (size_t oo = 0; oo < dim; oo++) {
            tmp = oo;
            for (int k = (int)L - 1; k >= 0; k--) {
                s_out[k] = (uint32_t)(tmp % d);
                tmp /= d;
            }
            /* Contract over bond chain b_0..b_L with b_0 = b_L = 0. */
            const uint32_t b_L_final = mpo->tensors[L - 1]->dims[3];
            /* row-vector running through: v[b] after site i. */
            uint32_t bond_max = 1;
            for (uint32_t i = 0; i < L; i++) {
                if (mpo->tensors[i]->dims[0] > bond_max)
                    bond_max = mpo->tensors[i]->dims[0];
                if (mpo->tensors[i]->dims[3] > bond_max)
                    bond_max = mpo->tensors[i]->dims[3];
            }
            double complex* v_curr = (double complex*)calloc(bond_max,
                                                             sizeof(double complex));
            double complex* v_next = (double complex*)calloc(bond_max,
                                                             sizeof(double complex));
            v_curr[0] = 1.0;
            uint32_t v_len = 1;
            for (uint32_t i = 0; i < L; i++) {
                const tensor_t* W = mpo->tensors[i];
                const uint32_t bL = W->dims[0], di = W->dims[1];
                const uint32_t do_ = W->dims[2], bR = W->dims[3];
                memset(v_next, 0, bond_max * sizeof(double complex));
                for (uint32_t bl = 0; bl < bL; bl++) {
                    for (uint32_t br = 0; br < bR; br++) {
                        const size_t widx = ((size_t)bl * di + s_in[i]) * do_ * bR
                                            + (size_t)s_out[i] * bR + br;
                        v_next[br] += v_curr[bl] * W->data[widx];
                    }
                }
                memcpy(v_curr, v_next, bond_max * sizeof(double complex));
                v_len = bR;
            }
            (void)v_len;
            (void)b_L_final;
            out[ii * dim + oo] = v_curr[0];
            free(v_curr); free(v_next);
        }
    }
    free(s_in); free(s_out);
    if (out_N) *out_N = dim;
    return out;
}

static void test_mpo_level_sign(void) {
    fprintf(stdout, "\n-- MPO-level sign(H) vs dense --\n");
    const uint32_t L = 3;
    mpo_t* H_src = mpo_tfim_create(L, 1.0, 0.8);
    tn_mpo_t* H = mpo_kpm_mpo_to_tn_mpo(H_src);

    /* Dense H and its sign via eigendecomp. */
    size_t Ndense = 0;
    double complex* Hd = build_dense_H(H_src, &Ndense);
    double complex* scratch = (double complex*)malloc(Ndense * Ndense
                                                       * sizeof(double complex));
    memcpy(scratch, Hd, Ndense * Ndense * sizeof(double complex));
    double* evals = (double*)malloc(Ndense * sizeof(double));
    diagonalise_hermitian(scratch, (int)Ndense, evals);
    const double E_min = evals[0], E_max = evals[Ndense - 1];
    free(scratch); free(evals);
    const double a = 0.5 * (E_max + E_min);
    const double b = 0.55 * (E_max - E_min);

    /* Rebuild M = H_tilde dense + diagonalise + form sign. */
    double complex* M = (double complex*)malloc(Ndense * Ndense
                                                 * sizeof(double complex));
    for (size_t i = 0; i < Ndense; i++) {
        for (size_t j = 0; j < Ndense; j++)
            M[i * Ndense + j] = Hd[i * Ndense + j] / b;
        M[i * Ndense + i] -= a / b;
    }
    double* w = (double*)malloc(Ndense * sizeof(double));
    diagonalise_hermitian(M, (int)Ndense, w);
    double complex* sign_dense = (double complex*)calloc(Ndense * Ndense,
                                                          sizeof(double complex));
    for (size_t i = 0; i < Ndense; i++) {
        for (size_t j = 0; j < Ndense; j++) {
            double complex acc = 0.0;
            for (size_t k = 0; k < Ndense; k++) {
                const double s = (w[k] > 0) ? 1.0 : (w[k] < 0 ? -1.0 : 0.0);
                acc += M[k * Ndense + i] * s * conj(M[k * Ndense + j]);
            }
            sign_dense[i * Ndense + j] = acc;
        }
    }
    free(M); free(w);

    /* MPO-level sign via Chebyshev. */
    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby = 300;
    params.E_shift = a;
    params.E_scale = b;
    params.max_bond_dim = 32;
    params.use_jackson = 1;
    tn_mpo_t* sign_mpo = mpo_kpm_sign_mpo(H, &params);
    CHECK(sign_mpo != NULL, "sign MPO built");

    size_t N2 = 0;
    double complex* sign_mpo_dense = tn_mpo_to_dense(sign_mpo, &N2);
    CHECK(sign_mpo_dense != NULL, "sign MPO -> dense matrix");
    CHECK(N2 == Ndense, "dims match (%zu)", N2);

    /* Frobenius-norm relative error. */
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < Ndense * Ndense; i++) {
        double complex d = sign_mpo_dense[i] - sign_dense[i];
        num += creal(d * conj(d));
        den += creal(sign_dense[i] * conj(sign_dense[i]));
    }
    const double rel = (den > 0) ? sqrt(num / den) : sqrt(num);
    fprintf(stdout, "    ||sign(H)_mpo - sign(H)_dense||_F / ||dense||_F = %.3e\n",
            rel);
    CHECK(rel < 1e-2, "MPO-level sign matches dense to 1%% at L=3, N_c=300");

    free(Hd); free(sign_dense); free(sign_mpo_dense);
    tn_mpo_free(sign_mpo);
    tn_mpo_free(H); mpo_free(H_src);
}

/* ---------------------------------------------------------------- */
/* 8. Cross-check: MPO projector applied to state vs MPS path         */
/* ---------------------------------------------------------------- */
static void test_projector_mpo_vs_mps_path(void) {
    fprintf(stdout, "\n-- projector_mpo applied to ket vs apply_projector --\n");
    const uint32_t L = 3;
    mpo_t* H_src = mpo_tfim_create(L, 1.0, 0.8);
    tn_mpo_t* H = mpo_kpm_mpo_to_tn_mpo(H_src);

    /* Spectrum bounds. */
    size_t Ndense = 0;
    double complex* Hd = build_dense_H(H_src, &Ndense);
    double complex* scratch = (double complex*)malloc(Ndense * Ndense
                                                       * sizeof(double complex));
    memcpy(scratch, Hd, Ndense * Ndense * sizeof(double complex));
    double* evals = (double*)malloc(Ndense * sizeof(double));
    diagonalise_hermitian(scratch, (int)Ndense, evals);
    const double a = 0.5 * (evals[0] + evals[Ndense - 1]);
    const double b = 0.55 * (evals[Ndense - 1] - evals[0]);
    free(Hd); free(scratch); free(evals);

    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby = 300;
    params.E_shift = a;
    params.E_scale = b;
    params.max_bond_dim = 32;

    tn_mps_state_t* ket = random_product_mps(L, 0xDEAD);

    /* Path A: apply_projector on the state. */
    tn_mps_state_t* P_ket_A = mpo_kpm_apply_projector(H, ket, &params);
    CHECK(P_ket_A != NULL, "apply_projector ok");

    /* Path B: build P as MPO, apply to the state. */
    tn_mpo_t* P_mpo = mpo_kpm_projector_mpo(H, &params);
    CHECK(P_mpo != NULL, "projector_mpo ok");
    tn_mps_state_t* P_ket_B = mpo_kpm_apply_mpo(P_mpo, ket, 64);
    CHECK(P_ket_B != NULL, "apply_mpo(P) ok");

    /* Compare via <ket|P|ket> from each. */
    const double complex me_A = tn_mps_overlap(ket, P_ket_A);
    const double complex me_B = tn_mps_overlap(ket, P_ket_B);
    fprintf(stdout, "    <ket|P|ket> MPS-route  = %+.6f%+.6fi\n",
            creal(me_A), cimag(me_A));
    fprintf(stdout, "    <ket|P|ket> MPO-route  = %+.6f%+.6fi\n",
            creal(me_B), cimag(me_B));
    CHECK(cabs(me_A - me_B) < 1e-3, "two routes agree on <ket|P|ket>");

    tn_mps_free(ket);
    tn_mps_free(P_ket_A); tn_mps_free(P_ket_B);
    tn_mpo_free(P_mpo);
    tn_mpo_free(H); mpo_free(H_src);
}

/* ---------------------------------------------------------------- */
/* 9. Dense -> MPO roundtrip                                          */
/* ---------------------------------------------------------------- */
static void test_mpo_from_dense_roundtrip(void) {
    fprintf(stdout, "\n-- mpo_from_dense roundtrip --\n");
    /* Build a random Hermitian 8x8 matrix (L = 3 sites). */
    const uint32_t L = 3;
    const size_t N = (size_t)1 << L;
    double complex* H_dense = (double complex*)malloc(N * N * sizeof(double complex));
    srand(0xFEEDC0DE);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double re = (double)rand() / RAND_MAX - 0.5;
            double im = (i == j) ? 0.0 : ((double)rand() / RAND_MAX - 0.5);
            H_dense[i * N + j] = re + im * I;
            H_dense[j * N + i] = re - im * I;
        }
    }

    tn_mpo_t* mpo = mpo_kpm_mpo_from_dense(H_dense, L, 0.0);
    CHECK(mpo != NULL, "mpo_from_dense returned non-null");
    CHECK(mpo->num_sites == L, "num_sites = L = %u", L);

    /* Dense-roundtrip via the tn_mpo_to_dense helper in this file. */
    size_t Nr = 0;
    double complex* H_roundtrip = tn_mpo_to_dense(mpo, &Nr);
    CHECK(H_roundtrip != NULL, "tn_mpo_to_dense roundtrip");
    CHECK(Nr == N, "dim matches");

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < N * N; i++) {
        double complex d = H_dense[i] - H_roundtrip[i];
        num += creal(d * conj(d));
        den += creal(H_dense[i] * conj(H_dense[i]));
    }
    const double rel = (den > 0) ? sqrt(num / den) : sqrt(num);
    fprintf(stdout, "    ||M - MPO(M)||_F / ||M||_F = %.3e\n", rel);
    CHECK(rel < 1e-12, "dense roundtrip is ULP-exact");

    free(H_dense); free(H_roundtrip);
    tn_mpo_free(mpo);
}

/* ---------------------------------------------------------------- */
/* 10. Chern-like closing: MPO-level P on a non-trivial 8x8 H        */
/* ---------------------------------------------------------------- */
/*
 * Builds a small random Hermitian 8x8 H with a clear zero-energy gap
 * (constructed via conjugation of a diagonal matrix with a random
 * unitary), converts to MPO via mpo_from_dense, computes the filled-
 * band projector via MPO-level Chebyshev, and confirms that
 *   MPO_to_dense(P_mpo) * psi ~ P_dense * psi
 * for a random psi.  This is the final self-consistency check: the
 * whole MPO-level pipeline (from-dense -> sign -> projector) lines
 * up with the dense Bianco-Resta projector on an arbitrary Hermitian
 * operator.
 */
static void test_projector_from_generic_dense(void) {
    fprintf(stdout, "\n-- projector_mpo on generic dense H --\n");
    const uint32_t L = 3;
    const size_t N = (size_t)1 << L;

    /* Start from diag(lambda) with lambda on either side of 0. */
    double lam[8] = {-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0};
    double complex* H = (double complex*)calloc(N * N, sizeof(double complex));
    /* Build a random unitary U via QR of a random complex matrix. */
    double complex* R = (double complex*)malloc(N * N * sizeof(double complex));
    srand(0xFEEDBACEu);
    for (size_t i = 0; i < N * N; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        R[i] = re + im * I;
    }
    /* Gram-Schmidt on columns. */
    double complex* U = (double complex*)calloc(N * N, sizeof(double complex));
    for (size_t j = 0; j < N; j++) {
        /* col j of U = R[:, j] minus projections onto U[:, 0..j-1]. */
        for (size_t i = 0; i < N; i++) U[i * N + j] = R[i * N + j];
        for (size_t k = 0; k < j; k++) {
            double complex proj = 0.0;
            for (size_t i = 0; i < N; i++) proj += conj(U[i * N + k]) * U[i * N + j];
            for (size_t i = 0; i < N; i++) U[i * N + j] -= proj * U[i * N + k];
        }
        double norm2 = 0.0;
        for (size_t i = 0; i < N; i++) norm2 += creal(U[i * N + j] * conj(U[i * N + j]));
        double invn = 1.0 / sqrt(norm2);
        for (size_t i = 0; i < N; i++) U[i * N + j] *= invn;
    }
    free(R);

    /* H = U diag(lam) U^H. */
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double complex acc = 0.0;
            for (size_t k = 0; k < N; k++) acc += U[i * N + k] * lam[k] * conj(U[j * N + k]);
            H[i * N + j] = acc;
        }
    }

    /* Dense P = U diag([1 for lam<0, 0 for lam>=0]) U^H. */
    double complex* Pd = (double complex*)calloc(N * N, sizeof(double complex));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double complex acc = 0.0;
            for (size_t k = 0; k < N; k++) {
                double occ = (lam[k] < 0) ? 1.0 : 0.0;
                acc += U[i * N + k] * occ * conj(U[j * N + k]);
            }
            Pd[i * N + j] = acc;
        }
    }
    free(U);

    /* Dense -> MPO for H. */
    tn_mpo_t* H_mpo = mpo_kpm_mpo_from_dense(H, L, 0.0);
    CHECK(H_mpo != NULL, "H -> MPO");

    /* Projector via MPO Chebyshev. */
    mpo_kpm_params_t params = mpo_kpm_params_default();
    params.n_cheby = 400;
    /* lam in [-3, 3]; shift 0, scale 3.5 to put spectrum in (-1, 1). */
    params.E_shift = 0.0;
    params.E_scale = 3.5;
    params.max_bond_dim = 32;

    tn_mpo_t* P_mpo = mpo_kpm_projector_mpo(H_mpo, &params);
    CHECK(P_mpo != NULL, "projector MPO built");

    size_t Nr = 0;
    double complex* P_dense_from_mpo = tn_mpo_to_dense(P_mpo, &Nr);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < N * N; i++) {
        double complex d = P_dense_from_mpo[i] - Pd[i];
        num += creal(d * conj(d));
        den += creal(Pd[i] * conj(Pd[i]));
    }
    const double rel = (den > 0) ? sqrt(num / den) : sqrt(num);
    fprintf(stdout, "    ||P_mpo - P_dense||_F / ||P_dense||_F = %.3e\n", rel);
    CHECK(rel < 1e-2, "MPO-projector agrees with dense projector");

    /* Trace check: tr(P) should equal number of negative eigenvalues. */
    double complex tr = 0.0;
    for (size_t i = 0; i < N; i++) tr += P_dense_from_mpo[i * N + i];
    fprintf(stdout, "    tr(P_mpo) = %+.4f%+.4fi (expect 4)\n",
            creal(tr), cimag(tr));
    CHECK(fabs(creal(tr) - 4.0) < 1e-2, "tr(P) ~ number of filled states");

    free(H); free(Pd); free(P_dense_from_mpo);
    tn_mpo_free(H_mpo); tn_mpo_free(P_mpo);
}

int main(void) {
    fprintf(stdout, "=== mpo_kpm unit tests ===\n");
    test_coefficients();
    test_mps_combine_norm(4, 0xC0DE, 0xBEEF);
    test_mps_combine_norm(6, 0x1234, 0x5678);
    test_sign_matrix_element_small_tfim();
    test_apply_sign_small_tfim();
    test_diagonal_sum_mpo();
    test_qxp_pipeline();
    test_mpo_level_sign();
    test_projector_mpo_vs_mps_path();
    test_mpo_from_dense_roundtrip();
    test_projector_from_generic_dense();

    fprintf(stdout, "\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
