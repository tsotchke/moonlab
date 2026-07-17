/*
 * t_svd.c -- tensor_svd edge cases.
 *
 * Built twice by CMake:
 *   t_svd_lapack   = harness + prebuilt libquantumsim   (Accelerate zgesvd)
 *   t_svd_fallback = harness + tensor.c compiled with
 *                    -DQSIM_FORCE_SVD_FALLBACK -DQSIM_FORCE_QR_FALLBACK
 *                    + gpu_stubs.c  (one-sided Jacobi SVD / Householder QR)
 *
 * For each test matrix A (m x n) we take the FULL SVD (max_rank=0, cutoff=0)
 * and verify implementation-agnostic invariants:
 *   1. no NaN/Inf in S, U, Vh
 *   2. singular values descending and >= 0
 *   3. reconstruction  A ~= U diag(S) Vh   (Frobenius)
 *   4. U columns orthonormal  (U^H U = I_k)     <- the property the fallback
 *   5. Vh rows orthonormal    (Vh Vh^H = I_k)       comment warns can break
 *
 * The wide (m<n) case specifically exercises the fallback's daggered-SVD
 * dispatch; rank-deficient / near-degenerate / tiny-huge singular values
 * probe ill-conditioning.
 */
#include "numerical_common.h"
#include "../../src/algorithms/tensor_network/tensor.h"

static tensor_t *mk(uint32_t m, uint32_t n, const complex_t *rowmajor) {
    tensor_t *t = tensor_create_matrix(m, n);
    if (!t) return NULL;
    for (uint32_t i = 0; i < m*n; i++) t->data[i] = rowmajor[i];
    return t;
}

static void check_svd(const char *name, const complex_t *A, uint32_t m, uint32_t n) {
    tensor_t *T = mk(m, n, A);
    if (!T) { NC_MISS("%s: tensor alloc", name); return; }

    tensor_svd_result_t *r = tensor_svd(T, 0, 0.0); /* full SVD */
    if (!r) { NC_MISS("%s: tensor_svd returned NULL", name); tensor_free(T); return; }

    uint32_t k = r->k;
    int bad = 0;
    for (uint32_t i = 0; i < k; i++) if (nc_is_bad(r->S[i])) bad = 1;
    for (uint32_t i = 0; i < m*k; i++) if (nc_is_bad_c(r->U->data[i])) bad = 1;
    for (uint32_t i = 0; i < k*n; i++) if (nc_is_bad_c(r->Vh->data[i])) bad = 1;
    g_checks++;
    if (bad) { NC_MISS("%s: NaN/Inf in SVD factors", name);
               tensor_svd_free(r); tensor_free(T); return; }

    /* descending + nonneg */
    g_checks++;
    for (uint32_t i = 0; i < k; i++) {
        if (r->S[i] < 0.0) { NC_MISS("%s: negative sv S[%u]=%.3e", name, i, r->S[i]); break; }
        if (i && r->S[i] > r->S[i-1] + 1e-10) {
            NC_MISS("%s: sv not descending S[%u]=%.6g > S[%u]=%.6g",
                    name, i, r->S[i], i-1, r->S[i-1]); break;
        }
    }

    /* scale for tolerances */
    double scale = 1.0;
    for (uint32_t i = 0; i < m*n; i++) { double a = cabs(A[i]); if (a > scale) scale = a; }
    double tol = 1e-9 * scale * (double)(m > n ? m : n);

    /* reconstruction A ~= U diag(S) Vh */
    double rerr = 0.0;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            complex_t acc = 0.0;
            for (uint32_t p = 0; p < k; p++)
                acc += r->U->data[i*k + p] * r->S[p] * r->Vh->data[p*n + j];
            double e = cabs(acc - A[i*n + j]);
            if (e > rerr) rerr = e;
        }
    }
    if (rerr > tol) NC_MISS("%s: reconstruction err=%.3e tol=%.3e (m=%u n=%u k=%u)",
                            name, rerr, tol, m, n, k);
    else g_checks++;

    /* U^H U = I_k */
    double ue = 0.0;
    for (uint32_t a = 0; a < k; a++)
        for (uint32_t b = 0; b < k; b++) {
            complex_t acc = 0.0;
            for (uint32_t i = 0; i < m; i++)
                acc += conj(r->U->data[i*k + a]) * r->U->data[i*k + b];
            double e = cabs(acc - (a==b ? 1.0 : 0.0));
            if (e > ue) ue = e;
        }
    if (ue > 1e-9 * (double)k) NC_MISS("%s: U not orthonormal, err=%.3e (k=%u)", name, ue, k);
    else g_checks++;

    /* Vh Vh^H = I_k */
    double ve = 0.0;
    for (uint32_t a = 0; a < k; a++)
        for (uint32_t b = 0; b < k; b++) {
            complex_t acc = 0.0;
            for (uint32_t j = 0; j < n; j++)
                acc += r->Vh->data[a*n + j] * conj(r->Vh->data[b*n + j]);
            double e = cabs(acc - (a==b ? 1.0 : 0.0));
            if (e > ve) ve = e;
        }
    if (ve > 1e-9 * (double)k) NC_MISS("%s: Vh not orthonormal, err=%.3e (k=%u)", name, ve, k);
    else g_checks++;

    tensor_svd_free(r);
    tensor_free(T);
}

int main(void) {
#ifdef SVD_FALLBACK_BUILD
    nc_begin("svd_fallback");
#else
    nc_begin("svd_lapack");
#endif
    uint64_t seed = 0x5EED1234ULL;
    NC_INFO("seed=0x%llx", (unsigned long long)seed);

    /* random square 6x6 */
    {
        uint32_t m=6, n=6; complex_t *A = calloc(m*n, sizeof(complex_t));
        for (uint32_t i=0;i<m*n;i++) A[i] = nc_urand(&seed) + nc_urand(&seed)*I;
        check_svd("rand_square_6", A, m, n); free(A);
    }
    /* wide 3x7 -- exercises fallback daggered dispatch */
    {
        uint32_t m=3, n=7; complex_t *A = calloc(m*n, sizeof(complex_t));
        for (uint32_t i=0;i<m*n;i++) A[i] = nc_urand(&seed) + nc_urand(&seed)*I;
        check_svd("rand_wide_3x7", A, m, n); free(A);
    }
    /* tall 7x3 */
    {
        uint32_t m=7, n=3; complex_t *A = calloc(m*n, sizeof(complex_t));
        for (uint32_t i=0;i<m*n;i++) A[i] = nc_urand(&seed) + nc_urand(&seed)*I;
        check_svd("rand_tall_7x3", A, m, n); free(A);
    }
    /* rank-1 3x3 A = u v^T (u=(1,2,3), v=(1,1,1)): minimal trigger for the
     * one-sided-Jacobi fallback's non-orthonormal-U bug on zero singular
     * values.  LAPACK: |U^H U - I| ~ 7e-16; fallback: ~0.80. */
    {
        uint32_t m=3,n=3; complex_t A[9];
        double u[3]={1,2,3}, v[3]={1,1,1};
        for (uint32_t i=0;i<m;i++) for (uint32_t j=0;j<n;j++) A[i*n+j]=u[i]*v[j];
        check_svd("rank1_3x3", A, m, n);
    }
    /* rank-2 6x6 (sum of two outer products) */
    {
        uint32_t m=6, n=6; complex_t *A = calloc(m*n, sizeof(complex_t));
        complex_t u1[6],v1[6],u2[6],v2[6];
        for (int i=0;i<6;i++){u1[i]=nc_urand(&seed)+nc_urand(&seed)*I;
                              v1[i]=nc_urand(&seed)+nc_urand(&seed)*I;
                              u2[i]=nc_urand(&seed)+nc_urand(&seed)*I;
                              v2[i]=nc_urand(&seed)+nc_urand(&seed)*I;}
        for (uint32_t i=0;i<m;i++) for (uint32_t j=0;j<n;j++)
            A[i*n+j] = u1[i]*conj(v1[j]) + u2[i]*conj(v2[j]);
        check_svd("rank2_6x6", A, m, n); free(A);
    }
    /* near-degenerate singular values: diag(1, 1+1e-11, 0.5) rotated */
    {
        uint32_t m=3,n=3; complex_t D[9]={0};
        D[0]=1.0; D[4]=1.0+1e-11; D[8]=0.5;
        check_svd("near_degenerate_sv", D, m, n);
    }
    /* tiny/huge singular values diag(1e14, 1, 1e-14) */
    {
        uint32_t m=3,n=3; complex_t D[9]={0};
        D[0]=1e14; D[4]=1.0; D[8]=1e-14;
        check_svd("tiny_huge_sv", D, m, n);
    }
    /* identity 5x5 -- fully degenerate singular spectrum */
    {
        uint32_t m=5,n=5; complex_t *A=calloc(m*n,sizeof(complex_t));
        for (uint32_t i=0;i<m;i++) A[i*n+i]=1.0;
        check_svd("identity_5", A, m, n); free(A);
    }
    /* near-zero-norm matrix ~1e-160 */
    {
        uint32_t m=4,n=4; complex_t *A=calloc(m*n,sizeof(complex_t));
        for (uint32_t i=0;i<m*n;i++) A[i]=1e-160*(nc_urand(&seed)+nc_urand(&seed)*I);
        check_svd("near_zero_norm", A, m, n); free(A);
    }
    /* exact-zero matrix (degenerate) */
    {
        uint32_t m=4,n=4; complex_t *A=calloc(m*n,sizeof(complex_t));
        check_svd("all_zero", A, m, n); free(A);
    }

    return nc_end();
}
