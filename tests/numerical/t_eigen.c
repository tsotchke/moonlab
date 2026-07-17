/*
 * t_eigen.c -- Hermitian eigendecomposition edge cases.
 *
 * hermitian_eigen_decomposition(matrix, n, eigenvalues, eigenvectors,
 *                               max_iter, tol)
 *   - eigenvalues:  n reals, sorted DESCENDING (documented convention)
 *   - eigenvectors: n x n, column j is the j-th eigenvector, stored so
 *                   that v_j[i] == eigenvectors[i*n + j].
 *
 * This one source is built twice by CMake:
 *   t_eigen_lapack  = harness + prebuilt libquantumsim  (Accelerate zheev)
 *   t_eigen_jacobi  = harness + matrix_math.c compiled with -U__APPLE__
 *                     (forces the no-LAPACK complex-Hermitian Jacobi path)
 *
 * Checks are implementation-agnostic (reconstruction / orthonormality /
 * residual / analytic spectra), so both builds run the identical battery
 * and any fallback-only bug shows up as a jacobi-build miss.
 *
 * Degenerate spectra intentionally verify only eigenVALUES + reconstruction
 * + orthonormality, never eigenVECTOR directions (which are gauge-arbitrary).
 */
#include "numerical_common.h"
#include "../../src/utils/matrix_math.h"

/* H[i*n+j], column-major eigenvectors V[i*n+j] = (v_j)_i. */
static double recon_error(const complex_t *H, const double *ev,
                          const complex_t *V, size_t n) {
    double maxerr = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            complex_t acc = 0.0;
            for (size_t j = 0; j < n; j++)
                acc += V[i*n + j] * ev[j] * conj(V[k*n + j]);
            double e = cabs(acc - H[i*n + k]);
            if (e > maxerr) maxerr = e;
        }
    }
    return maxerr;
}

static double ortho_error(const complex_t *V, size_t n) {
    double maxerr = 0.0;
    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            complex_t acc = 0.0;
            for (size_t i = 0; i < n; i++)
                acc += conj(V[i*n + a]) * V[i*n + b];
            complex_t want = (a == b) ? 1.0 : 0.0;
            double e = cabs(acc - want);
            if (e > maxerr) maxerr = e;
        }
    }
    return maxerr;
}

static double residual_error(const complex_t *H, const double *ev,
                             const complex_t *V, size_t n) {
    double maxerr = 0.0;
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            complex_t Hv = 0.0;
            for (size_t k = 0; k < n; k++)
                Hv += H[i*n + k] * V[k*n + j];
            double e = cabs(Hv - ev[j] * V[i*n + j]);
            if (e > maxerr) maxerr = e;
        }
    }
    return maxerr;
}

static double spectral_scale(const complex_t *H, size_t n) {
    double m = 1.0;
    for (size_t i = 0; i < n*n; i++) { double a = cabs(H[i]); if (a > m) m = a; }
    return m;
}

/* Run all implementation-agnostic checks on one Hermitian matrix. */
static void run_case(const char *name, const complex_t *H, size_t n) {
    double *ev = calloc(n, sizeof(double));
    complex_t *V = calloc(n * n, sizeof(complex_t));
    if (!ev || !V) { NC_MISS("%s: OOM", name); free(ev); free(V); return; }

    int rc = hermitian_eigen_decomposition(H, n, ev, V, 0, 1e-12);
    if (rc != 0) {
        NC_MISS("%s: decomposition returned %d (expected 0)", name, rc);
        free(ev); free(V); return;
    }

    /* finiteness */
    int bad = 0;
    for (size_t i = 0; i < n; i++) if (nc_is_bad(ev[i])) bad = 1;
    for (size_t i = 0; i < n*n; i++) if (nc_is_bad_c(V[i])) bad = 1;
    g_checks++;
    if (bad) { NC_MISS("%s: NaN/Inf in eigenpairs", name); free(ev); free(V); return; }

    /* descending order */
    g_checks++;
    for (size_t i = 1; i < n; i++) {
        if (ev[i] > ev[i-1] + 1e-9) {
            NC_MISS("%s: eigenvalues not descending: ev[%zu]=%.6g > ev[%zu]=%.6g",
                    name, i, ev[i], i-1, ev[i-1]);
            break;
        }
    }

    double sc = spectral_scale(H, n);
    double tol = 1e-8 * sc * (double)n;

    double re = recon_error(H, ev, V, n);
    if (re > tol) NC_MISS("%s: reconstruction err=%.3e tol=%.3e (n=%zu scale=%.3g)",
                          name, re, tol, n, sc);
    else g_checks++;

    double oe = ortho_error(V, n);
    if (oe > 1e-9 * (double)n) NC_MISS("%s: orthonormality err=%.3e (n=%zu)", name, oe, n);
    else g_checks++;

    double rse = residual_error(H, ev, V, n);
    if (rse > tol) NC_MISS("%s: residual ||Hv-lv|| err=%.3e tol=%.3e", name, rse, tol);
    else g_checks++;

    free(ev); free(V);
}

/* Like run_case but also asserts the sorted-descending eigenvalues match
 * an analytic reference. */
static void run_case_spectrum(const char *name, const complex_t *H, size_t n,
                              const double *expect_desc, double tol) {
    double *ev = calloc(n, sizeof(double));
    complex_t *V = calloc(n * n, sizeof(complex_t));
    if (!ev || !V) { NC_MISS("%s: OOM", name); free(ev); free(V); return; }
    int rc = hermitian_eigen_decomposition(H, n, ev, V, 0, 1e-12);
    if (rc != 0) { NC_MISS("%s: rc=%d", name, rc); free(ev); free(V); return; }
    for (size_t i = 0; i < n; i++) {
        char w[64]; snprintf(w, sizeof w, "%s ev[%zu]", name, i);
        nc_close(w, ev[i], expect_desc[i], tol);
    }
    free(ev); free(V);
    run_case(name, H, n);
}

/* ---- matrix builders --------------------------------------------------- */
static void set_h(complex_t *H, size_t n, size_t i, size_t j, complex_t v) {
    H[i*n + j] = v;
    H[j*n + i] = conj(v);
}

int main(void) {
#ifdef EIGEN_JACOBI_BUILD
    nc_begin("eigen_jacobi");
#else
    nc_begin("eigen_lapack");
#endif

    /* Pauli Z: diag(1,-1) -> descending {1,-1} */
    {
        complex_t H[4] = {1,0, 0,-1};
        double exp[2] = {1.0, -1.0};
        run_case_spectrum("pauli_z", H, 2, exp, 1e-12);
    }
    /* Pauli X: [[0,1],[1,0]] -> {1,-1} */
    {
        complex_t H[4] = {0,1, 1,0};
        double exp[2] = {1.0, -1.0};
        run_case_spectrum("pauli_x", H, 2, exp, 1e-12);
    }
    /* Pauli Y: [[0,-i],[i,0]] -> {1,-1}; exercises complex-Hermitian path */
    {
        complex_t H[4] = {0, -1.0*I, 1.0*I, 0};
        double exp[2] = {1.0, -1.0};
        run_case_spectrum("pauli_y", H, 2, exp, 1e-12);
    }
    /* Exactly-degenerate: diag(3,3,1) -> {3,3,1}. Eigenvectors arbitrary in
     * the 2D subspace; only values + reconstruction checked. */
    {
        size_t n = 3;
        complex_t *H = calloc(n*n, sizeof(complex_t));
        H[0]=3; H[4]=3; H[8]=1;
        double exp[3] = {3.0, 3.0, 1.0};
        run_case_spectrum("degenerate_331", H, n, exp, 1e-11);
        free(H);
    }
    /* Near-degenerate split: {2, 1+1e-12, 1} with an off-diagonal coupling. */
    {
        size_t n = 3;
        complex_t *H = calloc(n*n, sizeof(complex_t));
        H[0]=2.0; H[4]=1.0 + 1e-12; H[8]=1.0;
        set_h(H, n, 1, 2, 1e-13 + 1e-13*I);
        run_case("near_degenerate", H, n);
        free(H);
    }
    /* Rank-deficient projector |v><v|, v=(1,1,1,1)/2 -> eigenvalues {1,0,0,0} */
    {
        size_t n = 4;
        complex_t *H = calloc(n*n, sizeof(complex_t));
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                H[i*n + j] = 0.25; /* (1/2)(1/2) */
        double exp[4] = {1.0, 0.0, 0.0, 0.0};
        run_case_spectrum("rank1_projector", H, n, exp, 1e-11);
        free(H);
    }
    /* Tiny/huge spread: diag(1e14, 1.0, 1e-14) */
    {
        size_t n = 3;
        complex_t *H = calloc(n*n, sizeof(complex_t));
        H[0]=1e14; H[4]=1.0; H[8]=1e-14;
        double exp[3] = {1e14, 1.0, 1e-14};
        run_case_spectrum("tiny_huge_spread", H, n, exp, 1e-2 /*abs tol scaled by 1e14*/);
        free(H);
    }
    /* Near-zero-norm matrix: all magnitudes ~1e-160; must not NaN/overflow. */
    {
        size_t n = 4;
        complex_t *H = calloc(n*n, sizeof(complex_t));
        for (size_t i = 0; i < n; i++) H[i*n+i] = 1e-160 * (double)(i+1);
        set_h(H, n, 0, 1, 3e-161 + 2e-161*I);
        run_case("near_zero_norm", H, n);
        free(H);
    }
    /* Seeded random Hermitian, several sizes. */
    {
        uint64_t seed = 0xC0FFEEULL;
        NC_INFO("random-hermitian seed=0x%llx", (unsigned long long)seed);
        size_t sizes[] = {2, 4, 8, 16};
        for (size_t si = 0; si < 4; si++) {
            size_t n = sizes[si];
            complex_t *H = calloc(n*n, sizeof(complex_t));
            for (size_t i = 0; i < n; i++) {
                H[i*n+i] = nc_urand(&seed) * 5.0; /* real diagonal */
                for (size_t j = i+1; j < n; j++)
                    set_h(H, n, i, j, nc_urand(&seed) + nc_urand(&seed)*I);
            }
            char nm[48]; snprintf(nm, sizeof nm, "random_herm_n%zu", n);
            run_case(nm, H, n);
            free(H);
        }
    }

    return nc_end();
}
