/**
 * @file test_gpu_eshkol.cpp
 * @brief Parity test: moonlab_eshkol_zgemm vs Accelerate cblas_zgemm.
 *
 * Tolerance depends on the active precision tier. We parametrise.
 * When built without QSIM_ENABLE_ESHKOL, the bridge reports
 * NOT_BUILT and the test exits 0 after asserting the stub path
 * returns correctly.
 */

#include "../../src/optimization/gpu/backends/gpu_eshkol.h"

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* Portable CBLAS reference: prefer Apple Accelerate on macOS, fall back
 * to OpenBLAS / system cblas on Linux.  The test only needs cblas_zgemm
 * as the parity baseline for moonlab_eshkol_zgemm.  If no CBLAS header
 * can be located, the whole test turns into a no-op (exits 0) so the
 * ctest tier on bare-dependency runners still succeeds. */
#if defined(__APPLE__)
  #define MOONLAB_HAS_CBLAS 1
  #include <Accelerate/Accelerate.h>
#elif __has_include(<cblas.h>)
  #define MOONLAB_HAS_CBLAS 1
  #include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
  #define MOONLAB_HAS_CBLAS 1
  #include <openblas/cblas.h>
#else
  #define MOONLAB_HAS_CBLAS 0
#endif

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                     \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);      \
        failures++;                                                    \
    } else {                                                           \
        std::fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);      \
    }                                                                  \
} while (0)

using cplx = std::complex<double>;

static void randomise(cplx* m, std::size_t n, unsigned seed) {
    std::srand(seed);
    for (std::size_t i = 0; i < n; i++) {
        double re = (double)std::rand() / RAND_MAX - 0.5;
        double im = (double)std::rand() / RAND_MAX - 0.5;
        m[i] = { re, im };
    }
}

static double l2_rel(const cplx* a, const cplx* b, std::size_t n) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        cplx d = a[i] - b[i];
        num += std::norm(d);
        den += std::norm(a[i]);
    }
    return (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
}

#if MOONLAB_HAS_CBLAS
static void zgemm_cpu(const cplx* A, const cplx* B, cplx* C,
                     std::size_t M, std::size_t K, std::size_t N) {
    const cplx alpha(1.0, 0.0), beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, &alpha,
                A, (int)K, B, (int)N,
                &beta, C, (int)N);
}
#endif

#if MOONLAB_HAS_CBLAS
static void parity(std::size_t M, std::size_t K, std::size_t N,
                   moonlab_eshkol_precision_t tier, const char* label,
                   double tol) {
    std::fprintf(stdout, "\n-- parity %s @ (%zu,%zu,%zu) tol=%.1e --\n",
                 label, M, K, N, tol);
    cplx* A   = new cplx[M * K];
    cplx* B   = new cplx[K * N];
    cplx* Cref= new cplx[M * N];
    cplx* Cbr = new cplx[M * N];
    randomise(A, M * K, 0xA01);
    randomise(B, K * N, 0xB02);
    for (std::size_t i = 0; i < M * N; i++) Cbr[i] = Cref[i] = {0.0, 0.0};

    zgemm_cpu(A, B, Cref, M, K, N);

    moonlab_eshkol_set_precision(tier);
    /* std::complex<double> has the same ABI as C99 _Complex double
     * (two contiguous doubles), so it's safe to reinterpret through
     * the C-linkage boundary. We construct the alpha/beta scalars
     * the same way. */
    const cplx one(1.0, 0.0), zero(0.0, 0.0);
    moonlab_eshkol_status_t st = moonlab_eshkol_zgemm(
        reinterpret_cast<const moonlab_cplx_t*>(A), K,
        reinterpret_cast<const moonlab_cplx_t*>(B), N,
        reinterpret_cast<moonlab_cplx_t*>(Cbr), N,
        M, K, N,
        *reinterpret_cast<const moonlab_cplx_t*>(&one),
        *reinterpret_cast<const moonlab_cplx_t*>(&zero));

    if (st == MOONLAB_ESHKOL_NOT_BUILT) {
        CHECK(!moonlab_eshkol_available(),
              "stub path reports NOT_BUILT and not available");
    } else if (st == MOONLAB_ESHKOL_NO_GPU) {
        std::fprintf(stdout, "  SKIP  no Eshkol GPU backend on this host\n");
    } else {
        CHECK(st == MOONLAB_ESHKOL_OK,
              "moonlab_eshkol_zgemm returns OK (got %d)", (int)st);
        double err = l2_rel(Cref, Cbr, M * N);
        std::fprintf(stdout, "    L2 rel diff = %.2e\n", err);
        CHECK(err <= tol, "L2 rel %.2e <= %.1e", err, tol);
    }
    delete[] A; delete[] B; delete[] Cref; delete[] Cbr;
}
#endif

int main() {
    std::fprintf(stdout, "=== Moonlab-Eshkol bridge parity ===\n");
#if !MOONLAB_HAS_CBLAS
    std::fprintf(stdout,
        "No CBLAS reference available on this platform; skipping parity test.\n");
    return EXIT_SUCCESS;
#else
    if (!moonlab_eshkol_available()) {
        std::fprintf(stdout,
            "Eshkol not available (NOT_BUILT or NO_GPU). Testing stub.\n");
        /* With the stub, zgemm should return NOT_BUILT; with Eshkol
         * built but NO_GPU, it should return NO_GPU. The parity()
         * helper handles both. */
    }
    parity(64,  32,  32, MOONLAB_ESHKOL_PRECISION_EXACT, "EXACT", 1e-12);
    parity(128, 64,  64, MOONLAB_ESHKOL_PRECISION_HIGH,  "HIGH",  5e-6);
    parity(128, 64,  64, MOONLAB_ESHKOL_PRECISION_FAST,  "FAST",  5e-6);
#endif

    std::fprintf(stdout, "\n=== %d failure%s ===\n",
                 failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
