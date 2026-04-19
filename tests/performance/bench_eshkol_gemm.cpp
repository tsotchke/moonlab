/**
 * @file bench_eshkol_gemm.cpp
 * @brief Proof-of-concept: fp64 complex GEMM via Eshkol Metal GPU backend.
 *
 * Measures Eshkol's `eshkol_gpu_matmul_f64` (SF64 / Ozaki-II CRT
 * auto-dispatch, bit-exact IEEE-754 fp64 on Apple Silicon GPU)
 * against Accelerate cblas_zgemm on the CPU.  Complex-to-real lift:
 * each complex GEMM C = A * B expands into four real GEMMs
 *
 *   Re(C) = Ar * Br - Ai * Bi
 *   Im(C) = Ar * Bi + Ai * Br
 *
 * which we submit as four separate Eshkol dispatches (simplest first
 * cut; a fused 2x2 block form is the next optimisation).
 *
 * Sizes are chosen to match the hot inner-Lanczos tensor contraction
 * in dmrg.c: M ~ physical_dim * chi, K ~ chi, N ~ chi, at chi in
 * {32, 64, 128, 256}.  This is where the 15-20x ITensor perf gap
 * lives on Moonlab today.
 */

#include <Accelerate/Accelerate.h>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

#if defined(HAS_ESHKOL) && defined(__APPLE__)
#  include <eshkol/backend/gpu/gpu_memory.h>
#  define ESHKOL_PRESENT 1
#else
#  define ESHKOL_PRESENT 0
#endif

using cplx_t = std::complex<double>;

static double now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void randomise_matrix(cplx_t* m, uint64_t n, unsigned seed) {
    std::srand(seed);
    for (uint64_t i = 0; i < n; i++) {
        double re = (double)std::rand() / RAND_MAX - 0.5;
        double im = (double)std::rand() / RAND_MAX - 0.5;
        m[i] = { re, im };
    }
}

static void split_real_imag(const cplx_t* src, double* re, double* im, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        re[i] = src[i].real();
        im[i] = src[i].imag();
    }
}

static void merge_real_imag(const double* re, const double* im, cplx_t* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) dst[i] = { re[i], im[i] };
}

static double bench_cpu(const cplx_t* A, const cplx_t* B, cplx_t* C,
                        uint64_t M, uint64_t K, uint64_t N, int iters) {
    const cplx_t alpha(1.0, 0.0), beta(0.0, 0.0);
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)M, (int)N, (int)K,
                    &alpha, A, (int)K, B, (int)N,
                    &beta, C, (int)N);
    }
    return (now_us() - t0) / iters;
}

static double l2_rel_diff(const cplx_t* a, const cplx_t* b, uint64_t n) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        cplx_t d = a[i] - b[i];
        num += std::norm(d);
        den += std::norm(a[i]);
    }
    return (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
}

#if ESHKOL_PRESENT

static double bench_gpu(const cplx_t* Ain, const cplx_t* Bin, cplx_t* Cout,
                        uint64_t M, uint64_t K, uint64_t N, int iters) {
    auto* Ar  = (double*)std::malloc(M * K * sizeof(double));
    auto* Ai  = (double*)std::malloc(M * K * sizeof(double));
    auto* Br  = (double*)std::malloc(K * N * sizeof(double));
    auto* Bi  = (double*)std::malloc(K * N * sizeof(double));
    auto* Cr  = (double*)std::calloc(M * N, sizeof(double));
    auto* Ci  = (double*)std::calloc(M * N, sizeof(double));
    auto* tmp = (double*)std::malloc(M * N * sizeof(double));

    split_real_imag(Ain, Ar, Ai, M * K);
    split_real_imag(Bin, Br, Bi, K * N);

    EshkolGPUBuffer bAr, bAi, bBr, bBi, bCr, bCi, bTmp;
    eshkol_gpu_wrap_host(Ar,  M*K*sizeof(double), &bAr);
    eshkol_gpu_wrap_host(Ai,  M*K*sizeof(double), &bAi);
    eshkol_gpu_wrap_host(Br,  K*N*sizeof(double), &bBr);
    eshkol_gpu_wrap_host(Bi,  K*N*sizeof(double), &bBi);
    eshkol_gpu_wrap_host(Cr,  M*N*sizeof(double), &bCr);
    eshkol_gpu_wrap_host(Ci,  M*N*sizeof(double), &bCi);
    eshkol_gpu_wrap_host(tmp, M*N*sizeof(double), &bTmp);

    for (int w = 0; w < 3; w++) {
        eshkol_gpu_matmul_f64(&bAr, &bBr, &bCr,  M, K, N);
        eshkol_gpu_matmul_f64(&bAi, &bBi, &bTmp, M, K, N);
        eshkol_gpu_matmul_f64(&bAr, &bBi, &bCi,  M, K, N);
        eshkol_gpu_matmul_f64(&bAi, &bBr, &bTmp, M, K, N);
    }

    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        eshkol_gpu_matmul_f64(&bAr, &bBr, &bCr,  M, K, N);
        eshkol_gpu_matmul_f64(&bAi, &bBi, &bTmp, M, K, N);
        for (uint64_t k = 0; k < M*N; k++) Cr[k] -= tmp[k];
        eshkol_gpu_matmul_f64(&bAr, &bBi, &bCi,  M, K, N);
        eshkol_gpu_matmul_f64(&bAi, &bBr, &bTmp, M, K, N);
        for (uint64_t k = 0; k < M*N; k++) Ci[k] += tmp[k];
    }
    double dt = (now_us() - t0) / iters;

    merge_real_imag(Cr, Ci, Cout, M * N);

    eshkol_gpu_free(&bAr);  eshkol_gpu_free(&bAi);
    eshkol_gpu_free(&bBr);  eshkol_gpu_free(&bBi);
    eshkol_gpu_free(&bCr);  eshkol_gpu_free(&bCi);
    eshkol_gpu_free(&bTmp);
    std::free(Ar); std::free(Ai); std::free(Br); std::free(Bi);
    std::free(Cr); std::free(Ci); std::free(tmp);
    return dt;
}

static void bench_size(uint64_t M, uint64_t K, uint64_t N) {
    auto* A    = (cplx_t*)std::malloc(M * K * sizeof(cplx_t));
    auto* B    = (cplx_t*)std::malloc(K * N * sizeof(cplx_t));
    auto* Cref = (cplx_t*)std::malloc(M * N * sizeof(cplx_t));
    auto* Cgpu = (cplx_t*)std::malloc(M * N * sizeof(cplx_t));
    randomise_matrix(A, M * K, 0xBAD1);
    randomise_matrix(B, K * N, 0xBAD2);

    const int iters = 5;
    double cpu_us = bench_cpu(A, B, Cref, M, K, N, iters);
    double gpu_us = bench_gpu(A, B, Cgpu, M, K, N, iters);
    double err = l2_rel_diff(Cref, Cgpu, M * N);

    double flops = 8.0 * (double)M * (double)K * (double)N;
    double cpu_gf = flops / (cpu_us * 1e-6) / 1e9;
    double gpu_gf = flops / (gpu_us * 1e-6) / 1e9;

    std::printf("  M=%-5llu K=%-4llu N=%-4llu  CPU %9.2f us (%6.1f GF/s)   "
                "GPU %9.2f us (%6.1f GF/s)  %5.2fx  L2_rel=%.2e\n",
                (unsigned long long)M, (unsigned long long)K, (unsigned long long)N,
                cpu_us, cpu_gf, gpu_us, gpu_gf, cpu_us / gpu_us, err);

    std::free(A); std::free(B); std::free(Cref); std::free(Cgpu);
}

int main() {
    std::printf("=== Eshkol fp64 complex GEMM vs Accelerate cblas_zgemm ===\n");
    /* eshkol_gpu_init returns 1 when a GPU backend (Metal/CUDA) is
     * active, 0 when none found. */
    if (eshkol_gpu_init() != 1) {
        std::printf("No GPU backend available (Metal/CUDA); skipping.\n");
        return 0;
    }
    /* eshkol_gpu_supports_f64 returns 1 only for NATIVE fp64 hardware
     * (CUDA dGPUs). Apple GPUs report 0 but Eshkol still runs bit-
     * exact fp64 via SF64 software-float + Ozaki-II CRT. */
    int has_native_f64 = eshkol_gpu_supports_f64();
    std::printf("Native fp64 HW: %s (SF64 + Ozaki-II fp64 path always available on Metal)\n\n",
                has_native_f64 ? "yes" : "no");

    std::printf("  shape                          CPU (zgemm)          Eshkol GPU (fp64 via SF64 / Ozaki-II)\n");
    std::printf("  -----                          -----------          --------------------------------------\n");
    bench_size( 64,  32,  32);
    bench_size(128,  64,  64);
    bench_size(256, 128, 128);
    bench_size(512, 256, 256);
    bench_size(1024, 256, 256);
    bench_size(1024, 512, 512);
    bench_size(2048, 1024, 1024);
    bench_size(4096, 2048, 2048);

    eshkol_gpu_shutdown();
    return 0;
}

#else  /* !ESHKOL_PRESENT */

int main() {
    std::printf("Eshkol not compiled in (configure with -DQSIM_ENABLE_ESHKOL=ON).\n");
    return 0;
}

#endif
