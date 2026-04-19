/**
 * @file gpu_eshkol.cpp
 * @brief Moonlab bridge implementation for Eshkol's GPU backend.
 *
 * Split into two halves via the QSIM_ENABLE_ESHKOL macro:
 *   - When on:  actual Eshkol calls, zero-copy buffer wrap, 4-real
 *               GEMM lift for complex multiplication, precision-tier
 *               dispatch via Eshkol's ESHKOL_GPU_PRECISION semantics
 *               (communicated through setenv).
 *   - When off: stub implementation that reports NOT_BUILT so the
 *               rest of Moonlab can use this API unconditionally.
 */

#include "gpu_eshkol.h"

#ifdef QSIM_ENABLE_ESHKOL

#include <Accelerate/Accelerate.h>
#include <eshkol/backend/gpu/gpu_memory.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

std::atomic<int>          g_init_state{0};     /* 0 not-tried, 1 ok, -1 no-gpu */
std::atomic<int>          g_active_tier{-1};    /* last tier passed through init */
std::mutex                g_init_mutex;
std::atomic<int>          g_precision{MOONLAB_ESHKOL_PRECISION_FAST};

/* Map moonlab tier -> Eshkol env var value. */
const char* tier_name(moonlab_eshkol_precision_t t) {
    switch (t) {
        case MOONLAB_ESHKOL_PRECISION_EXACT: return "default";
        case MOONLAB_ESHKOL_PRECISION_HIGH:  return "high";
        case MOONLAB_ESHKOL_PRECISION_FAST:  return "fast";
        case MOONLAB_ESHKOL_PRECISION_ML:    return "ml";
    }
    return "fast";
}

/* Thin RAII wrapper around EshkolGPUBuffer. */
struct EshkolHostBuf {
    EshkolGPUBuffer buf{};
    bool            wrapped = false;
    int wrap(void* host_ptr, size_t bytes) {
        int rc = eshkol_gpu_wrap_host(host_ptr, bytes, &buf);
        wrapped = (rc == 0);
        return rc;
    }
    ~EshkolHostBuf() {
        if (wrapped) eshkol_gpu_free(&buf);
    }
};

/* Split interleaved complex (re, im) into separate real/imag
 * contiguous arrays. Honours leading dimension ld (elements not
 * bytes). In C++ the C99 creal/cimag/_Complex_I machinery is not
 * reliably exposed by <complex.h>, so we use the clang/gcc
 * __real__ / __imag__ extensions, which work identically in both
 * language modes. */
void split_complex(const moonlab_cplx_t* src, size_t M, size_t N, size_t ld,
                   double* re_out, double* im_out) {
    for (size_t i = 0; i < M; i++) {
        const moonlab_cplx_t* row = src + i * ld;
        for (size_t j = 0; j < N; j++) {
            re_out[i * N + j] = __real__(row[j]);
            im_out[i * N + j] = __imag__(row[j]);
        }
    }
}

static inline moonlab_cplx_t make_complex(double re, double im) {
    moonlab_cplx_t z;
    __real__(z) = re;
    __imag__(z) = im;
    return z;
}

void merge_complex(const double* re_in, const double* im_in,
                   size_t M, size_t N, size_t ld,
                   moonlab_cplx_t alpha, moonlab_cplx_t beta,
                   moonlab_cplx_t* dst) {
    /* dst = alpha * (re + i*im) + beta * dst */
    for (size_t i = 0; i < M; i++) {
        moonlab_cplx_t* row = dst + i * ld;
        for (size_t j = 0; j < N; j++) {
            moonlab_cplx_t prod = make_complex(re_in[i * N + j],
                                               im_in[i * N + j]);
            row[j] = alpha * prod + beta * row[j];
        }
    }
}

} /* namespace */

extern "C" {

int moonlab_eshkol_available(void) {
    if (moonlab_eshkol_init() != MOONLAB_ESHKOL_OK) return 0;
    return 1;
}

/* Internal: perform (or redo) the Eshkol init at the currently
 * requested precision tier. Caller must hold g_init_mutex. */
static moonlab_eshkol_status_t do_init_locked() {
    int requested = g_precision.load(std::memory_order_acquire);
    int active    = g_active_tier.load(std::memory_order_relaxed);

    /* If already initialised and at the right tier, nothing to do. */
    if (g_init_state.load(std::memory_order_relaxed) == 1 &&
        active == requested) {
        return MOONLAB_ESHKOL_OK;
    }

    /* Teardown if previously inited at a different tier. */
    if (g_init_state.load(std::memory_order_relaxed) != 0) {
        eshkol_gpu_shutdown();
        g_init_state.store(0, std::memory_order_relaxed);
        g_active_tier.store(-1, std::memory_order_relaxed);
    }

    /* Set the env Eshkol reads at init time, overwriting any prior
     * value (third arg = 1). */
    setenv("ESHKOL_GPU_PRECISION",
           tier_name((moonlab_eshkol_precision_t)requested), 1);

    int rc = eshkol_gpu_init();
    /* eshkol_gpu_init returns 1 on success (non-standard; we flip). */
    int ok = (rc == 1) ? 1 : -1;
    g_init_state.store(ok, std::memory_order_release);
    if (ok == 1) g_active_tier.store(requested, std::memory_order_release);
    return (ok == 1) ? MOONLAB_ESHKOL_OK : MOONLAB_ESHKOL_NO_GPU;
}

moonlab_eshkol_status_t moonlab_eshkol_init(void) {
    {
        /* Fast path: already OK at the requested tier. */
        int cached   = g_init_state.load(std::memory_order_acquire);
        int active   = g_active_tier.load(std::memory_order_acquire);
        int wanted   = g_precision.load(std::memory_order_acquire);
        if (cached == 1 && active == wanted) return MOONLAB_ESHKOL_OK;
        if (cached == -1)                     return MOONLAB_ESHKOL_NO_GPU;
    }
    std::lock_guard<std::mutex> lock(g_init_mutex);
    return do_init_locked();
}

void moonlab_eshkol_set_precision(moonlab_eshkol_precision_t tier) {
    int prev = g_precision.exchange((int)tier, std::memory_order_acq_rel);
    if (prev == (int)tier) return;
    /* Force a reinit on the next available() / init() / zgemm() call.
     * We also proactively reinit here if Eshkol was already initialised,
     * so that a subsequent zgemm inside an OpenMP region does not pay
     * the reinit cost. */
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_init_state.load(std::memory_order_relaxed) == 1) {
        do_init_locked();
    }
}

moonlab_eshkol_precision_t moonlab_eshkol_get_precision(void) {
    return (moonlab_eshkol_precision_t)g_precision.load(std::memory_order_acquire);
}

moonlab_eshkol_status_t moonlab_eshkol_zgemm(
    const moonlab_cplx_t* A, size_t lda,
    const moonlab_cplx_t* B, size_t ldb,
    moonlab_cplx_t*       C, size_t ldc,
    size_t M, size_t K, size_t N,
    moonlab_cplx_t alpha, moonlab_cplx_t beta)
{
    if (!A || !B || !C)           return MOONLAB_ESHKOL_INVALID_ARGS;
    if (lda < K || ldb < N || ldc < N)
                                   return MOONLAB_ESHKOL_INVALID_ARGS;

    moonlab_eshkol_status_t init = moonlab_eshkol_init();
    if (init != MOONLAB_ESHKOL_OK) return init;

    /* Allocate split-real scratch. Four MxK / KxN / MxN real buffers
     * plus one temporary accumulator. */
    double* Ar  = (double*)std::malloc(M * K * sizeof(double));
    double* Ai  = (double*)std::malloc(M * K * sizeof(double));
    double* Br  = (double*)std::malloc(K * N * sizeof(double));
    double* Bi  = (double*)std::malloc(K * N * sizeof(double));
    double* Cr  = (double*)std::calloc(M * N, sizeof(double));
    double* Ci  = (double*)std::calloc(M * N, sizeof(double));
    double* tmp = (double*)std::malloc(M * N * sizeof(double));
    if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci || !tmp) {
        std::free(Ar); std::free(Ai); std::free(Br); std::free(Bi);
        std::free(Cr); std::free(Ci); std::free(tmp);
        return MOONLAB_ESHKOL_OOM;
    }

    split_complex(A, M, K, lda, Ar, Ai);
    split_complex(B, K, N, ldb, Br, Bi);

    EshkolHostBuf bAr, bAi, bBr, bBi, bCr, bCi, bTmp;
    if (bAr.wrap(Ar,  M * K * sizeof(double)) != 0 ||
        bAi.wrap(Ai,  M * K * sizeof(double)) != 0 ||
        bBr.wrap(Br,  K * N * sizeof(double)) != 0 ||
        bBi.wrap(Bi,  K * N * sizeof(double)) != 0 ||
        bCr.wrap(Cr,  M * N * sizeof(double)) != 0 ||
        bCi.wrap(Ci,  M * N * sizeof(double)) != 0 ||
        bTmp.wrap(tmp, M * N * sizeof(double)) != 0) {
        std::free(Ar); std::free(Ai); std::free(Br); std::free(Bi);
        std::free(Cr); std::free(Ci); std::free(tmp);
        return MOONLAB_ESHKOL_DISPATCH_FAILED;
    }

    /* 4-real-GEMM lift:
     *   Re(C) = Ar * Br - Ai * Bi
     *   Im(C) = Ar * Bi + Ai * Br
     * Each Eshkol call overwrites its output buffer, so we emulate
     * the subtraction / addition on CPU after each pair. Future:
     * fused 2x2 block form once Eshkol exposes accumulation in a
     * single dispatch. */
    if (eshkol_gpu_matmul_f64(&bAr.buf, &bBr.buf, &bCr.buf,  M, K, N) != 0 ||
        eshkol_gpu_matmul_f64(&bAi.buf, &bBi.buf, &bTmp.buf, M, K, N) != 0) {
        return MOONLAB_ESHKOL_DISPATCH_FAILED;
    }
    for (size_t i = 0; i < M * N; i++) Cr[i] -= tmp[i];
    if (eshkol_gpu_matmul_f64(&bAr.buf, &bBi.buf, &bCi.buf,  M, K, N) != 0 ||
        eshkol_gpu_matmul_f64(&bAi.buf, &bBr.buf, &bTmp.buf, M, K, N) != 0) {
        return MOONLAB_ESHKOL_DISPATCH_FAILED;
    }
    for (size_t i = 0; i < M * N; i++) Ci[i] += tmp[i];

    merge_complex(Cr, Ci, M, N, ldc, alpha, beta, C);

    std::free(Ar); std::free(Ai); std::free(Br); std::free(Bi);
    std::free(Cr); std::free(Ci); std::free(tmp);
    return MOONLAB_ESHKOL_OK;
}

} /* extern "C" */

#else  /* !QSIM_ENABLE_ESHKOL : stub */

extern "C" {

int moonlab_eshkol_available(void) { return 0; }

moonlab_eshkol_status_t moonlab_eshkol_init(void) {
    return MOONLAB_ESHKOL_NOT_BUILT;
}

void moonlab_eshkol_set_precision(moonlab_eshkol_precision_t) {}

moonlab_eshkol_precision_t moonlab_eshkol_get_precision(void) {
    return MOONLAB_ESHKOL_PRECISION_FAST;
}

moonlab_eshkol_status_t moonlab_eshkol_zgemm(
    const moonlab_cplx_t*, size_t,
    const moonlab_cplx_t*, size_t,
    moonlab_cplx_t*,       size_t,
    size_t, size_t, size_t,
    moonlab_cplx_t, moonlab_cplx_t) {
    return MOONLAB_ESHKOL_NOT_BUILT;
}

} /* extern "C" */

#endif /* QSIM_ENABLE_ESHKOL */
