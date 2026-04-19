/**
 * @file gpu_eshkol.h
 * @brief Moonlab bridge to the Eshkol GPU backend.
 *
 * Eshkol (github.com/tsotchke/eshkol) is a Metal/CUDA GPU compute
 * library that exposes a family of precision tiers on top of the
 * same underlying hardware:
 *
 *  - EXACT (tier 0, fp53 / SF64): bit-exact IEEE-754 fp64 via
 *    software-float or Ozaki-II CRT. Slowest; use when ULP-level
 *    correctness is required (topology invariants, Berry curvature
 *    publication results).
 *  - HIGH  (tier 1, df64):        ~fp32-grade precision in practice
 *    (~1e-7 L2 rel error measured, not the docs' advertised ~48-bit).
 *    Good enough for normalised quantum observables; 1.6-1.7x
 *    Accelerate on large complex GEMM.
 *  - FAST  (tier 2, f32):         native Metal fp32. ~3e-7 L2 rel
 *    error. 4-6x Accelerate at 2048 <= M,K,N. The right mode for
 *    VQE / QAOA / Trotter where optimiser or shot noise dominates
 *    roundoff.
 *  - ML    (tier 3, fp24):        1.4x Accelerate at M>=2048, same
 *    precision band as FAST but lower throughput on our GEMM
 *    benchmarks. Kept for completeness.
 *
 * This header exposes one entry point -- a complex fp64 GEMM that
 * dispatches through Eshkol when available and the precision tier
 * is appropriate. Every subsequent Moonlab path that does matrix-
 * matrix work on state vectors, MPS tensors, or MPO blocks routes
 * through here.
 *
 * When Moonlab is built without QSIM_ENABLE_ESHKOL, this header
 * still declares the API but `moonlab_eshkol_available()` returns 0
 * and `moonlab_eshkol_zgemm()` returns `MOONLAB_ESHKOL_NOT_BUILT`.
 * Callers fall back to Accelerate cblas_zgemm.
 *
 * @since v0.2.0-dev
 */

#ifndef MOONLAB_GPU_ESHKOL_H
#define MOONLAB_GPU_ESHKOL_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double _Complex moonlab_cplx_t;

typedef enum {
    MOONLAB_ESHKOL_PRECISION_EXACT = 0,  /* tier 0: fp53 bit-exact fp64 */
    MOONLAB_ESHKOL_PRECISION_HIGH  = 1,  /* tier 1: df64 (~1e-7 in practice) */
    MOONLAB_ESHKOL_PRECISION_FAST  = 2,  /* tier 2: f32 */
    MOONLAB_ESHKOL_PRECISION_ML    = 3,  /* tier 3: fp24 */
} moonlab_eshkol_precision_t;

typedef enum {
    MOONLAB_ESHKOL_OK              =  0,
    MOONLAB_ESHKOL_NOT_BUILT       = -1, /* QSIM_ENABLE_ESHKOL was off at compile time */
    MOONLAB_ESHKOL_NO_GPU          = -2, /* Eshkol reports no GPU backend */
    MOONLAB_ESHKOL_INVALID_ARGS    = -3,
    MOONLAB_ESHKOL_DISPATCH_FAILED = -4,
    MOONLAB_ESHKOL_OOM             = -5,
} moonlab_eshkol_status_t;

/**
 * @brief Check whether Eshkol is both built and has a live GPU
 *        backend. Cheap; can be called on every dispatch decision.
 *
 * @return 1 when Eshkol + GPU are available, 0 otherwise.
 */
int moonlab_eshkol_available(void);

/**
 * @brief Idempotent lazy-init of the Eshkol GPU backend. Safe to
 *        call from multiple threads; the first caller initialises,
 *        subsequent calls are no-ops.
 *
 * @return MOONLAB_ESHKOL_OK, MOONLAB_ESHKOL_NOT_BUILT, or
 *         MOONLAB_ESHKOL_NO_GPU.
 */
moonlab_eshkol_status_t moonlab_eshkol_init(void);

/**
 * @brief Set the global precision tier used by subsequent
 *        moonlab_eshkol_zgemm calls. Maps to Eshkol's
 *        ESHKOL_GPU_PRECISION env var semantics but lets the
 *        process change tier at runtime without shelling out.
 */
void moonlab_eshkol_set_precision(moonlab_eshkol_precision_t tier);

moonlab_eshkol_precision_t moonlab_eshkol_get_precision(void);

/**
 * @brief Complex fp64 GEMM: C = alpha A B + beta C.
 *
 * A is M x K, B is K x N, C is M x N; all row-major with user-
 * specified leading dimensions. Complex-to-real lift (4 real GEMM
 * dispatches) is applied internally when the active precision tier
 * does not have a native complex path in Eshkol.
 *
 * For small problems (roughly M*K*N < 2^24 across all tiers, or the
 * whole problem fits in L3 at any tier), the caller should prefer
 * Accelerate cblas_zgemm directly -- dispatching to the GPU costs
 * more than it saves in that regime (see bench_eshkol_gemm.cpp).
 * This function does not apply that heuristic itself; callers
 * (notably tensor_contract) make the dispatch decision.
 *
 * @param A      pointer to interleaved (re, im) complex doubles
 * @param lda    leading dimension of A (stride between rows, >= K)
 * @param B      pointer to interleaved complex doubles
 * @param ldb    leading dimension of B (>= N)
 * @param C      pointer to interleaved complex doubles (in/out)
 * @param ldc    leading dimension of C (>= N)
 * @param M, K, N  matrix dimensions
 * @param alpha  scalar on A*B
 * @param beta   scalar on existing C (if beta != 0 + 0i, C is read
 *               before being written, which costs an extra traffic
 *               round)
 * @return MOONLAB_ESHKOL_OK on success, negative on error.
 */
moonlab_eshkol_status_t moonlab_eshkol_zgemm(
    const moonlab_cplx_t* A, size_t lda,
    const moonlab_cplx_t* B, size_t ldb,
    moonlab_cplx_t*       C, size_t ldc,
    size_t M, size_t K, size_t N,
    moonlab_cplx_t alpha, moonlab_cplx_t beta);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_GPU_ESHKOL_H */
