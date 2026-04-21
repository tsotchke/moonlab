/**
 * @file gpu_eshkol_stub.c
 * @brief Pure-C no-op stub for the Eshkol GPU backend.
 *
 * The canonical Eshkol backend lives in gpu_eshkol.cpp and is only
 * compiled when Eshkol is built in.  On target platforms where the
 * C++ file cannot be compiled (notably the Emscripten / WASM build
 * which uses C-only sources), this file provides link-compatible
 * no-op implementations of the moonlab_eshkol_* API so callers in
 * tensor.c and elsewhere can depend on the symbols unconditionally
 * and take the "eshkol not available -> fall back to CPU" branch at
 * runtime.
 *
 * Build gating: this file is included ONLY in builds that do not
 * also build gpu_eshkol.cpp (to avoid duplicate-symbol errors).
 */

#include "gpu_eshkol.h"

int moonlab_eshkol_available(void) { return 0; }

moonlab_eshkol_status_t moonlab_eshkol_init(void) {
    return MOONLAB_ESHKOL_NOT_BUILT;
}

void moonlab_eshkol_set_precision(moonlab_eshkol_precision_t tier) {
    (void)tier;
}

moonlab_eshkol_precision_t moonlab_eshkol_get_precision(void) {
    return MOONLAB_ESHKOL_PRECISION_FAST;
}

moonlab_eshkol_status_t moonlab_eshkol_zgemm(
    const moonlab_cplx_t* A, size_t lda,
    const moonlab_cplx_t* B, size_t ldb,
    moonlab_cplx_t*       C, size_t ldc,
    size_t M, size_t K, size_t N,
    moonlab_cplx_t alpha, moonlab_cplx_t beta)
{
    (void)A; (void)lda; (void)B; (void)ldb; (void)C; (void)ldc;
    (void)M; (void)K; (void)N; (void)alpha; (void)beta;
    return MOONLAB_ESHKOL_NOT_BUILT;
}
