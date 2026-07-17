/*
 * gpu_stubs.c -- minimal "no GPU" stubs so tensor.c can be compiled
 * standalone (with the SVD/QR fallbacks forced) into a harness without
 * dragging in the Metal / Eshkol backends.  Every entry reports
 * "unavailable", forcing the CPU code paths -- which is exactly what we
 * want to exercise.
 *
 * Signatures mirror src/optimization/gpu/backends/gpu_eshkol.h and
 * src/optimization/gpu_metal.h.
 */
#include <stddef.h>
#include <complex.h>

typedef double _Complex moonlab_cplx_t;

/* ---- Eshkol GEMM backend (not weak; must be defined) ------------------- */
int  moonlab_eshkol_available(void) { return 0; }
int  moonlab_eshkol_get_precision(void) { return 0; }        /* PRECISION_EXACT */
int  moonlab_eshkol_init(void) { return -1; }                /* NOT_BUILT */
void moonlab_eshkol_set_precision(int tier) { (void)tier; }

int moonlab_eshkol_zgemm(
    const moonlab_cplx_t* A, size_t lda,
    const moonlab_cplx_t* B, size_t ldb,
    moonlab_cplx_t*       C, size_t ldc,
    size_t M, size_t K, size_t N,
    moonlab_cplx_t alpha, moonlab_cplx_t beta) {
    (void)A;(void)lda;(void)B;(void)ldb;(void)C;(void)ldc;
    (void)M;(void)K;(void)N;(void)alpha;(void)beta;
    return -1; /* MOONLAB_ESHKOL_NOT_BUILT */
}

/* ---- Metal compute backend (weak imports; stub to be safe) ------------- */
void *metal_compute_init(void) { return NULL; }
void  metal_compute_free(void *ctx) { (void)ctx; }
void *metal_buffer_create(void *ctx, size_t size) { (void)ctx; (void)size; return NULL; }
void *metal_buffer_contents(void *buffer) { (void)buffer; return NULL; }
void  metal_buffer_free(void *buffer) { (void)buffer; }
