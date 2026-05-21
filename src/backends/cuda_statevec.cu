/**
 * @file  cuda_statevec.cu
 * @brief CUDA state-vector backend kernels + dispatch.
 *
 * Design notes (Jetson-first):
 *   - Allocations use cudaMallocManaged.  On Tegra SoCs this maps
 *     the same physical LPDDR pages into both CPU and GPU address
 *     spaces -- zero-copy across the host/device boundary.  On
 *     discrete GPUs the runtime still gives us a single pointer,
 *     and migrates pages lazily on access.  Same code, different
 *     bandwidth profile.
 *
 *   - Single-qubit gate kernel: each thread handles ONE pair of
 *     state indices that differ in the target bit.  The two
 *     amplitudes get a 2x2 matrix multiply.  Launch grid sized so
 *     that one thread runs per pair (total dim/2 pairs).
 *
 *   - CNOT kernel: each thread handles ONE pair where the control
 *     bit is 1.  Swap the target-bit-0 and target-bit-1 entries.
 *
 *   - Complex math: we use double2 to pack real/imag because
 *     cuComplex.h's cuDoubleComplex has the same layout but adds
 *     a wrapper dependency.  Our 2x2 multiply is inline.
 *
 *   - Error path: any cuda* failure returns
 *     MOONLAB_CUDA_ERR_DRIVER.  Caller can re-probe via the host's
 *     own logger; we don't propagate the cudaError_t enum to avoid
 *     exposing CUDA-runtime types in moonlab's C API.
 */

#include "cuda_statevec.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------- internal types ---------- */

struct moonlab_cuda_state {
    uint32_t        n_qubits;
    uint64_t        dim;
    double2        *amps;       /* unified-memory state vector */
    cudaStream_t    stream;
};

/* ---------- helpers ---------- */

#define CUDA_TRY(expr)                                              \
    do {                                                            \
        cudaError_t _e = (expr);                                    \
        if (_e != cudaSuccess) {                                    \
            fprintf(stderr,                                         \
                "moonlab_cuda: %s -> %s\n",                         \
                #expr, cudaGetErrorString(_e));                     \
            return MOONLAB_CUDA_ERR_DRIVER;                         \
        }                                                           \
    } while (0)

/* ---------- kernels ---------- */

/* 2x2 complex matmul:
 *   a0' = m00 * a0 + m01 * a1
 *   a1' = m10 * a0 + m11 * a1 */
__device__ static inline double2 cmul(double2 a, double2 b) {
    double2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

__device__ static inline double2 cadd(double2 a, double2 b) {
    double2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

/* Apply a 2x2 unitary M to qubit `target` of an n_qubits-state
 * vector.  Each thread handles ONE pair (i0, i1) where i0 has bit
 * `target` = 0 and i1 = i0 | (1 << target). */
__global__ void k_apply_1q(double2 * __restrict__ amps,
                           uint64_t              dim,
                           uint32_t              target,
                           double2 m00, double2 m01,
                           double2 m10, double2 m11)
{
    uint64_t pair = (uint64_t)blockIdx.x * (uint64_t)blockDim.x
                  +  (uint64_t)threadIdx.x;
    uint64_t n_pairs = dim >> 1;
    if (pair >= n_pairs) return;

    /* Map pair index -> i0 with bit `target` cleared.  We expand
     * `pair` over the bits below+above `target`, inserting a 0 at
     * position `target`.  Equivalent to:
     *   low  = pair & ((1 << target) - 1)
     *   high = pair >> target
     *   i0   = (high << (target + 1)) | low */
    uint64_t mask_lo = ((uint64_t)1 << target) - 1ULL;
    uint64_t low  = pair & mask_lo;
    uint64_t high = pair >> target;
    uint64_t i0   = (high << (target + 1)) | low;
    uint64_t i1   = i0 | ((uint64_t)1 << target);

    double2 a0 = amps[i0];
    double2 a1 = amps[i1];
    amps[i0] = cadd(cmul(m00, a0), cmul(m01, a1));
    amps[i1] = cadd(cmul(m10, a0), cmul(m11, a1));
}

/* CNOT(control, target): swap entries where control=1 differ in
 * target.  Each thread handles ONE pair (i0, i1) defined the same
 * way as in k_apply_1q but additionally requires the `control` bit
 * to be 1.  Half of the pairs are skipped (control=0). */
__global__ void k_apply_cnot(double2 * __restrict__ amps,
                             uint64_t              dim,
                             uint32_t              control,
                             uint32_t              target)
{
    uint64_t pair = (uint64_t)blockIdx.x * (uint64_t)blockDim.x
                  +  (uint64_t)threadIdx.x;
    uint64_t n_pairs = dim >> 1;
    if (pair >= n_pairs) return;

    uint64_t mask_lo = ((uint64_t)1 << target) - 1ULL;
    uint64_t low  = pair & mask_lo;
    uint64_t high = pair >> target;
    uint64_t i0   = (high << (target + 1)) | low;
    uint64_t i1   = i0 | ((uint64_t)1 << target);

    /* Only swap when control bit is 1 in BOTH i0 and i1.  If
     * control == target it's impossible -- guard at API level.
     * control bit is the same in i0 and i1 because they differ only
     * in `target`. */
    uint64_t ctrl_bit = ((uint64_t)1 << control);
    if ((i0 & ctrl_bit) == 0) return;

    double2 tmp = amps[i0];
    amps[i0] = amps[i1];
    amps[i1] = tmp;
}

/* ---------- public dispatch ---------- */

extern "C"
moonlab_cuda_status_t
moonlab_cuda_state_create(uint32_t n_qubits, moonlab_cuda_state_t **out_state)
{
    if (!out_state) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (n_qubits == 0 || n_qubits > 31)
        return MOONLAB_CUDA_ERR_NUM_QUBITS;

    int n_dev = 0;
    cudaError_t e = cudaGetDeviceCount(&n_dev);
    if (e != cudaSuccess || n_dev <= 0) return MOONLAB_CUDA_ERR_NO_DEVICE;

    moonlab_cuda_state_t *s = (moonlab_cuda_state_t *)calloc(1, sizeof(*s));
    if (!s) return MOONLAB_CUDA_ERR_OUT_OF_MEMORY;
    s->n_qubits = n_qubits;
    s->dim      = (uint64_t)1 << n_qubits;

    size_t bytes = (size_t)s->dim * sizeof(double2);
    if (cudaMallocManaged(&s->amps, bytes) != cudaSuccess) {
        free(s);
        return MOONLAB_CUDA_ERR_OUT_OF_MEMORY;
    }
    if (cudaStreamCreate(&s->stream) != cudaSuccess) {
        cudaFree(s->amps);
        free(s);
        return MOONLAB_CUDA_ERR_DRIVER;
    }

    /* Init to |0...0>: a[0] = 1+0i, all others 0.  cudaMemset
     * zeros the whole buffer; then set the first element from
     * host (unified mem: just a pointer write). */
    cudaMemsetAsync(s->amps, 0, bytes, s->stream);
    cudaStreamSynchronize(s->stream);
    s->amps[0].x = 1.0;
    s->amps[0].y = 0.0;

    *out_state = s;
    return MOONLAB_CUDA_OK;
}

extern "C"
void moonlab_cuda_state_free(moonlab_cuda_state_t *state)
{
    if (!state) return;
    if (state->amps)   cudaFree(state->amps);
    if (state->stream) cudaStreamDestroy(state->stream);
    free(state);
}

extern "C"
uint32_t moonlab_cuda_state_n_qubits(const moonlab_cuda_state_t *state)
{
    return state ? state->n_qubits : 0;
}

extern "C"
uint64_t moonlab_cuda_state_dim(const moonlab_cuda_state_t *state)
{
    return state ? state->dim : 0;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_apply_1q(moonlab_cuda_state_t *state,
                      uint32_t target,
                      const double m[8])
{
    if (!state || !m) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (target >= state->n_qubits) return MOONLAB_CUDA_ERR_INVALID_QUBIT;

    double2 m00 = { m[0], m[1] };
    double2 m01 = { m[2], m[3] };
    double2 m10 = { m[4], m[5] };
    double2 m11 = { m[6], m[7] };

    uint64_t n_pairs = state->dim >> 1;
    int  threads = 256;
    uint64_t blocks_u = (n_pairs + threads - 1) / threads;
    /* Clamp to int range for grid dim X (CUDA limit 2^31 - 1). */
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;

    k_apply_1q<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, target, m00, m01, m10, m11);
    CUDA_TRY(cudaGetLastError());
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_apply_cnot(moonlab_cuda_state_t *state,
                        uint32_t control,
                        uint32_t target)
{
    if (!state) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (control >= state->n_qubits || target >= state->n_qubits)
        return MOONLAB_CUDA_ERR_INVALID_QUBIT;
    if (control == target) return MOONLAB_CUDA_ERR_INVALID_QUBIT;

    uint64_t n_pairs = state->dim >> 1;
    int  threads = 256;
    uint64_t blocks_u = (n_pairs + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;

    k_apply_cnot<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, control, target);
    CUDA_TRY(cudaGetLastError());
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_state_copy_to_host(const moonlab_cuda_state_t *state,
                                double *out)
{
    if (!state || !out) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    CUDA_TRY(cudaStreamSynchronize(state->stream));
    /* On Tegra this is a plain memcpy from already-host-visible
     * pages.  On discrete it triggers a managed-memory copy via
     * the driver.  Either way, single API. */
    size_t bytes = (size_t)state->dim * sizeof(double2);
    memcpy(out, state->amps, bytes);
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_synchronize(moonlab_cuda_state_t *state)
{
    if (!state) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    CUDA_TRY(cudaStreamSynchronize(state->stream));
    return MOONLAB_CUDA_OK;
}
