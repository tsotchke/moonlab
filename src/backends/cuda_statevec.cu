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

/* Apply an arbitrary 4x4 unitary M to qubits (q0, q1).  Basis
 * ordering: |q1 q0>, with q0 the low bit.  Each thread handles
 * one quadruple (i00, i01, i10, i11) -- four state indices that
 * share all other-qubit values and differ only in (q0, q1).
 *
 * Matrix layout in constant args:
 *   M[r][c] = m[8*r + 2*c]  (real),  m[8*r + 2*c + 1] (imag)
 *
 * We pass the matrix as 16 double2's via __constant__ memory --
 * faster than 32 scalar doubles in the launch param list. */
__constant__ double2 k_apply_2q_M[16];

__global__ void k_apply_2q(double2 * __restrict__ amps,
                           uint64_t              dim,
                           uint32_t              q0,
                           uint32_t              q1)
{
    uint64_t quad = (uint64_t)blockIdx.x * (uint64_t)blockDim.x
                  +  (uint64_t)threadIdx.x;
    uint64_t n_quads = dim >> 2;
    if (quad >= n_quads) return;

    /* Expand `quad` over the bits below/between/above (q0, q1),
     * inserting 0s at positions q0 and q1.  Assume q0 < q1 for
     * the bit-insertion math; the public dispatch swaps if needed. */
    uint32_t lo = q0, hi = q1;
    uint64_t mask_lo  = ((uint64_t)1 << lo) - 1ULL;
    uint64_t mask_mid = (((uint64_t)1 << (hi - 1)) - 1ULL) & ~mask_lo;
    uint64_t a = quad & mask_lo;
    uint64_t b = (quad >> lo) & ((mask_mid) >> lo);
    uint64_t c = quad >> (hi - 1);
    /* base has bits at q0 and q1 cleared; index in the basis
     * |q1 q0> by setting those bits. */
    uint64_t base = a | (b << (lo + 1)) | (c << (hi + 1));
    uint64_t bit0 = (uint64_t)1 << q0;
    uint64_t bit1 = (uint64_t)1 << q1;

    uint64_t i00 = base;
    uint64_t i01 = base | bit0;
    uint64_t i10 = base | bit1;
    uint64_t i11 = base | bit0 | bit1;

    double2 v0 = amps[i00];
    double2 v1 = amps[i01];
    double2 v2 = amps[i10];
    double2 v3 = amps[i11];

    /* M is a row-major 4x4 acting on (|00>, |01>, |10>, |11>). */
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        double2 sum = { 0.0, 0.0 };
        sum = cadd(sum, cmul(k_apply_2q_M[4*r + 0], v0));
        sum = cadd(sum, cmul(k_apply_2q_M[4*r + 1], v1));
        sum = cadd(sum, cmul(k_apply_2q_M[4*r + 2], v2));
        sum = cadd(sum, cmul(k_apply_2q_M[4*r + 3], v3));
        switch (r) {
            case 0: amps[i00] = sum; break;
            case 1: amps[i01] = sum; break;
            case 2: amps[i10] = sum; break;
            case 3: amps[i11] = sum; break;
        }
    }
}

/* P(qubit=1) via per-block sum of |amps[k]|^2 for k with the
 * target bit set, followed by a host-side accumulate.  We write
 * one partial sum per block to a small scratch buffer.  This is
 * not the fastest reduction (atomicAdd on double would be lower
 * launch overhead) but is correct and portable across cc 6.0+. */
__global__ void k_prob_z(const double2 * __restrict__ amps,
                         uint64_t                    dim,
                         uint32_t                    target,
                         double        * __restrict__ block_partials)
{
    __shared__ double sdata[256];
    uint64_t pair = (uint64_t)blockIdx.x * (uint64_t)blockDim.x
                  +  (uint64_t)threadIdx.x;
    uint64_t n_pairs = dim >> 1;

    double local = 0.0;
    if (pair < n_pairs) {
        uint64_t mask_lo = ((uint64_t)1 << target) - 1ULL;
        uint64_t low  = pair & mask_lo;
        uint64_t high = pair >> target;
        uint64_t i1   = (high << (target + 1)) | low | ((uint64_t)1 << target);
        double2 a = amps[i1];
        local = a.x * a.x + a.y * a.y;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_partials[blockIdx.x] = sdata[0];
}

/* Total state norm via same reduction pattern.  Each thread
 * sums one amplitude's squared magnitude; per-block reduce; host
 * accumulates the block partials. */
__global__ void k_norm(const double2 * __restrict__ amps,
                       uint64_t                    dim,
                       double        * __restrict__ block_partials)
{
    __shared__ double sdata[256];
    uint64_t k = (uint64_t)blockIdx.x * (uint64_t)blockDim.x
              +  (uint64_t)threadIdx.x;

    double local = 0.0;
    if (k < dim) {
        double2 a = amps[k];
        local = a.x * a.x + a.y * a.y;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_partials[blockIdx.x] = sdata[0];
}

/* Fast-path Pauli-X: swap the two halves of each (target=0,
 * target=1) pair.  No matmul. */
__global__ void k_apply_x(double2 * __restrict__ amps,
                          uint64_t              dim,
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

    double2 tmp = amps[i0];
    amps[i0] = amps[i1];
    amps[i1] = tmp;
}

/* Fast-path Hadamard: (1/sqrt(2)) [[1,1],[1,-1]] inline. */
__global__ void k_apply_h(double2 * __restrict__ amps,
                          uint64_t              dim,
                          uint32_t              target,
                          double                inv_sqrt2)
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

    double2 a0 = amps[i0];
    double2 a1 = amps[i1];
    double2 out0 = { inv_sqrt2 * (a0.x + a1.x), inv_sqrt2 * (a0.y + a1.y) };
    double2 out1 = { inv_sqrt2 * (a0.x - a1.x), inv_sqrt2 * (a0.y - a1.y) };
    amps[i0] = out0;
    amps[i1] = out1;
}

/* Runtime probe of "is there a discrete CUDA GPU?".
 * Called by cuda_tegra_probe.c when the device-tree check fails
 * (i.e. we're not on a Jetson but might be on a stock PCIe NVIDIA).
 * Returns 1=discrete present, 0=no devices, -1=integrated, -2=error.
 * Marked extern "C" so the C-side weak symbol resolves correctly. */
extern "C"
int moonlab_cuda_runtime_probe_discrete(void)
{
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return -2;
    if (n <= 0) return 0;
    /* Look at device 0.  If it's integrated, return -1 so the
     * caller can decide.  If not, we have at least one discrete
     * GPU visible to the runtime. */
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) return -2;
    return props.integrated ? -1 : 1;
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
moonlab_cuda_state_copy_from_host(moonlab_cuda_state_t *state,
                                  const double *in)
{
    if (!state || !in) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    /* Same symmetry as copy_to_host: on managed memory both Tegra
     * and discrete just see this as memcpy + lazy migration on
     * next kernel launch.  We stream-sync first to make sure no
     * in-flight kernel is reading the buffer when we overwrite. */
    CUDA_TRY(cudaStreamSynchronize(state->stream));
    size_t bytes = (size_t)state->dim * sizeof(double2);
    memcpy(state->amps, in, bytes);
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

/* ---------- new gates: apply_2q / apply_x / apply_h ---------- */

extern "C"
moonlab_cuda_status_t
moonlab_cuda_apply_2q(moonlab_cuda_state_t *state,
                      uint32_t q0,
                      uint32_t q1,
                      const double m[32])
{
    if (!state || !m) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (q0 >= state->n_qubits || q1 >= state->n_qubits)
        return MOONLAB_CUDA_ERR_INVALID_QUBIT;
    if (q0 == q1) return MOONLAB_CUDA_ERR_INVALID_QUBIT;

    /* The kernel assumes q0 < q1 (used in bit-insertion math).
     * If caller passes them swapped, swap and reorder the matrix
     * accordingly.  Re-ordering swaps row/col bits of the basis
     * mapping: |q1 q0> -> |q0 q1> permutes rows/cols (0,1,2,3) ->
     * (0,2,1,3). */
    double m_ordered[32];
    if (q0 < q1) {
        for (int i = 0; i < 32; i++) m_ordered[i] = m[i];
    } else {
        const int perm[4] = {0, 2, 1, 3};  /* swap bits */
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int src = 8 * perm[r] + 2 * perm[c];
                int dst = 8 *      r  + 2 *      c;
                m_ordered[dst    ] = m[src    ];
                m_ordered[dst + 1] = m[src + 1];
            }
        }
        uint32_t tmp = q0; q0 = q1; q1 = tmp;
    }

    /* Upload matrix to constant memory.  16 double2 = 32 doubles. */
    double2 host_m[16];
    for (int i = 0; i < 16; i++) {
        host_m[i].x = m_ordered[2 * i];
        host_m[i].y = m_ordered[2 * i + 1];
    }
    CUDA_TRY(cudaMemcpyToSymbolAsync(k_apply_2q_M, host_m, sizeof(host_m),
                                     0, cudaMemcpyHostToDevice, state->stream));

    uint64_t n_quads = state->dim >> 2;
    int  threads = 256;
    uint64_t blocks_u = (n_quads + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;

    k_apply_2q<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, q0, q1);
    CUDA_TRY(cudaGetLastError());
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_apply_x(moonlab_cuda_state_t *state, uint32_t target)
{
    if (!state) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (target >= state->n_qubits) return MOONLAB_CUDA_ERR_INVALID_QUBIT;
    uint64_t n_pairs = state->dim >> 1;
    int  threads = 256;
    uint64_t blocks_u = (n_pairs + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;
    k_apply_x<<<blocks, threads, 0, state->stream>>>(state->amps, state->dim, target);
    CUDA_TRY(cudaGetLastError());
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_apply_h(moonlab_cuda_state_t *state, uint32_t target)
{
    if (!state) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (target >= state->n_qubits) return MOONLAB_CUDA_ERR_INVALID_QUBIT;
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    uint64_t n_pairs = state->dim >> 1;
    int  threads = 256;
    uint64_t blocks_u = (n_pairs + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;
    k_apply_h<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, target, inv_sqrt2);
    CUDA_TRY(cudaGetLastError());
    return MOONLAB_CUDA_OK;
}

/* ---------- reductions: prob_z / norm ---------- */

/* Helper: run a reduction kernel that writes one partial per block,
 * then sum partials on the host.  Returns the final scalar in *out. */
static moonlab_cuda_status_t
reduce_via_kernel(moonlab_cuda_state_t *state,
                  void (*launch)(double * /* partials */,
                                 int /* blocks */, int /* threads */,
                                 cudaStream_t),
                  double *out)
{
    /* Unused for now -- the dispatch below inlines launches with
     * different kernel signatures.  Kept as a forward stub for
     * the unified reduction path we may consolidate to later. */
    (void)state; (void)launch; (void)out;
    return MOONLAB_CUDA_ERR_DRIVER;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_prob_z(const moonlab_cuda_state_t *state,
                    uint32_t target,
                    double *out_prob)
{
    if (!state || !out_prob) return MOONLAB_CUDA_ERR_NUM_QUBITS;
    if (target >= state->n_qubits) return MOONLAB_CUDA_ERR_INVALID_QUBIT;

    uint64_t n_pairs = state->dim >> 1;
    int  threads = 256;
    uint64_t blocks_u = (n_pairs + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;

    double *partials = NULL;
    CUDA_TRY(cudaMallocManaged(&partials, (size_t)blocks * sizeof(double)));
    cudaMemsetAsync(partials, 0, (size_t)blocks * sizeof(double), state->stream);

    k_prob_z<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, target, partials);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaStreamSynchronize(state->stream));

    double sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += partials[i];
    cudaFree(partials);
    *out_prob = sum;
    return MOONLAB_CUDA_OK;
}

extern "C"
moonlab_cuda_status_t
moonlab_cuda_norm(const moonlab_cuda_state_t *state, double *out_norm)
{
    if (!state || !out_norm) return MOONLAB_CUDA_ERR_NUM_QUBITS;

    int  threads = 256;
    uint64_t blocks_u = (state->dim + threads - 1) / threads;
    if (blocks_u > (uint64_t)2147483647ULL) blocks_u = 2147483647ULL;
    int blocks = (int)blocks_u;
    if (blocks < 1) blocks = 1;

    double *partials = NULL;
    CUDA_TRY(cudaMallocManaged(&partials, (size_t)blocks * sizeof(double)));
    cudaMemsetAsync(partials, 0, (size_t)blocks * sizeof(double), state->stream);

    k_norm<<<blocks, threads, 0, state->stream>>>(
        state->amps, state->dim, partials);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaStreamSynchronize(state->stream));

    double sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += partials[i];
    cudaFree(partials);
    *out_norm = sum;
    return MOONLAB_CUDA_OK;
}
