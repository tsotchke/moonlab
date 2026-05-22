/**
 * @file  state_gpu.cu
 * @brief CUDA-backed quantum_state_t lifecycle + host/device sync.
 *
 * Compiled only when QSIM_HAS_CUDA is on (gated in CMakeLists.txt).
 * Without CUDA, callers get QS_ERROR_NOT_SUPPORTED via a weak stub
 * in state.c (or a link-time failure -- both acceptable; the public
 * documentation says these functions require QSIM_HAS_CUDA).
 *
 * This file also provides the two routing entry points that
 * gates.c looks up via weak externs:
 *
 *   int qsim_gpu_route_1q(quantum_state_t *, int qubit, const double m[8]);
 *   int qsim_gpu_route_cnot(quantum_state_t *, int control, int target);
 *
 * They forward to moonlab_cuda_apply_1q / apply_cnot when the state
 * has a non-NULL gpu_state.
 */

#include "state.h"
#include "../backends/cuda_statevec.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

extern "C"
qs_error_t quantum_state_create_gpu(size_t num_qubits, quantum_state_t **out_state)
{
    if (!out_state) return QS_ERROR_INVALID_PARAM;
    if (num_qubits == 0 || num_qubits > 31) return QS_ERROR_INVALID_PARAM;

    quantum_state_t *s = (quantum_state_t *)calloc(1, sizeof(*s));
    if (!s) return QS_ERROR_OUT_OF_MEMORY;

    s->num_qubits = num_qubits;
    s->state_dim  = ((size_t)1) << num_qubits;
    /* Host amplitudes buffer -- present so callers that read
     * state->amplitudes after a sync_to_host() see something
     * sensible, and so state_destroy()'s owns_memory branch
     * doesn't trip on a NULL deref later. */
    s->amplitudes = (complex_t *)calloc(s->state_dim, sizeof(complex_t));
    if (!s->amplitudes) { free(s); return QS_ERROR_OUT_OF_MEMORY; }
    /* |0...0>: a[0] = 1+0i; the calloc above already zeroed everything,
     * so we just write the real-1 component as two contiguous doubles.
     * Using _Complex_I would force C99-mode <complex.h>, which doesn't
     * compile cleanly through nvcc's C++ frontend on cc<13.  Writing
     * the underlying doubles directly is portable across both. */
    ((double *)s->amplitudes)[0] = 1.0;
    ((double *)s->amplitudes)[1] = 0.0;
    s->owns_memory = 1;

    moonlab_cuda_state_t *gpu = NULL;
    moonlab_cuda_status_t rc = moonlab_cuda_state_create((uint32_t)num_qubits, &gpu);
    if (rc != MOONLAB_CUDA_OK) {
        free(s->amplitudes);
        free(s);
        /* Map cuda status onto qs_error_t.  Most likely cause is
         * cudaMallocManaged out-of-memory or no CUDA device. */
        return (rc == MOONLAB_CUDA_ERR_DRIVER)
            ? QS_ERROR_NOT_SUPPORTED
            : QS_ERROR_OUT_OF_MEMORY;
    }
    s->gpu_state   = (void *)gpu;
    s->gpu_backend = 1;  /* CUDA */
    *out_state = s;
    return QS_SUCCESS;
}

extern "C"
qs_error_t quantum_state_sync_to_host(quantum_state_t *state)
{
    if (!state) return QS_ERROR_INVALID_PARAM;
    if (!state->gpu_state) return QS_SUCCESS;  /* CPU state; nothing to do */

    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_synchronize(gpu);
    moonlab_cuda_status_t rc = moonlab_cuda_state_copy_to_host(
        gpu, (double *)state->amplitudes);
    if (rc != MOONLAB_CUDA_OK) return QS_ERROR_DRIVER;
    return QS_SUCCESS;
}

extern "C"
qs_error_t quantum_state_sync_from_host(quantum_state_t *state)
{
    if (!state) return QS_ERROR_INVALID_PARAM;
    if (!state->gpu_state) return QS_SUCCESS;

    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_state_copy_from_host(
        gpu, (const double *)state->amplitudes);
    if (rc != MOONLAB_CUDA_OK) return QS_ERROR_DRIVER;
    return QS_SUCCESS;
}

/* ---------- routing entry points used by gates.c ---------- */
/* These are looked up as weak externs from gates.c.  When CUDA
 * is compiled in, they resolve here; otherwise gates.c sees NULL
 * and stays on the CPU path. */

extern "C"
int qsim_gpu_route_1q(quantum_state_t *state, int qubit, const double m[8])
{
    if (!state || !state->gpu_state || !m) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_1q(gpu, (uint32_t)qubit, m);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}

extern "C"
int qsim_gpu_route_cnot(quantum_state_t *state, int control, int target)
{
    if (!state || !state->gpu_state) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_cnot(
        gpu, (uint32_t)control, (uint32_t)target);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}

extern "C"
int qsim_gpu_route_hadamard(quantum_state_t *state, int qubit)
{
    if (!state || !state->gpu_state) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_h(gpu, (uint32_t)qubit);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}

extern "C"
int qsim_gpu_route_pauli_x(quantum_state_t *state, int qubit)
{
    if (!state || !state->gpu_state) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_x(gpu, (uint32_t)qubit);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}

/* Generic single-qubit unitary: apply any 2x2 complex matrix.
 * Matrix layout matches the kernel -- 8 doubles, row-major,
 * with each complex entry stored as (real, imag) pair:
 *   m[0..1] = M[0][0]   m[2..3] = M[0][1]
 *   m[4..5] = M[1][0]   m[6..7] = M[1][1]
 *
 * All single-qubit gates that don't have a specialized fast path
 * (apply_h, apply_x) route through here. */
extern "C"
int qsim_gpu_route_apply_1q_matrix(quantum_state_t *state, int qubit, const double m[8])
{
    if (!state || !state->gpu_state || !m) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_1q(gpu, (uint32_t)qubit, m);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}

/* Generic two-qubit unitary: apply any 4x4 complex matrix on
 * (q0, q1).  32 doubles, row-major, basis |q1 q0>:
 *   m[0..7]   = row 0 (|00>)
 *   m[8..15]  = row 1 (|01>)
 *   m[16..23] = row 2 (|10>)
 *   m[24..31] = row 3 (|11>)
 *
 * CZ, SWAP, cphase, cy, crx, cry, crz all route through here. */
extern "C"
int qsim_gpu_route_apply_2q_matrix(quantum_state_t *state, int q0, int q1, const double m[32])
{
    if (!state || !state->gpu_state || !m) return -1;
    moonlab_cuda_state_t *gpu = (moonlab_cuda_state_t *)state->gpu_state;
    moonlab_cuda_status_t rc = moonlab_cuda_apply_2q(gpu, (uint32_t)q0, (uint32_t)q1, m);
    return rc == MOONLAB_CUDA_OK ? 0 : -1;
}
