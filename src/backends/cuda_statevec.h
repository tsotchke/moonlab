/**
 * @file  cuda_statevec.h
 * @brief CUDA state-vector backend (Jetson-tuned).
 *
 * Minimal moonlab CUDA backend.  Designed FIRST for Jetson SoC GPUs
 * (Volta/Ampere/Orin), which share LPDDR with the CPU through a
 * unified memory controller.  Allocations use cudaMallocManaged, so
 * the same pointer is valid from both host code and kernel code --
 * no explicit cudaMemcpy is needed for round-trip.
 *
 * On discrete PCIe GPUs the same kernels work but you pay a
 * cudaMemcpy on every host<->device transition.  That's still much
 * faster than a CPU kernel for sufficiently large N; the
 * cuda_tegra_probe.h kind value lets calling code decide whether to
 * keep the state vector resident on device or shuttle it.
 *
 * Build: this header is plain C-visible (extern "C").  The kernels
 * are in cuda_statevec.cu and only get compiled when QSIM_ENABLE_CUDA
 * is ON.
 */

#ifndef MOONLAB_CUDA_STATEVEC_H
#define MOONLAB_CUDA_STATEVEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle.  Owns:
 *   - the unified-memory state-vector buffer  (2^n_qubits complex doubles)
 *   - the dimension + qubit count
 *   - one CUDA stream
 *   - a host-side scratch staging buffer for gate matrices */
typedef struct moonlab_cuda_state moonlab_cuda_state_t;

typedef enum {
    MOONLAB_CUDA_OK                 = 0,
    MOONLAB_CUDA_ERR_OUT_OF_MEMORY  = -1,
    MOONLAB_CUDA_ERR_NO_DEVICE      = -2,
    MOONLAB_CUDA_ERR_DRIVER         = -3,
    MOONLAB_CUDA_ERR_INVALID_QUBIT  = -4,
    MOONLAB_CUDA_ERR_NUM_QUBITS     = -5,
} moonlab_cuda_status_t;

/**
 * @brief Allocate a state vector of dimension 2^n_qubits, init to
 *        |0...0> (probability 1 in component 0).
 *
 *        n_qubits in [1, 31].  At 31 the state vector is
 *        2^31 complex doubles = 32 GiB -- larger than Jetson AGX
 *        Xavier's RAM, so practical Jetson limit is ~28.  On 64 GB+
 *        hosts we can go to 31; beyond that needs MPI sharding
 *        (separate path).
 *
 * @param[out] out_state  Set to a new opaque state on success.
 * @return MOONLAB_CUDA_OK on success.
 */
moonlab_cuda_status_t
moonlab_cuda_state_create(uint32_t n_qubits, moonlab_cuda_state_t **out_state);

/**
 * @brief Free a state vector.  Safe on NULL.
 */
void moonlab_cuda_state_free(moonlab_cuda_state_t *state);

/**
 * @brief Number of qubits the state holds.
 */
uint32_t moonlab_cuda_state_n_qubits(const moonlab_cuda_state_t *state);

/**
 * @brief Dimension of the state vector (2^n_qubits).
 */
uint64_t moonlab_cuda_state_dim(const moonlab_cuda_state_t *state);

/**
 * @brief Apply a 2x2 unitary to qubit ``target``.
 *        Matrix layout: [m00, m01, m10, m11] in row-major,
 *        each a complex pair {re, im}.  Eight doubles total.
 *
 *        Equivalent CPU operation:
 *            for each pair (i0, i1) of state indices differing
 *            only in bit `target`:
 *              [a0', a1']^T = M * [a0, a1]^T
 */
moonlab_cuda_status_t
moonlab_cuda_apply_1q(moonlab_cuda_state_t *state,
                      uint32_t target,
                      const double m[8]);

/**
 * @brief Apply a CNOT gate (control, target).
 *        Operationally: swap a_{...,1,...,0,...} with a_{...,1,...,1,...}
 *        for each combination of the other qubits.
 */
moonlab_cuda_status_t
moonlab_cuda_apply_cnot(moonlab_cuda_state_t *state,
                        uint32_t control,
                        uint32_t target);

/**
 * @brief Apply an arbitrary 4x4 unitary to qubits (q0, q1).
 *        ``m`` is 32 doubles: row-major 16 complex entries in
 *        {re, im} pairs.  Basis ordering is |q1 q0> with q0 the
 *        low bit -- matching CNOT(control=q1, target=q0) when
 *        the 4x4 is the standard CNOT matrix.
 *
 *        Kernel structure: one thread per quadruple
 *        (i00, i01, i10, i11) -- the four state indices that
 *        share all other-qubit values and differ only in (q0, q1).
 *        Each thread does a 4x4 complex matvec.
 */
moonlab_cuda_status_t
moonlab_cuda_apply_2q(moonlab_cuda_state_t *state,
                      uint32_t q0,
                      uint32_t q1,
                      const double m[32]);

/**
 * @brief Probability of measuring qubit `target` as |1>.
 *        Sums |amps[k]|^2 over all k with bit `target` set.
 *
 *        Kernel: parallel reduction on the GPU, single double
 *        scalar returned to host.
 */
moonlab_cuda_status_t
moonlab_cuda_prob_z(const moonlab_cuda_state_t *state,
                    uint32_t target,
                    double *out_prob);

/**
 * @brief Total state-vector squared norm.  For an ideal unitary
 *        circuit this stays at 1.0; useful as a quick health
 *        check on numerical drift through long circuits.
 */
moonlab_cuda_status_t
moonlab_cuda_norm(const moonlab_cuda_state_t *state,
                  double *out_norm);

/**
 * @brief Convenience: apply Pauli-X (NOT) to a qubit.  Equivalent
 *        to apply_1q with [[0,1],[1,0]] but avoids the matmul.
 */
moonlab_cuda_status_t
moonlab_cuda_apply_x(moonlab_cuda_state_t *state, uint32_t target);

/**
 * @brief Convenience: apply Hadamard to a qubit.  Same as
 *        apply_1q with (1/sqrt(2)) [[1,1],[1,-1]].
 */
moonlab_cuda_status_t
moonlab_cuda_apply_h(moonlab_cuda_state_t *state, uint32_t target);

/**
 * @brief Copy amplitudes out to caller-provided host buffer.
 *        ``out`` must hold dim * 2 doubles (real, imag interleaved).
 *
 *        On Tegra this is essentially a memcpy from unified memory.
 *        On discrete it's a cudaMemcpy(device -> host).
 */
moonlab_cuda_status_t
moonlab_cuda_state_copy_to_host(const moonlab_cuda_state_t *state,
                                double *out);

/**
 * @brief Synchronize the device stream.  Useful for benchmarks
 *        and for ensuring the result of asynchronous gate calls
 *        has settled before reading the state.
 */
moonlab_cuda_status_t
moonlab_cuda_synchronize(moonlab_cuda_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CUDA_STATEVEC_H */
