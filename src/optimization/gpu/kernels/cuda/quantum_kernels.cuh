/**
 * @file quantum_kernels.cuh
 * @brief CUDA kernel definitions for quantum computing operations
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef QUANTUM_KERNELS_CUH
#define QUANTUM_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

// ============================================================================
// Complex number type (double precision)
// ============================================================================

typedef cuDoubleComplex complex_t;

// Complex arithmetic helpers (device functions)
__device__ __forceinline__ complex_t cmul(complex_t a, complex_t b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

__device__ __forceinline__ complex_t cadd(complex_t a, complex_t b) {
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

__device__ __forceinline__ complex_t csub(complex_t a, complex_t b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

__device__ __forceinline__ complex_t cscale(complex_t z, double s) {
    return make_cuDoubleComplex(cuCreal(z) * s, cuCimag(z) * s);
}

__device__ __forceinline__ complex_t cneg(complex_t z) {
    return make_cuDoubleComplex(-cuCreal(z), -cuCimag(z));
}

__device__ __forceinline__ double cabs2(complex_t z) {
    return cuCreal(z) * cuCreal(z) + cuCimag(z) * cuCimag(z);
}

__device__ __forceinline__ complex_t cpolar(double r, double theta) {
    return make_cuDoubleComplex(r * cos(theta), r * sin(theta));
}

// ============================================================================
// QUANTUM GATE KERNELS
// ============================================================================

/**
 * @brief Hadamard transform on single qubit
 */
__global__ void hadamard_transform_kernel(
    complex_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Hadamard on all qubits (Walsh-Hadamard transform)
 */
__global__ void hadamard_all_kernel(
    complex_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Pauli-X gate (bit flip)
 */
__global__ void pauli_x_kernel(
    complex_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Pauli-Y gate
 */
__global__ void pauli_y_kernel(
    complex_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Pauli-Z gate (phase flip)
 */
__global__ void pauli_z_kernel(
    complex_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Phase rotation gate
 */
__global__ void phase_gate_kernel(
    complex_t* amplitudes,
    uint32_t qubit_index,
    double phase,
    uint64_t state_dim
);

/**
 * @brief CNOT (controlled-NOT) gate
 */
__global__ void cnot_kernel(
    complex_t* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
);

// ============================================================================
// GROVER'S ALGORITHM KERNELS
// ============================================================================

/**
 * @brief Oracle - single target phase flip
 */
__global__ void oracle_single_target_kernel(
    complex_t* amplitudes,
    uint64_t target
);

/**
 * @brief Sparse oracle - multiple targets
 */
__global__ void sparse_oracle_kernel(
    complex_t* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
);

/**
 * @brief Diffusion sum phase (parallel reduction)
 */
__global__ void diffusion_sum_kernel(
    const complex_t* amplitudes,
    complex_t* partial_sums,
    uint64_t state_dim
);

/**
 * @brief Diffusion apply phase
 */
__global__ void diffusion_apply_kernel(
    complex_t* amplitudes,
    const complex_t* avg_buffer,
    uint64_t state_dim
);

/**
 * @brief Fused Grover diffusion (small state dims)
 */
__global__ void grover_diffusion_fused_kernel(
    complex_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Batch Grover search kernel
 */
__global__ void grover_batch_search_kernel(
    complex_t* batch_states,
    const uint64_t* targets,
    uint64_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
);

// ============================================================================
// MEASUREMENT & UTILITY KERNELS
// ============================================================================

/**
 * @brief Compute probability distribution
 */
__global__ void compute_probabilities_kernel(
    const complex_t* amplitudes,
    double* probabilities,
    uint64_t state_dim
);

/**
 * @brief Normalize quantum state
 */
__global__ void normalize_state_kernel(
    complex_t* amplitudes,
    double inv_norm,
    uint64_t state_dim
);

/**
 * @brief Parallel reduction for sum of squared magnitudes
 */
__global__ void sum_squared_magnitudes_kernel(
    const complex_t* amplitudes,
    double* partial_sums,
    uint64_t state_dim
);

#endif /* QUANTUM_KERNELS_CUH */
