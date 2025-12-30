/**
 * @file stride_gates.h
 * @brief Stride-based optimized quantum gate operations
 *
 * Provides 4-6x speedup over bit-extraction implementations by using
 * predictable memory access patterns that eliminate branch misprediction.
 *
 * Key Insight:
 * The quantum state vector has a specific structure based on qubit positions.
 * For n qubits, the state has 2^n amplitudes. For a gate on qubit q:
 * - Amplitudes are grouped in blocks of size 2^(q+1)
 * - Within each block, first half has qubit q = 0, second half has q = 1
 * - The "stride" between paired amplitudes is 2^q
 *
 * Example (3 qubits, qubit 1):
 * State: [|000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩]
 *         [  0  ,   1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ]
 *
 * For qubit 1 (stride = 2^1 = 2):
 * - Pairs: (0,2), (1,3), (4,6), (5,7)
 * - Block size: 4
 *
 * For two-qubit gates (control c, target t):
 * - Process amplitudes where control = 1
 * - Apply gate to target within those blocks
 *
 * Performance:
 * - No bit extraction in inner loop
 * - Predictable stride pattern (no branch misprediction)
 * - Cache-friendly sequential access within blocks
 * - SIMD-friendly (can process multiple pairs)
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef STRIDE_GATES_H
#define STRIDE_GATES_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Complex type
typedef double _Complex complex_t;

// ============================================================================
// SINGLE-QUBIT GATE OPERATIONS (Stride-Based)
// ============================================================================

/**
 * @brief Apply Hadamard gate using stride-based iteration
 *
 * H = 1/√2 * [[1, 1], [1, -1]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index (0-indexed from LSB)
 */
void stride_hadamard(complex_t* amplitudes, int num_qubits, int target);

/**
 * @brief Apply Pauli-X gate using stride-based iteration
 *
 * X = [[0, 1], [1, 0]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 */
void stride_pauli_x(complex_t* amplitudes, int num_qubits, int target);

/**
 * @brief Apply Pauli-Y gate using stride-based iteration
 *
 * Y = [[0, -i], [i, 0]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 */
void stride_pauli_y(complex_t* amplitudes, int num_qubits, int target);

/**
 * @brief Apply Pauli-Z gate using stride-based iteration
 *
 * Z = [[1, 0], [0, -1]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 */
void stride_pauli_z(complex_t* amplitudes, int num_qubits, int target);

/**
 * @brief Apply Phase gate using stride-based iteration
 *
 * P(θ) = [[1, 0], [0, e^(iθ)]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 * @param theta      Phase angle in radians
 */
void stride_phase(complex_t* amplitudes, int num_qubits, int target, double theta);

/**
 * @brief Apply RX rotation gate using stride-based iteration
 *
 * RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_rx(complex_t* amplitudes, int num_qubits, int target, double theta);

/**
 * @brief Apply RY rotation gate using stride-based iteration
 *
 * RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_ry(complex_t* amplitudes, int num_qubits, int target, double theta);

/**
 * @brief Apply RZ rotation gate using stride-based iteration
 *
 * RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_rz(complex_t* amplitudes, int num_qubits, int target, double theta);

/**
 * @brief Apply arbitrary 2x2 unitary gate using stride-based iteration
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 * @param gate       2x2 complex matrix (row-major: [u00, u01, u10, u11])
 */
void stride_apply_gate(complex_t* amplitudes, int num_qubits, int target,
                       const complex_t gate[4]);

// ============================================================================
// TWO-QUBIT GATE OPERATIONS (Stride-Based)
// ============================================================================

/**
 * @brief Apply CNOT gate using stride-based iteration
 *
 * CNOT: Flips target when control is |1⟩
 *
 * Performance: 4-6x faster than bit-extraction method
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 */
void stride_cnot(complex_t* amplitudes, int num_qubits, int control, int target);

/**
 * @brief Apply CZ gate using stride-based iteration
 *
 * CZ: Applies phase -1 when both qubits are |1⟩
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 */
void stride_cz(complex_t* amplitudes, int num_qubits, int control, int target);

/**
 * @brief Apply CY gate using stride-based iteration
 *
 * CY: Applies Y gate to target when control is |1⟩
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 */
void stride_cy(complex_t* amplitudes, int num_qubits, int control, int target);

/**
 * @brief Apply Controlled-Phase gate using stride-based iteration
 *
 * CP(θ): Applies phase e^(iθ) when both qubits are |1⟩
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 * @param theta      Phase angle in radians
 */
void stride_cphase(complex_t* amplitudes, int num_qubits, int control, int target,
                   double theta);

/**
 * @brief Apply Controlled-RX gate using stride-based iteration
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_crx(complex_t* amplitudes, int num_qubits, int control, int target,
                double theta);

/**
 * @brief Apply Controlled-RY gate using stride-based iteration
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_cry(complex_t* amplitudes, int num_qubits, int control, int target,
                double theta);

/**
 * @brief Apply Controlled-RZ gate using stride-based iteration
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 * @param theta      Rotation angle in radians
 */
void stride_crz(complex_t* amplitudes, int num_qubits, int control, int target,
                double theta);

/**
 * @brief Apply SWAP gate using stride-based iteration
 *
 * SWAP: Exchanges states of two qubits
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param qubit1     First qubit index
 * @param qubit2     Second qubit index
 */
void stride_swap(complex_t* amplitudes, int num_qubits, int qubit1, int qubit2);

/**
 * @brief Apply iSWAP gate using stride-based iteration
 *
 * iSWAP: SWAP with i phase factors
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param qubit1     First qubit index
 * @param qubit2     Second qubit index
 */
void stride_iswap(complex_t* amplitudes, int num_qubits, int qubit1, int qubit2);

/**
 * @brief Apply arbitrary controlled-U gate using stride-based iteration
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 * @param gate       2x2 complex matrix for target gate
 */
void stride_controlled_gate(complex_t* amplitudes, int num_qubits,
                            int control, int target, const complex_t gate[4]);

// ============================================================================
// THREE-QUBIT GATE OPERATIONS (Stride-Based)
// ============================================================================

/**
 * @brief Apply Toffoli (CCNOT) gate using stride-based iteration
 *
 * Toffoli: Flips target when both controls are |1⟩
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control1   First control qubit index
 * @param control2   Second control qubit index
 * @param target     Target qubit index
 */
void stride_toffoli(complex_t* amplitudes, int num_qubits,
                    int control1, int control2, int target);

/**
 * @brief Apply Fredkin (CSWAP) gate using stride-based iteration
 *
 * Fredkin: Swaps target1 and target2 when control is |1⟩
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target1    First swap target
 * @param target2    Second swap target
 */
void stride_fredkin(complex_t* amplitudes, int num_qubits,
                    int control, int target1, int target2);

// ============================================================================
// PARALLEL VERSIONS (OpenMP)
// ============================================================================

#ifdef _OPENMP

/**
 * @brief Parallel CNOT gate for large state vectors
 *
 * Uses OpenMP for parallelization when state_dim > threshold.
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param control    Control qubit index
 * @param target     Target qubit index
 */
void stride_cnot_parallel(complex_t* amplitudes, int num_qubits,
                          int control, int target);

/**
 * @brief Parallel Hadamard gate
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param target     Target qubit index
 */
void stride_hadamard_parallel(complex_t* amplitudes, int num_qubits, int target);

#endif /* _OPENMP */

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/**
 * @brief Apply same single-qubit gate to multiple qubits
 *
 * More efficient than calling individual gates when applying
 * the same gate to multiple qubits (e.g., H on all qubits).
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 * @param targets    Array of target qubit indices
 * @param num_targets Number of target qubits
 * @param gate       2x2 complex matrix
 */
void stride_batch_single_gate(complex_t* amplitudes, int num_qubits,
                              const int* targets, int num_targets,
                              const complex_t gate[4]);

/**
 * @brief Apply Hadamard to all qubits (efficient tensor product)
 *
 * H⊗n: Applies Hadamard to every qubit
 *
 * @param amplitudes State amplitude array
 * @param num_qubits Number of qubits in state
 */
void stride_hadamard_all(complex_t* amplitudes, int num_qubits);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Get stride for a qubit position
 *
 * @param qubit Qubit index (0-indexed from LSB)
 * @return Stride value (2^qubit)
 */
static inline uint64_t get_stride(int qubit) {
    return 1ULL << qubit;
}

/**
 * @brief Get block size for a qubit position
 *
 * @param qubit Qubit index
 * @return Block size (2^(qubit+1))
 */
static inline uint64_t get_block_size(int qubit) {
    return 1ULL << (qubit + 1);
}

/**
 * @brief Benchmark stride-based vs bit-extraction methods
 *
 * Runs both implementations and reports speedup.
 *
 * @param num_qubits Number of qubits for benchmark
 * @param num_gates  Number of gate applications
 * @return Speedup factor (stride/bit-extract time ratio)
 */
double stride_benchmark_speedup(int num_qubits, int num_gates);

#ifdef __cplusplus
}
#endif

#endif /* STRIDE_GATES_H */
