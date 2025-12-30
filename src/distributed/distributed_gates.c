/**
 * @file distributed_gates.c
 * @brief Distributed quantum gate implementation
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "distributed_gates.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// ============================================================================
// STANDARD GATE MATRICES
// ============================================================================

const gate_matrix_2x2_t GATE_H = {{
    1.0/M_SQRT2 + 0.0*I,  1.0/M_SQRT2 + 0.0*I,
    1.0/M_SQRT2 + 0.0*I, -1.0/M_SQRT2 + 0.0*I
}};

const gate_matrix_2x2_t GATE_X = {{
    0.0 + 0.0*I, 1.0 + 0.0*I,
    1.0 + 0.0*I, 0.0 + 0.0*I
}};

const gate_matrix_2x2_t GATE_Y = {{
    0.0 + 0.0*I, 0.0 - 1.0*I,
    0.0 + 1.0*I, 0.0 + 0.0*I
}};

const gate_matrix_2x2_t GATE_Z = {{
    1.0 + 0.0*I,  0.0 + 0.0*I,
    0.0 + 0.0*I, -1.0 + 0.0*I
}};

const gate_matrix_2x2_t GATE_S = {{
    1.0 + 0.0*I, 0.0 + 0.0*I,
    0.0 + 0.0*I, 0.0 + 1.0*I
}};

const gate_matrix_2x2_t GATE_T = {{
    1.0 + 0.0*I, 0.0 + 0.0*I,
    0.0 + 0.0*I, M_SQRT2/2.0 + M_SQRT2/2.0*I
}};

const gate_matrix_2x2_t GATE_SDAG = {{
    1.0 + 0.0*I, 0.0 + 0.0*I,
    0.0 + 0.0*I, 0.0 - 1.0*I
}};

const gate_matrix_2x2_t GATE_TDAG = {{
    1.0 + 0.0*I, 0.0 + 0.0*I,
    0.0 + 0.0*I, M_SQRT2/2.0 - M_SQRT2/2.0*I
}};

// CNOT in computational basis |00⟩, |01⟩, |10⟩, |11⟩
const gate_matrix_4x4_t GATE_CNOT = {{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0
}};

const gate_matrix_4x4_t GATE_CZ = {{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, -1
}};

const gate_matrix_4x4_t GATE_SWAP = {{
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1
}};

const gate_matrix_4x4_t GATE_ISWAP = {{
    1, 0, 0, 0,
    0, 0, I, 0,
    0, I, 0, 0,
    0, 0, 0, 1
}};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Apply 2x2 matrix to amplitude pair
 */
static inline void apply_matrix_2x2(const gate_matrix_2x2_t* m,
                                    double complex* a0,
                                    double complex* a1) {
    double complex new_a0 = m->m[0] * (*a0) + m->m[1] * (*a1);
    double complex new_a1 = m->m[2] * (*a0) + m->m[3] * (*a1);
    *a0 = new_a0;
    *a1 = new_a1;
}

/**
 * @brief Check bit value at position
 */
static inline int get_bit(uint64_t value, uint32_t bit) {
    return (value >> bit) & 1;
}

/**
 * @brief Set bit at position
 */
static inline uint64_t set_bit(uint64_t value, uint32_t bit) {
    return value | (1ULL << bit);
}

/**
 * @brief Clear bit at position
 */
static inline uint64_t clear_bit(uint64_t value, uint32_t bit) {
    return value & ~(1ULL << bit);
}

/**
 * @brief Flip bit at position
 */
static inline uint64_t flip_bit(uint64_t value, uint32_t bit) {
    return value ^ (1ULL << bit);
}

// ============================================================================
// LOCAL SINGLE-QUBIT GATE (NO COMMUNICATION)
// ============================================================================

/**
 * @brief Apply single-qubit gate to local qubit
 */
static dist_gate_error_t apply_local_1q(partitioned_state_t* state,
                                        uint32_t target,
                                        const gate_matrix_2x2_t* matrix) {
    uint64_t stride = 1ULL << target;

    // Iterate over pairs of amplitudes differing in target bit
    for (uint64_t i = 0; i < state->local_count; i += 2 * stride) {
        for (uint64_t j = 0; j < stride; j++) {
            uint64_t idx0 = i + j;           // target bit = 0
            uint64_t idx1 = i + j + stride;  // target bit = 1

            if (idx1 < state->local_count) {
                apply_matrix_2x2(matrix,
                                &state->amplitudes[idx0],
                                &state->amplitudes[idx1]);
            }
        }
    }

    return DIST_GATE_SUCCESS;
}

// ============================================================================
// REMOTE SINGLE-QUBIT GATE (WITH COMMUNICATION)
// ============================================================================

/**
 * @brief Apply single-qubit gate to partition qubit (requires MPI exchange)
 */
static dist_gate_error_t apply_remote_1q(partitioned_state_t* state,
                                         uint32_t target,
                                         const gate_matrix_2x2_t* matrix) {
    exchange_descriptor_t desc;
    partition_error_t err = partition_plan_1q_exchange(state, target, &desc);
    if (err != PARTITION_SUCCESS) {
        return DIST_GATE_ERROR_COMM;
    }

    if (!desc.requires_exchange) {
        partition_free_exchange_desc(&desc);
        return apply_local_1q(state, target, matrix);
    }

    int rank = mpi_get_rank(state->dist_ctx);

    // Determine if we hold the |0⟩ or |1⟩ states for this partition bit
    uint32_t partition_bit = target - state->local_qubits;
    int holds_zero_states = (rank & (1 << partition_bit)) == 0;

    // Exchange all local amplitudes with partner
    mpi_bridge_error_t mpi_err = mpi_exchange_amplitudes(
        state->dist_ctx,
        state->amplitudes,
        state->recv_buffer,
        state->local_count,
        desc.partner_rank,
        0
    );

    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        partition_free_exchange_desc(&desc);
        return DIST_GATE_ERROR_COMM;
    }

    // Apply gate: we have one half, partner has the other
    // If we hold |0⟩ states: a0 = local, a1 = received
    // If we hold |1⟩ states: a0 = received, a1 = local
    for (uint64_t i = 0; i < state->local_count; i++) {
        double complex a0, a1;
        if (holds_zero_states) {
            a0 = state->amplitudes[i];
            a1 = state->recv_buffer[i];
        } else {
            a0 = state->recv_buffer[i];
            a1 = state->amplitudes[i];
        }

        double complex new_a0 = matrix->m[0] * a0 + matrix->m[1] * a1;
        double complex new_a1 = matrix->m[2] * a0 + matrix->m[3] * a1;

        if (holds_zero_states) {
            state->amplitudes[i] = new_a0;
            state->send_buffer[i] = new_a1;  // Will send back to partner
        } else {
            state->amplitudes[i] = new_a1;
            state->send_buffer[i] = new_a0;  // Will send back to partner
        }
    }

    // Exchange back updated amplitudes
    mpi_err = mpi_exchange_amplitudes(
        state->dist_ctx,
        state->send_buffer,
        state->recv_buffer,
        state->local_count,
        desc.partner_rank,
        1
    );

    partition_free_exchange_desc(&desc);

    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return DIST_GATE_ERROR_COMM;
    }

    return DIST_GATE_SUCCESS;
}

// ============================================================================
// SINGLE-QUBIT GATES
// ============================================================================

dist_gate_error_t dist_gate_1q(partitioned_state_t* state,
                               uint32_t target,
                               const gate_matrix_2x2_t* matrix) {
    if (!state || !matrix) {
        return DIST_GATE_ERROR_NOT_INITIALIZED;
    }

    if (target >= state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    if (partition_is_partition_qubit(state, target)) {
        return apply_remote_1q(state, target, matrix);
    } else {
        return apply_local_1q(state, target, matrix);
    }
}

dist_gate_error_t dist_hadamard(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_H);
}

dist_gate_error_t dist_hadamard_all(partitioned_state_t* state) {
    if (!state) {
        return DIST_GATE_ERROR_NOT_INITIALIZED;
    }

    // Apply to local qubits first (no communication)
    for (uint32_t q = 0; q < state->local_qubits; q++) {
        dist_gate_error_t err = apply_local_1q(state, q, &GATE_H);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    // Then partition qubits (with communication)
    for (uint32_t q = state->local_qubits; q < state->num_qubits; q++) {
        dist_gate_error_t err = apply_remote_1q(state, q, &GATE_H);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_pauli_x(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_X);
}

dist_gate_error_t dist_pauli_y(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_Y);
}

dist_gate_error_t dist_pauli_z(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_Z);
}

dist_gate_error_t dist_rz(partitioned_state_t* state, uint32_t target, double theta) {
    gate_matrix_2x2_t rz;
    dist_build_rz_matrix(theta, &rz);
    return dist_gate_1q(state, target, &rz);
}

dist_gate_error_t dist_rx(partitioned_state_t* state, uint32_t target, double theta) {
    gate_matrix_2x2_t rx;
    dist_build_rx_matrix(theta, &rx);
    return dist_gate_1q(state, target, &rx);
}

dist_gate_error_t dist_ry(partitioned_state_t* state, uint32_t target, double theta) {
    gate_matrix_2x2_t ry;
    dist_build_ry_matrix(theta, &ry);
    return dist_gate_1q(state, target, &ry);
}

dist_gate_error_t dist_phase(partitioned_state_t* state, uint32_t target, double phi) {
    gate_matrix_2x2_t p;
    dist_build_phase_matrix(phi, &p);
    return dist_gate_1q(state, target, &p);
}

dist_gate_error_t dist_s_gate(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_S);
}

dist_gate_error_t dist_t_gate(partitioned_state_t* state, uint32_t target) {
    return dist_gate_1q(state, target, &GATE_T);
}

// ============================================================================
// TWO-QUBIT GATES
// ============================================================================

/**
 * @brief Apply local two-qubit gate (both qubits local)
 */
static dist_gate_error_t apply_local_2q(partitioned_state_t* state,
                                        uint32_t q1, uint32_t q2,
                                        const gate_matrix_4x4_t* matrix) {
    if (q1 > q2) {
        uint32_t tmp = q1; q1 = q2; q2 = tmp;
    }

    uint64_t stride1 = 1ULL << q1;
    uint64_t stride2 = 1ULL << q2;

    // Iterate over groups of 4 amplitudes
    for (uint64_t i = 0; i < state->local_count; i++) {
        // Skip if not the first index in a group
        if ((i & stride1) || (i & stride2)) continue;

        uint64_t i00 = i;
        uint64_t i01 = i | stride1;
        uint64_t i10 = i | stride2;
        uint64_t i11 = i | stride1 | stride2;

        if (i11 >= state->local_count) continue;

        double complex a00 = state->amplitudes[i00];
        double complex a01 = state->amplitudes[i01];
        double complex a10 = state->amplitudes[i10];
        double complex a11 = state->amplitudes[i11];

        state->amplitudes[i00] = matrix->m[0]*a00 + matrix->m[1]*a01 +
                                  matrix->m[2]*a10 + matrix->m[3]*a11;
        state->amplitudes[i01] = matrix->m[4]*a00 + matrix->m[5]*a01 +
                                  matrix->m[6]*a10 + matrix->m[7]*a11;
        state->amplitudes[i10] = matrix->m[8]*a00 + matrix->m[9]*a01 +
                                  matrix->m[10]*a10 + matrix->m[11]*a11;
        state->amplitudes[i11] = matrix->m[12]*a00 + matrix->m[13]*a01 +
                                  matrix->m[14]*a10 + matrix->m[15]*a11;
    }

    return DIST_GATE_SUCCESS;
}

/**
 * @brief Apply remote two-qubit gate when exactly one qubit is a partition qubit
 *
 * This handles the case where one qubit is local and one is a partition qubit.
 * We exchange amplitudes with the partner rank that has the complementary
 * partition bit, apply the gate, and exchange back.
 */
static dist_gate_error_t apply_remote_2q_single_partition(
    partitioned_state_t* state,
    uint32_t local_qubit,
    uint32_t partition_qubit,
    const gate_matrix_4x4_t* matrix,
    int local_is_qubit1) {

    int rank = mpi_get_rank(state->dist_ctx);
    uint32_t partition_bit = partition_qubit - state->local_qubits;
    int partner_rank = rank ^ (1 << partition_bit);

    // Determine if we hold states with partition_qubit = 0 or 1
    int holds_pq_zero = (rank & (1 << partition_bit)) == 0;

    uint64_t local_stride = 1ULL << local_qubit;

    // Count pairs for exchange (half of local amplitudes participate)
    uint64_t exchange_count = state->local_count / 2;

    // Pack amplitudes to exchange (those where local qubit differs)
    // We need to exchange in pairs: one with local_qubit=0, one with local_qubit=1
    uint64_t pack_idx = 0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        if (!(i & local_stride)) {  // Only process where local_qubit = 0
            state->send_buffer[pack_idx++] = state->amplitudes[i];
            state->send_buffer[pack_idx++] = state->amplitudes[i | local_stride];
        }
    }

    // Exchange with partner
    mpi_bridge_error_t err = mpi_exchange_amplitudes(
        state->dist_ctx,
        state->send_buffer,
        state->recv_buffer,
        exchange_count * 2,  // Exchanging pairs
        partner_rank,
        0
    );

    if (err != MPI_BRIDGE_SUCCESS) {
        return DIST_GATE_ERROR_COMM;
    }

    // Apply the 4x4 matrix to each set of 4 amplitudes
    // a[0] = local with local_q=0, partition_q from our rank
    // a[1] = local with local_q=1, partition_q from our rank
    // a[2] = received with local_q=0, partition_q from partner
    // a[3] = received with local_q=1, partition_q from partner

    // The ordering depends on which qubit is which and partition bit values
    pack_idx = 0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        if (!(i & local_stride)) {
            double complex a00, a01, a10, a11;
            double complex new_a00, new_a01, new_a10, new_a11;

            double complex local_0 = state->amplitudes[i];
            double complex local_1 = state->amplitudes[i | local_stride];
            double complex remote_0 = state->recv_buffer[pack_idx];
            double complex remote_1 = state->recv_buffer[pack_idx + 1];

            // Arrange amplitudes based on qubit ordering and partition bit ownership
            // If local_is_qubit1: qubit1 is local, qubit2 is partition
            // Basis order: |q1 q2⟩ = |00⟩, |01⟩, |10⟩, |11⟩
            if (local_is_qubit1) {
                if (holds_pq_zero) {
                    // We have partition_qubit=0: local has |x0⟩
                    a00 = local_0;   // |00⟩ - local_q=0, partition_q=0 (local)
                    a01 = remote_0;  // |01⟩ - local_q=0, partition_q=1 (remote)
                    a10 = local_1;   // |10⟩ - local_q=1, partition_q=0 (local)
                    a11 = remote_1;  // |11⟩ - local_q=1, partition_q=1 (remote)
                } else {
                    // We have partition_qubit=1: local has |x1⟩
                    a00 = remote_0;  // |00⟩ - local_q=0, partition_q=0 (remote)
                    a01 = local_0;   // |01⟩ - local_q=0, partition_q=1 (local)
                    a10 = remote_1;  // |10⟩ - local_q=1, partition_q=0 (remote)
                    a11 = local_1;   // |11⟩ - local_q=1, partition_q=1 (local)
                }
            } else {
                // qubit1 is partition, qubit2 is local
                // Basis order: |q1 q2⟩ where q1=partition, q2=local
                if (holds_pq_zero) {
                    // We have partition_qubit=0: local has |0x⟩
                    a00 = local_0;   // |00⟩
                    a01 = local_1;   // |01⟩
                    a10 = remote_0;  // |10⟩
                    a11 = remote_1;  // |11⟩
                } else {
                    // We have partition_qubit=1: local has |1x⟩
                    a00 = remote_0;  // |00⟩
                    a01 = remote_1;  // |01⟩
                    a10 = local_0;   // |10⟩
                    a11 = local_1;   // |11⟩
                }
            }

            // Apply 4x4 matrix
            new_a00 = matrix->m[0]*a00 + matrix->m[1]*a01 + matrix->m[2]*a10 + matrix->m[3]*a11;
            new_a01 = matrix->m[4]*a00 + matrix->m[5]*a01 + matrix->m[6]*a10 + matrix->m[7]*a11;
            new_a10 = matrix->m[8]*a00 + matrix->m[9]*a01 + matrix->m[10]*a10 + matrix->m[11]*a11;
            new_a11 = matrix->m[12]*a00 + matrix->m[13]*a01 + matrix->m[14]*a10 + matrix->m[15]*a11;

            // Unpack back to local and send_buffer (for partner)
            if (local_is_qubit1) {
                if (holds_pq_zero) {
                    state->amplitudes[i] = new_a00;
                    state->send_buffer[pack_idx] = new_a01;  // Send to partner
                    state->amplitudes[i | local_stride] = new_a10;
                    state->send_buffer[pack_idx + 1] = new_a11;
                } else {
                    state->send_buffer[pack_idx] = new_a00;
                    state->amplitudes[i] = new_a01;
                    state->send_buffer[pack_idx + 1] = new_a10;
                    state->amplitudes[i | local_stride] = new_a11;
                }
            } else {
                if (holds_pq_zero) {
                    state->amplitudes[i] = new_a00;
                    state->amplitudes[i | local_stride] = new_a01;
                    state->send_buffer[pack_idx] = new_a10;
                    state->send_buffer[pack_idx + 1] = new_a11;
                } else {
                    state->send_buffer[pack_idx] = new_a00;
                    state->send_buffer[pack_idx + 1] = new_a01;
                    state->amplitudes[i] = new_a10;
                    state->amplitudes[i | local_stride] = new_a11;
                }
            }

            pack_idx += 2;
        }
    }

    // Exchange updated remote values back to partner
    err = mpi_exchange_amplitudes(
        state->dist_ctx,
        state->send_buffer,
        state->recv_buffer,
        exchange_count * 2,
        partner_rank,
        1
    );

    if (err != MPI_BRIDGE_SUCCESS) {
        return DIST_GATE_ERROR_COMM;
    }

    return DIST_GATE_SUCCESS;
}

/**
 * @brief Apply remote two-qubit gate when both qubits are partition qubits
 *
 * This requires communication with up to 3 other ranks to gather all 4 amplitudes.
 */
static dist_gate_error_t apply_remote_2q_both_partition(
    partitioned_state_t* state,
    uint32_t qubit1,
    uint32_t qubit2,
    const gate_matrix_4x4_t* matrix) {

    int rank = mpi_get_rank(state->dist_ctx);
    uint32_t pb1 = qubit1 - state->local_qubits;
    uint32_t pb2 = qubit2 - state->local_qubits;

    // Calculate the 4 ranks that own the 4 basis states
    // rank_xy = rank that owns states with qubit1=x, qubit2=y
    int base_rank = rank & ~((1 << pb1) | (1 << pb2));
    int rank_00 = base_rank;
    int rank_01 = base_rank | (1 << pb2);
    int rank_10 = base_rank | (1 << pb1);
    int rank_11 = base_rank | (1 << pb1) | (1 << pb2);

    // Each rank has 1/4 of the amplitudes needed for each group of 4
    // We process all local amplitudes, gathering/scattering with partners

    // For each local amplitude, we need to coordinate with 3 other ranks
    // to gather the 4-tuple, apply the gate, and scatter back

    // Use a batched approach: process in chunks that fit in buffers
    uint64_t batch_size = state->buffer_size / 4;  // 4 amplitudes per group
    if (batch_size > state->local_count) batch_size = state->local_count;

    for (uint64_t offset = 0; offset < state->local_count; offset += batch_size) {
        uint64_t current_batch = (state->local_count - offset < batch_size)
                                 ? (state->local_count - offset) : batch_size;

        // Pack local amplitudes for this batch
        for (uint64_t i = 0; i < current_batch; i++) {
            state->send_buffer[i] = state->amplitudes[offset + i];
        }

        // Gather from all 4 ranks to rank_00 (coordinator for this group)
        // Then apply gate and scatter back

        // Buffer layout: [batch from rank_00][batch from rank_01][batch from rank_10][batch from rank_11]
        double complex* gathered = (double complex*)malloc(4 * current_batch * sizeof(double complex));
        if (!gathered) return DIST_GATE_ERROR_ALLOC;

        // All ranks send to coordinator (rank_00)
        if (rank == rank_00) {
            // We are coordinator - gather from all
            memcpy(&gathered[0], state->send_buffer, current_batch * sizeof(double complex));

            // Receive from others
            mpi_bridge_error_t err;
            err = mpi_recv(state->dist_ctx, &gathered[current_batch],
                          current_batch * sizeof(double complex), rank_01, 0);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            err = mpi_recv(state->dist_ctx, &gathered[2 * current_batch],
                          current_batch * sizeof(double complex), rank_10, 0);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            err = mpi_recv(state->dist_ctx, &gathered[3 * current_batch],
                          current_batch * sizeof(double complex), rank_11, 0);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            // Apply gate to each group of 4
            for (uint64_t i = 0; i < current_batch; i++) {
                double complex a00 = gathered[i];
                double complex a01 = gathered[current_batch + i];
                double complex a10 = gathered[2 * current_batch + i];
                double complex a11 = gathered[3 * current_batch + i];

                gathered[i] = matrix->m[0]*a00 + matrix->m[1]*a01 +
                              matrix->m[2]*a10 + matrix->m[3]*a11;
                gathered[current_batch + i] = matrix->m[4]*a00 + matrix->m[5]*a01 +
                                               matrix->m[6]*a10 + matrix->m[7]*a11;
                gathered[2 * current_batch + i] = matrix->m[8]*a00 + matrix->m[9]*a01 +
                                                   matrix->m[10]*a10 + matrix->m[11]*a11;
                gathered[3 * current_batch + i] = matrix->m[12]*a00 + matrix->m[13]*a01 +
                                                   matrix->m[14]*a10 + matrix->m[15]*a11;
            }

            // Scatter results back
            memcpy(state->recv_buffer, &gathered[0], current_batch * sizeof(double complex));

            err = mpi_send(state->dist_ctx, &gathered[current_batch],
                          current_batch * sizeof(double complex), rank_01, 1);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            err = mpi_send(state->dist_ctx, &gathered[2 * current_batch],
                          current_batch * sizeof(double complex), rank_10, 1);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            err = mpi_send(state->dist_ctx, &gathered[3 * current_batch],
                          current_batch * sizeof(double complex), rank_11, 1);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }
        } else {
            // We are not coordinator - send to rank_00, receive result
            mpi_bridge_error_t err;
            err = mpi_send(state->dist_ctx, state->send_buffer,
                          current_batch * sizeof(double complex), rank_00, 0);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }

            err = mpi_recv(state->dist_ctx, state->recv_buffer,
                          current_batch * sizeof(double complex), rank_00, 1);
            if (err != MPI_BRIDGE_SUCCESS) { free(gathered); return DIST_GATE_ERROR_COMM; }
        }

        free(gathered);

        // Store results
        for (uint64_t i = 0; i < current_batch; i++) {
            state->amplitudes[offset + i] = state->recv_buffer[i];
        }
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_gate_2q(partitioned_state_t* state,
                               uint32_t qubit1,
                               uint32_t qubit2,
                               const gate_matrix_4x4_t* matrix) {
    if (!state || !matrix) {
        return DIST_GATE_ERROR_NOT_INITIALIZED;
    }

    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    if (qubit1 == qubit2) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    int q1_partition = partition_is_partition_qubit(state, qubit1);
    int q2_partition = partition_is_partition_qubit(state, qubit2);

    // Both local - simple case, no communication needed
    if (!q1_partition && !q2_partition) {
        return apply_local_2q(state, qubit1, qubit2, matrix);
    }

    // Exactly one partition qubit
    if (q1_partition && !q2_partition) {
        return apply_remote_2q_single_partition(state, qubit2, qubit1, matrix, 0);
    }

    if (!q1_partition && q2_partition) {
        return apply_remote_2q_single_partition(state, qubit1, qubit2, matrix, 1);
    }

    // Both qubits are partition qubits - most complex case
    return apply_remote_2q_both_partition(state, qubit1, qubit2, matrix);
}

dist_gate_error_t dist_cnot(partitioned_state_t* state,
                            uint32_t control,
                            uint32_t target) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (control >= state->num_qubits || target >= state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    if (control == target) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    int ctrl_partition = partition_is_partition_qubit(state, control);
    int tgt_partition = partition_is_partition_qubit(state, target);

    // Both local
    if (!ctrl_partition && !tgt_partition) {
        uint64_t ctrl_stride = 1ULL << control;
        uint64_t tgt_stride = 1ULL << target;

        for (uint64_t i = 0; i < state->local_count; i++) {
            // Only process if control bit is 1 and this is the |0⟩ target state
            if ((i & ctrl_stride) && !(i & tgt_stride)) {
                uint64_t partner = i ^ tgt_stride;
                if (partner < state->local_count) {
                    double complex tmp = state->amplitudes[i];
                    state->amplitudes[i] = state->amplitudes[partner];
                    state->amplitudes[partner] = tmp;
                }
            }
        }
        return DIST_GATE_SUCCESS;
    }

    // Target is partition qubit - need exchange
    if (tgt_partition && !ctrl_partition) {
        uint32_t partition_bit = target - state->local_qubits;
        int rank = mpi_get_rank(state->dist_ctx);
        int partner_rank = rank ^ (1 << partition_bit);

        // Only exchange amplitudes where control is 1
        uint64_t ctrl_stride = 1ULL << control;
        uint64_t exchange_count = state->local_count / 2;

        // Pack amplitudes with control=1
        uint64_t pack_idx = 0;
        for (uint64_t i = 0; i < state->local_count; i++) {
            if (i & ctrl_stride) {
                state->send_buffer[pack_idx++] = state->amplitudes[i];
            }
        }

        // Exchange
        mpi_bridge_error_t err = mpi_exchange_amplitudes(
            state->dist_ctx,
            state->send_buffer,
            state->recv_buffer,
            exchange_count,
            partner_rank,
            0
        );

        if (err != MPI_BRIDGE_SUCCESS) {
            return DIST_GATE_ERROR_COMM;
        }

        // Unpack
        pack_idx = 0;
        for (uint64_t i = 0; i < state->local_count; i++) {
            if (i & ctrl_stride) {
                state->amplitudes[i] = state->recv_buffer[pack_idx++];
            }
        }

        return DIST_GATE_SUCCESS;
    }

    // Control is partition qubit
    if (ctrl_partition) {
        uint32_t partition_bit = control - state->local_qubits;
        int rank = mpi_get_rank(state->dist_ctx);
        int control_is_one = (rank >> partition_bit) & 1;

        if (control_is_one) {
            // Apply X to target locally
            return dist_pauli_x(state, target);
        }
        // Control is 0 - do nothing
        return DIST_GATE_SUCCESS;
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_cz(partitioned_state_t* state,
                          uint32_t qubit1,
                          uint32_t qubit2) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    // CZ is diagonal - apply -1 phase when both qubits are |1⟩
    // This is simpler than CNOT because no amplitude exchange needed

    int q1_partition = partition_is_partition_qubit(state, qubit1);
    int q2_partition = partition_is_partition_qubit(state, qubit2);

    int rank = mpi_get_rank(state->dist_ctx);

    for (uint64_t i = 0; i < state->local_count; i++) {
        uint64_t global_idx = partition_local_to_global(state, i);

        int bit1 = q1_partition
                   ? ((rank >> (qubit1 - state->local_qubits)) & 1)
                   : get_bit(i, qubit1);
        int bit2 = q2_partition
                   ? ((rank >> (qubit2 - state->local_qubits)) & 1)
                   : get_bit(i, qubit2);

        if (bit1 && bit2) {
            state->amplitudes[i] *= -1.0;
        }
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_cphase(partitioned_state_t* state,
                              uint32_t control,
                              uint32_t target,
                              double phi) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    double complex phase = cexp(I * phi);

    int ctrl_partition = partition_is_partition_qubit(state, control);
    int tgt_partition = partition_is_partition_qubit(state, target);
    int rank = mpi_get_rank(state->dist_ctx);

    for (uint64_t i = 0; i < state->local_count; i++) {
        int ctrl_bit = ctrl_partition
                       ? ((rank >> (control - state->local_qubits)) & 1)
                       : get_bit(i, control);
        int tgt_bit = tgt_partition
                      ? ((rank >> (target - state->local_qubits)) & 1)
                      : get_bit(i, target);

        if (ctrl_bit && tgt_bit) {
            state->amplitudes[i] *= phase;
        }
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_swap(partitioned_state_t* state,
                            uint32_t qubit1,
                            uint32_t qubit2) {
    // SWAP = CNOT(1,2) CNOT(2,1) CNOT(1,2)
    dist_gate_error_t err;
    err = dist_cnot(state, qubit1, qubit2);
    if (err != DIST_GATE_SUCCESS) return err;
    err = dist_cnot(state, qubit2, qubit1);
    if (err != DIST_GATE_SUCCESS) return err;
    err = dist_cnot(state, qubit1, qubit2);
    return err;
}

dist_gate_error_t dist_iswap(partitioned_state_t* state,
                             uint32_t qubit1,
                             uint32_t qubit2) {
    return dist_gate_2q(state, qubit1, qubit2, &GATE_ISWAP);
}

dist_gate_error_t dist_sqrt_swap(partitioned_state_t* state,
                                 uint32_t qubit1,
                                 uint32_t qubit2) {
    // sqrt(SWAP) matrix
    gate_matrix_4x4_t sqrt_swap = {{
        1, 0, 0, 0,
        0, 0.5+0.5*I, 0.5-0.5*I, 0,
        0, 0.5-0.5*I, 0.5+0.5*I, 0,
        0, 0, 0, 1
    }};
    return dist_gate_2q(state, qubit1, qubit2, &sqrt_swap);
}

// ============================================================================
// MULTI-QUBIT GATES
// ============================================================================

dist_gate_error_t dist_toffoli(partitioned_state_t* state,
                               uint32_t control1,
                               uint32_t control2,
                               uint32_t target) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    // Toffoli: flip target when both controls are 1
    int c1_partition = partition_is_partition_qubit(state, control1);
    int c2_partition = partition_is_partition_qubit(state, control2);
    int tgt_partition = partition_is_partition_qubit(state, target);

    int rank = mpi_get_rank(state->dist_ctx);

    // If both controls are partition qubits and both 1, apply X to target
    if (c1_partition && c2_partition) {
        int c1_val = (rank >> (control1 - state->local_qubits)) & 1;
        int c2_val = (rank >> (control2 - state->local_qubits)) & 1;
        if (c1_val && c2_val) {
            return dist_pauli_x(state, target);
        }
        return DIST_GATE_SUCCESS;
    }

    // Local implementation
    if (!c1_partition && !c2_partition && !tgt_partition) {
        uint64_t c1_stride = 1ULL << control1;
        uint64_t c2_stride = 1ULL << control2;
        uint64_t tgt_stride = 1ULL << target;

        for (uint64_t i = 0; i < state->local_count; i++) {
            if ((i & c1_stride) && (i & c2_stride) && !(i & tgt_stride)) {
                uint64_t partner = i ^ tgt_stride;
                if (partner < state->local_count) {
                    double complex tmp = state->amplitudes[i];
                    state->amplitudes[i] = state->amplitudes[partner];
                    state->amplitudes[partner] = tmp;
                }
            }
        }
        return DIST_GATE_SUCCESS;
    }

    // Mixed case - decompose
    // Toffoli = H(t) CNOT(c2,t) Tdg(t) CNOT(c1,t) T(t) CNOT(c2,t) Tdg(t) CNOT(c1,t) T(c2) T(t) H(t) CNOT(c1,c2) T(c1) Tdg(c2) CNOT(c1,c2)
    // Simplified: use decomposition
    dist_gate_error_t err;

    err = dist_hadamard(state, target); if (err) return err;
    err = dist_cnot(state, control2, target); if (err) return err;
    err = dist_gate_1q(state, target, &GATE_TDAG); if (err) return err;
    err = dist_cnot(state, control1, target); if (err) return err;
    err = dist_t_gate(state, target); if (err) return err;
    err = dist_cnot(state, control2, target); if (err) return err;
    err = dist_gate_1q(state, target, &GATE_TDAG); if (err) return err;
    err = dist_cnot(state, control1, target); if (err) return err;
    err = dist_t_gate(state, control2); if (err) return err;
    err = dist_t_gate(state, target); if (err) return err;
    err = dist_hadamard(state, target); if (err) return err;
    err = dist_cnot(state, control1, control2); if (err) return err;
    err = dist_t_gate(state, control1); if (err) return err;
    err = dist_gate_1q(state, control2, &GATE_TDAG); if (err) return err;
    err = dist_cnot(state, control1, control2);

    return err;
}

dist_gate_error_t dist_fredkin(partitioned_state_t* state,
                               uint32_t control,
                               uint32_t target1,
                               uint32_t target2) {
    // Fredkin (CSWAP): swap targets when control is 1
    // = CNOT(t2,t1) Toffoli(c,t1,t2) CNOT(t2,t1)
    dist_gate_error_t err;
    err = dist_cnot(state, target2, target1); if (err) return err;
    err = dist_toffoli(state, control, target1, target2); if (err) return err;
    err = dist_cnot(state, target2, target1);
    return err;
}

dist_gate_error_t dist_mcz(partitioned_state_t* state,
                           const uint32_t* controls,
                           uint32_t num_controls) {
    if (!state || !controls || num_controls == 0) {
        return DIST_GATE_ERROR_NOT_INITIALIZED;
    }

    int rank = mpi_get_rank(state->dist_ctx);

    for (uint64_t i = 0; i < state->local_count; i++) {
        int all_ones = 1;

        for (uint32_t c = 0; c < num_controls; c++) {
            uint32_t qubit = controls[c];
            int bit_val;

            if (partition_is_partition_qubit(state, qubit)) {
                bit_val = (rank >> (qubit - state->local_qubits)) & 1;
            } else {
                bit_val = get_bit(i, qubit);
            }

            if (!bit_val) {
                all_ones = 0;
                break;
            }
        }

        if (all_ones) {
            state->amplitudes[i] *= -1.0;
        }
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_mcx(partitioned_state_t* state,
                           const uint32_t* controls,
                           uint32_t num_controls,
                           uint32_t target) {
    if (num_controls == 0) {
        return dist_pauli_x(state, target);
    }
    if (num_controls == 1) {
        return dist_cnot(state, controls[0], target);
    }
    if (num_controls == 2) {
        return dist_toffoli(state, controls[0], controls[1], target);
    }

    // General MCX: decompose into Toffolis
    // This is a simplified implementation
    // Full implementation uses gray code or recursion

    // For now, check all controls and apply X if all are 1
    int rank = mpi_get_rank(state->dist_ctx);
    int tgt_partition = partition_is_partition_qubit(state, target);

    // Check if all partition control qubits are 1
    int partition_controls_one = 1;
    for (uint32_t c = 0; c < num_controls; c++) {
        if (partition_is_partition_qubit(state, controls[c])) {
            int bit = (rank >> (controls[c] - state->local_qubits)) & 1;
            if (!bit) {
                partition_controls_one = 0;
                break;
            }
        }
    }

    if (!partition_controls_one) {
        return DIST_GATE_SUCCESS;  // Some partition control is 0
    }

    // Now handle remaining local controls + target
    // This is a simplification - proper implementation needs full exchange
    if (!tgt_partition) {
        uint64_t tgt_stride = 1ULL << target;

        for (uint64_t i = 0; i < state->local_count; i++) {
            // Check local control bits
            int all_local_ones = 1;
            for (uint32_t c = 0; c < num_controls; c++) {
                if (!partition_is_partition_qubit(state, controls[c])) {
                    if (!get_bit(i, controls[c])) {
                        all_local_ones = 0;
                        break;
                    }
                }
            }

            if (all_local_ones && !(i & tgt_stride)) {
                uint64_t partner = i ^ tgt_stride;
                if (partner < state->local_count) {
                    double complex tmp = state->amplitudes[i];
                    state->amplitudes[i] = state->amplitudes[partner];
                    state->amplitudes[partner] = tmp;
                }
            }
        }
    }

    return DIST_GATE_SUCCESS;
}

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

dist_gate_error_t dist_oracle_single(partitioned_state_t* state,
                                     uint64_t target_state) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (target_state >= state->total_amplitudes) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    // Check if target is local
    if (partition_is_local(state, target_state)) {
        uint64_t local_idx = partition_global_to_local(state, target_state);
        state->amplitudes[local_idx] *= -1.0;
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_oracle_multi(partitioned_state_t* state,
                                    const uint64_t* targets,
                                    uint32_t num_targets) {
    if (!state || !targets) return DIST_GATE_ERROR_NOT_INITIALIZED;

    for (uint32_t t = 0; t < num_targets; t++) {
        if (partition_is_local(state, targets[t])) {
            uint64_t local_idx = partition_global_to_local(state, targets[t]);
            state->amplitudes[local_idx] *= -1.0;
        }
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_grover_diffusion(partitioned_state_t* state) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    // Diffusion: 2|ψ⟩⟨ψ| - I where |ψ⟩ = (1/√N) Σ|i⟩

    // Step 1: Compute mean amplitude
    double complex local_sum = 0.0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        local_sum += state->amplitudes[i];
    }

    // All-reduce to get global sum
    double complex global_sum;
    mpi_bridge_error_t err = mpi_allreduce_sum_complex(
        state->dist_ctx, &local_sum, &global_sum, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return DIST_GATE_ERROR_COMM;
    }

    // Mean = sum / N
    double complex mean = global_sum / (double)state->total_amplitudes;

    // Step 2: Apply 2*mean - amplitude to each
    for (uint64_t i = 0; i < state->local_count; i++) {
        state->amplitudes[i] = 2.0 * mean - state->amplitudes[i];
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_grover_iteration(partitioned_state_t* state,
                                        uint64_t target_state) {
    dist_gate_error_t err;

    // Oracle
    err = dist_oracle_single(state, target_state);
    if (err != DIST_GATE_SUCCESS) return err;

    // Diffusion
    err = dist_grover_diffusion(state);

    return err;
}

dist_gate_error_t dist_grover_search(partitioned_state_t* state,
                                     uint64_t target_state,
                                     uint32_t num_iterations) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (num_iterations == 0) {
        num_iterations = dist_grover_optimal_iterations(state->num_qubits, 1);
    }

    // Ensure uniform superposition
    dist_gate_error_t err = partition_init_uniform(state);
    if (err != PARTITION_SUCCESS) return DIST_GATE_ERROR_NOT_INITIALIZED;

    // Run iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        err = dist_grover_iteration(state, target_state);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    return DIST_GATE_SUCCESS;
}

// ============================================================================
// QFT OPERATIONS
// ============================================================================

dist_gate_error_t dist_qft(partitioned_state_t* state,
                           uint32_t start_qubit,
                           uint32_t num_qft_qubits) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (start_qubit + num_qft_qubits > state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    dist_gate_error_t err;

    for (uint32_t i = 0; i < num_qft_qubits; i++) {
        uint32_t q = start_qubit + i;

        // Hadamard on qubit q
        err = dist_hadamard(state, q);
        if (err != DIST_GATE_SUCCESS) return err;

        // Controlled phase rotations
        for (uint32_t j = i + 1; j < num_qft_qubits; j++) {
            uint32_t control = start_qubit + j;
            double phase = M_PI / (1ULL << (j - i));
            err = dist_cphase(state, control, q, phase);
            if (err != DIST_GATE_SUCCESS) return err;
        }
    }

    // Swap qubits to reverse order (standard QFT convention)
    for (uint32_t i = 0; i < num_qft_qubits / 2; i++) {
        uint32_t q1 = start_qubit + i;
        uint32_t q2 = start_qubit + num_qft_qubits - 1 - i;
        err = dist_swap(state, q1, q2);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    return DIST_GATE_SUCCESS;
}

dist_gate_error_t dist_iqft(partitioned_state_t* state,
                            uint32_t start_qubit,
                            uint32_t num_qft_qubits) {
    if (!state) return DIST_GATE_ERROR_NOT_INITIALIZED;

    if (start_qubit + num_qft_qubits > state->num_qubits) {
        return DIST_GATE_ERROR_INVALID_QUBIT;
    }

    dist_gate_error_t err;

    // Swap qubits first
    for (uint32_t i = 0; i < num_qft_qubits / 2; i++) {
        uint32_t q1 = start_qubit + i;
        uint32_t q2 = start_qubit + num_qft_qubits - 1 - i;
        err = dist_swap(state, q1, q2);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    // Inverse of QFT: reverse order, conjugate phases
    for (int i = (int)num_qft_qubits - 1; i >= 0; i--) {
        uint32_t q = start_qubit + (uint32_t)i;

        // Controlled phase rotations (conjugate = negative)
        for (int j = (int)num_qft_qubits - 1; j > i; j--) {
            uint32_t control = start_qubit + (uint32_t)j;
            double phase = -M_PI / (1ULL << (j - i));
            err = dist_cphase(state, control, q, phase);
            if (err != DIST_GATE_SUCCESS) return err;
        }

        // Hadamard on qubit q
        err = dist_hadamard(state, q);
        if (err != DIST_GATE_SUCCESS) return err;
    }

    return DIST_GATE_SUCCESS;
}

// ============================================================================
// UTILITIES
// ============================================================================

void dist_build_rx_matrix(double theta, gate_matrix_2x2_t* matrix) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    matrix->m[0] = c + 0.0*I;
    matrix->m[1] = 0.0 - s*I;
    matrix->m[2] = 0.0 - s*I;
    matrix->m[3] = c + 0.0*I;
}

void dist_build_ry_matrix(double theta, gate_matrix_2x2_t* matrix) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    matrix->m[0] = c + 0.0*I;
    matrix->m[1] = -s + 0.0*I;
    matrix->m[2] = s + 0.0*I;
    matrix->m[3] = c + 0.0*I;
}

void dist_build_rz_matrix(double theta, gate_matrix_2x2_t* matrix) {
    matrix->m[0] = cexp(-I * theta / 2.0);
    matrix->m[1] = 0.0 + 0.0*I;
    matrix->m[2] = 0.0 + 0.0*I;
    matrix->m[3] = cexp(I * theta / 2.0);
}

void dist_build_phase_matrix(double phi, gate_matrix_2x2_t* matrix) {
    matrix->m[0] = 1.0 + 0.0*I;
    matrix->m[1] = 0.0 + 0.0*I;
    matrix->m[2] = 0.0 + 0.0*I;
    matrix->m[3] = cexp(I * phi);
}

void dist_build_cphase_matrix(double phi, gate_matrix_4x4_t* matrix) {
    memset(matrix->m, 0, sizeof(matrix->m));
    matrix->m[0] = 1.0;   // |00⟩
    matrix->m[5] = 1.0;   // |01⟩
    matrix->m[10] = 1.0;  // |10⟩
    matrix->m[15] = cexp(I * phi);  // |11⟩
}

const char* dist_gate_error_string(dist_gate_error_t error) {
    switch (error) {
        case DIST_GATE_SUCCESS:
            return "Success";
        case DIST_GATE_ERROR_INVALID_QUBIT:
            return "Invalid qubit index";
        case DIST_GATE_ERROR_COMM:
            return "Communication error";
        case DIST_GATE_ERROR_ALLOC:
            return "Memory allocation failed";
        case DIST_GATE_ERROR_NOT_INITIALIZED:
            return "State not initialized";
        case DIST_GATE_ERROR_INVALID_MATRIX:
            return "Invalid gate matrix";
        default:
            return "Unknown error";
    }
}

uint32_t dist_grover_optimal_iterations(uint32_t num_qubits, uint32_t num_targets) {
    double N = (double)(1ULL << num_qubits);
    double M = (double)num_targets;
    double theta = asin(sqrt(M / N));
    double iterations = (M_PI / (4.0 * theta)) - 0.5;
    return (uint32_t)(iterations + 0.5);
}
