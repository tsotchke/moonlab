/**
 * @file state_partition.c
 * @brief State vector partitioning implementation
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "state_partition.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Check for aligned allocation support
#ifdef __APPLE__
#include <malloc/malloc.h>
#define HAVE_POSIX_MEMALIGN 1
#elif defined(__linux__)
#define HAVE_POSIX_MEMALIGN 1
#endif

// SIMD alignment (64 bytes for AVX-512)
#define SIMD_ALIGNMENT 64

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Count trailing zeros (find lowest set bit position)
 */
static inline uint32_t count_trailing_zeros(uint64_t x) {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctzll(x);
#else
    uint32_t count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
#endif
}

/**
 * @brief Check if n is a power of 2
 */
static inline int is_power_of_2(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Calculate log2 of power of 2
 */
static inline uint32_t log2_of_power_of_2(uint64_t n) {
    return count_trailing_zeros(n);
}

/**
 * @brief Aligned memory allocation
 */
static void* aligned_alloc_impl(size_t size, size_t alignment) {
#ifdef HAVE_POSIX_MEMALIGN
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#else
    // Fallback: over-allocate and align manually
    void* raw = malloc(size + alignment + sizeof(void*));
    if (!raw) return NULL;

    void* aligned = (void*)(((uintptr_t)raw + alignment + sizeof(void*)) & ~(alignment - 1));
    ((void**)aligned)[-1] = raw;
    return aligned;
#endif
}

/**
 * @brief Free aligned memory
 */
static void aligned_free_impl(void* ptr) {
    if (!ptr) return;
#ifdef HAVE_POSIX_MEMALIGN
    free(ptr);
#else
    free(((void**)ptr)[-1]);
#endif
}

// ============================================================================
// PARTITION CREATION
// ============================================================================

partitioned_state_t* partition_state_create(distributed_ctx_t* dist_ctx,
                                            uint32_t num_qubits,
                                            const partition_config_t* config) {
    if (!dist_ctx || num_qubits == 0 || num_qubits > 50) {
        return NULL;
    }

    int size = mpi_get_size(dist_ctx);
    if (!is_power_of_2((uint64_t)size)) {
        fprintf(stderr, "partition_state_create: Number of processes must be power of 2\n");
        return NULL;
    }

    partitioned_state_t* state = (partitioned_state_t*)calloc(1, sizeof(partitioned_state_t));
    if (!state) return NULL;

    state->dist_ctx = dist_ctx;
    state->num_qubits = num_qubits;
    state->total_amplitudes = 1ULL << num_qubits;

    // Calculate partition bits (log2 of process count)
    state->partition_bits = log2_of_power_of_2((uint64_t)size);
    state->local_qubits = num_qubits - state->partition_bits;

    // Ensure we have enough qubits for this process count
    if (state->partition_bits > num_qubits) {
        fprintf(stderr, "partition_state_create: Too many processes for %u qubits\n", num_qubits);
        free(state);
        return NULL;
    }

    // Get local range from MPI context
    mpi_get_local_range(dist_ctx, &state->local_start, &state->local_end);
    state->local_count = state->local_end - state->local_start;

    // Allocate local amplitudes
    state->amplitudes_size = state->local_count * sizeof(double complex);

    int use_aligned = config ? config->use_aligned_memory : 1;
    if (use_aligned) {
        state->amplitudes = (double complex*)aligned_alloc_impl(state->amplitudes_size,
                                                                 SIMD_ALIGNMENT);
    } else {
        state->amplitudes = (double complex*)malloc(state->amplitudes_size);
    }

    if (!state->amplitudes) {
        free(state);
        return NULL;
    }
    state->owns_memory = 1;

    // Allocate communication buffers
    size_t buffer_size = config && config->comm_buffer_size > 0
                         ? config->comm_buffer_size
                         : (1ULL << (state->local_qubits > 20 ? 20 : state->local_qubits)) * sizeof(double complex);

    state->buffer_size = buffer_size / sizeof(double complex);

    if (use_aligned) {
        state->send_buffer = (double complex*)aligned_alloc_impl(buffer_size, SIMD_ALIGNMENT);
        state->recv_buffer = (double complex*)aligned_alloc_impl(buffer_size, SIMD_ALIGNMENT);
    } else {
        state->send_buffer = (double complex*)malloc(buffer_size);
        state->recv_buffer = (double complex*)malloc(buffer_size);
    }

    if (!state->send_buffer || !state->recv_buffer) {
        partition_state_free(state);
        return NULL;
    }

    // Initialize to |0⟩
    partition_init_zero(state);

    return state;
}

partitioned_state_t* partition_state_wrap(distributed_ctx_t* dist_ctx,
                                          uint32_t num_qubits,
                                          double complex* amplitudes,
                                          const partition_config_t* config) {
    if (!dist_ctx || !amplitudes || num_qubits == 0) {
        return NULL;
    }

    int size = mpi_get_size(dist_ctx);
    if (!is_power_of_2((uint64_t)size)) {
        return NULL;
    }

    partitioned_state_t* state = (partitioned_state_t*)calloc(1, sizeof(partitioned_state_t));
    if (!state) return NULL;

    state->dist_ctx = dist_ctx;
    state->num_qubits = num_qubits;
    state->total_amplitudes = 1ULL << num_qubits;
    state->partition_bits = log2_of_power_of_2((uint64_t)size);
    state->local_qubits = num_qubits - state->partition_bits;

    mpi_get_local_range(dist_ctx, &state->local_start, &state->local_end);
    state->local_count = state->local_end - state->local_start;

    state->amplitudes = amplitudes;
    state->amplitudes_size = state->local_count * sizeof(double complex);
    state->owns_memory = 0;  // Don't free external memory

    // Allocate communication buffers
    size_t buffer_size = config && config->comm_buffer_size > 0
                         ? config->comm_buffer_size
                         : (1ULL << 20) * sizeof(double complex);

    state->buffer_size = buffer_size / sizeof(double complex);
    state->send_buffer = (double complex*)malloc(buffer_size);
    state->recv_buffer = (double complex*)malloc(buffer_size);

    if (!state->send_buffer || !state->recv_buffer) {
        if (state->send_buffer) free(state->send_buffer);
        if (state->recv_buffer) free(state->recv_buffer);
        free(state);
        return NULL;
    }

    return state;
}

void partition_state_free(partitioned_state_t* state) {
    if (!state) return;

    if (state->owns_memory && state->amplitudes) {
        aligned_free_impl(state->amplitudes);
    }

    if (state->send_buffer) aligned_free_impl(state->send_buffer);
    if (state->recv_buffer) aligned_free_impl(state->recv_buffer);

    free(state);
}

// ============================================================================
// STATE INITIALIZATION
// ============================================================================

partition_error_t partition_init_zero(partitioned_state_t* state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    // Zero all local amplitudes
    memset(state->amplitudes, 0, state->amplitudes_size);

    // Only rank 0 owns index 0
    if (state->local_start == 0) {
        state->amplitudes[0] = 1.0 + 0.0 * I;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_init_uniform(partitioned_state_t* state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    double norm = 1.0 / sqrt((double)state->total_amplitudes);
    double complex uniform_amp = norm + 0.0 * I;

    for (uint64_t i = 0; i < state->local_count; i++) {
        state->amplitudes[i] = uniform_amp;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_init_basis(partitioned_state_t* state,
                                       uint64_t basis_state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    if (basis_state >= state->total_amplitudes) {
        return PARTITION_ERROR_INDEX_RANGE;
    }

    // Zero all local amplitudes
    memset(state->amplitudes, 0, state->amplitudes_size);

    // Set the basis state if local
    if (partition_is_local(state, basis_state)) {
        uint64_t local_idx = partition_global_to_local(state, basis_state);
        state->amplitudes[local_idx] = 1.0 + 0.0 * I;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_copy(partitioned_state_t* dest,
                                 const partitioned_state_t* src) {
    if (!dest || !src || !dest->amplitudes || !src->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    if (dest->num_qubits != src->num_qubits ||
        dest->local_count != src->local_count) {
        return PARTITION_ERROR_INVALID_QUBITS;
    }

    memcpy(dest->amplitudes, src->amplitudes, src->amplitudes_size);

    return PARTITION_SUCCESS;
}

// ============================================================================
// INDEX MAPPING
// ============================================================================

int partition_is_local(const partitioned_state_t* state, uint64_t global_index) {
    return global_index >= state->local_start && global_index < state->local_end;
}

uint64_t partition_global_to_local(const partitioned_state_t* state,
                                   uint64_t global_index) {
    if (!partition_is_local(state, global_index)) {
        return UINT64_MAX;
    }
    return global_index - state->local_start;
}

uint64_t partition_local_to_global(const partitioned_state_t* state,
                                   uint64_t local_index) {
    return state->local_start + local_index;
}

int partition_get_owner(const partitioned_state_t* state, uint64_t global_index) {
    // High-order bits determine owner
    // owner = global_index >> local_qubits
    return (int)(global_index >> state->local_qubits);
}

double complex partition_get_amplitude(const partitioned_state_t* state,
                                       uint64_t global_index) {
    if (!partition_is_local(state, global_index)) {
        return 0.0 + 0.0 * I;
    }
    uint64_t local_idx = partition_global_to_local(state, global_index);
    return state->amplitudes[local_idx];
}

partition_error_t partition_set_amplitude(partitioned_state_t* state,
                                          uint64_t global_index,
                                          double complex value) {
    if (!partition_is_local(state, global_index)) {
        return PARTITION_ERROR_INDEX_RANGE;
    }
    uint64_t local_idx = partition_global_to_local(state, global_index);
    state->amplitudes[local_idx] = value;
    return PARTITION_SUCCESS;
}

// ============================================================================
// EXCHANGE PLANNING
// ============================================================================

int partition_is_partition_qubit(const partitioned_state_t* state, uint32_t qubit) {
    // Partition qubits are the highest-order bits
    // With n qubits and p partition bits, qubits n-1 down to n-p are partition qubits
    return qubit >= state->local_qubits;
}

partition_error_t partition_plan_1q_exchange(const partitioned_state_t* state,
                                             uint32_t qubit,
                                             exchange_descriptor_t* desc) {
    if (!state || !desc) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    memset(desc, 0, sizeof(exchange_descriptor_t));

    // If not a partition qubit, no exchange needed
    if (!partition_is_partition_qubit(state, qubit)) {
        desc->requires_exchange = 0;
        return PARTITION_SUCCESS;
    }

    // For partition qubits, need to exchange with partner rank
    // Partner differs in the bit corresponding to this qubit
    int rank = mpi_get_rank(state->dist_ctx);
    uint32_t partition_bit = qubit - state->local_qubits;  // Which partition bit
    int partner = rank ^ (1 << partition_bit);

    desc->requires_exchange = 1;
    desc->partner_rank = partner;
    desc->count = state->local_count;

    // All local indices need to be exchanged for single-qubit gate on partition qubit
    desc->local_indices = (uint64_t*)malloc(desc->count * sizeof(uint64_t));
    desc->remote_indices = (uint64_t*)malloc(desc->count * sizeof(uint64_t));

    if (!desc->local_indices || !desc->remote_indices) {
        partition_free_exchange_desc(desc);
        return PARTITION_ERROR_ALLOC;
    }

    for (uint64_t i = 0; i < desc->count; i++) {
        uint64_t global_idx = partition_local_to_global(state, i);
        desc->local_indices[i] = global_idx;
        // Remote index has the partition bit flipped
        desc->remote_indices[i] = global_idx ^ (1ULL << qubit);
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_plan_2q_exchange(const partitioned_state_t* state,
                                             uint32_t qubit1,
                                             uint32_t qubit2,
                                             exchange_descriptor_t* desc) {
    if (!state || !desc) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    memset(desc, 0, sizeof(exchange_descriptor_t));

    int q1_partition = partition_is_partition_qubit(state, qubit1);
    int q2_partition = partition_is_partition_qubit(state, qubit2);

    // If neither is partition qubit, no exchange needed
    if (!q1_partition && !q2_partition) {
        desc->requires_exchange = 0;
        return PARTITION_SUCCESS;
    }

    // Determine partner rank
    int rank = mpi_get_rank(state->dist_ctx);
    int partner = rank;

    if (q1_partition) {
        uint32_t partition_bit = qubit1 - state->local_qubits;
        partner ^= (1 << partition_bit);
    }
    if (q2_partition) {
        uint32_t partition_bit = qubit2 - state->local_qubits;
        partner ^= (1 << partition_bit);
    }

    if (partner == rank) {
        // Both qubits flip the same bit, or cancel out - no exchange
        desc->requires_exchange = 0;
        return PARTITION_SUCCESS;
    }

    desc->requires_exchange = 1;
    desc->partner_rank = partner;

    // Count how many indices need exchange
    // For two-qubit gates, we need indices where the gate actually operates
    uint64_t exchange_count = 0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        uint64_t global_idx = partition_local_to_global(state, i);
        // Check if this index's partner is on the remote rank
        uint64_t partner_idx = global_idx;
        if (q1_partition) partner_idx ^= (1ULL << qubit1);
        if (q2_partition) partner_idx ^= (1ULL << qubit2);

        if (partition_get_owner(state, partner_idx) == partner) {
            exchange_count++;
        }
    }

    desc->count = exchange_count;
    if (exchange_count == 0) {
        desc->requires_exchange = 0;
        return PARTITION_SUCCESS;
    }

    desc->local_indices = (uint64_t*)malloc(exchange_count * sizeof(uint64_t));
    desc->remote_indices = (uint64_t*)malloc(exchange_count * sizeof(uint64_t));

    if (!desc->local_indices || !desc->remote_indices) {
        partition_free_exchange_desc(desc);
        return PARTITION_ERROR_ALLOC;
    }

    uint64_t idx = 0;
    for (uint64_t i = 0; i < state->local_count && idx < exchange_count; i++) {
        uint64_t global_idx = partition_local_to_global(state, i);
        uint64_t partner_idx = global_idx;
        if (q1_partition) partner_idx ^= (1ULL << qubit1);
        if (q2_partition) partner_idx ^= (1ULL << qubit2);

        if (partition_get_owner(state, partner_idx) == partner) {
            desc->local_indices[idx] = global_idx;
            desc->remote_indices[idx] = partner_idx;
            idx++;
        }
    }

    return PARTITION_SUCCESS;
}

void partition_free_exchange_desc(exchange_descriptor_t* desc) {
    if (!desc) return;
    if (desc->local_indices) free(desc->local_indices);
    if (desc->remote_indices) free(desc->remote_indices);
    memset(desc, 0, sizeof(exchange_descriptor_t));
}

// ============================================================================
// DATA EXCHANGE
// ============================================================================

partition_error_t partition_execute_exchange(partitioned_state_t* state,
                                             const exchange_descriptor_t* desc) {
    if (!state || !desc) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    if (!desc->requires_exchange || desc->count == 0) {
        return PARTITION_SUCCESS;
    }

    // Pack local amplitudes into send buffer
    uint64_t batch_size = state->buffer_size < desc->count ? state->buffer_size : desc->count;
    uint64_t offset = 0;

    while (offset < desc->count) {
        uint64_t current_batch = (desc->count - offset < batch_size)
                                 ? (desc->count - offset) : batch_size;

        // Pack send buffer
        for (uint64_t i = 0; i < current_batch; i++) {
            uint64_t local_idx = partition_global_to_local(state, desc->local_indices[offset + i]);
            state->send_buffer[i] = state->amplitudes[local_idx];
        }

        // Exchange with partner
        mpi_bridge_error_t err = mpi_exchange_amplitudes(state->dist_ctx,
                                                         state->send_buffer,
                                                         state->recv_buffer,
                                                         current_batch,
                                                         desc->partner_rank,
                                                         0);
        if (err != MPI_BRIDGE_SUCCESS) {
            return PARTITION_ERROR_MPI;
        }

        // The received amplitudes are the partner's values for remote_indices
        // Store in a temporary manner - actual gate application handles the update

        offset += current_batch;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_fetch_remote(partitioned_state_t* state,
                                         const uint64_t* global_indices,
                                         uint64_t count,
                                         double complex* buffer) {
    if (!state || !global_indices || !buffer) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    int size = mpi_get_size(state->dist_ctx);
    int rank = mpi_get_rank(state->dist_ctx);

    // Allocate arrays for MPI_Alltoallv
    int* send_counts = (int*)calloc(size, sizeof(int));
    int* recv_counts = (int*)calloc(size, sizeof(int));
    int* send_displs = (int*)calloc(size, sizeof(int));
    int* recv_displs = (int*)calloc(size, sizeof(int));

    if (!send_counts || !recv_counts || !send_displs || !recv_displs) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        return PARTITION_ERROR_ALLOC;
    }

    // Count how many indices each rank needs from us
    // First, count what we need from each rank
    for (uint64_t i = 0; i < count; i++) {
        int owner = partition_get_owner(state, global_indices[i]);
        recv_counts[owner]++;
    }

    // Exchange counts so each rank knows how many requests it will receive
    mpi_bridge_error_t err = mpi_allgather(state->dist_ctx,
                                           recv_counts, size * sizeof(int),
                                           send_counts);
    if (err != MPI_BRIDGE_SUCCESS) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        return PARTITION_ERROR_MPI;
    }

    // Compute displacements
    int total_send = 0, total_recv = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r] = total_send;
        recv_displs[r] = total_recv;
        total_send += send_counts[r];
        total_recv += recv_counts[r];
    }

    // Allocate index request buffers
    uint64_t* send_indices = (uint64_t*)malloc(total_send * sizeof(uint64_t));
    uint64_t* recv_indices = (uint64_t*)malloc(total_recv * sizeof(uint64_t));
    double complex* send_values = (double complex*)malloc(total_send * sizeof(double complex));
    double complex* recv_values = (double complex*)malloc(total_recv * sizeof(double complex));

    if (!send_indices || !recv_indices || !send_values || !recv_values) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        free(send_indices); free(recv_indices);
        free(send_values); free(recv_values);
        return PARTITION_ERROR_ALLOC;
    }

    // Pack indices we're requesting into recv_indices (sorted by owner)
    int* recv_offsets = (int*)calloc(size, sizeof(int));
    int* buffer_map = (int*)malloc(count * sizeof(int));  // Maps original index to recv position

    for (uint64_t i = 0; i < count; i++) {
        int owner = partition_get_owner(state, global_indices[i]);
        int pos = recv_displs[owner] + recv_offsets[owner]++;
        recv_indices[pos] = global_indices[i];
        buffer_map[i] = pos;
    }

    // Exchange indices: each rank sends what it needs, receives requests from others
    // Use point-to-point for index exchange
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            // Copy local requests
            for (int i = 0; i < recv_counts[rank]; i++) {
                send_indices[send_displs[rank] + i] = recv_indices[recv_displs[rank] + i];
            }
        } else {
            // Exchange with rank r
            if (recv_counts[r] > 0 || send_counts[r] > 0) {
                mpi_exchange_amplitudes(state->dist_ctx,
                                        &recv_indices[recv_displs[r]],
                                        &send_indices[send_displs[r]],
                                        (recv_counts[r] > send_counts[r] ? recv_counts[r] : send_counts[r]),
                                        r, 0);
            }
        }
    }

    // Look up local values for indices requested by other ranks
    for (int i = 0; i < total_send; i++) {
        if (partition_is_local(state, send_indices[i])) {
            uint64_t local_idx = partition_global_to_local(state, send_indices[i]);
            send_values[i] = state->amplitudes[local_idx];
        } else {
            send_values[i] = 0.0;  // Should not happen in correct usage
        }
    }

    // Exchange values
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            for (int i = 0; i < recv_counts[rank]; i++) {
                recv_values[recv_displs[rank] + i] = send_values[send_displs[rank] + i];
            }
        } else {
            if (recv_counts[r] > 0 || send_counts[r] > 0) {
                mpi_exchange_amplitudes(state->dist_ctx,
                                        &send_values[send_displs[r]],
                                        &recv_values[recv_displs[r]],
                                        (send_counts[r] > recv_counts[r] ? send_counts[r] : recv_counts[r]),
                                        r, 1);
            }
        }
    }

    // Unpack received values into buffer in original order
    for (uint64_t i = 0; i < count; i++) {
        buffer[i] = recv_values[buffer_map[i]];
    }

    // Cleanup
    free(send_counts); free(recv_counts);
    free(send_displs); free(recv_displs);
    free(recv_offsets); free(buffer_map);
    free(send_indices); free(recv_indices);
    free(send_values); free(recv_values);

    return PARTITION_SUCCESS;
}

partition_error_t partition_scatter_updates(partitioned_state_t* state,
                                            const uint64_t* global_indices,
                                            const double complex* values,
                                            uint64_t count) {
    if (!state || !global_indices || !values) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    int size = mpi_get_size(state->dist_ctx);
    int rank = mpi_get_rank(state->dist_ctx);

    // Allocate arrays for MPI communication
    int* send_counts = (int*)calloc(size, sizeof(int));
    int* recv_counts = (int*)calloc(size, sizeof(int));
    int* send_displs = (int*)calloc(size, sizeof(int));
    int* recv_displs = (int*)calloc(size, sizeof(int));

    if (!send_counts || !recv_counts || !send_displs || !recv_displs) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        return PARTITION_ERROR_ALLOC;
    }

    // Count how many updates go to each rank
    for (uint64_t i = 0; i < count; i++) {
        int owner = partition_get_owner(state, global_indices[i]);
        send_counts[owner]++;
    }

    // Exchange counts so each rank knows how many updates it will receive
    mpi_bridge_error_t err = mpi_alltoall_int(state->dist_ctx,
                                               send_counts, recv_counts, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        return PARTITION_ERROR_MPI;
    }

    // Compute displacements
    int total_send = 0, total_recv = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r] = total_send;
        recv_displs[r] = total_recv;
        total_send += send_counts[r];
        total_recv += recv_counts[r];
    }

    // Allocate buffers for indices and values
    uint64_t* send_indices = (uint64_t*)malloc(total_send * sizeof(uint64_t));
    uint64_t* recv_indices = (uint64_t*)malloc(total_recv * sizeof(uint64_t));
    double complex* send_values = (double complex*)malloc(total_send * sizeof(double complex));
    double complex* recv_values = (double complex*)malloc(total_recv * sizeof(double complex));

    if (!send_indices || !recv_indices || !send_values || !recv_values) {
        free(send_counts); free(recv_counts);
        free(send_displs); free(recv_displs);
        free(send_indices); free(recv_indices);
        free(send_values); free(recv_values);
        return PARTITION_ERROR_ALLOC;
    }

    // Pack updates by destination rank
    int* pack_offsets = (int*)calloc(size, sizeof(int));
    for (uint64_t i = 0; i < count; i++) {
        int owner = partition_get_owner(state, global_indices[i]);
        int pos = send_displs[owner] + pack_offsets[owner]++;
        send_indices[pos] = global_indices[i];
        send_values[pos] = values[i];
    }
    free(pack_offsets);

    // Exchange indices using point-to-point communication
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            // Local copy
            for (int i = 0; i < send_counts[rank]; i++) {
                recv_indices[recv_displs[rank] + i] = send_indices[send_displs[rank] + i];
                recv_values[recv_displs[rank] + i] = send_values[send_displs[rank] + i];
            }
        } else {
            // Exchange with rank r
            if (send_counts[r] > 0 || recv_counts[r] > 0) {
                // Send indices
                int max_count = send_counts[r] > recv_counts[r] ? send_counts[r] : recv_counts[r];
                if (max_count > 0) {
                    // Exchange indices as uint64_t (pack into double complex for exchange)
                    // Note: We use two exchanges - one for indices, one for values
                    double complex* temp_send_idx = (double complex*)malloc(max_count * sizeof(double complex));
                    double complex* temp_recv_idx = (double complex*)malloc(max_count * sizeof(double complex));

                    if (temp_send_idx && temp_recv_idx) {
                        // Pack indices into complex (real part holds index)
                        for (int i = 0; i < send_counts[r]; i++) {
                            temp_send_idx[i] = (double)send_indices[send_displs[r] + i];
                        }

                        mpi_exchange_amplitudes(state->dist_ctx,
                                                temp_send_idx, temp_recv_idx,
                                                max_count, r, 0);

                        // Unpack received indices
                        for (int i = 0; i < recv_counts[r]; i++) {
                            recv_indices[recv_displs[r] + i] = (uint64_t)creal(temp_recv_idx[i]);
                        }

                        // Exchange values
                        mpi_exchange_amplitudes(state->dist_ctx,
                                                &send_values[send_displs[r]],
                                                &recv_values[recv_displs[r]],
                                                max_count, r, 1);
                    }

                    free(temp_send_idx);
                    free(temp_recv_idx);
                }
            }
        }
    }

    // Apply received updates to local amplitudes
    for (int i = 0; i < total_recv; i++) {
        if (partition_is_local(state, recv_indices[i])) {
            uint64_t local_idx = partition_global_to_local(state, recv_indices[i]);
            state->amplitudes[local_idx] = recv_values[i];
        }
    }

    // Cleanup
    free(send_counts); free(recv_counts);
    free(send_displs); free(recv_displs);
    free(send_indices); free(recv_indices);
    free(send_values); free(recv_values);

    return PARTITION_SUCCESS;
}

// ============================================================================
// COLLECTIVE OPERATIONS
// ============================================================================

partition_error_t partition_global_norm_sq(const partitioned_state_t* state,
                                           double* norm_sq) {
    if (!state || !norm_sq || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    // Compute local sum of |amplitude|^2
    double local_sum = 0.0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        double re = creal(state->amplitudes[i]);
        double im = cimag(state->amplitudes[i]);
        local_sum += re * re + im * im;
    }

    // All-reduce sum
    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_sum, norm_sq, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return PARTITION_ERROR_MPI;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_normalize(partitioned_state_t* state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    double norm_sq;
    partition_error_t err = partition_global_norm_sq(state, &norm_sq);
    if (err != PARTITION_SUCCESS) {
        return err;
    }

    if (norm_sq < 1e-30) {
        return PARTITION_SUCCESS;  // Avoid division by zero
    }

    double scale = 1.0 / sqrt(norm_sq);

    for (uint64_t i = 0; i < state->local_count; i++) {
        state->amplitudes[i] *= scale;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_inner_product(const partitioned_state_t* state,
                                          const partitioned_state_t* other,
                                          double complex* result) {
    if (!state || !other || !result) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    if (state->num_qubits != other->num_qubits) {
        return PARTITION_ERROR_INVALID_QUBITS;
    }

    // Compute local inner product
    double complex local_sum = 0.0 + 0.0 * I;
    for (uint64_t i = 0; i < state->local_count; i++) {
        local_sum += conj(state->amplitudes[i]) * other->amplitudes[i];
    }

    // All-reduce sum of complex values
    mpi_bridge_error_t err = mpi_allreduce_sum_complex(state->dist_ctx,
                                                       &local_sum, result, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return PARTITION_ERROR_MPI;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_gather_to_root(const partitioned_state_t* state,
                                           double complex* full_state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    // Use MPI_Gather
    mpi_bridge_error_t err = mpi_gather(state->dist_ctx,
                                        state->amplitudes,
                                        state->local_count * sizeof(double complex),
                                        full_state, 0);
    if (err != MPI_BRIDGE_SUCCESS) {
        return PARTITION_ERROR_MPI;
    }

    return PARTITION_SUCCESS;
}

partition_error_t partition_scatter_from_root(partitioned_state_t* state,
                                              const double complex* full_state) {
    if (!state || !state->amplitudes) {
        return PARTITION_ERROR_NOT_INITIALIZED;
    }

    int rank = mpi_get_rank(state->dist_ctx);
    int size = mpi_get_size(state->dist_ctx);

    // Use MPI_Scatter to distribute portions of the full state to each rank
    // Each rank receives state->local_count amplitudes

    mpi_bridge_error_t err = mpi_scatter(state->dist_ctx,
                                          full_state,
                                          state->amplitudes,
                                          state->local_count * sizeof(double complex),
                                          0);
    if (err != MPI_BRIDGE_SUCCESS) {
        // Fall back to point-to-point if scatter not available
        if (rank == 0) {
            // Root: copy local portion
            if (full_state) {
                memcpy(state->amplitudes, full_state, state->amplitudes_size);
            }

            // Send each rank its portion
            for (int r = 1; r < size; r++) {
                uint64_t start = (uint64_t)r * state->local_count;
                err = mpi_send(state->dist_ctx,
                               &full_state[start],
                               state->amplitudes_size,
                               r, 0);
                if (err != MPI_BRIDGE_SUCCESS) {
                    return PARTITION_ERROR_MPI;
                }
            }
        } else {
            // Non-root: receive from root
            err = mpi_recv(state->dist_ctx,
                           state->amplitudes,
                           state->amplitudes_size,
                           0, 0);
            if (err != MPI_BRIDGE_SUCCESS) {
                return PARTITION_ERROR_MPI;
            }
        }
    }

    return PARTITION_SUCCESS;
}

// ============================================================================
// UTILITIES
// ============================================================================

void partition_print_info(const partitioned_state_t* state, int all_ranks) {
    if (!state) return;

    int rank = mpi_get_rank(state->dist_ctx);
    int size = mpi_get_size(state->dist_ctx);

    if (!all_ranks && rank != 0) return;

    printf("Partition Info [Rank %d/%d]:\n", rank, size);
    printf("  Total qubits: %u\n", state->num_qubits);
    printf("  Total amplitudes: %lu\n", (unsigned long)state->total_amplitudes);
    printf("  Partition bits: %u\n", state->partition_bits);
    printf("  Local qubits: %u\n", state->local_qubits);
    printf("  Local range: [%lu, %lu)\n",
           (unsigned long)state->local_start, (unsigned long)state->local_end);
    printf("  Local count: %lu (%.2f MB)\n",
           (unsigned long)state->local_count,
           (double)state->amplitudes_size / (1024.0 * 1024.0));
}

void partition_get_stats(const partitioned_state_t* state,
                         uint64_t* max_local,
                         uint64_t* min_local,
                         double* load_imbalance) {
    if (!state) return;

    uint64_t local = state->local_count;
    uint64_t global_max = local;
    uint64_t global_min = local;
    double global_sum = (double)local;

    int size = mpi_get_size(state->dist_ctx);

    // Use MPI_Allreduce to compute max, min, and sum across all ranks
    mpi_bridge_error_t err;

    // Get maximum local count across all ranks
    err = mpi_allreduce_max_uint64(state->dist_ctx, &local, &global_max, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        // Fallback: assume this rank's value
        global_max = local;
    }

    // Get minimum local count across all ranks
    err = mpi_allreduce_min_uint64(state->dist_ctx, &local, &global_min, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        // Fallback: assume this rank's value
        global_min = local;
    }

    // Get total sum for average calculation
    double local_double = (double)local;
    err = mpi_allreduce_sum_double(state->dist_ctx, &local_double, &global_sum, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        // Fallback: estimate total
        global_sum = (double)local * size;
    }

    // Calculate load imbalance: (max - min) / average
    double average = global_sum / (double)size;
    double imbalance = 0.0;
    if (average > 0.0) {
        imbalance = ((double)global_max - (double)global_min) / average;
    }

    if (max_local) *max_local = global_max;
    if (min_local) *min_local = global_min;
    if (load_imbalance) *load_imbalance = imbalance;
}

size_t partition_estimate_memory(uint32_t num_qubits, int num_processes) {
    if (num_qubits > 50 || !is_power_of_2((uint64_t)num_processes)) {
        return 0;
    }

    uint64_t total = 1ULL << num_qubits;
    uint64_t per_process = total / (uint64_t)num_processes;

    // State vector + 2 communication buffers
    size_t state_size = per_process * sizeof(double complex);
    size_t buffer_size = (1ULL << 20) * sizeof(double complex);  // 1M elements

    return state_size + 2 * buffer_size;
}

const char* partition_error_string(partition_error_t error) {
    switch (error) {
        case PARTITION_SUCCESS:
            return "Success";
        case PARTITION_ERROR_INVALID_QUBITS:
            return "Invalid qubit count";
        case PARTITION_ERROR_ALLOC:
            return "Memory allocation failed";
        case PARTITION_ERROR_MPI:
            return "MPI communication error";
        case PARTITION_ERROR_INDEX_RANGE:
            return "Index out of range";
        case PARTITION_ERROR_NOT_INITIALIZED:
            return "State not initialized";
        default:
            return "Unknown error";
    }
}
