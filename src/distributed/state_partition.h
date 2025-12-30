/**
 * @file state_partition.h
 * @brief State vector partitioning for distributed quantum simulation
 *
 * Handles distribution of quantum state amplitudes across MPI ranks,
 * including index mapping, load balancing, and memory management.
 *
 * Partitioning Strategy:
 * - State vector of 2^n amplitudes split across P processes
 * - Each process owns contiguous range of indices
 * - High-order qubits determine partition (qubit n-1...n-log2(P))
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef STATE_PARTITION_H
#define STATE_PARTITION_H

#include <stdint.h>
#include <stddef.h>
#include <complex.h>
#include "mpi_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PARTITION TYPES
// ============================================================================

/**
 * @brief Partitioned quantum state
 *
 * Represents a distributed state vector with local storage and
 * partition metadata for efficient gate operations.
 */
typedef struct {
    distributed_ctx_t* dist_ctx;     /**< MPI context */
    uint32_t num_qubits;             /**< Total qubits in system */
    uint64_t total_amplitudes;       /**< Total state dimension (2^n) */

    /* Local partition info */
    uint64_t local_start;            /**< First local index (inclusive) */
    uint64_t local_end;              /**< Last local index (exclusive) */
    uint64_t local_count;            /**< Number of local amplitudes */

    /* Local amplitude storage */
    double complex* amplitudes;      /**< Local amplitude array */
    size_t amplitudes_size;          /**< Size in bytes */
    int owns_memory;                 /**< Did we allocate amplitudes? */

    /* Partition qubits */
    uint32_t partition_bits;         /**< log2(num_processes) */
    uint32_t local_qubits;           /**< Qubits represented locally */

    /* Communication buffers */
    double complex* send_buffer;     /**< Pre-allocated send buffer */
    double complex* recv_buffer;     /**< Pre-allocated recv buffer */
    size_t buffer_size;              /**< Buffer capacity */
} partitioned_state_t;

/**
 * @brief Partition configuration options
 */
typedef struct {
    int use_aligned_memory;          /**< Use SIMD-aligned allocation */
    size_t comm_buffer_size;         /**< Communication buffer size (0=auto) */
    int prefetch_remote;             /**< Enable remote prefetching */
    int optimize_for_locality;       /**< Optimize partition for gate locality */
} partition_config_t;

/**
 * @brief Partition error codes
 */
typedef enum {
    PARTITION_SUCCESS = 0,
    PARTITION_ERROR_INVALID_QUBITS = -1,
    PARTITION_ERROR_ALLOC = -2,
    PARTITION_ERROR_MPI = -3,
    PARTITION_ERROR_INDEX_RANGE = -4,
    PARTITION_ERROR_NOT_INITIALIZED = -5
} partition_error_t;

/**
 * @brief Index exchange descriptor
 *
 * Describes amplitudes that need to be exchanged for a gate operation.
 */
typedef struct {
    uint64_t* local_indices;         /**< Local indices involved */
    uint64_t* remote_indices;        /**< Corresponding remote indices */
    uint64_t count;                  /**< Number of pairs */
    int partner_rank;                /**< Rank to exchange with */
    int requires_exchange;           /**< Whether exchange is needed */
} exchange_descriptor_t;

// ============================================================================
// PARTITION CREATION
// ============================================================================

/**
 * @brief Create partitioned quantum state
 *
 * Allocates distributed state vector initialized to |0⟩.
 *
 * @param dist_ctx MPI distributed context
 * @param num_qubits Number of qubits (state has 2^num_qubits amplitudes)
 * @param config Optional configuration (NULL for defaults)
 * @return Partitioned state or NULL on failure
 */
partitioned_state_t* partition_state_create(distributed_ctx_t* dist_ctx,
                                            uint32_t num_qubits,
                                            const partition_config_t* config);

/**
 * @brief Create partitioned state with external memory
 *
 * Uses provided amplitude array instead of allocating.
 *
 * @param dist_ctx MPI distributed context
 * @param num_qubits Number of qubits
 * @param amplitudes Pre-allocated amplitude array (must be correctly sized)
 * @param config Optional configuration
 * @return Partitioned state or NULL on failure
 */
partitioned_state_t* partition_state_wrap(distributed_ctx_t* dist_ctx,
                                          uint32_t num_qubits,
                                          double complex* amplitudes,
                                          const partition_config_t* config);

/**
 * @brief Free partitioned state
 *
 * @param state Partitioned state to free
 */
void partition_state_free(partitioned_state_t* state);

// ============================================================================
// STATE INITIALIZATION
// ============================================================================

/**
 * @brief Initialize state to |0⟩
 *
 * Sets amplitude[0] = 1, all others = 0.
 * Only rank 0 has non-zero amplitude.
 *
 * @param state Partitioned state
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_init_zero(partitioned_state_t* state);

/**
 * @brief Initialize to uniform superposition
 *
 * Sets all amplitudes to 1/sqrt(2^n).
 *
 * @param state Partitioned state
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_init_uniform(partitioned_state_t* state);

/**
 * @brief Initialize to specific basis state
 *
 * Sets amplitude[basis_state] = 1, all others = 0.
 *
 * @param state Partitioned state
 * @param basis_state Target basis state index
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_init_basis(partitioned_state_t* state,
                                       uint64_t basis_state);

/**
 * @brief Copy amplitudes from another state
 *
 * @param dest Destination state
 * @param src Source state
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_copy(partitioned_state_t* dest,
                                 const partitioned_state_t* src);

// ============================================================================
// INDEX MAPPING
// ============================================================================

/**
 * @brief Check if global index is owned by this rank
 *
 * @param state Partitioned state
 * @param global_index Global amplitude index
 * @return 1 if local, 0 if remote
 */
int partition_is_local(const partitioned_state_t* state, uint64_t global_index);

/**
 * @brief Convert global index to local index
 *
 * @param state Partitioned state
 * @param global_index Global amplitude index
 * @return Local index or UINT64_MAX if not local
 */
uint64_t partition_global_to_local(const partitioned_state_t* state,
                                   uint64_t global_index);

/**
 * @brief Convert local index to global index
 *
 * @param state Partitioned state
 * @param local_index Local array index
 * @return Global amplitude index
 */
uint64_t partition_local_to_global(const partitioned_state_t* state,
                                   uint64_t local_index);

/**
 * @brief Get rank that owns a global index
 *
 * @param state Partitioned state
 * @param global_index Global amplitude index
 * @return Owning rank (0 to size-1)
 */
int partition_get_owner(const partitioned_state_t* state, uint64_t global_index);

/**
 * @brief Get local amplitude by global index
 *
 * @param state Partitioned state
 * @param global_index Global amplitude index
 * @return Amplitude value (0 if not local)
 */
double complex partition_get_amplitude(const partitioned_state_t* state,
                                       uint64_t global_index);

/**
 * @brief Set local amplitude by global index
 *
 * @param state Partitioned state
 * @param global_index Global amplitude index
 * @param value New amplitude value
 * @return PARTITION_SUCCESS if local, PARTITION_ERROR_INDEX_RANGE if remote
 */
partition_error_t partition_set_amplitude(partitioned_state_t* state,
                                          uint64_t global_index,
                                          double complex value);

// ============================================================================
// EXCHANGE PLANNING
// ============================================================================

/**
 * @brief Plan exchange for single-qubit gate
 *
 * For gates on "partition qubits" (high-order bits), determines
 * which amplitudes need exchange.
 *
 * @param state Partitioned state
 * @param qubit Target qubit index
 * @param desc Output exchange descriptor
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_plan_1q_exchange(const partitioned_state_t* state,
                                             uint32_t qubit,
                                             exchange_descriptor_t* desc);

/**
 * @brief Plan exchange for two-qubit gate
 *
 * @param state Partitioned state
 * @param qubit1 First qubit (control for CNOT)
 * @param qubit2 Second qubit (target for CNOT)
 * @param desc Output exchange descriptor
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_plan_2q_exchange(const partitioned_state_t* state,
                                             uint32_t qubit1,
                                             uint32_t qubit2,
                                             exchange_descriptor_t* desc);

/**
 * @brief Free exchange descriptor resources
 *
 * @param desc Exchange descriptor to free
 */
void partition_free_exchange_desc(exchange_descriptor_t* desc);

/**
 * @brief Check if qubit is a partition qubit
 *
 * Partition qubits are high-order bits that determine amplitude ownership.
 * Gates on these qubits require inter-rank communication.
 *
 * @param state Partitioned state
 * @param qubit Qubit index to check
 * @return 1 if partition qubit, 0 if local qubit
 */
int partition_is_partition_qubit(const partitioned_state_t* state, uint32_t qubit);

// ============================================================================
// DATA EXCHANGE
// ============================================================================

/**
 * @brief Execute amplitude exchange
 *
 * Exchanges amplitudes according to descriptor.
 *
 * @param state Partitioned state
 * @param desc Exchange descriptor
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_execute_exchange(partitioned_state_t* state,
                                             const exchange_descriptor_t* desc);

/**
 * @brief Fetch remote amplitudes
 *
 * Gets amplitudes from remote ranks into local buffer.
 *
 * @param state Partitioned state
 * @param global_indices Array of global indices to fetch
 * @param count Number of indices
 * @param buffer Output buffer for amplitudes
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_fetch_remote(partitioned_state_t* state,
                                         const uint64_t* global_indices,
                                         uint64_t count,
                                         double complex* buffer);

/**
 * @brief Scatter amplitudes to remote ranks
 *
 * Sends amplitude updates to their owning ranks.
 *
 * @param state Partitioned state
 * @param global_indices Array of global indices
 * @param values Array of new values
 * @param count Number of updates
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_scatter_updates(partitioned_state_t* state,
                                            const uint64_t* global_indices,
                                            const double complex* values,
                                            uint64_t count);

// ============================================================================
// COLLECTIVE OPERATIONS
// ============================================================================

/**
 * @brief Compute global norm squared
 *
 * Sums |amplitude|^2 across all ranks.
 *
 * @param state Partitioned state
 * @param norm_sq Output norm squared
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_global_norm_sq(const partitioned_state_t* state,
                                           double* norm_sq);

/**
 * @brief Normalize state vector
 *
 * Scales all amplitudes so total probability = 1.
 *
 * @param state Partitioned state
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_normalize(partitioned_state_t* state);

/**
 * @brief Compute inner product with another state
 *
 * Computes <state|other> across all ranks.
 *
 * @param state First state
 * @param other Second state
 * @param result Output inner product
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_inner_product(const partitioned_state_t* state,
                                          const partitioned_state_t* other,
                                          double complex* result);

/**
 * @brief Gather full state to root
 *
 * Collects entire state vector on rank 0 (for small states only).
 *
 * @param state Partitioned state
 * @param full_state Output buffer (only significant at root)
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_gather_to_root(const partitioned_state_t* state,
                                           double complex* full_state);

/**
 * @brief Scatter state from root
 *
 * Distributes state vector from rank 0 to all ranks.
 *
 * @param state Partitioned state (receives data)
 * @param full_state Input buffer (only significant at root)
 * @return PARTITION_SUCCESS or error code
 */
partition_error_t partition_scatter_from_root(partitioned_state_t* state,
                                              const double complex* full_state);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Print partition info
 *
 * @param state Partitioned state
 * @param all_ranks Print from all ranks (1) or root only (0)
 */
void partition_print_info(const partitioned_state_t* state, int all_ranks);

/**
 * @brief Get partition statistics
 *
 * @param state Partitioned state
 * @param max_local Output max amplitudes on any rank
 * @param min_local Output min amplitudes on any rank
 * @param load_imbalance Output imbalance ratio (max/avg - 1)
 */
void partition_get_stats(const partitioned_state_t* state,
                         uint64_t* max_local,
                         uint64_t* min_local,
                         double* load_imbalance);

/**
 * @brief Estimate memory usage
 *
 * @param num_qubits Number of qubits
 * @param num_processes Number of MPI processes
 * @return Memory per process in bytes
 */
size_t partition_estimate_memory(uint32_t num_qubits, int num_processes);

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error message
 */
const char* partition_error_string(partition_error_t error);

#ifdef __cplusplus
}
#endif

#endif /* STATE_PARTITION_H */
