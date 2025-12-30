/**
 * @file mpi_bridge.h
 * @brief MPI bridge layer for distributed quantum simulation
 *
 * Provides MPI initialization, utilities, and distributed context management
 * for quantum simulations across multiple compute nodes.
 *
 * Supports:
 * - Standard MPI (MPICH, OpenMPI)
 * - RDMA-capable networks (InfiniBand, AWS EFA)
 * - Hybrid MPI+OpenMP parallelization
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef MPI_BRIDGE_H
#define MPI_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MPI BRIDGE TYPES
// ============================================================================

/**
 * @brief Distributed computing context
 *
 * Holds MPI state and partition information for distributed simulation.
 */
typedef struct {
    int rank;                    /**< This node's rank (0 to size-1) */
    int size;                    /**< Total number of nodes */
    int local_rank;              /**< Rank within node (for multi-GPU) */
    int local_size;              /**< Processes per node */
    uint64_t start_index;        /**< First amplitude index owned */
    uint64_t end_index;          /**< Last amplitude index owned (exclusive) */
    uint64_t local_count;        /**< Number of amplitudes owned */
    uint32_t num_qubits;         /**< Total qubits in simulation */
    uint32_t partition_qubits;   /**< Qubits used for partitioning */
    void* mpi_comm;              /**< MPI communicator (cast from MPI_Comm) */
    void* local_comm;            /**< Intra-node communicator */
    int initialized;             /**< MPI initialized by us */
    int thread_support;          /**< MPI thread support level */
    char processor_name[256];    /**< Processor/hostname */
} distributed_ctx_t;

/**
 * @brief MPI error codes
 */
typedef enum {
    MPI_BRIDGE_SUCCESS = 0,
    MPI_BRIDGE_ERROR_INIT = -1,
    MPI_BRIDGE_ERROR_COMM = -2,
    MPI_BRIDGE_ERROR_PARTITION = -3,
    MPI_BRIDGE_ERROR_ALLOC = -4,
    MPI_BRIDGE_ERROR_SYNC = -5,
    MPI_BRIDGE_ERROR_NOT_SUPPORTED = -6
} mpi_bridge_error_t;

/**
 * @brief Communication mode for data exchange
 */
typedef enum {
    MPI_COMM_BLOCKING,           /**< Blocking sends/receives */
    MPI_COMM_NONBLOCKING,        /**< Non-blocking with wait */
    MPI_COMM_PERSISTENT          /**< Persistent communication */
} mpi_comm_mode_t;

/**
 * @brief MPI initialization options
 */
typedef struct {
    int require_thread_multiple;  /**< Require MPI_THREAD_MULTIPLE */
    int enable_rdma;              /**< Enable RDMA optimizations */
    int enable_gpu_direct;        /**< Enable GPU-Direct RDMA */
    mpi_comm_mode_t comm_mode;    /**< Default communication mode */
} mpi_init_options_t;

// ============================================================================
// INITIALIZATION & FINALIZATION
// ============================================================================

/**
 * @brief Check if MPI is available
 *
 * @return 1 if MPI support compiled in, 0 otherwise
 */
int mpi_is_available(void);

/**
 * @brief Initialize MPI bridge
 *
 * Initializes MPI if not already done, sets up distributed context.
 *
 * @param argc Pointer to argc from main()
 * @param argv Pointer to argv from main()
 * @param options Optional initialization options (NULL for defaults)
 * @return Distributed context or NULL on failure
 */
distributed_ctx_t* mpi_bridge_init(int* argc, char*** argv,
                                   const mpi_init_options_t* options);

/**
 * @brief Initialize MPI bridge (no argc/argv)
 *
 * For use when MPI is already initialized.
 *
 * @param options Optional initialization options (NULL for defaults)
 * @return Distributed context or NULL on failure
 */
distributed_ctx_t* mpi_bridge_init_no_args(const mpi_init_options_t* options);

/**
 * @brief Free distributed context
 *
 * Frees context resources. Does NOT finalize MPI unless we initialized it.
 *
 * @param ctx Distributed context
 */
void mpi_bridge_free(distributed_ctx_t* ctx);

/**
 * @brief Finalize MPI
 *
 * Only call at program end. Safe to call multiple times.
 */
void mpi_bridge_finalize(void);

// ============================================================================
// CONTEXT QUERIES
// ============================================================================

/**
 * @brief Get current rank
 *
 * @param ctx Distributed context
 * @return Rank (0 to size-1)
 */
int mpi_get_rank(const distributed_ctx_t* ctx);

/**
 * @brief Get total number of processes
 *
 * @param ctx Distributed context
 * @return Number of MPI processes
 */
int mpi_get_size(const distributed_ctx_t* ctx);

/**
 * @brief Check if current process is root (rank 0)
 *
 * @param ctx Distributed context
 * @return 1 if root, 0 otherwise
 */
int mpi_is_root(const distributed_ctx_t* ctx);

/**
 * @brief Get amplitude index range for this rank
 *
 * @param ctx Distributed context
 * @param start Output: starting index (inclusive)
 * @param end Output: ending index (exclusive)
 */
void mpi_get_local_range(const distributed_ctx_t* ctx,
                         uint64_t* start, uint64_t* end);

/**
 * @brief Get rank that owns a specific amplitude index
 *
 * @param ctx Distributed context
 * @param index Amplitude index
 * @return Rank owning the index
 */
int mpi_get_owner_rank(const distributed_ctx_t* ctx, uint64_t index);

/**
 * @brief Check if amplitude index is local
 *
 * @param ctx Distributed context
 * @param index Amplitude index
 * @return 1 if local, 0 if remote
 */
int mpi_is_local_index(const distributed_ctx_t* ctx, uint64_t index);

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

/**
 * @brief Barrier synchronization
 *
 * Blocks until all ranks reach this point.
 *
 * @param ctx Distributed context
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_barrier(distributed_ctx_t* ctx);

/**
 * @brief Barrier with timeout
 *
 * @param ctx Distributed context
 * @param timeout_ms Timeout in milliseconds (0 for infinite)
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_barrier_timeout(distributed_ctx_t* ctx,
                                       uint32_t timeout_ms);

// ============================================================================
// POINT-TO-POINT COMMUNICATION
// ============================================================================

/**
 * @brief Send complex amplitudes to another rank
 *
 * @param ctx Distributed context
 * @param data Data to send
 * @param count Number of complex doubles
 * @param dest Destination rank
 * @param tag Message tag
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_send_amplitudes(distributed_ctx_t* ctx,
                                       const void* data, uint64_t count,
                                       int dest, int tag);

/**
 * @brief Receive complex amplitudes from another rank
 *
 * @param ctx Distributed context
 * @param data Buffer for received data
 * @param count Maximum count to receive
 * @param source Source rank (or MPI_ANY_SOURCE equivalent)
 * @param tag Message tag (or MPI_ANY_TAG equivalent)
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_recv_amplitudes(distributed_ctx_t* ctx,
                                       void* data, uint64_t count,
                                       int source, int tag);

/**
 * @brief Exchange amplitudes with another rank
 *
 * Simultaneous send and receive for paired exchanges.
 *
 * @param ctx Distributed context
 * @param send_data Data to send
 * @param recv_data Buffer for received data
 * @param count Number of complex doubles
 * @param partner Partner rank
 * @param tag Message tag
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_exchange_amplitudes(distributed_ctx_t* ctx,
                                           const void* send_data,
                                           void* recv_data,
                                           uint64_t count,
                                           int partner, int tag);

// ============================================================================
// COLLECTIVE OPERATIONS
// ============================================================================

/**
 * @brief Broadcast data from root to all ranks
 *
 * @param ctx Distributed context
 * @param data Data buffer (significant at root, filled at others)
 * @param count Number of bytes
 * @param root Root rank
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_broadcast(distributed_ctx_t* ctx,
                                 void* data, size_t count, int root);

/**
 * @brief All-reduce sum of complex values
 *
 * @param ctx Distributed context
 * @param send_data Local values
 * @param recv_data Output sum
 * @param count Number of complex doubles
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_allreduce_sum_complex(distributed_ctx_t* ctx,
                                             const void* send_data,
                                             void* recv_data,
                                             uint64_t count);

/**
 * @brief All-reduce sum of doubles
 *
 * @param ctx Distributed context
 * @param send_data Local values
 * @param recv_data Output sum
 * @param count Number of doubles
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_allreduce_sum_double(distributed_ctx_t* ctx,
                                            const double* send_data,
                                            double* recv_data,
                                            uint64_t count);

/**
 * @brief Gather data to root
 *
 * @param ctx Distributed context
 * @param send_data Local data
 * @param send_count Elements per rank
 * @param recv_data Receive buffer (only at root)
 * @param root Root rank
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_gather(distributed_ctx_t* ctx,
                              const void* send_data, size_t send_count,
                              void* recv_data, int root);

/**
 * @brief All-gather data from all ranks
 *
 * @param ctx Distributed context
 * @param send_data Local data
 * @param send_count Elements per rank
 * @param recv_data Receive buffer (all ranks get all data)
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_allgather(distributed_ctx_t* ctx,
                                 const void* send_data, size_t send_count,
                                 void* recv_data);

/**
 * @brief Scatter data from root to all ranks
 *
 * @param ctx Distributed context
 * @param send_data Data to scatter (only at root)
 * @param recv_data Buffer for received data
 * @param recv_count Bytes per rank
 * @param root Root rank
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_scatter(distributed_ctx_t* ctx,
                               const void* send_data,
                               void* recv_data,
                               size_t recv_count,
                               int root);

/**
 * @brief All-to-all exchange of integers
 *
 * Each rank sends count integers to every other rank.
 *
 * @param ctx Distributed context
 * @param send_data Send buffer (size * count integers)
 * @param recv_data Receive buffer (size * count integers)
 * @param count Integers per rank
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_alltoall_int(distributed_ctx_t* ctx,
                                    const int* send_data,
                                    int* recv_data,
                                    int count);

/**
 * @brief Send raw data to another rank
 *
 * @param ctx Distributed context
 * @param data Data to send
 * @param count Number of bytes
 * @param dest Destination rank
 * @param tag Message tag
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_send(distributed_ctx_t* ctx,
                            const void* data, size_t count,
                            int dest, int tag);

/**
 * @brief Receive raw data from another rank
 *
 * @param ctx Distributed context
 * @param data Buffer for received data
 * @param count Number of bytes to receive
 * @param source Source rank
 * @param tag Message tag
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_recv(distributed_ctx_t* ctx,
                            void* data, size_t count,
                            int source, int tag);

/**
 * @brief All-reduce max of uint64 values
 *
 * @param ctx Distributed context
 * @param send_data Local values
 * @param recv_data Output max
 * @param count Number of uint64_t values
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_allreduce_max_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count);

/**
 * @brief All-reduce min of uint64 values
 *
 * @param ctx Distributed context
 * @param send_data Local values
 * @param recv_data Output min
 * @param count Number of uint64_t values
 * @return MPI_BRIDGE_SUCCESS or error code
 */
mpi_bridge_error_t mpi_allreduce_min_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Print distributed context info
 *
 * Prints rank, size, partition info. Only root prints by default.
 *
 * @param ctx Distributed context
 * @param all_ranks Print from all ranks (1) or only root (0)
 */
void mpi_print_context_info(const distributed_ctx_t* ctx, int all_ranks);

/**
 * @brief Get processor name
 *
 * @param ctx Distributed context
 * @return Processor/hostname string
 */
const char* mpi_get_processor_name(const distributed_ctx_t* ctx);

/**
 * @brief Check if running in distributed mode
 *
 * @param ctx Distributed context
 * @return 1 if size > 1, 0 otherwise
 */
int mpi_is_distributed(const distributed_ctx_t* ctx);

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char* mpi_bridge_error_string(mpi_bridge_error_t error);

/**
 * @brief Abort all processes
 *
 * Emergency abort for unrecoverable errors.
 *
 * @param ctx Distributed context (or NULL)
 * @param error_code Exit code
 * @param message Error message
 */
void mpi_abort(distributed_ctx_t* ctx, int error_code, const char* message);

#ifdef __cplusplus
}
#endif

#endif /* MPI_BRIDGE_H */
