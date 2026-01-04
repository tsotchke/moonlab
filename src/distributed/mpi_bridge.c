/**
 * @file mpi_bridge.c
 * @brief MPI bridge layer implementation
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "mpi_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef HAS_MPI
#include <mpi.h>

// Track if we initialized MPI
static int mpi_initialized_by_us = 0;
static int mpi_finalized = 0;

// ============================================================================
// INITIALIZATION
// ============================================================================

int mpi_is_available(void) {
    return 1;
}

distributed_ctx_t* mpi_bridge_init(int* argc, char*** argv,
                                   const mpi_init_options_t* options) {
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);

    if (!already_initialized) {
        int required = MPI_THREAD_SINGLE;
        int provided;

        if (options && options->require_thread_multiple) {
            required = MPI_THREAD_MULTIPLE;
        }

        int err = MPI_Init_thread(argc, argv, required, &provided);
        if (err != MPI_SUCCESS) {
            return NULL;
        }

        mpi_initialized_by_us = 1;
    }

    return mpi_bridge_init_no_args(options);
}

distributed_ctx_t* mpi_bridge_init_no_args(const mpi_init_options_t* options) {
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);

    if (!already_initialized) {
        return NULL;
    }

    distributed_ctx_t* ctx = calloc(1, sizeof(distributed_ctx_t));
    if (!ctx) return NULL;

    ctx->mpi_comm = malloc(sizeof(MPI_Comm));
    if (!ctx->mpi_comm) {
        free(ctx);
        return NULL;
    }
    *(MPI_Comm*)ctx->mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);

    // Get processor name
    int name_len;
    MPI_Get_processor_name(ctx->processor_name, &name_len);

    // Query thread support level
    MPI_Query_thread(&ctx->thread_support);

    // Create local communicator for intra-node communication
    ctx->local_comm = malloc(sizeof(MPI_Comm));
    if (ctx->local_comm) {
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, (MPI_Comm*)ctx->local_comm);
        MPI_Comm_rank(*(MPI_Comm*)ctx->local_comm, &ctx->local_rank);
        MPI_Comm_size(*(MPI_Comm*)ctx->local_comm, &ctx->local_size);
    }

    ctx->initialized = 1;

    (void)options;  // Reserved for future use

    return ctx;
}

void mpi_bridge_free(distributed_ctx_t* ctx) {
    if (!ctx) return;

    if (ctx->local_comm) {
        MPI_Comm_free((MPI_Comm*)ctx->local_comm);
        free(ctx->local_comm);
    }

    if (ctx->mpi_comm) {
        free(ctx->mpi_comm);
    }

    free(ctx);
}

void mpi_bridge_finalize(void) {
    if (mpi_finalized) return;

    int already_finalized = 0;
    MPI_Finalized(&already_finalized);

    if (!already_finalized && mpi_initialized_by_us) {
        MPI_Finalize();
        mpi_finalized = 1;
    }
}

// ============================================================================
// CONTEXT QUERIES
// ============================================================================

int mpi_get_rank(const distributed_ctx_t* ctx) {
    return ctx ? ctx->rank : 0;
}

int mpi_get_size(const distributed_ctx_t* ctx) {
    return ctx ? ctx->size : 1;
}

int mpi_is_root(const distributed_ctx_t* ctx) {
    return ctx ? (ctx->rank == 0) : 1;
}

void mpi_get_local_range(const distributed_ctx_t* ctx,
                         uint64_t* start, uint64_t* end) {
    if (!ctx) {
        if (start) *start = 0;
        if (end) *end = 0;
        return;
    }

    if (start) *start = ctx->start_index;
    if (end) *end = ctx->end_index;
}

int mpi_get_owner_rank(const distributed_ctx_t* ctx, uint64_t index) {
    if (!ctx || ctx->size <= 1) return 0;

    // Uniform distribution across ranks
    uint64_t total = 1ULL << ctx->num_qubits;
    uint64_t per_rank = total / ctx->size;

    int rank = (int)(index / per_rank);
    if (rank >= ctx->size) rank = ctx->size - 1;

    return rank;
}

int mpi_is_local_index(const distributed_ctx_t* ctx, uint64_t index) {
    if (!ctx) return 1;
    return (index >= ctx->start_index && index < ctx->end_index);
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

mpi_bridge_error_t mpi_barrier(distributed_ctx_t* ctx) {
    if (!ctx) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Barrier(*(MPI_Comm*)ctx->mpi_comm);
    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_SYNC;
}

mpi_bridge_error_t mpi_barrier_timeout(distributed_ctx_t* ctx,
                                       uint32_t timeout_ms) {
    if (!ctx) return MPI_BRIDGE_ERROR_INIT;

    // Use non-blocking barrier with polling for timeout support
    MPI_Request request;
    int err = MPI_Ibarrier(*(MPI_Comm*)ctx->mpi_comm, &request);
    if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_SYNC;

    // Poll with timeout
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int completed = 0;
    while (!completed) {
        MPI_Test(&request, &completed, MPI_STATUS_IGNORE);

        if (completed) break;

        // Check timeout
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t elapsed_ms = (now.tv_sec - start.tv_sec) * 1000 +
                              (now.tv_nsec - start.tv_nsec) / 1000000;

        if (elapsed_ms >= timeout_ms) {
            // Timeout: cancel the request
            MPI_Cancel(&request);
            MPI_Request_free(&request);
            return MPI_BRIDGE_ERROR_TIMEOUT;
        }

        // Brief sleep to avoid busy-waiting (100 microseconds)
        struct timespec sleep_time = {0, 100000};
        nanosleep(&sleep_time, NULL);
    }

    return MPI_BRIDGE_SUCCESS;
}

// ============================================================================
// POINT-TO-POINT COMMUNICATION
// ============================================================================

mpi_bridge_error_t mpi_send_amplitudes(distributed_ctx_t* ctx,
                                       const void* data, uint64_t count,
                                       int dest, int tag) {
    if (!ctx || !data) return MPI_BRIDGE_ERROR_INIT;

    // MPI uses int for count, handle large transfers in chunks
    uint64_t remaining = count * 16;  // 16 bytes per complex double
    const char* ptr = (const char*)data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Send(ptr, chunk, MPI_BYTE, dest, tag,
                           *(MPI_Comm*)ctx->mpi_comm);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

mpi_bridge_error_t mpi_recv_amplitudes(distributed_ctx_t* ctx,
                                       void* data, uint64_t count,
                                       int source, int tag) {
    if (!ctx || !data) return MPI_BRIDGE_ERROR_INIT;

    uint64_t remaining = count * 16;
    char* ptr = (char*)data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Recv(ptr, chunk, MPI_BYTE, source, tag,
                           *(MPI_Comm*)ctx->mpi_comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

mpi_bridge_error_t mpi_exchange_amplitudes(distributed_ctx_t* ctx,
                                           const void* send_data,
                                           void* recv_data,
                                           uint64_t count,
                                           int partner, int tag) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    uint64_t remaining = count * 16;
    const char* send_ptr = (const char*)send_data;
    char* recv_ptr = (char*)recv_data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Sendrecv(send_ptr, chunk, MPI_BYTE, partner, tag,
                               recv_ptr, chunk, MPI_BYTE, partner, tag,
                               *(MPI_Comm*)ctx->mpi_comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        send_ptr += chunk;
        recv_ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

// ============================================================================
// COLLECTIVE OPERATIONS
// ============================================================================

mpi_bridge_error_t mpi_broadcast(distributed_ctx_t* ctx,
                                 void* data, size_t count, int root) {
    if (!ctx || !data) return MPI_BRIDGE_ERROR_INIT;

    size_t remaining = count;
    char* ptr = (char*)data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Bcast(ptr, chunk, MPI_BYTE, root,
                            *(MPI_Comm*)ctx->mpi_comm);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

mpi_bridge_error_t mpi_allreduce_sum_complex(distributed_ctx_t* ctx,
                                             const void* send_data,
                                             void* recv_data,
                                             uint64_t count) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    // Complex double = 2 doubles
    int err = MPI_Allreduce(send_data, recv_data, (int)(count * 2),
                            MPI_DOUBLE, MPI_SUM, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_allreduce_sum_double(distributed_ctx_t* ctx,
                                            const double* send_data,
                                            double* recv_data,
                                            uint64_t count) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Allreduce(send_data, recv_data, (int)count,
                            MPI_DOUBLE, MPI_SUM, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_gather(distributed_ctx_t* ctx,
                              const void* send_data, size_t send_count,
                              void* recv_data, int root) {
    if (!ctx || !send_data) return MPI_BRIDGE_ERROR_INIT;
    if (ctx->rank == root && !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Gather(send_data, (int)send_count, MPI_BYTE,
                         recv_data, (int)send_count, MPI_BYTE,
                         root, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_allgather(distributed_ctx_t* ctx,
                                 const void* send_data, size_t send_count,
                                 void* recv_data) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Allgather(send_data, (int)send_count, MPI_BYTE,
                            recv_data, (int)send_count, MPI_BYTE,
                            *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_scatter(distributed_ctx_t* ctx,
                               const void* send_data,
                               void* recv_data,
                               size_t recv_count,
                               int root) {
    if (!ctx || !recv_data) return MPI_BRIDGE_ERROR_INIT;
    if (ctx->rank == root && !send_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Scatter(send_data, (int)recv_count, MPI_BYTE,
                          recv_data, (int)recv_count, MPI_BYTE,
                          root, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_alltoall_int(distributed_ctx_t* ctx,
                                    const int* send_data,
                                    int* recv_data,
                                    int count) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Alltoall(send_data, count, MPI_INT,
                           recv_data, count, MPI_INT,
                           *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_send(distributed_ctx_t* ctx,
                            const void* data, size_t count,
                            int dest, int tag) {
    if (!ctx || !data) return MPI_BRIDGE_ERROR_INIT;

    size_t remaining = count;
    const char* ptr = (const char*)data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Send(ptr, chunk, MPI_BYTE, dest, tag,
                           *(MPI_Comm*)ctx->mpi_comm);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

mpi_bridge_error_t mpi_recv(distributed_ctx_t* ctx,
                            void* data, size_t count,
                            int source, int tag) {
    if (!ctx || !data) return MPI_BRIDGE_ERROR_INIT;

    size_t remaining = count;
    char* ptr = (char*)data;

    while (remaining > 0) {
        int chunk = (remaining > INT32_MAX) ? INT32_MAX : (int)remaining;
        int err = MPI_Recv(ptr, chunk, MPI_BYTE, source, tag,
                           *(MPI_Comm*)ctx->mpi_comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) return MPI_BRIDGE_ERROR_COMM;
        ptr += chunk;
        remaining -= chunk;
    }

    return MPI_BRIDGE_SUCCESS;
}

mpi_bridge_error_t mpi_allreduce_max_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Allreduce(send_data, recv_data, (int)count,
                            MPI_UINT64_T, MPI_MAX, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

mpi_bridge_error_t mpi_allreduce_min_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count) {
    if (!ctx || !send_data || !recv_data) return MPI_BRIDGE_ERROR_INIT;

    int err = MPI_Allreduce(send_data, recv_data, (int)count,
                            MPI_UINT64_T, MPI_MIN, *(MPI_Comm*)ctx->mpi_comm);

    return (err == MPI_SUCCESS) ? MPI_BRIDGE_SUCCESS : MPI_BRIDGE_ERROR_COMM;
}

// ============================================================================
// UTILITIES
// ============================================================================

void mpi_print_context_info(const distributed_ctx_t* ctx, int all_ranks) {
    if (!ctx) return;

    if (all_ranks || ctx->rank == 0) {
        printf("[Rank %d/%d] %s\n", ctx->rank, ctx->size, ctx->processor_name);
        printf("  Local rank: %d/%d\n", ctx->local_rank, ctx->local_size);
        printf("  Thread support: %d\n", ctx->thread_support);
        printf("  Index range: [%llu, %llu)\n",
               (unsigned long long)ctx->start_index,
               (unsigned long long)ctx->end_index);
        printf("  Local count: %llu\n",
               (unsigned long long)ctx->local_count);
    }

    if (all_ranks) {
        MPI_Barrier(*(MPI_Comm*)ctx->mpi_comm);
    }
}

const char* mpi_get_processor_name(const distributed_ctx_t* ctx) {
    return ctx ? ctx->processor_name : "unknown";
}

int mpi_is_distributed(const distributed_ctx_t* ctx) {
    return ctx ? (ctx->size > 1) : 0;
}

const char* mpi_bridge_error_string(mpi_bridge_error_t error) {
    switch (error) {
        case MPI_BRIDGE_SUCCESS: return "Success";
        case MPI_BRIDGE_ERROR_INIT: return "Initialization error";
        case MPI_BRIDGE_ERROR_COMM: return "Communication error";
        case MPI_BRIDGE_ERROR_PARTITION: return "Partition error";
        case MPI_BRIDGE_ERROR_ALLOC: return "Allocation error";
        case MPI_BRIDGE_ERROR_SYNC: return "Synchronization error";
        case MPI_BRIDGE_ERROR_NOT_SUPPORTED: return "Not supported";
        default: return "Unknown error";
    }
}

void mpi_abort(distributed_ctx_t* ctx, int error_code, const char* message) {
    fprintf(stderr, "[MPI ABORT] %s\n", message ? message : "Unknown error");

    if (ctx && ctx->mpi_comm) {
        MPI_Abort(*(MPI_Comm*)ctx->mpi_comm, error_code);
    } else {
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }
}

#else /* !HAS_MPI */

// Stub implementations when MPI is not available

int mpi_is_available(void) { return 0; }

distributed_ctx_t* mpi_bridge_init(int* argc, char*** argv,
                                   const mpi_init_options_t* options) {
    (void)argc; (void)argv; (void)options;
    return NULL;
}

distributed_ctx_t* mpi_bridge_init_no_args(const mpi_init_options_t* options) {
    (void)options;
    return NULL;
}

void mpi_bridge_free(distributed_ctx_t* ctx) { (void)ctx; }
void mpi_bridge_finalize(void) {}

int mpi_get_rank(const distributed_ctx_t* ctx) { (void)ctx; return 0; }
int mpi_get_size(const distributed_ctx_t* ctx) { (void)ctx; return 1; }
int mpi_is_root(const distributed_ctx_t* ctx) { (void)ctx; return 1; }

void mpi_get_local_range(const distributed_ctx_t* ctx,
                         uint64_t* start, uint64_t* end) {
    (void)ctx;
    if (start) *start = 0;
    if (end) *end = 0;
}

int mpi_get_owner_rank(const distributed_ctx_t* ctx, uint64_t index) {
    (void)ctx; (void)index;
    return 0;
}

int mpi_is_local_index(const distributed_ctx_t* ctx, uint64_t index) {
    (void)ctx; (void)index;
    return 1;
}

mpi_bridge_error_t mpi_barrier(distributed_ctx_t* ctx) {
    (void)ctx;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_barrier_timeout(distributed_ctx_t* ctx, uint32_t timeout_ms) {
    (void)ctx; (void)timeout_ms;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_send_amplitudes(distributed_ctx_t* ctx,
                                       const void* data, uint64_t count,
                                       int dest, int tag) {
    (void)ctx; (void)data; (void)count; (void)dest; (void)tag;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_recv_amplitudes(distributed_ctx_t* ctx,
                                       void* data, uint64_t count,
                                       int source, int tag) {
    (void)ctx; (void)data; (void)count; (void)source; (void)tag;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_exchange_amplitudes(distributed_ctx_t* ctx,
                                           const void* send_data,
                                           void* recv_data,
                                           uint64_t count,
                                           int partner, int tag) {
    (void)ctx; (void)send_data; (void)recv_data;
    (void)count; (void)partner; (void)tag;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_broadcast(distributed_ctx_t* ctx,
                                 void* data, size_t count, int root) {
    (void)ctx; (void)data; (void)count; (void)root;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_allreduce_sum_complex(distributed_ctx_t* ctx,
                                             const void* send_data,
                                             void* recv_data,
                                             uint64_t count) {
    (void)ctx; (void)send_data; (void)recv_data; (void)count;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_allreduce_sum_double(distributed_ctx_t* ctx,
                                            const double* send_data,
                                            double* recv_data,
                                            uint64_t count) {
    (void)ctx; (void)send_data; (void)recv_data; (void)count;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_gather(distributed_ctx_t* ctx,
                              const void* send_data, size_t send_count,
                              void* recv_data, int root) {
    (void)ctx; (void)send_data; (void)send_count; (void)recv_data; (void)root;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_allgather(distributed_ctx_t* ctx,
                                 const void* send_data, size_t send_count,
                                 void* recv_data) {
    (void)ctx; (void)send_data; (void)send_count; (void)recv_data;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_scatter(distributed_ctx_t* ctx,
                               const void* send_data,
                               void* recv_data,
                               size_t recv_count,
                               int root) {
    (void)ctx; (void)send_data; (void)recv_data; (void)recv_count; (void)root;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_alltoall_int(distributed_ctx_t* ctx,
                                    const int* send_data,
                                    int* recv_data,
                                    int count) {
    (void)ctx; (void)send_data; (void)recv_data; (void)count;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_send(distributed_ctx_t* ctx,
                            const void* data, size_t count,
                            int dest, int tag) {
    (void)ctx; (void)data; (void)count; (void)dest; (void)tag;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_recv(distributed_ctx_t* ctx,
                            void* data, size_t count,
                            int source, int tag) {
    (void)ctx; (void)data; (void)count; (void)source; (void)tag;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_allreduce_max_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count) {
    (void)ctx; (void)send_data; (void)recv_data; (void)count;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

mpi_bridge_error_t mpi_allreduce_min_uint64(distributed_ctx_t* ctx,
                                             const uint64_t* send_data,
                                             uint64_t* recv_data,
                                             uint64_t count) {
    (void)ctx; (void)send_data; (void)recv_data; (void)count;
    return MPI_BRIDGE_ERROR_NOT_SUPPORTED;
}

void mpi_print_context_info(const distributed_ctx_t* ctx, int all_ranks) {
    (void)ctx; (void)all_ranks;
    printf("MPI not available\n");
}

const char* mpi_get_processor_name(const distributed_ctx_t* ctx) {
    (void)ctx;
    return "localhost";
}

int mpi_is_distributed(const distributed_ctx_t* ctx) {
    (void)ctx;
    return 0;
}

const char* mpi_bridge_error_string(mpi_bridge_error_t error) {
    (void)error;
    return "MPI not available";
}

void mpi_abort(distributed_ctx_t* ctx, int error_code, const char* message) {
    (void)ctx;
    fprintf(stderr, "[ABORT] %s (code %d)\n",
            message ? message : "Unknown error", error_code);
    exit(error_code);
}

#endif /* HAS_MPI */
