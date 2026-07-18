/**
 * @file  large_state_ghz_gpu.c
 * @brief v1.1 follow-up #6 demo: sharded MPI + CUDA GHZ state.
 *
 * Same workload as examples/distributed/large_state_ghz.c, but each
 * rank's local shard lives on a CUDA GPU.  Demonstrates the new
 * partition_state_create_gpu / partition_sync_* path:
 *   - dist_hadamard(0) is a "local" gate (target = local qubit on
 *     every rank) and dispatches entirely on GPU with NO MPI.
 *   - dist_cnot(0, k) for k in the partition region requires an
 *     MPI exchange; the new path syncs GPU->host before send and
 *     host->GPU after recv.
 *
 * Run on a CUDA-equipped MPI host:
 *   mpirun -n 2 large_state_ghz_gpu [N]   (default N=10, must fit
 *                                          on a single GPU per rank)
 *
 * For multi-host setups CUDA must be installed on every rank.
 */

#include "../../src/distributed/state_partition.h"
#include "../../src/distributed/distributed_gates.h"
#include "../../src/distributed/mpi_bridge.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char host[256];
    int device_id;
} gpu_rank_record_t;

static uint32_t partition_bits_for_size(int size)
{
    uint32_t bits = 0;
    while (size > 1) {
        size >>= 1;
        ++bits;
    }
    return bits;
}

static int count_unique_hosts(const gpu_rank_record_t *records, int count)
{
    int unique = 0;
    for (int i = 0; i < count; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) {
            if (strcmp(records[i].host, records[j].host) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) ++unique;
    }
    return unique;
}

static int count_unique_gpu_endpoints(const gpu_rank_record_t *records, int count)
{
    int unique = 0;
    for (int i = 0; i < count; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) {
            if (records[i].device_id == records[j].device_id &&
                strcmp(records[i].host, records[j].host) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) ++unique;
    }
    return unique;
}

static int has_two_hosts_with_two_ranks_each(const gpu_rank_record_t *records,
                                              int count)
{
    if (count != 4 || count_unique_hosts(records, count) != 2) return 0;

    for (int i = 0; i < count; ++i) {
        int host_ranks = 0;
        int seen = 0;
        for (int j = 0; j < i; ++j) {
            if (strcmp(records[i].host, records[j].host) == 0) {
                seen = 1;
                break;
            }
        }
        if (seen) continue;

        for (int j = 0; j < count; ++j) {
            if (strcmp(records[i].host, records[j].host) == 0) ++host_ranks;
        }
        if (host_ranks != 2) return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    distributed_ctx_t *ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) return 1;

    const int rank = mpi_get_rank(ctx);
    const int size = mpi_get_size(ctx);
    const uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 10;

    if (size < 2 || (size & (size - 1)) != 0 || N > 50 ||
        N <= partition_bits_for_size(size)) {
        if (rank == 0) {
            fprintf(stderr,
                "requires a power-of-two MPI size >= 2 and partitionable N <= 50\n");
        }
        mpi_bridge_free(ctx);
        mpi_bridge_finalize();
        return 2;
    }

    const uint32_t partition_bits = partition_bits_for_size(size);
    const uint32_t local_qubits = N - partition_bits;
    const uint64_t local_count = 1ULL << local_qubits;

    /* This GHZ circuit's only remote operation is CNOT(control=0,
     * target=partition-qubit).  Use a tiny fixed halo so the example can run
     * under the N=33 proof on two-node / four-rank style hardware profiles
     * after the kernel-side chunked exchange landed.
     *
     * 1<<26 complex amplitudes = 1 GiB per buffer.  This leaves ample host
     * memory headroom at N=33 while avoiding thousands of cross-zone MPI
     * round trips for each partition-boundary CNOT. */
    const size_t comm_halo_elements = 1ULL << 26;
    partition_config_t config = {0};
    config.use_aligned_memory = 1;
    config.comm_buffer_size = (size_t)comm_halo_elements * sizeof(double complex);

    if (rank == 0) {
        fprintf(stdout,
            "=== Sharded MPI+CUDA GHZ demo: N=%u across %d ranks ===\n", N, size);
        fprintf(stdout, "    total amplitudes: 2^%u = %llu\n", N, (1ULL << N));
        fprintf(stdout, "    per-rank GPU mem: %.2f MB\n",
                (double)((1ULL << N) * sizeof(double complex)) /
                ((double)size * 1024.0 * 1024.0));
        fprintf(stdout, "    per-rank host staging+halos: %.2f MB\n",
                (double)(local_count * sizeof(double complex) +
                         2 * config.comm_buffer_size) /
                (1024.0 * 1024.0));
        fprintf(stdout, "    configured comm buffer: %.2f MB\n",
                (double)(config.comm_buffer_size) / (1024.0 * 1024.0));
    }

    partitioned_state_t *state = partition_state_create_gpu(ctx, N, &config);
    uint64_t create_failed = state ? 0 : 1;
    uint64_t any_create_failed = 0;
    if (mpi_allreduce_max_uint64(ctx, &create_failed,
                                 &any_create_failed, 1) != MPI_BRIDGE_SUCCESS) {
        any_create_failed = 1;
    }
    if (any_create_failed) {
        if (rank == 0) {
            fprintf(stderr,
                "partition_state_create_gpu failed (no CUDA, no GPU, or "
                "non-CUDA libquantumsim build)\n");
        }
        partition_state_free(state);
        mpi_bridge_free(ctx);
        mpi_bridge_finalize();
        return 3;
    }

    fprintf(stdout,
            "    rank %d/%d host=%s local_rank=%d CUDA device=%d/%d "
            "shard=[%llu,%llu)\n",
            rank, size, ctx->processor_name, ctx->local_rank,
            state->gpu_device_id, state->gpu_device_count,
            (unsigned long long)state->local_start,
            (unsigned long long)state->local_end);

    gpu_rank_record_t record = {{0}, state->gpu_device_id};
    strncpy(record.host, ctx->processor_name, sizeof(record.host) - 1);
    gpu_rank_record_t *records = NULL;
    if (rank == 0) {
        records = (gpu_rank_record_t *)calloc((size_t)size, sizeof(*records));
    }
    uint64_t record_alloc_failed = (rank == 0 && !records) ? 1 : 0;
    uint64_t any_record_alloc_failed = 0;
    if (mpi_allreduce_max_uint64(ctx, &record_alloc_failed,
                                 &any_record_alloc_failed, 1) != MPI_BRIDGE_SUCCESS ||
        any_record_alloc_failed) {
        if (rank == 0) fprintf(stderr, "FAIL: could not allocate GPU rank map\n");
        free(records);
        partition_state_free(state);
        mpi_bridge_free(ctx);
        mpi_bridge_finalize();
        return 4;
    }
    uint64_t gather_failed =
        mpi_gather(ctx, &record, sizeof(record), records, 0) == MPI_BRIDGE_SUCCESS
        ? 0 : 1;
    uint64_t any_gather_failed = 0;
    if (mpi_allreduce_max_uint64(ctx, &gather_failed,
                                 &any_gather_failed, 1) != MPI_BRIDGE_SUCCESS) {
        any_gather_failed = 1;
    }
    if (any_gather_failed) {
        if (rank == 0) fprintf(stderr, "FAIL: could not gather GPU rank mapping\n");
        free(records);
        partition_state_free(state);
        mpi_bridge_free(ctx);
        mpi_bridge_finalize();
        return 4;
    }

    const uint32_t halo_swaps = partition_bits;

    const clock_t t0 = clock();

    /* GHZ: H_0 then CNOT(0, k) for k=1..N-1.  When k >= local_qubits
     * the CNOT spans the partition boundary and triggers the MPI
     * exchange path with GPU<->host sync wrapped around it. */
    if (dist_hadamard(state, 0) != DIST_GATE_SUCCESS) goto fail;
    for (uint32_t k = 1; k < N; k++) {
        if (dist_cnot(state, 0, k) != DIST_GATE_SUCCESS) goto fail;
    }

    /* Pull final GPU shard back to host for inspection. */
    if (partition_sync_to_host(state) != PARTITION_SUCCESS) goto fail;

    const clock_t t1 = clock();
    const double sim_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    /* Same verification as the CPU-only demo. */
    const uint64_t last_idx = (1ULL << N) - 1;
    const int owner_last = partition_get_owner(state, last_idx);
    double mN1_local = 0.0;
    if (rank == owner_last) {
        const double complex aN1 = partition_get_amplitude(state, last_idx);
        mN1_local = creal(aN1) * creal(aN1) + cimag(aN1) * cimag(aN1);
    }
    double mN1_global = 0.0;
    uint64_t verify_failed =
        mpi_allreduce_sum_double(ctx, &mN1_local, &mN1_global, 1) == MPI_BRIDGE_SUCCESS
        ? 0 : 1;

    if (rank == 0) {
        const double complex a0 = partition_get_amplitude(state, 0);
        const double m0 = creal(a0) * creal(a0) + cimag(a0) * cimag(a0);
        const int hosts = count_unique_hosts(records, size);
        const int gpu_endpoints = count_unique_gpu_endpoints(records, size);
        const int host_slots_2x2 =
            has_two_hosts_with_two_ranks_each(records, size);
        fprintf(stdout,
            "\n    simulation time:  %.4f s (wall, rank 0)\n", sim_s);
        fprintf(stdout, "    P(|0...0>) = %.6f  (rank 0)\n", m0);
        fprintf(stdout, "    P(|1...1>) = %.6f  (rank %d, reduced)\n",
                mN1_global, owner_last);
        fprintf(stdout, "    expected   = 0.500000 each\n");

        if (fabs(m0 - 0.5) < 1e-9 && fabs(mN1_global - 0.5) < 1e-9) {
            fprintf(stdout,
                "\n    GHZ verified: sharded MPI+CUDA at N=%u, %d ranks.\n",
                N, size);
            fprintf(stdout,
                "    Local gates ran on GPU; cross-partition CNOTs used\n"
                "    the GPU->host->MPI->host->GPU sync path.\n");
            fprintf(stdout,
                "MOONLAB_MPI_SHARDED_GPU PASS n=%u ranks=%d hosts=%d "
                "host_slots_2x2=%d gpu_endpoints=%d halo_swaps=%u "
                "local_qubits=%u\n",
                N, size, hosts, host_slots_2x2, gpu_endpoints, halo_swaps,
                local_qubits);
        } else {
            fprintf(stderr, "\n    FAIL: GHZ amplitudes off\n");
            verify_failed = 1;
        }
    }

    uint64_t any_verify_failed = 0;
    if (mpi_allreduce_max_uint64(ctx, &verify_failed,
                                 &any_verify_failed, 1) != MPI_BRIDGE_SUCCESS) {
        any_verify_failed = 1;
    }

    free(records);
    partition_state_free(state);
    mpi_barrier(ctx);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return any_verify_failed ? 5 : 0;

fail:
    if (rank == 0) fprintf(stderr, "FAIL during dist_* call\n");
    partition_state_free(state);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();
    return 6;
}
