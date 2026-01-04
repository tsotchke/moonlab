/**
 * @file bench_tensor_networks.c
 * @brief Performance benchmarks for tensor network operations
 *
 * Measures performance of:
 * - MPS creation and manipulation
 * - SVD truncation
 * - DMRG sweeps
 * - Entanglement entropy calculation
 *
 * @stability stable
 * @since v1.0.0
 */

#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"
#include "../../src/algorithms/tensor_network/tn_measurement.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define WARMUP_ITERATIONS 2
#define BENCH_ITERATIONS 5

typedef struct {
    double min_us;
    double max_us;
    double avg_us;
    double std_us;
} bench_result_t;

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static bench_result_t run_benchmark(void (*func)(void *), void *data, int iterations) {
    double times[BENCH_ITERATIONS];

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        func(data);
    }

    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        func(data);
        double end = get_time_us();
        times[i] = end - start;
    }

    bench_result_t result;
    result.min_us = times[0];
    result.max_us = times[0];
    double sum = 0.0;

    for (int i = 0; i < iterations; i++) {
        if (times[i] < result.min_us) result.min_us = times[i];
        if (times[i] > result.max_us) result.max_us = times[i];
        sum += times[i];
    }

    result.avg_us = sum / iterations;

    double sum_sq = 0.0;
    for (int i = 0; i < iterations; i++) {
        double diff = times[i] - result.avg_us;
        sum_sq += diff * diff;
    }
    result.std_us = sqrt(sum_sq / iterations);

    return result;
}

static void print_result(const char *name, bench_result_t *r) {
    if (r->avg_us >= 1e6) {
        printf("  %-40s %10.2f s  (±%.2f)\n",
               name, r->avg_us / 1e6, r->std_us / 1e6);
    } else if (r->avg_us >= 1e3) {
        printf("  %-40s %10.2f ms (±%.2f)\n",
               name, r->avg_us / 1e3, r->std_us / 1e3);
    } else {
        printf("  %-40s %10.2f μs (±%.2f)\n",
               name, r->avg_us, r->std_us);
    }
}

// ============================================================================
// MPS CREATION BENCHMARKS
// ============================================================================

typedef struct {
    uint32_t num_sites;
    uint32_t bond_dim;
    tn_mps_t *mps;
} mps_data_t;

static void bench_mps_create(void *data) {
    mps_data_t *d = (mps_data_t *)data;
    d->mps = tn_mps_create(d->num_sites, d->bond_dim);
    tn_mps_free(d->mps);
}

static void benchmark_mps_creation(void) {
    printf("\n=== MPS Creation Benchmarks ===\n");

    int site_counts[] = {20, 50, 100, 200};
    int bond_dims[] = {16, 32, 64};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            mps_data_t data = {
                .num_sites = site_counts[i],
                .bond_dim = bond_dims[j]
            };

            char name[64];
            snprintf(name, sizeof(name), "MPS %d sites, χ=%d",
                     site_counts[i], bond_dims[j]);

            bench_result_t r = run_benchmark(bench_mps_create, &data, BENCH_ITERATIONS);
            print_result(name, &r);
        }
    }
}

// ============================================================================
// GATE APPLICATION BENCHMARKS
// ============================================================================

typedef struct {
    tn_mps_t *mps;
    uint32_t site;
} gate_data_t;

static void bench_tn_hadamard(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    tn_apply_h(d->mps, d->site);
}

static void bench_tn_cnot(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    tn_apply_cnot(d->mps, d->site, d->site + 1);
}

static void benchmark_tn_gates(void) {
    printf("\n=== MPS Gate Application Benchmarks ===\n");

    uint32_t num_sites = 50;
    uint32_t bond_dim = 32;

    tn_mps_t *mps = tn_mps_create(num_sites, bond_dim);
    tn_mps_init_zero(mps);

    gate_data_t data = {.mps = mps, .site = num_sites / 2};

    bench_result_t r;

    r = run_benchmark(bench_tn_hadamard, &data, BENCH_ITERATIONS);
    print_result("Hadamard on MPS (50 sites, χ=32)", &r);

    r = run_benchmark(bench_tn_cnot, &data, BENCH_ITERATIONS);
    print_result("CNOT on MPS (50 sites, χ=32)", &r);

    tn_mps_free(mps);
}

// ============================================================================
// ENTANGLEMENT ENTROPY BENCHMARKS
// ============================================================================

typedef struct {
    tn_mps_t *mps;
    uint32_t bond;
} entropy_data_t;

static void bench_entropy(void *data) {
    entropy_data_t *d = (entropy_data_t *)data;
    tn_mps_entanglement_entropy(d->mps, d->bond);
}

static void benchmark_entropy(void) {
    printf("\n=== Entanglement Entropy Benchmarks ===\n");

    int site_counts[] = {20, 50, 100};
    int bond_dims[] = {16, 32, 64};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            tn_mps_t *mps = tn_mps_create(site_counts[i], bond_dims[j]);
            tn_mps_init_zero(mps);

            // Create some entanglement
            for (uint32_t s = 0; s < site_counts[i]; s++) {
                tn_apply_h(mps, s);
            }
            for (uint32_t s = 0; s < site_counts[i] - 1; s++) {
                tn_apply_cnot(mps, s, s + 1);
            }

            entropy_data_t data = {
                .mps = mps,
                .bond = site_counts[i] / 2
            };

            char name[64];
            snprintf(name, sizeof(name), "Entropy %d sites, χ=%d",
                     site_counts[i], bond_dims[j]);

            bench_result_t r = run_benchmark(bench_entropy, &data, BENCH_ITERATIONS);
            print_result(name, &r);

            tn_mps_free(mps);
        }
    }
}

// ============================================================================
// DMRG BENCHMARKS
// ============================================================================

typedef struct {
    tn_mps_t *mps;
    dmrg_config_t config;
} dmrg_data_t;

static void bench_dmrg(void *data) {
    dmrg_data_t *d = (dmrg_data_t *)data;
    dmrg_ground_state(d->mps, &d->config);
}

static void benchmark_dmrg(void) {
    printf("\n=== DMRG Benchmarks ===\n");

    int site_counts[] = {10, 16, 20};
    int bond_dims[] = {16, 32};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            tn_mps_t *mps = tn_mps_create(site_counts[i], bond_dims[j]);
            tn_mps_init_zero(mps);

            dmrg_data_t data = {
                .mps = mps,
                .config = {
                    .num_sites = site_counts[i],
                    .bond_dim = bond_dims[j],
                    .max_sweeps = 5,
                    .convergence_threshold = 1e-6,
                    .J = 1.0,
                    .delta = 1.0,
                    .h_field = 0.0
                }
            };

            char name[64];
            snprintf(name, sizeof(name), "DMRG %d sites, χ=%d (5 sweeps)",
                     site_counts[i], bond_dims[j]);

            bench_result_t r = run_benchmark(bench_dmrg, &data, 3);  // Fewer iterations for slow benchmark
            print_result(name, &r);

            tn_mps_free(mps);
        }
    }
}

// ============================================================================
// SCALING BENCHMARKS
// ============================================================================

static void benchmark_scaling(void) {
    printf("\n=== Bond Dimension Scaling ===\n");

    uint32_t num_sites = 30;
    int bond_dims[] = {8, 16, 32, 64, 128};

    for (int j = 0; j < 5; j++) {
        tn_mps_t *mps = tn_mps_create(num_sites, bond_dims[j]);
        tn_mps_init_zero(mps);

        // Time a sequence of gates
        double start = get_time_us();

        for (int iter = 0; iter < 10; iter++) {
            for (uint32_t s = 0; s < num_sites; s++) {
                tn_apply_h(mps, s);
            }
            for (uint32_t s = 0; s < num_sites - 1; s++) {
                tn_apply_cnot(mps, s, s + 1);
            }
        }

        double end = get_time_us();
        double time_ms = (end - start) / 1000.0;

        printf("  χ=%3d: 10 layers on %d sites in %.2f ms\n",
               bond_dims[j], num_sites, time_ms);

        tn_mps_free(mps);
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       TENSOR NETWORK PERFORMANCE BENCHMARKS                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Measuring MPS, DMRG, and entropy operations                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    benchmark_mps_creation();
    benchmark_tn_gates();
    benchmark_entropy();
    benchmark_dmrg();
    benchmark_scaling();

    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Benchmarks complete\n");
    printf("════════════════════════════════════════════════════════════════\n");

    return 0;
}
