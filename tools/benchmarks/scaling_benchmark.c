/**
 * @file scaling_benchmark.c
 * @brief Comprehensive scaling benchmark tool
 *
 * Measures how the simulator scales with:
 * - Number of qubits (memory and time)
 * - Circuit depth
 * - Bond dimension (for tensor networks)
 *
 * Outputs results in CSV format for analysis.
 *
 * Usage: ./scaling_benchmark [options]
 *   --qubits MIN MAX    Range of qubit counts (default: 4 20)
 *   --depth MIN MAX     Range of circuit depths (default: 10 100)
 *   --bond MIN MAX      Range of bond dimensions (default: 8 64)
 *   --output FILE       Output CSV file (default: benchmark_results.csv)
 *   --type TYPE         Benchmark type: state, tensor, both (default: both)
 *
 * @stability stable
 * @since v1.0.0
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include <math.h>

#define DEFAULT_MIN_QUBITS 4
#define DEFAULT_MAX_QUBITS 20
#define DEFAULT_MIN_DEPTH 10
#define DEFAULT_MAX_DEPTH 100
#define DEFAULT_MIN_BOND 8
#define DEFAULT_MAX_BOND 64

typedef struct {
    int min_qubits;
    int max_qubits;
    int min_depth;
    int max_depth;
    int min_bond;
    int max_bond;
    char output_file[256];
    int bench_state;
    int bench_tensor;
} config_t;

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static double get_memory_mb(size_t num_qubits) {
    size_t state_dim = 1ULL << num_qubits;
    return (state_dim * sizeof(double _Complex)) / (1024.0 * 1024.0);
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --qubits MIN MAX    Range of qubit counts (default: %d %d)\n",
           DEFAULT_MIN_QUBITS, DEFAULT_MAX_QUBITS);
    printf("  --depth MIN MAX     Range of circuit depths (default: %d %d)\n",
           DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH);
    printf("  --bond MIN MAX      Range of bond dimensions (default: %d %d)\n",
           DEFAULT_MIN_BOND, DEFAULT_MAX_BOND);
    printf("  --output FILE       Output CSV file (default: benchmark_results.csv)\n");
    printf("  --type TYPE         Benchmark type: state, tensor, both (default: both)\n");
    printf("  --help              Show this help\n");
}

static void parse_args(int argc, char **argv, config_t *config) {
    config->min_qubits = DEFAULT_MIN_QUBITS;
    config->max_qubits = DEFAULT_MAX_QUBITS;
    config->min_depth = DEFAULT_MIN_DEPTH;
    config->max_depth = DEFAULT_MAX_DEPTH;
    config->min_bond = DEFAULT_MIN_BOND;
    config->max_bond = DEFAULT_MAX_BOND;
    strcpy(config->output_file, "benchmark_results.csv");
    config->bench_state = 1;
    config->bench_tensor = 1;

    static struct option long_options[] = {
        {"qubits", required_argument, 0, 'q'},
        {"depth", required_argument, 0, 'd'},
        {"bond", required_argument, 0, 'b'},
        {"output", required_argument, 0, 'o'},
        {"type", required_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "q:d:b:o:t:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'q':
                sscanf(optarg, "%d", &config->min_qubits);
                if (optind < argc && argv[optind][0] != '-') {
                    config->max_qubits = atoi(argv[optind++]);
                }
                break;
            case 'd':
                sscanf(optarg, "%d", &config->min_depth);
                if (optind < argc && argv[optind][0] != '-') {
                    config->max_depth = atoi(argv[optind++]);
                }
                break;
            case 'b':
                sscanf(optarg, "%d", &config->min_bond);
                if (optind < argc && argv[optind][0] != '-') {
                    config->max_bond = atoi(argv[optind++]);
                }
                break;
            case 'o':
                strncpy(config->output_file, optarg, sizeof(config->output_file) - 1);
                break;
            case 't':
                if (strcmp(optarg, "state") == 0) {
                    config->bench_tensor = 0;
                } else if (strcmp(optarg, "tensor") == 0) {
                    config->bench_state = 0;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                break;
        }
    }
}

/**
 * @brief Apply random circuit layer
 */
static void apply_random_layer(quantum_state_t *state) {
    size_t n = state->num_qubits;

    // Single-qubit rotations
    for (size_t q = 0; q < n; q++) {
        gate_rz(state, q, (double)rand() / RAND_MAX * 2 * M_PI);
        gate_ry(state, q, (double)rand() / RAND_MAX * M_PI);
    }

    // Entangling layer
    for (size_t q = 0; q < n - 1; q += 2) {
        gate_cnot(state, q, q + 1);
    }
    for (size_t q = 1; q < n - 1; q += 2) {
        gate_cnot(state, q, q + 1);
    }
}

/**
 * @brief Benchmark state vector simulation
 */
static void benchmark_state_vector(config_t *config, FILE *out) {
    printf("\n=== State Vector Scaling ===\n");
    fprintf(out, "# State Vector Benchmarks\n");
    fprintf(out, "qubits,memory_mb,init_us,layer_us,depth_100_ms\n");

    for (int n = config->min_qubits; n <= config->max_qubits; n += 2) {
        double mem_mb = get_memory_mb(n);

        if (mem_mb > 16000) {  // Skip if > 16GB
            printf("  %2d qubits: Skipped (%.1f GB required)\n", n, mem_mb / 1024);
            continue;
        }

        // Benchmark initialization
        double start = get_time_us();
        quantum_state_t state;
        qs_error_t err = quantum_state_init(&state, n);
        double init_time = get_time_us() - start;

        if (err != QS_SUCCESS) {
            printf("  %2d qubits: Failed to allocate\n", n);
            continue;
        }

        // Benchmark single layer
        start = get_time_us();
        apply_random_layer(&state);
        double layer_time = get_time_us() - start;

        // Benchmark 100 layers
        quantum_state_reset(&state);
        start = get_time_us();
        for (int d = 0; d < 100; d++) {
            apply_random_layer(&state);
        }
        double depth_100_time = (get_time_us() - start) / 1000.0;  // ms

        printf("  %2d qubits: mem=%.1f MB, init=%.0f μs, layer=%.0f μs, 100 layers=%.1f ms\n",
               n, mem_mb, init_time, layer_time, depth_100_time);

        fprintf(out, "%d,%.2f,%.2f,%.2f,%.2f\n",
                n, mem_mb, init_time, layer_time, depth_100_time);

        quantum_state_free(&state);
    }
}

/**
 * @brief Benchmark tensor network simulation
 */
static void benchmark_tensor_network(config_t *config, FILE *out) {
    printf("\n=== Tensor Network Scaling ===\n");
    fprintf(out, "\n# Tensor Network Benchmarks\n");
    fprintf(out, "sites,bond_dim,init_us,layer_us,depth_100_ms\n");

    int site_counts[] = {20, 50, 100, 200};
    int num_sites_options = 4;

    for (int s = 0; s < num_sites_options; s++) {
        int num_sites = site_counts[s];

        for (int chi = config->min_bond; chi <= config->max_bond; chi *= 2) {
            // Create config with desired bond dimension
            tn_state_config_t tn_config = tn_state_config_create(chi, 1e-12);

            // Benchmark initialization
            double start = get_time_us();
            tn_mps_state_t *mps = tn_mps_create_zero(num_sites, &tn_config);
            double init_time = get_time_us() - start;

            if (!mps) {
                printf("  %3d sites, χ=%3d: Failed\n", num_sites, chi);
                continue;
            }

            // Benchmark single layer
            start = get_time_us();
            for (int q = 0; q < num_sites; q++) {
                tn_apply_h(mps, q);
            }
            for (int q = 0; q < num_sites - 1; q++) {
                tn_apply_cnot(mps, q, q + 1);
            }
            double layer_time = get_time_us() - start;

            // Reset and benchmark 100 layers
            tn_mps_free(mps);
            mps = tn_mps_create_zero(num_sites, &tn_config);
            start = get_time_us();
            for (int d = 0; d < 100; d++) {
                for (int q = 0; q < num_sites; q++) {
                    tn_apply_h(mps, q);
                }
                for (int q = 0; q < num_sites - 1; q++) {
                    tn_apply_cnot(mps, q, q + 1);
                }
            }
            double depth_100_time = (get_time_us() - start) / 1000.0;  // ms

            printf("  %3d sites, χ=%3d: init=%.0f μs, layer=%.0f μs, 100 layers=%.1f ms\n",
                   num_sites, chi, init_time, layer_time, depth_100_time);

            fprintf(out, "%d,%d,%.2f,%.2f,%.2f\n",
                    num_sites, chi, init_time, layer_time, depth_100_time);

            tn_mps_free(mps);
        }
    }
}

/**
 * @brief Benchmark depth scaling
 */
static void benchmark_depth_scaling(config_t *config, FILE *out) {
    printf("\n=== Depth Scaling ===\n");
    fprintf(out, "\n# Depth Scaling Benchmarks\n");
    fprintf(out, "qubits,depth,time_ms\n");

    int qubit_counts[] = {10, 12, 14, 16};
    int depths[] = {10, 25, 50, 100, 200, 500};

    for (int q = 0; q < 4; q++) {
        int n = qubit_counts[q];
        quantum_state_t state;
        quantum_state_init(&state, n);

        printf("  %d qubits:\n", n);

        for (int d = 0; d < 6; d++) {
            int depth = depths[d];

            quantum_state_reset(&state);

            double start = get_time_us();
            for (int layer = 0; layer < depth; layer++) {
                apply_random_layer(&state);
            }
            double time_ms = (get_time_us() - start) / 1000.0;

            printf("    depth %3d: %.2f ms\n", depth, time_ms);
            fprintf(out, "%d,%d,%.2f\n", n, depth, time_ms);
        }

        quantum_state_free(&state);
    }
}

int main(int argc, char **argv) {
    config_t config;
    parse_args(argc, argv, &config);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       QUANTUM SIMULATOR SCALING BENCHMARK                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Measuring scaling with qubits, depth, and bond dimension    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    printf("\nConfiguration:\n");
    printf("  Qubits: %d - %d\n", config.min_qubits, config.max_qubits);
    printf("  Depth: %d - %d\n", config.min_depth, config.max_depth);
    printf("  Bond dim: %d - %d\n", config.min_bond, config.max_bond);
    printf("  Output: %s\n", config.output_file);

    FILE *out = fopen(config.output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot open output file %s\n", config.output_file);
        return 1;
    }

    fprintf(out, "# Quantum Simulator Scaling Benchmark Results\n");
    fprintf(out, "# Generated: %s\n", __DATE__);

    srand(42);  // Reproducible

    if (config.bench_state) {
        benchmark_state_vector(&config, out);
    }

    if (config.bench_tensor) {
        benchmark_tensor_network(&config, out);
    }

    benchmark_depth_scaling(&config, out);

    fclose(out);

    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Results written to: %s\n", config.output_file);
    printf("════════════════════════════════════════════════════════════════\n");

    return 0;
}
