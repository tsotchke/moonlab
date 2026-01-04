/**
 * @file memory_profiler.c
 * @brief Memory profiling tool for quantum simulator
 *
 * Profiles memory usage patterns across:
 * - State vector allocation and deallocation
 * - Tensor network memory scaling
 * - Peak memory during circuit execution
 * - Memory fragmentation analysis
 *
 * Usage: ./memory_profiler [options]
 *   --qubits MIN MAX    Range of qubit counts (default: 4 24)
 *   --iterations N      Allocation iterations (default: 100)
 *   --type TYPE         Profile type: state, tensor, both (default: both)
 *   --output FILE       Output file (default: memory_profile.csv)
 *
 * @stability stable
 * @since v1.0.0
 */

#include "../../src/quantum/state.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <getopt.h>
#include <math.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif

#define DEFAULT_MIN_QUBITS 4
#define DEFAULT_MAX_QUBITS 24
#define DEFAULT_ITERATIONS 100

typedef struct {
    int min_qubits;
    int max_qubits;
    int iterations;
    char output_file[256];
    int profile_state;
    int profile_tensor;
} profiler_config_t;

/**
 * @brief Get current memory usage in bytes
 */
static size_t get_memory_usage(void) {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#else
    // Linux: read from /proc/self/statm
    FILE *f = fopen("/proc/self/statm", "r");
    if (f) {
        long pages = 0;
        fscanf(f, "%*d %ld", &pages);
        fclose(f);
        return pages * sysconf(_SC_PAGESIZE);
    }
    return 0;
#endif
}

/**
 * @brief Get peak memory usage
 */
static size_t get_peak_memory(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss;  // Already in bytes on macOS
#else
        return usage.ru_maxrss * 1024;  // Convert from KB to bytes on Linux
#endif
    }
    return 0;
}

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --qubits MIN MAX    Range of qubit counts (default: %d %d)\n",
           DEFAULT_MIN_QUBITS, DEFAULT_MAX_QUBITS);
    printf("  --iterations N      Allocation iterations (default: %d)\n", DEFAULT_ITERATIONS);
    printf("  --type TYPE         Profile type: state, tensor, both (default: both)\n");
    printf("  --output FILE       Output file (default: memory_profile.csv)\n");
    printf("  --help              Show this help\n");
}

static void parse_args(int argc, char **argv, profiler_config_t *config) {
    config->min_qubits = DEFAULT_MIN_QUBITS;
    config->max_qubits = DEFAULT_MAX_QUBITS;
    config->iterations = DEFAULT_ITERATIONS;
    strcpy(config->output_file, "memory_profile.csv");
    config->profile_state = 1;
    config->profile_tensor = 1;

    static struct option long_options[] = {
        {"qubits", required_argument, 0, 'q'},
        {"iterations", required_argument, 0, 'i'},
        {"type", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "q:i:t:o:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'q':
                sscanf(optarg, "%d", &config->min_qubits);
                if (optind < argc && argv[optind][0] != '-') {
                    config->max_qubits = atoi(argv[optind++]);
                }
                break;
            case 'i':
                config->iterations = atoi(optarg);
                break;
            case 't':
                if (strcmp(optarg, "state") == 0) {
                    config->profile_tensor = 0;
                } else if (strcmp(optarg, "tensor") == 0) {
                    config->profile_state = 0;
                }
                break;
            case 'o':
                strncpy(config->output_file, optarg, sizeof(config->output_file) - 1);
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
 * @brief Profile state vector memory allocation
 */
static void profile_state_allocation(profiler_config_t *config, FILE *out) {
    printf("\n=== State Vector Memory Profile ===\n");
    fprintf(out, "# State Vector Memory Profile\n");
    fprintf(out, "qubits,theoretical_mb,actual_mb,alloc_us,free_us,overhead_pct\n");

    for (int n = config->min_qubits; n <= config->max_qubits; n += 2) {
        // Theoretical memory
        size_t state_dim = 1ULL << n;
        double theoretical_mb = (state_dim * sizeof(double _Complex)) / (1024.0 * 1024.0);

        if (theoretical_mb > 16000) {
            printf("  %2d qubits: Skipped (%.1f GB theoretical)\n", n, theoretical_mb / 1024);
            continue;
        }

        // Measure allocation
        size_t mem_before = get_memory_usage();
        double start = get_time_us();

        quantum_state_t state;
        qs_error_t err = quantum_state_init(&state, n);

        double alloc_time = get_time_us() - start;

        if (err != QS_SUCCESS) {
            printf("  %2d qubits: Failed to allocate\n", n);
            continue;
        }

        size_t mem_after = get_memory_usage();
        double actual_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

        // Measure deallocation
        start = get_time_us();
        quantum_state_free(&state);
        double free_time = get_time_us() - start;

        // Calculate overhead
        double overhead_pct = 0.0;
        if (theoretical_mb > 0) {
            overhead_pct = ((actual_mb - theoretical_mb) / theoretical_mb) * 100.0;
        }

        printf("  %2d qubits: theory=%.1f MB, actual=%.1f MB, alloc=%.0f μs, free=%.0f μs\n",
               n, theoretical_mb, actual_mb, alloc_time, free_time);

        fprintf(out, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                n, theoretical_mb, actual_mb > 0 ? actual_mb : theoretical_mb,
                alloc_time, free_time, overhead_pct);
    }
}

/**
 * @brief Profile tensor network memory allocation
 */
static void profile_tensor_allocation(profiler_config_t *config, FILE *out) {
    printf("\n=== Tensor Network Memory Profile ===\n");
    fprintf(out, "\n# Tensor Network Memory Profile\n");
    fprintf(out, "sites,bond_dim,theoretical_mb,actual_mb,alloc_us,free_us\n");

    int site_counts[] = {50, 100, 200, 500, 1000};
    int num_site_options = 5;
    int bond_dims[] = {8, 16, 32, 64, 128};
    int num_bond_dims = 5;

    for (int s = 0; s < num_site_options; s++) {
        int num_sites = site_counts[s];

        for (int b = 0; b < num_bond_dims; b++) {
            int chi = bond_dims[b];

            // Theoretical memory for MPS: ~N * chi^2 * d^2 complex numbers
            // d=2 for qubits
            double theoretical_mb = (num_sites * chi * chi * 4 * sizeof(double _Complex)) / (1024.0 * 1024.0);

            if (theoretical_mb > 8000) {
                printf("  %4d sites, χ=%3d: Skipped (%.1f GB theoretical)\n",
                       num_sites, chi, theoretical_mb / 1024);
                continue;
            }

            // Create config with desired bond dimension
            tn_state_config_t tn_config = tn_state_config_create(chi, 1e-12);

            size_t mem_before = get_memory_usage();
            double start = get_time_us();

            tn_mps_state_t *mps = tn_mps_create_zero(num_sites, &tn_config);

            double alloc_time = get_time_us() - start;

            if (!mps) {
                printf("  %4d sites, χ=%3d: Failed\n", num_sites, chi);
                continue;
            }

            size_t mem_after = get_memory_usage();
            double actual_mb = (mem_after - mem_before) / (1024.0 * 1024.0);

            // Get memory after full initialization
            size_t mem_initialized = get_memory_usage();
            double initialized_mb = (mem_initialized - mem_before) / (1024.0 * 1024.0);

            start = get_time_us();
            tn_mps_free(mps);
            double free_time = get_time_us() - start;

            printf("  %4d sites, χ=%3d: theory=%.1f MB, actual=%.1f MB, alloc=%.0f μs\n",
                   num_sites, chi, theoretical_mb,
                   initialized_mb > 0 ? initialized_mb : theoretical_mb, alloc_time);

            fprintf(out, "%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                    num_sites, chi, theoretical_mb,
                    initialized_mb > 0 ? initialized_mb : theoretical_mb,
                    alloc_time, free_time);
        }
    }
}

/**
 * @brief Profile allocation/deallocation patterns
 */
static void profile_alloc_patterns(profiler_config_t *config, FILE *out) {
    printf("\n=== Allocation Pattern Analysis ===\n");
    fprintf(out, "\n# Allocation Pattern Analysis\n");
    fprintf(out, "pattern,qubits,iterations,total_time_ms,avg_alloc_us,avg_free_us,peak_mb\n");

    int qubits = 14;  // Fixed size for pattern analysis

    // Pattern 1: Sequential allocate/free
    {
        printf("\n  Sequential pattern (%d iterations):\n", config->iterations);
        double total_alloc = 0, total_free = 0;

        for (int i = 0; i < config->iterations; i++) {
            double start = get_time_us();
            quantum_state_t state;
            quantum_state_init(&state, qubits);
            total_alloc += get_time_us() - start;

            start = get_time_us();
            quantum_state_free(&state);
            total_free += get_time_us() - start;
        }

        double avg_alloc = total_alloc / config->iterations;
        double avg_free = total_free / config->iterations;
        double total_ms = (total_alloc + total_free) / 1000.0;
        double peak_mb = get_peak_memory() / (1024.0 * 1024.0);

        printf("    Avg alloc: %.0f μs, Avg free: %.0f μs, Peak: %.1f MB\n",
               avg_alloc, avg_free, peak_mb);

        fprintf(out, "sequential,%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                qubits, config->iterations, total_ms, avg_alloc, avg_free, peak_mb);
    }

    // Pattern 2: Batch allocate then free
    {
        int batch_size = 10;
        printf("\n  Batch pattern (batch=%d, iterations=%d):\n",
               batch_size, config->iterations / batch_size);

        double total_time = 0;
        double peak_mb = 0;

        for (int iter = 0; iter < config->iterations / batch_size; iter++) {
            quantum_state_t states[10];

            double start = get_time_us();
            for (int i = 0; i < batch_size; i++) {
                quantum_state_init(&states[i], qubits);
            }

            double current_mb = get_memory_usage() / (1024.0 * 1024.0);
            if (current_mb > peak_mb) peak_mb = current_mb;

            for (int i = 0; i < batch_size; i++) {
                quantum_state_free(&states[i]);
            }
            total_time += get_time_us() - start;
        }

        double total_ms = total_time / 1000.0;
        printf("    Total time: %.1f ms, Peak memory: %.1f MB\n", total_ms, peak_mb);

        fprintf(out, "batch,%d,%d,%.2f,0,0,%.2f\n",
                qubits, config->iterations, total_ms, peak_mb);
    }

    // Pattern 3: Growing sizes
    {
        printf("\n  Growing size pattern:\n");
        double start = get_time_us();
        double max_mb = 0;

        for (int n = 4; n <= 18; n += 2) {
            quantum_state_t state;
            quantum_state_init(&state, n);

            double current_mb = get_memory_usage() / (1024.0 * 1024.0);
            if (current_mb > max_mb) max_mb = current_mb;

            quantum_state_free(&state);
        }

        double total_ms = (get_time_us() - start) / 1000.0;
        printf("    Total time: %.1f ms, Peak: %.1f MB\n", total_ms, max_mb);

        fprintf(out, "growing,4-18,8,%.2f,0,0,%.2f\n", total_ms, max_mb);
    }
}

/**
 * @brief Analyze memory fragmentation
 */
static void analyze_fragmentation(profiler_config_t *config, FILE *out) {
    printf("\n=== Memory Fragmentation Analysis ===\n");
    fprintf(out, "\n# Memory Fragmentation Analysis\n");
    fprintf(out, "iteration,allocated_mb,resident_mb,fragmentation_pct\n");

    size_t base_memory = get_memory_usage();
    int num_states = 20;
    quantum_state_t *states = malloc(num_states * sizeof(quantum_state_t));

    printf("  Allocating %d quantum states of varying sizes...\n", num_states);

    // Allocate states of varying sizes
    size_t total_allocated = 0;
    for (int i = 0; i < num_states; i++) {
        int qubits = 8 + (i % 5) * 2;  // 8, 10, 12, 14, 16 pattern
        quantum_state_init(&states[i], qubits);
        total_allocated += (1ULL << qubits) * sizeof(double _Complex);
    }

    double allocated_mb = total_allocated / (1024.0 * 1024.0);
    double resident_mb = (get_memory_usage() - base_memory) / (1024.0 * 1024.0);
    double frag_initial = resident_mb > 0 ? ((resident_mb - allocated_mb) / resident_mb) * 100 : 0;

    printf("  Initial: allocated=%.1f MB, resident=%.1f MB, frag=%.1f%%\n",
           allocated_mb, resident_mb, frag_initial);
    fprintf(out, "0,%.2f,%.2f,%.2f\n", allocated_mb, resident_mb, frag_initial);

    // Free every other state to create fragmentation
    printf("  Freeing every other state...\n");
    for (int i = 0; i < num_states; i += 2) {
        quantum_state_free(&states[i]);
        states[i].amplitudes = NULL;  // Mark as freed
    }

    total_allocated = 0;
    for (int i = 0; i < num_states; i++) {
        if (states[i].amplitudes) {
            total_allocated += (1ULL << states[i].num_qubits) * sizeof(double _Complex);
        }
    }

    allocated_mb = total_allocated / (1024.0 * 1024.0);
    resident_mb = (get_memory_usage() - base_memory) / (1024.0 * 1024.0);
    double frag_after = resident_mb > 0 ? ((resident_mb - allocated_mb) / resident_mb) * 100 : 0;

    printf("  After partial free: allocated=%.1f MB, resident=%.1f MB, frag=%.1f%%\n",
           allocated_mb, resident_mb, frag_after);
    fprintf(out, "1,%.2f,%.2f,%.2f\n", allocated_mb, resident_mb, frag_after);

    // Reallocate to test fragmentation impact
    printf("  Reallocating freed slots...\n");
    for (int i = 0; i < num_states; i += 2) {
        int qubits = 10 + (i % 4) * 2;  // Different sizes
        quantum_state_init(&states[i], qubits);
    }

    total_allocated = 0;
    for (int i = 0; i < num_states; i++) {
        total_allocated += (1ULL << states[i].num_qubits) * sizeof(double _Complex);
    }

    allocated_mb = total_allocated / (1024.0 * 1024.0);
    resident_mb = (get_memory_usage() - base_memory) / (1024.0 * 1024.0);
    double frag_realloc = resident_mb > 0 ? ((resident_mb - allocated_mb) / resident_mb) * 100 : 0;

    printf("  After realloc: allocated=%.1f MB, resident=%.1f MB, frag=%.1f%%\n",
           allocated_mb, resident_mb, frag_realloc);
    fprintf(out, "2,%.2f,%.2f,%.2f\n", allocated_mb, resident_mb, frag_realloc);

    // Cleanup
    for (int i = 0; i < num_states; i++) {
        quantum_state_free(&states[i]);
    }
    free(states);
}

int main(int argc, char **argv) {
    profiler_config_t config;
    parse_args(argc, argv, &config);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       QUANTUM SIMULATOR MEMORY PROFILER                      ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Analyzing memory allocation patterns and overhead           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    printf("\nConfiguration:\n");
    printf("  Qubits: %d - %d\n", config.min_qubits, config.max_qubits);
    printf("  Iterations: %d\n", config.iterations);
    printf("  Output: %s\n", config.output_file);

    FILE *out = fopen(config.output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot open output file %s\n", config.output_file);
        return 1;
    }

    fprintf(out, "# Quantum Simulator Memory Profile\n");
    fprintf(out, "# Generated: %s\n", __DATE__);

    // Initial memory baseline
    printf("\nBaseline memory: %.1f MB\n", get_memory_usage() / (1024.0 * 1024.0));

    if (config.profile_state) {
        profile_state_allocation(&config, out);
    }

    if (config.profile_tensor) {
        profile_tensor_allocation(&config, out);
    }

    profile_alloc_patterns(&config, out);
    analyze_fragmentation(&config, out);

    // Final summary
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Peak memory usage: %.1f MB\n", get_peak_memory() / (1024.0 * 1024.0));
    printf("  Results written to: %s\n", config.output_file);
    printf("════════════════════════════════════════════════════════════════\n");

    fclose(out);
    return 0;
}
