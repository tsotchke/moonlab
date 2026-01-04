#include "parallel_ops.h"
#include "../algorithms/grover.h"
#include "../quantum/gates.h"
#include "../utils/constants.h"
#include "../applications/hardware_entropy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <unistd.h>
#endif

// ============================================================================
// HARDWARE DETECTION
// ============================================================================

size_t grover_parallel_get_optimal_batch_size(void) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    return (size_t)num_threads;
#else
    // Fallback: try to detect cores
#ifdef __APPLE__
    int num_cores = 0;
    size_t size = sizeof(num_cores);
    if (sysctlbyname("hw.ncpu", &num_cores, &size, NULL, 0) == 0) {
        return (size_t)num_cores;
    }
#endif
    return 1;  // Sequential fallback
#endif
}

// ============================================================================
// PARALLEL BATCH PROCESSING
// ============================================================================

grover_parallel_result_t grover_parallel_batch(
    const grover_parallel_config_t *config,
    const uint64_t *marked_states,
    quantum_entropy_ctx_t *entropy_pools
) {
    grover_parallel_result_t result = {0};
    
    if (!config || !marked_states || !entropy_pools) {
        return result;
    }
    
    size_t num_searches = config->num_parallel_searches;
    if (num_searches == 0) {
        return result;
    }
    
    // Allocate results array
    result.results = calloc(num_searches, sizeof(grover_result_t));
    if (!result.results) {
        return result;
    }
    result.num_results = num_searches;
    
    // Allocate quantum states (one per thread)
    quantum_state_t *states = calloc(num_searches, sizeof(quantum_state_t));
    if (!states) {
        free(result.results);
        memset(&result, 0, sizeof(result));
        return result;
    }
    
    // Initialize quantum states
    for (size_t i = 0; i < num_searches; i++) {
        if (quantum_state_init(&states[i], config->num_qubits) != QS_SUCCESS) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                quantum_state_free(&states[j]);
            }
            free(states);
            free(result.results);
            memset(&result, 0, sizeof(result));
            return result;
        }
    }
    
    // Start timing
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // PARALLEL EXECUTION - M2 ULTRA 24-CORE OPTIMIZATION
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(num_searches)
#endif
    for (size_t i = 0; i < num_searches; i++) {
        // Each thread runs independent Grover search
        grover_config_t search_config = {
            .num_qubits = config->num_qubits,
            .marked_state = marked_states[i],
            .num_iterations = 0,
            .use_optimal_iterations = config->use_optimal_iterations
        };
        
        // Thread-safe: each thread has its own state and entropy context
        result.results[i] = grover_search(
            &states[i],
            &search_config,
            &entropy_pools[i]
        );
    }
    
    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate execution time
    result.total_time_seconds = 
        (end_time.tv_sec - start_time.tv_sec) +
        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    result.searches_per_second = num_searches / result.total_time_seconds;
    
    // Find best result (highest success probability)
    result.best_result = result.results[0];
    for (size_t i = 1; i < num_searches; i++) {
        if (result.results[i].success_probability > result.best_result.success_probability) {
            result.best_result = result.results[i];
        }
    }
    
    // Cleanup quantum states
    for (size_t i = 0; i < num_searches; i++) {
        quantum_state_free(&states[i]);
    }
    free(states);
    
    return result;
}

grover_parallel_result_t grover_parallel_random_batch(
    size_t num_searches,
    size_t num_qubits,
    quantum_entropy_ctx_t *entropy_pool
) {
    grover_parallel_result_t result = {0};
    
    if (num_searches == 0 || num_qubits == 0 || !entropy_pool) {
        return result;
    }
    
    // Generate random target states
    uint64_t *marked_states = calloc(num_searches, sizeof(uint64_t));
    if (!marked_states) {
        return result;
    }
    
    uint64_t search_space = 1ULL << num_qubits;
    for (size_t i = 0; i < num_searches; i++) {
        uint64_t random_target;
        if (quantum_entropy_get_uint64(entropy_pool, &random_target) != 0) {
            free(marked_states);
            return result;
        }
        marked_states[i] = random_target % search_space;
    }
    
    // Create entropy contexts for each thread
    quantum_entropy_ctx_t *entropy_pools = calloc(num_searches, sizeof(quantum_entropy_ctx_t));
    if (!entropy_pools) {
        free(marked_states);
        return result;
    }
    
    // Each thread gets its own entropy context (thread-safe)
    for (size_t i = 0; i < num_searches; i++) {
        // Copy entropy context structure (each thread will have independent state)
        memcpy(&entropy_pools[i], entropy_pool, sizeof(quantum_entropy_ctx_t));
    }
    
    // Configure parallel execution
    grover_parallel_config_t config = {
        .num_parallel_searches = num_searches,
        .num_qubits = num_qubits,
        .use_optimal_iterations = 1,
        .pin_to_performance_cores = 0
    };
    
    // Execute parallel batch
    result = grover_parallel_batch(&config, marked_states, entropy_pools);
    
    // Cleanup
    free(marked_states);
    free(entropy_pools);
    
    return result;
}

int grover_parallel_random_samples(
    size_t num_qubits,
    uint64_t *samples,
    size_t num_samples,
    quantum_entropy_ctx_t *entropy_pool
) {
    if (!samples || num_samples == 0 || !entropy_pool) {
        return -1;
    }
    
    // Determine optimal batch size
    size_t batch_size = grover_parallel_get_optimal_batch_size();
    if (batch_size > num_samples) {
        batch_size = num_samples;
    }
    
    size_t samples_generated = 0;
    
    while (samples_generated < num_samples) {
        size_t current_batch = batch_size;
        if (samples_generated + current_batch > num_samples) {
            current_batch = num_samples - samples_generated;
        }
        
        // Run parallel batch
        grover_parallel_result_t batch_result = grover_parallel_random_batch(
            current_batch,
            num_qubits,
            entropy_pool
        );
        
        if (!batch_result.results) {
            return -1;
        }
        
        // Extract found states
        for (size_t i = 0; i < current_batch; i++) {
            samples[samples_generated + i] = batch_result.results[i].found_state;
        }
        
        samples_generated += current_batch;
        
        grover_parallel_free_result(&batch_result);
    }
    
    return 0;
}

// ============================================================================
// PARTITIONED SEARCH (Experimental)
// ============================================================================

grover_result_t grover_parallel_partitioned_search(
    size_t num_qubits,
    uint64_t marked_state,
    size_t num_partitions,
    quantum_entropy_ctx_t *entropy_pool
) {
    grover_result_t result = {0};

    if (num_partitions == 0 || !entropy_pool || num_qubits == 0) {
        return result;
    }

    // =========================================================================
    // Partitioned Grover Search Implementation
    //
    // Divides the search space into num_partitions regions and runs parallel
    // Grover searches on each partition. This is useful when:
    // 1. The marked state's location is unknown but might be in a specific region
    // 2. Multiple QPUs/cores are available for parallel execution
    // 3. We want to reduce iterations per search at cost of more searches
    //
    // Each partition uses a modified oracle that only marks states within
    // that partition, effectively searching a smaller space faster.
    // =========================================================================

    uint64_t total_states = 1ULL << num_qubits;
    uint64_t states_per_partition = total_states / num_partitions;
    if (states_per_partition == 0) states_per_partition = 1;

    // Find which partition contains the marked state
    uint64_t target_partition = marked_state / states_per_partition;
    if (target_partition >= num_partitions) {
        target_partition = num_partitions - 1;
    }

    // Allocate results array for parallel execution
    grover_result_t *partition_results = calloc(num_partitions, sizeof(grover_result_t));
    int *found_flags = calloc(num_partitions, sizeof(int));

    if (!partition_results || !found_flags) {
        free(partition_results);
        free(found_flags);
        return result;
    }

    // Run Grover searches in parallel on each partition
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t p = 0; p < num_partitions; p++) {
        quantum_state_t state;
        if (quantum_state_init(&state, num_qubits) != QS_SUCCESS) {
            continue;
        }

        // Calculate partition boundaries
        uint64_t partition_start = p * states_per_partition;
        uint64_t partition_end = (p == num_partitions - 1)
            ? total_states
            : (p + 1) * states_per_partition;
        uint64_t partition_size = partition_end - partition_start;

        // Calculate optimal iterations for this partition size
        // Grover iterations ~ π/4 * sqrt(N/M) where N is search space, M is marked states
        size_t optimal_iters = (size_t)(M_PI / 4.0 * sqrt((double)partition_size));
        if (optimal_iters < 1) optimal_iters = 1;

        // Initialize to uniform superposition over partition
        quantum_state_reset(&state);

        // Prepare initial superposition using Hadamard gates
        for (size_t q = 0; q < num_qubits; q++) {
            gate_hadamard(&state, q);
        }

        // Run Grover iterations
        for (size_t iter = 0; iter < optimal_iters; iter++) {
            // Oracle: flip phase of marked state if it's in this partition
            if (marked_state >= partition_start && marked_state < partition_end) {
                state.amplitudes[marked_state] = -state.amplitudes[marked_state];
            }

            // Diffusion operator (amplitude amplification)
            // Reflect about the mean amplitude
            double complex mean = 0.0;
            for (size_t i = 0; i < state.state_dim; i++) {
                mean += state.amplitudes[i];
            }
            mean /= (double)state.state_dim;

            for (size_t i = 0; i < state.state_dim; i++) {
                state.amplitudes[i] = 2.0 * mean - state.amplitudes[i];
            }
        }

        // Measure - get probabilities and find most likely state in partition
        double max_prob = 0.0;
        uint64_t most_likely = partition_start;

        for (uint64_t i = partition_start; i < partition_end; i++) {
            double prob = cabs(state.amplitudes[i]) * cabs(state.amplitudes[i]);
            if (prob > max_prob) {
                max_prob = prob;
                most_likely = i;
            }
        }

        partition_results[p].found_state = most_likely;
        partition_results[p].success_probability = max_prob;
        partition_results[p].iterations_performed = optimal_iters;

        // Mark if we found the target
        if (most_likely == marked_state) {
            found_flags[p] = 1;
        }

        quantum_state_free(&state);
    }

    // Find best result (highest probability or exact match)
    result.success_probability = 0.0;

    for (size_t p = 0; p < num_partitions; p++) {
        if (found_flags[p]) {
            // Found exact match
            result = partition_results[p];
            result.found_state = marked_state;
            break;
        }
        if (partition_results[p].success_probability > result.success_probability) {
            result = partition_results[p];
        }
    }

    free(partition_results);
    free(found_flags);

    return result;
}

// ============================================================================
// PERFORMANCE BENCHMARKING
// ============================================================================

grover_parallel_benchmark_t grover_parallel_benchmark(
    size_t num_searches,
    size_t num_qubits,
    quantum_entropy_ctx_t *entropy_pool
) {
    grover_parallel_benchmark_t benchmark = {0};
    
    if (num_searches == 0 || num_qubits == 0 || !entropy_pool) {
        return benchmark;
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     M2 ULTRA PARALLEL GROVER BENCHMARK                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║                                                           ║\n");
    printf("║  Configuration:                                           ║\n");
    printf("║    Searches:            %6zu                            ║\n", num_searches);
    printf("║    Qubits per search:   %6zu                            ║\n", num_qubits);
    printf("║    Search space:        %6llu (2^%zu)                   ║\n",
           1ULL << num_qubits, num_qubits);
    
#ifdef _OPENMP
    printf("║    OpenMP threads:      %6d                            ║\n", omp_get_max_threads());
#else
    printf("║    OpenMP:              NOT ENABLED                       ║\n");
#endif
    
    printf("║                                                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // ========================================
    // SEQUENTIAL BENCHMARK
    // ========================================
    
    printf("Running SEQUENTIAL benchmark (%zu searches)...\n", num_searches);
    fflush(stdout);
    
    struct timespec seq_start, seq_end;
    clock_gettime(CLOCK_MONOTONIC, &seq_start);
    
    quantum_state_t state;
    quantum_state_init(&state, num_qubits);
    
    uint64_t search_space = 1ULL << num_qubits;
    
    for (size_t i = 0; i < num_searches; i++) {
        uint64_t target;
        quantum_entropy_get_uint64(entropy_pool, &target);
        target = target % search_space;
        
        grover_config_t config = {
            .num_qubits = num_qubits,
            .marked_state = target,
            .num_iterations = 0,
            .use_optimal_iterations = 1
        };
        
        grover_search(&state, &config, entropy_pool);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &seq_end);
    quantum_state_free(&state);
    
    benchmark.sequential_time = 
        (seq_end.tv_sec - seq_start.tv_sec) +
        (seq_end.tv_nsec - seq_start.tv_nsec) / 1e9;
    
    printf("Sequential time: %.6f seconds\n", benchmark.sequential_time);
    printf("Throughput: %.2f searches/second\n\n", num_searches / benchmark.sequential_time);
    
    // ========================================
    // PARALLEL BENCHMARK
    // ========================================
    
    printf("Running PARALLEL benchmark (%zu searches on %zu cores)...\n",
           num_searches, grover_parallel_get_optimal_batch_size());
    fflush(stdout);
    
    struct timespec par_start, par_end;
    clock_gettime(CLOCK_MONOTONIC, &par_start);
    
    grover_parallel_result_t par_result = grover_parallel_random_batch(
        num_searches,
        num_qubits,
        entropy_pool
    );
    
    clock_gettime(CLOCK_MONOTONIC, &par_end);
    
    benchmark.parallel_time = 
        (par_end.tv_sec - par_start.tv_sec) +
        (par_end.tv_nsec - par_start.tv_nsec) / 1e9;
    
    printf("Parallel time: %.6f seconds\n", benchmark.parallel_time);
    printf("Throughput: %.2f searches/second\n\n", num_searches / benchmark.parallel_time);
    
    grover_parallel_free_result(&par_result);
    
    // ========================================
    // ANALYSIS
    // ========================================
    
    benchmark.speedup = benchmark.sequential_time / benchmark.parallel_time;
    benchmark.num_cores_used = grover_parallel_get_optimal_batch_size();
    benchmark.efficiency = benchmark.speedup / benchmark.num_cores_used;
    
    return benchmark;
}

void grover_parallel_print_benchmark(const grover_parallel_benchmark_t *benchmark) {
    if (!benchmark) return;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     PARALLEL PERFORMANCE RESULTS                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║                                                           ║\n");
    printf("║  Execution Time:                                          ║\n");
    printf("║    Sequential:          %10.6f seconds                 ║\n", benchmark->sequential_time);
    printf("║    Parallel:            %10.6f seconds                 ║\n", benchmark->parallel_time);
    printf("║                                                           ║\n");
    printf("║  Performance:                                             ║\n");
    printf("║    Speedup:             %10.2fx                        ║\n", benchmark->speedup);
    printf("║    Cores used:          %10zu                          ║\n", benchmark->num_cores_used);
    printf("║    Efficiency:          %9.1f%%                        ║\n", 
           benchmark->efficiency * 100.0);
    printf("║                                                           ║\n");
    
    // Evaluation
    if (benchmark->speedup >= benchmark->num_cores_used * 0.8) {
        printf("║  ┌─────────────────────────────────────────────────────┐  ║\n");
        printf("║  │   ✓ EXCELLENT SCALING                               │  ║\n");
        printf("║  │   Near-linear speedup achieved!                     │  ║\n");
        printf("║  └─────────────────────────────────────────────────────┘  ║\n");
    } else if (benchmark->speedup >= benchmark->num_cores_used * 0.5) {
        printf("║  ┌─────────────────────────────────────────────────────┐  ║\n");
        printf("║  │   ✓ GOOD SCALING                                    │  ║\n");
        printf("║  │   Effective multi-core utilization                  │  ║\n");
        printf("║  └─────────────────────────────────────────────────────┘  ║\n");
    } else {
        printf("║  ┌─────────────────────────────────────────────────────┐  ║\n");
        printf("║  │   ⚠ SUBOPTIMAL SCALING                              │  ║\n");
        printf("║  │   Consider larger batch sizes or tune OpenMP        │  ║\n");
        printf("║  └─────────────────────────────────────────────────────┘  ║\n");
    }
    
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

// ============================================================================
// UTILITIES
// ============================================================================

void grover_parallel_free_result(grover_parallel_result_t *result) {
    if (!result) return;
    
    if (result->results) {
        free(result->results);
        result->results = NULL;
    }
    
    result->num_results = 0;
}

void grover_parallel_print_config(const grover_parallel_config_t *config) {
    if (!config) return;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     PARALLEL GROVER CONFIGURATION                         ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║                                                           ║\n");
    printf("║  Parallel searches:     %6zu                            ║\n", 
           config->num_parallel_searches);
    printf("║  Qubits per search:     %6zu                            ║\n",
           config->num_qubits);
    printf("║  Search space each:     %6llu (2^%zu)                   ║\n",
           1ULL << config->num_qubits, config->num_qubits);
    printf("║  Optimal iterations:    %s                              ║\n",
           config->use_optimal_iterations ? "YES" : "NO ");
    printf("║  Pin to P-cores:        %s                              ║\n",
           config->pin_to_performance_cores ? "YES" : "NO ");
    printf("║                                                           ║\n");
    
#ifdef _OPENMP
    printf("║  OpenMP Status:         ENABLED                           ║\n");
    printf("║  Max threads:           %6d                            ║\n", omp_get_max_threads());
#else
    printf("║  OpenMP Status:         DISABLED                          ║\n");
    printf("║  Note: Compile with -fopenmp for parallel execution      ║\n");
#endif
    
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}