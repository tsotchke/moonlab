/**
 * @file profiler.h
 * @brief Performance profiling for quantum simulator
 *
 * High-precision timing and performance analysis:
 * - Function-level timing
 * - Gate operation statistics
 * - Memory usage tracking
 * - Cache efficiency metrics
 * - SIMD utilization analysis
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef TOOLS_PROFILER_H
#define TOOLS_PROFILER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PROFILER CONFIGURATION
// ============================================================================

#define PROFILER_MAX_REGIONS    256
#define PROFILER_MAX_NAME_LEN   64
#define PROFILER_MAX_DEPTH      32

/**
 * @brief Profiler enable flags
 */
typedef enum {
    PROFILER_TIMING       = 0x01,  /**< Enable timing measurements */
    PROFILER_MEMORY       = 0x02,  /**< Enable memory tracking */
    PROFILER_COUNTERS     = 0x04,  /**< Enable hardware counters */
    PROFILER_CALL_GRAPH   = 0x08,  /**< Enable call graph tracking */
    PROFILER_ALL          = 0x0F   /**< Enable all features */
} profiler_flags_t;

// ============================================================================
// TIMING STRUCTURES
// ============================================================================

/**
 * @brief High-precision timestamp
 */
typedef struct {
    uint64_t ticks;         /**< CPU ticks or nanoseconds */
} profiler_time_t;

/**
 * @brief Timing statistics for a profiled region
 */
typedef struct {
    char name[PROFILER_MAX_NAME_LEN];

    uint64_t call_count;    /**< Number of calls */
    uint64_t total_time;    /**< Total time (nanoseconds) */
    uint64_t min_time;      /**< Minimum call time */
    uint64_t max_time;      /**< Maximum call time */
    uint64_t self_time;     /**< Time excluding children */

    // Running statistics
    double mean;            /**< Running mean */
    double m2;              /**< For variance calculation */

    // Parent tracking
    int parent_id;          /**< Parent region ID */
} profiler_region_t;

/**
 * @brief Memory allocation statistics
 */
typedef struct {
    uint64_t total_allocated;     /**< Total bytes allocated */
    uint64_t total_freed;         /**< Total bytes freed */
    uint64_t current_usage;       /**< Current memory usage */
    uint64_t peak_usage;          /**< Peak memory usage */
    uint64_t allocation_count;    /**< Number of allocations */
    uint64_t free_count;          /**< Number of frees */
    uint64_t realloc_count;       /**< Number of reallocs */
} profiler_memory_stats_t;

/**
 * @brief Gate operation statistics
 */
typedef struct {
    uint64_t single_qubit_gates;  /**< Single-qubit gate count */
    uint64_t two_qubit_gates;     /**< Two-qubit gate count */
    uint64_t multi_qubit_gates;   /**< Multi-qubit gate count */
    uint64_t measurements;        /**< Measurement count */

    // Per-gate type counts
    uint64_t hadamard_count;
    uint64_t pauli_x_count;
    uint64_t pauli_y_count;
    uint64_t pauli_z_count;
    uint64_t cnot_count;
    uint64_t cz_count;
    uint64_t swap_count;
    uint64_t rotation_count;

    // Timing
    uint64_t gate_time_total;     /**< Total time in gates */
} profiler_gate_stats_t;

/**
 * @brief Main profiler context
 */
typedef struct {
    int enabled;
    profiler_flags_t flags;

    // Timing regions
    profiler_region_t regions[PROFILER_MAX_REGIONS];
    int num_regions;

    // Call stack for hierarchical timing
    int call_stack[PROFILER_MAX_DEPTH];
    int stack_depth;

    // Memory stats
    profiler_memory_stats_t memory;

    // Gate stats
    profiler_gate_stats_t gates;

    // Global timing
    profiler_time_t start_time;
    uint64_t total_runtime;

} profiler_ctx_t;

// ============================================================================
// GLOBAL PROFILER
// ============================================================================

/**
 * @brief Initialize global profiler
 *
 * @param flags Profiling features to enable
 * @return 0 on success, -1 on error
 */
int profiler_init(profiler_flags_t flags);

/**
 * @brief Shutdown global profiler
 */
void profiler_shutdown(void);

/**
 * @brief Reset all profiling data
 */
void profiler_reset(void);

/**
 * @brief Get global profiler context
 *
 * @return Profiler context or NULL if not initialized
 */
profiler_ctx_t* profiler_get_context(void);

/**
 * @brief Enable/disable profiling
 *
 * @param enable 1 to enable, 0 to disable
 */
void profiler_set_enabled(int enable);

/**
 * @brief Check if profiler is enabled
 *
 * @return 1 if enabled, 0 otherwise
 */
int profiler_is_enabled(void);

// ============================================================================
// TIMING FUNCTIONS
// ============================================================================

/**
 * @brief Get current high-precision timestamp
 *
 * @return Current time
 */
profiler_time_t profiler_get_time(void);

/**
 * @brief Calculate elapsed time in nanoseconds
 *
 * @param start Start time
 * @param end End time
 * @return Elapsed nanoseconds
 */
uint64_t profiler_elapsed_ns(profiler_time_t start, profiler_time_t end);

/**
 * @brief Start timing a named region
 *
 * @param name Region name
 * @return Region ID for matching with end call
 */
int profiler_region_start(const char* name);

/**
 * @brief End timing a region
 *
 * @param region_id Region ID from start call
 */
void profiler_region_end(int region_id);

/**
 * @brief Get timing statistics for a region
 *
 * @param name Region name
 * @return Region statistics or NULL if not found
 */
const profiler_region_t* profiler_get_region(const char* name);

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#ifdef QSIM_ENABLE_PROFILING

#define PROFILER_REGION_START(name) \
    int _profiler_region_##__LINE__ = profiler_region_start(name)

#define PROFILER_REGION_END() \
    profiler_region_end(_profiler_region_##__LINE__)

#define PROFILER_FUNCTION() \
    int _profiler_func_region = profiler_region_start(__func__); \
    (void)_profiler_func_region

#define PROFILER_FUNCTION_END() \
    profiler_region_end(_profiler_func_region)

#else

#define PROFILER_REGION_START(name) ((void)0)
#define PROFILER_REGION_END() ((void)0)
#define PROFILER_FUNCTION() ((void)0)
#define PROFILER_FUNCTION_END() ((void)0)

#endif

// ============================================================================
// MEMORY TRACKING
// ============================================================================

/**
 * @brief Track memory allocation
 *
 * @param size Bytes allocated
 */
void profiler_track_alloc(size_t size);

/**
 * @brief Track memory free
 *
 * @param size Bytes freed
 */
void profiler_track_free(size_t size);

/**
 * @brief Track reallocation
 *
 * @param old_size Previous size
 * @param new_size New size
 */
void profiler_track_realloc(size_t old_size, size_t new_size);

/**
 * @brief Get memory statistics
 *
 * @return Pointer to memory stats
 */
const profiler_memory_stats_t* profiler_get_memory_stats(void);

// ============================================================================
// GATE TRACKING
// ============================================================================

/**
 * @brief Track gate application
 *
 * @param gate_type Gate type string
 * @param num_qubits Number of qubits affected
 */
void profiler_track_gate(const char* gate_type, int num_qubits);

/**
 * @brief Track measurement
 */
void profiler_track_measurement(void);

/**
 * @brief Get gate statistics
 *
 * @return Pointer to gate stats
 */
const profiler_gate_stats_t* profiler_get_gate_stats(void);

// ============================================================================
// REPORTING
// ============================================================================

/**
 * @brief Print profiling summary to stdout
 */
void profiler_print_summary(void);

/**
 * @brief Print detailed timing report
 */
void profiler_print_timing_report(void);

/**
 * @brief Print memory report
 */
void profiler_print_memory_report(void);

/**
 * @brief Print gate statistics
 */
void profiler_print_gate_report(void);

/**
 * @brief Export profiling data to JSON file
 *
 * @param path Output file path
 * @return 0 on success, -1 on error
 */
int profiler_export_json(const char* path);

/**
 * @brief Export profiling data to CSV file
 *
 * @param path Output file path
 * @return 0 on success, -1 on error
 */
int profiler_export_csv(const char* path);

// ============================================================================
// BENCHMARKING UTILITIES
// ============================================================================

/**
 * @brief Benchmark result
 */
typedef struct {
    double mean_ns;         /**< Mean time in nanoseconds */
    double std_ns;          /**< Standard deviation */
    double min_ns;          /**< Minimum time */
    double max_ns;          /**< Maximum time */
    int iterations;         /**< Number of iterations */
    double ops_per_sec;     /**< Operations per second */
} benchmark_result_t;

/**
 * @brief Run benchmark function
 *
 * @param func Function to benchmark
 * @param data User data for function
 * @param warmup_iterations Warmup iterations
 * @param bench_iterations Benchmark iterations
 * @param result Output result
 * @return 0 on success, -1 on error
 */
int profiler_benchmark(void (*func)(void*), void* data,
                       int warmup_iterations, int bench_iterations,
                       benchmark_result_t* result);

/**
 * @brief Print benchmark result
 *
 * @param name Benchmark name
 * @param result Benchmark result
 */
void profiler_print_benchmark(const char* name, const benchmark_result_t* result);

// ============================================================================
// SCOPED PROFILER (C++ style RAII for C)
// ============================================================================

/**
 * @brief Scoped profiler handle
 */
typedef struct {
    int region_id;
    profiler_time_t start;
} profiler_scope_t;

/**
 * @brief Begin scoped profiling
 *
 * @param name Region name
 * @return Scope handle
 */
profiler_scope_t profiler_scope_begin(const char* name);

/**
 * @brief End scoped profiling
 *
 * @param scope Scope handle
 */
void profiler_scope_end(profiler_scope_t* scope);

#ifdef __cplusplus
}
#endif

#endif /* TOOLS_PROFILER_H */
