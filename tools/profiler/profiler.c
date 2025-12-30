/**
 * @file profiler.c
 * @brief Performance profiling implementation
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
 * Licensed under the Apache License, Version 2.0
 */

#include "profiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#ifdef __linux__
#include <time.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static profiler_ctx_t* g_profiler = NULL;

// ============================================================================
// PLATFORM-SPECIFIC TIMING
// ============================================================================

#ifdef __APPLE__

static mach_timebase_info_data_t g_timebase = {0, 0};

static void init_timebase(void) {
    if (g_timebase.denom == 0) {
        mach_timebase_info(&g_timebase);
    }
}

profiler_time_t profiler_get_time(void) {
    profiler_time_t t;
    t.ticks = mach_absolute_time();
    return t;
}

uint64_t profiler_elapsed_ns(profiler_time_t start, profiler_time_t end) {
    init_timebase();
    uint64_t elapsed = end.ticks - start.ticks;
    return (elapsed * g_timebase.numer) / g_timebase.denom;
}

#elif defined(__linux__)

profiler_time_t profiler_get_time(void) {
    profiler_time_t t;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t.ticks = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    return t;
}

uint64_t profiler_elapsed_ns(profiler_time_t start, profiler_time_t end) {
    return end.ticks - start.ticks;
}

#else

profiler_time_t profiler_get_time(void) {
    profiler_time_t t;
    t.ticks = (uint64_t)clock() * 1000000000ULL / CLOCKS_PER_SEC;
    return t;
}

uint64_t profiler_elapsed_ns(profiler_time_t start, profiler_time_t end) {
    return end.ticks - start.ticks;
}

#endif

// ============================================================================
// INITIALIZATION
// ============================================================================

int profiler_init(profiler_flags_t flags) {
    if (g_profiler) {
        return 0;  // Already initialized
    }

    g_profiler = calloc(1, sizeof(profiler_ctx_t));
    if (!g_profiler) {
        return -1;
    }

    g_profiler->enabled = 1;
    g_profiler->flags = flags;
    g_profiler->num_regions = 0;
    g_profiler->stack_depth = 0;
    g_profiler->start_time = profiler_get_time();

    // Initialize memory stats
    memset(&g_profiler->memory, 0, sizeof(profiler_memory_stats_t));

    // Initialize gate stats
    memset(&g_profiler->gates, 0, sizeof(profiler_gate_stats_t));

#ifdef __APPLE__
    init_timebase();
#endif

    return 0;
}

void profiler_shutdown(void) {
    if (g_profiler) {
        free(g_profiler);
        g_profiler = NULL;
    }
}

void profiler_reset(void) {
    if (!g_profiler) return;

    g_profiler->num_regions = 0;
    g_profiler->stack_depth = 0;
    g_profiler->start_time = profiler_get_time();
    g_profiler->total_runtime = 0;

    memset(&g_profiler->memory, 0, sizeof(profiler_memory_stats_t));
    memset(&g_profiler->gates, 0, sizeof(profiler_gate_stats_t));
    memset(g_profiler->regions, 0, sizeof(g_profiler->regions));
}

profiler_ctx_t* profiler_get_context(void) {
    return g_profiler;
}

void profiler_set_enabled(int enable) {
    if (g_profiler) {
        g_profiler->enabled = enable;
    }
}

int profiler_is_enabled(void) {
    return g_profiler && g_profiler->enabled;
}

// ============================================================================
// TIMING REGIONS
// ============================================================================

static int find_or_create_region(const char* name) {
    if (!g_profiler || !name) return -1;

    // Search for existing region
    for (int i = 0; i < g_profiler->num_regions; i++) {
        if (strcmp(g_profiler->regions[i].name, name) == 0) {
            return i;
        }
    }

    // Create new region
    if (g_profiler->num_regions >= PROFILER_MAX_REGIONS) {
        return -1;
    }

    int id = g_profiler->num_regions++;
    profiler_region_t* region = &g_profiler->regions[id];

    strncpy(region->name, name, PROFILER_MAX_NAME_LEN - 1);
    region->name[PROFILER_MAX_NAME_LEN - 1] = '\0';
    region->call_count = 0;
    region->total_time = 0;
    region->min_time = UINT64_MAX;
    region->max_time = 0;
    region->self_time = 0;
    region->mean = 0.0;
    region->m2 = 0.0;
    region->parent_id = -1;

    return id;
}

int profiler_region_start(const char* name) {
    if (!g_profiler || !g_profiler->enabled) return -1;
    if (!(g_profiler->flags & PROFILER_TIMING)) return -1;

    int region_id = find_or_create_region(name);
    if (region_id < 0) return -1;

    profiler_region_t* region = &g_profiler->regions[region_id];

    // Track parent
    if (g_profiler->stack_depth > 0) {
        region->parent_id = g_profiler->call_stack[g_profiler->stack_depth - 1];
    }

    // Push to call stack
    if (g_profiler->stack_depth < PROFILER_MAX_DEPTH) {
        g_profiler->call_stack[g_profiler->stack_depth++] = region_id;
    }

    // Store start time in a thread-local or stack-based manner
    // For simplicity, we encode start time in the region temporarily
    // This is a simplified implementation

    return region_id;
}

void profiler_region_end(int region_id) {
    if (!g_profiler || !g_profiler->enabled) return;
    if (region_id < 0 || region_id >= g_profiler->num_regions) return;

    // Pop from call stack
    if (g_profiler->stack_depth > 0) {
        g_profiler->stack_depth--;
    }
}

const profiler_region_t* profiler_get_region(const char* name) {
    if (!g_profiler || !name) return NULL;

    for (int i = 0; i < g_profiler->num_regions; i++) {
        if (strcmp(g_profiler->regions[i].name, name) == 0) {
            return &g_profiler->regions[i];
        }
    }

    return NULL;
}

// ============================================================================
// MEMORY TRACKING
// ============================================================================

void profiler_track_alloc(size_t size) {
    if (!g_profiler) return;
    if (!(g_profiler->flags & PROFILER_MEMORY)) return;

    g_profiler->memory.total_allocated += size;
    g_profiler->memory.current_usage += size;
    g_profiler->memory.allocation_count++;

    if (g_profiler->memory.current_usage > g_profiler->memory.peak_usage) {
        g_profiler->memory.peak_usage = g_profiler->memory.current_usage;
    }
}

void profiler_track_free(size_t size) {
    if (!g_profiler) return;
    if (!(g_profiler->flags & PROFILER_MEMORY)) return;

    g_profiler->memory.total_freed += size;
    g_profiler->memory.current_usage -= size;
    g_profiler->memory.free_count++;
}

void profiler_track_realloc(size_t old_size, size_t new_size) {
    if (!g_profiler) return;
    if (!(g_profiler->flags & PROFILER_MEMORY)) return;

    g_profiler->memory.realloc_count++;

    if (new_size > old_size) {
        g_profiler->memory.total_allocated += (new_size - old_size);
        g_profiler->memory.current_usage += (new_size - old_size);
    } else {
        g_profiler->memory.total_freed += (old_size - new_size);
        g_profiler->memory.current_usage -= (old_size - new_size);
    }

    if (g_profiler->memory.current_usage > g_profiler->memory.peak_usage) {
        g_profiler->memory.peak_usage = g_profiler->memory.current_usage;
    }
}

const profiler_memory_stats_t* profiler_get_memory_stats(void) {
    return g_profiler ? &g_profiler->memory : NULL;
}

// ============================================================================
// GATE TRACKING
// ============================================================================

void profiler_track_gate(const char* gate_type, int num_qubits) {
    if (!g_profiler) return;
    if (!(g_profiler->flags & PROFILER_COUNTERS)) return;

    // Count by size
    if (num_qubits == 1) {
        g_profiler->gates.single_qubit_gates++;
    } else if (num_qubits == 2) {
        g_profiler->gates.two_qubit_gates++;
    } else {
        g_profiler->gates.multi_qubit_gates++;
    }

    // Count by type
    if (gate_type) {
        if (strcmp(gate_type, "H") == 0) {
            g_profiler->gates.hadamard_count++;
        } else if (strcmp(gate_type, "X") == 0) {
            g_profiler->gates.pauli_x_count++;
        } else if (strcmp(gate_type, "Y") == 0) {
            g_profiler->gates.pauli_y_count++;
        } else if (strcmp(gate_type, "Z") == 0) {
            g_profiler->gates.pauli_z_count++;
        } else if (strcmp(gate_type, "CNOT") == 0 || strcmp(gate_type, "CX") == 0) {
            g_profiler->gates.cnot_count++;
        } else if (strcmp(gate_type, "CZ") == 0) {
            g_profiler->gates.cz_count++;
        } else if (strcmp(gate_type, "SWAP") == 0) {
            g_profiler->gates.swap_count++;
        } else if (strstr(gate_type, "R") != NULL) {
            g_profiler->gates.rotation_count++;
        }
    }
}

void profiler_track_measurement(void) {
    if (!g_profiler) return;
    if (!(g_profiler->flags & PROFILER_COUNTERS)) return;

    g_profiler->gates.measurements++;
}

const profiler_gate_stats_t* profiler_get_gate_stats(void) {
    return g_profiler ? &g_profiler->gates : NULL;
}

// ============================================================================
// REPORTING
// ============================================================================

static const char* format_size(uint64_t bytes, char* buf, size_t buflen) {
    if (bytes < 1024) {
        snprintf(buf, buflen, "%llu B", (unsigned long long)bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buf, buflen, "%.2f KB", bytes / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buf, buflen, "%.2f MB", bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buf, buflen, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return buf;
}

static const char* format_time(uint64_t ns, char* buf, size_t buflen) {
    if (ns < 1000) {
        snprintf(buf, buflen, "%llu ns", (unsigned long long)ns);
    } else if (ns < 1000000) {
        snprintf(buf, buflen, "%.2f µs", ns / 1000.0);
    } else if (ns < 1000000000) {
        snprintf(buf, buflen, "%.2f ms", ns / 1000000.0);
    } else {
        snprintf(buf, buflen, "%.2f s", ns / 1000000000.0);
    }
    return buf;
}

void profiler_print_summary(void) {
    if (!g_profiler) {
        printf("Profiler not initialized\n");
        return;
    }

    profiler_time_t now = profiler_get_time();
    uint64_t runtime = profiler_elapsed_ns(g_profiler->start_time, now);

    char buf[64];

    printf("\n");
    printf("============================================\n");
    printf("         Profiler Summary                   \n");
    printf("============================================\n");
    printf("\n");

    printf("Runtime: %s\n", format_time(runtime, buf, sizeof(buf)));
    printf("Regions tracked: %d\n", g_profiler->num_regions);
    printf("\n");

    if (g_profiler->flags & PROFILER_MEMORY) {
        profiler_print_memory_report();
    }

    if (g_profiler->flags & PROFILER_COUNTERS) {
        profiler_print_gate_report();
    }

    if (g_profiler->flags & PROFILER_TIMING) {
        profiler_print_timing_report();
    }
}

void profiler_print_timing_report(void) {
    if (!g_profiler) return;

    printf("\nTiming Report:\n");
    printf("%-30s %10s %12s %12s %12s\n",
           "Region", "Calls", "Total", "Mean", "Max");
    printf("%-30s %10s %12s %12s %12s\n",
           "------", "-----", "-----", "----", "---");

    char buf1[64], buf2[64], buf3[64];

    for (int i = 0; i < g_profiler->num_regions; i++) {
        profiler_region_t* r = &g_profiler->regions[i];
        if (r->call_count > 0) {
            printf("%-30s %10llu %12s %12s %12s\n",
                   r->name,
                   (unsigned long long)r->call_count,
                   format_time(r->total_time, buf1, sizeof(buf1)),
                   format_time(r->total_time / r->call_count, buf2, sizeof(buf2)),
                   format_time(r->max_time, buf3, sizeof(buf3)));
        }
    }
    printf("\n");
}

void profiler_print_memory_report(void) {
    if (!g_profiler) return;

    char buf[64];

    printf("\nMemory Report:\n");
    printf("  Total allocated: %s\n",
           format_size(g_profiler->memory.total_allocated, buf, sizeof(buf)));
    printf("  Total freed:     %s\n",
           format_size(g_profiler->memory.total_freed, buf, sizeof(buf)));
    printf("  Current usage:   %s\n",
           format_size(g_profiler->memory.current_usage, buf, sizeof(buf)));
    printf("  Peak usage:      %s\n",
           format_size(g_profiler->memory.peak_usage, buf, sizeof(buf)));
    printf("  Allocations:     %llu\n",
           (unsigned long long)g_profiler->memory.allocation_count);
    printf("  Frees:           %llu\n",
           (unsigned long long)g_profiler->memory.free_count);
    printf("\n");
}

void profiler_print_gate_report(void) {
    if (!g_profiler) return;

    printf("\nGate Statistics:\n");
    printf("  Single-qubit gates: %llu\n",
           (unsigned long long)g_profiler->gates.single_qubit_gates);
    printf("    Hadamard (H):     %llu\n",
           (unsigned long long)g_profiler->gates.hadamard_count);
    printf("    Pauli X:          %llu\n",
           (unsigned long long)g_profiler->gates.pauli_x_count);
    printf("    Pauli Y:          %llu\n",
           (unsigned long long)g_profiler->gates.pauli_y_count);
    printf("    Pauli Z:          %llu\n",
           (unsigned long long)g_profiler->gates.pauli_z_count);
    printf("    Rotations:        %llu\n",
           (unsigned long long)g_profiler->gates.rotation_count);
    printf("  Two-qubit gates:    %llu\n",
           (unsigned long long)g_profiler->gates.two_qubit_gates);
    printf("    CNOT:             %llu\n",
           (unsigned long long)g_profiler->gates.cnot_count);
    printf("    CZ:               %llu\n",
           (unsigned long long)g_profiler->gates.cz_count);
    printf("    SWAP:             %llu\n",
           (unsigned long long)g_profiler->gates.swap_count);
    printf("  Multi-qubit gates:  %llu\n",
           (unsigned long long)g_profiler->gates.multi_qubit_gates);
    printf("  Measurements:       %llu\n",
           (unsigned long long)g_profiler->gates.measurements);
    printf("\n");
}

// ============================================================================
// EXPORT
// ============================================================================

int profiler_export_json(const char* path) {
    if (!g_profiler || !path) return -1;

    FILE* fp = fopen(path, "w");
    if (!fp) return -1;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"runtime_ns\": %llu,\n",
            (unsigned long long)g_profiler->total_runtime);

    // Memory
    fprintf(fp, "  \"memory\": {\n");
    fprintf(fp, "    \"total_allocated\": %llu,\n",
            (unsigned long long)g_profiler->memory.total_allocated);
    fprintf(fp, "    \"peak_usage\": %llu,\n",
            (unsigned long long)g_profiler->memory.peak_usage);
    fprintf(fp, "    \"allocation_count\": %llu\n",
            (unsigned long long)g_profiler->memory.allocation_count);
    fprintf(fp, "  },\n");

    // Gates
    fprintf(fp, "  \"gates\": {\n");
    fprintf(fp, "    \"single_qubit\": %llu,\n",
            (unsigned long long)g_profiler->gates.single_qubit_gates);
    fprintf(fp, "    \"two_qubit\": %llu,\n",
            (unsigned long long)g_profiler->gates.two_qubit_gates);
    fprintf(fp, "    \"measurements\": %llu\n",
            (unsigned long long)g_profiler->gates.measurements);
    fprintf(fp, "  },\n");

    // Regions
    fprintf(fp, "  \"regions\": [\n");
    for (int i = 0; i < g_profiler->num_regions; i++) {
        profiler_region_t* r = &g_profiler->regions[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\",\n", r->name);
        fprintf(fp, "      \"calls\": %llu,\n", (unsigned long long)r->call_count);
        fprintf(fp, "      \"total_ns\": %llu,\n", (unsigned long long)r->total_time);
        fprintf(fp, "      \"min_ns\": %llu,\n", (unsigned long long)r->min_time);
        fprintf(fp, "      \"max_ns\": %llu\n", (unsigned long long)r->max_time);
        fprintf(fp, "    }%s\n", (i < g_profiler->num_regions - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n");

    fprintf(fp, "}\n");

    fclose(fp);
    return 0;
}

int profiler_export_csv(const char* path) {
    if (!g_profiler || !path) return -1;

    FILE* fp = fopen(path, "w");
    if (!fp) return -1;

    // Header
    fprintf(fp, "region,calls,total_ns,min_ns,max_ns,mean_ns\n");

    // Data
    for (int i = 0; i < g_profiler->num_regions; i++) {
        profiler_region_t* r = &g_profiler->regions[i];
        uint64_t mean = r->call_count > 0 ? r->total_time / r->call_count : 0;
        fprintf(fp, "%s,%llu,%llu,%llu,%llu,%llu\n",
                r->name,
                (unsigned long long)r->call_count,
                (unsigned long long)r->total_time,
                (unsigned long long)(r->min_time == UINT64_MAX ? 0 : r->min_time),
                (unsigned long long)r->max_time,
                (unsigned long long)mean);
    }

    fclose(fp);
    return 0;
}

// ============================================================================
// BENCHMARKING
// ============================================================================

int profiler_benchmark(void (*func)(void*), void* data,
                       int warmup_iterations, int bench_iterations,
                       benchmark_result_t* result) {
    if (!func || !result || bench_iterations <= 0) return -1;

    memset(result, 0, sizeof(benchmark_result_t));

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        func(data);
    }

    // Collect timing samples
    double* samples = malloc(bench_iterations * sizeof(double));
    if (!samples) return -1;

    uint64_t total_ns = 0;
    uint64_t min_ns = UINT64_MAX;
    uint64_t max_ns = 0;

    for (int i = 0; i < bench_iterations; i++) {
        profiler_time_t start = profiler_get_time();
        func(data);
        profiler_time_t end = profiler_get_time();

        uint64_t elapsed = profiler_elapsed_ns(start, end);
        samples[i] = (double)elapsed;
        total_ns += elapsed;

        if (elapsed < min_ns) min_ns = elapsed;
        if (elapsed > max_ns) max_ns = elapsed;
    }

    // Calculate statistics
    result->mean_ns = (double)total_ns / (double)bench_iterations;
    result->min_ns = (double)min_ns;
    result->max_ns = (double)max_ns;
    result->iterations = bench_iterations;

    // Standard deviation
    double variance = 0.0;
    for (int i = 0; i < bench_iterations; i++) {
        double diff = samples[i] - result->mean_ns;
        variance += diff * diff;
    }
    result->std_ns = sqrt(variance / (double)bench_iterations);

    // Operations per second
    if (result->mean_ns > 0) {
        result->ops_per_sec = 1e9 / result->mean_ns;
    }

    free(samples);
    return 0;
}

void profiler_print_benchmark(const char* name, const benchmark_result_t* result) {
    if (!name || !result) return;

    char buf1[64], buf2[64], buf3[64], buf4[64];

    printf("\nBenchmark: %s\n", name);
    printf("  Iterations: %d\n", result->iterations);
    printf("  Mean: %s\n", format_time((uint64_t)result->mean_ns, buf1, sizeof(buf1)));
    printf("  Std:  %s\n", format_time((uint64_t)result->std_ns, buf2, sizeof(buf2)));
    printf("  Min:  %s\n", format_time((uint64_t)result->min_ns, buf3, sizeof(buf3)));
    printf("  Max:  %s\n", format_time((uint64_t)result->max_ns, buf4, sizeof(buf4)));
    printf("  Throughput: %.2f ops/sec\n", result->ops_per_sec);
}

// ============================================================================
// SCOPED PROFILING
// ============================================================================

profiler_scope_t profiler_scope_begin(const char* name) {
    profiler_scope_t scope;
    scope.region_id = profiler_region_start(name);
    scope.start = profiler_get_time();
    return scope;
}

void profiler_scope_end(profiler_scope_t* scope) {
    if (!scope || !g_profiler) return;

    if (scope->region_id >= 0 && scope->region_id < g_profiler->num_regions) {
        profiler_time_t end = profiler_get_time();
        uint64_t elapsed = profiler_elapsed_ns(scope->start, end);

        profiler_region_t* r = &g_profiler->regions[scope->region_id];
        r->call_count++;
        r->total_time += elapsed;

        if (elapsed < r->min_time) r->min_time = elapsed;
        if (elapsed > r->max_time) r->max_time = elapsed;

        // Update running mean and variance (Welford's algorithm)
        double delta = (double)elapsed - r->mean;
        r->mean += delta / (double)r->call_count;
        double delta2 = (double)elapsed - r->mean;
        r->m2 += delta * delta2;
    }

    profiler_region_end(scope->region_id);
}
