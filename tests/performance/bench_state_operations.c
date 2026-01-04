/**
 * @file bench_state_operations.c
 * @brief Performance benchmarks for quantum state operations
 *
 * Measures performance of:
 * - State initialization at various qubit counts
 * - Single-qubit gate application
 * - Two-qubit gate application
 * - Measurement operations
 * - State normalization
 *
 * @stability stable
 * @since v1.0.0
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/utils/quantum_entropy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define WARMUP_ITERATIONS 3
#define BENCH_ITERATIONS 10

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

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        func(data);
    }

    // Timed runs
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        func(data);
        double end = get_time_us();
        times[i] = end - start;
    }

    // Compute statistics
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
    printf("  %-40s %10.2f μs (±%.2f, min=%.2f, max=%.2f)\n",
           name, r->avg_us, r->std_us, r->min_us, r->max_us);
}

// ============================================================================
// STATE INITIALIZATION BENCHMARKS
// ============================================================================

typedef struct {
    quantum_state_t state;
    size_t num_qubits;
} init_data_t;

static void bench_state_init(void *data) {
    init_data_t *d = (init_data_t *)data;
    quantum_state_init(&d->state, d->num_qubits);
    quantum_state_free(&d->state);
}

static void benchmark_state_initialization(void) {
    printf("\n=== State Initialization Benchmarks ===\n");

    int qubit_counts[] = {10, 12, 14, 16, 18, 20};
    int num_counts = sizeof(qubit_counts) / sizeof(qubit_counts[0]);

    for (int i = 0; i < num_counts; i++) {
        init_data_t data = {.num_qubits = qubit_counts[i]};

        char name[64];
        snprintf(name, sizeof(name), "Init %d qubits (2^%d states)",
                 qubit_counts[i], qubit_counts[i]);

        bench_result_t r = run_benchmark(bench_state_init, &data, BENCH_ITERATIONS);
        print_result(name, &r);
    }
}

// ============================================================================
// SINGLE-QUBIT GATE BENCHMARKS
// ============================================================================

typedef struct {
    quantum_state_t *state;
    size_t target;
} gate_data_t;

static void bench_hadamard(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    gate_hadamard(d->state, d->target);
}

static void bench_x_gate(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    gate_x(d->state, d->target);
}

static void bench_t_gate(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    gate_t(d->state, d->target);
}

static void bench_rz_gate(void *data) {
    gate_data_t *d = (gate_data_t *)data;
    gate_rz(d->state, d->target, 0.5);
}

static void benchmark_single_qubit_gates(void) {
    printf("\n=== Single-Qubit Gate Benchmarks (16 qubits) ===\n");

    quantum_state_t state;
    quantum_state_init(&state, 16);

    gate_data_t data = {.state = &state, .target = 8};

    bench_result_t r;

    r = run_benchmark(bench_hadamard, &data, BENCH_ITERATIONS);
    print_result("Hadamard gate", &r);

    r = run_benchmark(bench_x_gate, &data, BENCH_ITERATIONS);
    print_result("Pauli-X gate", &r);

    r = run_benchmark(bench_t_gate, &data, BENCH_ITERATIONS);
    print_result("T gate", &r);

    r = run_benchmark(bench_rz_gate, &data, BENCH_ITERATIONS);
    print_result("Rz(0.5) gate", &r);

    quantum_state_free(&state);
}

// ============================================================================
// TWO-QUBIT GATE BENCHMARKS
// ============================================================================

typedef struct {
    quantum_state_t *state;
    size_t control;
    size_t target;
} two_gate_data_t;

static void bench_cnot(void *data) {
    two_gate_data_t *d = (two_gate_data_t *)data;
    gate_cnot(d->state, d->control, d->target);
}

static void bench_cz(void *data) {
    two_gate_data_t *d = (two_gate_data_t *)data;
    gate_cz(d->state, d->control, d->target);
}

static void bench_swap(void *data) {
    two_gate_data_t *d = (two_gate_data_t *)data;
    gate_swap(d->state, d->control, d->target);
}

static void benchmark_two_qubit_gates(void) {
    printf("\n=== Two-Qubit Gate Benchmarks (16 qubits) ===\n");

    quantum_state_t state;
    quantum_state_init(&state, 16);

    // Prepare non-trivial state
    for (int q = 0; q < 16; q++) {
        gate_hadamard(&state, q);
    }

    two_gate_data_t data = {.state = &state, .control = 0, .target = 15};

    bench_result_t r;

    r = run_benchmark(bench_cnot, &data, BENCH_ITERATIONS);
    print_result("CNOT (adjacent)", &r);

    data.control = 0;
    data.target = 15;
    r = run_benchmark(bench_cnot, &data, BENCH_ITERATIONS);
    print_result("CNOT (long-range)", &r);

    r = run_benchmark(bench_cz, &data, BENCH_ITERATIONS);
    print_result("CZ (long-range)", &r);

    r = run_benchmark(bench_swap, &data, BENCH_ITERATIONS);
    print_result("SWAP", &r);

    quantum_state_free(&state);
}

// ============================================================================
// MEASUREMENT BENCHMARKS
// ============================================================================

static int bench_entropy_cb(void *ud, uint8_t *buf, size_t sz) {
    (void)ud;
    for (size_t i = 0; i < sz; i++) buf[i] = rand() & 0xFF;
    return 0;
}

typedef struct {
    quantum_state_t *state;
    quantum_entropy_ctx_t *entropy;
} measure_data_t;

static void bench_probability(void *data) {
    measure_data_t *d = (measure_data_t *)data;
    for (size_t i = 0; i < d->state->state_dim; i++) {
        quantum_state_get_probability(d->state, i);
    }
}

static void bench_normalize(void *data) {
    measure_data_t *d = (measure_data_t *)data;
    quantum_state_normalize(d->state);
}

static void benchmark_measurement(void) {
    printf("\n=== Measurement Benchmarks (14 qubits) ===\n");

    quantum_state_t state;
    quantum_state_init(&state, 14);

    // Prepare random state
    for (int q = 0; q < 14; q++) {
        gate_hadamard(&state, q);
    }

    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, bench_entropy_cb, NULL);

    measure_data_t data = {.state = &state, .entropy = &entropy};

    bench_result_t r;

    r = run_benchmark(bench_probability, &data, BENCH_ITERATIONS);
    print_result("Get all probabilities", &r);

    r = run_benchmark(bench_normalize, &data, BENCH_ITERATIONS);
    print_result("Normalize state", &r);

    quantum_state_free(&state);
}

// ============================================================================
// CIRCUIT DEPTH BENCHMARKS
// ============================================================================

typedef struct {
    quantum_state_t *state;
    int depth;
} circuit_data_t;

static void bench_random_circuit(void *data) {
    circuit_data_t *d = (circuit_data_t *)data;
    int n = d->state->num_qubits;

    for (int layer = 0; layer < d->depth; layer++) {
        // Single-qubit layer
        for (int q = 0; q < n; q++) {
            gate_hadamard(d->state, q);
        }

        // Entangling layer
        for (int q = 0; q < n - 1; q += 2) {
            gate_cnot(d->state, q, q + 1);
        }
        for (int q = 1; q < n - 1; q += 2) {
            gate_cnot(d->state, q, q + 1);
        }
    }
}

static void benchmark_circuits(void) {
    printf("\n=== Circuit Depth Benchmarks (12 qubits) ===\n");

    quantum_state_t state;
    quantum_state_init(&state, 12);

    int depths[] = {10, 50, 100, 200};
    int num_depths = sizeof(depths) / sizeof(depths[0]);

    for (int i = 0; i < num_depths; i++) {
        quantum_state_reset(&state);

        circuit_data_t data = {.state = &state, .depth = depths[i]};

        char name[64];
        snprintf(name, sizeof(name), "Depth-%d random circuit", depths[i]);

        bench_result_t r = run_benchmark(bench_random_circuit, &data, BENCH_ITERATIONS);
        print_result(name, &r);
    }

    quantum_state_free(&state);
}

// ============================================================================
// MAIN
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       QUANTUM STATE PERFORMANCE BENCHMARKS                   ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Measuring core operation performance                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    srand(42);

    benchmark_state_initialization();
    benchmark_single_qubit_gates();
    benchmark_two_qubit_gates();
    benchmark_measurement();
    benchmark_circuits();

    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  Benchmarks complete\n");
    printf("════════════════════════════════════════════════════════════════\n");

    return 0;
}
