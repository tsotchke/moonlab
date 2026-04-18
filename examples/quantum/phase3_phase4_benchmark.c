/**
 * @file phase3_phase4_benchmark.c
 * @brief State-vector gate throughput benchmark (measured).
 *
 * Times single- and two-qubit gates plus a Grover iteration across a
 * range of qubit counts. Reports wall-clock time and amplitudes/s
 * throughput as measured on this machine — no expected / theoretical
 * speedups. To compare backends, run twice and diff the numbers.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/algorithms/grover.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    const char *name;
    double seconds_per_gate;
    size_t dim;
} timing_t;

static timing_t bench_single_qubit(int num_qubits, int reps, const char *name,
                                   qs_error_t (*op)(quantum_state_t*, int)) {
    quantum_state_t s;
    quantum_state_init(&s, num_qubits);
    gate_hadamard(&s, 0);

    double t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        for (int q = 0; q < num_qubits; q++) {
            op(&s, q);
        }
    }
    double t1 = now_seconds();
    quantum_state_free(&s);
    timing_t out;
    out.name = name;
    out.seconds_per_gate = (t1 - t0) / ((double)reps * num_qubits);
    out.dim = (size_t)1 << num_qubits;
    return out;
}

static timing_t bench_cnot(int num_qubits, int reps) {
    quantum_state_t s;
    quantum_state_init(&s, num_qubits);
    gate_hadamard(&s, 0);
    double t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        for (int q = 0; q + 1 < num_qubits; q++) {
            gate_cnot(&s, q, q + 1);
        }
    }
    double t1 = now_seconds();
    quantum_state_free(&s);
    timing_t out;
    out.name = "cnot";
    out.seconds_per_gate = (t1 - t0) / ((double)reps * (num_qubits - 1));
    out.dim = (size_t)1 << num_qubits;
    return out;
}

static void print_row(int n, timing_t t) {
    double us = t.seconds_per_gate * 1e6;
    double amp_per_sec = (double)t.dim / t.seconds_per_gate;
    printf("  %-10s  n=%2d  dim=%10zu  %9.2f us/gate  %10.2e amp/s\n",
           t.name, n, t.dim, us, amp_per_sec);
}

static void bench_grover(int n_qubits) {
    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    quantum_state_t s;
    quantum_state_init(&s, n_qubits);
    for (int q = 0; q < n_qubits; q++) gate_hadamard(&s, q);

    grover_config_t cfg = {0};
    cfg.marked_state = ((uint64_t)1 << n_qubits) - 1;
    cfg.num_qubits = (size_t)n_qubits;
    cfg.num_iterations = grover_optimal_iterations((size_t)n_qubits);
    cfg.use_optimal_iterations = 1;

    double t0 = now_seconds();
    grover_result_t r = grover_search(&s, &cfg, (void*)&e);
    (void)r;
    double t1 = now_seconds();
    double success = 0.0;
    if (cfg.marked_state < s.state_dim) {
        double mag = cabs(s.amplitudes[cfg.marked_state]);
        success = mag * mag;
    }
    printf("  grover     n=%2d  iterations=%-4zu  %8.3f s  P(marked)=%.4f\n",
           n_qubits, cfg.num_iterations, t1 - t0, success);
    quantum_state_free(&s);
}

int main(int argc, char *argv[]) {
    int max_n = 20;
    if (argc > 1) max_n = atoi(argv[1]);
    if (max_n < 4) max_n = 4;
    if (max_n > 28) max_n = 28;

    printf("=== state-vector gate throughput (measured) ===\n");
    printf("  max qubits: %d (override via argv[1])\n", max_n);
    printf("  one n=24 run needs ~256 MiB of state-vector memory.\n\n");

    int ns[] = {8, 12, 16, 20, 24};
    for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); i++) {
        int n = ns[i];
        if (n > max_n) break;
        int reps = (n <= 12) ? 50 : (n <= 16 ? 10 : 2);

        timing_t h = bench_single_qubit(n, reps, "hadamard", gate_hadamard);
        timing_t x = bench_single_qubit(n, reps, "pauli_x",  gate_pauli_x);
        timing_t z = bench_single_qubit(n, reps, "pauli_z",  gate_pauli_z);
        timing_t c = bench_cnot(n, reps);
        print_row(n, h);
        print_row(n, x);
        print_row(n, z);
        print_row(n, c);
    }

    printf("\n=== Grover (measured) ===\n");
    int grover_sizes[] = {10, 14, 18};
    for (size_t i = 0; i < sizeof(grover_sizes) / sizeof(grover_sizes[0]); i++) {
        if (grover_sizes[i] > max_n) break;
        bench_grover(grover_sizes[i]);
    }
    return 0;
}
