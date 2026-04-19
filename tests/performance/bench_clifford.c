/**
 * @file bench_clifford.c
 * @brief Clifford backend throughput benchmark.
 *
 * Exercises the Aaronson-Gottesman tableau at scales well beyond the
 * dense simulator's 32-qubit ceiling.  For each n:
 *   - Builds a GHZ circuit (1 H + (n-1) CNOTs) and measures all qubits.
 *   - Applies n random Clifford 2-qubit gates on random pairs.
 * Reports wall-clock per gate and per measurement.
 *
 * Expected asymptotics (AG paper): O(n^2) per gate, O(n^2) per
 * measurement, O(n^3) per sample_all.  Nothing exponential.
 */

#include "../../src/backends/clifford/clifford.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static uint64_t xrng = 0x1234ABCD5678EF01ULL;
static uint32_t rng_u32(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return (uint32_t)xrng;
}

static void bench_ghz(size_t n) {
    clifford_tableau_t* t = clifford_tableau_create(n);
    uint64_t rng = 0xBADDCAFEu;

    double t0 = now_us();
    clifford_h(t, 0);
    for (size_t q = 1; q < n; q++) clifford_cnot(t, 0, q);
    double t_prep = now_us() - t0;

    double t1 = now_us();
    int first = 0;
    clifford_measure(t, 0, &rng, &first, NULL);
    int consistent = 1;
    for (size_t q = 1; q < n; q++) {
        int b = -1;
        clifford_measure(t, q, &rng, &b, NULL);
        if (b != first) consistent = 0;
    }
    double t_meas = now_us() - t1;

    size_t n_gates = n; /* 1 H + (n-1) CNOTs */
    printf("  n=%-6zu  GHZ prep: %.2f ms (%.1f us/gate)  "
           "measure-all: %.2f ms (%.1f us/qubit)  consistent=%s\n",
           n, t_prep / 1000.0, t_prep / (double)n_gates,
           t_meas / 1000.0, t_meas / (double)n,
           consistent ? "YES" : "NO");

    clifford_tableau_free(t);
}

static void bench_random_cliffords(size_t n, size_t num_gates) {
    clifford_tableau_t* t = clifford_tableau_create(n);
    double t0 = now_us();
    for (size_t i = 0; i < num_gates; i++) {
        size_t a = rng_u32() % n;
        size_t b = rng_u32() % n;
        if (a == b) b = (a + 1) % n;
        switch (rng_u32() % 6) {
            case 0: clifford_cnot(t, a, b); break;
            case 1: clifford_cz(t, a, b); break;
            case 2: clifford_swap(t, a, b); break;
            case 3: clifford_h(t, a); break;
            case 4: clifford_s(t, a); break;
            case 5: clifford_x(t, a); break;
        }
    }
    double dt = now_us() - t0;
    printf("  n=%-6zu  random Cliffords: %zu gates in %.2f ms  "
           "(%.2f us/gate)\n",
           n, num_gates, dt / 1000.0, dt / (double)num_gates);

    clifford_tableau_free(t);
}

int main(void) {
    printf("=== Clifford backend throughput ===\n");
    printf("GHZ-n preparation + all-qubits measurement:\n");
    for (size_t n = 100; n <= 4000; n *= 2) bench_ghz(n);
    printf("\nRandom Clifford streams:\n");
    for (size_t n = 100; n <= 2000; n *= 2) bench_random_cliffords(n, 2 * n);
    return 0;
}
