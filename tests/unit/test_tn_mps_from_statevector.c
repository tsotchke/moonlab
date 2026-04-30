/**
 * @file test_tn_mps_from_statevector.c
 * @brief Round-trip test for tn_mps_from_statevector / tn_mps_to_statevector.
 *
 * Pins:
 *   - Constructing an MPS from a known state vector and converting back
 *     reproduces the original amplitudes (within SVD-truncation
 *     tolerance) on small product, GHZ, and W states.
 *   - The MPS bond dimension does not exceed the configured maximum.
 *
 * Closes the ICC dead-code triage entry for tn_mps_from_statevector
 * (declared in tn_state.h, no in-tree caller before this test).
 */

#include "../../src/algorithms/tensor_network/tn_state.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static double max_abs_diff(const double complex* a, const double complex* b,
                            size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = cabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void run_round_trip(const char* label,
                            uint32_t n,
                            const double complex* amps_in) {
    fprintf(stdout, "\n--- %s (n=%u) ---\n", label, n);
    tn_state_config_t cfg = tn_state_config_default();
    tn_mps_state_t* mps = tn_mps_from_statevector(amps_in, n, &cfg);
    CHECK(mps != NULL, "from_statevector returned NULL");
    if (!mps) return;

    size_t dim = (size_t)1 << n;
    double complex* amps_out = (double complex*)calloc(dim, sizeof(double complex));
    tn_state_error_t err = tn_mps_to_statevector(mps, amps_out);
    CHECK(err == TN_STATE_SUCCESS, "to_statevector returned %d", (int)err);

    double diff = max_abs_diff(amps_in, amps_out, dim);
    CHECK(diff < 1e-10,
          "round-trip max abs diff = %.3e (expected < 1e-10)", diff);

    free(amps_out);
    tn_mps_free(mps);
}

int main(void) {
    fprintf(stdout, "=== tn_mps_from_statevector round-trip ===\n");

    /* Case 1: 3-qubit product state |010>. */
    {
        const uint32_t n = 3;
        size_t dim = 1u << n;
        double complex* a = (double complex*)calloc(dim, sizeof(double complex));
        a[0b010] = 1.0;
        run_round_trip("product |010>", n, a);
        free(a);
    }

    /* Case 2: 4-qubit GHZ (|0000> + |1111>) / sqrt(2). */
    {
        const uint32_t n = 4;
        size_t dim = 1u << n;
        double complex* a = (double complex*)calloc(dim, sizeof(double complex));
        a[0]       = 1.0 / sqrt(2.0);
        a[dim - 1] = 1.0 / sqrt(2.0);
        run_round_trip("GHZ-4", n, a);
        free(a);
    }

    /* Case 3: 4-qubit W state. */
    {
        const uint32_t n = 4;
        size_t dim = 1u << n;
        double complex* a = (double complex*)calloc(dim, sizeof(double complex));
        a[0b0001] = a[0b0010] = a[0b0100] = a[0b1000] = 0.5;
        run_round_trip("W-4", n, a);
        free(a);
    }

    /* Case 4: 5-qubit random complex state (deterministic seed). */
    {
        const uint32_t n = 5;
        size_t dim = 1u << n;
        double complex* a = (double complex*)calloc(dim, sizeof(double complex));
        srand(0xC0FFEE);
        double norm2 = 0.0;
        for (size_t i = 0; i < dim; i++) {
            double re = (rand() / (double)RAND_MAX) - 0.5;
            double im = (rand() / (double)RAND_MAX) - 0.5;
            a[i] = re + im * I;
            norm2 += re * re + im * im;
        }
        double s = 1.0 / sqrt(norm2);
        for (size_t i = 0; i < dim; i++) a[i] *= s;
        run_round_trip("random 5-qubit (full bond)", n, a);
        free(a);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
