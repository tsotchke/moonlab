/**
 * @file test_correctness_properties.c
 * @brief Property-based correctness checks for core simulator primitives.
 *
 * These tests exercise invariants that must hold regardless of the exact
 * implementation:
 *
 *  1. Unitary gates preserve state norm exactly (to within floating-point
 *     round-off).
 *  2. Applying a single-qubit rotation followed by its inverse returns the
 *     state to within 1e-12 of its original amplitudes.
 *  3. Hadamard squared is the identity.
 *  4. Pauli-X and Pauli-Z squared are the identity.
 *  5. CNOT squared on any 2-qubit state is the identity.
 *  6. `create_bell_state_phi_plus` produces exactly the amplitudes
 *     [1/sqrt(2), 0, 0, 1/sqrt(2)] to within 1e-14.
 *
 * This file pins down the physical correctness of the lowest-level
 * primitives that every higher-level feature depends on. A regression here
 * is a red-alert scientific-correctness bug.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/algorithms/bell_tests.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static double norm_squared(const quantum_state_t* s) {
    double n = 0.0;
    for (size_t i = 0; i < s->state_dim; ++i) {
        double m = cabs(s->amplitudes[i]);
        n += m * m;
    }
    return n;
}

static double l2_amplitude_diff(const quantum_state_t* a,
                                const quantum_state_t* b) {
    if (a->state_dim != b->state_dim) return INFINITY;
    double acc = 0.0;
    for (size_t i = 0; i < a->state_dim; ++i) {
        complex_t d = a->amplitudes[i] - b->amplitudes[i];
        acc += cabs(d) * cabs(d);
    }
    return sqrt(acc);
}

static void randomize_state(quantum_state_t* s) {
    double norm = 0.0;
    for (size_t i = 0; i < s->state_dim; ++i) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        s->amplitudes[i] = re + I * im;
        norm += re * re + im * im;
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < s->state_dim; ++i) {
        s->amplitudes[i] /= norm;
    }
}

static void test_norm_preservation(void) {
    fprintf(stdout, "\n-- norm preservation under gate application --\n");
    srand(0xC0FFEE);
    const int N = 3;
    quantum_state_t s;
    quantum_state_init(&s, N);

    for (int trial = 0; trial < 50; ++trial) {
        randomize_state(&s);
        double angle = 2.0 * M_PI * ((double)rand() / RAND_MAX);
        int q = rand() % N;
        gate_rx(&s, q, angle);
        gate_ry(&s, q, angle * 0.5);
        gate_rz(&s, q, angle * 0.25);
        gate_hadamard(&s, q);
        gate_cnot(&s, q, (q + 1) % N);
        double n2 = norm_squared(&s);
        if (fabs(n2 - 1.0) > 1e-10) {
            fprintf(stderr,
                    "  FAIL  trial %d: norm^2 = %.17g (drift %.3e)\n",
                    trial, n2, fabs(n2 - 1.0));
            failures++;
            break;
        }
    }
    fprintf(stdout, "  OK    50 random circuits preserve ||psi||^2 = 1\n");
    quantum_state_free(&s);
}

static void test_rotation_inverse(void) {
    fprintf(stdout, "\n-- rotation gate inverses --\n");
    srand(0xBEEF);
    const int N = 3;
    quantum_state_t s, original;
    quantum_state_init(&s, N);

    for (int trial = 0; trial < 50; ++trial) {
        randomize_state(&s);
        quantum_state_clone(&original, &s);

        double theta = 4.0 * M_PI * ((double)rand() / RAND_MAX) - 2.0 * M_PI;
        int q = rand() % N;
        int kind = rand() % 3;

        if (kind == 0)      { gate_rx(&s, q, theta);  gate_rx(&s, q, -theta); }
        else if (kind == 1) { gate_ry(&s, q, theta);  gate_ry(&s, q, -theta); }
        else                { gate_rz(&s, q, theta);  gate_rz(&s, q, -theta); }

        double diff = l2_amplitude_diff(&s, &original);
        if (diff > 1e-12) {
            fprintf(stderr,
                    "  FAIL  trial %d (kind=%d): ||U(theta)U(-theta)|psi> - |psi>||_2 = %.3e\n",
                    trial, kind, diff);
            failures++;
            quantum_state_free(&original);
            break;
        }
        quantum_state_free(&original);
    }
    fprintf(stdout, "  OK    50 random Rx/Ry/Rz +/-theta inverse pairs restore |psi> (L2 < 1e-12)\n");
    quantum_state_free(&s);
}

static void test_involutions(void) {
    fprintf(stdout, "\n-- HH = XX = ZZ = CNOT*CNOT = I --\n");
    srand(0x1234);
    const int N = 2;
    quantum_state_t s, original;
    quantum_state_init(&s, N);

    const char* names[] = { "HH", "XX", "ZZ", "CNOTxCNOT" };
    for (int kind = 0; kind < 4; ++kind) {
        for (int trial = 0; trial < 25; ++trial) {
            randomize_state(&s);
            quantum_state_clone(&original, &s);

            if (kind == 0)      { gate_hadamard(&s, 0); gate_hadamard(&s, 0); }
            else if (kind == 1) { gate_pauli_x(&s, 0);  gate_pauli_x(&s, 0);  }
            else if (kind == 2) { gate_pauli_z(&s, 0);  gate_pauli_z(&s, 0);  }
            else                { gate_cnot(&s, 0, 1);  gate_cnot(&s, 0, 1);  }

            double diff = l2_amplitude_diff(&s, &original);
            if (diff > 1e-14) {
                fprintf(stderr,
                        "  FAIL  %s trial %d: L2 diff = %.3e\n",
                        names[kind], trial, diff);
                failures++;
                quantum_state_free(&original);
                quantum_state_free(&s);
                return;
            }
            quantum_state_free(&original);
        }
        fprintf(stdout, "  OK    %-9s is the identity over 25 random inputs (L2 < 1e-14)\n",
                names[kind]);
    }
    quantum_state_free(&s);
}

static void test_swap_involution(void) {
    fprintf(stdout, "\n-- SWAP(a,b) applied twice is identity --\n");
    srand(0xFEED);
    quantum_state_t s, original;
    quantum_state_init(&s, 3);
    for (int trial = 0; trial < 20; ++trial) {
        randomize_state(&s);
        quantum_state_clone(&original, &s);
        int a = rand() % 3, b = (a + 1 + rand() % 2) % 3;
        gate_swap(&s, a, b);
        gate_swap(&s, a, b);
        double diff = l2_amplitude_diff(&s, &original);
        if (diff > 1e-14) {
            fprintf(stderr,
                    "  FAIL  SWAP(%d,%d) trial %d diff %.3e\n",
                    a, b, trial, diff);
            failures++;
            quantum_state_free(&original);
            quantum_state_free(&s);
            return;
        }
        quantum_state_free(&original);
    }
    fprintf(stdout, "  OK    20 random SWAP pairs are identity (L2 < 1e-14)\n");
    quantum_state_free(&s);
}

static void test_qft_iqft_roundtrip(void) {
    fprintf(stdout, "\n-- QFT . IQFT = I on random states --\n");
    srand(0xFACADE);
    for (int nq = 2; nq <= 4; ++nq) {
        quantum_state_t s, original;
        quantum_state_init(&s, nq);
        randomize_state(&s);
        quantum_state_clone(&original, &s);

        int qubits[8];
        for (int q = 0; q < nq; ++q) qubits[q] = q;

        gate_qft(&s, qubits, nq);
        gate_iqft(&s, qubits, nq);

        double diff = l2_amplitude_diff(&s, &original);
        if (diff > 1e-12) {
            fprintf(stderr,
                    "  FAIL  n=%d QFT/IQFT round-trip L2 diff %.3e > 1e-12\n",
                    nq, diff);
            failures++;
        } else {
            fprintf(stdout,
                    "  OK    n=%d: QFT . IQFT matches input (L2 = %.3e)\n",
                    nq, diff);
        }
        quantum_state_free(&original);
        quantum_state_free(&s);
    }
}

static void test_ghz_bipartite_entanglement(void) {
    fprintf(stdout, "\n-- GHZ state bipartite entropy = 1 ebit --\n");
    /* |GHZ> = (|000> + |111>)/sqrt(2). Any bipartition (A = {0},
     * B = {1,2}) gives maximally mixed reduced density rho_A = I/2
     * with S(rho_A) = 1 bit. Exercises zheev_ in entanglement.c. */
    for (int nq = 3; nq <= 5; ++nq) {
        quantum_state_t s;
        quantum_state_init(&s, nq);
        gate_hadamard(&s, 0);
        for (int q = 1; q < nq; ++q) gate_cnot(&s, 0, q);

        int subsys[1] = { 0 };
        double S = quantum_state_entanglement_entropy(&s, subsys, 1);
        if (fabs(S - 1.0) > 1e-10) {
            fprintf(stderr,
                    "  FAIL  n=%d GHZ entropy = %.10f, expected 1.0\n",
                    nq, S);
            failures++;
        } else {
            fprintf(stdout,
                    "  OK    n=%d GHZ: S(rho_A) = %.10f ebit\n", nq, S);
        }
        quantum_state_free(&s);
    }
}

static void test_bell_phi_plus_amplitudes(void) {
    fprintf(stdout, "\n-- exact amplitudes of |Phi+> --\n");
    quantum_state_t s;
    quantum_state_init(&s, 2);
    create_bell_state_phi_plus(&s, 0, 1);

    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    struct { size_t idx; complex_t expected; } exp[] = {
        { 0, inv_sqrt2 + 0.0 * I },
        { 1, 0.0 + 0.0 * I },
        { 2, 0.0 + 0.0 * I },
        { 3, inv_sqrt2 + 0.0 * I },
    };

    for (size_t i = 0; i < 4; ++i) {
        complex_t got  = s.amplitudes[exp[i].idx];
        complex_t want = exp[i].expected;
        double diff = cabs(got - want);
        if (diff > 1e-14) {
            fprintf(stderr,
                    "  FAIL  idx %zu: got %.17g+%.17gi  want %.17g+%.17gi  diff %.3e\n",
                    exp[i].idx, creal(got), cimag(got),
                    creal(want), cimag(want), diff);
            failures++;
        } else {
            fprintf(stdout,
                    "  OK    idx %zu: %.17g + %.17gi\n",
                    exp[i].idx, creal(got), cimag(got));
        }
    }
    quantum_state_free(&s);
}

int main(void) {
    fprintf(stdout, "=== core correctness property tests ===\n");
    test_norm_preservation();
    test_rotation_inverse();
    test_involutions();
    test_swap_involution();
    test_qft_iqft_roundtrip();
    test_ghz_bipartite_entanglement();
    test_bell_phi_plus_amplitudes();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
