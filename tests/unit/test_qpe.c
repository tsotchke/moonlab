/**
 * @file test_qpe.c
 * @brief QPE smoke test: recovers a known phase on a phase-gate unitary.
 *
 * Exercises:
 *  - `qpe_bitstring_to_phase` round-trips phase <-> bitstring.
 *  - `qpe_estimate_phase` on the pre-built T gate (phi = 1/8) recovers
 *    the eigenphase to within 2 * 2^-m precision.
 */

#include "../../src/algorithms/qpe.h"
#include "../../src/quantum/gates.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_phase_bitstring_roundtrip(void) {
    fprintf(stdout, "\n-- QPE: phase <-> bitstring round trip --\n");
    /* phi = y / 2^m where y = integer stored with bit k == qubit k outcome. */
    double phi = qpe_bitstring_to_phase(1, 4);
    CHECK(fabs(phi - 1.0/16.0) < 1e-15,
          "bitstring=1 m=4 -> phi=1/16 (got %.15g)", phi);

    phi = qpe_bitstring_to_phase(8, 4);
    CHECK(fabs(phi - 0.5) < 1e-15,
          "bitstring=8 m=4 -> phi=1/2 (got %.15g)", phi);

    phi = qpe_bitstring_to_phase(0, 8);
    CHECK(fabs(phi) < 1e-15,
          "bitstring=0 -> phi=0 (got %.15g)", phi);

    phi = qpe_bitstring_to_phase(255, 8);
    CHECK(fabs(phi - 255.0/256.0) < 1e-15,
          "bitstring=255 m=8 -> phi=255/256 (got %.15g)", phi);
}

static void test_t_gate_phase(void) {
    fprintf(stdout, "\n-- QPE: T-gate eigenphase recovery --\n");
    /* T|1> = e^(i pi/4) |1>. The phase in the normalised [0,1)
     * convention is phi = (pi/4) / (2 pi) = 1/8 = 0.125.
     * With m = 6 precision bits, the exact phase 1/8 is representable
     * as bitstring 8 / 64 = 0.125, so QPE should recover it exactly. */
    unitary_operator_t* U = qpe_create_t_gate();
    CHECK(U != NULL, "create T-gate unitary");
    if (!U) return;

    /* Build eigenstate |1> on U's num_qubits qubits. */
    eigenstate_t* es = eigenstate_create(U->num_qubits);
    CHECK(es != NULL, "create eigenstate");
    if (!es) { unitary_operator_free(U); return; }
    gate_pauli_x(es->state, 0);
    es->phase = 1.0 / 8.0;

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    const size_t M = 6;
    qpe_result_t r = qpe_estimate_phase(U, es, M, &e);

    fprintf(stdout,
            "    phi_est = %.6f   (true = 0.125)   bitstring = 0x%llx   confidence = %.4f\n",
            r.estimated_phase,
            (unsigned long long)r.phase_bitstring,
            r.confidence);

    CHECK(r.estimated_phase >= 0.0 && r.estimated_phase < 1.0,
          "estimated phase is in [0, 1)");
    CHECK(r.precision_bits == M,
          "result reports %zu precision bits (expected %zu)",
          r.precision_bits, (size_t)M);
    CHECK(r.phase_bitstring == 8ULL,
          "bitstring 8 at m=6 (got 0x%llx)",
          (unsigned long long)r.phase_bitstring);
    CHECK(fabs(r.estimated_phase - 0.125) < 1e-12,
          "phi_est == 1/8 for T-gate (got %.12f)", r.estimated_phase);

    eigenstate_free(es);
    unitary_operator_free(U);
}

int main(void) {
    fprintf(stdout, "=== QPE smoke tests ===\n");
    test_phase_bitstring_roundtrip();
    test_t_gate_phase();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
