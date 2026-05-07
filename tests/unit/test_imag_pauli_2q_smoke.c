/**
 * @file test_imag_pauli_2q_smoke.c
 * @brief 2-qubit single-shot smoke for moonlab_ca_mps_imag_pauli_rotation.
 *
 * Starts in |00>, applies exp(-tau XX) once at tau=0.5. Asserts |11>
 * acquires a non-negligible amplitude. This is the minimum test the
 * weight-2-adjacent fast path must satisfy: if it returns the identity
 * here, this test fails immediately.
 *
 * Mathematically:
 *   exp(-tau X⊗X) = cosh(tau) * I⊗I - sinh(tau) * X⊗X
 * On |00>:
 *   cosh(tau) * |00> - sinh(tau) * |11>
 * After normalisation: |alpha|^2 + |beta|^2 = cosh² + sinh² = cosh(2 tau).
 * Normalised amplitudes:
 *   alpha = cosh(tau) / sqrt(cosh(2 tau)),
 *   beta  = -sinh(tau) / sqrt(cosh(2 tau)).
 * <Z_0> = |alpha|^2 - |beta|^2 = (cosh² - sinh²) / cosh(2 tau) = 1 / cosh(2 tau).
 * For tau = 0.5:  cosh(1) ≈ 1.5430806,  <Z_0> ≈ 0.6480543.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; \
    } \
} while (0)

int main(void) {
    fprintf(stdout, "=== imag_pauli_rotation 2q smoke (XX on |00>) ===\n");

    moonlab_ca_mps_t* s = moonlab_ca_mps_create(/*n=*/2, /*chi=*/16);
    CHECK(s != NULL, "create");
    if (!s) return 1;

    const double tau = 0.5;
    uint8_t pauli[2] = { 1, 1 };  /* X on q0, X on q1 */
    ca_mps_error_t e = moonlab_ca_mps_imag_pauli_rotation(s, pauli, tau);
    CHECK(e == CA_MPS_SUCCESS, "imag_pauli_rotation returned %d", (int)e);

    e = moonlab_ca_mps_normalize(s);
    CHECK(e == CA_MPS_SUCCESS, "normalize returned %d", (int)e);

    /* <Z_0> on the resulting normalised state should be 1 / cosh(2 tau). */
    uint8_t z0[2] = { 3, 0 };  /* Z on q0, I on q1 */
    double _Complex z_expect;
    e = moonlab_ca_mps_expect_pauli(s, z0, &z_expect);
    CHECK(e == CA_MPS_SUCCESS, "expect_pauli returned %d", (int)e);
    double got_z = creal(z_expect);
    double want_z = 1.0 / cosh(2.0 * tau);

    fprintf(stdout, "  tau         = %.6f\n", tau);
    fprintf(stdout, "  <Z_0> got   = %.10f\n", got_z);
    fprintf(stdout, "  <Z_0> want  = %.10f (= 1/cosh(2 tau))\n", want_z);
    fprintf(stdout, "  abs delta   = %.2e\n", fabs(got_z - want_z));

    CHECK(fabs(got_z - want_z) < 1e-6,
          "<Z_0> drift > 1e-6: got %.10f want %.10f",
          got_z, want_z);
    CHECK(got_z < 0.99,
          "<Z_0> = %.10f means state is still ~|00>; the gate did nothing.",
          got_z);

    moonlab_ca_mps_free(s);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
