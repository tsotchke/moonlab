/**
 * @file test_ca_mps_linux_probe.c
 * @brief Instrumented probe for the Linux OpenBLAS divergence in CA-MPS
 *        imaginary-time evolution.  Not a production test -- the sole
 *        purpose is to print the intermediate state at every step so
 *        we can diff Linux vs macOS outputs.
 *
 * Runs the dimer case (N=2, 1 bond) for a single short Trotter schedule
 * (tau=0.1, 3 steps).  On each step, prints:
 *   - raw MPS bond dimensions
 *   - <Z_0 Z_1>, <X_0 X_1>, <Y_0 Y_1>, <Z_0>, <X_0>, <Y_0>
 *   - state norm (raw MPS norm BEFORE renormalize)
 *   - final energy <psi|H|psi>
 *
 * Push this, read the CI output, compare platforms.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    fprintf(stdout, "=== CA-MPS Linux divergence probe ===\n");

    uint8_t zz[2] = {3, 3};
    uint8_t xx[2] = {1, 1};
    uint8_t yy[2] = {2, 2};
    uint8_t z0[2] = {3, 0};
    uint8_t x0[2] = {1, 0};
    uint8_t y0[2] = {2, 0};

    moonlab_ca_mps_t* s = moonlab_ca_mps_create(2, 16);

    /* Neel + Hadamard kick -- same initial state as test_ca_mps_imag_time. */
    moonlab_ca_mps_x(s, 1);
    moonlab_ca_mps_h(s, 0);

    fprintf(stdout, "\n--- After Clifford prep ---\n");
    double norm = moonlab_ca_mps_norm(s);
    fprintf(stdout, "  norm^2 = %.9f\n", norm);
    double _Complex ezz, exx, eyy, ez0, ex0, ey0;
    moonlab_ca_mps_expect_pauli(s, zz, &ezz);
    moonlab_ca_mps_expect_pauli(s, xx, &exx);
    moonlab_ca_mps_expect_pauli(s, yy, &eyy);
    moonlab_ca_mps_expect_pauli(s, z0, &ez0);
    moonlab_ca_mps_expect_pauli(s, x0, &ex0);
    moonlab_ca_mps_expect_pauli(s, y0, &ey0);
    fprintf(stdout, "  <ZZ> = %+.9f %+.9fi\n", creal(ezz), cimag(ezz));
    fprintf(stdout, "  <XX> = %+.9f %+.9fi\n", creal(exx), cimag(exx));
    fprintf(stdout, "  <YY> = %+.9f %+.9fi\n", creal(eyy), cimag(eyy));
    fprintf(stdout, "  <Z_0>= %+.9f %+.9fi\n", creal(ez0), cimag(ez0));
    fprintf(stdout, "  <X_0>= %+.9f %+.9fi\n", creal(ex0), cimag(ex0));
    fprintf(stdout, "  <Y_0>= %+.9f %+.9fi\n", creal(ey0), cimag(ey0));
    fprintf(stdout, "  <H>  = %+.9f\n", creal(exx + eyy + ezz));

    for (int step = 0; step < 3; step++) {
        fprintf(stdout, "\n--- Step %d: apply exp(-0.1 XX) exp(-0.1 YY) exp(-0.1 ZZ) ---\n", step);
        moonlab_ca_mps_imag_pauli_rotation(s, xx, 0.1);
        moonlab_ca_mps_imag_pauli_rotation(s, yy, 0.1);
        moonlab_ca_mps_imag_pauli_rotation(s, zz, 0.1);
        fprintf(stdout, "  pre-normalize norm^2 = %.9f\n", moonlab_ca_mps_norm(s));
        moonlab_ca_mps_normalize(s);
        moonlab_ca_mps_expect_pauli(s, zz, &ezz);
        moonlab_ca_mps_expect_pauli(s, xx, &exx);
        moonlab_ca_mps_expect_pauli(s, yy, &eyy);
        moonlab_ca_mps_expect_pauli(s, z0, &ez0);
        moonlab_ca_mps_expect_pauli(s, x0, &ex0);
        moonlab_ca_mps_expect_pauli(s, y0, &ey0);
        fprintf(stdout, "  post-normalize:\n");
        fprintf(stdout, "    <ZZ> = %+.9f %+.9fi\n", creal(ezz), cimag(ezz));
        fprintf(stdout, "    <XX> = %+.9f %+.9fi\n", creal(exx), cimag(exx));
        fprintf(stdout, "    <YY> = %+.9f %+.9fi\n", creal(eyy), cimag(eyy));
        fprintf(stdout, "    <Z_0>= %+.9f %+.9fi\n", creal(ez0), cimag(ez0));
        fprintf(stdout, "    <X_0>= %+.9f %+.9fi\n", creal(ex0), cimag(ex0));
        fprintf(stdout, "    <Y_0>= %+.9f %+.9fi\n", creal(ey0), cimag(ey0));
        fprintf(stdout, "    <H>  = %+.9f\n", creal(exx + eyy + ezz));
        fprintf(stdout, "    MPS bond = %u\n", moonlab_ca_mps_current_bond_dim(s));
    }

    moonlab_ca_mps_free(s);
    /* Intentionally fail so CI --output-on-failure prints the trace.
     * Removed once the divergence is characterised. */
    return 1;
}
