/**
 * @file ca_mps_imag_time.c
 * @brief Imaginary-time evolution of a Heisenberg dimer (N=2) under
 *        CA-MPS, demonstrating convergence to the singlet ground state.
 *
 * The Heisenberg Hamiltonian H = X.X + Y.Y + Z.Z on two qubits has
 * ground-state energy -3 (singlet) and first-excited energy +1.  Apply
 * imaginary-time evolution exp(-tau H) for many small steps and watch
 * <H> converge to -3.
 *
 * The Trotterised step is exp(-dtau X.X) . exp(-dtau Y.Y) . exp(-dtau Z.Z),
 * each of which is a Pauli-string rotation.  Under CA-MPS,
 * imag_pauli_rotation handles the non-unitary part directly.
 *
 * Run:
 *   ./build/example_ca_mps_imag_time [steps=400] [dtau=0.05]
 *
 * Expect convergence to within 1e-8 of -3 by the default 400 steps.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    uint32_t steps = (argc > 1) ? (uint32_t)atoi(argv[1]) : 400;
    double   dtau  = (argc > 2) ? atof(argv[2])         : 0.05;

    /* N=2 dimer is small enough that no truncation is ever necessary;
     * chi=4 is overkill but safe. */
    moonlab_ca_mps_t* s = moonlab_ca_mps_create(2, 4);
    if (!s) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Initial state |10> has 1/sqrt(2) overlap with the singlet (the
     * unique ground state of XX+YY+ZZ).  |00> would be a triplet
     * eigenstate with energy +1 and exp(-tau H) couldn't move it to
     * the ground sector. */
    moonlab_ca_mps_x(s, 0);

    /* Heisenberg H = XX + YY + ZZ.  Three Pauli strings, each of
     * length 2 (one per qubit). */
    uint8_t xx[2] = {1, 1};
    uint8_t yy[2] = {2, 2};
    uint8_t zz[2] = {3, 3};
    /* Observable: same H, expressed as a Pauli sum. */
    uint8_t paulis[3 * 2] = {1,1, 2,2, 3,3};
    double _Complex coeffs[3] = {1.0, 1.0, 1.0};

    /* Track <H> every 25 steps so the user sees the convergence trajectory. */
    printf("Imaginary-time evolution of the Heisenberg dimer (N=2)\n");
    printf("  Trotter step = %.3f, total steps = %u, target E0 = -3.0\n\n",
           dtau, steps);
    printf("  step    tau       <H>        |<H> - E0|\n");
    printf("  ----  ------  ----------  ------------\n");

    double _Complex e;
    moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, 3, &e);
    printf("  %4u  %6.3f  %+10.6f  %.3e\n",
           0u, 0.0, creal(e), fabs(creal(e) - (-3.0)));

    for (uint32_t k = 1; k <= steps; k++) {
        moonlab_ca_mps_imag_pauli_rotation(s, xx, dtau);
        moonlab_ca_mps_imag_pauli_rotation(s, yy, dtau);
        moonlab_ca_mps_imag_pauli_rotation(s, zz, dtau);
        moonlab_ca_mps_normalize(s);
        if (k % 25 == 0 || k == steps) {
            moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, 3, &e);
            printf("  %4u  %6.3f  %+10.6f  %.3e\n",
                   k, k * dtau, creal(e), fabs(creal(e) - (-3.0)));
        }
    }

    moonlab_ca_mps_free(s);
    return 0;
}
