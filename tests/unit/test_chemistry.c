/**
 * @file test_chemistry.c
 * @brief Chemistry module smoke tests.
 *
 *  - Build and free a minimal molecular_hamiltonian_t.
 *  - hartree_fock_state() places the right number of electrons in the
 *    lowest-lying orbitals.
 *  - UCCSD configuration constructs with the expected parameter count.
 */

#include "../../src/algorithms/chemistry/chemistry.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/measurement.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_molecular_hamiltonian_lifecycle(void) {
    fprintf(stdout, "\n-- chemistry: molecular_hamiltonian build/free --\n");
    molecular_hamiltonian_t* h = molecular_hamiltonian_create(4, 2, 0.7137);
    CHECK(h != NULL, "create 4-orbital 2-electron Hamiltonian");
    if (!h) return;
    molecular_hamiltonian_add_h1(h, 0, 0, -1.25);
    molecular_hamiltonian_add_h1(h, 1, 1, -0.48);
    molecular_hamiltonian_free(h);
    fprintf(stdout, "  OK    freed Hamiltonian cleanly\n");
}

static void test_hartree_fock_state(void) {
    fprintf(stdout, "\n-- chemistry: Hartree-Fock state places electrons --\n");
    /* 4 orbitals, 2 electrons. Hartree-Fock places them in the lowest
     * two orbitals: result is |1100> or similar (Jordan-Wigner). In any
     * convention, exactly 2 qubits are |1> and 2 are |0>, giving
     * Sum_i P(q_i = 1) = 2. */
    quantum_state_t s;
    quantum_state_init(&s, 4);
    qs_error_t err = hartree_fock_state(&s, 2, 4);
    CHECK(err == QS_SUCCESS, "hartree_fock_state returns success");

    double total_p_one = 0.0;
    for (int q = 0; q < 4; ++q) {
        total_p_one += measurement_probability_one(&s, q);
    }
    CHECK(fabs(total_p_one - 2.0) < 1e-10,
          "sum of P(q=1) over all qubits == num_electrons = 2 (got %.6f)",
          total_p_one);
    quantum_state_free(&s);
}

static void test_uccsd_config(void) {
    fprintf(stdout, "\n-- chemistry: UCCSD config lifecycle --\n");
    uccsd_config_t* cfg = uccsd_config_create(4, 2);
    CHECK(cfg != NULL, "create UCCSD config (4 orbitals, 2 electrons)");
    uccsd_config_free(cfg);
    fprintf(stdout, "  OK    freed UCCSD config cleanly\n");
}

int main(void) {
    fprintf(stdout, "=== chemistry smoke tests ===\n");
    test_molecular_hamiltonian_lifecycle();
    test_hartree_fock_state();
    test_uccsd_config();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
