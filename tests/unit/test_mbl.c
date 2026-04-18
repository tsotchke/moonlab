/**
 * @file test_mbl.c
 * @brief MBL (many-body localization) smoke tests.
 *
 * Exercises the XXZ Hamiltonian builder and the sparse-matrix path.
 */

#include "../../src/algorithms/mbl/mbl.h"
#include <stdbool.h>
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

static void test_xxz_hamiltonian_lifecycle(void) {
    fprintf(stdout, "\n-- MBL: xxz_hamiltonian build/free --\n");
    /* 4-site XXZ chain: J=1, Delta=1 (isotropic Heisenberg),
     * disorder strength W=3, open boundary conditions, seed=42. */
    xxz_hamiltonian_t* h = xxz_hamiltonian_create(4, 1.0, 1.0, 3.0, false, 42);
    CHECK(h != NULL, "create 4-site XXZ Hamiltonian");
    if (!h) return;

    sparse_hamiltonian_t* sp = xxz_build_sparse(h);
    CHECK(sp != NULL, "build sparse form");
    if (sp) sparse_hamiltonian_free(sp);
    xxz_hamiltonian_free(h);
    fprintf(stdout, "  OK    freed XXZ and sparse Hamiltonians\n");
}

int main(void) {
    fprintf(stdout, "=== MBL smoke tests ===\n");
    test_xxz_hamiltonian_lifecycle();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
