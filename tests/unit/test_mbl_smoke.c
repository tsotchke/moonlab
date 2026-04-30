/**
 * @file test_mbl_smoke.c
 * @brief Smoke harness for the MBL research-path public APIs.
 *
 * Closes two ICC dead-code triage entries (construct_lioms, 165 LOC,
 * and scan_phase_diagram, 103 LOC) by calling each with synthetic
 * small inputs and checking the returned structures are well-formed.
 *
 * The MBL module ships a level-statistics + LIOM + phase-diagram
 * pipeline for disordered Heisenberg XXZ chains.  Until this test
 * landed, none of those entry points had any in-tree caller, so an
 * accidental regression on them would not have been caught.
 */

#include "../../src/algorithms/mbl/mbl.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static void test_construct_lioms_smoke(void) {
    fprintf(stdout, "\n--- construct_lioms (N=4 disordered XXZ) ---\n");

    xxz_hamiltonian_t* xxz =
        xxz_hamiltonian_create(/*N=*/4, /*J=*/1.0, /*delta=*/1.0,
                                /*W=*/2.0, /*pbc=*/false, /*seed=*/0xC0FFEE);
    CHECK(xxz != NULL, "xxz_hamiltonian_create");
    if (!xxz) return;

    sparse_hamiltonian_t* h = xxz_build_sparse(xxz);
    CHECK(h != NULL, "xxz_build_sparse");
    if (h) {
        /* construct_lioms requires the eigensystem to have been
         * computed (h->eigensystem_computed must be true). */
        qs_error_t e = sparse_hamiltonian_diagonalize(h);
        CHECK(e == QS_SUCCESS, "sparse_hamiltonian_diagonalize -> %d", (int)e);
        if (e == QS_SUCCESS) {
            liom_system_t* sys = construct_lioms(h);
            CHECK(sys != NULL, "construct_lioms returned NULL");
            if (sys) liom_system_free(sys);
        }
        sparse_hamiltonian_free(h);
    }
    xxz_hamiltonian_free(xxz);
}

static void test_scan_phase_diagram_smoke(void) {
    fprintf(stdout, "\n--- scan_phase_diagram (N=4, 2 W points, 1 realisation) ---\n");

    phase_diagram_t* pd = scan_phase_diagram(/*N=*/4,
                                                /*J=*/1.0,
                                                /*delta=*/1.0,
                                                /*W_min=*/0.5,
                                                /*W_max=*/2.5,
                                                /*num_W_points=*/2,
                                                /*num_realizations=*/1,
                                                /*pbc=*/false);
    CHECK(pd != NULL, "scan_phase_diagram returned NULL");
    if (pd) phase_diagram_free(pd);
}

int main(void) {
    fprintf(stdout, "=== MBL dead-code smoke harness ===\n");
    test_construct_lioms_smoke();
    test_scan_phase_diagram_smoke();

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
