/**
 * @file test_surface_code_clifford.c
 * @brief Unit tests for the Clifford-backed surface code.
 *
 * Validates syndrome response to data-qubit errors at distances well
 * beyond the dense simulator's reach.  At d=7 the code uses 121
 * qubits; d=15 uses 617.  Both are impossible on the 32-qubit dense
 * engine but tractable on the tableau.
 *
 * The state is initialised as |0⟩^⊗N (not projected onto the full
 * logical |0⟩_L), which is already +1 for every Z-stabilizer. X-errors
 * on data qubits must flip exactly the Z-syndromes for the vertices
 * whose stabilizer includes the errored qubit.  This is the classic
 * surface-code syndrome-detection story.
 */

#include "../../src/algorithms/topological/topological.h"

#include <stdint.h>
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

static uint32_t idx(const surface_code_clifford_t* code,
                    uint32_t r, uint32_t c) {
    return surface_code_clifford_data_index(code, r, c);
}

static void test_fresh_state_zero_syndrome(uint32_t d) {
    fprintf(stdout, "\n-- surface_code_clifford d=%u: fresh |0>^N has zero Z-syndrome --\n", d);
    surface_code_clifford_t* c = surface_code_clifford_create(d, 0xDEADBEEFu);
    surface_code_clifford_measure_z_syndromes(c);
    uint32_t total = 0;
    uint32_t ns = c->num_ancilla_qubits;
    for (uint32_t i = 0; i < ns; i++) total += c->z_syndrome[i];
    CHECK(total == 0, "all %u Z-syndromes read 0", ns);
    surface_code_clifford_free(c);
}

static void test_single_x_error_response(void) {
    fprintf(stdout, "\n-- surface_code_clifford d=7: single X error flips exactly adjacent Z-syndromes --\n");
    const uint32_t d = 7;
    surface_code_clifford_t* c = surface_code_clifford_create(d, 0xC0FFEE01u);

    /* Apply X on an interior data qubit at (3, 3).  The data qubit at
     * (r, c) participates in the Z-stabilizers at interior vertices
     * (r-1, c-1), (r-1, c), (r, c-1), (r, c) -- those that have it in
     * their 2x2 plaquette. At (3, 3) all four of those vertices are
     * interior (valid range 0..d-2 == 0..5). */
    surface_code_clifford_apply_error(c, idx(c, 3, 3), 'X');
    surface_code_clifford_measure_z_syndromes(c);

    uint32_t total = 0;
    uint32_t ns = c->num_ancilla_qubits;
    for (uint32_t i = 0; i < ns; i++) total += c->z_syndrome[i];
    CHECK(total == 4, "total Z-syndrome weight = 4 (got %u)", total);

    CHECK(c->z_syndrome[2 * (d - 1) + 2] == 1, "vertex (2,2) flipped");
    CHECK(c->z_syndrome[2 * (d - 1) + 3] == 1, "vertex (2,3) flipped");
    CHECK(c->z_syndrome[3 * (d - 1) + 2] == 1, "vertex (3,2) flipped");
    CHECK(c->z_syndrome[3 * (d - 1) + 3] == 1, "vertex (3,3) flipped");

    surface_code_clifford_free(c);
}

static void test_two_x_errors_annihilate(void) {
    fprintf(stdout, "\n-- surface_code_clifford d=9: adjacent X errors share syndromes --\n");
    const uint32_t d = 9;
    surface_code_clifford_t* c = surface_code_clifford_create(d, 0xC0FFEE02u);

    /* Two adjacent data qubits: (4, 4) and (4, 5). They share the
     * Z-stabilizers at vertices (3, 4) and (4, 4).  Those two should
     * end up with syndrome 0 (double-flipped), and the 4 non-shared
     * vertices should be 1. Net weight 4. */
    surface_code_clifford_apply_error(c, idx(c, 4, 4), 'X');
    surface_code_clifford_apply_error(c, idx(c, 4, 5), 'X');
    surface_code_clifford_measure_z_syndromes(c);

    uint32_t total = 0;
    uint32_t ns = c->num_ancilla_qubits;
    for (uint32_t i = 0; i < ns; i++) total += c->z_syndrome[i];
    CHECK(total == 4, "total Z-syndrome weight = 4 (got %u)", total);

    CHECK(c->z_syndrome[3 * (d - 1) + 4] == 0, "shared vertex (3,4) stays 0");
    CHECK(c->z_syndrome[4 * (d - 1) + 4] == 0, "shared vertex (4,4) stays 0");
    CHECK(c->z_syndrome[3 * (d - 1) + 3] == 1, "non-shared vertex (3,3) = 1");
    CHECK(c->z_syndrome[3 * (d - 1) + 5] == 1, "non-shared vertex (3,5) = 1");
    CHECK(c->z_syndrome[4 * (d - 1) + 3] == 1, "non-shared vertex (4,3) = 1");
    CHECK(c->z_syndrome[4 * (d - 1) + 5] == 1, "non-shared vertex (4,5) = 1");

    surface_code_clifford_free(c);
}

static void test_distance_scaling(void) {
    fprintf(stdout, "\n-- surface_code_clifford d=15: scaling past dense ceiling --\n");
    const uint32_t d = 15;
    /* 225 data + 392 ancilla = 617 qubits -- impossible dense. */
    surface_code_clifford_t* c = surface_code_clifford_create(d, 0xBADDCAFEu);
    CHECK(c != NULL, "create at d=15 succeeds (617-qubit tableau)");
    surface_code_clifford_apply_error(c, idx(c, 7, 7), 'X');
    surface_code_clifford_measure_z_syndromes(c);

    uint32_t total = 0;
    uint32_t ns = c->num_ancilla_qubits;
    for (uint32_t i = 0; i < ns; i++) total += c->z_syndrome[i];
    CHECK(total == 4, "single interior X error flips 4 Z-syndromes at d=15");
    surface_code_clifford_free(c);
}

int main(void) {
    fprintf(stdout, "=== Clifford-backed surface code ===\n");
    test_fresh_state_zero_syndrome(3);
    test_fresh_state_zero_syndrome(7);
    test_single_x_error_response();
    test_two_x_errors_annihilate();
    test_distance_scaling();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
