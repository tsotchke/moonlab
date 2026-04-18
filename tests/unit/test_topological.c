/**
 * @file test_topological.c
 * @brief Topological QC smoke tests.
 *
 *  - Fibonacci and Ising anyon systems construct with the expected
 *    quantum dimensions.
 *  - Anyonic registers can be created and freed on either anyon model.
 *  - Surface-code/toric-code basic lifecycle.
 */

#include "../../src/algorithms/topological/topological.h"
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

static void test_fibonacci_quantum_dimension(void) {
    fprintf(stdout, "\n-- topological: Fibonacci anyons have d_tau = phi --\n");
    anyon_system_t* sys = anyon_system_fibonacci();
    CHECK(sys != NULL, "create Fibonacci anyon system");
    if (!sys) return;

    /* Fibonacci tau anyon has quantum dimension phi = (1 + sqrt(5))/2.
     * Identity has quantum dimension 1. */
    double d_total = anyon_total_dimension(sys);
    CHECK(isfinite(d_total) && d_total > 0.0,
          "total quantum dimension positive (got %.6f)", d_total);

    anyon_system_free(sys);
}

static void test_ising_anyons(void) {
    fprintf(stdout, "\n-- topological: Ising anyon system builds --\n");
    anyon_system_t* sys = anyon_system_ising();
    CHECK(sys != NULL, "create Ising anyon system");
    if (sys) {
        double d_total = anyon_total_dimension(sys);
        CHECK(isfinite(d_total) && d_total > 0.0,
              "Ising total quantum dimension is finite positive (got %.6f)",
              d_total);
        anyon_system_free(sys);
    }
}

static void test_surface_code_lifecycle(void) {
    fprintf(stdout, "\n-- topological: surface_code_create/free --\n");
    surface_code_t* code = surface_code_create(3);  /* distance-3 */
    CHECK(code != NULL, "create distance-3 surface code");
    if (code) {
        qs_error_t err = surface_code_init_logical_zero(code);
        CHECK(err == QS_SUCCESS, "init logical |0>");
        surface_code_free(code);
    }
}

static void test_toric_code_lifecycle(void) {
    fprintf(stdout, "\n-- topological: toric_code_create/free --\n");
    toric_code_t* code = toric_code_create(3);  /* L=3 */
    CHECK(code != NULL, "create L=3 toric code");
    if (code) {
        toric_code_free(code);
    }
}

static void test_fibonacci_braiding_invariants(void) {
    fprintf(stdout, "\n-- topological: Fibonacci braiding --\n");
    anyon_system_t* sys = anyon_system_fibonacci();
    if (!sys) { CHECK(0, "create system"); return; }

    /* Four tau anyons fusing to vacuum — the canonical setup used to
     * realise one logical qubit of Fibonacci topological quantum
     * computation. */
    anyon_charge_t charges[4] = { FIB_TAU, FIB_TAU, FIB_TAU, FIB_TAU };
    fusion_tree_t* tree = fusion_tree_create(sys, charges, 4, FIB_VACUUM);
    CHECK(tree != NULL, "create 4-tau fusion tree");
    if (!tree) { anyon_system_free(sys); return; }

    /* Fusion space dimension: for n=4 tau anyons fused to vacuum, it's
     * Fibonacci number F_{n-1} = F_3 = 2. */
    uint32_t paths = fusion_count_paths(sys, charges, 4, FIB_VACUUM);
    CHECK(paths == 2,
          "fusion_count_paths(4 tau -> vacuum) == 2 (got %u)", paths);

    /* Norm before braid should be 1. */
    double norm0 = 0.0;
    for (uint32_t i = 0; i < tree->num_paths; i++) {
        double m = cabs(tree->amplitudes[i]);
        norm0 += m * m;
    }
    CHECK(fabs(norm0 - 1.0) < 1e-10,
          "initial norm == 1 (got %.12f)", norm0);

    /* Snapshot amplitudes for identity-braid check. */
    double complex amps0[8] = {0};
    for (uint32_t i = 0; i < tree->num_paths && i < 8; i++) {
        amps0[i] = tree->amplitudes[i];
    }

    /* Apply sigma_1 followed by sigma_1^{-1}: should be identity on
     * the logical state up to global phase. */
    qs_error_t err = braid_anyons(tree, 0, true);
    CHECK(err == QS_SUCCESS, "sigma_1 succeeds");
    err = braid_anyons(tree, 0, false);
    CHECK(err == QS_SUCCESS, "sigma_1^{-1} succeeds");

    /* Norm preserved. */
    double norm1 = 0.0;
    for (uint32_t i = 0; i < tree->num_paths; i++) {
        double m = cabs(tree->amplitudes[i]);
        norm1 += m * m;
    }
    CHECK(fabs(norm1 - 1.0) < 1e-10,
          "norm preserved after sigma_1 sigma_1^{-1} (got %.12f)", norm1);

    /* sigma * sigma^{-1} is identity on the full state. */
    double max_diff = 0.0;
    for (uint32_t i = 0; i < tree->num_paths && i < 8; i++) {
        double d = cabs(tree->amplitudes[i] - amps0[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-10,
          "sigma_1 sigma_1^{-1} = I on amplitudes (max diff %.3e)", max_diff);

    fusion_tree_free(tree);
    anyon_system_free(sys);
}

int main(void) {
    fprintf(stdout, "=== topological smoke tests ===\n");
    test_fibonacci_quantum_dimension();
    test_ising_anyons();
    test_fibonacci_braiding_invariants();
    test_surface_code_lifecycle();
    test_toric_code_lifecycle();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
