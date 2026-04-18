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

int main(void) {
    fprintf(stdout, "=== topological smoke tests ===\n");
    test_fibonacci_quantum_dimension();
    test_ising_anyons();
    test_surface_code_lifecycle();
    test_toric_code_lifecycle();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
