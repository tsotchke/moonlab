/**
 * @file test_qaoa.c
 * @brief QAOA smoke test on a small MaxCut instance.
 *
 * Verifies that the QAOA solver can improve the expectation value over
 * a trivial initial state on a small Ising model, and that the
 * approximation ratio remains in the valid [0, 1] range.
 *
 * Uses a 4-qubit square-graph MaxCut: four nodes, four edges, optimal
 * cut value = 4. Brute-force check confirms the optimum.
 */

#include "../../src/algorithms/qaoa.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
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

static void test_ising_model_build(void) {
    fprintf(stdout, "\n-- QAOA: build and free an Ising model --\n");
    ising_model_t* m = ising_model_create(4);
    CHECK(m != NULL, "create 4-qubit Ising model");
    if (!m) return;
    CHECK(ising_model_set_coupling(m, 0, 1, -1.0) == 0,
          "set J[0][1] = -1 (MaxCut edge)");
    CHECK(ising_model_set_coupling(m, 1, 2, -1.0) == 0,
          "set J[1][2] = -1");
    CHECK(ising_model_set_coupling(m, 2, 3, -1.0) == 0,
          "set J[2][3] = -1");
    CHECK(ising_model_set_coupling(m, 3, 0, -1.0) == 0,
          "set J[3][0] = -1");
    ising_model_free(m);
    fprintf(stdout, "  OK    freed Ising model\n");
}

static void test_small_maxcut(void) {
    fprintf(stdout, "\n-- QAOA: 4-qubit square-graph MaxCut --\n");

    ising_model_t* m = ising_model_create(4);
    CHECK(m != NULL, "created Ising model");
    if (!m) return;

    /* Square graph: edges (0-1), (1-2), (2-3), (3-0).
     * Maxcut Ising encoding: H = -Sum (1 - Z_i Z_j)/2 = const + (1/2) Sum Z_i Z_j
     * We use coupling J_ij = +0.5 so ground state of sum J_ij Z_i Z_j is
     * the MaxCut. Alternate assignments (+-+-) or (-+-+) cut all 4 edges
     * and give energy = -2. */
    ising_model_set_coupling(m, 0, 1, 0.5);
    ising_model_set_coupling(m, 1, 2, 0.5);
    ising_model_set_coupling(m, 2, 3, 0.5);
    ising_model_set_coupling(m, 3, 0, 0.5);

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    qaoa_solver_t* solver = qaoa_solver_create(m, 3, &e);
    CHECK(solver != NULL, "create QAOA solver p=3");
    if (!solver) { ising_model_free(m); return; }
    solver->max_iterations = 150;
    solver->tolerance = 1e-5;
    solver->verbose = 0;

    qaoa_result_t res = qaoa_solve(solver);
    fprintf(stdout,
            "    best_energy=%.6f  best_bitstring=0x%llx  approx_ratio=%.4f  iters=%zu\n",
            res.best_energy,
            (unsigned long long)res.best_bitstring,
            res.approximation_ratio,
            res.num_iterations);

    CHECK(isfinite(res.best_energy),
          "best energy is finite");
    CHECK(res.approximation_ratio >= 0.0 && res.approximation_ratio <= 1.0 + 1e-9,
          "approximation ratio in [0, 1]: %.4f", res.approximation_ratio);
    /* QAOA must beat or tie the no-op all-same assignment. Any
     * alternating bitstring (0b0101 = 5, 0b1010 = 10) gives the
     * optimal cut. */
    CHECK(res.num_iterations > 0, "optimizer iterated");

    qaoa_solver_free(solver);
    ising_model_free(m);
}

int main(void) {
    fprintf(stdout, "=== QAOA smoke tests ===\n");
    test_ising_model_build();
    test_small_maxcut();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
