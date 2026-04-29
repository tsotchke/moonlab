/**
 * @file test_ca_mps_var_d.c
 * @brief Validation: greedy local-Clifford search reduces variational
 *        energy on a known-easy TFIM problem.
 *
 * Setup:
 *  - n = 4 qubit chain, TFIM Hamiltonian H = -sum Z_i Z_{i+1} - g sum X_i
 *    with g = 2.5 (paramagnetic regime; ground state ~ |+...+>).
 *  - Initial CA-MPS state: D = I, |phi> = |0...0>.  So |psi> = |0...0>.
 *    <psi|H|psi> at this point is purely the ZZ contribution: -3 (the
 *    three Z_i Z_{i+1} bonds each give +1, with the leading minus,
 *    summing to -3).  The X terms vanish on |0...0>.
 *  - Run the greedy Clifford-only search.
 *
 * Expected: applying H on each qubit takes |0...0> -> |+...+>, which is
 * the exact paramagnetic GS.  <psi|H|psi> after that is exactly -g*n
 * + (corrections from the ZZ terms) ~ -10 for n=4, g=2.5.  The greedy
 * search should reach much closer to this than the initial -3.
 *
 * Pass criteria:
 *  - Search must reduce <psi|H|psi> by at least 5 (any non-trivial
 *    descent is a positive validation; we don't insist on exact
 *    optimum because the greedy local search may find a different
 *    minimum than H_all).
 *  - Search must accept at least one gate.
 *  - Final energy must be lower than initial energy by more than the
 *    convergence epsilon.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; \
    } \
} while (0)

/* Build the TFIM Pauli sum: H = -sum_{i<n-1} Z_i Z_{i+1} - g sum_i X_i.
 *
 * Pauli encoding: 0=I, 1=X, 2=Y, 3=Z.  Layout: paulis[k * n + q].
 * Returns the term count via *out_num_terms; caller frees *out_paulis
 * and *out_coeffs. */
static void build_tfim(uint32_t n, double g,
                        uint8_t** out_paulis, double** out_coeffs,
                        uint32_t* out_num_terms) {
    uint32_t num_zz = (n >= 2) ? (n - 1) : 0;
    uint32_t num_x  = n;
    uint32_t T = num_zz + num_x;

    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, 1);
    double*  coeffs = (double*)calloc(T, sizeof(double));
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        paulis[k * n + i]     = 3;     /* Z */
        paulis[k * n + i + 1] = 3;     /* Z */
        coeffs[k] = -1.0;
        k++;
    }
    for (uint32_t i = 0; i < n; i++) {
        paulis[k * n + i] = 1;         /* X */
        coeffs[k] = -g;
        k++;
    }
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = T;
}

int main(void) {
    fprintf(stdout, "=== CA-MPS variational-D Clifford-only search test ===\n");

    const uint32_t n = 4;
    const double g = 2.5;

    /* Build TFIM Pauli sum. */
    uint8_t*  paulis;
    double*   coeffs;
    uint32_t  T;
    build_tfim(n, g, &paulis, &coeffs, &T);
    fprintf(stdout, "TFIM N=%u, g=%.3f, %u Pauli terms\n", n, g, T);

    /* CA-MPS in default state: D = I, |phi> = |0...0>. */
    moonlab_ca_mps_t* state = moonlab_ca_mps_create(n, /*max_bond=*/8);
    CHECK(state != NULL, "moonlab_ca_mps_create returned NULL");
    if (!state) { free(paulis); free(coeffs); return 1; }

    /* Run greedy variational search. */
    ca_mps_var_d_config_t cfg = ca_mps_var_d_config_default();
    cfg.verbose = 1;
    cfg.improvement_eps = 1e-9;
    ca_mps_var_d_result_t res = {0};

    ca_mps_error_t e = moonlab_ca_mps_optimize_var_d_clifford_only(
        state, paulis, coeffs, T, &cfg, &res);
    CHECK(e == CA_MPS_SUCCESS, "optimize_var_d returned %d", (int)e);

    /* Validation. */
    double dE = res.final_energy - res.initial_energy;
    fprintf(stdout, "\nInitial E = %.6f, Final E = %.6f, dE = %.6f\n",
            res.initial_energy, res.final_energy, dE);
    fprintf(stdout, "Initial S(phi) = %.6f, Final S(phi) = %.6f\n",
            res.initial_phi_entropy, res.final_phi_entropy);
    fprintf(stdout, "gates_added = %d, passes = %d, converged = %d\n",
            res.gates_added, res.passes, res.converged);

    /* Initial energy: |0...0> gives -3 (ZZ terms only contribute, X terms = 0). */
    CHECK(fabs(res.initial_energy + 3.0) < 1e-10,
          "expected initial E = -3, got %.10f", res.initial_energy);

    /* Greedy search must accept at least one gate. */
    CHECK(res.gates_added > 0, "search accepted 0 gates");

    /* Energy must drop by at least 5 (loose threshold; the true GS at
     * g=2.5 has E ~ -10 for n=4). */
    CHECK(dE < -5.0,
          "expected energy drop > 5, got %.6f", dE);

    /* The Clifford-only search at fixed |phi> = |0...0> doesn't change
     * |phi> -- entropy stays at 0. */
    CHECK(fabs(res.final_phi_entropy - res.initial_phi_entropy) < 1e-12,
          "S(phi) changed during Clifford-only search: %.6f -> %.6f",
          res.initial_phi_entropy, res.final_phi_entropy);

    /* Search must terminate cleanly: either by hitting a local minimum
     * (converged = 1) or by hitting max_passes.  Both are acceptable;
     * we just want to confirm the loop doesn't run away. */
    CHECK(res.passes <= cfg.max_passes,
          "passes = %d exceeds max_passes = %d",
          res.passes, cfg.max_passes);

    moonlab_ca_mps_free(state);
    free(paulis);
    free(coeffs);

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
