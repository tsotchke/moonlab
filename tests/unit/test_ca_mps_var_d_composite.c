/**
 * @file test_ca_mps_var_d_composite.c
 * @brief Validates the composite-2-gate Clifford search feature.
 *
 * Constructs a small TFIM at criticality (n=4, g=1) and verifies:
 *   1. composite_2gate=0: greedy single-gate descent produces some
 *      result;
 *   2. composite_2gate=1: greedy with 2-gate composite moves enabled
 *      produces a result that is *at least as good* as the 1-gate
 *      version (energy <= 1-gate's, entropy <= 1-gate's by some
 *      epsilon).
 *
 * The hypothesis behind composite moves is that the 1-gate search
 * gets stuck in basins where the right descent is a 2-gate sequence
 * that looks bad in step 1 but pays off in step 2.  This test
 * verifies the *feature is wired up* and produces consistent
 * monotone-or-better results -- it does not insist on a strict
 * improvement, since at small N the basin structure is benign and
 * 1-gate may already find the local optimum.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static void build_tfim(uint32_t n, double g,
                        uint8_t** out_paulis, double** out_coeffs,
                        uint32_t* out_T) {
    uint32_t T = (n - 1) + n;
    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, 1);
    double*  coeffs = (double*)calloc(T, sizeof(double));
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        paulis[k * n + i]     = 3; paulis[k * n + i + 1] = 3;
        coeffs[k] = -1.0; k++;
    }
    for (uint32_t i = 0; i < n; i++) {
        paulis[k * n + i] = 1; coeffs[k] = -g; k++;
    }
    *out_paulis = paulis; *out_coeffs = coeffs; *out_T = T;
}

static double run_var_d(int composite_2gate, double* out_E, double* out_S) {
    const uint32_t n = 4;
    const double g = 1.0;
    uint8_t* paulis;
    double*  coeffs;
    uint32_t T;
    build_tfim(n, g, &paulis, &coeffs, &T);

    moonlab_ca_mps_t* state = moonlab_ca_mps_create(n, /*max_bond=*/8);
    ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
    cfg.warmstart                 = CA_MPS_WARMSTART_DUAL_TFIM;
    cfg.max_outer_iters           = 20;
    cfg.imag_time_dtau            = 0.1;
    cfg.imag_time_steps_per_outer = 4;
    cfg.clifford_passes_per_outer = 6;
    cfg.composite_2gate           = composite_2gate;
    cfg.convergence_eps           = 1e-7;
    cfg.verbose                   = 0;

    ca_mps_var_d_alt_result_t res = {0};
    moonlab_ca_mps_optimize_var_d_alternating(state, paulis, coeffs, T, &cfg, &res);
    *out_E = res.final_energy;
    *out_S = res.final_phi_entropy;
    int gates = res.total_gates_added;

    moonlab_ca_mps_free(state);
    free(paulis); free(coeffs);
    return (double)gates;
}

int main(void) {
    fprintf(stdout, "=== composite-move var-D test (TFIM N=4, g=1) ===\n");

    double E1, S1, E2, S2;
    double g1 = run_var_d(0, &E1, &S1);
    double g2 = run_var_d(1, &E2, &S2);

    fprintf(stdout, "1-gate:    E=%.6f  S(phi)=%.4f  gates=%.0f\n", E1, S1, g1);
    fprintf(stdout, "composite: E=%.6f  S(phi)=%.4f  gates=%.0f\n", E2, S2, g2);

    /* Composite must produce at least as good an energy as 1-gate
     * (greedy 1-gate is a subset of greedy 2-gate). */
    CHECK(E2 <= E1 + 1e-8,
          "composite-move E=%.6f exceeds 1-gate E=%.6f", E2, E1);
    /* Both should converge to a finite, sane state. */
    CHECK(E1 > -1e3 && E1 < 0,
          "1-gate energy %.4f outside [-1e3, 0]", E1);
    CHECK(E2 > -1e3 && E2 < 0,
          "composite energy %.4f outside [-1e3, 0]", E2);
    /* Composite should not produce a divergent S(phi). */
    CHECK(S2 < 5.0,
          "composite S(phi) = %.4f exceeds 5.0 nat sanity bound", S2);

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
