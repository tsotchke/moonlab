/**
 * @file test_ca_mps_var_d_large.c
 * @brief Regression for the var-D delta cache above 255 qubits.
 *
 * The per-pass term cache stored support qubit indices and lengths as uint8_t
 * while ca_mps accepts up to 100000 qubits, so any qubit index above 255 was
 * silently truncated on write.  This test runs the greedy Clifford var-D
 * optimizer (which builds and consumes the delta cache) on a 300-qubit chain
 * and checks that the energy tracked through the cache's incremental deltas
 * agrees with an independent full Pauli-sum re-evaluation of the final state.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

int main(void) {
    fprintf(stdout, "=== var-D delta cache: >255-qubit consistency ===\n");

    const uint32_t n = 300;          /* > 255: exercises the widened cache */
    const uint32_t max_bond = 4;

    moonlab_ca_mps_t *s = moonlab_ca_mps_create(n, max_bond);
    CHECK(s != NULL, "moonlab_ca_mps_create(%u)", n);
    if (!s) return 1;

    /* Cheap Hamiltonian: single-site Z on every qubit, H = sum_i Z_i.  Each
     * term has support on exactly one qubit, and the support spans indices
     * 0..299 -- covering qubit indices well above 255. */
    const uint32_t num_terms = n;
    uint8_t *paulis = calloc((size_t)num_terms * n, sizeof(uint8_t));
    double  *coeffs = calloc(num_terms, sizeof(double));
    CHECK(paulis && coeffs, "alloc Hamiltonian");
    if (!paulis || !coeffs) { moonlab_ca_mps_free(s); free(paulis); free(coeffs); return 1; }

    for (uint32_t k = 0; k < num_terms; k++) {
        paulis[(size_t)k * n + k] = 3;   /* Z on qubit k */
        coeffs[k] = 1.0;
    }

    ca_mps_var_d_config_t cfg = ca_mps_var_d_config_default();
    cfg.max_passes = 3;
    cfg.include_2q_gates = 0;
    cfg.composite_2gate = 0;
    cfg.verbose = 0;

    ca_mps_var_d_result_t res = {0};
    ca_mps_error_t err = moonlab_ca_mps_optimize_var_d_clifford_only(
        s, paulis, coeffs, num_terms, &cfg, &res);
    CHECK(err == CA_MPS_SUCCESS, "optimize_var_d rc=%d", err);

    /* Independent full re-evaluation of <psi|H|psi> on the final state. */
    double complex *cz = calloc(num_terms, sizeof(double complex));
    for (uint32_t k = 0; k < num_terms; k++) cz[k] = coeffs[k];
    double complex e_full = 0.0;
    ca_mps_error_t ferr = moonlab_ca_mps_expect_pauli_sum(s, paulis, cz, num_terms, &e_full);
    CHECK(ferr == CA_MPS_SUCCESS, "expect_pauli_sum rc=%d", ferr);

    double e_cache = res.final_energy;
    double e_ref = creal(e_full);
    double diff = fabs(e_cache - e_ref);
    fprintf(stdout, "  gates=%d passes=%d converged=%d\n",
            res.gates_added, res.passes, res.converged);
    fprintf(stdout, "  E(delta-cache) = %+.12f\n", e_cache);
    fprintf(stdout, "  E(full re-eval)= %+.12f\n", e_ref);
    fprintf(stdout, "  |diff| = %.3e\n", diff);

    CHECK(diff < 1e-9,
          "cached energy diverges from full re-evaluation by %.3e at n=%u",
          diff, n);
    /* The greedy search should have found improving moves (H on each qubit
     * takes <Z> from +1 to 0), so the energy must not have increased. */
    CHECK(res.final_energy <= res.initial_energy + 1e-9,
          "final energy %.6f exceeds initial %.6f",
          res.final_energy, res.initial_energy);

    free(cz);
    free(paulis);
    free(coeffs);
    moonlab_ca_mps_free(s);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
