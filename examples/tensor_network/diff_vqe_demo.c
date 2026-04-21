/**
 * @file diff_vqe_demo.c
 * @brief VQE-style minimisation driven by native Moonlab autograd.
 *
 * Minimises H = <Z_0> + 0.5 <Z_0 Z_1> on a two-qubit RY ansatz using
 * gradient descent.  Gradients come entirely from the adjoint method
 * in src/algorithms/diff/; no Python, no PyTorch, no finite
 * differences -- the whole cost is expressed as a multi-Pauli sum and
 * both value and gradient are evaluated in one call each.
 */

#include "../../src/algorithms/diff/differentiable.h"
#include "../../src/quantum/state.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    printf("=== diff_vqe_demo: grad descent on 2-qubit RY ansatz ===\n");

    moonlab_diff_circuit_t *c = moonlab_diff_circuit_create(2);
    double theta[2] = { 0.2, -0.3 };
    moonlab_diff_ry(c, 0, theta[0]);
    moonlab_diff_ry(c, 1, theta[1]);
    moonlab_diff_cnot(c, 0, 1);

    /* Cost as a Pauli sum: 1.0 * Z_0 + 0.5 * Z_0 Z_1. */
    int q_z0[1]  = {0};
    int q_zz[2]  = {0, 1};
    moonlab_diff_observable_t p_z[1]  = { MOONLAB_DIFF_OBS_Z };
    moonlab_diff_observable_t p_zz[2] = { MOONLAB_DIFF_OBS_Z,
                                           MOONLAB_DIFF_OBS_Z };
    moonlab_diff_pauli_term_t terms[2] = {
        { 1.0, 1, q_z0, p_z  },
        { 0.5, 2, q_zz, p_zz },
    };

    quantum_state_t s;
    quantum_state_init(&s, 2);

    const double lr       = 0.3;
    const size_t max_iter = 80;
    const double tol      = 1e-10;
    double prev_cost = 1e9;

    printf("%4s  %12s  %12s  %12s  %12s\n",
           "iter", "theta0", "theta1", "<H>", "|grad|");
    for (size_t iter = 0; iter < max_iter; iter++) {
        moonlab_diff_set_theta(c, 0, theta[0]);
        moonlab_diff_set_theta(c, 1, theta[1]);
        moonlab_diff_forward(c, &s);

        const double cost = moonlab_diff_expect_pauli_sum(&s, terms, 2);

        double g[2] = {0};
        int rc = moonlab_diff_backward_pauli_sum(c, &s, terms, 2, g);
        if (rc != 0) {
            fprintf(stderr, "backward_pauli_sum failed (rc=%d)\n", rc);
            quantum_state_free(&s);
            moonlab_diff_circuit_free(c);
            return 1;
        }

        theta[0] -= lr * g[0];
        theta[1] -= lr * g[1];

        const double gmag = sqrt(g[0] * g[0] + g[1] * g[1]);
        if (iter % 5 == 0 || iter == max_iter - 1) {
            printf("%4zu  %+.6f  %+.6f  %+.6f  %.2e\n",
                   iter, theta[0], theta[1], cost, gmag);
        }
        if (fabs(prev_cost - cost) < tol) {
            printf("  converged after %zu iterations  final <H>=%+.8f\n",
                   iter + 1, cost);
            break;
        }
        prev_cost = cost;
    }

    /* Analytic minimum: at theta_0 = pi the ansatz becomes
     * CNOT(0,1)(|1> |psi(theta_1)>) = |1>|~psi>, giving <Z_0> = -1 and
     * <Z_0 Z_1> = -<Z_1_after_CNOT> averaged to -1, so <H> = -1.5. */
    moonlab_diff_set_theta(c, 0, theta[0]);
    moonlab_diff_set_theta(c, 1, theta[1]);
    moonlab_diff_forward(c, &s);
    const double final_cost = moonlab_diff_expect_pauli_sum(&s, terms, 2);
    printf("Final <H> = %+.6f  (analytic minimum -1.5)\n", final_cost);

    quantum_state_free(&s);
    moonlab_diff_circuit_free(c);
    return 0;
}
