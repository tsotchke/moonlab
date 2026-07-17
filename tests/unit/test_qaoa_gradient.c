/**
 * @file test_qaoa_gradient.c
 * @brief Exactness check for the QAOA analytic gradient.
 *
 * qaoa_compute_gradient claims to be an exact gradient of <H_C>(gamma, beta).
 * A QAOA layer parameter multiplies a weighted sum of commuting Pauli
 * generators, so the old single global +-pi/2 shift was not the gradient.
 * This test compares the analytic gradient against a central-difference of the
 * exact statevector expectation (qaoa_expectation_exact) for every parameter
 * and requires agreement to 1e-6.  The Ising instance carries both couplings
 * (ZZ terms) and local fields (Z terms) so both gradient contributions are
 * exercised, over two QAOA layers.
 */

#include "../../src/algorithms/qaoa.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"

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
    fprintf(stdout, "=== QAOA analytic gradient vs central difference ===\n");

    const size_t n = 4;
    const size_t p = 2;

    ising_model_t *m = ising_model_create(n);
    CHECK(m != NULL, "ising_model_create");
    if (!m) return 1;

    /* Couplings (ZZ terms) on a small graph plus non-trivial local fields
     * (Z terms) so both branches of the gradient are non-zero. */
    ising_model_set_coupling(m, 0, 1, 0.5);
    ising_model_set_coupling(m, 1, 2, -0.8);
    ising_model_set_coupling(m, 2, 3, 0.3);
    ising_model_set_coupling(m, 0, 3, 0.7);
    ising_model_set_coupling(m, 0, 2, -0.4);
    ising_model_set_field(m, 0, 0.6);
    ising_model_set_field(m, 2, -0.5);

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    qaoa_solver_t *solver = qaoa_solver_create(m, p, &e);
    CHECK(solver != NULL, "qaoa_solver_create");
    if (!solver) { ising_model_free(m); return 1; }
    solver->verbose = 0;   /* keep test output focused on the gradient check */

    /* Fixed, generic parameter point. */
    double gamma[2] = { 0.37, 0.91 };
    double beta[2]  = { 0.55, 0.22 };

    double grad_gamma[2] = {0}, grad_beta[2] = {0};
    int rc = qaoa_compute_gradient(solver, gamma, beta, grad_gamma, grad_beta);
    CHECK(rc == 0, "qaoa_compute_gradient rc=%d", rc);

    const double eps = 1e-4;
    double max_err = 0.0;

    for (size_t L = 0; L < p; L++) {
        /* Central difference w.r.t. gamma[L]. */
        double gp[2] = { gamma[0], gamma[1] };
        gp[L] = gamma[L] + eps;
        double fp = qaoa_expectation_exact(solver, gp, beta);
        gp[L] = gamma[L] - eps;
        double fm = qaoa_expectation_exact(solver, gp, beta);
        double cd_gamma = (fp - fm) / (2.0 * eps);
        double err_g = fabs(cd_gamma - grad_gamma[L]);
        if (err_g > max_err) max_err = err_g;
        fprintf(stdout, "  d/dgamma[%zu]: analytic=%+.9f  central=%+.9f  |err|=%.3e\n",
                L, grad_gamma[L], cd_gamma, err_g);

        /* Central difference w.r.t. beta[L]. */
        double bp[2] = { beta[0], beta[1] };
        bp[L] = beta[L] + eps;
        double fpb = qaoa_expectation_exact(solver, gamma, bp);
        bp[L] = beta[L] - eps;
        double fmb = qaoa_expectation_exact(solver, gamma, bp);
        double cd_beta = (fpb - fmb) / (2.0 * eps);
        double err_b = fabs(cd_beta - grad_beta[L]);
        if (err_b > max_err) max_err = err_b;
        fprintf(stdout, "  d/dbeta[%zu] : analytic=%+.9f  central=%+.9f  |err|=%.3e\n",
                L, grad_beta[L], cd_beta, err_b);
    }

    fprintf(stdout, "  max |analytic - central| = %.3e\n", max_err);
    CHECK(max_err < 1e-6, "gradient mismatch %.3e exceeds 1e-6", max_err);

    /* Exercise the new result_free on a real solve result. */
    qaoa_result_t result = qaoa_solve(solver);
    qaoa_result_free(&result);

    qaoa_solver_free(solver);
    ising_model_free(m);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
