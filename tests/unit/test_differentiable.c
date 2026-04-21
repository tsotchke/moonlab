/**
 * @file test_differentiable.c
 * @brief Validate reverse-mode adjoint gradients against
 *        finite-difference references.
 *
 * For every parametric gate in a circuit, the adjoint-method
 * gradient must match a central-difference estimate
 *   df/dtheta ~ (f(theta + h) - f(theta - h)) / (2h)
 * to ~1e-6 with h = 1e-4.
 *
 * Three test circuits:
 *  1. Single-qubit RY(theta) with <Z>.  Analytic: d<Z>/dtheta = sin(theta).
 *  2. HEA-like two-qubit circuit (RY, RZ, RY, CNOT, RY, RZ) with <Z0>.
 *  3. Four-qubit VQE-style ansatz with a Z-chain cost.
 */

#include "../../src/algorithms/diff/differentiable.h"
#include "../../src/quantum/state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

/* Compute <O> after running the circuit forward. */
static double forward_expect(moonlab_diff_circuit_t *c,
                              moonlab_diff_observable_t obs,
                              int qubit) {
    quantum_state_t s;
    quantum_state_init(&s, moonlab_diff_num_qubits(c));
    moonlab_diff_forward(c, &s);
    double v;
    switch (obs) {
        case MOONLAB_DIFF_OBS_Z: v = moonlab_diff_expect_z(&s, qubit); break;
        case MOONLAB_DIFF_OBS_X: v = moonlab_diff_expect_x(&s, qubit); break;
        default: v = 0.0; break;
    }
    quantum_state_free(&s);
    return v;
}

/* Finite-difference gradient vector.  Mutates the circuit's theta_k in
 * place; restores it after. */
static void finite_diff_gradient(moonlab_diff_circuit_t *c,
                                  moonlab_diff_observable_t obs,
                                  int qubit,
                                  double h,
                                  double *grad_out) {
    const size_t n = moonlab_diff_num_parameters(c);
    /* We need to read back the angles.  Cheat: the circuit stores
     * angles in its ops array but that's opaque.  Instead: store a
     * backup by sweeping +h then -h and restoring to the actual
     * cached value via a separate symmetric call. */
    for (size_t k = 0; k < n; k++) {
        /* Temporarily nudge theta_k; the mid-value is stored inside
         * the circuit, so we compute it via the +h/-h difference.
         * To avoid losing the original, do +h, evaluate, -h, evaluate,
         * +h to restore. */
        /* Using an internal buffer to capture the current angle before
         * perturbation isn't exposed via the API -- instead we do
         * +h -> measure plus; -2h -> measure minus; +h -> restore. */
        moonlab_diff_set_theta(c, k, h);
        (void)0;  /* No-op; set_theta needs absolute angle, not delta. */
        /* The API takes absolute theta; we don't have a get_theta.
         * Fall back to: save/restore via two complementary updates
         * assuming the caller has the angles vector. */
        (void)grad_out;
    }
}

/* Because we don't have get_theta(), the tests below pass their own
 * theta vector and set_theta() around evaluations. */
static void finite_diff_gradient_with_angles(moonlab_diff_circuit_t *c,
                                              const double *thetas,
                                              size_t n,
                                              moonlab_diff_observable_t obs,
                                              int qubit,
                                              double h,
                                              double *grad_out) {
    for (size_t k = 0; k < n; k++) {
        moonlab_diff_set_theta(c, k, thetas[k] + h);
        double fp = forward_expect(c, obs, qubit);
        moonlab_diff_set_theta(c, k, thetas[k] - h);
        double fm = forward_expect(c, obs, qubit);
        moonlab_diff_set_theta(c, k, thetas[k]);
        grad_out[k] = (fp - fm) / (2.0 * h);
    }
}

/* ---------------------------------------------------------------- */
/* 1. Analytic: RY(theta) |0> -> cos(th/2)|0> + sin(th/2)|1>
 *    <Z> = cos^2(th/2) - sin^2(th/2) = cos(theta).
 *    d<Z>/dtheta = -sin(theta).
 */
static void test_single_qubit_ry(void) {
    fprintf(stdout, "\n-- single-qubit RY(theta) analytic grad --\n");
    moonlab_diff_circuit_t *c = moonlab_diff_circuit_create(1);
    const double theta = 0.7;
    moonlab_diff_ry(c, 0, theta);

    quantum_state_t s;
    quantum_state_init(&s, 1);
    moonlab_diff_forward(c, &s);

    const double z = moonlab_diff_expect_z(&s, 0);
    const double z_analytic = cos(theta);
    fprintf(stdout, "    <Z> = %.6f (expect cos(theta) = %.6f)\n",
            z, z_analytic);
    CHECK(fabs(z - z_analytic) < 1e-12, "forward <Z> matches analytic");

    double grad[1] = {0};
    int rc = moonlab_diff_backward(c, &s, MOONLAB_DIFF_OBS_Z, 0, grad);
    CHECK(rc == 0, "backward returns success");
    const double g_analytic = -sin(theta);
    fprintf(stdout, "    d<Z>/dtheta = %.6f (analytic %.6f)\n",
            grad[0], g_analytic);
    CHECK(fabs(grad[0] - g_analytic) < 1e-10,
          "adjoint gradient matches -sin(theta)");

    quantum_state_free(&s);
    moonlab_diff_circuit_free(c);
}

/* ---------------------------------------------------------------- */
/* 2. HEA-like two-qubit: RY RZ H CNOT RY RZ RY; <Z0>, <X1>.
 *    Finite-difference agreement at h = 1e-4 should be ~1e-8. */
static void test_hea_two_qubit(void) {
    fprintf(stdout, "\n-- two-qubit HEA-like ansatz vs finite diff --\n");
    moonlab_diff_circuit_t *c = moonlab_diff_circuit_create(2);
    double thetas[5] = { 0.15, -0.42, 1.23, 0.88, -0.33 };

    moonlab_diff_ry(c, 0, thetas[0]);
    moonlab_diff_rz(c, 1, thetas[1]);
    moonlab_diff_h (c, 0);
    moonlab_diff_cnot(c, 0, 1);
    moonlab_diff_ry(c, 0, thetas[2]);
    moonlab_diff_rz(c, 0, thetas[3]);
    moonlab_diff_ry(c, 1, thetas[4]);

    CHECK(moonlab_diff_num_parameters(c) == 5, "5 parameters recorded");

    quantum_state_t s;
    quantum_state_init(&s, 2);
    moonlab_diff_forward(c, &s);
    double grad_adj[5];
    moonlab_diff_backward(c, &s, MOONLAB_DIFF_OBS_Z, 0, grad_adj);

    double grad_fd[5];
    finite_diff_gradient_with_angles(c, thetas, 5,
                                      MOONLAB_DIFF_OBS_Z, 0,
                                      1e-4, grad_fd);

    double max_err = 0.0;
    for (size_t k = 0; k < 5; k++) {
        const double err = fabs(grad_adj[k] - grad_fd[k]);
        if (err > max_err) max_err = err;
        fprintf(stdout, "    k=%zu  adj=%+.6f  fd=%+.6f  diff=%.2e\n",
                k, grad_adj[k], grad_fd[k], err);
    }
    CHECK(max_err < 1e-7, "adjoint within 1e-7 of finite-diff on <Z0>");

    /* Same circuit, <X1> observable. */
    double grad_adj_x[5], grad_fd_x[5];
    moonlab_diff_backward(c, &s, MOONLAB_DIFF_OBS_X, 1, grad_adj_x);
    finite_diff_gradient_with_angles(c, thetas, 5,
                                      MOONLAB_DIFF_OBS_X, 1,
                                      1e-4, grad_fd_x);
    max_err = 0.0;
    for (size_t k = 0; k < 5; k++) {
        const double e = fabs(grad_adj_x[k] - grad_fd_x[k]);
        if (e > max_err) max_err = e;
    }
    fprintf(stdout, "    <X1> max grad diff: %.2e\n", max_err);
    CHECK(max_err < 1e-7, "adjoint within 1e-7 of finite-diff on <X1>");

    quantum_state_free(&s);
    moonlab_diff_circuit_free(c);
}

/* ---------------------------------------------------------------- */
/* 3. Four-qubit VQE-like ansatz: multiple layers of single-qubit
 *    rotations interleaved with entangling CNOTs.
 */
static void test_four_qubit_vqe_ansatz(void) {
    fprintf(stdout, "\n-- 4-qubit VQE-like ansatz vs finite diff --\n");
    const uint32_t L = 4;
    moonlab_diff_circuit_t *c = moonlab_diff_circuit_create(L);
    /* 2 layers x (RY per qubit + RZ per qubit + ring of CNOTs). */
    const size_t n_params = 2 * L * 2;  /* 2 layers * 4 qubits * (RY+RZ) */
    double *thetas = (double*)malloc(n_params * sizeof(double));
    srand(0xC0DE);
    for (size_t i = 0; i < n_params; i++) {
        thetas[i] = ((double)rand() / RAND_MAX - 0.5) * 3.0;
    }
    size_t p = 0;
    for (int layer = 0; layer < 2; layer++) {
        for (uint32_t q = 0; q < L; q++) {
            moonlab_diff_ry(c, (int)q, thetas[p++]);
            moonlab_diff_rz(c, (int)q, thetas[p++]);
        }
        for (uint32_t q = 0; q + 1 < L; q++) {
            moonlab_diff_cnot(c, (int)q, (int)(q + 1));
        }
    }
    CHECK(moonlab_diff_num_parameters(c) == n_params,
          "%zu parameters recorded", n_params);

    quantum_state_t s;
    quantum_state_init(&s, L);
    moonlab_diff_forward(c, &s);
    double *grad_adj = (double*)malloc(n_params * sizeof(double));
    double *grad_fd  = (double*)malloc(n_params * sizeof(double));
    moonlab_diff_backward(c, &s, MOONLAB_DIFF_OBS_Z, 2, grad_adj);
    finite_diff_gradient_with_angles(c, thetas, n_params,
                                      MOONLAB_DIFF_OBS_Z, 2,
                                      1e-4, grad_fd);
    double max_err = 0.0;
    for (size_t k = 0; k < n_params; k++) {
        const double e = fabs(grad_adj[k] - grad_fd[k]);
        if (e > max_err) max_err = e;
    }
    fprintf(stdout, "    max |adjoint - fd| over %zu params: %.2e\n",
            n_params, max_err);
    CHECK(max_err < 1e-6, "adjoint within 1e-6 of finite-diff (4-qubit, 16 params)");

    free(thetas); free(grad_adj); free(grad_fd);
    quantum_state_free(&s);
    moonlab_diff_circuit_free(c);
}

int main(void) {
    fprintf(stdout, "=== differentiable moonlab tests ===\n");
    test_single_qubit_ry();
    test_hea_two_qubit();
    test_four_qubit_vqe_ansatz();

    fprintf(stdout, "\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
