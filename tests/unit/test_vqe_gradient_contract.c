/**
 * @file test_vqe_gradient_contract.c
 * @brief VQE exact-or-error gradient contract (Eshkol RFC 4.4).
 *
 * vqe_compute_gradient must return an exact gradient or an explicit error:
 *   - noise-free: succeeds (adjoint or exact parameter shift);
 *   - noise attached: returns VQE_GRADIENT_ERR_NOT_EXACT unless the caller
 *     opts into stochastic PSR via vqe_solver_set_allow_stochastic_gradient;
 *   - NULL ansatz: returns VQE_GRADIENT_ERR_INVALID (guarded, no deref).
 * Also exercises vqe_result_free.
 */

#include "../../src/algorithms/vqe.h"
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
    fprintf(stdout, "=== VQE exact-or-error gradient contract ===\n");

    pauli_hamiltonian_t *H = vqe_create_h2_hamiltonian(0.74);
    CHECK(H != NULL, "H2 Hamiltonian");
    if (!H) return 1;

    vqe_ansatz_t *ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 2);
    vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);
    vqe_solver_t *solver = vqe_solver_create(H, ansatz, opt, &e);
    CHECK(solver != NULL, "solver");
    if (!solver) { return 1; }

    double *grad = calloc(ansatz->num_parameters, sizeof(double));

    /* 1. Noise-free: exact gradient succeeds. */
    int rc0 = vqe_compute_gradient(solver, ansatz->parameters, grad);
    fprintf(stdout, "  noise-free rc = %d (expect %d)\n", rc0, VQE_GRADIENT_SUCCESS);
    CHECK(rc0 == VQE_GRADIENT_SUCCESS, "noise-free gradient must be exact");

    /* 2. Noise attached, default contract: refuse with NOT_EXACT. */
    noise_model_t *noise = vqe_create_depolarizing_noise(1e-3, 1e-2, 1e-2);
    CHECK(noise != NULL, "noise model");
    vqe_solver_set_noise(solver, noise);
    int rc1 = vqe_compute_gradient(solver, ansatz->parameters, grad);
    fprintf(stdout, "  noisy default rc = %d (expect %d)\n", rc1, VQE_GRADIENT_ERR_NOT_EXACT);
    CHECK(rc1 == VQE_GRADIENT_ERR_NOT_EXACT,
          "noisy gradient must refuse (exact-or-error), got %d", rc1);

    /* 3. Opt into stochastic PSR: now allowed. */
    vqe_solver_set_allow_stochastic_gradient(solver, 1);
    int rc2 = vqe_compute_gradient(solver, ansatz->parameters, grad);
    fprintf(stdout, "  noisy opted-in rc = %d (expect %d)\n", rc2, VQE_GRADIENT_SUCCESS);
    CHECK(rc2 == VQE_GRADIENT_SUCCESS, "opted-in stochastic PSR must run, got %d", rc2);

    /* 4. NULL-ansatz guard. */
    vqe_ansatz_t *saved = solver->ansatz;
    solver->ansatz = NULL;
    int rc3 = vqe_compute_gradient(solver, ansatz->parameters, grad);
    solver->ansatz = saved;
    fprintf(stdout, "  null-ansatz rc = %d (expect %d)\n", rc3, VQE_GRADIENT_ERR_INVALID);
    CHECK(rc3 == VQE_GRADIENT_ERR_INVALID, "null ansatz must be guarded, got %d", rc3);

    /* 5. vqe_result_free on a solve result (drop noise first to keep it fast). */
    vqe_solver_set_noise(solver, NULL);
    vqe_result_t result = vqe_solve(solver);
    vqe_result_free(&result);
    CHECK(result.optimal_parameters == NULL, "vqe_result_free nulls owned array");

    free(grad);
    vqe_solver_free(solver);
    vqe_ansatz_free(ansatz);
    vqe_optimizer_free(opt);
    pauli_hamiltonian_free(H);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
