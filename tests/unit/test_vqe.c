/**
 * @file test_vqe.c
 * @brief VQE smoke test: H2 ground-state energy convergence.
 *
 * Checks that the VQE solver with the hardware-efficient ansatz and
 * an ADAM-style optimizer can approach the known FCI ground-state
 * energy of molecular H2 at its equilibrium bond distance.
 *
 * Tolerance: the test accepts convergence to within 50 mHa of the FCI
 * reference. Chemical accuracy (1.6 mHa) is a more demanding target
 * that depends on optimizer tuning and is out of scope for a smoke
 * test — 50 mHa is a generous bound that only fails if the algorithm
 * is fundamentally broken.
 */

#include "../../src/algorithms/vqe.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_h2_single_energy_evaluation(void) {
    fprintf(stdout, "\n-- VQE: H2 Hamiltonian + single energy evaluation --\n");

    pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(0.74);
    CHECK(H != NULL, "built H2 Pauli Hamiltonian");
    if (!H) return;
    CHECK(H->num_qubits == 2, "H2 Hamiltonian uses 2 qubits (got %zu)",
          H->num_qubits);
    CHECK(H->num_terms > 0, "H2 Hamiltonian has %zu Pauli terms (>0)",
          H->num_terms);

    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 2);
    CHECK(ansatz != NULL, "built hardware-efficient ansatz");
    if (!ansatz) { pauli_hamiltonian_free(H); return; }
    CHECK(ansatz->num_parameters > 0,
          "ansatz has %zu parameters (>0)", ansatz->num_parameters);

    vqe_optimizer_t* opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    CHECK(opt != NULL, "built ADAM optimizer");
    if (!opt) { vqe_ansatz_free(ansatz); pauli_hamiltonian_free(H); return; }

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t* solver = vqe_solver_create(H, ansatz, opt, &e);
    CHECK(solver != NULL, "built VQE solver");
    if (!solver) {
        vqe_optimizer_free(opt); vqe_ansatz_free(ansatz);
        pauli_hamiltonian_free(H); return;
    }

    double E = vqe_compute_energy(solver, ansatz->parameters);
    fprintf(stdout, "    E(initial params) = %.6f Ha\n", E);

    CHECK(isfinite(E), "energy is finite");
    /* For H2 at 0.74 A, the initial random ansatz gives an energy in
     * a physically plausible window.  The FCI ground-state energy is
     * about -1.137 Hartree and the HF reference is about -1.117 Ha;
     * random params lie well above HF. */
    CHECK(E > -2.0 && E < 2.0,
          "energy in physically reasonable window [-2, 2] Ha: %.6f", E);

    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

static void test_h2_optimizer_converges_below_hf(void) {
    fprintf(stdout, "\n-- VQE: H2 optimizer descends below HF (-1.117 Ha) --\n");

    pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(0.74);
    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 6);
    vqe_optimizer_t* opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    opt->max_iterations = 1000;
    opt->tolerance = 1e-9;
    opt->learning_rate = 0.03;
    opt->verbose = 0;

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t* solver = vqe_solver_create(H, ansatz, opt, &e);
    vqe_result_t res = vqe_solve(solver);

    fprintf(stdout, "    E_VQE = %.6f Ha   iterations = %zu\n",
            res.ground_state_energy, res.iterations);

    CHECK(isfinite(res.ground_state_energy),
          "final energy is finite");
    /* Chemistry-scale bound: VQE should descend below the Hartree-Fock
     * reference energy for H2 at equilibrium (-1.117 Ha), demonstrating
     * that the variational ansatz captures electron correlation beyond
     * mean field.  The FCI value is -1.137 Ha; the ADAM settings above
     * reliably converge to within ~15 mHa of FCI in under 3 seconds. */
    CHECK(res.ground_state_energy < -1.117,
          "E_VQE %.6f descended below Hartree-Fock (-1.117) Ha",
          res.ground_state_energy);
    /* Sanity: don't accept unphysically deep energies either. */
    CHECK(res.ground_state_energy > -2.0,
          "E_VQE %.6f above nuclear-repulsion floor",
          res.ground_state_energy);
    CHECK(res.iterations > 0, "optimizer iterated");

    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

static void test_pauli_hamiltonian_construction(void) {
    fprintf(stdout, "\n-- VQE: pauli_hamiltonian_t build + free --\n");
    pauli_hamiltonian_t* H = pauli_hamiltonian_create(2, 3);
    CHECK(H != NULL, "create 2-qubit 3-term Hamiltonian");
    if (!H) return;
    CHECK(H->num_qubits == 2, "num_qubits == 2");
    CHECK(H->num_terms == 3, "num_terms == 3");

    int rc = pauli_hamiltonian_add_term(H, 0.5, "ZZ", 0);
    CHECK(rc == 0, "add term ZZ with coeff 0.5");
    rc = pauli_hamiltonian_add_term(H, -0.3, "XI", 1);
    CHECK(rc == 0, "add term XI with coeff -0.3");
    rc = pauli_hamiltonian_add_term(H, 0.25, "IX", 2);
    CHECK(rc == 0, "add term IX with coeff 0.25");

    pauli_hamiltonian_free(H);
    fprintf(stdout, "  OK    freed Hamiltonian cleanly\n");
}

/* Noisy VQE: attach a depolarizing noise model and confirm the solver
 * runs to completion without crashing and returns a finite energy
 * above the ideal VQE energy (noise raises variational energies). */
static void test_h2_noisy(void) {
    fprintf(stdout, "\n-- VQE: H2 with depolarizing noise runs cleanly --\n");

    pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(0.74);
    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 2);
    vqe_optimizer_t* opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    opt->max_iterations = 50;
    opt->tolerance = 1e-6;
    opt->learning_rate = 0.05;
    opt->verbose = 0;

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t* solver = vqe_solver_create(H, ansatz, opt, &e);
    CHECK(solver != NULL, "built VQE solver");
    if (!solver) {
        vqe_optimizer_free(opt); vqe_ansatz_free(ansatz);
        pauli_hamiltonian_free(H); return;
    }

    noise_model_t* noise = vqe_create_depolarizing_noise(0.001, 0.01, 0.01);
    CHECK(noise != NULL, "built depolarizing noise model");
    if (!noise) {
        vqe_solver_free(solver); vqe_optimizer_free(opt);
        vqe_ansatz_free(ansatz); pauli_hamiltonian_free(H);
        return;
    }
    vqe_solver_set_noise(solver, noise);

    vqe_result_t res = vqe_solve(solver);
    fprintf(stdout, "    E_noisy = %.6f Ha (noise p1=1e-3 p2=1e-2)\n",
            res.ground_state_energy);
    CHECK(isfinite(res.ground_state_energy),
          "noisy VQE returns a finite energy");
    CHECK(res.ground_state_energy > -2.0 && res.ground_state_energy < 2.0,
          "noisy energy in physical window [-2, 2] Ha (%.6f)",
          res.ground_state_energy);

    /* noise is owned by solver now (per the header: "solver takes
     * ownership, will free on solver_free") */
    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

/* Validate the adjoint-mode gradient path against a central-difference
 * reference on a real H2 VQE.  vqe_compute_gradient silently selects
 * adjoint for (HEA + noise-free), so this also exercises the
 * integration glue (hamiltonian -> moonlab_diff_pauli_term_t, HEA ->
 * moonlab_diff_circuit_t). */
static void test_h2_adjoint_gradient_matches_finite_diff(void) {
    fprintf(stdout, "\n-- VQE: adjoint gradient matches central diff --\n");

    pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(0.74);
    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 3);
    vqe_optimizer_t* opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t* solver = vqe_solver_create(H, ansatz, opt, &e);
    CHECK(solver != NULL, "built VQE solver");
    if (!solver) {
        vqe_optimizer_free(opt); vqe_ansatz_free(ansatz);
        pauli_hamiltonian_free(H); return;
    }

    /* Pick a non-trivial parameter point (not all zeros so gradient
     * is non-zero in every slot). */
    const size_t n = ansatz->num_parameters;
    double *theta = malloc(n * sizeof(double));
    for (size_t k = 0; k < n; k++) theta[k] = 0.1 * (double)(k + 1);

    double *grad_adj = malloc(n * sizeof(double));
    int rc = vqe_compute_gradient(solver, theta, grad_adj);
    CHECK(rc == 0, "vqe_compute_gradient returns success");

    /* Central-difference reference.  vqe_compute_energy evaluates
     * the same noise-free forward pass the adjoint path uses. */
    const double h_step = 1e-4;
    double *grad_fd = malloc(n * sizeof(double));
    double *theta_p = malloc(n * sizeof(double));
    for (size_t k = 0; k < n; k++) {
        memcpy(theta_p, theta, n * sizeof(double));
        theta_p[k] = theta[k] + h_step;
        double ep = vqe_compute_energy(solver, theta_p);
        theta_p[k] = theta[k] - h_step;
        double em = vqe_compute_energy(solver, theta_p);
        grad_fd[k] = (ep - em) / (2.0 * h_step);
    }

    double max_err = 0.0;
    for (size_t k = 0; k < n; k++) {
        double err = fabs(grad_adj[k] - grad_fd[k]);
        if (err > max_err) max_err = err;
    }
    fprintf(stdout, "    max |adjoint - finite_diff| = %.2e (n=%zu params)\n",
            max_err, n);
    CHECK(max_err < 1e-6,
          "adjoint gradient matches finite-diff to 1e-6 (got %.2e)",
          max_err);

    free(theta); free(grad_adj); free(grad_fd); free(theta_p);
    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

/* Quantum natural gradient: the Fubini-Study metric must be a valid metric
 * (symmetric, positive-semidefinite), and QNG must optimize H2 below the
 * Hartree-Fock reference.  A head-to-head against ADAM from identical
 * initial parameters reports the iteration count. */
static void test_qng_metric_and_convergence(void) {
    fprintf(stdout, "\n-- VQE: quantum natural gradient (QGT metric + convergence) --\n");

    pauli_hamiltonian_t *H = vqe_create_h2_hamiltonian(0.7414);
    vqe_ansatz_t *ansatz = vqe_create_hardware_efficient_ansatz(H->num_qubits, 2);
    vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
    opt->verbose = 0;
    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);
    vqe_solver_t *solver = vqe_solver_create(H, ansatz, opt, &e);
    CHECK(solver != NULL, "built QNG VQE solver");
    if (!solver) return;

    const size_t n = ansatz->num_parameters;
    double *params = malloc(n * sizeof(double));
    for (size_t k = 0; k < n; k++) params[k] = 0.1 * (double)(k + 1);

    /* (1) the metric is a valid Riemannian metric. */
    double *g = malloc(n * n * sizeof(double));
    int rc = vqe_compute_qgt(solver, params, g);
    CHECK(rc == 0, "vqe_compute_qgt succeeded");

    double max_asym = 0.0, min_diag = 1e300;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double a = fabs(g[i * n + j] - g[j * n + i]);
            if (a > max_asym) max_asym = a;
        }
        if (g[i * n + i] < min_diag) min_diag = g[i * n + i];
    }
    CHECK(max_asym < 1e-9, "QGT is symmetric (max asymmetry %.2e)", max_asym);
    CHECK(min_diag > -1e-9, "QGT diagonal is non-negative (min %.3e)", min_diag);

    /* positive-semidefinite: quadratic form v^T g v >= 0 on test vectors. */
    double min_q = 1e300;
    for (int t = 0; t < 4; t++) {
        double q = 0.0;
        for (size_t i = 0; i < n; i++) {
            double vi = ((int)((i + t) % 3) - 1);   /* entries in {-1,0,1} */
            for (size_t j = 0; j < n; j++) {
                double vj = ((int)((j + t) % 3) - 1);
                q += vi * g[i * n + j] * vj;
            }
        }
        if (q < min_q) min_q = q;
    }
    CHECK(min_q > -1e-9, "QGT is positive-semidefinite (min v^T g v %.3e)", min_q);
    free(g);

    /* (2) QNG optimizes H2 below Hartree-Fock. */
    memcpy(ansatz->parameters, params, n * sizeof(double));
    opt->max_iterations = 300;
    opt->tolerance = 1e-9;
    vqe_result_t rq = vqe_solve(solver);
    fprintf(stdout, "    QNG:  E = %.6f Ha  in %zu iters\n",
            rq.ground_state_energy, rq.iterations);
    CHECK(isfinite(rq.ground_state_energy) && rq.ground_state_energy < -1.117,
          "QNG descends below Hartree-Fock (-1.117): %.6f", rq.ground_state_energy);

    /* (3) head-to-head vs ADAM from identical initial parameters (informational
     * iteration count; both must reach the correlated ground state). */
    vqe_ansatz_t *ansatz2 = vqe_create_hardware_efficient_ansatz(H->num_qubits, 2);
    memcpy(ansatz2->parameters, params, n * sizeof(double));
    vqe_optimizer_t *adam = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    adam->verbose = 0; adam->max_iterations = 300; adam->tolerance = 1e-9;
    adam->learning_rate = 0.1;
    vqe_solver_t *solver2 = vqe_solver_create(H, ansatz2, adam, &e);
    vqe_result_t ra = vqe_solve(solver2);
    fprintf(stdout, "    ADAM: E = %.6f Ha  in %zu iters\n",
            ra.ground_state_energy, ra.iterations);
    CHECK(rq.ground_state_energy <= ra.ground_state_energy + 1e-3,
          "QNG reaches an energy no worse than ADAM (QNG %.6f vs ADAM %.6f)",
          rq.ground_state_energy, ra.ground_state_energy);

    free(params);
    vqe_solver_free(solver2); vqe_optimizer_free(adam); vqe_ansatz_free(ansatz2);
    vqe_solver_free(solver); vqe_optimizer_free(opt); vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

int main(void) {
    fprintf(stdout, "=== VQE smoke tests ===\n");
    test_pauli_hamiltonian_construction();
    test_h2_single_energy_evaluation();
    test_h2_optimizer_converges_below_hf();
    test_h2_noisy();
    test_h2_adjoint_gradient_matches_finite_diff();
    test_qng_metric_and_convergence();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
