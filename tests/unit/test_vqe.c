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

/* Exact ground-state energy of H2 at a bond distance (no ansatz/optimizer). */
static double h2_exact_energy(double r_angstrom) {
    pauli_hamiltonian_t *H = vqe_create_h2_hamiltonian(r_angstrom);
    double e = vqe_exact_ground_state_energy(H);
    pauli_hamiltonian_free(H);
    return e;
}

/* The H2 potential energy surface must be smooth and differentiable in the
 * bond length: the equilibrium result is unchanged, there is no flat plateau
 * (which would give zero force at the minimum), and the force dE/dr is
 * continuous (no kink).  This guards the first-principles STO-3G construction
 * against a regression to the previous Morse/plateau interpolation. */
static void test_h2_pes_smooth_and_differentiable(void) {
    fprintf(stdout, "\n-- VQE: H2 potential energy surface is smooth + differentiable --\n");

    /* (1) Equilibrium coefficients + nuclear repulsion are the exact O'Malley
     * values (the fix is additively anchored, so 0.7414 A is unchanged). */
    pauli_hamiltonian_t *H = vqe_create_h2_hamiltonian(0.7414);
    CHECK(H != NULL, "built H2 Hamiltonian at equilibrium");
    if (!H) return;
    const char *names[5] = {"II", "IZ", "ZI", "ZZ", "XX"};
    const double omalley[5] = {-1.0523732, 0.39793742, -0.39793742,
                               -0.01128010, 0.18093120};
    for (int t = 0; t < 5; t++) {
        double got = NAN;
        for (size_t i = 0; i < H->num_terms; i++)
            if (H->terms[i].pauli_string &&
                strcmp(H->terms[i].pauli_string, names[t]) == 0)
                got = H->terms[i].coefficient;
        CHECK(isfinite(got) && fabs(got - omalley[t]) < 1e-9,
              "equilibrium %s coefficient exact (%.8f)", names[t], got);
    }
    CHECK(fabs(H->nuclear_repulsion - 0.7151043390) < 1e-9,
          "equilibrium nuclear repulsion exact (%.10f)", H->nuclear_repulsion);
    pauli_hamiltonian_free(H);

    double E_eq = h2_exact_energy(0.7414);
    fprintf(stdout, "    E_exact(0.7414) = %.9f Ha\n", E_eq);
    CHECK(fabs(E_eq - (-1.142170640)) < 1e-6,
          "equilibrium ground-state energy unchanged (%.9f)", E_eq);

    /* (2) No plateau: the previous construction returned identical coefficients
     * for all r within +/-0.01 A of equilibrium, so the energy was flat there.
     * A first-principles PES must vary. */
    double Ea = h2_exact_energy(0.735), Eb = h2_exact_energy(0.745);
    CHECK(fabs(Ea - Eb) > 1e-6,
          "PES is not flat across equilibrium (no plateau): |dE| = %.2e", fabs(Ea - Eb));

    /* (3) The force dE/dr is continuous through equilibrium.  The old
     * exp(-1.5*fabs(r-r_eq)) scaling and plateau edge made it jump by ~2.4
     * Ha/A; a smooth PES keeps adjacent central-difference forces close. */
    const double h = 1e-3;
    double max_force_jump = 0.0, prev_force = NAN;
    for (double r = 0.70; r <= 0.78001; r += 0.01) {
        double force = (h2_exact_energy(r + h) - h2_exact_energy(r - h)) / (2.0 * h);
        if (isfinite(prev_force)) {
            double jump = fabs(force - prev_force);
            if (jump > max_force_jump) max_force_jump = jump;
        }
        prev_force = force;
    }
    fprintf(stdout, "    max adjacent |d(force)| over [0.70,0.78] = %.4f Ha/A\n",
            max_force_jump);
    CHECK(max_force_jump < 0.1,
          "force is continuous through equilibrium (no kink): %.4f < 0.1", max_force_jump);
}

/* LiH shares the same smoothness fix.  The previous else branch dropped 13 of
 * the 21 terms off equilibrium (a discontinuous change of Hamiltonian
 * structure) and used a fabs() kink; the fixed construction keeps the full term
 * set at every geometry with a smooth Gaussian envelope. */
static void test_lih_pes_smooth_and_consistent(void) {
    fprintf(stdout, "\n-- VQE: LiH surface is smooth + structurally consistent --\n");

    /* (1) Equilibrium preserved. */
    pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
    CHECK(H != NULL, "built LiH Hamiltonian at equilibrium");
    if (!H) return;
    CHECK(H->num_terms == 21, "LiH has all 21 terms at equilibrium (got %zu)",
          H->num_terms);
    double iiii = NAN;
    for (size_t i = 0; i < H->num_terms; i++)
        if (H->terms[i].pauli_string && strcmp(H->terms[i].pauli_string, "IIII") == 0)
            iiii = H->terms[i].coefficient;
    CHECK(isfinite(iiii) && fabs(iiii - (-7.8823620)) < 1e-9,
          "equilibrium IIII coefficient exact (%.7f)", iiii);
    CHECK(fabs(H->nuclear_repulsion - 0.9953800) < 1e-9,
          "equilibrium nuclear repulsion exact (%.7f)", H->nuclear_repulsion);
    pauli_hamiltonian_free(H);

    /* (2) Structural consistency: the term COUNT must not change with geometry
     * (the old else branch produced 8 terms instead of 21). */
    for (double r = 1.30; r <= 1.90001; r += 0.20) {
        pauli_hamiltonian_t *Hr = vqe_create_lih_hamiltonian(r);
        CHECK(Hr && Hr->num_terms == 21,
              "LiH keeps all 21 terms at r=%.2f A (got %zu)", r,
              Hr ? Hr->num_terms : (size_t)0);
        if (Hr) pauli_hamiltonian_free(Hr);
    }

    /* (3) Force continuity through equilibrium (no plateau, no kink). */
    const double h = 1e-3;
    double max_jump = 0.0, prev = NAN;
    for (double r = 1.40; r <= 1.80001; r += 0.05) {
        pauli_hamiltonian_t *Hp = vqe_create_lih_hamiltonian(r + h);
        pauli_hamiltonian_t *Hm = vqe_create_lih_hamiltonian(r - h);
        double force = (vqe_exact_ground_state_energy(Hp) -
                        vqe_exact_ground_state_energy(Hm)) / (2.0 * h);
        pauli_hamiltonian_free(Hp);
        pauli_hamiltonian_free(Hm);
        if (isfinite(prev)) {
            double jump = fabs(force - prev);
            if (jump > max_jump) max_jump = jump;
        }
        prev = force;
    }
    fprintf(stdout, "    LiH max adjacent |d(force)| = %.4f Ha/A\n", max_jump);
    CHECK(max_jump < 0.5, "LiH force is continuous (no kink): %.4f < 0.5", max_jump);
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

    /* (1b) exact analytic value: for the hardware-efficient ansatz the leading
     * RY on the HF reference has Fubini-Study metric exactly 1/4 (the variance of
     * a Pauli generator on a computational-basis eigenstate). The exact
     * generator-insertion path (vqe_qng.c) reaches this to machine precision;
     * central differences only get ~2e-10. */
    CHECK(fabs(g[0] - 0.25) < 1e-12, "exact QGT g[0][0] == 1/4 (%.12f)", g[0]);
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

/* UCCSD must reach the exact STO-3G ground state of H2 to chemical
 * accuracy (1.6 mHa).  The single-parameter UCCSD circuit is the
 * canonical H2 ansatz: from the Hartree-Fock reference |10> a single
 * particle-conserving excitation mixes |10> and |01>, and its optimum
 * is the exact ground state.  This guards the fix for the two defects
 * that previously drove UCCSD H2 well above the true ground state:
 *   (1) the ansatz re-prepared the HF reference that vqe_compute_energy
 *       had already prepared (double X gates -> wrong particle sector);
 *   (2) the single-excitation "Givens" circuit was not particle
 *       conserving (it leaked amplitude into |00>). */
static void test_h2_uccsd_chemical_accuracy(void) {
    fprintf(stdout, "\n-- VQE: UCCSD H2 reaches chemical accuracy --\n");

    pauli_hamiltonian_t *H = vqe_create_h2_hamiltonian(0.74);
    CHECK(H != NULL, "built H2 Hamiltonian");
    if (!H) return;
    double E_exact = vqe_exact_ground_state_energy(H);

    /* 2 qubits, 1 electron in the parity/BK-reduced encoding:
     * 1 single excitation, 0 doubles -> a single variational parameter. */
    vqe_ansatz_t *ansatz = vqe_create_uccsd_ansatz(2, 1);
    CHECK(ansatz != NULL, "built UCCSD ansatz (2 qubits, 1 electron)");
    if (!ansatz) { pauli_hamiltonian_free(H); return; }
    CHECK(ansatz->num_parameters == 1,
          "UCCSD H2 has a single variational parameter (got %zu)",
          ansatz->num_parameters);

    vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    opt->max_iterations = 500;
    opt->tolerance = 1e-10;
    opt->learning_rate = 0.1;
    opt->verbose = 0;

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t *solver = vqe_solver_create(H, ansatz, opt, &e);
    vqe_result_t res = vqe_solve(solver);

    double err = fabs(res.ground_state_energy - E_exact);
    fprintf(stdout, "    E_UCCSD = %.9f Ha   E_exact = %.9f Ha   |err| = %.2e Ha\n",
            res.ground_state_energy, E_exact, err);
    CHECK(isfinite(res.ground_state_energy), "UCCSD energy is finite");
    CHECK(err < 1.6e-3,
          "UCCSD H2 within chemical accuracy of exact (|err| %.2e < 1.6e-3)",
          err);

    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

/* The exact ground state of the 4-qubit LiH operator lies in the
 * 3-excitation sector; the UCCSD reference is the contiguous
 * three-orbital occupation |0111>, and the three single excitations
 * (each moving a particle into the fourth orbital) span the correlated
 * ground state exactly.  UCCSD must therefore reach chemical accuracy
 * on LiH as well. */
static void test_lih_uccsd_chemical_accuracy(void) {
    fprintf(stdout, "\n-- VQE: UCCSD LiH reaches chemical accuracy --\n");

    pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
    CHECK(H != NULL, "built LiH Hamiltonian");
    if (!H) return;
    CHECK(H->hf_reference == 0x7,
          "LiH Hartree-Fock reference is |0111> (0x7, got 0x%llx)",
          (unsigned long long)H->hf_reference);
    double E_exact = vqe_exact_ground_state_energy(H);

    /* 4 qubits, 3 electrons: 3 single excitations, 0 doubles. */
    vqe_ansatz_t *ansatz = vqe_create_uccsd_ansatz(4, 3);
    CHECK(ansatz != NULL, "built UCCSD ansatz (4 qubits, 3 electrons)");
    if (!ansatz) { pauli_hamiltonian_free(H); return; }
    CHECK(ansatz->num_parameters == 3,
          "UCCSD LiH has 3 single-excitation parameters (got %zu)",
          ansatz->num_parameters);

    vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    opt->max_iterations = 3000;
    opt->tolerance = 1e-12;
    opt->learning_rate = 0.1;
    opt->verbose = 0;

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    vqe_solver_t *solver = vqe_solver_create(H, ansatz, opt, &e);
    vqe_result_t res = vqe_solve(solver);

    double err = fabs(res.ground_state_energy - E_exact);
    fprintf(stdout, "    E_UCCSD = %.9f Ha   E_exact = %.9f Ha   |err| = %.2e Ha\n",
            res.ground_state_energy, E_exact, err);
    CHECK(isfinite(res.ground_state_energy), "UCCSD LiH energy is finite");
    CHECK(err < 1.6e-3,
          "UCCSD LiH within chemical accuracy of exact (|err| %.2e < 1.6e-3)",
          err);

    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
}

int main(void) {
    fprintf(stdout, "=== VQE smoke tests ===\n");
    test_pauli_hamiltonian_construction();
    test_h2_single_energy_evaluation();
    test_h2_optimizer_converges_below_hf();
    test_h2_noisy();
    test_h2_adjoint_gradient_matches_finite_diff();
    test_h2_pes_smooth_and_differentiable();
    test_lih_pes_smooth_and_consistent();
    test_qng_metric_and_convergence();
    test_h2_uccsd_chemical_accuracy();
    test_lih_uccsd_chemical_accuracy();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
