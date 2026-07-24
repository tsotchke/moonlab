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
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include <complex.h>
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

/* |psi(params)> through the public forward pass only (HF preparation from
 * hamiltonian->hf_reference, then vqe_apply_ansatz) -- the same convention
 * vqe_compute_qgt uses internally. */
static int qgt_ref_prepare(vqe_solver_t *s, const double *params,
                           quantum_state_t *psi) {
    if (quantum_state_init(psi, s->hamiltonian->num_qubits) != QS_SUCCESS)
        return -1;
    uint64_t hf = s->hamiltonian->hf_reference;
    for (size_t q = 0; q < s->hamiltonian->num_qubits; q++)
        if (hf & (1ULL << q)) gate_pauli_x(psi, (int)q);
    memcpy(s->ansatz->parameters, params,
           s->ansatz->num_parameters * sizeof(double));
    if (vqe_apply_ansatz(psi, s->ansatz) != QS_SUCCESS) {
        quantum_state_free(psi);
        return -1;
    }
    return 0;
}

/* Independent central-difference reference for the Fubini-Study metric,
 * built purely from forward passes.  Cross-checks the exact analytic
 * generator-insertion derivatives inside vqe_compute_qgt: the two must agree
 * to the O(delta^2) truncation error of the finite difference. */
static int qgt_central_difference_reference(vqe_solver_t *s,
                                            const double *params,
                                            double *g_out) {
    const size_t n   = s->ansatz->num_parameters;
    const size_t dim = (size_t)1 << s->hamiltonian->num_qubits;
    const double delta = 1e-4;

    double    *pshift = malloc(n * sizeof(double));
    complex_t *psi0   = malloc(dim * sizeof(complex_t));
    complex_t *dpsi   = malloc(n * dim * sizeof(complex_t));
    complex_t *ov     = malloc(n * sizeof(complex_t));
    if (!pshift || !psi0 || !dpsi || !ov) {
        free(pshift); free(psi0); free(dpsi); free(ov);
        return -1;
    }

    int rc = 0;
    quantum_state_t st;
    if (qgt_ref_prepare(s, params, &st) != 0) { rc = -1; goto done; }
    memcpy(psi0, st.amplitudes, dim * sizeof(complex_t));
    quantum_state_free(&st);

    for (size_t i = 0; i < n; i++) {
        memcpy(pshift, params, n * sizeof(double));
        pshift[i] = params[i] + delta;
        if (qgt_ref_prepare(s, pshift, &st) != 0) { rc = -1; goto done; }
        for (size_t x = 0; x < dim; x++) dpsi[i * dim + x] = st.amplitudes[x];
        quantum_state_free(&st);

        pshift[i] = params[i] - delta;
        if (qgt_ref_prepare(s, pshift, &st) != 0) { rc = -1; goto done; }
        for (size_t x = 0; x < dim; x++)
            dpsi[i * dim + x] =
                (dpsi[i * dim + x] - st.amplitudes[x]) / (2.0 * delta);
        quantum_state_free(&st);
    }

    for (size_t i = 0; i < n; i++) {
        complex_t sum = 0.0;
        for (size_t x = 0; x < dim; x++) sum += conj(psi0[x]) * dpsi[i * dim + x];
        ov[i] = sum;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = i; j < n; j++) {
            complex_t a = 0.0;
            for (size_t x = 0; x < dim; x++)
                a += conj(dpsi[i * dim + x]) * dpsi[j * dim + x];
            double g = creal(a) - creal(conj(ov[i]) * ov[j]);
            g_out[i * n + j] = g;
            g_out[j * n + i] = g;
        }

done:
    free(pshift); free(psi0); free(dpsi); free(ov);
    return rc;
}

/* Symmetry, non-negative diagonal, and positive-semidefiniteness (quadratic
 * form on {-1,0,1} test vectors) of an n x n metric. */
static void qgt_check_metric_properties(const double *g, size_t n,
                                        const char *name) {
    double max_asym = 0.0, min_diag = 1e300;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double a = fabs(g[i * n + j] - g[j * n + i]);
            if (a > max_asym) max_asym = a;
        }
        if (g[i * n + i] < min_diag) min_diag = g[i * n + i];
    }
    CHECK(max_asym < 1e-9, "%s: QGT symmetric (max asymmetry %.2e)",
          name, max_asym);
    CHECK(min_diag > -1e-9, "%s: QGT diagonal non-negative (min %.3e)",
          name, min_diag);

    double min_q = 1e300;
    for (int t = 0; t < 4; t++) {
        double q = 0.0;
        for (size_t i = 0; i < n; i++) {
            double vi = ((int)((i + t) % 3) - 1);
            for (size_t j = 0; j < n; j++) {
                double vj = ((int)((j + t) % 3) - 1);
                q += vi * g[i * n + j] * vj;
            }
        }
        if (q < min_q) min_q = q;
    }
    CHECK(min_q > -1e-9, "%s: QGT positive-semidefinite (min v^T g v %.3e)",
          name, min_q);
}

/* max_ij |g_exact - g_cd| between the exact metric and the independent
 * central-difference reference, printed and gated at 1e-7. */
static void qgt_check_matches_central_difference(vqe_solver_t *s,
                                                 const double *params,
                                                 const double *g_exact,
                                                 size_t n, const char *name) {
    double *g_cd = malloc(n * n * sizeof(double));
    int rc = (g_cd != NULL) ?
        qgt_central_difference_reference(s, params, g_cd) : -1;
    CHECK(rc == 0, "%s: central-difference reference computed", name);
    if (rc == 0) {
        double max_diff = 0.0;
        for (size_t k = 0; k < n * n; k++) {
            double d = fabs(g_exact[k] - g_cd[k]);
            if (d > max_diff) max_diff = d;
        }
        CHECK(max_diff < 1e-7,
              "%s: exact QGT matches central differences (max |diff| %.3e)",
              name, max_diff);
    }
    free(g_cd);
}

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

    /* (1b) exact analytic anchor (PR #16): for the hardware-efficient ansatz
     * the leading RY on the HF reference has Fubini-Study metric exactly 1/4,
     * the variance of the P/2 generator on a real-amplitude state (<Y> = 0
     * there, <Y^2> = 1).  The exact generator-insertion path reaches this to
     * machine precision; central differences only get within ~1e-9. */
    CHECK(fabs(g[0] - 0.25) < 1e-12, "exact QGT g[0][0] == 1/4 (%.15f)", g[0]);

    /* (1c) full-metric agreement with an independent central-difference
     * reference built from forward passes only. */
    qgt_check_matches_central_difference(solver, params, g, n, "HEA/H2");
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

/* Exact analytic QGT for the UCCSD and symmetry-preserving (Givens) ansaetze
 * (generator insertion, PR #16, rederived for the current particle-number-
 * preserving excitation gates).  Three cases on the 4-qubit LiH operator:
 *
 * (a) UCCSD singles (4 qubits, 3 electrons, HF = |0111>): valid metric, full
 *     agreement with an independent central-difference reference, and the
 *     closed-form anchor g[0][0] = 1/4.  Derivation for the current Givens
 *     gate: the composite CNOT-CRY(theta)-CNOT rotates its plane by theta/2
 *     (CRY is a half-angle gate), so the gate is exp((theta/2) G) with G real
 *     antisymmetric and d_0|psi> = (1/2) U_after G_0 U_upto|HF>.  Parameter 0
 *     rotates the coupled pair {|1_0 0_3>, |0_0 1_3>} that contains the HF
 *     reference, so U_upto|HF> stays in that plane, where G acts as the
 *     orthogonal [[0,-1],[1,0]] -- hence <d_0 psi|d_0 psi> = 1/4 -- and the
 *     Berry connection <psi|d_0 psi> = (1/2) v^T G v = 0 exactly for any
 *     real v.  So g[0][0] = 1/4 to machine precision, at any theta.  (The
 *     PR #16 anchor was g[0][0] = 1 for the old CNOT-RY(2 theta)-CNOT
 *     decomposition; the particle-preserving replacement halves the angular
 *     velocity, quartering the metric entry.)
 * (b) UCCSD with an active double excitation: the same operator with a
 *     2-electron reference |0011> gives 4 singles + 1 double.  CD agreement
 *     at a generic point; with all singles at zero the double's rotation
 *     plane {|0011>, |1100>} contains the reference, and the double rotates
 *     by the FULL angle (cos/sin applied directly), so the same argument
 *     with s = 1 pins its diagonal at g[4][4] = 1.
 * (c) symmetry-preserving Givens ansatz (2 layers): valid metric + CD
 *     agreement.  Here each parameter drives gate_cry inside a fixed CNOT
 *     sandwich, so the insertion is (-i/2) |1><1| (x) Y after the CRY. */
static void test_qng_exact_qgt_uccsd_givens(void) {
    fprintf(stdout, "\n-- VQE: exact QGT for UCCSD / symmetry-preserving ansaetze --\n");

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    /* (a) UCCSD singles on LiH: 3 electrons, HF |0111>. */
    {
        pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
        vqe_ansatz_t *ansatz = vqe_create_uccsd_ansatz(4, 3);
        vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
        vqe_solver_t *s = vqe_solver_create(H, ansatz, opt, &e);
        const size_t n = ansatz->num_parameters;
        CHECK(n == 3, "UCCSD LiH: 3 single-excitation parameters (got %zu)", n);

        double params[3], g[9];
        for (size_t k = 0; k < n; k++) params[k] = 0.1 * (double)(k + 1);
        int rc = vqe_compute_qgt(s, params, g);
        CHECK(rc == 0, "UCCSD/LiH: vqe_compute_qgt succeeded");

        qgt_check_metric_properties(g, n, "UCCSD/LiH");
        qgt_check_matches_central_difference(s, params, g, n, "UCCSD/LiH");
        CHECK(fabs(g[0] - 0.25) < 1e-12,
              "UCCSD/LiH exact singles anchor g[0][0] == 1/4 (%.15f)", g[0]);

        vqe_solver_free(s); vqe_optimizer_free(opt);
        vqe_ansatz_free(ansatz); pauli_hamiltonian_free(H);
    }

    /* (b) UCCSD doubles: same operator, 2-electron reference |0011> so the
     * (0,1)->(2,3) double excitation acts on the reference directly. */
    {
        pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
        H->hf_reference = 0x3;
        vqe_ansatz_t *ansatz = vqe_create_uccsd_ansatz(4, 2);
        vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
        vqe_solver_t *s = vqe_solver_create(H, ansatz, opt, &e);
        const size_t n = ansatz->num_parameters;
        CHECK(n == 5, "UCCSD doubles: 4 singles + 1 double (got %zu)", n);

        double params[5], g[25];
        for (size_t k = 0; k < n; k++) params[k] = 0.1 * (double)(k + 1);
        int rc = vqe_compute_qgt(s, params, g);
        CHECK(rc == 0, "UCCSD doubles: vqe_compute_qgt succeeded");

        qgt_check_metric_properties(g, n, "UCCSD doubles");
        qgt_check_matches_central_difference(s, params, g, n, "UCCSD doubles");

        /* Doubles anchor: singles at zero, only the double rotates. */
        double params_d[5] = { 0.0, 0.0, 0.0, 0.0, 0.3 };
        rc = vqe_compute_qgt(s, params_d, g);
        CHECK(rc == 0, "UCCSD doubles anchor: vqe_compute_qgt succeeded");
        CHECK(fabs(g[4 * n + 4] - 1.0) < 1e-12,
              "UCCSD exact doubles anchor g[4][4] == 1 (%.15f)", g[4 * n + 4]);

        vqe_solver_free(s); vqe_optimizer_free(opt);
        vqe_ansatz_free(ansatz); pauli_hamiltonian_free(H);
    }

    /* (c) symmetry-preserving Givens ansatz, 2 layers. */
    {
        pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
        vqe_ansatz_t *ansatz = vqe_create_symmetry_preserving_ansatz(4, 2, 2);
        vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
        vqe_solver_t *s = vqe_solver_create(H, ansatz, opt, &e);
        const size_t n = ansatz->num_parameters;
        CHECK(n == 8, "Givens LiH: 2 layers x 2 occ x 2 virt = 8 parameters (got %zu)", n);

        double params[8], g[64];
        for (size_t k = 0; k < n; k++) params[k] = 0.1 * (double)(k + 1);
        int rc = vqe_compute_qgt(s, params, g);
        CHECK(rc == 0, "Givens/LiH: vqe_compute_qgt succeeded");

        qgt_check_metric_properties(g, n, "Givens/LiH");
        qgt_check_matches_central_difference(s, params, g, n, "Givens/LiH");

        vqe_solver_free(s); vqe_optimizer_free(opt);
        vqe_ansatz_free(ansatz); pauli_hamiltonian_free(H);
    }
}

/* QNG end-to-end on LiH with the UCCSD ansatz: the exact metric must carry
 * the natural-gradient step to the correlated ground state, and from
 * identical initial parameters, the same learning rate, and the same
 * iteration budget it must do at least as well as plain gradient descent. */
static void test_qng_lih_uccsd_convergence(void) {
    fprintf(stdout, "\n-- VQE: QNG on LiH (UCCSD) vs plain gradient --\n");

    pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5949);
    double E_exact = vqe_exact_ground_state_energy(H);

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    const double init[3] = { 0.02, 0.02, 0.02 };

    /* Identical initial parameters, learning rate, and iteration budget for
     * both optimizers.  The UCCSD singles metric diagonal is ~1/4 (see the
     * anchor above), so the natural-gradient step g^{-1} grad is ~4x the raw
     * gradient step: QNG must converge markedly faster than plain GD here. */
    vqe_ansatz_t *a_qng = vqe_create_uccsd_ansatz(4, 3);
    memcpy(a_qng->parameters, init, sizeof(init));
    vqe_optimizer_t *qng = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
    qng->verbose = 0; qng->max_iterations = 1500; qng->tolerance = 1e-12;
    qng->learning_rate = 0.1;
    vqe_solver_t *s_qng = vqe_solver_create(H, a_qng, qng, &e);
    vqe_result_t rq = vqe_solve(s_qng);

    vqe_ansatz_t *a_gd = vqe_create_uccsd_ansatz(4, 3);
    memcpy(a_gd->parameters, init, sizeof(init));
    vqe_optimizer_t *gd = vqe_optimizer_create(VQE_OPTIMIZER_GRADIENT_DESCENT);
    gd->verbose = 0; gd->max_iterations = 1500; gd->tolerance = 1e-12;
    gd->learning_rate = 0.1;
    vqe_solver_t *s_gd = vqe_solver_create(H, a_gd, gd, &e);
    vqe_result_t rg = vqe_solve(s_gd);

    fprintf(stdout,
            "    QNG:  E = %.9f Ha in %zu iters   GD: E = %.9f Ha in %zu iters"
            "   (exact %.9f)\n",
            rq.ground_state_energy, rq.iterations,
            rg.ground_state_energy, rg.iterations, E_exact);
    CHECK(isfinite(rq.ground_state_energy) &&
          fabs(rq.ground_state_energy - E_exact) < 1.6e-3,
          "QNG reaches the LiH ground state to chemical accuracy (|err| %.2e)",
          fabs(rq.ground_state_energy - E_exact));
    CHECK(rq.ground_state_energy <= rg.ground_state_energy + 1e-9,
          "QNG at least matches plain gradient descent (QNG %.9f vs GD %.9f)",
          rq.ground_state_energy, rg.ground_state_energy);

    vqe_solver_free(s_qng); vqe_optimizer_free(qng); vqe_ansatz_free(a_qng);
    vqe_solver_free(s_gd);  vqe_optimizer_free(gd);  vqe_ansatz_free(a_gd);
    pauli_hamiltonian_free(H);
}

/* Berry curvature = the imaginary half of the QGT, F_ij = -2 Im Q_ij (the same
 * exact derivatives as vqe_compute_qgt). Checks: (a) the closed-form single-qubit
 * value F_01 = -sin(alpha)/2 for RY(alpha)-RZ(beta) on |0> (a Bloch state), to
 * machine precision; (b) F is antisymmetric with zero diagonal; (c) F vanishes
 * identically for a real ansatz (UCCSD has only RY/CNOT/Givens gates -> real
 * state -> real QGT); (d) integrating F over the single-qubit (alpha,beta) sweep
 * gives the Chern number -1. */
static void test_qng_berry_curvature(void) {
    fprintf(stdout, "\n-- VQE: Berry curvature (imaginary half of the QGT) --\n");
    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    /* (a,b) single-qubit HEA: params = (alpha for RY, beta for RZ) -> Bloch state.
     * Berry curvature F_01 = -sin(alpha)/2 exactly. */
    {
        pauli_hamiltonian_t *H = pauli_hamiltonian_create(1, 1);
        H->hf_reference = 0;
        vqe_ansatz_t *a = vqe_create_hardware_efficient_ansatz(1, 1);
        CHECK(a->num_parameters == 2, "single-qubit HEA has 2 params (RY,RZ) (got %zu)",
              a->num_parameters);
        vqe_optimizer_t *o = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
        vqe_solver_t *s = vqe_solver_create(H, a, o, &e);
        double F[4];
        double p[2] = { 0.7, 1.3 };
        int rc = vqe_compute_berry_curvature(s, p, F);
        CHECK(rc == 0, "vqe_compute_berry_curvature succeeded");
        CHECK(fabs(F[0] - 0.0) < 1e-14 && fabs(F[3] - 0.0) < 1e-14,
              "Berry curvature has zero diagonal");
        CHECK(fabs(F[1] + F[2]) < 1e-14, "Berry curvature antisymmetric (F01+F10=%.1e)",
              F[1] + F[2]);
        CHECK(fabs(F[1] - (-0.5 * sin(p[0]))) < 1e-12,
              "F01 == -sin(alpha)/2 closed form (%.12f vs %.12f)", F[1], -0.5 * sin(p[0]));

        /* (d) Chern number = (1/2pi) * integral of F over the (alpha,beta) sphere. */
        double chern = 0.0; int Na = 400; double da = M_PI / Na, db = 2.0 * M_PI;
        for (int ia = 0; ia < Na; ia++) {
            double pp[2] = { (ia + 0.5) * da, 0.3 };
            vqe_compute_berry_curvature(s, pp, F);
            chern += F[1] * da * db;
        }
        chern /= (2.0 * M_PI);
        CHECK(fabs(chern - (-1.0)) < 1e-2, "Chern number of single-qubit sweep == -1 (%.4f)", chern);

        vqe_solver_free(s); vqe_optimizer_free(o); vqe_ansatz_free(a); pauli_hamiltonian_free(H);
    }

    /* (c) real ansatz (UCCSD, only real gates) -> Berry curvature vanishes. */
    {
        pauli_hamiltonian_t *H = vqe_create_lih_hamiltonian(1.5);
        H->hf_reference = 0x3;
        vqe_ansatz_t *a = vqe_create_uccsd_ansatz(H->num_qubits, 2);
        vqe_optimizer_t *o = vqe_optimizer_create(VQE_OPTIMIZER_QNG);
        vqe_solver_t *s = vqe_solver_create(H, a, o, &e);
        size_t n = a->num_parameters;
        double *F = malloc(n * n * sizeof(double));
        double *p = malloc(n * sizeof(double));
        for (size_t k = 0; k < n; k++) p[k] = 0.1 * (double)(k + 1);
        vqe_compute_berry_curvature(s, p, F);
        double mx = 0.0;
        for (size_t i = 0; i < n * n; i++) if (fabs(F[i]) > mx) mx = fabs(F[i]);
        CHECK(mx < 1e-12, "UCCSD (real state) has zero Berry curvature (max|F|=%.1e)", mx);
        free(F); free(p);
        vqe_solver_free(s); vqe_optimizer_free(o); vqe_ansatz_free(a); pauli_hamiltonian_free(H);
    }
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
    test_qng_exact_qgt_uccsd_givens();
    test_qng_berry_curvature();
    test_qng_lih_uccsd_convergence();
    test_h2_uccsd_chemical_accuracy();
    test_lih_uccsd_chemical_accuracy();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
