/**
 * @file vqe_qng.c
 * @brief Quantum Natural Gradient for VQE: the Fubini-Study metric of the
 *        ansatz state, used to precondition the gradient.
 *
 * The existing VQE optimizers (COBYLA / L-BFGS / ADAM / gradient descent) step
 * on the raw parameter-shift gradient.  Quantum Natural Gradient (Stokes et al.,
 * Quantum 4, 269 (2020)) instead steps along
 *
 *     theta <- theta - lr * (g + eps*I)^{-1} grad,
 *
 * where g is the quantum geometric (Fubini-Study) tensor of the ansatz state,
 *
 *     g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> ].
 *
 * g is the natural Riemannian metric on the state manifold; preconditioning by
 * it makes the step invariant to the parameterization and typically converges
 * in far fewer iterations, especially where the metric is ill-conditioned.
 *
 * The state derivatives d_i|psi> are obtained by central differences on the
 * ideal (noise-free) statevector produced by vqe_apply_ansatz -- no new
 * autodiff machinery is required, only the existing forward pass.
 */
#include "vqe.h"
#include "../quantum/state.h"
#include "../quantum/gates.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Exact analytic QGT state derivatives for the UCCSD / symmetry-preserving
 * ansaetze — defined in vqe.c (they need its private circuit-data structs). */
int qng_uccsd_state_derivatives(vqe_solver_t *solver, const double *params,
                                complex_t *dpsi, size_t dim);
int qng_symmetry_state_derivatives(vqe_solver_t *solver, const double *params,
                                   complex_t *dpsi, size_t dim);

/* Prepare the ideal trial state |psi(params)> = U(params)|HF>, mirroring the
 * state preparation in vqe_compute_energy (HF reference then ansatz, no noise).
 * Leaves solver->ansatz->parameters set to `params`. */
static int qng_prepare_state(vqe_solver_t *solver, const double *params,
                             quantum_state_t *psi) {
    if (quantum_state_init(psi, solver->hamiltonian->num_qubits) != QS_SUCCESS)
        return -1;
    uint64_t hf = solver->hamiltonian->hf_reference;
    for (size_t q = 0; q < solver->hamiltonian->num_qubits; q++)
        if (hf & (1ULL << q)) gate_pauli_x(psi, (int)q);
    memcpy(solver->ansatz->parameters, params,
           solver->ansatz->num_parameters * sizeof(double));
    if (vqe_apply_ansatz(psi, solver->ansatz) != QS_SUCCESS) {
        quantum_state_free(psi);
        return -1;
    }
    return 0;
}

/* Exact state derivatives d_i|psi> for the hardware-efficient ansatz, via
 * analytic gate differentiation instead of a finite-difference step.  A rotation
 * gate is exp(-i theta P/2) with generator P (Y for gate_ry, Z for gate_rz), so
 *   d|psi>/dtheta_i = (-i/2) U_after P_i U_upto |HF>,
 * i.e. replay the ansatz and insert the Pauli generator P_i immediately after the
 * gate driven by parameter i (P_i commutes with its own rotation).  Exact (no
 * delta) and cheaper than central differences: n circuit replays vs 2n.  Rows of
 * `dpsi` are d_i|psi>, i in [0, n).  Mirrors vqe_apply_hardware_efficient_ansatz
 * exactly (per-qubit RY,RZ then a linear CNOT chain, per layer). */
static int qng_hea_state_derivatives(vqe_solver_t *solver, const double *params,
                                     complex_t *dpsi, size_t dim) {
    const vqe_ansatz_t *ansatz = solver->ansatz;
    const size_t n  = ansatz->num_parameters;
    const size_t nq = ansatz->num_qubits;

    /* Hartree-Fock reference, copied as the start of every replay. */
    quantum_state_t hf;
    if (quantum_state_init(&hf, solver->hamiltonian->num_qubits) != QS_SUCCESS) return -1;
    uint64_t hfref = solver->hamiltonian->hf_reference;
    for (size_t q = 0; q < nq; q++)
        if (hfref & (1ULL << q)) gate_pauli_x(&hf, (int)q);

    int rc = 0;
    for (size_t i = 0; i < n; i++) {
        quantum_state_t st;
        if (quantum_state_init(&st, solver->hamiltonian->num_qubits) != QS_SUCCESS) { rc = -1; break; }
        memcpy(st.amplitudes, hf.amplitudes, dim * sizeof(complex_t));

        size_t p = 0;
        for (size_t layer = 0; layer < ansatz->num_layers; layer++) {
            for (size_t q = 0; q < nq; q++) {
                gate_ry(&st, (int)q, params[p]);
                if (p == i) gate_pauli_y(&st, (int)q);   /* generator of RY is Y */
                p++;
                gate_rz(&st, (int)q, params[p]);
                if (p == i) gate_pauli_z(&st, (int)q);   /* generator of RZ is Z */
                p++;
            }
            for (size_t q = 0; q + 1 < nq; q++) gate_cnot(&st, (int)q, (int)(q + 1));
        }
        /* apply the global (-i/2) factor: d_i|psi> = (-i/2) U_after P_i U_upto |HF> */
        for (size_t x = 0; x < dim; x++)
            dpsi[i * dim + x] = (-0.5 * I) * st.amplitudes[x];
        quantum_state_free(&st);
    }
    quantum_state_free(&hf);
    return rc;
}

/* Shared core for the geometric-tensor verbs.  Fills psi0[dim] = |psi(params)>,
 * dpsi[n*dim] (row i = d_i|psi>) and ov[n] = <psi|d_i psi>, using exact analytic
 * derivatives (generator insertion, n replays not 2n) for the hardware-efficient,
 * UCCSD and symmetry-preserving ansaetze, and a central-difference fallback for a
 * CUSTOM ansatz whose gate structure is unknown.  Restores the ansatz parameters
 * before returning.  Both vqe_compute_qgt (the metric, Re) and
 * vqe_compute_berry_curvature (the curvature, -2 Im) are assembled from these
 * three arrays: they are the two halves of the one Hermitian quantum geometric
 * tensor Q_ij = <d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>. */
static int qng_state_and_derivatives(vqe_solver_t *solver, const double *params,
                                     complex_t *psi0, complex_t *dpsi, complex_t *ov,
                                     size_t n, size_t dim) {
    const double delta = 1e-4;   /* central-difference step for the CUSTOM path */
    double *saved  = malloc(n * sizeof(double));
    double *pshift = malloc(n * sizeof(double));
    if (!saved || !pshift) { free(saved); free(pshift); return -1; }
    memcpy(saved, solver->ansatz->parameters, n * sizeof(double));

    int rc = 0;
    quantum_state_t st;

    if (qng_prepare_state(solver, params, &st) != 0) { rc = -1; goto done; }
    memcpy(psi0, st.amplitudes, dim * sizeof(complex_t));
    quantum_state_free(&st);

    if (solver->ansatz->type == VQE_ANSATZ_HARDWARE_EFFICIENT) {
        if (qng_hea_state_derivatives(solver, params, dpsi, dim) != 0) { rc = -1; goto done; }
    } else if (solver->ansatz->type == VQE_ANSATZ_UCCSD) {
        if (qng_uccsd_state_derivatives(solver, params, dpsi, dim) != 0) { rc = -1; goto done; }
    } else if (solver->ansatz->type == VQE_ANSATZ_SYMMETRY_PRESERVING) {
        if (qng_symmetry_state_derivatives(solver, params, dpsi, dim) != 0) { rc = -1; goto done; }
    } else {
        for (size_t i = 0; i < n; i++) {
            memcpy(pshift, params, n * sizeof(double));
            pshift[i] = params[i] + delta;
            if (qng_prepare_state(solver, pshift, &st) != 0) { rc = -1; goto done; }
            for (size_t x = 0; x < dim; x++) dpsi[i * dim + x] = st.amplitudes[x];
            quantum_state_free(&st);

            pshift[i] = params[i] - delta;
            if (qng_prepare_state(solver, pshift, &st) != 0) { rc = -1; goto done; }
            for (size_t x = 0; x < dim; x++)
                dpsi[i * dim + x] = (dpsi[i * dim + x] - st.amplitudes[x]) / (2.0 * delta);
            quantum_state_free(&st);
        }
    }

    for (size_t i = 0; i < n; i++) {
        complex_t s = 0.0;
        for (size_t x = 0; x < dim; x++) s += conj(psi0[x]) * dpsi[i * dim + x];
        ov[i] = s;
    }

done:
    memcpy(solver->ansatz->parameters, saved, n * sizeof(double));   /* restore */
    free(saved); free(pshift);
    return rc;
}

int vqe_compute_qgt(vqe_solver_t *solver, const double *params, double *qgt_out) {
    if (!solver || !params || !qgt_out) return -1;
    const size_t n   = solver->ansatz->num_parameters;
    const size_t dim = (size_t)1 << solver->hamiltonian->num_qubits;

    complex_t *psi0 = malloc(dim * sizeof(complex_t));
    complex_t *dpsi = malloc(n * dim * sizeof(complex_t));   /* row i = d_i psi */
    complex_t *ov   = malloc(n * sizeof(complex_t));         /* <psi | d_i psi> */
    if (!psi0 || !dpsi || !ov) { free(psi0); free(dpsi); free(ov); return -1; }

    int rc = qng_state_and_derivatives(solver, params, psi0, dpsi, ov, n, dim);
    if (rc == 0) {
        /* g_ij = Re[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>], symmetrized. */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) {
                complex_t a = 0.0;
                for (size_t x = 0; x < dim; x++)
                    a += conj(dpsi[i * dim + x]) * dpsi[j * dim + x];
                /* <d_i psi|psi> = conj(<psi|d_i psi>) = conj(ov[i]) */
                double g = creal(a) - creal(conj(ov[i]) * ov[j]);
                qgt_out[i * n + j] = g;
                qgt_out[j * n + i] = g;
            }
        }
    }
    free(psi0); free(dpsi); free(ov);
    return rc;
}

int vqe_compute_berry_curvature(vqe_solver_t *solver, const double *params,
                                double *berry_out) {
    if (!solver || !params || !berry_out) return -1;
    const size_t n   = solver->ansatz->num_parameters;
    const size_t dim = (size_t)1 << solver->hamiltonian->num_qubits;

    complex_t *psi0 = malloc(dim * sizeof(complex_t));
    complex_t *dpsi = malloc(n * dim * sizeof(complex_t));
    complex_t *ov   = malloc(n * sizeof(complex_t));
    if (!psi0 || !dpsi || !ov) { free(psi0); free(dpsi); free(ov); return -1; }

    int rc = qng_state_and_derivatives(solver, params, psi0, dpsi, ov, n, dim);
    if (rc == 0) {
        /* Berry curvature is the antisymmetric imaginary half of the QGT:
         *   F_ij = -2 Im[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>].
         * Q is Hermitian, so F is real with F_ii = 0 and F_ji = -F_ij.  The
         * -2 Im convention makes the surface flux of F integrate (via Stokes)
         * to the Berry phase around a closed parameter loop, and over a closed
         * parameter 2-manifold to 2*pi times the Chern number.  For a real
         * ansatz (only RY / CNOT / Givens / double-excitation gates) the state
         * is real, Q is real, and F vanishes identically. */
        for (size_t i = 0; i < n; i++) {
            berry_out[i * n + i] = 0.0;
            for (size_t j = i + 1; j < n; j++) {
                complex_t a = 0.0;
                for (size_t x = 0; x < dim; x++)
                    a += conj(dpsi[i * dim + x]) * dpsi[j * dim + x];
                double F = -2.0 * (cimag(a) - cimag(conj(ov[i]) * ov[j]));
                berry_out[i * n + j] =  F;
                berry_out[j * n + i] = -F;
            }
        }
    }
    free(psi0); free(dpsi); free(ov);
    return rc;
}

/* Solve the symmetric positive-definite system A x = b (A is n*n row-major,
 * overwritten) by Gaussian elimination with partial pivoting.  Returns 0 on
 * success, -1 if the system is singular. */
static int spd_solve(double *A, const double *b, double *x, size_t n) {
    double *w = malloc(n * sizeof(double));
    if (!w) return -1;
    memcpy(w, b, n * sizeof(double));
    for (size_t col = 0; col < n; col++) {
        size_t piv = col;
        double best = fabs(A[col * n + col]);
        for (size_t r = col + 1; r < n; r++) {
            double v = fabs(A[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) { free(w); return -1; }
        if (piv != col) {
            for (size_t c = 0; c < n; c++) {
                double t = A[col * n + c]; A[col * n + c] = A[piv * n + c]; A[piv * n + c] = t;
            }
            double t = w[col]; w[col] = w[piv]; w[piv] = t;
        }
        double diag = A[col * n + col];
        for (size_t r = col + 1; r < n; r++) {
            double f = A[r * n + col] / diag;
            if (f == 0.0) continue;
            for (size_t c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
            w[r] -= f * w[col];
        }
    }
    for (size_t ii = n; ii-- > 0; ) {
        double s = w[ii];
        for (size_t c = ii + 1; c < n; c++) s -= A[ii * n + c] * x[c];
        x[ii] = s / A[ii * n + ii];
    }
    free(w);
    return 0;
}

int vqe_natural_gradient_direction(vqe_solver_t *solver, const double *params,
                                   const double *gradient, double regularization,
                                   double *direction_out) {
    if (!solver || !params || !gradient || !direction_out) return -1;
    const size_t n = solver->ansatz->num_parameters;
    double *g = malloc(n * n * sizeof(double));
    if (!g) return -1;
    if (vqe_compute_qgt(solver, params, g) != 0) { free(g); return -1; }
    for (size_t i = 0; i < n; i++) g[i * n + i] += regularization;   /* Tikhonov */
    int rc = spd_solve(g, gradient, direction_out, n);
    free(g);
    return rc;   /* on failure the caller falls back to the plain gradient */
}
