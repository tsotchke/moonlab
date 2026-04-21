/**
 * @file differentiable.h
 * @brief Reverse-mode automatic differentiation for parameterized
 *        quantum circuits (P5.19).
 *
 * Moonlab ships a native autograd tape rather than a PyTorch
 * adapter.  The core primitive is an adjoint-method gradient for
 * @f$\langle \psi(\vec\theta) | \hat O | \psi(\vec\theta) \rangle@f$
 * with respect to every parametric angle in a recorded circuit.
 * This is the workhorse for VQE / QAOA / variational quantum
 * eigensolver-style optimisation, without a Python dependency.
 *
 * DESIGN
 * ------
 * A @c moonlab_diff_circuit_t is a record of gate applications (mix
 * of parametric and fixed).  After the caller builds the circuit:
 *
 *   1. @c moonlab_diff_forward evolves @f$|0\ldots 0\rangle@f$
 *      through the circuit into a supplied @c quantum_state_t.
 *   2. @c moonlab_diff_expect_z computes
 *      @f$\langle \psi | Z_q | \psi \rangle@f$ on a single qubit.
 *   3. @c moonlab_diff_backward runs the adjoint loop: for each
 *      parameter @f$\theta_k@f$ in reverse order, undoes the gate
 *      on a working copy of the forward state AND on a cotangent
 *      state @f$|\eta\rangle = U_N^\dagger \ldots U_{k+1}^\dagger \hat O |\psi\rangle@f$,
 *      and accumulates @c grad[k] = @f$\operatorname{Im}\langle\eta|G_k|\xi_k\rangle@f$
 *      where @f$G_k@f$ is the Hermitian generator of the
 *      @f$k@f$-th parametric gate
 *      (@f$U_k = e^{-i\theta_k G_k/2}@f$).
 *
 * This is the same algorithm used by PennyLane, Qiskit Aer, and
 * JAX-Qsim for exact (non-stochastic) quantum gradients.  Total cost
 * is roughly two forward passes: one for @c moonlab_diff_forward and
 * one for the adjoint rewind, regardless of the number of
 * parameters @f$N_\theta@f$.  Parameter-shift-rule gradients would
 * cost @f$2 N_\theta@f$ forward passes.
 *
 * SUPPORTED GATES
 * ---------------
 *  Parametric (contribute one gradient entry each):
 *    - @c moonlab_diff_rx : @f$e^{-i\theta X/2}@f$, generator X.
 *    - @c moonlab_diff_ry : @f$e^{-i\theta Y/2}@f$, generator Y.
 *    - @c moonlab_diff_rz : @f$e^{-i\theta Z/2}@f$, generator Z.
 *
 *  Fixed (no gradient entry, unitary):
 *    - @c moonlab_diff_h        : Hadamard.
 *    - @c moonlab_diff_x / _y / _z : Paulis.
 *    - @c moonlab_diff_cnot, @c moonlab_diff_cz.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_DIFFERENTIABLE_H
#define MOONLAB_DIFFERENTIABLE_H

#include "../../quantum/state.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque circuit record. */
typedef struct moonlab_diff_circuit moonlab_diff_circuit_t;

/**
 * @brief Build an empty circuit on @p num_qubits qubits.
 */
moonlab_diff_circuit_t* moonlab_diff_circuit_create(uint32_t num_qubits);

/**
 * @brief Release the circuit record.  NULL-safe.
 */
void moonlab_diff_circuit_free(moonlab_diff_circuit_t *c);

/**
 * @brief Number of qubits the circuit acts on.
 */
uint32_t moonlab_diff_num_qubits(const moonlab_diff_circuit_t *c);

/**
 * @brief Number of parametric gates currently recorded.  Size of
 *        the gradient vector returned by @c moonlab_diff_backward.
 */
size_t moonlab_diff_num_parameters(const moonlab_diff_circuit_t *c);

/* -------- Non-parametric gates ------------------------------- */

int moonlab_diff_h (moonlab_diff_circuit_t *c, int qubit);
int moonlab_diff_x (moonlab_diff_circuit_t *c, int qubit);
int moonlab_diff_y (moonlab_diff_circuit_t *c, int qubit);
int moonlab_diff_z (moonlab_diff_circuit_t *c, int qubit);
int moonlab_diff_cnot(moonlab_diff_circuit_t *c, int ctrl, int target);
int moonlab_diff_cz  (moonlab_diff_circuit_t *c, int q0,   int q1);

/* -------- Parametric gates ----------------------------------- */

/**
 * @brief Record a parametric RX / RY / RZ gate.  The gradient with
 *        respect to @p theta will be accumulated into the
 *        corresponding slot of the @c moonlab_diff_backward output.
 *
 * Returns 0 on success.
 */
int moonlab_diff_rx(moonlab_diff_circuit_t *c, int qubit, double theta);
int moonlab_diff_ry(moonlab_diff_circuit_t *c, int qubit, double theta);
int moonlab_diff_rz(moonlab_diff_circuit_t *c, int qubit, double theta);

/**
 * @brief Update the @p k-th parametric angle in place (useful when
 *        an optimiser updates parameters across iterations without
 *        rebuilding the circuit).
 */
int moonlab_diff_set_theta(moonlab_diff_circuit_t *c,
                            size_t k, double theta);

/* -------- Forward + expectation ------------------------------ */

/**
 * @brief Evolve @c state from @f$|0\ldots 0\rangle@f$ through the
 *        circuit.  @p state must be initialised to the correct
 *        num_qubits via @c quantum_state_init; this function resets
 *        it to @c |0..0> before applying gates.
 */
int moonlab_diff_forward(const moonlab_diff_circuit_t *c,
                          quantum_state_t *state);

/**
 * @brief Compute @f$\langle \psi | Z_q | \psi \rangle@f$ for a
 *        single qubit @p q.  No tape entry; pure read-only.
 */
double moonlab_diff_expect_z(const quantum_state_t *state, int qubit);

/**
 * @brief Compute @f$\langle \psi | X_q | \psi \rangle@f$.
 */
double moonlab_diff_expect_x(const quantum_state_t *state, int qubit);

/* -------- Adjoint gradient ----------------------------------- */

/**
 * @brief Observable selector for @c moonlab_diff_backward.  Exactly
 *        one single-qubit Pauli on one qubit; extended observables
 *        can be expressed as a linear combination with a small
 *        wrapper by the caller.
 */
typedef enum {
    MOONLAB_DIFF_OBS_Z = 0,
    MOONLAB_DIFF_OBS_X = 1,
    MOONLAB_DIFF_OBS_Y = 2,
} moonlab_diff_observable_t;

/**
 * @brief Compute gradient of @f$\langle O_{obs,qubit}\rangle@f$ with
 *        respect to every parametric angle.
 *
 * Expects @c moonlab_diff_forward has already populated @p state
 * (the function will clone it internally, so the caller's state is
 * not modified).  @p grad must point to at least
 * @c moonlab_diff_num_parameters(c) doubles.
 *
 * @return 0 on success, non-zero on invalid input or allocation
 *         failure.
 */
int moonlab_diff_backward(const moonlab_diff_circuit_t *c,
                           const quantum_state_t *forward_state,
                           moonlab_diff_observable_t obs,
                           int obs_qubit,
                           double *grad_out);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_DIFFERENTIABLE_H */
