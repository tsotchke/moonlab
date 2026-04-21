/**
 * @file differentiable.c
 * @brief Native reverse-mode autograd for parameterized circuits.
 *
 * See differentiable.h for the design.  The adjoint loop uses
 * two quantum_state_t buffers concurrently (forward + cotangent)
 * and unwinds the circuit in reverse, accumulating
 *   grad_k = Im(<eta_k | G_k | xi_k>)
 * where G_k is the Hermitian Pauli generator of the k-th
 * parametric rotation and |xi_k>, |eta_k> are the forward and
 * cotangent states mid-adjoint at step k.
 */

#include "differentiable.h"
#include "../../quantum/gates.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_H, OP_X, OP_Y, OP_Z,
    OP_CNOT, OP_CZ,
    OP_RX, OP_RY, OP_RZ
} op_kind_t;

typedef struct {
    op_kind_t kind;
    int q0;
    int q1;       /* control qubit for 2q gates, unused otherwise */
    double theta; /* angle for parametric ops */
    size_t param_index; /* slot in gradient output; SIZE_MAX for fixed */
} op_t;

struct moonlab_diff_circuit {
    uint32_t num_qubits;
    op_t   *ops;
    size_t  n_ops;
    size_t  cap_ops;
    size_t  n_params;
};

/* ------------------------------------------------------------- */
/* Circuit lifecycle                                              */
/* ------------------------------------------------------------- */

moonlab_diff_circuit_t* moonlab_diff_circuit_create(uint32_t num_qubits) {
    if (num_qubits == 0 || num_qubits > 63) return NULL;
    moonlab_diff_circuit_t *c =
        (moonlab_diff_circuit_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->num_qubits = num_qubits;
    c->cap_ops    = 32;
    c->ops        = (op_t*)calloc(c->cap_ops, sizeof(op_t));
    if (!c->ops) { free(c); return NULL; }
    return c;
}

void moonlab_diff_circuit_free(moonlab_diff_circuit_t *c) {
    if (!c) return;
    free(c->ops);
    free(c);
}

uint32_t moonlab_diff_num_qubits(const moonlab_diff_circuit_t *c) {
    return c ? c->num_qubits : 0;
}

size_t moonlab_diff_num_parameters(const moonlab_diff_circuit_t *c) {
    return c ? c->n_params : 0;
}

/* ------------------------------------------------------------- */
/* Append helpers                                                 */
/* ------------------------------------------------------------- */

static int ensure_cap(moonlab_diff_circuit_t *c) {
    if (c->n_ops < c->cap_ops) return 0;
    size_t new_cap = c->cap_ops * 2;
    op_t *grown = (op_t*)realloc(c->ops, new_cap * sizeof(op_t));
    if (!grown) return -1;
    c->ops     = grown;
    c->cap_ops = new_cap;
    return 0;
}

static int append_fixed(moonlab_diff_circuit_t *c, op_kind_t k,
                         int q0, int q1) {
    if (!c) return -1;
    if (q0 < 0 || (uint32_t)q0 >= c->num_qubits) return -2;
    if (k == OP_CNOT || k == OP_CZ) {
        if (q1 < 0 || (uint32_t)q1 >= c->num_qubits) return -2;
        if (q0 == q1) return -3;
    }
    if (ensure_cap(c) != 0) return -4;
    op_t *op = &c->ops[c->n_ops++];
    op->kind = k;
    op->q0   = q0;
    op->q1   = q1;
    op->theta = 0.0;
    op->param_index = (size_t)-1;
    return 0;
}

static int append_param(moonlab_diff_circuit_t *c, op_kind_t k,
                         int qubit, double theta) {
    if (!c) return -1;
    if (qubit < 0 || (uint32_t)qubit >= c->num_qubits) return -2;
    if (ensure_cap(c) != 0) return -4;
    op_t *op = &c->ops[c->n_ops++];
    op->kind = k;
    op->q0   = qubit;
    op->q1   = -1;
    op->theta = theta;
    op->param_index = c->n_params++;
    return 0;
}

int moonlab_diff_h (moonlab_diff_circuit_t *c, int q) { return append_fixed(c, OP_H, q, -1); }
int moonlab_diff_x (moonlab_diff_circuit_t *c, int q) { return append_fixed(c, OP_X, q, -1); }
int moonlab_diff_y (moonlab_diff_circuit_t *c, int q) { return append_fixed(c, OP_Y, q, -1); }
int moonlab_diff_z (moonlab_diff_circuit_t *c, int q) { return append_fixed(c, OP_Z, q, -1); }
int moonlab_diff_cnot(moonlab_diff_circuit_t *c, int ctrl, int tgt) {
    return append_fixed(c, OP_CNOT, ctrl, tgt);
}
int moonlab_diff_cz(moonlab_diff_circuit_t *c, int a, int b) {
    return append_fixed(c, OP_CZ, a, b);
}

int moonlab_diff_rx(moonlab_diff_circuit_t *c, int q, double t) { return append_param(c, OP_RX, q, t); }
int moonlab_diff_ry(moonlab_diff_circuit_t *c, int q, double t) { return append_param(c, OP_RY, q, t); }
int moonlab_diff_rz(moonlab_diff_circuit_t *c, int q, double t) { return append_param(c, OP_RZ, q, t); }

int moonlab_diff_set_theta(moonlab_diff_circuit_t *c, size_t k, double t) {
    if (!c) return -1;
    if (k >= c->n_params) return -2;
    for (size_t i = 0; i < c->n_ops; i++) {
        if (c->ops[i].param_index == k) {
            c->ops[i].theta = t;
            return 0;
        }
    }
    return -3;
}

/* ------------------------------------------------------------- */
/* Forward application                                            */
/* ------------------------------------------------------------- */

static qs_error_t apply_op(quantum_state_t *s, const op_t *op) {
    switch (op->kind) {
        case OP_H:     return gate_hadamard(s, op->q0);
        case OP_X:     return gate_pauli_x(s, op->q0);
        case OP_Y:     return gate_pauli_y(s, op->q0);
        case OP_Z:     return gate_pauli_z(s, op->q0);
        case OP_CNOT:  return gate_cnot(s, op->q0, op->q1);
        case OP_CZ:    return gate_cz(s, op->q0, op->q1);
        case OP_RX:    return gate_rx(s, op->q0, op->theta);
        case OP_RY:    return gate_ry(s, op->q0, op->theta);
        case OP_RZ:    return gate_rz(s, op->q0, op->theta);
    }
    return QS_ERROR_INVALID_STATE;
}

/* Apply the inverse of op by negating the angle for parametrics or
 * reapplying self-inverse fixed gates.  H, X, Y, Z, CNOT, CZ are all
 * involutions so U = U^-1; RX, RY, RZ invert by negating theta. */
static qs_error_t apply_op_inverse(quantum_state_t *s, const op_t *op) {
    op_t inv = *op;
    switch (op->kind) {
        case OP_RX:
        case OP_RY:
        case OP_RZ:
            inv.theta = -op->theta;
            break;
        default:
            /* Involution: U^-1 = U. */
            break;
    }
    return apply_op(s, &inv);
}

int moonlab_diff_forward(const moonlab_diff_circuit_t *c,
                          quantum_state_t *state) {
    if (!c || !state) return -1;
    if (state->num_qubits != c->num_qubits) return -2;
    /* Reset to |0...0>. */
    memset(state->amplitudes, 0,
           state->state_dim * sizeof(*state->amplitudes));
    state->amplitudes[0] = 1.0;
    for (size_t i = 0; i < c->n_ops; i++) {
        if (apply_op(state, &c->ops[i]) != QS_SUCCESS) return -3;
    }
    return 0;
}

/* ------------------------------------------------------------- */
/* Observables                                                    */
/* ------------------------------------------------------------- */

double moonlab_diff_expect_z(const quantum_state_t *s, int q) {
    if (!s || q < 0 || (uint32_t)q >= s->num_qubits) return 0.0;
    const uint64_t mask = (uint64_t)1 << q;
    double acc = 0.0;
    for (uint64_t i = 0; i < s->state_dim; i++) {
        const double re = creal(s->amplitudes[i]);
        const double im = cimag(s->amplitudes[i]);
        const double p  = re * re + im * im;
        acc += ((i & mask) ? -1.0 : +1.0) * p;
    }
    return acc;
}

double moonlab_diff_expect_x(const quantum_state_t *s, int q) {
    if (!s || q < 0 || (uint32_t)q >= s->num_qubits) return 0.0;
    const uint64_t mask = (uint64_t)1 << q;
    /* <psi|X_q|psi> = 2 * Re(sum_{i: bit_q(i)=0} conj(psi[i]) * psi[i | mask]). */
    double acc = 0.0;
    for (uint64_t i = 0; i < s->state_dim; i++) {
        if (i & mask) continue;
        const uint64_t j = i | mask;
        acc += 2.0 * creal(conj(s->amplitudes[i]) * s->amplitudes[j]);
    }
    return acc;
}

/* Apply the single-qubit Pauli @p obs to @p state in place. */
static qs_error_t apply_observable_in_place(quantum_state_t *s,
                                             moonlab_diff_observable_t obs,
                                             int qubit) {
    switch (obs) {
        case MOONLAB_DIFF_OBS_Z: return gate_pauli_z(s, qubit);
        case MOONLAB_DIFF_OBS_X: return gate_pauli_x(s, qubit);
        case MOONLAB_DIFF_OBS_Y: return gate_pauli_y(s, qubit);
    }
    return QS_ERROR_INVALID_STATE;
}

/* ------------------------------------------------------------- */
/* Adjoint backward pass                                          */
/* ------------------------------------------------------------- */

/* Inner product <a|b>. */
static double complex inner_product(const quantum_state_t *a,
                                     const quantum_state_t *b) {
    double complex acc = 0.0;
    const uint64_t n = a->state_dim;
    for (uint64_t i = 0; i < n; i++) {
        acc += conj(a->amplitudes[i]) * b->amplitudes[i];
    }
    return acc;
}

/* Apply Hermitian Pauli generator G in place to a state copy and
 * return <eta|G|xi> in one combined pass.  Reuses a scratch state
 * to avoid an extra alloc per step. */
static double complex eta_G_xi(const quantum_state_t *eta,
                                const quantum_state_t *xi,
                                op_kind_t gate_kind,
                                int qubit,
                                quantum_state_t *scratch) {
    /* Copy xi into scratch, apply G (= X / Y / Z for RX / RY / RZ). */
    memcpy(scratch->amplitudes, xi->amplitudes,
           xi->state_dim * sizeof(*xi->amplitudes));
    switch (gate_kind) {
        case OP_RX: gate_pauli_x(scratch, qubit); break;
        case OP_RY: gate_pauli_y(scratch, qubit); break;
        case OP_RZ: gate_pauli_z(scratch, qubit); break;
        default: return 0.0;
    }
    return inner_product(eta, scratch);
}

int moonlab_diff_backward(const moonlab_diff_circuit_t *c,
                           const quantum_state_t *forward_state,
                           moonlab_diff_observable_t obs,
                           int obs_qubit,
                           double *grad_out) {
    if (!c || !forward_state || !grad_out) return -1;
    if (forward_state->num_qubits != c->num_qubits) return -2;
    if (obs_qubit < 0 || (uint32_t)obs_qubit >= c->num_qubits) return -3;

    for (size_t i = 0; i < c->n_params; i++) grad_out[i] = 0.0;

    /* xi = forward state (clone so we can mutate). */
    quantum_state_t xi;
    if (quantum_state_clone(&xi, forward_state) != QS_SUCCESS) return -4;
    /* eta = O |xi>.  Clone then apply Pauli. */
    quantum_state_t eta;
    if (quantum_state_clone(&eta, forward_state) != QS_SUCCESS) {
        quantum_state_free(&xi); return -5;
    }
    if (apply_observable_in_place(&eta, obs, obs_qubit) != QS_SUCCESS) {
        quantum_state_free(&xi); quantum_state_free(&eta); return -6;
    }
    /* Scratch buffer for G|xi> computation. */
    quantum_state_t scratch;
    if (quantum_state_clone(&scratch, forward_state) != QS_SUCCESS) {
        quantum_state_free(&xi); quantum_state_free(&eta); return -7;
    }

    /* Walk ops in reverse.  At step k (original index i = k), both
     * xi and eta currently represent the state "after U_k", i.e.
     * U_k applied.  For parametric U_k, the gradient is
     *     grad[param_index] = Im(<eta | G_k | xi>).
     * Then undo U_k on both xi and eta. */
    for (size_t idx = c->n_ops; idx > 0; idx--) {
        const op_t *op = &c->ops[idx - 1];
        if (op->kind == OP_RX || op->kind == OP_RY || op->kind == OP_RZ) {
            double complex v = eta_G_xi(&eta, &xi, op->kind, op->q0, &scratch);
            /* U = exp(-i theta G/2) with Hermitian G.  See
             * differentiable.h: grad = Im(<eta|G|xi>). */
            grad_out[op->param_index] = cimag(v);
        }
        /* Undo the gate on both states. */
        if (apply_op_inverse(&xi,  op) != QS_SUCCESS) {
            quantum_state_free(&xi);
            quantum_state_free(&eta);
            quantum_state_free(&scratch);
            return -8;
        }
        if (apply_op_inverse(&eta, op) != QS_SUCCESS) {
            quantum_state_free(&xi);
            quantum_state_free(&eta);
            quantum_state_free(&scratch);
            return -9;
        }
    }

    quantum_state_free(&xi);
    quantum_state_free(&eta);
    quantum_state_free(&scratch);
    return 0;
}
