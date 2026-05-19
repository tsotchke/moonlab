/**
 * @file    moonlab_qgtl_backend.c
 * @brief   Implementation of the QGTL-shaped circuit-ingestion surface.
 *
 * The circuit-record is just a growable array of `qgtl_gate_record_t`
 * tuples.  `execute` runs through them once, dispatches each to the
 * matching `gate_*` call against a freshly-initialised
 * `quantum_state_t`, then optionally extracts the probability
 * distribution + samples measurement outcomes.
 *
 * No optimisation, transpilation, or gate fusion -- QGTL handles that
 * upstream.  This TU is intentionally the thinnest possible bridge.
 */

#include "moonlab_qgtl_backend.h"

#include "../quantum/state.h"
#include "../quantum/gates.h"
#include "../quantum/measurement.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    moonlab_qgtl_gate_t type;
    int     target;
    int     control;
    double  theta;     /**< Captured at `add_gate` time so params[] can vanish. */
    int     has_param; /**< 1 if `theta` is meaningful. */
} qgtl_gate_record_t;

struct moonlab_qgtl_circuit {
    int                 num_qubits;
    int                 num_gates;
    int                 capacity;
    qgtl_gate_record_t *gates;
};

moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_create(int num_qubits)
{
    if (num_qubits < 1 || num_qubits > 32) return NULL;
    moonlab_qgtl_circuit_t *c = (moonlab_qgtl_circuit_t *)
        calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->num_qubits = num_qubits;
    return c;
}

void moonlab_qgtl_circuit_free(moonlab_qgtl_circuit_t *c)
{
    if (!c) return;
    free(c->gates);
    free(c);
}

int moonlab_qgtl_circuit_num_qubits(const moonlab_qgtl_circuit_t *c)
{
    return c ? c->num_qubits : MOONLAB_QGTL_BAD_ARG;
}

int moonlab_qgtl_circuit_num_gates(const moonlab_qgtl_circuit_t *c)
{
    return c ? c->num_gates : MOONLAB_QGTL_BAD_ARG;
}

static int gate_takes_param(moonlab_qgtl_gate_t t)
{
    return t == MOONLAB_QGTL_GATE_RX ||
           t == MOONLAB_QGTL_GATE_RY ||
           t == MOONLAB_QGTL_GATE_RZ;
}

static int gate_is_two_qubit(moonlab_qgtl_gate_t t)
{
    return t == MOONLAB_QGTL_GATE_CNOT ||
           t == MOONLAB_QGTL_GATE_CY   ||
           t == MOONLAB_QGTL_GATE_CZ   ||
           t == MOONLAB_QGTL_GATE_SWAP;
}

int moonlab_qgtl_add_gate(moonlab_qgtl_circuit_t *c,
                          moonlab_qgtl_gate_t    type,
                          int                    target,
                          int                    control,
                          const double          *params)
{
    if (!c) return MOONLAB_QGTL_BAD_ARG;
    if (target < 0 || target >= c->num_qubits) return MOONLAB_QGTL_BAD_ARG;
    if (gate_is_two_qubit(type)) {
        if (control < 0 || control >= c->num_qubits) return MOONLAB_QGTL_BAD_ARG;
        if (control == target) return MOONLAB_QGTL_BAD_ARG;
    }
    if (gate_takes_param(type) && params == NULL) return MOONLAB_QGTL_BAD_ARG;

    if (c->num_gates >= c->capacity) {
        const int new_cap = c->capacity ? c->capacity * 2 : 16;
        qgtl_gate_record_t *n = (qgtl_gate_record_t *)
            realloc(c->gates, (size_t)new_cap * sizeof(*n));
        if (!n) return MOONLAB_QGTL_OOM;
        c->gates    = n;
        c->capacity = new_cap;
    }

    qgtl_gate_record_t *g = &c->gates[c->num_gates++];
    g->type    = type;
    g->target  = target;
    g->control = control;
    g->theta     = gate_takes_param(type) ? params[0] : 0.0;
    g->has_param = gate_takes_param(type) ? 1 : 0;
    return MOONLAB_QGTL_OK;
}

static int apply_one_gate(quantum_state_t *s, const qgtl_gate_record_t *g)
{
    qs_error_t rc;
    switch (g->type) {
    case MOONLAB_QGTL_GATE_I:
        rc = 0; /* identity is a no-op */
        break;
    case MOONLAB_QGTL_GATE_X:    rc = gate_pauli_x(s, g->target); break;
    case MOONLAB_QGTL_GATE_Y:    rc = gate_pauli_y(s, g->target); break;
    case MOONLAB_QGTL_GATE_Z:    rc = gate_pauli_z(s, g->target); break;
    case MOONLAB_QGTL_GATE_H:    rc = gate_hadamard(s, g->target); break;
    case MOONLAB_QGTL_GATE_S:    rc = gate_s(s, g->target); break;
    case MOONLAB_QGTL_GATE_T:    rc = gate_t(s, g->target); break;
    case MOONLAB_QGTL_GATE_RX:   rc = gate_rx(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_RY:   rc = gate_ry(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_RZ:   rc = gate_rz(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_CNOT: rc = gate_cnot(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_CZ:   rc = gate_cz(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_SWAP: rc = gate_swap(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_CY:
        /* CY = (I tensor S^dagger) * CNOT * (I tensor S).  Build by composition
         * since moonlab doesn't ship a direct gate_cy. */
        rc = gate_s(s, g->target);
        if (rc == 0) rc = gate_cnot(s, g->control, g->target);
        if (rc == 0) rc = gate_s_dagger(s, g->target);
        break;
    default:
        return MOONLAB_QGTL_UNSUPPORTED;
    }
    return (rc == 0) ? MOONLAB_QGTL_OK : MOONLAB_QGTL_INTERNAL;
}

static uint64_t default_seed(void)
{
    /* Deterministic-from-clock seed; QGTL passes a real seed when it
     * needs reproducibility. */
    return (uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ULL;
}

/* xorshift64 -- one-line PRNG good enough for shot sampling. */
static inline uint64_t xorshift64_step(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

int moonlab_qgtl_execute(moonlab_qgtl_circuit_t           *c,
                         const moonlab_qgtl_exec_options_t *opts,
                         moonlab_qgtl_results_t           *out)
{
    if (!c || !opts || !out) return MOONLAB_QGTL_BAD_ARG;

    /* Initialise an N-qubit |0...0> state. */
    quantum_state_t state = {0};
    if (quantum_state_init(&state, (size_t)c->num_qubits) != 0) {
        return MOONLAB_QGTL_OOM;
    }

    /* Walk the gate list. */
    int rc = MOONLAB_QGTL_OK;
    for (int i = 0; i < c->num_gates; i++) {
        rc = apply_one_gate(&state, &c->gates[i]);
        if (rc != MOONLAB_QGTL_OK) {
            quantum_state_free(&state);
            return rc;
        }
    }

    /* Initialise output record. */
    memset(out, 0, sizeof(*out));
    out->num_qubits = c->num_qubits;
    out->num_shots  = opts->num_shots;

    const size_t dim = (size_t)1 << c->num_qubits;

    /* Optional: dump the probability vector. */
    if (opts->return_probabilities) {
        out->probabilities = (double *)malloc(dim * sizeof(double));
        if (!out->probabilities) {
            quantum_state_free(&state);
            return MOONLAB_QGTL_OOM;
        }
        for (size_t b = 0; b < dim; b++) {
            const double re = creal(state.amplitudes[b]);
            const double im = cimag(state.amplitudes[b]);
            out->probabilities[b] = re * re + im * im;
        }
    }

    /* Optional: shot sampling.  We need to preserve the post-circuit
     * state across shots, so clone-and-measure each shot.  At
     * num_shots > ~1000 the per-shot clone overhead dominates --
     * QGTL is expected to either keep shots small (it's a
     * cross-validation step, not a production run) or pre-flag
     * stim-style sampling, which can come in a later release. */
    if (opts->num_shots > 0) {
        out->outcomes = (uint64_t *)malloc((size_t)opts->num_shots * sizeof(uint64_t));
        if (!out->outcomes) {
            free(out->probabilities);
            out->probabilities = NULL;
            quantum_state_free(&state);
            return MOONLAB_QGTL_OOM;
        }
        uint64_t seed = opts->rng_seed != 0 ? opts->rng_seed : default_seed();
        for (int s = 0; s < opts->num_shots; s++) {
            quantum_state_t shot = {0};
            if (quantum_state_clone(&shot, &state) != 0) {
                free(out->outcomes);
                free(out->probabilities);
                out->outcomes = NULL;
                out->probabilities = NULL;
                quantum_state_free(&state);
                return MOONLAB_QGTL_OOM;
            }
            /* Draw a uniform random in [0, 1). */
            const uint64_t r = xorshift64_step(&seed);
            const double u = (double)(r >> 11) / (double)(1ULL << 53);
            out->outcomes[s] = measurement_all_qubits(&shot, u);
            quantum_state_free(&shot);
        }
    }

    quantum_state_free(&state);
    return MOONLAB_QGTL_OK;
}

void moonlab_qgtl_results_free(moonlab_qgtl_results_t *r)
{
    if (!r) return;
    free(r->outcomes);
    free(r->probabilities);
    r->outcomes      = NULL;
    r->probabilities = NULL;
}
