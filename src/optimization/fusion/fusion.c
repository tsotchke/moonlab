/**
 * @file fusion.c
 * @brief Gate-fusion DAG implementation.
 */

#include "fusion.h"
#include "../../quantum/gates.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct fuse_circuit_t {
    size_t       n;
    fuse_gate_t* gates;
    size_t       len;
    size_t       cap;
};

/* ---- 2x2 matrix helpers ------------------------------------------------ */

typedef struct { complex_t m[2][2]; } mat2_t;

static mat2_t mat_identity(void) {
    mat2_t r = { { { 1.0, 0.0 }, { 0.0, 1.0 } } };
    return r;
}

static mat2_t mat_mul(const mat2_t* a, const mat2_t* b) {
    mat2_t r;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            r.m[i][j] = a->m[i][0] * b->m[0][j] + a->m[i][1] * b->m[1][j];
        }
    }
    return r;
}

static mat2_t mat_of_gate(const fuse_gate_t* g) {
    const double INV_SQRT2 = 0.7071067811865475244;
    mat2_t r;
    switch (g->kind) {
        case FUSE_GATE_H:
            r.m[0][0] = INV_SQRT2; r.m[0][1] = INV_SQRT2;
            r.m[1][0] = INV_SQRT2; r.m[1][1] = -INV_SQRT2;
            return r;
        case FUSE_GATE_X:
            r.m[0][0] = 0; r.m[0][1] = 1;
            r.m[1][0] = 1; r.m[1][1] = 0;
            return r;
        case FUSE_GATE_Y:
            r.m[0][0] = 0;   r.m[0][1] = -1.0 * _Complex_I;
            r.m[1][0] = _Complex_I; r.m[1][1] = 0;
            return r;
        case FUSE_GATE_Z:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = -1;
            return r;
        case FUSE_GATE_S:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = _Complex_I;
            return r;
        case FUSE_GATE_SDG:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = -1.0 * _Complex_I;
            return r;
        case FUSE_GATE_T:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = cos(M_PI_4) + _Complex_I * sin(M_PI_4);
            return r;
        case FUSE_GATE_TDG:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = cos(M_PI_4) - _Complex_I * sin(M_PI_4);
            return r;
        case FUSE_GATE_PHASE:
            r.m[0][0] = 1; r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = cos(g->p[0]) + _Complex_I * sin(g->p[0]);
            return r;
        case FUSE_GATE_RX: {
            double c = cos(g->p[0] / 2.0);
            double s = sin(g->p[0] / 2.0);
            r.m[0][0] = c;                         r.m[0][1] = -_Complex_I * s;
            r.m[1][0] = -_Complex_I * s;           r.m[1][1] = c;
            return r;
        }
        case FUSE_GATE_RY: {
            double c = cos(g->p[0] / 2.0);
            double s = sin(g->p[0] / 2.0);
            r.m[0][0] = c;  r.m[0][1] = -s;
            r.m[1][0] = s;  r.m[1][1] = c;
            return r;
        }
        case FUSE_GATE_RZ: {
            double a = g->p[0] / 2.0;
            r.m[0][0] = cos(a) - _Complex_I * sin(a); r.m[0][1] = 0;
            r.m[1][0] = 0; r.m[1][1] = cos(a) + _Complex_I * sin(a);
            return r;
        }
        case FUSE_GATE_U3: {
            double th = g->p[0], ph = g->p[1], la = g->p[2];
            double c = cos(th / 2.0);
            double s = sin(th / 2.0);
            complex_t eil  = cos(la) + _Complex_I * sin(la);
            complex_t eip  = cos(ph) + _Complex_I * sin(ph);
            complex_t eipl = cos(ph + la) + _Complex_I * sin(ph + la);
            r.m[0][0] = c;         r.m[0][1] = -eil * s;
            r.m[1][0] = eip * s;   r.m[1][1] = eipl * c;
            return r;
        }
        case FUSE_GATE_FUSED_1Q:
            memcpy(r.m, g->u, sizeof(r.m));
            return r;
        default:
            return mat_identity();
    }
}

static int is_single_qubit(fuse_gate_kind_t k) {
    switch (k) {
        case FUSE_GATE_H: case FUSE_GATE_X: case FUSE_GATE_Y:
        case FUSE_GATE_Z: case FUSE_GATE_S: case FUSE_GATE_SDG:
        case FUSE_GATE_T: case FUSE_GATE_TDG: case FUSE_GATE_PHASE:
        case FUSE_GATE_RX: case FUSE_GATE_RY: case FUSE_GATE_RZ:
        case FUSE_GATE_U3: case FUSE_GATE_FUSED_1Q:
            return 1;
        default:
            return 0;
    }
}

/* ---- circuit container ------------------------------------------------- */

fuse_circuit_t* fuse_circuit_create(size_t num_qubits) {
    if (num_qubits == 0 || num_qubits > MAX_QUBITS) return NULL;
    fuse_circuit_t* c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n = num_qubits;
    c->cap = 16;
    c->gates = calloc(c->cap, sizeof(fuse_gate_t));
    if (!c->gates) { free(c); return NULL; }
    return c;
}

void fuse_circuit_free(fuse_circuit_t* c) {
    if (!c) return;
    free(c->gates);
    free(c);
}

size_t fuse_circuit_len(const fuse_circuit_t* c) { return c ? c->len : 0; }
size_t fuse_circuit_num_qubits(const fuse_circuit_t* c) { return c ? c->n : 0; }

static int push_gate(fuse_circuit_t* c, const fuse_gate_t* g) {
    if (!c) return -1;
    if (c->len == c->cap) {
        size_t ncap = c->cap * 2;
        fuse_gate_t* p = realloc(c->gates, ncap * sizeof(fuse_gate_t));
        if (!p) return -1;
        c->gates = p;
        c->cap = ncap;
    }
    c->gates[c->len++] = *g;
    return 0;
}

#define VALIDATE1(q)  do { if ((q) < 0 || (size_t)(q) >= c->n) return -1; } while (0)
#define VALIDATE2(a, b) do {                                          \
    if ((a) < 0 || (size_t)(a) >= c->n) return -1;                    \
    if ((b) < 0 || (size_t)(b) >= c->n) return -1;                    \
    if ((a) == (b)) return -1;                                        \
} while (0)

static int append_1q(fuse_circuit_t* c, fuse_gate_kind_t k, int q, double p0) {
    VALIDATE1(q);
    fuse_gate_t g = { .kind = k, .q = { q, -1 }, .p = { p0, 0, 0 } };
    return push_gate(c, &g);
}
static int append_2q(fuse_circuit_t* c, fuse_gate_kind_t k,
                     int a, int b, double p0) {
    VALIDATE2(a, b);
    fuse_gate_t g = { .kind = k, .q = { a, b }, .p = { p0, 0, 0 } };
    return push_gate(c, &g);
}

int fuse_append_h(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_H,   q, 0); }
int fuse_append_x(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_X,   q, 0); }
int fuse_append_y(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_Y,   q, 0); }
int fuse_append_z(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_Z,   q, 0); }
int fuse_append_s(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_S,   q, 0); }
int fuse_append_sdg(fuse_circuit_t* c, int q) { return append_1q(c, FUSE_GATE_SDG, q, 0); }
int fuse_append_t(fuse_circuit_t* c, int q)   { return append_1q(c, FUSE_GATE_T,   q, 0); }
int fuse_append_tdg(fuse_circuit_t* c, int q) { return append_1q(c, FUSE_GATE_TDG, q, 0); }
int fuse_append_phase(fuse_circuit_t* c, int q, double t) { return append_1q(c, FUSE_GATE_PHASE, q, t); }
int fuse_append_rx(fuse_circuit_t* c, int q, double t) { return append_1q(c, FUSE_GATE_RX, q, t); }
int fuse_append_ry(fuse_circuit_t* c, int q, double t) { return append_1q(c, FUSE_GATE_RY, q, t); }
int fuse_append_rz(fuse_circuit_t* c, int q, double t) { return append_1q(c, FUSE_GATE_RZ, q, t); }

int fuse_append_u3(fuse_circuit_t* c, int q,
                   double theta, double phi, double lambda) {
    VALIDATE1(q);
    fuse_gate_t g = {
        .kind = FUSE_GATE_U3, .q = { q, -1 },
        .p = { theta, phi, lambda }
    };
    return push_gate(c, &g);
}

int fuse_append_cnot(fuse_circuit_t* c, int a, int b)   { return append_2q(c, FUSE_GATE_CNOT,  a, b, 0); }
int fuse_append_cz(fuse_circuit_t* c, int a, int b)     { return append_2q(c, FUSE_GATE_CZ,    a, b, 0); }
int fuse_append_cy(fuse_circuit_t* c, int a, int b)     { return append_2q(c, FUSE_GATE_CY,    a, b, 0); }
int fuse_append_swap(fuse_circuit_t* c, int a, int b)   { return append_2q(c, FUSE_GATE_SWAP,  a, b, 0); }
int fuse_append_cphase(fuse_circuit_t* c, int a, int b, double t) { return append_2q(c, FUSE_GATE_CPHASE, a, b, t); }
int fuse_append_crx(fuse_circuit_t* c, int a, int b, double t) { return append_2q(c, FUSE_GATE_CRX, a, b, t); }
int fuse_append_cry(fuse_circuit_t* c, int a, int b, double t) { return append_2q(c, FUSE_GATE_CRY, a, b, t); }
int fuse_append_crz(fuse_circuit_t* c, int a, int b, double t) { return append_2q(c, FUSE_GATE_CRZ, a, b, t); }

/* ---- compilation ------------------------------------------------------- */

typedef struct {
    mat2_t m;
    int    has;
    int    count;   /* how many original gates folded into m */
} pending_t;

static int emit_pending(fuse_circuit_t* dst, pending_t* p, int qubit,
                        fuse_stats_t* stats) {
    if (!p->has) return 0;
    fuse_gate_t g = { 0 };
    if (p->count == 1) {
        /* A single-gate run — emit as FUSED_1Q too; the matrix-based
         * application cost is the same as a native native-gate dispatch. */
        g.kind = FUSE_GATE_FUSED_1Q;
        g.q[0] = qubit; g.q[1] = -1;
        memcpy(g.u, p->m.m, sizeof(g.u));
    } else {
        g.kind = FUSE_GATE_FUSED_1Q;
        g.q[0] = qubit; g.q[1] = -1;
        memcpy(g.u, p->m.m, sizeof(g.u));
        if (stats) stats->merges_applied += (size_t)(p->count - 1);
    }
    p->has = 0;
    p->count = 0;
    p->m = mat_identity();
    return push_gate(dst, &g);
}

fuse_circuit_t* fuse_compile(const fuse_circuit_t* src, fuse_stats_t* stats) {
    if (!src) return NULL;
    fuse_circuit_t* dst = fuse_circuit_create(src->n);
    if (!dst) return NULL;
    if (stats) {
        stats->original_gates = src->len;
        stats->fused_gates = 0;
        stats->merges_applied = 0;
    }

    pending_t* pend = calloc(src->n, sizeof(pending_t));
    if (!pend) { fuse_circuit_free(dst); return NULL; }
    for (size_t i = 0; i < src->n; i++) pend[i].m = mat_identity();

    for (size_t i = 0; i < src->len; i++) {
        const fuse_gate_t* g = &src->gates[i];
        if (is_single_qubit(g->kind)) {
            int q = g->q[0];
            mat2_t gm = mat_of_gate(g);
            /* new = gm * pending  (gm is applied AFTER pending) */
            pending_t* pq = &pend[q];
            pq->m = pq->has ? mat_mul(&gm, &pq->m) : gm;
            pq->has = 1;
            pq->count += 1;
        } else {
            /* multi-qubit: flush pending on every touched qubit */
            int a = g->q[0], b = g->q[1];
            if (emit_pending(dst, &pend[a], a, stats) != 0) goto oom;
            if (emit_pending(dst, &pend[b], b, stats) != 0) goto oom;
            if (push_gate(dst, g) != 0) goto oom;
        }
    }
    /* End of circuit — flush all remaining pending ops. */
    for (size_t q = 0; q < src->n; q++) {
        if (emit_pending(dst, &pend[q], (int)q, stats) != 0) goto oom;
    }

    free(pend);
    if (stats) stats->fused_gates = dst->len;
    return dst;

oom:
    free(pend);
    fuse_circuit_free(dst);
    return NULL;
}

/* ---- execution --------------------------------------------------------- */

qs_error_t fuse_execute(const fuse_circuit_t* c, quantum_state_t* state) {
    if (!c || !state) return QS_ERROR_INVALID_STATE;
    if (state->num_qubits < c->n) return QS_ERROR_INVALID_DIMENSION;

    for (size_t i = 0; i < c->len; i++) {
        const fuse_gate_t* g = &c->gates[i];
        qs_error_t rc = QS_SUCCESS;
        switch (g->kind) {
            case FUSE_GATE_H:   rc = gate_hadamard(state, g->q[0]); break;
            case FUSE_GATE_X:   rc = gate_pauli_x(state, g->q[0]); break;
            case FUSE_GATE_Y:   rc = gate_pauli_y(state, g->q[0]); break;
            case FUSE_GATE_Z:   rc = gate_pauli_z(state, g->q[0]); break;
            case FUSE_GATE_S:   rc = gate_s(state, g->q[0]); break;
            case FUSE_GATE_SDG: rc = gate_s_dagger(state, g->q[0]); break;
            case FUSE_GATE_T:   rc = gate_t(state, g->q[0]); break;
            case FUSE_GATE_TDG: rc = gate_t_dagger(state, g->q[0]); break;
            case FUSE_GATE_PHASE: rc = gate_phase(state, g->q[0], g->p[0]); break;
            case FUSE_GATE_RX:  rc = gate_rx(state, g->q[0], g->p[0]); break;
            case FUSE_GATE_RY:  rc = gate_ry(state, g->q[0], g->p[0]); break;
            case FUSE_GATE_RZ:  rc = gate_rz(state, g->q[0], g->p[0]); break;
            case FUSE_GATE_U3:  rc = gate_u3(state, g->q[0], g->p[0], g->p[1], g->p[2]); break;
            case FUSE_GATE_FUSED_1Q: {
                complex_t m[2][2];
                memcpy(m, g->u, sizeof(m));
                rc = apply_single_qubit_gate(state, g->q[0], m);
                break;
            }
            case FUSE_GATE_CNOT:   rc = gate_cnot(state, g->q[0], g->q[1]); break;
            case FUSE_GATE_CZ:     rc = gate_cz(state, g->q[0], g->q[1]); break;
            case FUSE_GATE_CY:     rc = gate_cy(state, g->q[0], g->q[1]); break;
            case FUSE_GATE_SWAP:   rc = gate_swap(state, g->q[0], g->q[1]); break;
            case FUSE_GATE_CPHASE: rc = gate_cphase(state, g->q[0], g->q[1], g->p[0]); break;
            case FUSE_GATE_CRX:    rc = gate_crx(state, g->q[0], g->q[1], g->p[0]); break;
            case FUSE_GATE_CRY:    rc = gate_cry(state, g->q[0], g->q[1], g->p[0]); break;
            case FUSE_GATE_CRZ:    rc = gate_crz(state, g->q[0], g->q[1], g->p[0]); break;
            default: return QS_ERROR_INVALID_STATE;
        }
        if (rc != QS_SUCCESS) return rc;
    }
    return QS_SUCCESS;
}
