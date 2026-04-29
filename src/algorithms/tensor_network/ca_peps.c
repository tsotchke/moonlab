/**
 * @file ca_peps.c
 * @brief Clifford-Assisted PEPS scaffold -- not yet implemented.
 *
 * Every public entry point currently returns
 * @c CA_PEPS_ERR_NOT_IMPLEMENTED or NULL.  This is the
 * scaffolding commit so downstream consumers (notably QGTL) can
 * plan against the API surface and tooling can compile.  The full
 * implementation requires:
 *
 *   1. A @c tn_peps_state_t type (plain PEPS) -- doesn't exist yet
 *      in Moonlab.  Needs site tensors with 4 bond indices + physical
 *      index, and per-bond chi caps.
 *   2. Split-CTMRG environment-tensor contraction for measurements
 *      and bond-dim truncation after gate application.
 *   3. PEPS-specific SVD compression after each non-Clifford
 *      Pauli rotation.
 *   4. The CA-PEPS wrapper itself: tableau D plus PEPS factor
 *      |phi>, with Clifford gates on D and Pauli rotations on PEPS
 *      following the §2.2 conjugation rules from the CA-MPS design.
 *
 * Estimated 2 weeks for the full implementation.  The scaffold in
 * this file makes the public API *callable* (no link-time errors)
 * while signalling NOT_IMPLEMENTED at runtime so consumers don't
 * silently get wrong results.
 */

#include "ca_peps.h"

#include <stdlib.h>

/* Internal struct -- intentionally minimal until the real PEPS
 * machinery lands.  Just enough to support clone/free without UB. */
struct moonlab_ca_peps_t {
    uint32_t Lx;
    uint32_t Ly;
    uint32_t chi_bond;
};

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

moonlab_ca_peps_t* moonlab_ca_peps_create(uint32_t Lx, uint32_t Ly,
                                            uint32_t chi_bond) {
    if (Lx == 0 || Ly == 0 || chi_bond == 0) return NULL;
    moonlab_ca_peps_t* s = (moonlab_ca_peps_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->Lx = Lx; s->Ly = Ly; s->chi_bond = chi_bond;
    return s;
}

void moonlab_ca_peps_free(moonlab_ca_peps_t* s) { free(s); }

moonlab_ca_peps_t* moonlab_ca_peps_clone(const moonlab_ca_peps_t* s) {
    if (!s) return NULL;
    return moonlab_ca_peps_create(s->Lx, s->Ly, s->chi_bond);
}

uint32_t moonlab_ca_peps_lx(const moonlab_ca_peps_t* s) {
    return s ? s->Lx : 0;
}
uint32_t moonlab_ca_peps_ly(const moonlab_ca_peps_t* s) {
    return s ? s->Ly : 0;
}
uint32_t moonlab_ca_peps_num_qubits(const moonlab_ca_peps_t* s) {
    return s ? s->Lx * s->Ly : 0;
}
uint32_t moonlab_ca_peps_max_bond_dim(const moonlab_ca_peps_t* s) {
    return s ? s->chi_bond : 0;
}

/* ------------------------------------------------------------------ */
/*  Stub gate / measurement entry points.  Returning NOT_IMPLEMENTED   */
/*  so callers fail loudly rather than silently producing the wrong   */
/*  state.                                                             */
/* ------------------------------------------------------------------ */

#define STUB1Q(name) \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, uint32_t q) { \
        (void)s; (void)q; return CA_PEPS_ERR_NOT_IMPLEMENTED; \
    }
#define STUB2Q(name) \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, \
                                            uint32_t a, uint32_t b) { \
        (void)s; (void)a; (void)b; return CA_PEPS_ERR_NOT_IMPLEMENTED; \
    }
#define STUB1Q_THETA(name) \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, \
                                            uint32_t q, double theta) { \
        (void)s; (void)q; (void)theta; return CA_PEPS_ERR_NOT_IMPLEMENTED; \
    }

STUB1Q(h)
STUB1Q(s)
STUB1Q(sdag)
STUB1Q(x)
STUB1Q(y)
STUB1Q(z)
STUB2Q(cnot)
STUB2Q(cz)
STUB1Q_THETA(rx)
STUB1Q_THETA(ry)
STUB1Q_THETA(rz)

ca_peps_error_t moonlab_ca_peps_expect_pauli(const moonlab_ca_peps_t* s,
                                              const uint8_t* pauli,
                                              double _Complex* out_expval) {
    (void)s; (void)pauli; (void)out_expval;
    return CA_PEPS_ERR_NOT_IMPLEMENTED;
}
