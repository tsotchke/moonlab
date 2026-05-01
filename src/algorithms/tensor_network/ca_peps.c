/**
 * @file ca_peps.c
 * @brief Clifford-Assisted PEPS (v0.2.1 implementation).
 *
 * The 2D extension promised in section 7 of docs/research/ca_mps.md.
 * The state lives on an Lx by Ly square lattice; sites are identified
 * by linear index q = x + Lx * y.  As of v0.2.1 the physical factor
 * |phi> is stored as a row-major MPS over Lx * Ly qubits via the
 * existing CA-MPS engine.  This is *not* a full PEPS factor (the real
 * 2D tensor with four bond indices per site is a v0.3 milestone), but
 * it gives correct numerical results for any circuit and lets the
 * public API surface be exercised end-to-end.
 *
 * Why this is useful even with a 1D back-end:
 *   - All Clifford gates (single + 2-qubit) update the tableau D only
 *     and are completely free in both 1D and 2D layouts -- this is the
 *     core CA-* advantage and it transfers to 2D unchanged.
 *   - Single-qubit non-Clifford rotations (RX/RY/RZ) push through D as
 *     a single-site Pauli rotation on |phi>; locality is irrelevant.
 *   - Pauli-string expectation values delegate to the existing CA-MPS
 *     contraction; correctness is independent of geometry.
 *
 * What you give up versus a full PEPS:
 *   - Bond growth scales like a snake-MPS over a 2D area, so the bond
 *     dimension cap may saturate quickly for circuits with non-Clifford
 *     gates dense across both directions.  Clifford-heavy circuits stay
 *     cheap because the tableau absorbs them.
 *
 * Two-qubit non-Clifford gates are not in the public API yet (only
 * Clifford 2-qubit + single-qubit rotations are exposed), so the row-
 * major embedding is sufficient to run correct circuits at the current
 * ABI surface.
 */

#include "ca_peps.h"
#include "ca_mps.h"

#include <stdlib.h>

struct moonlab_ca_peps_t {
    uint32_t Lx;
    uint32_t Ly;
    uint32_t chi_bond;
    moonlab_ca_mps_t* mps;  /* underlying CA-MPS over Lx * Ly qubits */
};

/* Map a CA-MPS error code into a CA-PEPS error code.  Both enums share
 * SUCCESS = 0 and use distinct negative tags otherwise. */
static ca_peps_error_t map_err(ca_mps_error_t e) {
    switch (e) {
    case CA_MPS_SUCCESS:    return CA_PEPS_SUCCESS;
    case CA_MPS_ERR_INVALID: return CA_PEPS_ERR_INVALID;
    case CA_MPS_ERR_QUBIT:   return CA_PEPS_ERR_QUBIT;
    case CA_MPS_ERR_OOM:     return CA_PEPS_ERR_OOM;
    case CA_MPS_ERR_BACKEND: return CA_PEPS_ERR_BACKEND;
    default:                 return CA_PEPS_ERR_BACKEND;
    }
}

/* Validate a single-site lattice coordinate.  Returns CA_PEPS_SUCCESS on
 * success; sets the index range error otherwise. */
static int site_in_range(const moonlab_ca_peps_t* s, uint32_t q) {
    return s != NULL && q < s->Lx * s->Ly;
}

/* Adjacency check on the square lattice with no PBC: two linear indices
 * are neighbours iff they differ by exactly 1 within the same row, or by
 * exactly Lx between rows. */
static int sites_adjacent(const moonlab_ca_peps_t* s, uint32_t a, uint32_t b) {
    if (!s) return 0;
    if (a == b) return 0;
    const uint32_t Lx = s->Lx;
    const uint32_t ax = a % Lx, ay = a / Lx;
    const uint32_t bx = b % Lx, by = b / Lx;
    if (ay == by && (ax + 1 == bx || bx + 1 == ax)) return 1;
    if (ax == bx && (ay + 1 == by || by + 1 == ay)) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

moonlab_ca_peps_t* moonlab_ca_peps_create(uint32_t Lx, uint32_t Ly,
                                            uint32_t chi_bond) {
    if (Lx == 0 || Ly == 0 || chi_bond == 0) return NULL;
    moonlab_ca_peps_t* s = (moonlab_ca_peps_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->Lx = Lx;
    s->Ly = Ly;
    s->chi_bond = chi_bond;
    s->mps = moonlab_ca_mps_create(Lx * Ly, chi_bond);
    if (!s->mps) { free(s); return NULL; }
    return s;
}

void moonlab_ca_peps_free(moonlab_ca_peps_t* s) {
    if (!s) return;
    if (s->mps) moonlab_ca_mps_free(s->mps);
    free(s);
}

moonlab_ca_peps_t* moonlab_ca_peps_clone(const moonlab_ca_peps_t* s) {
    if (!s) return NULL;
    moonlab_ca_peps_t* c = (moonlab_ca_peps_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->Lx = s->Lx;
    c->Ly = s->Ly;
    c->chi_bond = s->chi_bond;
    c->mps = moonlab_ca_mps_clone(s->mps);
    if (!c->mps) { free(c); return NULL; }
    return c;
}

uint32_t moonlab_ca_peps_lx(const moonlab_ca_peps_t* s) { return s ? s->Lx : 0; }
uint32_t moonlab_ca_peps_ly(const moonlab_ca_peps_t* s) { return s ? s->Ly : 0; }
uint32_t moonlab_ca_peps_num_qubits(const moonlab_ca_peps_t* s) {
    return s ? s->Lx * s->Ly : 0;
}
uint32_t moonlab_ca_peps_max_bond_dim(const moonlab_ca_peps_t* s) {
    return s ? s->chi_bond : 0;
}
uint32_t moonlab_ca_peps_current_bond_dim(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_current_bond_dim(s->mps) : 0;
}
double moonlab_ca_peps_max_half_cut_entropy(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_max_half_cut_entropy(s->mps) : 0.0;
}

/* ------------------------------------------------------------------ */
/*  Clifford gates -- tableau-only, geometry-free.  Delegate.          */
/* ------------------------------------------------------------------ */

#define DELEGATE_1Q(name) \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, uint32_t q) { \
        if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT; \
        return map_err(moonlab_ca_mps_##name(s->mps, q)); \
    }

DELEGATE_1Q(h)
DELEGATE_1Q(s)
DELEGATE_1Q(sdag)
DELEGATE_1Q(x)
DELEGATE_1Q(y)
DELEGATE_1Q(z)

#undef DELEGATE_1Q

ca_peps_error_t moonlab_ca_peps_cnot(moonlab_ca_peps_t* s,
                                      uint32_t c, uint32_t t) {
    if (!site_in_range(s, c) || !site_in_range(s, t)) return CA_PEPS_ERR_QUBIT;
    /* For Clifford gates, the tableau update is independent of lattice
     * geometry, but we still validate adjacency so callers get an honest
     * error if they treat distant sites as "neighbours". */
    if (!sites_adjacent(s, c, t)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_cnot(s->mps, c, t));
}

ca_peps_error_t moonlab_ca_peps_cz(moonlab_ca_peps_t* s,
                                    uint32_t a, uint32_t b) {
    if (!site_in_range(s, a) || !site_in_range(s, b)) return CA_PEPS_ERR_QUBIT;
    if (!sites_adjacent(s, a, b)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_cz(s->mps, a, b));
}

/* ------------------------------------------------------------------ */
/*  Non-Clifford single-qubit rotations.  Delegate.                   */
/* ------------------------------------------------------------------ */

#define DELEGATE_1Q_THETA(name) \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, \
                                            uint32_t q, double theta) { \
        if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT; \
        return map_err(moonlab_ca_mps_##name(s->mps, q, theta)); \
    }

DELEGATE_1Q_THETA(rx)
DELEGATE_1Q_THETA(ry)
DELEGATE_1Q_THETA(rz)
DELEGATE_1Q_THETA(phase)

#undef DELEGATE_1Q_THETA

ca_peps_error_t moonlab_ca_peps_t_gate(moonlab_ca_peps_t* s, uint32_t q) {
    if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_t_gate(s->mps, q));
}
ca_peps_error_t moonlab_ca_peps_t_dagger(moonlab_ca_peps_t* s, uint32_t q) {
    if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_t_dagger(s->mps, q));
}

ca_peps_error_t moonlab_ca_peps_pauli_rotation(moonlab_ca_peps_t* s,
                                                const uint8_t* pauli,
                                                double theta) {
    if (!s || !pauli) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_pauli_rotation(s->mps, pauli, theta));
}

ca_peps_error_t moonlab_ca_peps_imag_pauli_rotation(moonlab_ca_peps_t* s,
                                                     const uint8_t* pauli,
                                                     double tau) {
    if (!s || !pauli) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_imag_pauli_rotation(s->mps, pauli, tau));
}

ca_peps_error_t moonlab_ca_peps_normalize(moonlab_ca_peps_t* s) {
    if (!s) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_normalize(s->mps));
}

double moonlab_ca_peps_norm(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_norm(s->mps) : 0.0;
}

/* ------------------------------------------------------------------ */
/*  Pauli expectation.  Delegate -- the underlying contraction is     */
/*  geometry-agnostic given the row-major embedding.                   */
/* ------------------------------------------------------------------ */

ca_peps_error_t moonlab_ca_peps_expect_pauli(const moonlab_ca_peps_t* s,
                                              const uint8_t* pauli,
                                              double _Complex* out_expval) {
    if (!s || !pauli || !out_expval) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_expect_pauli(s->mps, pauli, out_expval));
}

ca_peps_error_t moonlab_ca_peps_expect_pauli_sum(const moonlab_ca_peps_t* s,
                                                  const uint8_t* paulis,
                                                  const double _Complex* coeffs,
                                                  uint32_t num_terms,
                                                  double _Complex* out_expval) {
    if (!s || !paulis || !coeffs || !out_expval) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_expect_pauli_sum(
        s->mps, paulis, coeffs, num_terms, out_expval));
}

ca_peps_error_t moonlab_ca_peps_prob_z(const moonlab_ca_peps_t* s,
                                        uint32_t q, double* out_prob) {
    if (!s || !out_prob) return CA_PEPS_ERR_INVALID;
    if (q >= s->Lx * s->Ly) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_prob_z(s->mps, q, out_prob));
}
