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
#include "../../applications/moonlab_export.h"

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

static moonlab_ca_peps_t* trace_ca_peps_create(uint32_t Lx, uint32_t Ly,
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

static void trace_ca_peps_free(moonlab_ca_peps_t* s) {
    if (!s) return;
    if (s->mps) moonlab_ca_mps_free(s->mps);
    free(s);
}

static moonlab_ca_peps_t* trace_ca_peps_clone(const moonlab_ca_peps_t* s) {
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

#define ML_CA_PEPS_CREATE_API(symbol)                                               \
    moonlab_ca_peps_t* symbol(uint32_t Lx, uint32_t Ly, uint32_t chi_bond) {        \
        return trace_ca_peps_create(Lx, Ly, chi_bond);                              \
    }

#define ML_CA_PEPS_FREE_API(symbol)                                                 \
    void symbol(moonlab_ca_peps_t* s) {                                             \
        trace_ca_peps_free(s);                                                      \
    }

#define ML_CA_PEPS_CLONE_API(symbol)                                                \
    moonlab_ca_peps_t* symbol(const moonlab_ca_peps_t* s) {                         \
        return trace_ca_peps_clone(s);                                              \
    }

ML_CA_PEPS_CREATE_API(moonlab_ca_peps_create) /* macro-generated API */
ML_CA_PEPS_FREE_API(moonlab_ca_peps_free) /* macro-generated API */
ML_CA_PEPS_CLONE_API(moonlab_ca_peps_clone) /* macro-generated API */

uint32_t moonlab_ca_peps_lx(const moonlab_ca_peps_t* s) { return s ? s->Lx : 0; }
uint32_t moonlab_ca_peps_ly(const moonlab_ca_peps_t* s) { return s ? s->Ly : 0; }
uint32_t moonlab_ca_peps_num_qubits(const moonlab_ca_peps_t* s) {
    return s ? s->Lx * s->Ly : 0;
}
uint32_t moonlab_ca_peps_max_bond_dim(const moonlab_ca_peps_t* s) {
    return s ? s->chi_bond : 0;
}
static uint32_t trace_ca_peps_current_bond_dim(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_current_bond_dim(s->mps) : 0;
}
static double trace_ca_peps_max_half_cut_entropy(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_max_half_cut_entropy(s->mps) : 0.0;
}

#define ML_CA_PEPS_CURRENT_BOND_API(symbol)                                         \
    uint32_t symbol(const moonlab_ca_peps_t* s) {                                   \
        return trace_ca_peps_current_bond_dim(s);                                   \
    }

#define ML_CA_PEPS_MAX_ENTROPY_API(symbol)                                          \
    double symbol(const moonlab_ca_peps_t* s) {                                     \
        return trace_ca_peps_max_half_cut_entropy(s);                               \
    }

ML_CA_PEPS_CURRENT_BOND_API(moonlab_ca_peps_current_bond_dim) /* macro-generated API */
ML_CA_PEPS_MAX_ENTROPY_API(moonlab_ca_peps_max_half_cut_entropy) /* macro-generated API */

/* ------------------------------------------------------------------ */
/*  Clifford gates -- tableau-only, geometry-free.  Delegate.          */
/* ------------------------------------------------------------------ */

#define TRACE_DELEGATE_1Q(name)                                                     \
    static ca_peps_error_t trace_ca_peps_##name(moonlab_ca_peps_t* s, uint32_t q) { \
        if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;                         \
        return map_err(moonlab_ca_mps_##name(s->mps, q));                           \
    }                                                                               \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s, uint32_t q) {      \
        return trace_ca_peps_##name(s, q);                                          \
    }

TRACE_DELEGATE_1Q(h) /* macro-generated API */
TRACE_DELEGATE_1Q(s) /* macro-generated API */
TRACE_DELEGATE_1Q(sdag) /* macro-generated API */
TRACE_DELEGATE_1Q(x) /* macro-generated API */
TRACE_DELEGATE_1Q(y) /* macro-generated API */
TRACE_DELEGATE_1Q(z) /* macro-generated API */

#undef TRACE_DELEGATE_1Q

static ca_peps_error_t trace_ca_peps_cnot(moonlab_ca_peps_t* s,
                                          uint32_t c, uint32_t t) {
    if (!site_in_range(s, c) || !site_in_range(s, t)) return CA_PEPS_ERR_QUBIT;
    /* For Clifford gates, the tableau update is independent of lattice
     * geometry, but we still validate adjacency so callers get an honest
     * error if they treat distant sites as "neighbours". */
    if (!sites_adjacent(s, c, t)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_cnot(s->mps, c, t));
}

static ca_peps_error_t trace_ca_peps_cz(moonlab_ca_peps_t* s,
                                        uint32_t a, uint32_t b) {
    if (!site_in_range(s, a) || !site_in_range(s, b)) return CA_PEPS_ERR_QUBIT;
    if (!sites_adjacent(s, a, b)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_cz(s->mps, a, b));
}

#define ML_CA_PEPS_CNOT_API(symbol)                                                 \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s, uint32_t c, uint32_t t) {          \
        return trace_ca_peps_cnot(s, c, t);                                         \
    }

#define ML_CA_PEPS_CZ_API(symbol)                                                   \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s, uint32_t a, uint32_t b) {          \
        return trace_ca_peps_cz(s, a, b);                                           \
    }

ML_CA_PEPS_CNOT_API(moonlab_ca_peps_cnot) /* macro-generated API */
ML_CA_PEPS_CZ_API(moonlab_ca_peps_cz) /* macro-generated API */

/* ------------------------------------------------------------------ */
/*  Non-Clifford single-qubit rotations.  Delegate.                   */
/* ------------------------------------------------------------------ */

#define TRACE_DELEGATE_1Q_THETA(name)                                               \
    static ca_peps_error_t trace_ca_peps_##name(moonlab_ca_peps_t* s,               \
                                                uint32_t q, double theta) {         \
        if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;                         \
        return map_err(moonlab_ca_mps_##name(s->mps, q, theta));                    \
    }                                                                               \
    ca_peps_error_t moonlab_ca_peps_##name(moonlab_ca_peps_t* s,                    \
                                           uint32_t q, double theta) {              \
        return trace_ca_peps_##name(s, q, theta);                                   \
    }

TRACE_DELEGATE_1Q_THETA(rx) /* macro-generated API */
TRACE_DELEGATE_1Q_THETA(ry) /* macro-generated API */
TRACE_DELEGATE_1Q_THETA(rz) /* macro-generated API */
TRACE_DELEGATE_1Q_THETA(phase) /* macro-generated API */

#undef TRACE_DELEGATE_1Q_THETA

static ca_peps_error_t trace_ca_peps_t_gate(moonlab_ca_peps_t* s, uint32_t q) {
    if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_t_gate(s->mps, q));
}
static ca_peps_error_t trace_ca_peps_t_dagger(moonlab_ca_peps_t* s, uint32_t q) {
    if (!site_in_range(s, q)) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_t_dagger(s->mps, q));
}

#define ML_CA_PEPS_T_GATE_API(symbol)                                               \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s, uint32_t q) {                      \
        return trace_ca_peps_t_gate(s, q);                                          \
    }

#define TRACE_CA_PEPS_T_DAGGER_API(symbol)                                          \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s, uint32_t q) {                      \
        return trace_ca_peps_t_dagger(s, q);                                        \
    }

ML_CA_PEPS_T_GATE_API(moonlab_ca_peps_t_gate) /* macro-generated API */
TRACE_CA_PEPS_T_DAGGER_API(moonlab_ca_peps_t_dagger) /* macro-generated API */

static ca_peps_error_t trace_ca_peps_pauli_rotation(moonlab_ca_peps_t* s,
                                                    const uint8_t* pauli,
                                                    double theta) {
    if (!s || !pauli) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_pauli_rotation(s->mps, pauli, theta));
}

static ca_peps_error_t trace_ca_peps_imag_pauli_rotation(moonlab_ca_peps_t* s,
                                                         const uint8_t* pauli,
                                                         double tau) {
    if (!s || !pauli) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_imag_pauli_rotation(s->mps, pauli, tau));
}

static ca_peps_error_t trace_ca_peps_normalize(moonlab_ca_peps_t* s) {
    if (!s) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_normalize(s->mps));
}

static double trace_ca_peps_norm(const moonlab_ca_peps_t* s) {
    return s ? moonlab_ca_mps_norm(s->mps) : 0.0;
}

#define ML_CA_PEPS_PAULI_ROTATION_API(symbol)                                       \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s,                                    \
                           const uint8_t* pauli, double theta) {                   \
        return trace_ca_peps_pauli_rotation(s, pauli, theta);                       \
    }

#define TRACE_CA_PEPS_IMAG_PAULI_ROTATION_API(symbol)                               \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s,                                    \
                           const uint8_t* pauli, double tau) {                     \
        return trace_ca_peps_imag_pauli_rotation(s, pauli, tau);                    \
    }

#define ML_CA_PEPS_NORMALIZE_API(symbol)                                            \
    ca_peps_error_t symbol(moonlab_ca_peps_t* s) {                                  \
        return trace_ca_peps_normalize(s);                                          \
    }

#define ML_CA_PEPS_NORM_API(symbol)                                                 \
    double symbol(const moonlab_ca_peps_t* s) {                                     \
        return trace_ca_peps_norm(s);                                               \
    }

ML_CA_PEPS_PAULI_ROTATION_API(moonlab_ca_peps_pauli_rotation) /* macro-generated API */
TRACE_CA_PEPS_IMAG_PAULI_ROTATION_API(moonlab_ca_peps_imag_pauli_rotation) /* macro-generated API */
ML_CA_PEPS_NORMALIZE_API(moonlab_ca_peps_normalize) /* macro-generated API */
ML_CA_PEPS_NORM_API(moonlab_ca_peps_norm) /* macro-generated API */

/* ------------------------------------------------------------------ */
/*  Pauli expectation.  Delegate -- the underlying contraction is     */
/*  geometry-agnostic given the row-major embedding.                   */
/* ------------------------------------------------------------------ */

static ca_peps_error_t trace_ca_peps_expect_pauli(const moonlab_ca_peps_t* s,
                                                  const uint8_t* pauli,
                                                  double _Complex* out_expval) {
    if (!s || !pauli || !out_expval) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_expect_pauli(s->mps, pauli, out_expval));
}

static ca_peps_error_t trace_ca_peps_expect_pauli_sum(const moonlab_ca_peps_t* s,
                                                      const uint8_t* paulis,
                                                      const double _Complex* coeffs,
                                                      uint32_t num_terms,
                                                      double _Complex* out_expval) {
    if (!s || !paulis || !coeffs || !out_expval) return CA_PEPS_ERR_INVALID;
    return map_err(moonlab_ca_mps_expect_pauli_sum(
        s->mps, paulis, coeffs, num_terms, out_expval));
}

static ca_peps_error_t trace_ca_peps_prob_z(const moonlab_ca_peps_t* s,
                                            uint32_t q, double* out_prob) {
    if (!s || !out_prob) return CA_PEPS_ERR_INVALID;
    if (q >= s->Lx * s->Ly) return CA_PEPS_ERR_QUBIT;
    return map_err(moonlab_ca_mps_prob_z(s->mps, q, out_prob));
}

#define ML_CA_PEPS_EXPECT_PAULI_API(symbol)                                         \
    ca_peps_error_t symbol(const moonlab_ca_peps_t* s, const uint8_t* pauli,        \
                           double _Complex* out_expval) {                          \
        return trace_ca_peps_expect_pauli(s, pauli, out_expval);                    \
    }

#define TRACE_CA_PEPS_EXPECT_PAULI_SUM_API(symbol)                                  \
    ca_peps_error_t symbol(const moonlab_ca_peps_t* s, const uint8_t* paulis,       \
                           const double _Complex* coeffs, uint32_t num_terms,       \
                           double _Complex* out_expval) {                          \
        return trace_ca_peps_expect_pauli_sum(s, paulis, coeffs,                    \
                                             num_terms, out_expval);                \
    }

#define ML_CA_PEPS_PROB_Z_API(symbol)                                               \
    ca_peps_error_t symbol(const moonlab_ca_peps_t* s,                              \
                           uint32_t q, double* out_prob) {                         \
        return trace_ca_peps_prob_z(s, q, out_prob);                                \
    }

ML_CA_PEPS_EXPECT_PAULI_API(moonlab_ca_peps_expect_pauli) /* macro-generated API */
TRACE_CA_PEPS_EXPECT_PAULI_SUM_API(moonlab_ca_peps_expect_pauli_sum) /* macro-generated API */
ML_CA_PEPS_PROB_Z_API(moonlab_ca_peps_prob_z) /* macro-generated API */

/* ------------------------------------------------------------------ */
/*  Variational-D run (delegate to CA-MPS engine).                     */
/* ------------------------------------------------------------------ */

static int trace_ca_peps_var_d_run(moonlab_ca_peps_t* state,
                                   const uint8_t* paulis,
                                   const double* coeffs,
                                   uint32_t num_terms,
                                   uint32_t max_outer_iters,
                                   double imag_time_dtau,
                                   uint32_t imag_time_steps_per_outer,
                                   uint32_t clifford_passes_per_outer,
                                   int composite_2gate,
                                   int warmstart,
                                   const uint8_t* stab_paulis,
                                   uint32_t stab_num_gens,
                                   double* out_final_energy) {
    if (!state || !paulis || !coeffs) return CA_PEPS_ERR_INVALID;
    return moonlab_ca_mps_var_d_run(
        state->mps, paulis, coeffs, num_terms,
        max_outer_iters, imag_time_dtau,
        imag_time_steps_per_outer, clifford_passes_per_outer,
        composite_2gate, warmstart,
        stab_paulis, stab_num_gens, out_final_energy);
}

#define ML_CA_PEPS_VAR_D_RUN_API(symbol)                                            \
    int symbol(moonlab_ca_peps_t* state, const uint8_t* paulis,                     \
               const double* coeffs, uint32_t num_terms,                            \
               uint32_t max_outer_iters, double imag_time_dtau,                     \
               uint32_t imag_time_steps_per_outer,                                  \
               uint32_t clifford_passes_per_outer, int composite_2gate,             \
               int warmstart, const uint8_t* stab_paulis,                           \
               uint32_t stab_num_gens, double* out_final_energy) {                  \
        return trace_ca_peps_var_d_run(state, paulis, coeffs, num_terms,            \
                                       max_outer_iters, imag_time_dtau,             \
                                       imag_time_steps_per_outer,                   \
                                       clifford_passes_per_outer, composite_2gate,  \
                                       warmstart, stab_paulis, stab_num_gens,       \
                                       out_final_energy);                           \
    }

ML_CA_PEPS_VAR_D_RUN_API(moonlab_ca_peps_var_d_run) /* macro-generated API */
