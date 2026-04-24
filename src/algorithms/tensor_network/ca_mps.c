/**
 * @file ca_mps.c
 * @brief Clifford-Assisted Matrix Product State implementation.
 *
 * See docs/research/ca_mps.md for the design and math.  This file glues
 * the Aaronson-Gottesman tableau (src/backends/clifford/) to the MPS
 * machinery (tn_state, tn_gates) via a Pauli-string-rotation MPO primitive.
 */
#include "ca_mps.h"

#include "tn_state.h"
#include "tn_gates.h"
#include "tn_measurement.h"
#include "tensor.h"
#include "../../backends/clifford/clifford.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct moonlab_ca_mps_t {
    uint32_t n;                     /* number of qubits */
    uint32_t max_bond;              /* MPS truncation cap */
    clifford_tableau_t* D;          /* tableau storing C^dagger */
    tn_mps_state_t*     phi;        /* |phi>; |psi> = D^dagger |phi> = C |phi> */
};

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t n, uint32_t max_bond) {
    if (n == 0 || max_bond == 0) return NULL;
    moonlab_ca_mps_t* s = (moonlab_ca_mps_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->n = n;
    s->max_bond = max_bond;
    s->D = clifford_tableau_create((size_t)n);
    if (!s->D) { free(s); return NULL; }

    tn_state_config_t cfg = tn_state_config_create(max_bond, 1e-12);
    s->phi = tn_mps_create_zero(n, &cfg);
    if (!s->phi) {
        clifford_tableau_free(s->D);
        free(s);
        return NULL;
    }
    return s;
}

void moonlab_ca_mps_free(moonlab_ca_mps_t* s) {
    if (!s) return;
    if (s->D)   clifford_tableau_free(s->D);
    if (s->phi) tn_mps_free(s->phi);
    free(s);
}

moonlab_ca_mps_t* moonlab_ca_mps_clone(const moonlab_ca_mps_t* s) {
    if (!s) return NULL;
    moonlab_ca_mps_t* c = (moonlab_ca_mps_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n = s->n;
    c->max_bond = s->max_bond;
    c->D = clifford_tableau_clone(s->D);
    c->phi = tn_mps_copy(s->phi);
    if (!c->D || !c->phi) {
        moonlab_ca_mps_free(c);
        return NULL;
    }
    return c;
}

uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s) {
    return s ? s->n : 0;
}
uint32_t moonlab_ca_mps_max_bond_dim(const moonlab_ca_mps_t* s) {
    return s ? s->max_bond : 0;
}
uint32_t moonlab_ca_mps_current_bond_dim(const moonlab_ca_mps_t* s) {
    if (!s || !s->phi) return 0;
    /* The MPS struct exposes bond_dims via tn_state.h; walk and find the max. */
    uint32_t max_b = 1;
    for (uint32_t i = 0; i + 1 < s->n; i++) {
        uint32_t b = tn_mps_bond_dim(s->phi, i);
        if (b > max_b) max_b = b;
    }
    return max_b;
}

/* ------------------------------------------------------------------ */
/*  Clifford gates -- tableau-only updates.                           */
/*                                                                    */
/*  The tableau D stores C^dagger, where |psi> = C|phi>.  To apply    */
/*  Clifford gate G on |psi> we want G|psi> = GC|phi>, i.e. C <- GC.  */
/*  Equivalently D <- (GC)^dagger = C^dagger G^dagger = D G^dagger.    */
/*                                                                    */
/*  The existing clifford_{h,s,...}(tableau, q) call applies a gate   */
/*  to the tableau such that after the call the tableau represents    */
/*  the product of the previous tableau's C with the new gate.  Given */
/*  our convention (tableau holds C^dagger), we need to apply         */
/*  clifford_*(D, q) with the GATE DAGGER.  For self-inverse gates    */
/*  (H, X, Y, Z, CNOT, CZ, SWAP) the dagger is the gate itself.  For  */
/*  S the dagger is S^dagger and vice versa.                          */
/* ------------------------------------------------------------------ */

#define CKEY(e) ((e) == CLIFFORD_SUCCESS ? CA_MPS_SUCCESS : CA_MPS_ERR_BACKEND)

/* Convention: the tableau `s->D` stores the FORWARD Clifford C that has
 * been applied to the state.  After each Clifford gate call the tableau
 * is updated by standard Heisenberg conjugation (left-multiplication by
 * the gate). */
ca_mps_error_t moonlab_ca_mps_h(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_h(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_s(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_s(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_sdag(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_s_dag(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_x(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_x(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_y(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_y(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_z(moonlab_ca_mps_t* s, uint32_t q) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_z(s->D, q));
}
ca_mps_error_t moonlab_ca_mps_cnot(moonlab_ca_mps_t* s, uint32_t c, uint32_t t) {
    if (!s || c >= s->n || t >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_cnot(s->D, c, t));
}
ca_mps_error_t moonlab_ca_mps_cz(moonlab_ca_mps_t* s, uint32_t a, uint32_t b) {
    if (!s || a >= s->n || b >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_cz(s->D, a, b));
}
ca_mps_error_t moonlab_ca_mps_swap(moonlab_ca_mps_t* s, uint32_t a, uint32_t b) {
    if (!s || a >= s->n || b >= s->n) return CA_MPS_ERR_QUBIT;
    return CKEY(clifford_swap(s->D, a, b));
}

/* ------------------------------------------------------------------ */
/*  Pauli-string rotation MPO                                         */
/*                                                                    */
/*  For a Pauli string P = P_{q1} ... P_{qk} (q1 < ... < qk) on n     */
/*  qubits, exp(i theta P) is a sum of two branches:                  */
/*    - identity branch with weight cos(theta)                        */
/*    - Pauli branch with weight i sin(theta)                         */
/*                                                                    */
/*  We build a bond-dim-2 MPO:                                        */
/*    - Sites 0..q1-1 carry bond (1, 1) and apply identity.           */
/*    - Site q1 (bond (1, 2)): W[0, 0] = cos*I, W[0, 1] = i sin * P_q1. */
/*    - Sites q1+1..qk-1 (bond (2, 2)): W[0, 0] = I, W[1, 1] = P_q    */
/*       (if q in string) or I (if q not in string).                  */
/*    - Site qk (bond (2, 1)): W[0, 0] = I, W[1, 0] = P_qk.           */
/*    - Sites qk+1..n-1 carry bond (1, 1) and apply identity.         */
/*                                                                    */
/*  Special case: weight 0 (identity Pauli string) -> overall phase   */
/*  exp(i theta); skip the MPS apply.  Weight 1 works with the        */
/*  general construction too.                                         */
/* ------------------------------------------------------------------ */

/* Pauli matrix elements in row-major [out][in] form.  Access as
 * PAULI_MAT[p][out][in] where p in {0,1,2,3}. */
static const double _Complex PAULI_MAT[4][2][2] = {
    /* I */ { {1.0, 0.0}, {0.0, 1.0} },
    /* X */ { {0.0, 1.0}, {1.0, 0.0} },
    /* Y */ { {0.0, -1.0*I}, {1.0*I, 0.0} },
    /* Z */ { {1.0, 0.0}, {0.0, -1.0} },
};

static int pauli_weight(const uint8_t* p, uint32_t n) {
    int w = 0;
    for (uint32_t i = 0; i < n; i++) if (p[i] != 0) w++;
    return w;
}

static tn_mpo_t* build_pauli_rotation_mpo(uint32_t n,
                                          const uint8_t* pauli, int phase_in,
                                          double theta) {
    /* phase_in is the accumulated phase (in powers of i) from Clifford
     * conjugation of the generator.  Fold it into the angle so that
     * exp(i theta P) (with P understood to include phase_in) is correctly
     * represented as exp(i (theta + phase_in * pi/2) * P_bare) ... but
     * that mixes cos/sin coefficients badly.  Cleaner: apply phase_in as
     * a scalar multiplier to the "Pauli branch" coefficient.
     *
     * Specifically: the incoming Pauli is phase_in * P_bare where
     * phase_in in {+1, +i, -1, -i} = i^phase_in.  The generator is thus
     * G = i^phase_in * P_bare.  exp(i theta G) = cos(theta) I +
     * i sin(theta) i^phase_in * P_bare.
     *
     * We store the Pauli branch coefficient as i * sin(theta) * i^phase_in
     * = sin(theta) * i^(phase_in + 1). */
    double _Complex i_pow[4] = { 1.0, 1.0*I, -1.0, -1.0*I };
    double _Complex branch_coef = sin(theta) * i_pow[(phase_in + 1) & 3];
    double cos_t = cos(theta);

    /* Find first and last non-identity position. */
    uint32_t q_first = n, q_last = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (pauli[i] != 0) {
            if (q_first == n) q_first = i;
            q_last = i;
        }
    }
    if (q_first == n) {
        /* Identity Pauli string: whole operator is cos*I + i sin*I =
         * exp(i theta) * I.  Caller should handle identity specially; we
         * return NULL to signal "no MPO needed, apply global phase". */
        return NULL;
    }

    tn_mpo_t* mpo = (tn_mpo_t*)calloc(1, sizeof(*mpo));
    if (!mpo) return NULL;
    mpo->num_sites = n;
    mpo->tensors = (tensor_t**)calloc(n, sizeof(tensor_t*));
    mpo->bond_dims = (uint32_t*)calloc(n > 1 ? n - 1 : 1, sizeof(uint32_t));
    if (!mpo->tensors || !mpo->bond_dims) { tn_mpo_free(mpo); return NULL; }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t lb, rb;
        if (i < q_first) { lb = 1; rb = 1; }
        else if (i == q_first && i == q_last) { lb = 1; rb = 1; }
        else if (i == q_first) { lb = 1; rb = 2; }
        else if (i == q_last) { lb = 2; rb = 1; }
        else if (i > q_first && i < q_last) { lb = 2; rb = 2; }
        else { lb = 1; rb = 1; }

        uint32_t dims[4] = {lb, 2, 2, rb};  /* TN_PHYSICAL_DIM = 2 */
        mpo->tensors[i] = tensor_create(4, dims);
        if (!mpo->tensors[i]) { tn_mpo_free(mpo); return NULL; }

        uint8_t pi_code = pauli[i];
        const double _Complex (*Ppi)[2] = PAULI_MAT[pi_code];
        const double _Complex (*Ipi)[2] = PAULI_MAT[0];

        if (i < q_first || i > q_last) {
            /* Identity passthrough, bond (1,1). */
            for (uint32_t pin = 0; pin < 2; pin++)
                for (uint32_t pout = 0; pout < 2; pout++) {
                    uint32_t idx[4] = {0, pin, pout, 0};
                    tensor_set(mpo->tensors[i], idx, Ipi[pout][pin]);
                }
        } else if (i == q_first && i == q_last) {
            /* Weight-1 string: entire rotation at one site, bond (1,1).
             * W[0, pi, po, 0] = cos * I[po,pi] + branch_coef * P[po,pi]. */
            for (uint32_t pin = 0; pin < 2; pin++)
                for (uint32_t pout = 0; pout < 2; pout++) {
                    uint32_t idx[4] = {0, pin, pout, 0};
                    double _Complex v = cos_t * Ipi[pout][pin]
                                      + branch_coef * Ppi[pout][pin];
                    tensor_set(mpo->tensors[i], idx, v);
                }
        } else if (i == q_first) {
            /* bond (1, 2): row vector.
             *   W[0, pin, pout, 0] = cos * I[pout, pin]              (identity branch)
             *   W[0, pin, pout, 1] = branch_coef * P[pout, pin]       (Pauli branch) */
            for (uint32_t pin = 0; pin < 2; pin++)
                for (uint32_t pout = 0; pout < 2; pout++) {
                    uint32_t idx0[4] = {0, pin, pout, 0};
                    uint32_t idx1[4] = {0, pin, pout, 1};
                    tensor_set(mpo->tensors[i], idx0, cos_t * Ipi[pout][pin]);
                    tensor_set(mpo->tensors[i], idx1, branch_coef * Ppi[pout][pin]);
                }
        } else if (i == q_last) {
            /* bond (2, 1): column vector.
             *   W[0, pin, pout, 0] = I[pout, pin]                     (identity branch)
             *   W[1, pin, pout, 0] = P[pout, pin]                     (Pauli branch) */
            for (uint32_t pin = 0; pin < 2; pin++)
                for (uint32_t pout = 0; pout < 2; pout++) {
                    uint32_t idx0[4] = {0, pin, pout, 0};
                    uint32_t idx1[4] = {1, pin, pout, 0};
                    tensor_set(mpo->tensors[i], idx0, Ipi[pout][pin]);
                    tensor_set(mpo->tensors[i], idx1, Ppi[pout][pin]);
                }
        } else {
            /* bond (2, 2): diagonal passthrough.
             *   W[0, pin, pout, 0] = I[pout, pin]
             *   W[1, pin, pout, 1] = P[pout, pin]  (P = I if pi_code==0) */
            for (uint32_t pin = 0; pin < 2; pin++)
                for (uint32_t pout = 0; pout < 2; pout++) {
                    uint32_t idx00[4] = {0, pin, pout, 0};
                    uint32_t idx11[4] = {1, pin, pout, 1};
                    tensor_set(mpo->tensors[i], idx00, Ipi[pout][pin]);
                    tensor_set(mpo->tensors[i], idx11, Ppi[pout][pin]);
                }
        }

        if (i + 1 < n) mpo->bond_dims[i] = rb;
    }

    return mpo;
}

/* ------------------------------------------------------------------ */
/*  Apply a Pauli rotation as if it were exp(i theta * phase_in * P).  */
/*  Handles identity case (global phase), weight-1 (direct MPS op),    */
/*  and general weight via the rotation MPO.                            */
/* ------------------------------------------------------------------ */
static ca_mps_error_t apply_conjugated_rotation(moonlab_ca_mps_t* s,
                                                const uint8_t* pauli,
                                                int phase_in, double theta) {
    int w = pauli_weight(pauli, s->n);
    if (w == 0) {
        /* P = i^phase_in * I_n, so exp(i theta P) = exp(i theta i^phase_in) I_n.
         * Apply as global phase to the MPS. */
        double _Complex i_pow[4] = { 1.0, 1.0*I, -1.0, -1.0*I };
        double _Complex scalar = cexp(I * theta * i_pow[phase_in & 3]);
        double global_phase = carg(scalar);
        tn_gate_error_t e = tn_apply_global_phase(s->phi, global_phase);
        return (e == TN_GATE_SUCCESS) ? CA_MPS_SUCCESS : CA_MPS_ERR_BACKEND;
    }

    tn_mpo_t* mpo = build_pauli_rotation_mpo(s->n, pauli, phase_in, theta);
    if (!mpo) return CA_MPS_ERR_OOM;
    double trunc = 0.0;
    tn_gate_error_t e = tn_apply_mpo(s->phi, mpo, &trunc);
    tn_mpo_free(mpo);
    return (e == TN_GATE_SUCCESS) ? CA_MPS_SUCCESS : CA_MPS_ERR_BACKEND;
}

/* ------------------------------------------------------------------ */
/*  Non-Clifford single-qubit rotations:                               */
/*  exp(i theta P_q)|psi> = C (C^dagger exp(i theta P_q) C) |phi>      */
/*                        = C exp(i theta (D P_q D^dagger)) |phi>       */
/*                        = C exp(i theta P') |phi>                     */
/*  where P' = D P_q D^dagger is read from the tableau.                */
/* ------------------------------------------------------------------ */

static ca_mps_error_t apply_single_qubit_generator(moonlab_ca_mps_t* s,
                                                   uint32_t q, uint8_t gen_code,
                                                   double theta) {
    if (!s || q >= s->n) return CA_MPS_ERR_QUBIT;
    uint8_t* in_pauli = (uint8_t*)calloc(s->n, sizeof(uint8_t));
    uint8_t* out_pauli = (uint8_t*)calloc(s->n, sizeof(uint8_t));
    if (!in_pauli || !out_pauli) {
        free(in_pauli); free(out_pauli);
        return CA_MPS_ERR_OOM;
    }
    in_pauli[q] = gen_code;
    int out_phase = 0;
    /* Non-Clifford pushthrough needs C^dagger P C, hence inverse conjugation. */
    clifford_conjugate_pauli_inverse(s->D, in_pauli, 0, out_pauli, &out_phase);
    ca_mps_error_t e = apply_conjugated_rotation(s, out_pauli, out_phase, theta);
    free(in_pauli); free(out_pauli);
    return e;
}

/* Public rotation API follows the standard Qiskit/Cirq convention:
 *   R_P(theta) = exp(-i theta P / 2)
 * so we convert to our internal generator (which applies exp(+i phi P))
 * via phi = -theta / 2. */
ca_mps_error_t moonlab_ca_mps_rx(moonlab_ca_mps_t* s, uint32_t q, double theta) {
    return apply_single_qubit_generator(s, q, 1 /* X */, -theta / 2.0);
}
ca_mps_error_t moonlab_ca_mps_ry(moonlab_ca_mps_t* s, uint32_t q, double theta) {
    return apply_single_qubit_generator(s, q, 2 /* Y */, -theta / 2.0);
}
ca_mps_error_t moonlab_ca_mps_rz(moonlab_ca_mps_t* s, uint32_t q, double theta) {
    return apply_single_qubit_generator(s, q, 3 /* Z */, -theta / 2.0);
}
ca_mps_error_t moonlab_ca_mps_t_gate(moonlab_ca_mps_t* s, uint32_t q) {
    /* T = diag(1, e^{i pi/4}) differs from R_Z(pi/4) = diag(e^{-i pi/8},
     * e^{i pi/8}) only by a global phase e^{i pi/8}, which doesn't affect
     * observables.  So T is moonlab_ca_mps_rz(q, M_PI/4). */
    return moonlab_ca_mps_rz(s, q, M_PI / 4.0);
}

ca_mps_error_t moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
                                             const uint8_t* pauli, double theta) {
    if (!s || !pauli) return CA_MPS_ERR_INVALID;
    uint8_t* out_pauli = (uint8_t*)calloc(s->n, sizeof(uint8_t));
    if (!out_pauli) return CA_MPS_ERR_OOM;
    int out_phase = 0;
    clifford_conjugate_pauli_inverse(s->D, pauli, 0, out_pauli, &out_phase);
    ca_mps_error_t e = apply_conjugated_rotation(s, out_pauli, out_phase, theta);
    free(out_pauli);
    return e;
}

/* ------------------------------------------------------------------ */
/*  Expectation <psi | P | psi> = <phi | D P D^dagger | phi>          */
/* ------------------------------------------------------------------ */
ca_mps_error_t moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
                                           const uint8_t* pauli,
                                           double _Complex* out_expval) {
    if (!s || !pauli || !out_expval) return CA_MPS_ERR_INVALID;
    uint8_t* conj_pauli = (uint8_t*)calloc(s->n, sizeof(uint8_t));
    if (!conj_pauli) return CA_MPS_ERR_OOM;
    int conj_phase = 0;
    /* <psi|P|psi> = <phi|C^dagger P C|phi>, so apply inverse conjugation. */
    clifford_conjugate_pauli_inverse(s->D, pauli, 0, conj_pauli, &conj_phase);

    double _Complex raw = tn_expectation_pauli_string(s->phi, conj_pauli);
    double _Complex i_pow[4] = { 1.0, 1.0*I, -1.0, -1.0*I };
    *out_expval = raw * i_pow[conj_phase & 3];
    free(conj_pauli);
    return CA_MPS_SUCCESS;
}

ca_mps_error_t moonlab_ca_mps_expect_pauli_sum(const moonlab_ca_mps_t* s,
                                                const uint8_t* paulis,
                                                const double _Complex* coeffs,
                                                uint32_t num_terms,
                                                double _Complex* out_expval) {
    if (!s || !paulis || !coeffs || !out_expval) return CA_MPS_ERR_INVALID;
    double _Complex acc = 0.0;
    for (uint32_t k = 0; k < num_terms; k++) {
        double _Complex term;
        ca_mps_error_t e = moonlab_ca_mps_expect_pauli(s, paulis + (size_t)k * s->n, &term);
        if (e != CA_MPS_SUCCESS) return e;
        acc += coeffs[k] * term;
    }
    *out_expval = acc;
    return CA_MPS_SUCCESS;
}
