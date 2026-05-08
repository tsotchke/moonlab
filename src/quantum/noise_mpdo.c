/**
 * @file noise_mpdo.c
 * @brief Matrix-product density operator implementation.
 *
 * The MPDO is represented as an MPS with physical leg dimension 4
 * (vectorised local 2x2 density matrix).  A site tensor at site i
 * has shape (left_bond, 4, right_bond), stored row-major:
 *   tensors[i][L * 4 * R + p * R + r]
 * with index ordering (left, phys, right).
 *
 * Bond dimensions start at chi=1 for the |0><0| product initial state
 * and grow under multi-site channels and gates.  The first scaffold
 * implements only single-qubit channels, which keep the bond
 * dimension fixed at the level it has reached -- they only modify
 * the physical leg, leaving virtual bonds untouched.
 *
 * Layout choice: we keep the "vec-of-rho" basis (rho_00, rho_01,
 * rho_10, rho_11) directly, not the Pauli basis (I, X, Y, Z).  This
 * makes Kraus-superoperator construction explicit in 2x2 matrix
 * elements at the cost of a final basis change for Pauli
 * expectation values (a single 4x1 contraction).
 */

#include "noise_mpdo.h"

#include <stdlib.h>
#include <string.h>

struct moonlab_mpdo_t {
    uint32_t n;              /* number of qubits */
    uint32_t max_bond;       /* MPS truncation cap */
    uint32_t* bond_dims;     /* length n+1; bond_dims[0]=bond_dims[n]=1 */
    /* tensors[i] is a flat row-major array of shape
     *   (bond_dims[i], 4, bond_dims[i+1])
     * indexed as tensors[i][L * 4 * R + p * R + r]. */
    mpdo_complex_t** tensors;
};

static size_t tensor_size(uint32_t L, uint32_t R) {
    return (size_t)L * 4u * (size_t)R;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

moonlab_mpdo_t* moonlab_mpdo_create(uint32_t n, uint32_t max_bond) {
    if (n == 0 || max_bond == 0) return NULL;
    moonlab_mpdo_t* m = (moonlab_mpdo_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->n = n;
    m->max_bond = max_bond;
    m->bond_dims = (uint32_t*)calloc(n + 1, sizeof(uint32_t));
    m->tensors = (mpdo_complex_t**)calloc(n, sizeof(mpdo_complex_t*));
    if (!m->bond_dims || !m->tensors) {
        moonlab_mpdo_free(m);
        return NULL;
    }
    for (uint32_t i = 0; i <= n; i++) m->bond_dims[i] = 1;
    /* |0><0| product state: each site's vec(rho_i) = (1, 0, 0, 0).
     * tensor[i] has shape (1, 4, 1); only the (0, 0, 0) element is 1. */
    for (uint32_t i = 0; i < n; i++) {
        size_t sz = tensor_size(1, 1);
        m->tensors[i] = (mpdo_complex_t*)calloc(sz, sizeof(mpdo_complex_t));
        if (!m->tensors[i]) {
            moonlab_mpdo_free(m);
            return NULL;
        }
        m->tensors[i][0] = 1.0;   /* (L=0, phys=0, R=0) -> rho_00 = 1 */
    }
    return m;
}

void moonlab_mpdo_free(moonlab_mpdo_t* m) {
    if (!m) return;
    if (m->tensors) {
        for (uint32_t i = 0; i < m->n; i++) free(m->tensors[i]);
        free(m->tensors);
    }
    free(m->bond_dims);
    free(m);
}

moonlab_mpdo_t* moonlab_mpdo_clone(const moonlab_mpdo_t* m) {
    if (!m) return NULL;
    moonlab_mpdo_t* c = (moonlab_mpdo_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n = m->n;
    c->max_bond = m->max_bond;
    c->bond_dims = (uint32_t*)malloc((m->n + 1) * sizeof(uint32_t));
    c->tensors = (mpdo_complex_t**)calloc(m->n, sizeof(mpdo_complex_t*));
    if (!c->bond_dims || !c->tensors) { moonlab_mpdo_free(c); return NULL; }
    memcpy(c->bond_dims, m->bond_dims, (m->n + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < m->n; i++) {
        size_t sz = tensor_size(m->bond_dims[i], m->bond_dims[i + 1]);
        c->tensors[i] = (mpdo_complex_t*)malloc(sz * sizeof(mpdo_complex_t));
        if (!c->tensors[i]) { moonlab_mpdo_free(c); return NULL; }
        memcpy(c->tensors[i], m->tensors[i], sz * sizeof(mpdo_complex_t));
    }
    return c;
}

/* ------------------------------------------------------------------ */
/*  Introspection                                                     */
/* ------------------------------------------------------------------ */

uint32_t moonlab_mpdo_num_qubits(const moonlab_mpdo_t* m) {
    return m ? m->n : 0;
}

uint32_t moonlab_mpdo_max_bond_dim(const moonlab_mpdo_t* m) {
    return m ? m->max_bond : 0;
}

uint32_t moonlab_mpdo_current_bond_dim(const moonlab_mpdo_t* m) {
    if (!m || m->n == 0) return 0;
    uint32_t maxb = 1;
    for (uint32_t i = 1; i < m->n; i++) {
        if (m->bond_dims[i] > maxb) maxb = m->bond_dims[i];
    }
    return maxb;
}

/* Trace of the local 2x2 density matrix is rho_00 + rho_11 in the
 * vec basis -- physical-index components 0 and 3.  Tr(rho) is then
 * the contraction over all sites of the trace projector
 * vec_trace = (1, 0, 0, 1) on each physical leg. */
double moonlab_mpdo_trace(const moonlab_mpdo_t* m) {
    if (!m) return 0.0;
    /* Contract from left to right: maintain a running row vector v
     * on the right-bond side of the partial trace.  Initially v has
     * shape (1) holding amplitude 1.  After contracting site i we
     * sum tensors[i][L][p_trace][R] * v[L] over (L, p_trace) into
     * a new v of length bond_dims[i+1]. */
    uint32_t Lcap = m->max_bond > 1 ? m->max_bond : 1;
    mpdo_complex_t* v   = (mpdo_complex_t*)calloc(Lcap, sizeof(mpdo_complex_t));
    mpdo_complex_t* vn  = (mpdo_complex_t*)calloc(Lcap, sizeof(mpdo_complex_t));
    if (!v || !vn) { free(v); free(vn); return 0.0; }
    v[0] = 1.0;
    uint32_t cur_bond = 1;
    for (uint32_t i = 0; i < m->n; i++) {
        uint32_t L = m->bond_dims[i];
        uint32_t R = m->bond_dims[i + 1];
        /* Allocate larger if needed -- defensive; bond never exceeds
         * Lcap by construction. */
        for (uint32_t r = 0; r < R; r++) vn[r] = 0.0;
        const mpdo_complex_t* T = m->tensors[i];
        for (uint32_t l = 0; l < L; l++) {
            const mpdo_complex_t coef = v[l];
            if (cabs(coef) == 0.0) continue;
            /* Sum over phys = 0 (rho_00) and phys = 3 (rho_11) only. */
            for (uint32_t r = 0; r < R; r++) {
                vn[r] += coef * (T[l * 4 * R + 0 * R + r]
                                + T[l * 4 * R + 3 * R + r]);
            }
        }
        memcpy(v, vn, R * sizeof(mpdo_complex_t));
        cur_bond = R;
    }
    /* End: v has shape (1) = bond_dims[n]; v[0] = Tr(rho). */
    double tr_re = creal(v[0]);
    (void)cur_bond;
    free(v); free(vn);
    return tr_re;
}

/* ------------------------------------------------------------------ */
/*  Channel application                                               */
/* ------------------------------------------------------------------ */

/* Build the 4x4 superoperator S in vec-basis from a list of 2x2 Kraus
 * operators.  S[(rout * 2 + cout) * 4 + (rin * 2 + cin)]
 *   = sum_a K_a[rout, rin] * conj(K_a[cout, cin]).
 * That is: vec(K rho K^dag)[rout, cout] = sum_{rin, cin} K[rout, rin] *
 *   conj(K[cout, cin]) * vec(rho)[rin, cin].
 *
 * Vec basis convention: index = 2 * row + col, so rho_{rc} maps to
 * vec_index 2r + c.  Physical leg index 0 = rho_00, 1 = rho_01,
 * 2 = rho_10, 3 = rho_11. */
static void build_kraus_superop(const mpdo_complex_t* kraus,
                                 uint32_t num_kraus,
                                 mpdo_complex_t S[16]) {
    for (int i = 0; i < 16; i++) S[i] = 0.0;
    for (uint32_t a = 0; a < num_kraus; a++) {
        const mpdo_complex_t* K = &kraus[a * 4];
        for (uint32_t rout = 0; rout < 2; rout++)
        for (uint32_t cout = 0; cout < 2; cout++)
        for (uint32_t rin = 0; rin < 2; rin++)
        for (uint32_t cin = 0; cin < 2; cin++) {
            uint32_t out_idx = 2 * rout + cout;
            uint32_t in_idx  = 2 * rin  + cin;
            S[out_idx * 4 + in_idx] +=
                K[rout * 2 + rin] * conj(K[cout * 2 + cin]);
        }
    }
}

mpdo_error_t moonlab_mpdo_apply_kraus_1q(moonlab_mpdo_t* state,
                                          uint32_t qubit,
                                          const mpdo_complex_t* kraus,
                                          uint32_t num_kraus) {
    if (!state || !kraus || num_kraus == 0) return MPDO_ERR_INVALID;
    if (qubit >= state->n) return MPDO_ERR_QUBIT;

    mpdo_complex_t S[16];
    build_kraus_superop(kraus, num_kraus, S);

    /* Apply S to the physical leg of site `qubit`:
     * T'[L, p_out, R] = sum_{p_in} S[p_out, p_in] * T[L, p_in, R]. */
    uint32_t L = state->bond_dims[qubit];
    uint32_t R = state->bond_dims[qubit + 1];
    mpdo_complex_t* T = state->tensors[qubit];
    mpdo_complex_t* Tnew =
        (mpdo_complex_t*)malloc(tensor_size(L, R) * sizeof(mpdo_complex_t));
    if (!Tnew) return MPDO_ERR_OOM;
    for (uint32_t l = 0; l < L; l++)
    for (uint32_t pout = 0; pout < 4; pout++)
    for (uint32_t r = 0; r < R; r++) {
        mpdo_complex_t acc = 0.0;
        for (uint32_t pin = 0; pin < 4; pin++) {
            acc += S[pout * 4 + pin] * T[l * 4 * R + pin * R + r];
        }
        Tnew[l * 4 * R + pout * R + r] = acc;
    }
    memcpy(T, Tnew, tensor_size(L, R) * sizeof(mpdo_complex_t));
    free(Tnew);
    return MPDO_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Observables                                                       */
/* ------------------------------------------------------------------ */

/* Pauli matrices in row-major (rout, cin) form. */
static const mpdo_complex_t PAULI[4][4] = {
    /* I */ { 1.0,    0.0,    0.0,    1.0    },
    /* X */ { 0.0,    1.0,    1.0,    0.0    },
    /* Y */ { 0.0,    -_Complex_I, _Complex_I, 0.0 },
    /* Z */ { 1.0,    0.0,    0.0,    -1.0   },
};

mpdo_error_t moonlab_mpdo_expect_pauli_1q(const moonlab_mpdo_t* state,
                                           uint32_t qubit,
                                           uint8_t  pauli_code,
                                           double*  out_expval) {
    if (!state || !out_expval) return MPDO_ERR_INVALID;
    if (qubit >= state->n) return MPDO_ERR_QUBIT;
    if (pauli_code > 3) return MPDO_ERR_INVALID;

    /* For each site != qubit, contract with the partial-trace
     * projector vec_trace = (1, 0, 0, 1).  At site `qubit`,
     * contract with the Pauli observable's vec form:
     *   vec(P)[2r+c] = P[r, c]
     * so that Tr(rho P) = sum_{r,c} rho[r,c] * P[c,r] in matrix
     * notation, but in vec-basis: Tr(rho P) =
     *   sum_{r,c} rho_vec[2r+c] * P[c, r] = vec_P_dag . vec(rho)
     * with vec_P_dag[2r+c] = P[c, r] (the conjugate-transpose of vec).
     * For Hermitian P this equals conj(P[r, c]) (= P[c, r]).
     */
    const mpdo_complex_t* P = PAULI[pauli_code];
    /* vec(P^T) component at index 2r+c is P[c, r]. */
    mpdo_complex_t obs_vec[4];
    obs_vec[0] = P[0 * 2 + 0];   /* (r=0, c=0): P[0, 0] */
    obs_vec[1] = P[1 * 2 + 0];   /* (r=0, c=1): P[1, 0] */
    obs_vec[2] = P[0 * 2 + 1];   /* (r=1, c=0): P[0, 1] */
    obs_vec[3] = P[1 * 2 + 1];   /* (r=1, c=1): P[1, 1] */

    uint32_t Lcap = state->max_bond > 1 ? state->max_bond : 1;
    mpdo_complex_t* v   = (mpdo_complex_t*)calloc(Lcap, sizeof(mpdo_complex_t));
    mpdo_complex_t* vn  = (mpdo_complex_t*)calloc(Lcap, sizeof(mpdo_complex_t));
    if (!v || !vn) { free(v); free(vn); return MPDO_ERR_OOM; }
    v[0] = 1.0;
    for (uint32_t i = 0; i < state->n; i++) {
        uint32_t L = state->bond_dims[i];
        uint32_t R = state->bond_dims[i + 1];
        for (uint32_t r = 0; r < R; r++) vn[r] = 0.0;
        const mpdo_complex_t* T = state->tensors[i];
        const mpdo_complex_t* contractor;
        mpdo_complex_t trace_vec[4] = { 1.0, 0.0, 0.0, 1.0 };
        if (i == qubit) {
            contractor = obs_vec;
        } else {
            contractor = trace_vec;
        }
        for (uint32_t l = 0; l < L; l++) {
            const mpdo_complex_t coef = v[l];
            if (creal(coef) == 0.0 && cimag(coef) == 0.0) continue;
            for (uint32_t p = 0; p < 4; p++) {
                if (creal(contractor[p]) == 0.0 && cimag(contractor[p]) == 0.0)
                    continue;
                for (uint32_t r = 0; r < R; r++) {
                    vn[r] += coef * contractor[p] * T[l * 4 * R + p * R + r];
                }
            }
        }
        memcpy(v, vn, R * sizeof(mpdo_complex_t));
    }
    *out_expval = creal(v[0]);
    free(v); free(vn);
    return MPDO_SUCCESS;
}
