/**
 * @file ca_mps_var_d_stab_warmstart.c
 * @brief Implementation of the stabilizer-subgroup warmstart Clifford builder.
 *
 * Algorithm (Aaronson-Gottesman, "Improved Simulation of Stabilizer
 * Circuits", arXiv:quant-ph/0406196, generalized to non-square
 * tableaux):
 *
 *   Encode each generator as a row in a (k, 2n+1) F_2 tableau:
 *     row i = [ x_{i,0} x_{i,1} ... x_{i,n-1} | z_{i,0} ... z_{i,n-1} | r_i ]
 *   where the Pauli on qubit q is given by the symplectic pair
 *     (x_{i,q}, z_{i,q}) in {(0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y}
 *   and r_i is the phase bit (0 = +, 1 = -; the AG i^{x.z} convention
 *   handled implicitly).
 *
 *   Symplectic Gauss-Jordan (column ops emit Clifford gates, row ops
 *   are free):
 *
 *     For pivot p = 0, 1, ..., k-1:
 *       (A) Permute rows so row p has a non-I Pauli on some qubit not
 *           yet used as a pivot.  (Free; just a row reorder.)
 *       (B) Pick the pivot qubit q for row p (smallest unused qubit
 *           with non-I in row p).
 *       (C) Rotate (x_{p,q}, z_{p,q}) to (1,0)=X using H or S on
 *           qubit q (column op).
 *       (D) For every other qubit q' with non-I in row p:
 *             rotate (x_{p,q'}, z_{p,q'}) to X (H or S on q'),
 *             apply CNOT(q, q'); now row p has X only at qubit q.
 *       (E) For every other row r != p with x_{r,q} = 1, XOR row p
 *           into row r.  (Free; commutation guarantees z_{r,q} = 0.)
 *       (F) Apply H on qubit q to convert row p from X_q to Z_q.
 *
 *   After elimination row p is +/- Z_{q_p}.  The state |b> with
 *   b_{q_p} = r_p satisfies (transformed g_p) |b> = +|b> for all p.
 *
 *   The state |psi> = G_1^dagger G_2^dagger ... G_M^dagger |b>
 *   (where G_1...G_M are the column-op gates emitted in order) is
 *   stabilized by every original generator g_p.  We apply the
 *   reversed-and-daggered gate list as Clifford operations on the
 *   CA-MPS state (which absorbs them into the D factor) preceded
 *   by X gates on pivot qubits whose phase came out negative.
 *
 *   Cost: O(n^2) gate emissions, O(k * n^2) tableau operations.  For
 *   the Z2 LGT use case (k = N - 2 ~ n/2), this is well below the
 *   alternating-loop cost and runs effectively for free.
 */

#include "ca_mps_var_d_stab_warmstart.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Recorded gate type (column ops emitted during elimination).        */
/* ------------------------------------------------------------------ */

typedef enum {
    SW_GATE_H,
    SW_GATE_S,
    SW_GATE_CNOT
} sw_gate_kind_t;

typedef struct {
    sw_gate_kind_t kind;
    uint32_t q1;       /* H, S: target qubit; CNOT: control */
    uint32_t q2;       /* CNOT: target qubit; ignored for 1Q gates */
} sw_gate_t;

typedef struct {
    sw_gate_t* data;
    size_t     len;
    size_t     cap;
} sw_gate_log_t;

static int sw_log_push(sw_gate_log_t* log, sw_gate_kind_t k,
                       uint32_t q1, uint32_t q2) {
    if (log->len == log->cap) {
        size_t new_cap = log->cap == 0 ? 32 : log->cap * 2;
        sw_gate_t* p = (sw_gate_t*)realloc(log->data,
                                            new_cap * sizeof(sw_gate_t));
        if (!p) return -1;
        log->data = p;
        log->cap  = new_cap;
    }
    log->data[log->len].kind = k;
    log->data[log->len].q1   = q1;
    log->data[log->len].q2   = q2;
    log->len++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Tableau: per-row {x, z, r} flat F_2 matrix.                        */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t k;          /* number of generators */
    uint32_t n;          /* number of qubits */
    uint8_t* x;          /* (k, n) row-major; values in {0, 1} */
    uint8_t* z;          /* (k, n) row-major; values in {0, 1} */
    uint8_t* r;          /* (k,)  values in {0, 1} */
} sw_tableau_t;

static int sw_tab_alloc(sw_tableau_t* t, uint32_t k, uint32_t n) {
    t->k = k;
    t->n = n;
    t->x = (uint8_t*)calloc((size_t)k * n, 1);
    t->z = (uint8_t*)calloc((size_t)k * n, 1);
    t->r = (uint8_t*)calloc(k, 1);
    if (!t->x || !t->z || !t->r) return -1;
    return 0;
}

static void sw_tab_free(sw_tableau_t* t) {
    free(t->x); t->x = NULL;
    free(t->z); t->z = NULL;
    free(t->r); t->r = NULL;
}

static inline uint8_t* sw_xrow(sw_tableau_t* t, uint32_t i) {
    return &t->x[(size_t)i * t->n];
}
static inline uint8_t* sw_zrow(sw_tableau_t* t, uint32_t i) {
    return &t->z[(size_t)i * t->n];
}

/* ------------------------------------------------------------------ */
/* Aaronson-Gottesman conjugation rules (column ops, mutate tableau). */
/* Rules apply uniformly to every row.                                */
/* ------------------------------------------------------------------ */

static void sw_apply_h(sw_tableau_t* t, uint32_t q) {
    for (uint32_t i = 0; i < t->k; i++) {
        uint8_t* x = sw_xrow(t, i);
        uint8_t* z = sw_zrow(t, i);
        t->r[i] ^= (uint8_t)(x[q] & z[q]);
        uint8_t tmp = x[q];
        x[q] = z[q];
        z[q] = tmp;
    }
}

static void sw_apply_s(sw_tableau_t* t, uint32_t q) {
    /* S: (x, z) -> (x, x XOR z); r ^= x & z. */
    for (uint32_t i = 0; i < t->k; i++) {
        uint8_t* x = sw_xrow(t, i);
        uint8_t* z = sw_zrow(t, i);
        t->r[i] ^= (uint8_t)(x[q] & z[q]);
        z[q] ^= x[q];
    }
}

static void sw_apply_cnot(sw_tableau_t* t, uint32_t c, uint32_t tgt) {
    for (uint32_t i = 0; i < t->k; i++) {
        uint8_t* x = sw_xrow(t, i);
        uint8_t* z = sw_zrow(t, i);
        /* AG rule: r ^= x[c] & z[t] & (x[t] XOR z[c] XOR 1). */
        uint8_t bit = x[c] & z[tgt];
        bit &= (uint8_t)(x[tgt] ^ z[c] ^ 1u);
        t->r[i] ^= bit;
        x[tgt] ^= x[c];
        z[c]   ^= z[tgt];
    }
}

/* ------------------------------------------------------------------ */
/* Row XOR (multiply two stabilizer generators).  Free in gate cost.  */
/* Phase composition: if g_a has phase a, g_b has phase b, then       */
/*   g_a * g_b in the symplectic + bit form picks up an extra phase   */
/*   from the cross-term i^{x_a . z_b - z_a . x_b mod 4} -- but for   */
/*   our use case (commuting Hermitian generators with phase in       */
/*   {+1,-1} only), and given the way we use row XOR (only after the  */
/*   pivot row has been reduced to a single Pauli), the cross-term    */
/*   contribution simplifies to x_a . z_b - z_a . x_b mod 4 evaluated */
/*   when row p has only one non-zero qubit (the pivot).  We compute  */
/*   it generically using the AG i-term to stay correct for future    */
/*   non-Hermitian generator sets.                                    */
/* ------------------------------------------------------------------ */

static void sw_row_xor(sw_tableau_t* t, uint32_t dst, uint32_t src) {
    /* Phase under multiplication of two Pauli strings:
     *   sign bit r ^= r_src
     *   plus a "sum of imaginary cross-terms" from each qubit.
     * AG eq. (4.1) / Lemma in arXiv:quant-ph/0406196: define
     *   g(x1,z1,x2,z2) = number of i factors mod 4 from
     *   multiplying single-qubit Paulis (x1,z1)*(x2,z2) on one qubit.
     *   Values: g(0,0,*,*) = 0; g(1,1,1,0) = 1; g(1,0,1,1) = -1;
     *           g(0,1,1,1) = 1; g(1,1,0,1) = -1; g(1,0,0,1) = 1;
     *           g(0,1,1,0) = -1; etc.  The closed form:
     *     g = z1*x2 - x1*z2  (for x1+z1+x2+z2 != 0 with the cases
     *     where one operand is I giving 0 trivially).
     * Sum mod 4 across qubits gives total i-power from the merge;
     * we add to (2*r) and re-extract.  For our use case (commuting
     * generators), the imaginary part vanishes (sum is in {0,2})
     * and the result projects cleanly to a sign bit.
     */
    int32_t i_power = (int32_t)(2 * t->r[dst] + 2 * t->r[src]);
    uint8_t* xa = sw_xrow(t, dst);
    uint8_t* za = sw_zrow(t, dst);
    uint8_t* xb = sw_xrow(t, src);
    uint8_t* zb = sw_zrow(t, src);
    for (uint32_t q = 0; q < t->n; q++) {
        /* Per-qubit imaginary contribution = z_a*x_b - x_a*z_b. */
        i_power += (int32_t)za[q] * (int32_t)xb[q]
                 - (int32_t)xa[q] * (int32_t)zb[q];
        xa[q] ^= xb[q];
        za[q] ^= zb[q];
    }
    /* Reduce i_power mod 4; rows commute => result must be in {0,2}. */
    int32_t m = ((i_power % 4) + 4) % 4;
    t->r[dst] = (uint8_t)(m / 2);  /* 0 -> 0, 2 -> 1; 1,3 should not occur */
}

/* ------------------------------------------------------------------ */
/* Row swap (free).                                                   */
/* ------------------------------------------------------------------ */

static void sw_row_swap(sw_tableau_t* t, uint32_t a, uint32_t b) {
    if (a == b) return;
    uint8_t* xa = sw_xrow(t, a);
    uint8_t* za = sw_zrow(t, a);
    uint8_t* xb = sw_xrow(t, b);
    uint8_t* zb = sw_zrow(t, b);
    for (uint32_t q = 0; q < t->n; q++) {
        uint8_t tx = xa[q]; xa[q] = xb[q]; xb[q] = tx;
        uint8_t tz = za[q]; za[q] = zb[q]; zb[q] = tz;
    }
    uint8_t tr = t->r[a]; t->r[a] = t->r[b]; t->r[b] = tr;
}

/* ------------------------------------------------------------------ */
/* Decode {0=I, 1=X, 2=Y, 3=Z} into symplectic (x, z) bits.           */
/* ------------------------------------------------------------------ */

static inline void pauli_byte_to_xz(uint8_t b, uint8_t* x, uint8_t* z) {
    /* I=00 X=10 Y=11 Z=01 in the (x, z) convention.
     * Our byte encoding: 0=I, 1=X, 2=Y, 3=Z. */
    switch (b) {
        case 0: *x = 0; *z = 0; break;
        case 1: *x = 1; *z = 0; break;
        case 2: *x = 1; *z = 1; break;
        case 3: *x = 0; *z = 1; break;
        default: *x = 0; *z = 0; break;
    }
}

/* ------------------------------------------------------------------ */
/* Find the first qubit q (excluding the "used" set) where row r has  */
/* a non-I Pauli.  Returns t->n if none.                              */
/* ------------------------------------------------------------------ */

static uint32_t sw_first_unused_nonI(const sw_tableau_t* t, uint32_t r,
                                       const uint8_t* qubit_used) {
    const uint8_t* x = &t->x[(size_t)r * t->n];
    const uint8_t* z = &t->z[(size_t)r * t->n];
    for (uint32_t q = 0; q < t->n; q++) {
        if (!qubit_used[q] && (x[q] | z[q])) return q;
    }
    return t->n;
}

/* ------------------------------------------------------------------ */
/* Rotate the qubit-q Pauli of row p to X (1, 0) using {H, S}.         */
/* Emit the chosen gates into the log and apply them to the tableau   */
/* (which propagates the column op to all rows).                      */
/* ------------------------------------------------------------------ */

static int sw_rotate_to_X(sw_tableau_t* t, sw_gate_log_t* log,
                          uint32_t row, uint32_t q) {
    uint8_t x = t->x[(size_t)row * t->n + q];
    uint8_t z = t->z[(size_t)row * t->n + q];
    if (x == 1 && z == 0) return 0;             /* X already */
    if (x == 0 && z == 1) {                     /* Z -> X via H */
        if (sw_log_push(log, SW_GATE_H, q, 0) != 0) return -1;
        sw_apply_h(t, q);
        return 0;
    }
    if (x == 1 && z == 1) {                     /* Y -> X via S */
        /* S takes (1,1) -> (1, 0), i.e. Y -> X (with phase update). */
        if (sw_log_push(log, SW_GATE_S, q, 0) != 0) return -1;
        sw_apply_s(t, q);
        return 0;
    }
    /* Caller guarantees (x, z) != (0, 0). */
    return -1;
}

/* ------------------------------------------------------------------ */
/* Verify that all rows pairwise commute.  Returns 1 if yes, 0 if no. */
/* ------------------------------------------------------------------ */

static int sw_verify_commute(const sw_tableau_t* t) {
    for (uint32_t i = 0; i < t->k; i++) {
        for (uint32_t j = i + 1; j < t->k; j++) {
            const uint8_t* xi = &t->x[(size_t)i * t->n];
            const uint8_t* zi = &t->z[(size_t)i * t->n];
            const uint8_t* xj = &t->x[(size_t)j * t->n];
            const uint8_t* zj = &t->z[(size_t)j * t->n];
            uint32_t parity = 0;
            for (uint32_t q = 0; q < t->n; q++) {
                parity ^= (uint32_t)(xi[q] & zj[q]) ^ (uint32_t)(zi[q] & xj[q]);
            }
            if (parity & 1u) return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Apply a single recorded gate (in dagger form, since we walk the    */
/* log in reverse) to a CA-MPS state.                                 */
/* ------------------------------------------------------------------ */

static ca_mps_error_t sw_apply_dagger_to_ca_mps(moonlab_ca_mps_t* state,
                                                  const sw_gate_t* g) {
    switch (g->kind) {
        case SW_GATE_H:    return moonlab_ca_mps_h(state, g->q1);
        case SW_GATE_S:    return moonlab_ca_mps_sdag(state, g->q1);
        case SW_GATE_CNOT: return moonlab_ca_mps_cnot(state, g->q1, g->q2);
    }
    return CA_MPS_ERR_INVALID;
}

/* ------------------------------------------------------------------ */
/* Public entry point.                                                */
/* ------------------------------------------------------------------ */

ca_mps_error_t moonlab_ca_mps_apply_stab_subgroup_warmstart(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    uint32_t num_gens) {
    if (!state || !paulis || num_gens == 0) return CA_MPS_ERR_INVALID;

    uint32_t n = moonlab_ca_mps_num_qubits(state);
    if (num_gens > n) return CA_MPS_ERR_INVALID;

    sw_tableau_t tab = {0};
    if (sw_tab_alloc(&tab, num_gens, n) != 0) {
        sw_tab_free(&tab);
        return CA_MPS_ERR_OOM;
    }

    /* Decode input Pauli strings into symplectic (x, z) form. */
    for (uint32_t i = 0; i < num_gens; i++) {
        for (uint32_t q = 0; q < n; q++) {
            uint8_t x = 0, z = 0;
            pauli_byte_to_xz(paulis[(size_t)i * n + q], &x, &z);
            tab.x[(size_t)i * n + q] = x;
            tab.z[(size_t)i * n + q] = z;
        }
        tab.r[i] = 0;
    }

    if (!sw_verify_commute(&tab)) {
        sw_tab_free(&tab);
        return CA_MPS_ERR_INVALID;
    }

    sw_gate_log_t log = {0};
    uint8_t* qubit_used = (uint8_t*)calloc(n, 1);
    uint32_t* pivot_qubit = (uint32_t*)calloc(num_gens, sizeof(uint32_t));
    if (!qubit_used || !pivot_qubit) {
        free(qubit_used); free(pivot_qubit); free(log.data);
        sw_tab_free(&tab);
        return CA_MPS_ERR_OOM;
    }

    ca_mps_error_t err = CA_MPS_SUCCESS;

    for (uint32_t p = 0; p < num_gens; p++) {
        /* (A) Find a row >= p with a non-I Pauli on an unused qubit. */
        uint32_t pivot_row = num_gens;
        uint32_t pivot_q   = n;
        for (uint32_t i = p; i < num_gens; i++) {
            uint32_t q = sw_first_unused_nonI(&tab, i, qubit_used);
            if (q < n) {
                pivot_row = i;
                pivot_q   = q;
                break;
            }
        }
        if (pivot_row == num_gens) {
            /* All remaining rows are zero on unused qubits ->
             * generators not independent. */
            err = CA_MPS_ERR_INVALID;
            goto cleanup;
        }
        sw_row_swap(&tab, p, pivot_row);
        pivot_qubit[p] = pivot_q;
        qubit_used[pivot_q] = 1;

        /* (C) Rotate qubit pivot_q of row p to X. */
        if (sw_rotate_to_X(&tab, &log, p, pivot_q) != 0) {
            err = CA_MPS_ERR_OOM;
            goto cleanup;
        }

        /* (D) For every other qubit q' with non-I in row p: rotate
         * to X, then apply CNOT(pivot_q, q') to clear it. */
        for (uint32_t q = 0; q < n; q++) {
            if (q == pivot_q) continue;
            uint8_t x = tab.x[(size_t)p * n + q];
            uint8_t z = tab.z[(size_t)p * n + q];
            if ((x | z) == 0) continue;
            if (sw_rotate_to_X(&tab, &log, p, q) != 0) {
                err = CA_MPS_ERR_OOM;
                goto cleanup;
            }
            if (sw_log_push(&log, SW_GATE_CNOT, pivot_q, q) != 0) {
                err = CA_MPS_ERR_OOM;
                goto cleanup;
            }
            sw_apply_cnot(&tab, pivot_q, q);
        }

        /* (E) Eliminate column pivot_q from every other row by row XOR. */
        for (uint32_t r = 0; r < num_gens; r++) {
            if (r == p) continue;
            if (tab.x[(size_t)r * n + pivot_q] == 1) {
                sw_row_xor(&tab, r, p);
            }
        }

        /* (F) Convert qubit pivot_q from X back to Z. */
        if (sw_log_push(&log, SW_GATE_H, pivot_q, 0) != 0) {
            err = CA_MPS_ERR_OOM;
            goto cleanup;
        }
        sw_apply_h(&tab, pivot_q);
    }

    /* The tableau is now in canonical form: row p has Z on pivot
     * qubit pivot_qubit[p] and identity elsewhere, with phase bit
     * tab.r[p].  Build the preparation circuit:
     *
     *   1. Apply X on every pivot qubit whose final phase is -1.
     *      This brings |0^n> to the bit string |b> with b_q = r_p
     *      for q = pivot_qubit[p], so |b> is in the +1 eigenspace
     *      of every transformed generator.
     *
     *   2. Apply the recorded gate list in REVERSE, with each gate
     *      replaced by its inverse:
     *        H  -> H        (self-inverse)
     *        S  -> S^dagger
     *        CNOT(c,t) -> CNOT(c,t)  (self-inverse)
     */
    for (uint32_t p = 0; p < num_gens; p++) {
        if (tab.r[p]) {
            err = moonlab_ca_mps_x(state, pivot_qubit[p]);
            if (err != CA_MPS_SUCCESS) goto cleanup;
        }
    }
    for (size_t i = log.len; i-- > 0;) {
        err = sw_apply_dagger_to_ca_mps(state, &log.data[i]);
        if (err != CA_MPS_SUCCESS) goto cleanup;
    }

cleanup:
    free(qubit_used);
    free(pivot_qubit);
    free(log.data);
    sw_tab_free(&tab);
    return err;
}
