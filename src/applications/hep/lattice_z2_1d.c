/**
 * @file lattice_z2_1d.c
 * @brief Implementation of 1+1D Z2 lattice gauge theory Pauli-sum builder.
 */

#include "lattice_z2_1d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint32_t z2_lgt_1d_num_qubits(const z2_lgt_config_t* cfg) {
    if (!cfg || cfg->num_matter_sites < 2) return 0;
    return 2u * cfg->num_matter_sites - 1u;
}

/* ---- helpers ---------------------------------------------------------- */

/* Add a Pauli string with the given coefficient to the running term list.
 * @p pauli is a length n_qubits array; the function copies it. */
static void add_term(uint8_t* paulis_buf, double* coeffs_buf,
                      uint32_t* term_idx, uint32_t n_qubits,
                      const uint8_t* pauli, double c) {
    if (c == 0.0) return;
    memcpy(paulis_buf + (size_t)(*term_idx) * n_qubits, pauli, n_qubits);
    coeffs_buf[*term_idx] = c;
    (*term_idx)++;
}

/* Reset a Pauli string to identity. */
static void zero_pauli(uint8_t* p, uint32_t n) {
    memset(p, 0, n);
}

/* ---- main builder ----------------------------------------------------- */

int z2_lgt_1d_build_pauli_sum(const z2_lgt_config_t* cfg,
                                uint8_t** out_paulis,
                                double**  out_coeffs,
                                uint32_t* out_num_terms,
                                uint32_t* out_num_qubits) {
    if (!cfg || !out_paulis || !out_coeffs ||
        !out_num_terms || !out_num_qubits) return -1;
    if (cfg->num_matter_sites < 2) return -1;

    const uint32_t N    = cfg->num_matter_sites;
    const uint32_t nq   = 2u * N - 1u;

    /* Term-count upper bound:
     *   hopping:   2 * (N - 1)         (XYY + YYX per bond, gauge-invariant)
     *   electric:  (N - 1)             (Z on each link)
     *   mass:      N                   (Z on each matter site)
     *   gauss:     2 * (N - 2)         (lambda * I + (-lambda) * G_x)
     *   constant:  1                   (the lambda * sum_x I term, lumped)
     * Allocate generously and trim by counting. */
    const uint32_t cap = 2u * (N - 1u) + (N - 1u) + N + 2u * (N >= 2 ? (N - 2u) : 0u) + 1u;

    uint8_t* paulis = (uint8_t*)calloc((size_t)cap * nq, 1);
    double*  coeffs = (double*)calloc(cap, sizeof(double));
    if (!paulis || !coeffs) {
        free(paulis); free(coeffs);
        return -2;
    }

    uint8_t* p = (uint8_t*)calloc(nq, 1);
    if (!p) { free(paulis); free(coeffs); return -2; }

    uint32_t k = 0;

    /* --- matter hopping with parallel transport --- */
    /*
     * Gauge-invariant matter hopping with the link operator U = X
     * absorbed into the JW expression.  Earlier Moonlab releases used
     *   X_{2x} X_{2x+1} X_{2x+2} + Y_{2x} X_{2x+1} Y_{2x+2}
     * which is the bare JW form; both pieces anti-commute with G_x =
     * X_{2x-1} Z_{2x} X_{2x+1} (the Z-X overlap at qubit 2x flips
     * parity once -> odd -> anti-commute), so the lambda penalty was
     * enforcing gauge invariance only energetically.
     *
     * Inserting the Z2 gauge-link operator U_{2x+1} = X_{2x+1} into
     * the JW expression and combining with the JW string Z_{2x+1}
     * gives X * Z = -i Y on the link qubit, and the combined hop
     * (after taking the Hermitian h.c. partner) reduces to
     *
     *   K_x = (1/2) * [X_{2x} Y_{2x+1} Y_{2x+2} - Y_{2x} Y_{2x+1} X_{2x+2}]
     *
     * Each piece commutes with G_x and G_{x+1} term-by-term:
     *   X_{2x} Y_{2x+1} Y_{2x+2}  vs G_x = X Z X (qubits 2x-1, 2x, 2x+1):
     *     qubit 2x:   X vs Z -> anti
     *     qubit 2x+1: Y vs X -> anti
     *     -> 2 anti = even = commute.
     *   Same operator vs G_{x+1} = X Z X (qubits 2x+1, 2x+2, 2x+3):
     *     qubit 2x+1: Y vs X -> anti
     *     qubit 2x+2: Y vs Z -> anti
     *     -> 2 anti = even = commute.
     * Y X Y form is symmetric under x <-> x+1 and gives the same
     * commutativity.  Verified by tests/unit/test_z2_lgt_pauli_sum.c
     * (commutativity check) and tests/unit/test_gauge_warmstart.c
     * (post-warmstart-after-evolution Gauss-law violation stays at
     * machine zero for N=4).
     *
     * Coefficient: -t/2 per piece so the resulting hop matrix
     * element <x|H|x+1> = -t (matching the bare JW + gauge link
     * expansion -- we factor out the 1/2 from the symmetrisation).
     */
    for (uint32_t x = 0; x + 1 < N; x++) {
        const uint32_t qm0 = 2 * x;
        const uint32_t ql  = 2 * x + 1;
        const uint32_t qm1 = 2 * x + 2;

        zero_pauli(p, nq);
        p[qm0] = 1; p[ql] = 2; p[qm1] = 2;          /* XYY */
        add_term(paulis, coeffs, &k, nq, p, -0.5 * cfg->t_hop);

        zero_pauli(p, nq);
        p[qm0] = 2; p[ql] = 2; p[qm1] = 1;          /* YYX */
        add_term(paulis, coeffs, &k, nq, p, +0.5 * cfg->t_hop);
    }

    /* --- electric field on each link --- */
    /* -h * Z_{2x+1} for each link x = 0..N-2 */
    for (uint32_t x = 0; x + 1 < N; x++) {
        zero_pauli(p, nq);
        p[2 * x + 1] = 3;
        add_term(paulis, coeffs, &k, nq, p, -cfg->h_link);
    }

    /* --- staggered mass --- */
    /* (m/2) * sum_x (-1)^x Z_{2x} -- the constant (-1)^x * m/2 piece is a
     * scalar shift we drop; only the Z-on-matter contributions enter. */
    for (uint32_t x = 0; x < N; x++) {
        zero_pauli(p, nq);
        p[2 * x] = 3;
        double sign = (x % 2 == 0) ? +1.0 : -1.0;
        add_term(paulis, coeffs, &k, nq, p, 0.5 * cfg->mass * sign);
    }

    /* --- Gauss-law penalty on interior sites x = 1..N-2 --- */
    /* H_gauss = lambda * sum_x (I - G_x) where G_x = X_{2x-1} X_{2x+1} Z_{2x}.
     * The constant (lambda * (N - 2)) is dropped; we add -lambda * G_x. */
    if (cfg->gauss_penalty != 0.0 && N >= 3) {
        for (uint32_t x = 1; x + 1 < N; x++) {
            zero_pauli(p, nq);
            p[2 * x - 1] = 1;   /* X on left link */
            p[2 * x]     = 3;   /* Z on matter   */
            p[2 * x + 1] = 1;   /* X on right link */
            add_term(paulis, coeffs, &k, nq, p, -cfg->gauss_penalty);
        }
    }

    free(p);
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = k;
    *out_num_qubits = nq;
    return 0;
}

int z2_lgt_1d_gauss_law_pauli(const z2_lgt_config_t* cfg,
                                 uint32_t site_x,
                                 uint8_t* out_pauli) {
    if (!cfg || !out_pauli) return -1;
    const uint32_t N = cfg->num_matter_sites;
    if (N < 3) return -1;
    if (site_x < 1 || site_x + 1 >= N) return -1;
    const uint32_t nq = 2u * N - 1u;
    memset(out_pauli, 0, nq);
    out_pauli[2 * site_x - 1] = 1;   /* X */
    out_pauli[2 * site_x]     = 3;   /* Z */
    out_pauli[2 * site_x + 1] = 1;   /* X */
    return 0;
}

int z2_lgt_1d_wilson_line_pauli(const z2_lgt_config_t* cfg,
                                  uint32_t link_start,
                                  uint32_t link_end,
                                  uint8_t* out_pauli) {
    if (!cfg || !out_pauli) return -1;
    const uint32_t N = cfg->num_matter_sites;
    if (N < 2) return -1;
    if (link_start > link_end || link_end + 1 >= N) return -1;
    const uint32_t nq = 2u * N - 1u;
    memset(out_pauli, 0, nq);
    for (uint32_t l = link_start; l <= link_end; l++) {
        out_pauli[2 * l + 1] = 3;   /* Z on each link */
    }
    return 0;
}
