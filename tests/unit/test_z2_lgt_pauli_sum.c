/**
 * @file test_z2_lgt_pauli_sum.c
 * @brief Unit test for the 1+1D Z2 lattice gauge theory Pauli-sum builder.
 *
 * Pins:
 *   - Qubit count: N matter sites give 2*N - 1 qubits.
 *   - Term count breakdown: 2*(N-1) hopping + (N-1) electric +
 *     (#non-zero-mass) + (N-2) Gauss penalty.
 *   - Each Gauss-law operator has weight 3 in the right shape.
 *   - Coefficients have the documented signs.
 *   - Wilson-line operator runs across the right link qubits.
 */

#include "../../src/applications/hep/lattice_z2_1d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static int has_term_like(const uint8_t* paulis, const double* coeffs,
                          uint32_t T, uint32_t nq,
                          const uint8_t* target_pauli, double target_coeff,
                          double tol) {
    for (uint32_t k = 0; k < T; k++) {
        if (memcmp(paulis + (size_t)k * nq, target_pauli, nq) != 0) continue;
        double diff = coeffs[k] - target_coeff;
        if (diff < -tol || diff > tol) continue;
        return 1;
    }
    return 0;
}

int main(void) {
    fprintf(stdout, "=== Z2 LGT 1D Pauli-sum builder test ===\n");

    /* N = 4 matter sites, 7 qubits total (3 link qubits) */
    z2_lgt_config_t cfg = {
        .num_matter_sites = 4,
        .t_hop = 1.0,
        .h_link = 0.7,
        .mass = 0.5,
        .gauss_penalty = 5.0,
    };
    CHECK(z2_lgt_1d_num_qubits(&cfg) == 7, "num_qubits should be 7, got %u",
          z2_lgt_1d_num_qubits(&cfg));

    uint8_t* paulis = NULL;
    double* coeffs = NULL;
    uint32_t T = 0, nq = 0;
    int rc = z2_lgt_1d_build_pauli_sum(&cfg, &paulis, &coeffs, &T, &nq);
    CHECK(rc == 0, "build_pauli_sum returned %d", rc);
    CHECK(nq == 7, "nq mismatch: %u", nq);
    /* Expected term count for N = 4:
     *   hopping  : 2 * 3 = 6
     *   electric : 3
     *   mass     : 4 (m != 0 -> all 4 staggered terms present)
     *   gauss    : N - 2 = 2 (interior sites x = 1, 2)
     *   total    : 15
     */
    CHECK(T == 15, "term count mismatch: %u (expected 15)", T);

    /* Gauge-invariant kinetic terms (since the XYY/YYX rewrite):
     *   K_x = -(t/2) X_{2x} Y_{2x+1} Y_{2x+2}
     *       + (t/2) Y_{2x} Y_{2x+1} X_{2x+2}
     * Bond 0 -> qubits 0, 1, 2; with t_hop = 1.0, the coefficients
     * are -0.5 and +0.5. */
    uint8_t target[7];
    memset(target, 0, 7);
    target[0] = 1; target[1] = 2; target[2] = 2;     /* X Y Y */
    CHECK(has_term_like(paulis, coeffs, T, nq, target, -0.5, 1e-12),
          "missing -(t/2) * XYY on bond 0");

    memset(target, 0, 7);
    target[0] = 2; target[1] = 2; target[2] = 1;     /* Y Y X */
    CHECK(has_term_like(paulis, coeffs, T, nq, target, +0.5, 1e-12),
          "missing +(t/2) * YYX on bond 0");

    /* Term-by-term commutativity with the interior Gauss-law
     * operators is the headline correctness property of the
     * gauge-invariant rewrite.  Verify it on the kinetic term
     * acting at bond x = 0 vs G_1 = X_1 Z_2 X_3 (interior site x=1
     * -> qubits 1, 2, 3). */
    {
        const uint8_t G1[7] = { 0, 1, 3, 1, 0, 0, 0 };
        const uint8_t XYY[7] = { 1, 2, 2, 0, 0, 0, 0 };
        const uint8_t YYX[7] = { 2, 2, 1, 0, 0, 0, 0 };
        const uint8_t* hop_terms[2] = { XYY, YYX };
        for (int t = 0; t < 2; t++) {
            int parity = 0;
            for (uint32_t q = 0; q < 7; q++) {
                uint8_t a = hop_terms[t][q], b = G1[q];
                if (a == 0 || b == 0 || a == b) continue;
                parity ^= 1;
            }
            CHECK(parity == 0,
                  "kinetic term %d does NOT commute with G_1 (parity=%d)",
                  t, parity);
        }
    }

    /* Electric field: -h * Z on link 0 (qubit 1) */
    memset(target, 0, 7);
    target[1] = 3;
    CHECK(has_term_like(paulis, coeffs, T, nq, target, -0.7, 1e-12),
          "missing -h * Z on link 0");

    /* Staggered mass: matter site 0 -> +m/2 * Z; matter site 1 -> -m/2 * Z */
    memset(target, 0, 7);
    target[0] = 3;
    CHECK(has_term_like(paulis, coeffs, T, nq, target, +0.25, 1e-12),
          "missing +m/2 * Z on matter site 0");

    memset(target, 0, 7);
    target[2] = 3;
    CHECK(has_term_like(paulis, coeffs, T, nq, target, -0.25, 1e-12),
          "missing -m/2 * Z on matter site 1");

    /* Gauss-law penalty: -lambda * X_{2x-1} X_{2x+1} Z_{2x} for x = 1, 2 */
    /* x = 1: qubits 1 (X), 2 (Z), 3 (X) */
    memset(target, 0, 7);
    target[1] = 1; target[2] = 3; target[3] = 1;
    CHECK(has_term_like(paulis, coeffs, T, nq, target, -5.0, 1e-12),
          "missing -lambda * G_1");
    /* x = 2: qubits 3 (X), 4 (Z), 5 (X) */
    memset(target, 0, 7);
    target[3] = 1; target[4] = 3; target[5] = 1;
    CHECK(has_term_like(paulis, coeffs, T, nq, target, -5.0, 1e-12),
          "missing -lambda * G_2");

    /* Direct Gauss-law accessor on interior site x = 1 should give XZX
     * on qubits 1, 2, 3 with no other support. */
    uint8_t gx[7];
    rc = z2_lgt_1d_gauss_law_pauli(&cfg, 1, gx);
    CHECK(rc == 0, "gauss_law_pauli x=1 rc = %d", rc);
    CHECK(gx[1] == 1 && gx[2] == 3 && gx[3] == 1,
          "G_1 layout wrong: %u %u %u %u %u %u %u",
          gx[0], gx[1], gx[2], gx[3], gx[4], gx[5], gx[6]);
    CHECK(gx[0] == 0 && gx[4] == 0 && gx[5] == 0 && gx[6] == 0,
          "G_1 has unexpected Pauli outside its support");

    /* Wilson line spanning links 0..2 = qubits 1, 3, 5 -> Z Z Z */
    uint8_t wl[7];
    rc = z2_lgt_1d_wilson_line_pauli(&cfg, /*link_start=*/0, /*link_end=*/2, wl);
    CHECK(rc == 0, "wilson_line_pauli rc = %d", rc);
    CHECK(wl[1] == 3 && wl[3] == 3 && wl[5] == 3,
          "Wilson-line ZZZ pattern wrong");
    CHECK(wl[0] == 0 && wl[2] == 0 && wl[4] == 0 && wl[6] == 0,
          "Wilson line has Pauli on a matter qubit");

    /* Out-of-range Gauss / Wilson should reject. */
    CHECK(z2_lgt_1d_gauss_law_pauli(&cfg, 0, gx) != 0, "G_0 should be rejected (boundary)");
    CHECK(z2_lgt_1d_gauss_law_pauli(&cfg, 3, gx) != 0, "G_3 should be rejected (boundary)");
    CHECK(z2_lgt_1d_wilson_line_pauli(&cfg, 5, 6, wl) != 0,
          "Wilson line beyond chain should be rejected");

    free(paulis); free(coeffs);

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
