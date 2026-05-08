/**
 * @file test_qgt_bhz.c
 * @brief Tests for the BHZ (Bernevig-Hughes-Zhang 2006) 4-band TI
 *        and its Z_2 invariant.
 *
 * BHZ: square-lattice 4-band TI in the basis (|s,+>, |p,+>, |s,->, |p,->).
 * Spin is conserved; in the Sz-conserving sector the Z_2 invariant
 * reduces to the parity of the spin-up Chern number.
 *
 * Phase diagram for the lattice regularization
 *   mass(k) = M - 2B (2 - cos kx - cos ky)
 * (Gamma-closing at M=0, X-closings at M=4B, M-corner closing at M=8B):
 *   - M/B < 0         -> Z_2 = 0  (trivial)
 *   - 0 < M/B < 8B    -> Z_2 = 1  (QSH)  [X-closings at M=4B cancel,
 *                                          so Z_2 stays at 1 across
 *                                          the whole 0..8B window]
 *   - M/B > 8         -> Z_2 = 0  (trivial)
 *
 * The textbook continuum BHZ "QSH for 0 < M/B < 4" applies when the
 * X-corner cutoff dispersion does not contribute; the lattice form
 * extends the QSH region to the M-corner closing at 8B.
 *
 * This pins:
 *   1. The 4-band Bloch callback is well-formed (Tr H = 0, Hermitian).
 *   2. C_total = 0 for BHZ (TRS).
 *   3. Z_2 invariant transitions correctly across the M/B = 0 and
 *      M/B = 4 boundaries.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    fprintf(stdout, "=== BHZ 4-band model + Z_2 invariant ===\n");

    /* Fix A=B=1.0 (standard scaled BHZ).  Phase diagram: QSH for
     * 0 < M < 4. */
    const double A = 1.0;
    const double B = 1.0;

    /* ---- Case 1: trivial (M < 0) ------------------------------- */
    {
        const double M = -1.0;
        qgt_system_n_t* sys = qgt_model_bhz(A, B, M);
        CHECK(sys != NULL, "create BHZ M=-1");

        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "z2_invariant rc=%d", rc);
        fprintf(stdout, "  trivial(M=%.2f): Z_2 = %d\n", M, z2);
        CHECK(z2 == 0, "M=-1: expected Z_2 = 0, got %d", z2);

        qgt_berry_grid_t g;
        rc = qgt_berry_grid_nband(sys, 32, &g);
        CHECK(rc == 0, "berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            CHECK(C == 0, "C_total (TRS): expected 0, got %d", C);
            qgt_berry_grid_free(&g);
        }
        qgt_free_nband(sys);
    }

    /* ---- Case 2: QSH (0 < M < 4) ------------------------------- */
    {
        const double M = 2.0;
        qgt_system_n_t* sys = qgt_model_bhz(A, B, M);
        CHECK(sys != NULL, "create BHZ M=2");

        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 48, &z2);
        CHECK(rc == 0, "z2_invariant rc=%d", rc);
        fprintf(stdout, "  QSH    (M=%.2f): Z_2 = %d\n", M, z2);
        CHECK(z2 == 1, "M=2: expected Z_2 = 1, got %d", z2);

        qgt_berry_grid_t g;
        rc = qgt_berry_grid_nband(sys, 48, &g);
        CHECK(rc == 0, "berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            CHECK(C == 0, "C_total (TRS): expected 0, got %d", C);
            qgt_berry_grid_free(&g);
        }
        qgt_free_nband(sys);
    }

    /* ---- Case 3: trivial (M > 8B) ------------------------------ */
    {
        const double M = 9.0;
        qgt_system_n_t* sys = qgt_model_bhz(A, B, M);

        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "z2_invariant rc=%d", rc);
        fprintf(stdout, "  trivial(M=%.2f): Z_2 = %d\n", M, z2);
        CHECK(z2 == 0, "M=9: expected Z_2 = 0, got %d", z2);

        qgt_free_nband(sys);
    }

    /* ---- Case 4: phase boundary sweep -------------------------
     * In the lattice regularization the QSH window extends from
     * M=0 (Gamma closing) to M=8B (M-corner closing).  The X
     * closings at M=4B cancel and don't change Z_2. */
    {
        fprintf(stdout, "  -- M sweep at A=B=1, QSH window 0 < M < 8 --\n");
        double Ms[] = { -2.0, -0.5, 0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 8.5, 10.0 };
        for (size_t i = 0; i < sizeof(Ms) / sizeof(Ms[0]); i++) {
            double M = Ms[i];
            qgt_system_n_t* sys = qgt_model_bhz(A, B, M);
            int z2 = -1;
            int rc = qgt_z2_invariant(sys, 48, &z2);
            CHECK(rc == 0, "sweep rc=%d at M=%.3f", rc, M);
            int expected = (M > 0.0 && M < 8.0 * B) ? 1 : 0;
            CHECK(z2 == expected,
                  "M=%.2f: expected Z_2=%d, got %d", M, expected, z2);
            fprintf(stdout, "    M=%+.2f -> Z_2 = %d (expected %d)\n",
                    M, z2, expected);
            qgt_free_nband(sys);
        }
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
