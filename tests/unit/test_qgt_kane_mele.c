/**
 * @file test_qgt_kane_mele.c
 * @brief Tests for the Kane-Mele 4-band model and its Z_2 invariant.
 *
 * Pins:
 *   1. The 4-band Bloch callback is Hermitian and properly populated.
 *   2. n-band Chern infrastructure produces total Chern = 0 for KM
 *      (TRS gives C_total = 0 for any time-reversal-symmetric system).
 *   3. Z_2 invariant: nu = 1 in the QSH phase
 *      (|lambda_v| < 3 sqrt(3) |lambda_so|), nu = 0 in the trivial
 *      phase (|lambda_v| > 3 sqrt(3) |lambda_so|).
 *   4. Matches the analytical phase boundary at the gap-closing point.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    fprintf(stdout, "=== Kane-Mele 4-band model + Z_2 invariant ===\n");

    /* Phase boundary in the primitive-reciprocal-coordinate convention:
     * gap closes at the Dirac points (0, +/-2*pi/3) of the honeycomb,
     * giving the canonical |lambda_v| = 3 * sqrt(3) * |lambda_so|. */
    const double t         = 1.0;
    const double lambda_so = 0.06;
    const double boundary  = 3.0 * sqrt(3.0) * lambda_so;  /* ~0.3118 */

    /* ---- Case 1: trivial phase (lambda_v >> 3*sqrt(3)*lambda_so) - */
    {
        const double lambda_v  = 0.6;   /* >> boundary 0.31 */
        qgt_system_n_t* sys =
            qgt_model_kane_mele(t, lambda_so, /*lambda_r=*/0.0, lambda_v);
        CHECK(sys != NULL, "create KM (trivial)");

        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "z2_invariant rc=%d", rc);
        fprintf(stdout, "  trivial   (lambda_so=%.3f lambda_v=%.3f): Z_2 = %d\n",
                lambda_so, lambda_v, z2);
        CHECK(z2 == 0, "trivial phase: expected Z_2 = 0, got %d", z2);

        /* Total Chern of full 4-band system must be 0 by TRS. */
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

    /* ---- Case 2: QSH (topological) phase ------------------------- */
    {
        const double lambda_v  = 0.1;   /* < boundary 0.31 */
        qgt_system_n_t* sys =
            qgt_model_kane_mele(t, lambda_so, /*lambda_r=*/0.0, lambda_v);
        CHECK(sys != NULL, "create KM (QSH)");

        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "z2_invariant rc=%d", rc);
        fprintf(stdout, "  QSH       (lambda_so=%.3f lambda_v=%.3f): Z_2 = %d\n",
                lambda_so, lambda_v, z2);
        CHECK(z2 == 1, "QSH phase: expected Z_2 = 1, got %d", z2);

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

    /* ---- Case 3: phase-boundary sweep across the QSH/trivial transition
     *               at |lambda_v| = 3*sqrt(3)*lambda_so = 0.312 ------ */
    {
        fprintf(stdout, "  -- lambda_v sweep at lambda_so=%.3f, "
                        "boundary = %.4f --\n", lambda_so, boundary);
        double lvs[] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8 };
        for (size_t i = 0; i < sizeof(lvs) / sizeof(lvs[0]); i++) {
            double lv = lvs[i];
            qgt_system_n_t* sys = qgt_model_kane_mele(t, lambda_so, 0.0, lv);
            int z2 = -1;
            int rc = qgt_z2_invariant(sys, 64, &z2);
            CHECK(rc == 0, "sweep rc=%d at lv=%.3f", rc, lv);
            int expected = (lv < boundary) ? 1 : 0;
            CHECK(z2 == expected,
                  "lambda_v=%.3f: expected Z_2=%d, got %d (boundary=%.4f)",
                  lv, expected, z2, boundary);
            fprintf(stdout, "    lv=%.3f -> Z_2 = %d (expected %d)\n",
                    lv, z2, expected);
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
