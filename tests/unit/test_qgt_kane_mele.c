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

static double max_spin_mixing(const qgt_complex_t H[16]) {
    const int up[2] = {0, 1};
    const int dn[2] = {2, 3};
    double maxv = 0.0;
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            double v1 = cabs(H[up[a] * 4 + dn[b]]);
            double v2 = cabs(H[dn[a] * 4 + up[b]]);
            if (v1 > maxv) maxv = v1;
            if (v2 > maxv) maxv = v2;
        }
    }
    return maxv;
}

static void check_hermitian(const qgt_complex_t H[16], const char* label) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double err = cabs(H[i * 4 + j] - conj(H[j * 4 + i]));
            CHECK(err < 1e-10, "%s: H[%d,%d] Hermiticity err %.3e",
                  label, i, j, err);
        }
    }
}

static void check_time_reversal(const qgt_complex_t Hk[16],
                                const qgt_complex_t Hmk[16],
                                const char* label) {
    const int flip[4] = {2, 3, 0, 1};
    const double eta[4] = {+1.0, +1.0, -1.0, -1.0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            qgt_complex_t transformed =
                eta[flip[i]] * eta[flip[j]] *
                conj(Hmk[flip[i] * 4 + flip[j]]);
            double err = cabs(Hk[i * 4 + j] - transformed);
            CHECK(err < 1e-10, "%s: TRS err H[%d,%d] %.3e",
                  label, i, j, err);
        }
    }
}

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

    /* ---- Case 4: Rashba SOC is physically present and TRS ---------- */
    {
        const double lambda_v = 0.1;
        const double lambda_r = 0.05;
        const double k[2] = {0.41, -0.29};
        const double mk[2] = {-k[0], -k[1]};
        qgt_complex_t H[16], Hm[16];
        qgt_system_n_t* sys =
            qgt_model_kane_mele(t, lambda_so, lambda_r, lambda_v);
        CHECK(sys != NULL, "create KM (Rashba QSH)");

        int rc = qgt_eval_nband_hamiltonian(sys, k, H, 16);
        CHECK(rc == 0, "eval Rashba H(k) rc=%d", rc);
        rc = qgt_eval_nband_hamiltonian(sys, mk, Hm, 16);
        CHECK(rc == 0, "eval Rashba H(-k) rc=%d", rc);
        check_hermitian(H, "Rashba H(k)");
        check_time_reversal(H, Hm, "Rashba Kane-Mele");
        CHECK(max_spin_mixing(H) > 1e-3,
              "Rashba term should mix spin sectors");

        int z2 = -1;
        rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "Rashba z2 rc=%d", rc);
        CHECK(z2 == 1, "Rashba QSH phase: expected Z_2 = 1, got %d", z2);

        qgt_berry_grid_t g;
        rc = qgt_berry_grid_nband(sys, 32, &g);
        CHECK(rc == 0, "Rashba berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            CHECK(C == 0, "Rashba C_total (TRS): expected 0, got %d", C);
            qgt_berry_grid_free(&g);
        }
        qgt_free_nband(sys);
    }

    /* ---- Case 5: Rashba SOC in a trivial gapped phase -------------- */
    {
        const double lambda_v = 0.6;
        const double lambda_r = 0.05;
        qgt_system_n_t* sys =
            qgt_model_kane_mele(t, lambda_so, lambda_r, lambda_v);
        CHECK(sys != NULL, "create KM (Rashba trivial)");
        int z2 = -1;
        int rc = qgt_z2_invariant(sys, 32, &z2);
        CHECK(rc == 0, "Rashba trivial z2 rc=%d", rc);
        CHECK(z2 == 0, "Rashba trivial phase: expected Z_2 = 0, got %d", z2);
        qgt_free_nband(sys);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
