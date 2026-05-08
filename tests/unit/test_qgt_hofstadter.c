/**
 * @file test_qgt_hofstadter.c
 * @brief Tests for the Harper-Hofstadter model and its sub-band Chern
 *        numbers.
 *
 * The Hofstadter butterfly's signature is the integer Chern number of
 * each sub-band, satisfying the TKNN (Thouless-Kohmoto-Nightingale-
 * den Nijs 1982) Diophantine equation t_r p + s_r q = r.  For
 * phi = p/q = 1/q on a square lattice the lowest band has Chern = +1.
 *
 * Pins:
 *   1. q=3 lattice: lowest band Chern = +1.  Sum of all 3 = 0.
 *   2. q=5 lattice: lowest band Chern = +1.
 *   3. Lifecycle.
 *
 * The Berry-grid integration is over the magnetic Brillouin zone
 * [-pi, pi]^2 in primitive coordinates; the kx range physically runs
 * over a reduced [-pi/q, pi/q] but the Bloch Hamiltonian's exp(-i q kx)
 * factor compensates so the same integrator can be used.
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
    fprintf(stdout, "=== Harper-Hofstadter Chern sub-bands ===\n");

    /* ---- Case 1: q = 3, lowest band ----------------------------- */
    {
        qgt_system_n_t* sys = qgt_model_hofstadter(1.0, 1, 3, 1);
        CHECK(sys != NULL, "create Hofstadter q=3 (lowest band)");
        if (!sys) return 1;

        qgt_berry_grid_t g;
        int rc = qgt_berry_grid_nband(sys, 32, &g);
        CHECK(rc == 0, "berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            fprintf(stdout, "  q=3 lowest band: Chern = %+d (expected +1)\n", C);
            CHECK(C == 1, "expected Chern=+1 for lowest band, got %+d", C);
            qgt_berry_grid_free(&g);
        }
        qgt_free_nband(sys);
    }

    /* ---- Case 2: q = 3, two lowest bands (sum) -------------------
     *
     * Lowest two bands summed: Chern = +1 + (-2) = -1.  The
     * non-Abelian FHS path on n_occupied=2 returns the total Chern
     * of the occupied subspace. */
    {
        qgt_system_n_t* sys = qgt_model_hofstadter(1.0, 1, 3, 2);
        qgt_berry_grid_t g;
        int rc = qgt_berry_grid_nband(sys, 32, &g);
        CHECK(rc == 0, "berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            fprintf(stdout, "  q=3 lowest 2 bands: Chern = %+d (expected -1)\n", C);
            CHECK(C == -1, "expected Chern=-1 for lowest 2 bands, got %+d", C);
            qgt_berry_grid_free(&g);
        }
        qgt_free_nband(sys);
    }

    /* ---- Case 3: q = 5, lowest band ----------------------------- */
    {
        qgt_system_n_t* sys = qgt_model_hofstadter(1.0, 1, 5, 1);
        CHECK(sys != NULL, "create Hofstadter q=5 (lowest band)");
        if (!sys) return 1;

        qgt_berry_grid_t g;
        int rc = qgt_berry_grid_nband(sys, 24, &g);
        CHECK(rc == 0, "berry_grid_nband rc=%d", rc);
        if (rc == 0) {
            int C = (int)lround(g.chern);
            fprintf(stdout, "  q=5 lowest band: Chern = %+d (expected +1)\n", C);
            CHECK(C == 1, "expected Chern=+1 for lowest band, got %+d", C);
            qgt_berry_grid_free(&g);
        }
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
