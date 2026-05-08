/**
 * @file test_qgt_integrators.c
 * @brief Verify that the three Berry-grid integrators agree on QWZ.
 *
 * The original eigvec-FHS path (qgt_berry_grid), the parallel-
 * transport-gauge path (qgt_berry_grid_pt), and the projector-trace
 * path (qgt_berry_grid_proj) should all give the same integer
 * Chern number for any gapped 2-band Bloch Hamiltonian.  The QWZ
 * model has a clean phase diagram:
 *   m in (-2, 0) -> C = +1
 *   m in (0, +2) -> C = -1
 *   |m| > 2     -> C = 0
 * (sign convention determined by the existing qgt_berry_grid path
 * which the other two are calibrated against).
 *
 * This test pins that all three integrators return the same Chern
 * across the QWZ phase diagram.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

static int chern_match(qgt_system_t* sys, size_t N) {
    qgt_berry_grid_t g_old, g_pt, g_proj;
    int rc1 = qgt_berry_grid     (sys, N, &g_old);
    int rc2 = qgt_berry_grid_pt  (sys, N, &g_pt);
    int rc3 = qgt_berry_grid_proj(sys, N, &g_proj);
    int c_old  = (int)lround(g_old.chern);
    int c_pt   = (int)lround(g_pt.chern);
    int c_proj = (int)lround(g_proj.chern);
    fprintf(stdout, "    old=%+.3f (rounded %+d), pt=%+.3f (%+d), proj=%+.3f (%+d)\n",
            g_old.chern, c_old, g_pt.chern, c_pt, g_proj.chern, c_proj);
    qgt_berry_grid_free(&g_old);
    qgt_berry_grid_free(&g_pt);
    qgt_berry_grid_free(&g_proj);
    if (rc1 || rc2 || rc3) return -1;
    return (c_old == c_pt && c_pt == c_proj) ? c_old : -99999;
}

int main(void) {
    fprintf(stdout, "=== QGT integrator agreement (QWZ phase diagram) ===\n");

    /* QWZ topological points */
    {
        fprintf(stdout, "\n-- m=-1.5 (topological, C=+1) --\n");
        qgt_system_t* sys = qgt_model_qwz(-1.5);
        int c = chern_match(sys, 48);
        CHECK(c == 1, "all three integrators agree at +1, got %d", c);
        qgt_free(sys);
    }
    {
        fprintf(stdout, "\n-- m=+0.5 (topological, C=-1) --\n");
        qgt_system_t* sys = qgt_model_qwz(0.5);
        int c = chern_match(sys, 48);
        CHECK(c == -1, "all three integrators agree at -1, got %d", c);
        qgt_free(sys);
    }
    {
        fprintf(stdout, "\n-- m=+1.0 (topological, C=-1) --\n");
        qgt_system_t* sys = qgt_model_qwz(1.0);
        int c = chern_match(sys, 48);
        CHECK(c == -1, "all three integrators agree at -1, got %d", c);
        qgt_free(sys);
    }

    /* QWZ trivial points */
    {
        fprintf(stdout, "\n-- m=+3.0 (trivial, C=0) --\n");
        qgt_system_t* sys = qgt_model_qwz(3.0);
        int c = chern_match(sys, 48);
        CHECK(c == 0, "all three integrators agree at 0, got %d", c);
        qgt_free(sys);
    }
    {
        fprintf(stdout, "\n-- m=-3.0 (trivial, C=0) --\n");
        qgt_system_t* sys = qgt_model_qwz(-3.0);
        int c = chern_match(sys, 48);
        CHECK(c == 0, "all three integrators agree at 0, got %d", c);
        qgt_free(sys);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
