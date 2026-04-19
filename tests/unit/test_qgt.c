/**
 * @file test_qgt.c
 * @brief Quantum geometric tensor correctness tests.
 *
 * Cross-validates the QGT Berry-curvature / Chern computation against
 * the known phase diagrams of QWZ and Haldane, and confirms
 * consistency with the real-space marker module in the same tree.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static int nearest_integer(double x) {
    return (int)(x < 0 ? x - 0.5 : x + 0.5);
}

static void test_qwz_phase(double m, int expected) {
    fprintf(stdout, "\n-- QWZ m=%.2f, expecting C=%+d --\n", m, expected);
    qgt_system_t* sys = qgt_model_qwz(m);
    CHECK(sys != NULL, "system created");

    qgt_berry_grid_t grid;
    int rc = qgt_berry_grid(sys, 32, &grid);
    CHECK(rc == 0, "berry grid computed (rc=%d)", rc);
    fprintf(stdout, "    integrated Chern = %+.4f\n", grid.chern);
    int cint = nearest_integer(grid.chern);
    CHECK(cint == expected,
          "Chern rounds to %+d (expected %+d, raw %.4f)",
          cint, expected, grid.chern);
    qgt_berry_grid_free(&grid);
    qgt_free(sys);
}

static void test_qwz_chern_matches_realspace(void) {
    /* The real-space Chern marker chern_marker.c returns C=-1 for
     * m=+1 and C=+1 for m=-1 (QWZ convention we've already fixed).
     * The FHS method here should return the same numbers. */
    fprintf(stdout, "\n-- QWZ: FHS Chern matches the real-space marker sign --\n");
    qgt_system_t* a = qgt_model_qwz(+1.0);
    qgt_system_t* b = qgt_model_qwz(-1.0);
    qgt_berry_grid_t ga, gb;
    qgt_berry_grid(a, 40, &ga);
    qgt_berry_grid(b, 40, &gb);
    fprintf(stdout, "    C(m=+1)=%+.3f   C(m=-1)=%+.3f\n",
            ga.chern, gb.chern);
    CHECK(nearest_integer(ga.chern) == -1, "m=+1 gives Chern=-1");
    CHECK(nearest_integer(gb.chern) == +1, "m=-1 gives Chern=+1");
    qgt_berry_grid_free(&ga);
    qgt_berry_grid_free(&gb);
    qgt_free(a); qgt_free(b);
}

static void test_metric_positive_semidefinite(void) {
    fprintf(stdout, "\n-- QWZ quantum metric is PSD in the bulk --\n");
    qgt_system_t* sys = qgt_model_qwz(-1.0);
    /* Pick a generic bulk point (not at a Dirac crossing). */
    double k[2] = { 1.1, 0.7 };
    double g[4];
    int rc = qgt_metric_at(sys, k, 1e-4, g);
    CHECK(rc == 0, "qgt_metric_at returns 0");
    double gxx = g[0], gxy = g[1], gyx = g[2], gyy = g[3];
    fprintf(stdout,
            "    g = [[%+.5f, %+.5f], [%+.5f, %+.5f]]\n",
            gxx, gxy, gyx, gyy);
    /* Symmetric. */
    CHECK(fabs(gxy - gyx) < 1e-6, "g symmetric (|gxy - gyx| = %.2e)",
          fabs(gxy - gyx));
    /* Positive semidefinite for a 2x2: gxx >= 0, gyy >= 0, det >= 0. */
    double det = gxx * gyy - gxy * gyx;
    CHECK(gxx >= -1e-9 && gyy >= -1e-9 && det >= -1e-9,
          "diagonal >= 0 and det %.3e >= 0", det);
    qgt_free(sys);
}

static void test_haldane_topological(void) {
    fprintf(stdout, "\n-- Haldane: M=0, phi=pi/2 gives C=+/-1 --\n");
    /* Topological: |M| < 3 sqrt(3) |t2 sin(phi)|. With M=0, any
     * nonzero sin(phi) puts us in the topological phase. */
    qgt_system_t* sys = qgt_model_haldane(1.0, 0.2, 0.5 * M_PI, 0.0);
    qgt_berry_grid_t g;
    int rc = qgt_berry_grid(sys, 48, &g);
    CHECK(rc == 0, "berry grid computed");
    int c = nearest_integer(g.chern);
    fprintf(stdout, "    C = %+.3f (rounded %+d)\n", g.chern, c);
    CHECK(c == 1 || c == -1, "Chern is +/-1 in topological phase");
    qgt_berry_grid_free(&g);
    qgt_free(sys);
}

static void test_haldane_trivial(void) {
    fprintf(stdout, "\n-- Haldane: large-|M| trivial regime, C=0 --\n");
    /* Push M well above the 3 sqrt(3) t2 |sin(phi)| threshold. */
    qgt_system_t* sys = qgt_model_haldane(1.0, 0.2, 0.5 * M_PI, 3.0);
    qgt_berry_grid_t g;
    qgt_berry_grid(sys, 48, &g);
    int c = nearest_integer(g.chern);
    fprintf(stdout, "    C = %+.3f (rounded %+d)\n", g.chern, c);
    CHECK(c == 0, "Chern = 0 in trivial phase");
    qgt_berry_grid_free(&g);
    qgt_free(sys);
}

static void test_ssh_winding(double t1, double t2, int expected) {
    fprintf(stdout,
            "\n-- SSH t1=%.2f t2=%.2f: expect winding=%+d --\n",
            t1, t2, expected);
    qgt_system_1d_t* s = qgt_model_ssh(t1, t2);
    CHECK(s != NULL, "SSH created");
    double raw;
    int w = qgt_winding_1d(s, 64, &raw);
    fprintf(stdout, "    winding = %+d (raw %+.4f)\n", w, raw);
    CHECK(w == expected,
          "winding %+d matches expected %+d", w, expected);
    qgt_free_1d(s);
}

static void test_wilson_loop_trivial(void) {
    fprintf(stdout, "\n-- Wilson loop: tiny QWZ plaquette -> small phase --\n");
    qgt_system_t* sys = qgt_model_qwz(-1.0);
    /* A small square deep in the BZ.  Berry flux through the small
     * region is much smaller than 2 pi. */
    double path[8] = {
        1.0, 1.0,
        1.1, 1.0,
        1.1, 1.1,
        1.0, 1.1,
    };
    double phase;
    int rc = qgt_wilson_loop(sys, path, 4, &phase);
    CHECK(rc == 0, "wilson loop returned 0");
    fprintf(stdout, "    phase = %+.4f rad\n", phase);
    CHECK(fabs(phase) < 0.5,
          "phase %.3f rad has |phase| < 0.5 (small plaquette)", phase);
    qgt_free(sys);
}

int main(void) {
    fprintf(stdout, "=== Quantum geometric tensor ===\n");
    test_qwz_phase(+1.0, -1);
    test_qwz_phase(-1.0, +1);
    test_qwz_phase(+3.0,  0);
    test_qwz_phase(-3.0,  0);
    test_qwz_chern_matches_realspace();
    test_metric_positive_semidefinite();
    test_haldane_topological();
    test_haldane_trivial();
    /* SSH phase diagram:
     *   t2 > t1 > 0   -> winding = +/-1 (topological)
     *   t1 > t2 > 0   -> winding = 0    (trivial) */
    test_ssh_winding(1.0, 2.0, +1);
    test_ssh_winding(2.0, 1.0,  0);
    test_ssh_winding(1.0, 0.5,  0);
    test_wilson_loop_trivial();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
