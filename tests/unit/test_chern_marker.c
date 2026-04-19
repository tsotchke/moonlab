/**
 * @file test_chern_marker.c
 * @brief Verify the dense local Chern marker reproduces known
 *        topological invariants on the Qi-Wu-Zhang model.
 *
 * QWZ phase diagram (2D, half-filling, 2 bands):
 *   -2 < m < 0   -> Chern = +/-1 (topological)
 *    0 < m < 2   -> Chern = +/-1 (opposite sign)
 *   |m|  > 2     -> Chern =  0   (trivial insulator)
 *
 * On an open boundary lattice of size L x L the bulk local Chern
 * marker should converge to the Chern number at sites away from the
 * boundary. We check |c(r) - C_expected| < 0.1 at the lattice center
 * for a modest L and confirm the trivial phase reads ~0.
 */

#include "../../src/algorithms/topology_realspace/chern_marker.h"

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

static double center_marker(chern_system_t* sys) {
    size_t ctr = sys->L / 2;
    return chern_local_marker(sys, ctr, ctr);
}

static void run_phase(const char* label, double m, double expected,
                      double tol, size_t L) {
    fprintf(stdout, "\n-- %s: QWZ m=%.2f on L=%zu, expecting C~%.1f --\n",
            label, m, L, expected);
    chern_system_t* sys = chern_qwz_create(L, m);
    CHECK(sys != NULL, "system created");
    int rc = chern_build_projector(sys);
    CHECK(rc == 0, "projector built (rc=%d)", rc);

    double cr = center_marker(sys);
    fprintf(stdout, "    c(L/2, L/2) = %+.4f\n", cr);
    CHECK(fabs(cr - expected) < tol,
          "center marker %.3f within %.2f of expected %.1f",
          cr, tol, expected);

    /* Also look at a 4x4 bulk patch centered at L/2, L/2.  The mean
     * should converge even faster than a single site. */
    size_t lo = L / 2 - 2, hi = L / 2 + 2;
    double bulk_sum = chern_bulk_sum(sys, lo, hi);
    double bulk_mean = bulk_sum / 16.0;
    fprintf(stdout, "    mean over 4x4 bulk patch = %+.4f\n", bulk_mean);
    CHECK(fabs(bulk_mean - expected) < tol * 0.7,
          "bulk-patch mean %.3f within %.2f of expected %.1f",
          bulk_mean, tol * 0.7, expected);

    chern_system_free(sys);
}

int main(void) {
    fprintf(stdout, "=== Local Chern marker (QWZ dense reference) ===\n");
    run_phase("topological m=+1",      1.0,  -1.0, 0.2, 14);
    run_phase("topological m=-1",     -1.0,  +1.0, 0.2, 14);
    run_phase("trivial m=+3",          3.0,   0.0, 0.1, 10);
    run_phase("trivial m=-3",         -3.0,   0.0, 0.1, 10);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
