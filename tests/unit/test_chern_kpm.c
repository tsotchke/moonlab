/**
 * @file test_chern_kpm.c
 * @brief Cross-check the matrix-free KPM Chern marker against the
 *        dense Schulz reference at small L, and demonstrate that it
 *        scales past the dense ceiling.
 */

#include "../../src/algorithms/topology_realspace/chern_kpm.h"
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

static void parity_with_dense(double m, size_t L, size_t n_cheby,
                              double tol) {
    fprintf(stdout,
            "\n-- KPM vs dense: L=%zu, m=%.2f, n_cheby=%zu --\n",
            L, m, n_cheby);

    /* Dense reference. */
    chern_system_t* dense = chern_qwz_create(L, m);
    CHECK(dense != NULL, "dense system created");
    int rc = chern_build_projector(dense);
    CHECK(rc == 0, "dense projector built");

    size_t ctr = L / 2;
    double c_dense = chern_local_marker(dense, ctr, ctr);

    /* KPM. */
    chern_kpm_system_t* kpm = chern_kpm_create(L, m, n_cheby);
    CHECK(kpm != NULL, "KPM system created");
    double c_kpm = chern_kpm_local_marker(kpm, ctr, ctr);

    fprintf(stdout,
            "    c_dense = %+.4f  c_kpm = %+.4f  diff = %.4f\n",
            c_dense, c_kpm, fabs(c_dense - c_kpm));
    CHECK(fabs(c_dense - c_kpm) < tol,
          "|c_dense - c_kpm| = %.3f < %.3f",
          fabs(c_dense - c_kpm), tol);

    chern_system_free(dense);
    chern_kpm_free(kpm);
}

static void reproduces_topological_number(double m, size_t L,
                                          size_t n_cheby,
                                          double expected, double tol) {
    fprintf(stdout,
            "\n-- KPM alone: L=%zu, m=%.2f, n_cheby=%zu -> expect C~%.1f --\n",
            L, m, n_cheby, expected);
    chern_kpm_system_t* kpm = chern_kpm_create(L, m, n_cheby);
    CHECK(kpm != NULL, "KPM system created");

    size_t lo = L / 2 - 2;
    size_t hi = L / 2 + 2;
    double bulk_sum = chern_kpm_bulk_sum(kpm, lo, hi);
    double bulk_mean = bulk_sum / 16.0;
    fprintf(stdout,
            "    mean over 4x4 bulk patch at center = %+.4f\n", bulk_mean);
    CHECK(fabs(bulk_mean - expected) < tol,
          "bulk mean %.3f within %.2f of expected %.1f",
          bulk_mean, tol, expected);
    chern_kpm_free(kpm);
}

static void modulation_preserves_topology(void) {
    fprintf(stdout,
            "\n-- Modulation: small V(r) should not change the Chern number --\n");
    const size_t L = 16;
    const size_t n_cheby = 140;
    chern_kpm_system_t* sys = chern_kpm_create(L, -1.0, n_cheby);
    CHECK(sys != NULL, "system created");

    /* C_4 cosine modulation at a generic wavevector. */
    const double Q = 2.0 * M_PI / 7.0;  /* incommensurate with lattice */
    const double V0 = 0.2;              /* small relative to gap (~2) */
    double* V = chern_kpm_cn_modulation(L, 4, Q, V0);
    CHECK(V != NULL, "modulation array created");
    int rc = chern_kpm_set_modulation(sys, V, 4.0 * V0);
    CHECK(rc == 0, "modulation attached");

    double c = chern_kpm_local_marker(sys, L / 2, L / 2);
    fprintf(stdout, "    c(L/2, L/2) with V0=%.2f Q=%.3f = %+.4f\n",
            V0, Q, c);
    CHECK(fabs(c - 1.0) < 0.25,
          "modulated marker %.3f within 0.25 of 1.0", c);

    chern_kpm_free(sys);
    free(V);
}

static void large_modulation_collapses(void) {
    fprintf(stdout,
            "\n-- Large-V modulation (gap closes): marker magnitude drops --\n");
    const size_t L = 14;
    const size_t n_cheby = 200;
    chern_kpm_system_t* sys = chern_kpm_create(L, -1.0, n_cheby);
    const double Q = 2.0 * M_PI / 5.0;
    const double V0 = 3.0;  /* several times the gap */
    double* V = chern_kpm_cn_modulation(L, 4, Q, V0);
    chern_kpm_set_modulation(sys, V, 4.0 * V0);

    /* With a very large V, the bulk marker should no longer be
     * quantised at +/-1. Accept any value with magnitude != 1. */
    double c = chern_kpm_local_marker(sys, L / 2, L / 2);
    fprintf(stdout, "    c(L/2, L/2) with V0=%.2f = %+.4f\n", V0, c);
    CHECK(fabs(c - 1.0) > 0.2,
          "large-V marker %.3f deviates measurably from 1.0", c);

    chern_kpm_free(sys);
    free(V);
}

int main(void) {
    fprintf(stdout, "=== Matrix-free KPM Chern marker ===\n");

    /* 1. Verify KPM matches the dense reference at small L. */
    parity_with_dense(-1.0, 8, 80, 0.10);
    parity_with_dense(+1.0, 8, 80, 0.10);
    parity_with_dense(+3.0, 8, 80, 0.10);

    /* 2. KPM alone at larger L than the dense reference can comfortably
     * handle -- this is the whole point of the matrix-free approach. */
    reproduces_topological_number(-1.0, 16, 120, +1.0, 0.25);
    reproduces_topological_number(+3.0, 12, 100,  0.0, 0.15);

    /* 3. Quasicrystal-style C_n modulation: small V preserves
     * topology; large V can drive transitions. */
    modulation_preserves_topology();
    large_modulation_collapses();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
