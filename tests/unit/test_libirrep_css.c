/**
 * @file  test_libirrep_css.c
 * @brief Validate the CSS-code bridge across the QEC zoo.
 *
 * v0.6.1: surface code d = 3, 5.
 * v0.6.2: toric code, Steane color, Hamming [[15,7,3]], IBM BB
 *         [[72,12,6]] / [[144,12,12]] / [[288,12,18]], hypergraph
 *         product [[13,1,3]] / [[25,1,4]] / [[41,1,5]].
 *
 * Each instance is validated against its published [[n, k, d]]
 * shape and (where tractable) brute-force distance.
 *
 * Built only when QSIM_ENABLE_LIBIRREP=ON; otherwise exits 77 (CTest
 * SKIP) so default configurations stay green.
 */

#include "../../src/integration/libirrep_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

/* Verify a CSS handle reports the expected (n, m_X, m_Z, k) shape and
 * optionally cross-check the distance via brute force.  Each entry in
 * the QEC zoo collapses to one call of this helper. */
static void check_shape(moonlab_libirrep_qec_t *q, const char *label,
                        int expect_n, int expect_m_X, int expect_m_Z,
                        int expect_k, int expect_d, int run_distance_check)
{
    if (!q) { CHECK(0, "%s: handle is NULL", label); return; }
    const int n  = moonlab_libirrep_qec_n_qubits(q);
    const int mX = moonlab_libirrep_qec_n_x_stabs(q);
    const int mZ = moonlab_libirrep_qec_n_z_stabs(q);
    const int k  = moonlab_libirrep_qec_logical_qubits(q);
    fprintf(stdout, "    %s: n=%d  m_X=%d  m_Z=%d  k=%d  [[%d,%d,%d]] expected\n",
            label, n, mX, mZ, k, expect_n, expect_k, expect_d);
    CHECK(n == expect_n,  "%s: n = %d (expected %d)", label, n, expect_n);
    CHECK(mX == expect_m_X, "%s: m_X = %d (expected %d)", label, mX, expect_m_X);
    CHECK(mZ == expect_m_Z, "%s: m_Z = %d (expected %d)", label, mZ, expect_m_Z);
    CHECK(k == expect_k,   "%s: k = %d (expected %d)", label, k, expect_k);
    if (run_distance_check) {
        const int d = moonlab_libirrep_qec_distance(q);
        CHECK(d == expect_d, "%s: distance = %d (expected %d)", label, d, expect_d);
    }
}

static void exercise_surface(int d, int expect_m_X, int expect_m_Z,
                             int run_distance_check)
{
    fprintf(stdout, "\n--- distance d = %d surface code ---\n", d);
    moonlab_libirrep_qec_t *q = NULL;
    const int rc = moonlab_libirrep_surface_code_new(d, &q);
    CHECK(rc == 0 && q, "moonlab_libirrep_surface_code_new(%d) rc=%d", d, rc);
    if (rc != 0) return;

    const int n = moonlab_libirrep_qec_n_qubits(q);
    const int mX = moonlab_libirrep_qec_n_x_stabs(q);
    const int mZ = moonlab_libirrep_qec_n_z_stabs(q);
    const int k  = moonlab_libirrep_qec_logical_qubits(q);
    CHECK(n == d * d, "n = %d (expected d^2 = %d)", n, d * d);
    CHECK(mX == expect_m_X, "m_X = %d (expected %d)", mX, expect_m_X);
    CHECK(mZ == expect_m_Z, "m_Z = %d (expected %d)", mZ, expect_m_Z);
    CHECK(k == 1, "k = %d (expected 1)", k);

    /* Sanity-check first X-row + first Z-row: support size should be
     * 2 (boundary) or 4 (bulk).  No row should be all-zeros. */
    unsigned char support[64];
    memset(support, 0, sizeof(support));
    int rrc = moonlab_libirrep_qec_get_x_check_row(q, 0, support);
    CHECK(rrc == 0, "get_x_check_row(0) rc=%d", rrc);
    int x0_weight = 0;
    for (int i = 0; i < n; i++) x0_weight += support[i];
    CHECK(x0_weight == 2 || x0_weight == 4,
          "X-row 0 weight = %d (expect 2 or 4)", x0_weight);

    memset(support, 0, sizeof(support));
    rrc = moonlab_libirrep_qec_get_z_check_row(q, 0, support);
    CHECK(rrc == 0, "get_z_check_row(0) rc=%d", rrc);
    int z0_weight = 0;
    for (int i = 0; i < n; i++) z0_weight += support[i];
    CHECK(z0_weight == 2 || z0_weight == 4,
          "Z-row 0 weight = %d (expect 2 or 4)", z0_weight);

    /* Bounds check: out-of-range row indices should fail cleanly. */
    rrc = moonlab_libirrep_qec_get_x_check_row(q, mX, support);
    CHECK(rrc != 0, "get_x_check_row(out-of-range) rejected with rc=%d", rrc);

    if (run_distance_check) {
        const int dist = moonlab_libirrep_qec_distance(q);
        CHECK(dist == d, "code distance = %d (expected %d)", dist, d);
    } else {
        fprintf(stdout, "  --    skipped distance brute-force (n=%d too large)\n", n);
    }

    moonlab_libirrep_qec_free(q);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== libirrep CSS-code bridge: QEC zoo ===\n");

    if (!moonlab_libirrep_available()) {
        fprintf(stdout, "  libirrep not linked -- test skipped (CTest exit 77).\n");
        return 77;
    }

    /* --- Surface code (v0.6.1 -- d=3 brute-force, d=5 structural) --- */
    exercise_surface(/*d=*/3, /*m_X=*/4, /*m_Z=*/4, /*distance=*/1);
    exercise_surface(/*d=*/5, /*m_X=*/12, /*m_Z=*/12, /*distance=*/0);

    /* --- Toric code (v0.6.2) --- */
    fprintf(stdout, "\n--- toric code (Kitaev, periodic) ---\n");
    {
        moonlab_libirrep_qec_t *q = NULL;
        /* L=3 torus: n = 2 * 3 * 3 = 18, m_X = m_Z = 9, k = 2, d = 3. */
        const int rc = moonlab_libirrep_toric_code_new(3, 3, &q);
        CHECK(rc == 0, "toric_code_new(3, 3) rc=%d", rc);
        check_shape(q, "toric L=3", 18, 9, 9, 2, 3, /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        /* L=4 torus: n = 32, m_X = m_Z = 16, k = 2, d = 4. */
        q = NULL;
        const int rc2 = moonlab_libirrep_toric_code_new(4, 4, &q);
        CHECK(rc2 == 0, "toric_code_new(4, 4) rc=%d", rc2);
        check_shape(q, "toric L=4", 32, 16, 16, 2, 4, /*distance=*/0);
        moonlab_libirrep_qec_free(q);
    }

    /* --- Color codes (v0.6.2) --- */
    fprintf(stdout, "\n--- color codes (2D Steane + Hamming) ---\n");
    {
        moonlab_libirrep_qec_t *q = NULL;
        const int rc = moonlab_libirrep_color_steane_new(&q);
        CHECK(rc == 0, "color_steane_new rc=%d", rc);
        /* Steane [[7, 1, 3]] CSS code. */
        check_shape(q, "Steane",   7, 3, 3, 1, 3, /*distance=*/1);
        moonlab_libirrep_qec_free(q);

        q = NULL;
        const int rc2 = moonlab_libirrep_color_hamming_15_7_3_new(&q);
        CHECK(rc2 == 0, "color_hamming_15_7_3_new rc=%d", rc2);
        /* Hamming-based CSS [[15, 7, 3]]: n = 15, k = 7, d = 3. */
        check_shape(q, "Hamming",  15, 4, 4, 7, 3, /*distance=*/0);
        moonlab_libirrep_qec_free(q);
    }

    /* --- Bivariate-bicycle qLDPC (Bravyi et al. 2024, Nature 627, 778) --- */
    fprintf(stdout, "\n--- bivariate-bicycle qLDPC (IBM Gross) ---\n");
    {
        moonlab_libirrep_qec_t *q = NULL;
        int rc = moonlab_libirrep_bb_72_12_6_new(&q);
        CHECK(rc == 0, "bb_72_12_6_new rc=%d", rc);
        check_shape(q, "BB 72",   72, 36, 36, 12, 6,  /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        q = NULL;
        rc = moonlab_libirrep_bb_144_12_12_new(&q);
        CHECK(rc == 0, "bb_144_12_12_new rc=%d", rc);
        check_shape(q, "BB 144", 144, 72, 72, 12, 12, /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        q = NULL;
        rc = moonlab_libirrep_bb_288_12_18_new(&q);
        CHECK(rc == 0, "bb_288_12_18_new rc=%d", rc);
        check_shape(q, "BB 288", 288, 144, 144, 12, 18, /*distance=*/0);
        moonlab_libirrep_qec_free(q);
    }

    /* --- Hypergraph product (Tillich-Zemor 2009) --- */
    fprintf(stdout, "\n--- hypergraph product (rep_d x rep_d) ---\n");
    {
        moonlab_libirrep_qec_t *q = NULL;
        int rc = moonlab_libirrep_hgp_repetition_new(3, &q);
        CHECK(rc == 0, "hgp_repetition_new(3) rc=%d", rc);
        check_shape(q, "HGP rep3", 13,  6,  6, 1, 3, /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        q = NULL;
        rc = moonlab_libirrep_hgp_repetition_new(4, &q);
        CHECK(rc == 0, "hgp_repetition_new(4) rc=%d", rc);
        check_shape(q, "HGP rep4", 25, 12, 12, 1, 4, /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        q = NULL;
        rc = moonlab_libirrep_hgp_repetition_new(5, &q);
        CHECK(rc == 0, "hgp_repetition_new(5) rc=%d", rc);
        check_shape(q, "HGP rep5", 41, 20, 20, 1, 5, /*distance=*/0);
        moonlab_libirrep_qec_free(q);

        /* Out-of-range d is rejected. */
        q = NULL;
        rc = moonlab_libirrep_hgp_repetition_new(6, &q);
        CHECK(rc != 0, "hgp_repetition_new(6) rejects d > 5 (rc=%d)", rc);
    }

    fprintf(stdout, "\n--- error path ---\n");
    moonlab_libirrep_qec_t *q = NULL;
    int rc = moonlab_libirrep_surface_code_new(/*d=*/1, &q);
    CHECK(rc != 0, "distance < 2 rejected (rc=%d)", rc);
    rc = moonlab_libirrep_surface_code_new(/*d=*/3, NULL);
    CHECK(rc != 0, "NULL output rejected (rc=%d)", rc);
    rc = moonlab_libirrep_toric_code_new(/*Lx=*/1, /*Ly=*/3, &q);
    CHECK(rc != 0, "toric_code_new(1, 3) rejects Lx < 2 (rc=%d)", rc);
    moonlab_libirrep_qec_free(NULL); /* must not crash */
    fprintf(stdout, "  OK    qec_free(NULL) does not crash\n");

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
