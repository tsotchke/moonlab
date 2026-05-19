/**
 * @file  test_libirrep_css.c
 * @brief Validate the CSS-code bridge (v0.6.1 phase B).
 *
 * Confirms that `moonlab_libirrep_surface_code_new(d)` builds a
 * well-formed CSS code with the expected shape:
 *
 *   d  |  n = d^2  |  m_X  |  m_Z  |  k  |  distance
 *   3  |    9      |   4   |   4   |  1  |    3
 *   5  |   25      |  12   |  12   |  1  |    5
 *
 * Exercises the stabiliser-row accessors to make sure the
 * `irrep_parity_matrix_get` plumbing returns sensible bit values.
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
    fprintf(stdout, "=== libirrep CSS-code bridge: surface code d = 3, 5 ===\n");

    if (!moonlab_libirrep_available()) {
        fprintf(stdout, "  libirrep not linked -- test skipped (CTest exit 77).\n");
        return 77;
    }

    /* Distance 3: 9 qubits, 4 X-stabs (one per face, bulk + 2 boundary),
     * 4 Z-stabs.  Brute-force distance is tractable. */
    exercise_surface(/*d=*/3, /*m_X=*/4, /*m_Z=*/4, /*distance=*/1);

    /* Distance 5: 25 qubits, 12 of each stabiliser type. */
    exercise_surface(/*d=*/5, /*m_X=*/12, /*m_Z=*/12,
                     /*distance=*/0); /* skip brute-force at n=25 */

    fprintf(stdout, "\n--- error path ---\n");
    moonlab_libirrep_qec_t *q = NULL;
    int rc = moonlab_libirrep_surface_code_new(/*d=*/1, &q);
    CHECK(rc != 0, "distance < 2 rejected (rc=%d)", rc);
    rc = moonlab_libirrep_surface_code_new(/*d=*/3, NULL);
    CHECK(rc != 0, "NULL output rejected (rc=%d)", rc);
    moonlab_libirrep_qec_free(NULL); /* must not crash */
    fprintf(stdout, "  OK    qec_free(NULL) does not crash\n");

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
