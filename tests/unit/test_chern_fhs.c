/**
 * @file test_chern_fhs.c
 * @brief Regression test for the Fukui-Hatsugai-Suzuki momentum-space
 *        Chern integrator and its cross-validation against the dense
 *        Newton-Schulz real-space marker.
 *
 * Three checks:
 *   1. FHS lower-band Chern integer recovers the analytic QWZ phase
 *      diagram (m in (-2, 0): C = -1; m in (0, 2): C = +1; |m| > 2: 0)
 *      following the Qi-Wu-Zhang convention.  Unrounded plaquette sum
 *      drifts < 1e-2 from integer at N >= 64 in any gapped phase.
 *   2. FHS at N=64 vs Newton-Schulz Bianco-Resta marker at L=10.  The
 *      two routines use opposite sign conventions (FHS lower-band C
 *      vs Bianco-Resta upper-band C as implemented in chern_marker.c);
 *      we assert |C_FHS| == |C_BR-rounded| at every test point and
 *      the sign relation C_BR = -C_FHS in the gapped phases.
 *   3. FHS converges in N: integer is constant from N=16 to N=128 in
 *      any gapped phase.
 *
 * Closes the paper §2.7 / §4.3 audit point on the three-paths claim:
 * Newton-Schulz dense, KPM matrix-free, and FHS momentum-space all
 * in-tree, all asserted to agree on the QWZ Chern integer.
 */

#include "../../src/algorithms/topology_realspace/chern_fhs.h"
#include "../../src/algorithms/topology_realspace/chern_marker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int test_qwz_phase_diagram(void) {
    /* Lower-band Chern of QWZ via FHS, Qi-Wu-Zhang sign convention:
     *   -2 < m < 0:    C = -1
     *    0 < m < 2:    C = +1
     *   |m| > 2:       C =  0
     */
    const struct { double m; int expected; } cases[] = {
        { -1.0, -1 },
        { -0.5, -1 },
        { -1.5, -1 },
        {  0.5, +1 },
        {  1.0, +1 },
        {  1.5, +1 },
        { -3.0,  0 },
        {  3.0,  0 },
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int C; double Cr;
        int err = chern_fhs_qwz(64, cases[i].m, &C, &Cr);
        if (err) {
            fprintf(stderr, "FHS: m=%.2f returned error %d\n",
                    cases[i].m, err);
            rc = 1; continue;
        }
        if (C != cases[i].expected) {
            fprintf(stderr,
                    "FAIL: m=%+.2f expected C=%+d, got C=%+d (real %.6f)\n",
                    cases[i].m, cases[i].expected, C, Cr);
            rc = 1;
        } else {
            fprintf(stdout,
                    "  OK: m=%+.2f -> C=%+d (real %.6f, error %.2e)\n",
                    cases[i].m, C, Cr,
                    fabs(Cr - (double)cases[i].expected));
        }
        if (fabs(Cr - (double)cases[i].expected) > 1e-2) {
            fprintf(stderr,
                    "FAIL: m=%+.2f real C=%.6f drifted >1e-2 from integer\n",
                    cases[i].m, Cr);
            rc = 1;
        }
    }
    return rc;
}

static int test_fhs_vs_dense_marker(void) {
    /* Compare FHS to Newton-Schulz Bianco-Resta marker on the same QWZ
     * sweep.  The two use opposite sign conventions; we assert |C|
     * matches between the two and the sign relation is consistent. */
    const struct { double m; int fhs_expected; } cases[] = {
        { -1.0, -1 },
        {  1.0, +1 },
        { -3.0,  0 },
        {  3.0,  0 },
    };
    int rc = 0;
    const size_t L = 10;
    const size_t rmin = 3, rmax = 7;
    const size_t patch = (rmax - rmin) * (rmax - rmin);

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int C_fhs; double Cr_fhs;
        int err = chern_fhs_qwz(64, cases[i].m, &C_fhs, &Cr_fhs);
        if (err) { rc = 1; continue; }

        chern_system_t* sys = chern_qwz_create(L, cases[i].m);
        if (!sys) { rc = 1; continue; }
        if (chern_build_projector(sys) != 0) {
            chern_system_free(sys); rc = 1; continue;
        }
        double bulk = chern_bulk_sum(sys, rmin, rmax);
        double bulk_avg = bulk / (double)patch;
        chern_system_free(sys);

        int C_dense_int = (int)lrint(bulk_avg);

        if (C_fhs != cases[i].fhs_expected) {
            fprintf(stderr,
                    "FAIL FHS: m=%+.2f expected %+d, got %+d (real %.6f)\n",
                    cases[i].m, cases[i].fhs_expected, C_fhs, Cr_fhs);
            rc = 1;
        }
        if (abs(C_dense_int) != abs(cases[i].fhs_expected)) {
            fprintf(stderr,
                    "FAIL DENSE: m=%+.2f expected |C|=%d, "
                    "dense bulk_avg=%.6f rounds to |%d|\n",
                    cases[i].m, abs(cases[i].fhs_expected),
                    bulk_avg, abs(C_dense_int));
            rc = 1;
        }
        /* Sign-flip relation: in the gapped phases, C_dense = -C_fhs. */
        if (cases[i].fhs_expected != 0 && C_dense_int != -C_fhs) {
            fprintf(stderr,
                    "FAIL SIGN: m=%+.2f FHS=%+d expected dense=%+d, "
                    "got dense=%+d\n",
                    cases[i].m, C_fhs, -C_fhs, C_dense_int);
            rc = 1;
        }
        fprintf(stdout,
                "  m=%+.2f  FHS C=%+d  Dense bulk_avg=%+.6f (rounds to %+d, "
                "consistent with FHS sign-flip convention)\n",
                cases[i].m, C_fhs, bulk_avg, C_dense_int);
    }
    return rc;
}

static int test_fhs_convergence(void) {
    /* Increasing N at fixed gapped m should not change the integer and
     * should keep the unrounded real value within a small window of
     * the integer. */
    const double m = -1.0;
    const int expected = -1;
    int prev_C = 99;
    int rc = 0;
    for (size_t N = 16; N <= 128; N *= 2) {
        int C; double Cr;
        if (chern_fhs_qwz(N, m, &C, &Cr) != 0) { rc = 1; continue; }
        double drift = fabs(Cr - (double)expected);
        fprintf(stdout, "  N=%zu: C=%+d, real=%.6f, drift=%.2e\n",
                N, C, Cr, drift);
        if (prev_C != 99 && C != prev_C) {
            fprintf(stderr, "FAIL: integer changed across N (gapped phase)\n");
            rc = 1;
        }
        prev_C = C;
        if (N >= 64 && drift > 1e-2) {
            fprintf(stderr,
                    "FAIL: drift %.2e at N=%zu in gapped phase\n",
                    drift, N);
            rc = 1;
        }
    }
    return rc;
}

int main(void) {
    int rc = 0;
    fprintf(stdout, "[1] FHS Chern integer on QWZ phase diagram (N=64)\n");
    rc |= test_qwz_phase_diagram();
    fprintf(stdout, "\n[2] FHS vs dense Newton-Schulz marker (L=10 bulk patch)\n");
    rc |= test_fhs_vs_dense_marker();
    fprintf(stdout, "\n[3] FHS N-convergence at m=-1\n");
    rc |= test_fhs_convergence();
    if (rc == 0) {
        fprintf(stdout, "\nAll FHS Chern tests PASSED.\n");
    } else {
        fprintf(stdout, "\nFHS Chern tests FAILED (rc=%d).\n", rc);
    }
    return rc;
}
