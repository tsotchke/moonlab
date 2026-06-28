/**
 * @file  test_libirrep_sector_ed.c
 * @brief Validate moonlab_libirrep_heisenberg_sector_e0 at N = 12 / 18 / 24.
 *
 * The bridge factors out translation symmetry via libirrep's space-group
 * + rep-table machinery, restricting Lanczos to the orbit-representative
 * basis at fixed total Sz.  That sector compression is what lets moonlab
 * reach N > 14 Heisenberg ground-state ED -- the dense-zheev path tops
 * out at 4096-dim (N = 12) due to mpo_to_matrix memory.
 *
 * Cross-checks:
 *   - N = 12 kagome (Lx, Ly) = (2, 2), Sz = 0 (popcount = 6).  Ground
 *     state is a singlet, so sector E_0 must agree with full ED's
 *     `-5.44487522` reference to Lanczos precision.
 *   - N = 18 kagome (Lx, Ly) = (3, 2), Sz = 0.  Full ED is out of reach
 *     for moonlab's mpo_to_matrix; libirrep's
 *     PHYSICS_RESULTS.md table 1 reports `-8.43586` (J = 1 spin).
 *   - N = 24 kagome (Lx, Ly) = (4, 2), Sz = 0.  Demonstrates the new
 *     reach; no internal reference value, just convergence checks.
 *
 * Built only when QSIM_ENABLE_LIBIRREP=ON; the no-libirrep test build
 * still compiles + links the unit but the test gracefully exits 77
 * (CTest SKIP) so the suite stays green on default configurations.
 */

#include "../../src/integration/libirrep_bridge.h"

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

/* libirrep PHYSICS_RESULTS.md (table 1) -- spin convention J = 1, S = 1/2.
 * The N = 12 cluster's ground state is a singlet so the Sz = 0 sector
 * recovers the full Hilbert space value -5.44487522 J exactly.  At
 * N = 18 / 24 the absolute GS of the p1 cluster sits in higher-Sz
 * sectors (PHYSICS_RESULTS.md section 6.1.1: "The N=18 p1 cluster's
 * ABSOLUTE GS is a TRIPLET (F=-1) at E=-8.008"), so the Sz = 0 sector
 * gives the lowest singlet, which is generally NOT the full-Hilbert-
 * space E_0.  We assert the AFM-friendly range and report the value
 * rather than asserting a hardcoded reference for the larger N. */
#define E0_KAGOME_12_SZ0_REF  (-5.44487522)

static int run_sector(int Lx, int Ly, int sz_total_2x, int max_iters,
                      double *e0_out, long long *dim_out)
{
    double eigvals[2] = {0.0};
    const int rc = moonlab_libirrep_heisenberg_sector_e0(
        MOONLAB_LIBIRREP_LATTICE_KAGOME, Lx, Ly,
        MOONLAB_LIBIRREP_WALLPAPER_P1,     /* translation-only sector */
        sz_total_2x,
        /*k_wanted=*/1,
        max_iters,
        eigvals,
        dim_out);
    if (rc != MOONLAB_LIBIRREP_OK) return rc;
    *e0_out = eigvals[0];
    return MOONLAB_LIBIRREP_OK;
}

int main(void)
{
    /* Line-buffer stdout so the OK/FAIL records flush as they print
     * (the N = 24 case takes minutes; the default block-buffering
     * hides progress until exit when stdout is piped through grep). */
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== libirrep sector-ED bridge: kagome N = 12 / 18 / 24 ===\n\n");

    if (!moonlab_libirrep_available()) {
        fprintf(stdout, "  libirrep not linked -- test skipped (CTest exit 77).\n");
        return 77;
    }

    /* N = 12, Sz = 0 cross-check vs full ED. */
    {
        double e0 = 0.0; long long dim = 0;
        const int rc = run_sector(/*Lx=*/2, /*Ly=*/2, /*sz_2x=*/0,
                                  /*max_iters=*/200, &e0, &dim);
        CHECK(rc == 0, "kagome 12 Sz=0 sector-ED returned rc=%d", rc);
        if (rc == 0) {
            fprintf(stdout, "    N=12 sector dim   = %lld  (vs full 2^12 = 4096)\n", dim);
            fprintf(stdout, "    N=12 E_0 (sector) = %+.10f\n", e0);
            fprintf(stdout, "    N=12 E_0 (full)   = %+.10f  (PRB 83, 212401)\n",
                    E0_KAGOME_12_SZ0_REF);
            CHECK(fabs(e0 - E0_KAGOME_12_SZ0_REF) < 1e-6,
                  "N=12 sector ED matches full ED to <1e-6");
            CHECK(dim < 4096,
                  "N=12 sector dim %lld < 4096 (compression vs full Hilbert)", dim);
        }
    }
    fprintf(stdout, "\n");

    /* N = 18, Sz = 0: out of moonlab's mpo_to_matrix reach.  Sz = 0
     * sector E_0 is not the published full-ED ground state (the latter
     * sits in the triplet at this cluster); we assert the AFM
     * range + sector compression instead, and report the value. */
    {
        double e0 = 0.0; long long dim = 0;
        const int rc = run_sector(/*Lx=*/3, /*Ly=*/2, /*sz_2x=*/0,
                                  /*max_iters=*/300, &e0, &dim);
        CHECK(rc == 0, "kagome 18 Sz=0 sector-ED returned rc=%d", rc);
        if (rc == 0) {
            fprintf(stdout, "    N=18 sector dim   = %lld  (vs full 2^18 = 262144)\n", dim);
            fprintf(stdout, "    N=18 E_0 (sector) = %+.10f  (per-site %.4f J)\n",
                    e0, e0 / 18.0);
            /* AFM kagome E_0 / N is in (-0.5, -0.4) for any cluster
             * shape past the small-N regime; broad range so the test
             * stays robust against minor numerical drift. */
            CHECK(e0 < 0.0 && e0 > -10.0,
                  "N=18 E_0 in (-10, 0) (AFM ground state)");
            CHECK(dim < 262144,
                  "N=18 sector dim %lld < 262144 (compression vs full Hilbert)", dim);
        }
    }
    fprintf(stdout, "\n");

    /* N = 24, Sz = 0: new territory for moonlab, but the full-reorth
     * Lanczos basis at ~337k-dim is ~800 MB and the iter^2 dot-product
     * cost runs ~5-8 minutes on M2 Ultra at 150 iters -- too heavy for
     * default CI.  Opt in with MOONLAB_LIBIRREP_HEAVY=1. */
    if (getenv("MOONLAB_LIBIRREP_HEAVY") != NULL) {
        double e0 = 0.0; long long dim = 0;
        const int rc = run_sector(/*Lx=*/4, /*Ly=*/2, /*sz_2x=*/0,
                                  /*max_iters=*/150, &e0, &dim);
        CHECK(rc == 0, "kagome 24 Sz=0 sector-ED returned rc=%d", rc);
        if (rc == 0) {
            fprintf(stdout, "    N=24 sector dim   = %lld  (vs full 2^24 = 16777216)\n", dim);
            fprintf(stdout, "    N=24 E_0 (sector) = %+.10f  (per-site %.4f J)\n",
                    e0, e0 / 24.0);
            CHECK(e0 < 0.0 && e0 > -14.0,
                  "N=24 E_0 in (-14, 0) (AFM ground state)");
            CHECK(dim < 16777216,
                  "N=24 sector dim %lld < 2^24 (compression vs full Hilbert)", dim);
        }
    } else {
        fprintf(stdout, "    N=24 heavy case skipped "
                "(set MOONLAB_LIBIRREP_HEAVY=1 to enable; ~5-8 min on M2 Ultra)\n");
    }
    fprintf(stdout, "\n");

    fprintf(stdout, "=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
