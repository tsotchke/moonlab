/**
 * @file bench_chern_mosaic.c
 * @brief Computes a full 2D Chern mosaic: c(r) for every bulk site on
 *        a modulated QWZ lattice, and prints a coarse text heatmap
 *        along with the bulk statistics. This is the moonlab mirror
 *        of Antao-Sun-Fumega-Lado (PRL 136, 156601 (2026)) Fig. 2
 *        rendered on a tractable L x L lattice.
 *
 * The current matrix-free backend runs on a single CPU core; an
 * L=30 bulk region with ~400 sites takes about a second.  With the
 * upcoming MPO/QTCI backend (plan §2I), the same computation will
 * scale to 10^6-10^8 sites.
 */

#include "../../src/algorithms/topology_realspace/chern_kpm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static const char* shade(double c) {
    /* Map c in roughly [-1, 1] to a 7-step ramp. */
    if      (c < -0.75) return "##";
    else if (c < -0.25) return "==";
    else if (c < -0.05) return "--";
    else if (c <  0.05) return "..";
    else if (c <  0.25) return "++";
    else if (c <  0.75) return "**";
    else                return "@@";
}

static void print_mosaic(const chern_kpm_system_t* sys,
                         size_t rmin, size_t rmax,
                         double* out_min, double* out_max,
                         double* out_mean) {
    size_t side = rmax - rmin;
    double* vals = malloc(side * side * sizeof(double));
    double t0 = now_s();
    chern_kpm_bulk_map(sys, rmin, rmax, vals);
    double dt = now_s() - t0;

    double mn = 1e30, mx = -1e30, sum = 0.0;
    for (size_t i = 0; i < side * side; i++) {
        if (vals[i] < mn) mn = vals[i];
        if (vals[i] > mx) mx = vals[i];
        sum += vals[i];
    }
    *out_min = mn; *out_max = mx; *out_mean = sum / (side * side);
    printf("  bulk region [%zu..%zu)^2 (%zu sites): min=%+.3f max=%+.3f "
           "mean=%+.3f  time=%.2f s (%.0f ms/site)\n",
           rmin, rmax, side * side, mn, mx, *out_mean,
           dt, dt * 1000.0 / (side * side));
    for (size_t y = 0; y < side; y++) {
        printf("    ");
        for (size_t x = 0; x < side; x++) {
            printf("%s", shade(vals[y * side + x]));
        }
        printf("\n");
    }
    free(vals);
}

int main(void) {
    printf("=== Chern mosaic (QWZ m=-1 + C_4 modulation) ===\n");
    printf("Shading: @@ = +1, ** = +0.5, ++ = +0.15, .. = 0, "
           "-- = -0.15, == = -0.5, ## = -1\n\n");

    const size_t L = 40;
    const size_t n_cheby = 160;
    chern_kpm_system_t* sys = chern_kpm_create(L, -1.0, n_cheby);

    /* Small modulation: topology preserved. */
    double* V_small = chern_kpm_cn_modulation(L, 4, 2.0 * M_PI / 7.0, 0.25);
    chern_kpm_set_modulation(sys, V_small, 4.0 * 0.25);
    printf("Small V0=0.25 (deep in topological phase):\n");
    double mn, mx, me;
    print_mosaic(sys, 4, 36, &mn, &mx, &me);

    /* Large modulation: gap closed, marker near 0. */
    double* V_large = chern_kpm_cn_modulation(L, 4, 2.0 * M_PI / 7.0, 3.0);
    chern_kpm_set_modulation(sys, V_large, 4.0 * 3.0);
    printf("\nLarge V0=3.0 (gap closed; expect uniform ~0):\n");
    print_mosaic(sys, 4, 36, &mn, &mx, &me);

    /* Transition: expect mixed map. */
    double* V_mid = chern_kpm_cn_modulation(L, 4, 2.0 * M_PI / 5.0, 1.2);
    chern_kpm_set_modulation(sys, V_mid, 4.0 * 1.2);
    printf("\nMid V0=1.2 (near transition; expect structure):\n");
    print_mosaic(sys, 4, 36, &mn, &mx, &me);

    chern_kpm_free(sys);
    free(V_small); free(V_large); free(V_mid);
    return 0;
}
