/**
 * @file bench_chern_mosaic_hq.c
 * @brief Research-grade Chern mosaic renderer for modulated QWZ.
 *
 * Reproduces the Antao-Sun-Fumega-Lado (PRL 136, 156601 (2026)) Fig. 2
 * motif at whatever scale the sparse-stencil backend can reach today:
 * the local Bianco-Resta Chern marker on a QWZ lattice with a C_n
 * quasicrystalline onsite modulation.  Compared to the legacy
 * @c bench_chern_mosaic, this version:
 *
 *  - Accepts @c --L, @c --n, @c --V0, @c --Q, @c --n-cheby as flags.
 *  - Emits the full per-site map to @c $MOONLAB_CHERN_OUT_CSV
 *    (x,y,c rows).
 *  - Emits an 8-bit PPM image to @c $MOONLAB_CHERN_OUT_PPM, using a
 *    diverging blue-white-red colour map centred on zero so the
 *    sign of the marker is visually unambiguous.
 *  - Emits a reproducibility manifest to @c $MOONLAB_MANIFEST_OUT.
 *
 * The scale ceiling on the sparse stencil path is about L = 300 on a
 * single modern host.  The P5.08 milestones (MPO Chebyshev-KPM +
 * QTCI position operators) will lift that to 10^6-10^8 sites per
 * Antao et al.; this bench is the honest upper bound today and doubles
 * as the reference any MPS-level implementation must reproduce.
 */

#include "../../src/algorithms/topology_realspace/chern_kpm.h"
#include "../../src/utils/manifest.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Default parameters reproducing the legacy bench behaviour when no
 * flags are passed. */
#define DEFAULT_L         64
#define DEFAULT_N_SYM      4
#define DEFAULT_V0         1.0
#define DEFAULT_Q          (2.0 * M_PI / 7.0)
#define DEFAULT_N_CHEBY  200
#define BOUNDARY_PAD       4

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Parse --key=value or --key value from argv; returns the value or the
 * supplied default.  Flags not present return the default unchanged. */
static const char* flag_arg(int argc, char** argv, const char* key,
                            const char* def) {
    size_t klen = strlen(key);
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], key, klen) == 0 && argv[i][klen] == '=') {
            return argv[i] + klen + 1;
        }
        if (strcmp(argv[i], key) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return def;
}

/* ---------------------------------------------------------------- */
/* Diverging blue-white-red colour map                               */
/* ---------------------------------------------------------------- */
static void diverging_rgb(double c, double vmax,
                          unsigned char* r, unsigned char* g,
                          unsigned char* b)
{
    double t = c / vmax;
    if (t >  1.0) t =  1.0;
    if (t < -1.0) t = -1.0;
    /* Linear interpolation: -1 -> (23, 41, 120) blue,
     *                         0 -> (246, 247, 247) off-white,
     *                        +1 -> (153, 20,  42) red. */
    double tn = (t + 1.0) * 0.5;   /* 0..1 from blue end to red end */
    unsigned char blue_r  = 23,  blue_g  = 41,  blue_b  = 120;
    unsigned char white_r = 246, white_g = 247, white_b = 247;
    unsigned char red_r   = 153, red_g   = 20,  red_b   = 42;
    if (tn < 0.5) {
        double u = tn * 2.0;
        *r = (unsigned char)(blue_r  + u * (white_r - blue_r));
        *g = (unsigned char)(blue_g  + u * (white_g - blue_g));
        *b = (unsigned char)(blue_b  + u * (white_b - blue_b));
    } else {
        double u = (tn - 0.5) * 2.0;
        *r = (unsigned char)(white_r + u * (red_r - white_r));
        *g = (unsigned char)(white_g + u * (red_g - white_g));
        *b = (unsigned char)(white_b + u * (red_b - white_b));
    }
}

/* ---------------------------------------------------------------- */
/* CSV + PPM writers                                                  */
/* ---------------------------------------------------------------- */

static int write_csv(const char* path, size_t side, size_t rmin,
                     const double* vals)
{
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "x,y,c\n");
    for (size_t y = 0; y < side; y++) {
        for (size_t x = 0; x < side; x++) {
            fprintf(f, "%zu,%zu,%.6f\n",
                    rmin + x, rmin + y, vals[y * side + x]);
        }
    }
    fclose(f);
    return 0;
}

static int write_ppm(const char* path, size_t side,
                     const double* vals, double vmax)
{
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%zu %zu\n255\n", side, side);
    unsigned char px[3];
    for (size_t y = 0; y < side; y++) {
        for (size_t x = 0; x < side; x++) {
            diverging_rgb(vals[y * side + x], vmax, &px[0], &px[1], &px[2]);
            fwrite(px, 1, 3, f);
        }
    }
    fclose(f);
    return 0;
}

/* ---------------------------------------------------------------- */
/* main                                                              */
/* ---------------------------------------------------------------- */

int main(int argc, char** argv) {
    const size_t L       = (size_t)atoi(flag_arg(argc, argv, "--L",        "64"));
    const int    n_sym   =          atoi(flag_arg(argc, argv, "--n",        "4"));
    const double V0      =          atof(flag_arg(argc, argv, "--V0",       "1.0"));
    const double Q       =          atof(flag_arg(argc, argv, "--Q",        "0.8976"));
    const size_t n_cheby = (size_t)atoi(flag_arg(argc, argv, "--n-cheby",  "200"));

    if (L < BOUNDARY_PAD * 2 + 4) {
        fprintf(stderr, "L too small: need L >= %d\n", BOUNDARY_PAD * 2 + 4);
        return 1;
    }

    printf("=== Chern mosaic (QWZ m=-1 + C_%d modulation) ===\n", n_sym);
    printf("  L = %zu, bulk padding = %d, N_c = %zu\n",
           L, BOUNDARY_PAD, n_cheby);
    printf("  modulation: V_0 = %.4f, Q = %.4f, symmetry = C_%d\n",
           V0, Q, n_sym);

    moonlab_manifest_t manifest;
    moonlab_manifest_capture(&manifest, "bench_chern_mosaic_hq", 0);

    chern_kpm_system_t* sys = chern_kpm_create(L, -1.0, n_cheby);
    if (!sys) {
        fprintf(stderr, "chern_kpm_create(L=%zu) failed\n", L);
        moonlab_manifest_release(&manifest);
        return 1;
    }

    double* V = chern_kpm_cn_modulation(L, n_sym, Q, V0);
    /* Modulation amplitude bound: each of the n cosine terms is in
     * [-V0, V0], so the peak |V(r)| can be as large as n * V0.  We use
     * that as the rescale-widening upper bound. */
    chern_kpm_set_modulation(sys, V, (double)n_sym * fabs(V0));

    /* Full bulk region minus BOUNDARY_PAD on each side. */
    size_t rmin = BOUNDARY_PAD;
    size_t rmax = L - BOUNDARY_PAD;
    size_t side = rmax - rmin;
    size_t n_sites = side * side;
    double* vals = (double*)malloc(n_sites * sizeof(double));

    double t0 = now_s();
    int rc = chern_kpm_bulk_map(sys, rmin, rmax, vals);
    double dt = now_s() - t0;
    if (rc != 0) {
        fprintf(stderr, "bulk_map failed rc=%d\n", rc);
        free(vals); chern_kpm_free(sys); free(V);
        moonlab_manifest_release(&manifest);
        return 1;
    }

    double mn = 1e30, mx = -1e30, sum = 0.0;
    for (size_t i = 0; i < n_sites; i++) {
        if (vals[i] < mn) mn = vals[i];
        if (vals[i] > mx) mx = vals[i];
        sum += vals[i];
    }
    double mean = sum / (double)n_sites;

    printf("  bulk region [%zu, %zu)^2 (%zu sites)\n", rmin, rmax, n_sites);
    printf("  min=%+.4f  max=%+.4f  mean=%+.4f\n", mn, mx, mean);
    printf("  total time: %.2f s   (%.2f ms / site)\n",
           dt, dt * 1000.0 / (double)n_sites);

    const double vmax = fmax(fabs(mn), fabs(mx));

    const char* csv_out = getenv("MOONLAB_CHERN_OUT_CSV");
    if (csv_out && *csv_out) {
        if (write_csv(csv_out, side, rmin, vals) == 0) {
            printf("  csv: %s\n", csv_out);
        } else {
            fprintf(stderr, "  csv write failed: %s\n", csv_out);
        }
    }
    const char* ppm_out = getenv("MOONLAB_CHERN_OUT_PPM");
    if (ppm_out && *ppm_out) {
        if (write_ppm(ppm_out, side, vals, vmax) == 0) {
            printf("  ppm: %s  (vmax=%.4f)\n", ppm_out, vmax);
        } else {
            fprintf(stderr, "  ppm write failed: %s\n", ppm_out);
        }
    }

    char metrics[4096];
    snprintf(metrics, sizeof metrics,
             "{\"L\":%zu,\"n_sym\":%d,\"V0\":%.6f,\"Q\":%.6f,"
             "\"n_cheby\":%zu,\"rmin\":%zu,\"rmax\":%zu,"
             "\"sites\":%zu,\"min\":%.6f,\"max\":%.6f,\"mean\":%.6f,"
             "\"vmax_abs\":%.6f,\"wall_s\":%.4f,\"ms_per_site\":%.4f,"
             "\"csv_path\":\"%s\",\"ppm_path\":\"%s\"}",
             L, n_sym, V0, Q, n_cheby, rmin, rmax,
             n_sites, mn, mx, mean, vmax, dt,
             dt * 1000.0 / (double)n_sites,
             csv_out ? csv_out : "",
             ppm_out ? ppm_out : "");
    manifest.metrics_json = metrics;

    moonlab_manifest_stamp_finish(&manifest);

    const char* manifest_out = getenv("MOONLAB_MANIFEST_OUT");
    if (manifest_out && *manifest_out) {
        FILE* f = fopen(manifest_out, "w");
        if (f) {
            moonlab_manifest_write_json_pretty(&manifest, f);
            fclose(f);
            fprintf(stderr, "[manifest] written to %s\n", manifest_out);
        }
    }

    free(vals);
    chern_kpm_free(sys);
    free(V);
    moonlab_manifest_release(&manifest);
    return 0;
}
