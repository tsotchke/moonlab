/**
 * @file bench_chern_kpm.c
 * @brief Scaling demonstration for the matrix-free KPM Chern marker.
 *
 * Wall-clock for a single bulk-site marker computation at various L,
 * showing memory stays O(N) and time scales as L^2 * n_cheby * L^2
 * (three projector applications, each O(n_cheby * nnz) with nnz = 5N).
 */

#include "../../src/algorithms/topology_realspace/chern_kpm.h"
#include "../../src/utils/manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static char*  g_metrics = NULL;
static size_t g_metrics_cap = 0;

static void bench_one(size_t L, size_t n_cheby, double m) {
    chern_kpm_system_t* sys = chern_kpm_create(L, m, n_cheby);
    size_t ctr = L / 2;
    double t0 = now_us();
    double c = chern_kpm_local_marker(sys, ctr, ctr);
    double dt = now_us() - t0;
    printf("  L=%-4zu  N=%-6zu  n_cheby=%-4zu  c(ctr)=%+.4f  time=%.1f ms\n",
           L, L * L * 2, n_cheby, c, dt / 1000.0);

    if (g_metrics && g_metrics_cap) {
        size_t cur = strlen(g_metrics);
        snprintf(g_metrics + cur, g_metrics_cap - cur,
                 "%s{\"L\":%zu,\"N\":%zu,\"n_cheby\":%zu,"
                 "\"c_center\":%.6f,\"time_ms\":%.3f}",
                 cur == 0 ? "" : ",",
                 L, L * L * 2, n_cheby, c, dt / 1000.0);
    }
    chern_kpm_free(sys);
}

int main(int argc, char** argv) {
    int extended = (argc >= 2 && argv[1][0] == 'x');
    printf("=== KPM Chern marker scaling (QWZ m=-1, one bulk site) ===\n");

    moonlab_manifest_t manifest;
    moonlab_manifest_capture(&manifest, "bench_chern_kpm", 0);
    char buf[4096] = "[";
    g_metrics = buf + 1;
    g_metrics_cap = sizeof buf - 2;

    bench_one(12,  80, -1.0);
    bench_one(20, 100, -1.0);
    bench_one(30, 120, -1.0);
    bench_one(40, 140, -1.0);
    bench_one(60, 160, -1.0);
    bench_one(100, 180, -1.0);
    if (extended) {
        bench_one(200, 200, -1.0);
        bench_one(300, 220, -1.0);
    }

    size_t mlen = strlen(buf);
    buf[mlen] = ']'; buf[mlen + 1] = '\0';
    char metrics_obj[5120];
    snprintf(metrics_obj, sizeof metrics_obj, "{\"rows\":%s}", buf);
    manifest.metrics_json = metrics_obj;
    moonlab_manifest_stamp_finish(&manifest);

    const char* out_path = getenv("MOONLAB_MANIFEST_OUT");
    if (out_path && *out_path) {
        FILE* f = fopen(out_path, "w");
        if (f) {
            moonlab_manifest_write_json_pretty(&manifest, f);
            fclose(f);
            fprintf(stderr, "[manifest] written to %s\n", out_path);
        }
    }
    moonlab_manifest_release(&manifest);
    return 0;
}
