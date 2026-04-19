/**
 * @file bench_chern_kpm.c
 * @brief Scaling demonstration for the matrix-free KPM Chern marker.
 *
 * Wall-clock for a single bulk-site marker computation at various L,
 * showing memory stays O(N) and time scales as L^2 * n_cheby * L^2
 * (three projector applications, each O(n_cheby * nnz) with nnz = 5N).
 */

#include "../../src/algorithms/topology_realspace/chern_kpm.h"

#include <stdio.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void bench_one(size_t L, size_t n_cheby, double m) {
    chern_kpm_system_t* sys = chern_kpm_create(L, m, n_cheby);
    size_t ctr = L / 2;
    double t0 = now_us();
    double c = chern_kpm_local_marker(sys, ctr, ctr);
    double dt = now_us() - t0;
    printf("  L=%-4zu  N=%-6zu  n_cheby=%-4zu  c(ctr)=%+.4f  time=%.1f ms\n",
           L, L * L * 2, n_cheby, c, dt / 1000.0);
    chern_kpm_free(sys);
}

int main(int argc, char** argv) {
    int extended = (argc >= 2 && argv[1][0] == 'x');
    printf("=== KPM Chern marker scaling (QWZ m=-1, one bulk site) ===\n");
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
    return 0;
}
