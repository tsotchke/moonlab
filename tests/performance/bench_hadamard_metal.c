/**
 * @file bench_hadamard_metal.c
 * @brief Metal GPU Hadamard throughput micro-benchmark (fp32).
 *
 * Runs metal_hadamard against a zero-copy Metal buffer holding an
 * fp32-complex state of the same dimension as the CPU bench. Reports
 * wall-clock and effective bandwidth (4 bytes / component instead of
 * 8 on the CPU path, so the traffic is half).
 *
 * Skipped (prints a note, exits 0) when Metal is unavailable.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if defined(__APPLE__) && defined(HAS_METAL)
#  include "../../src/optimization/gpu_metal.h"
#  define METAL_PRESENT 1
#else
#  define METAL_PRESENT 0
#endif

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

#if METAL_PRESENT

typedef struct { float re; float im; } fcomplex_t;

static void bench_metal_H(metal_compute_ctx_t* ctx, size_t n) {
    uint64_t dim   = 1ULL << n;
    size_t   bytes = (size_t)dim * sizeof(fcomplex_t);   /* fp32 complex = 8 B */
    fcomplex_t* host = (fcomplex_t*)aligned_alloc(64, bytes);
    if (!host) { printf("  n=%2zu  alloc fail\n", n); return; }
    /* Initialise to |0...0> with a small perturbation so the state isn't
     * trivially invariant under H. */
    memset(host, 0, bytes);
    host[0].re = 0.9999f;
    host[1].re = 0.01414f;

    metal_buffer_t* buf = metal_buffer_create_from_data(ctx, host, bytes);
    if (!buf) { free(host); printf("  n=%2zu  buffer fail\n", n); return; }

    /* Warm up: Metal kernel compilation and first dispatch. */
    for (int w = 0; w < 5; w++) metal_hadamard(ctx, buf, (uint32_t)(n / 2), (uint32_t)dim);

    /* Measurement. */
    const int iters = 10;
    double t0 = now_us();
    for (int i = 0; i < iters; i++)
        metal_hadamard(ctx, buf, (uint32_t)(n / 2), (uint32_t)dim);
    double dt = (now_us() - t0) / iters;

    /* Effective R+W bandwidth: each pair reads 2x8=16 bytes and writes 16 bytes. */
    double bw = ((double)dim * 8.0 * 2.0 / (dt * 1e-6)) / 1e9;
    printf("  H-fp32  n=%2zu  %10llu  %9.2f us    %6.1f GB/s (GPU)\n",
           n, (unsigned long long)dim, dt, bw);

    metal_buffer_free(buf);
    free(host);
}

int main(void) {
    metal_compute_ctx_t* ctx = metal_compute_init();
    if (!ctx) {
        printf("Metal not available on this host.\n");
        return 0;
    }
    char name[256];
    uint32_t max_threads = 0, num_cores = 0;
    metal_get_device_info(ctx, name, &max_threads, &num_cores);
    printf("=== Metal GPU Hadamard micro-benchmark (fp32) ===\n");
    printf("Device: %s  cores=%u  max_threads_per_group=%u\n\n",
           name, num_cores, max_threads);
    printf("  gate         n     dim        time         R+W bw\n");
    printf("  ----         --    ----       ----         ------\n");
    for (size_t n = 16; n <= 26; n += 2) bench_metal_H(ctx, n);
    metal_compute_free(ctx);
    return 0;
}

#else

int main(void) {
    printf("Metal not compiled in (HAS_METAL not defined).\n");
    return 0;
}

#endif
