/**
 * @file bench_tensor_matmul_eshkol.c
 * @brief End-to-end timing of tensor_matmul through the Eshkol
 *        dispatch path vs CPU Accelerate.
 *
 * `tensor_matmul` auto-dispatches to Eshkol when the problem size
 * exceeds a per-tier threshold (src/algorithms/tensor_network/tensor.c).
 * This bench sweeps a range of sizes straddling the crossover and at
 * each size reports CPU-only vs GPU-dispatched wall-clock.  We force
 * CPU by setting the threshold to ~infinity before the CPU reference
 * pass.
 */

#include "../../src/algorithms/tensor_network/tensor.h"
#include "../../src/optimization/gpu/backends/gpu_eshkol.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void randomise(double complex* buf, uint64_t n, unsigned seed) {
    srand(seed);
    for (uint64_t i = 0; i < n; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        buf[i] = re + im * I;
    }
}

static double l2_rel(const double complex* a, const double complex* b, uint64_t n) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double dr = creal(a[i] - b[i]);
        double di = cimag(a[i] - b[i]);
        double ar = creal(a[i]), ai = cimag(a[i]);
        num += dr*dr + di*di;
        den += ar*ar + ai*ai;
    }
    return (den > 0) ? sqrt(num / den) : sqrt(num);
}

static void bench(uint32_t M, uint32_t K, uint32_t N) {
    tensor_t* A = tensor_create_matrix(M, K);
    tensor_t* B = tensor_create_matrix(K, N);
    if (!A || !B) { tensor_free(A); tensor_free(B); return; }
    randomise(A->data, (uint64_t)M * K, 0xA01);
    randomise(B->data, (uint64_t)K * N, 0xB02);

    /* CPU reference: force the threshold to infinity so we never
     * dispatch to the GPU path. */
    setenv("MOONLAB_TENSOR_GPU_THRESHOLD_MUL", "99999999999999999", 1);
    tensor_t* Cref = tensor_matmul(A, B);
    const int iters = 3;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        tensor_t* tmp = tensor_matmul(A, B);
        tensor_free(tmp);
    }
    double cpu_us = (now_us() - t0) / iters;

    /* GPU: restore the tier-aware threshold default. */
    unsetenv("MOONLAB_TENSOR_GPU_THRESHOLD_MUL");
    tensor_t* Cgpu = tensor_matmul(A, B);
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        tensor_t* tmp = tensor_matmul(A, B);
        tensor_free(tmp);
    }
    double gpu_us = (now_us() - t0) / iters;

    double err = Cref && Cgpu ? l2_rel(Cref->data, Cgpu->data, (uint64_t)M * N) : 0.0;

    double flops = 8.0 * (double)M * (double)K * (double)N;
    double cpu_gf = flops / (cpu_us * 1e-6) / 1e9;
    double gpu_gf = flops / (gpu_us * 1e-6) / 1e9;

    printf("  M=%-5u K=%-4u N=%-4u  CPU %9.2f us (%6.1f GF/s)  "
           "tensor_matmul %9.2f us (%6.1f GF/s)  %5.2fx  err=%.2e\n",
           M, K, N, cpu_us, cpu_gf, gpu_us, gpu_gf,
           cpu_us / gpu_us, err);

    tensor_free(A); tensor_free(B);
    tensor_free(Cref); tensor_free(Cgpu);
}

int main(void) {
    printf("=== tensor_matmul: CPU-forced vs auto-dispatched ===\n");
    if (moonlab_eshkol_available()) {
        const char* tiers[] = { "EXACT", "HIGH", "FAST", "ML" };
        int t = (int)moonlab_eshkol_get_precision();
        printf("Eshkol active, precision tier: %s\n\n", tiers[t]);
    } else {
        printf("Eshkol not available; GPU pass will run the CPU path.\n\n");
    }

    /* Sweep across the crossover for the default FAST tier (2^27). */
    bench(128,  64,  64);        /* well below */
    bench(256, 128, 128);        /* below */
    bench(512, 256, 256);        /* right at crossover */
    bench(1024, 512, 512);       /* above */
    bench(2048, 1024, 1024);     /* well above */
    bench(4096, 2048, 2048);     /* large */
    return 0;
}
