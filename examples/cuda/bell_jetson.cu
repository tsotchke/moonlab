/**
 * @file  bell_jetson.cu
 * @brief First moonlab CUDA backend smoke -- Bell state on Jetson.
 *
 * Builds against src/backends/cuda_statevec.h.  Runs:
 *   1. create N=12 state vector (4096 amplitudes)
 *   2. apply H on qubit 0
 *   3. apply CNOT(0, 1)
 *   4. copy back, print P(|00>), P(|01>), P(|10>), P(|11>)
 *   5. assert  |a[00]|^2 + |a[(N-1) low bits)]|^2 = 1
 *
 * Standalone -- no moonlab core dependency, so this is the
 * tightest possible loop to verify the new CUDA backend works on
 * the Jetson.
 */

#include "../../src/backends/cuda_statevec.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 12;
    printf("=== moonlab CUDA backend Bell smoke (N=%u) ===\n", N);
    moonlab_gpu_kind_t kind = moonlab_gpu_probe_kind();
    printf("    detected GPU kind: %s\n", moonlab_gpu_probe_kind_str(kind));

    moonlab_cuda_state_t *s = NULL;
    moonlab_cuda_status_t rc = moonlab_cuda_state_create(N, &s);
    if (rc != MOONLAB_CUDA_OK) {
        fprintf(stderr, "    state_create failed: %d\n", rc);
        return 1;
    }
    printf("    state dim: %llu\n", (unsigned long long)moonlab_cuda_state_dim(s));

    /* H = (1/sqrt(2)) [[1, 1], [1, -1]] in row-major,
     * pairs of {re, im} per matrix element. */
    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    double H[8] = {
         inv_sqrt2, 0.0,   inv_sqrt2, 0.0,
         inv_sqrt2, 0.0,  -inv_sqrt2, 0.0,
    };
    rc = moonlab_cuda_apply_1q(s, 0, H);
    if (rc != MOONLAB_CUDA_OK) { fprintf(stderr, "    H failed\n"); return 2; }

    rc = moonlab_cuda_apply_cnot(s, /*ctrl=*/0, /*tgt=*/1);
    if (rc != MOONLAB_CUDA_OK) { fprintf(stderr, "    CNOT failed\n"); return 3; }

    moonlab_cuda_synchronize(s);

    /* Pull state back to host -- on Tegra this is unified memory,
     * no actual copy.  On discrete this would be a real cudaMemcpy. */
    uint64_t dim = moonlab_cuda_state_dim(s);
    double *host = (double *)malloc(dim * 2 * sizeof(double));
    moonlab_cuda_state_copy_to_host(s, host);

    /* Bell on the first two qubits with the rest as zeroes:
     *   amp(00...0) = 1/sqrt(2),  amp(11 followed by 0...0) = 1/sqrt(2). */
    uint64_t idx_00 = 0ULL;
    uint64_t idx_11 = 0ULL | (1ULL << 0) | (1ULL << 1);
    double p00 = host[2*idx_00]*host[2*idx_00] + host[2*idx_00+1]*host[2*idx_00+1];
    double p11 = host[2*idx_11]*host[2*idx_11] + host[2*idx_11+1]*host[2*idx_11+1];

    /* Probability mass elsewhere should be zero. */
    double rest = 0.0;
    for (uint64_t k = 0; k < dim; ++k) {
        if (k == idx_00 || k == idx_11) continue;
        rest += host[2*k]*host[2*k] + host[2*k+1]*host[2*k+1];
    }

    printf("    P(|0...0>) = %.6f  (expected 0.5)\n", p00);
    printf("    P(|...011>) = %.6f  (qubits 0,1 = |11>, rest |0>; expected 0.5)\n", p11);
    printf("    P(everything else) = %.2e  (expected 0)\n", rest);

    int ok = (fabs(p00 - 0.5) < 1e-9)
          && (fabs(p11 - 0.5) < 1e-9)
          && (rest < 1e-12);
    printf("    Bell-state assertion: %s\n", ok ? "PASS" : "FAIL");

    free(host);
    moonlab_cuda_state_free(s);
    return ok ? 0 : 4;
}
