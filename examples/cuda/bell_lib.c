/**
 * @file  bell_lib.c
 * @brief moonlab CUDA backend Bell via libquantumsim.so.
 *
 * Same circuit as bell_jetson.cu but compiles as plain C against
 * the installed library -- proves the CUDA backend's headers are
 * C-callable and the symbols are exported from libquantumsim.so.
 * Also exercises the new gate / measurement / norm primitives.
 */

#include "../../src/backends/cuda_statevec.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    uint32_t N = (argc > 1) ? (uint32_t)atoi(argv[1]) : 16;
    printf("=== moonlab CUDA (libquantumsim) Bell at N=%u ===\n", N);
    printf("    GPU kind: %s\n",
        moonlab_gpu_probe_kind_str(moonlab_gpu_probe_kind()));

    moonlab_cuda_state_t *s = NULL;
    if (moonlab_cuda_state_create(N, &s) != MOONLAB_CUDA_OK) {
        fprintf(stderr, "create failed\n"); return 1;
    }

    /* Bell on (0, 1).  Use the new convenience entry points. */
    moonlab_cuda_apply_h(s, 0);
    moonlab_cuda_apply_cnot(s, /*ctrl=*/0, /*tgt=*/1);
    moonlab_cuda_synchronize(s);

    /* New: query probability and norm directly via GPU reductions. */
    double p1_q0 = 0.0, p1_q1 = 0.0, norm = 0.0;
    moonlab_cuda_prob_z(s, 0, &p1_q0);
    moonlab_cuda_prob_z(s, 1, &p1_q1);
    moonlab_cuda_norm  (s,    &norm);

    printf("    P(q0=1) = %.6f  (expected 0.5)\n", p1_q0);
    printf("    P(q1=1) = %.6f  (expected 0.5; perfectly correlated with q0)\n", p1_q1);
    printf("    norm    = %.12f  (expected 1.0)\n", norm);

    int ok = fabs(p1_q0 - 0.5)  < 1e-9
          && fabs(p1_q1 - 0.5)  < 1e-9
          && fabs(norm  - 1.0)  < 1e-12;
    printf("    Bell + reductions: %s\n", ok ? "PASS" : "FAIL");

    moonlab_cuda_state_free(s);
    return ok ? 0 : 4;
}
