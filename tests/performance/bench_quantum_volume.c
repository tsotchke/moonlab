/**
 * @file bench_quantum_volume.c
 * @brief Quantum Volume benchmark runner.
 *
 * Runs the QV protocol at widths 3..10 with 100 trials each and prints
 * mean HOP, stddev, CI lower bound, and pass/fail.  A noiseless
 * simulator should pass every width: QV_max = 2^width_max.
 */

#include "../../src/applications/quantum_volume.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=== Quantum Volume benchmark (noiseless statevector) ===\n");
    printf("%-6s %-8s %-10s %-10s %-10s %s\n",
           "width", "trials", "mean_hop", "stddev", "lo_ci97.5", "pass");
    for (size_t w = 3; w <= 10; w++) {
        qv_result_t r;
        int rc = quantum_volume_run(w, 100, 0xD15EA5Eu + w, &r);
        if (rc != 0) {
            printf("  width=%zu ERROR rc=%d\n", w, rc);
            continue;
        }
        printf("%-6zu %-8zu %-10.4f %-10.4f %-10.4f %s\n",
               r.width, r.num_trials, r.mean_hop, r.stddev_hop,
               r.lower_ci_97p5, r.passed ? "YES" : "NO");
    }
    printf("\nQV = 2^w_max where w_max is the largest passing width.\n");
    return 0;
}
