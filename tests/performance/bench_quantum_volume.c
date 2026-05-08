/**
 * @file bench_quantum_volume.c
 * @brief Quantum Volume benchmark runner with JSON output.
 *
 * Runs the QV protocol at widths 3..10 with 100 trials each and prints
 * mean HOP, stddev, CI lower bound, and pass/fail.  A noiseless
 * simulator should pass every width: QV_max = 2^width_max.
 *
 * Emits JSON with schema "moonlab/quantum_volume_v1" so the paper
 * can pull QV claims directly.  Pass the output path as argv[1]
 * (default ./quantum_volume.json).
 */

#include "../../src/applications/quantum_volume.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "quantum_volume.json";
    const size_t w_min = 3, w_max = 10;
    const size_t trials = 100;

    printf("=== Quantum Volume benchmark (noiseless statevector) ===\n");
    printf("  schema: moonlab/quantum_volume_v1  out: %s\n\n", out_path);
    printf("%-6s %-8s %-10s %-10s %-10s %s\n",
           "width", "trials", "mean_hop", "stddev", "lo_ci97.5", "pass");

    qv_result_t* rows = (qv_result_t*)calloc(w_max - w_min + 1, sizeof(qv_result_t));
    int* ok = (int*)calloc(w_max - w_min + 1, sizeof(int));
    size_t qv_max_passed = 0;

    for (size_t w = w_min; w <= w_max; w++) {
        const size_t i = w - w_min;
        int rc = quantum_volume_run(w, trials, 0xD15EA5Eu + (unsigned)w, &rows[i]);
        ok[i] = (rc == 0);
        if (!ok[i]) {
            printf("  width=%zu ERROR rc=%d\n", w, rc);
            continue;
        }
        printf("%-6zu %-8zu %-10.4f %-10.4f %-10.4f %s\n",
               rows[i].width, rows[i].num_trials, rows[i].mean_hop,
               rows[i].stddev_hop, rows[i].lower_ci_97p5,
               rows[i].passed ? "YES" : "NO");
        if (rows[i].passed && w > qv_max_passed) qv_max_passed = w;
    }
    printf("\nQV = 2^%zu = %llu (largest passing width).\n",
           qv_max_passed, 1ULL << qv_max_passed);

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); free(rows); free(ok); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/quantum_volume_v1\",\n");
    fprintf(f, "  \"description\": \"IBM Quantum Volume protocol on the "
               "noiseless state-vector engine.  HOP = heavy-output "
               "probability; pass at width w iff the lower 97.5%% CI on "
               "mean HOP exceeds 2/3.  QV = 2^w_max.\",\n");
    fprintf(f, "  \"trials_per_width\": %zu,\n", trials);
    fprintf(f, "  \"qv_max_width\": %zu,\n", qv_max_passed);
    fprintf(f, "  \"qv_value\": %llu,\n", 1ULL << qv_max_passed);
    fprintf(f, "  \"rows\": [");
    int first = 1;
    for (size_t w = w_min; w <= w_max; w++) {
        const size_t i = w - w_min;
        if (!ok[i]) continue;
        fprintf(f, "%s\n    {\"width\": %zu, \"trials\": %zu, "
                   "\"mean_hop\": %.6f, \"stddev_hop\": %.6f, "
                   "\"lower_ci_97p5\": %.6f, \"passed\": %s}",
                first ? "" : ",",
                rows[i].width, rows[i].num_trials, rows[i].mean_hop,
                rows[i].stddev_hop, rows[i].lower_ci_97p5,
                rows[i].passed ? "true" : "false");
        first = 0;
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("wrote %s\n", out_path);

    free(rows); free(ok);
    return 0;
}
