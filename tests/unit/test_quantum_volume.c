/**
 * @file test_quantum_volume.c
 * @brief Unit test for the Quantum Volume harness.
 *
 * Width 4, 40 trials on an ideal simulator: mean HOP must land above
 * the 2/3 passing threshold with a positive CI margin.  Asymptotically
 * a noiseless simulator approaches mean HOP ≈ (1 + ln 2)/2 ≈ 0.847;
 * at small widths it's somewhat lower but still well above 2/3.
 */

#include "../../src/applications/quantum_volume.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void run_width(size_t w, size_t trials, uint64_t seed) {
    fprintf(stdout, "\n-- QV width=%zu, trials=%zu --\n", w, trials);
    qv_result_t r = { 0 };
    int rc = quantum_volume_run(w, trials, seed, &r);
    CHECK(rc == 0, "quantum_volume_run returns 0 (got %d)", rc);
    fprintf(stdout,
            "    mean_hop=%.4f  sd=%.4f  lower_ci=%.4f  passed=%d\n",
            r.mean_hop, r.stddev_hop, r.lower_ci_97p5, r.passed);
    CHECK(r.mean_hop > 0.75,
          "mean HOP %.3f > 0.75 (ideal simulator expectation)", r.mean_hop);
    CHECK(r.passed == 1, "CI lower bound clears the 2/3 QV threshold");
}

int main(void) {
    fprintf(stdout, "=== Quantum Volume ===\n");
    run_width(3, 40, 0xC0FFEE01u);
    run_width(4, 40, 0xC0FFEE02u);
    run_width(5, 40, 0xC0FFEE03u);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
