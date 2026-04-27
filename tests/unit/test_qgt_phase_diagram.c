/**
 * @file test_qgt_phase_diagram.c
 * @brief Pin the QWZ Chern phase diagram against known integer phases.
 *
 * The Qi-Wu-Zhang 2-band Chern insulator
 *   H(k) = sin(k_x) sigma_x + sin(k_y) sigma_y
 *        + (m + cos(k_x) + cos(k_y)) sigma_z
 * has integer Chern number C(m) of the lower band; the existing
 * qgt_model_qwz convention (matching the moonlab_qwz_chern ABI test
 * which expects C(+1.0) = -1) gives:
 *   m < -2: C =  0
 *   -2 < m <  0: C = +1
 *    0 < m < +2: C = -1
 *   m > +2: C =  0
 *
 * Sweep m in [-3, +3] at K = 25 samples; expect Chern jumps at the
 * phase boundaries m = -2, 0, +2.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

/* Factory: (void*)NULL, parameter = QWZ mass m. */
static qgt_system_t* qwz_factory(void* user, double m) {
    (void)user;
    return qgt_model_qwz(m);
}

int main(void) {
    fprintf(stdout, "=== QWZ phase diagram via qgt_phase_diagram_chern ===\n");
    const size_t K = 25;
    const size_t N = 32;
    const double m_min = -3.0, m_max = +3.0;

    int chern[K];
    int rc = qgt_phase_diagram_chern(qwz_factory, NULL,
                                      m_min, m_max, K, N, chern);
    CHECK(rc == 0, "qgt_phase_diagram_chern returned %d", rc);
    if (rc != 0) return 1;

    fprintf(stdout, "    m       C\n");
    for (size_t i = 0; i < K; i++) {
        const double m = m_min + (m_max - m_min) * (double)i / (double)(K - 1);
        fprintf(stdout, "  %+5.2f    %+d\n", m, chern[i]);

        /* Build the analytic expectation in the qgt_model_qwz sign
         * convention (matches the moonlab_qwz_chern ABI test). */
        int expected;
        if      (m < -2.0) expected = 0;
        else if (m <  0.0) expected = +1;
        else if (m <  2.0) expected = -1;
        else               expected = 0;

        /* Skip the exact phase-boundary points; the gap closes there
         * and the FHS integral is undefined. */
        const double dist_from_boundary =
            fmin(fmin(fabs(m + 2.0), fabs(m)), fabs(m - 2.0));
        if (dist_from_boundary < 0.05) continue;

        CHECK(chern[i] == expected,
              "QWZ at m = %+.2f: got C = %+d, expected %+d",
              m, chern[i], expected);
    }
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
