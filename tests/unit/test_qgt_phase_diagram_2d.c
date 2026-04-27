/**
 * @file test_qgt_phase_diagram_2d.c
 * @brief Pin qgt_phase_diagram_chern_2d on a known-good test system.
 *
 * Uses QWZ as the underlying model, treating its mass `m` as the
 * x-axis knob and a no-op `dummy` as the y-axis knob.  The expected
 * phase diagram is therefore the QWZ 1D phase diagram replicated
 * across every dummy column:
 *   m < -2: C =  0
 *   -2 < m <  0: C = +1
 *    0 < m < +2: C = -1
 *   m > +2: C =  0
 *
 * This exercises the 2D-sweep mechanics independently of any
 * specific 2-parameter physical model.  When the Haldane Bloch
 * implementation in qgt.c is hardened against grid-size oscillation
 * its (t2, phi) phase diagram becomes the natural physics test.
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

/* x = QWZ mass m, y = dummy (ignored). */
static qgt_system_t* qwz_factory_2d(void* user, double m, double dummy) {
    (void)user;
    (void)dummy;
    return qgt_model_qwz(m);
}

int main(void) {
    fprintf(stdout, "=== QWZ phase diagram (2D harness, dummy y-axis) ===\n");
    const size_t Kx = 11;       /* m axis */
    const size_t Ky = 4;        /* dummy axis */
    const size_t N = 32;
    const double m_min = -2.6, m_max = +2.6;
    const double y_min =  0.0, y_max = 1.0;

    int chern[Kx * Ky];
    int rc = qgt_phase_diagram_chern_2d(qwz_factory_2d, NULL,
                                         m_min, m_max, y_min, y_max,
                                         Kx, Ky, N, chern);
    CHECK(rc == 0, "qgt_phase_diagram_chern_2d returned %d", rc);
    if (rc != 0) return 1;

    fprintf(stdout, "  m  \\ y :");
    for (size_t iy = 0; iy < Ky; iy++) {
        const double y = y_min + (y_max - y_min) * (double)iy / (double)(Ky - 1);
        fprintf(stdout, "  %.2f", y);
    }
    fprintf(stdout, "\n");
    for (size_t ix = 0; ix < Kx; ix++) {
        const double m = m_min + (m_max - m_min) * (double)ix / (double)(Kx - 1);
        fprintf(stdout, "  %+5.2f   :", m);
        for (size_t iy = 0; iy < Ky; iy++) {
            fprintf(stdout, "  %+3d", chern[ix * Ky + iy]);
        }
        fprintf(stdout, "\n");

        /* Expected QWZ Chern (matches qgt_model_qwz convention). */
        int expected;
        if      (m < -2.0) expected = 0;
        else if (m <  0.0) expected = +1;
        else if (m <  2.0) expected = -1;
        else               expected = 0;

        const double dist = fmin(fmin(fabs(m + 2.0), fabs(m)), fabs(m - 2.0));
        if (dist < 0.05) continue;

        for (size_t iy = 0; iy < Ky; iy++) {
            CHECK(chern[ix * Ky + iy] == expected,
                  "QWZ at m=%+.2f, y_idx=%zu: got %+d, expected %+d",
                  m, iy, chern[ix * Ky + iy], expected);
        }
    }
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
