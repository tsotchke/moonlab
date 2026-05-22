/**
 * @file test_qgt_vs_chern_marker.c
 * @brief Cross-check the momentum-space FHS / projector-trace Chern
 *        integrators (qgt.h) against the real-space Bianco-Resta
 *        local Chern marker (chern_marker.h) on the QWZ model.
 *
 * Both methods compute the same topological invariant of the same
 * Hamiltonian; agreement to within a few percent is a strong sanity
 * check on both implementations and the cleanest possible
 * verification that what the QGT module exposes via momentum-space
 * Bloch states equals what real-space simulations pick up via the
 * projector formulation.
 *
 * QWZ phase diagram:
 *   -2 < m < 0  -> Chern = +1
 *    0 < m < 2  -> Chern = -1
 *   |m| > 2     -> Chern =  0
 *
 * Method:
 *   - momentum: qgt_model_qwz(m), qgt_berry_grid_proj on N=48 BZ grid.
 *   - real-space: chern_qwz_create(L, m), chern_local_marker at the
 *                 lattice center.
 * Both must round to the same integer.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"
#include "../../src/algorithms/topology_realspace/chern_marker.h"

#include <math.h>
#include <stdint.h>
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

static int momentum_chern(double m) {
    qgt_system_t* sys = qgt_model_qwz(m);
    if (!sys) return INT32_MIN;
    qgt_berry_grid_t g;
    int rc = qgt_berry_grid_proj(sys, 48, &g);
    int c = (rc == 0) ? (int)lround(g.chern) : INT32_MIN;
    if (rc == 0) qgt_berry_grid_free(&g);
    qgt_free(sys);
    return c;
}

static int realspace_chern(double m, size_t L) {
    chern_system_t* sys = chern_qwz_create(L, m);
    if (!sys) return INT32_MIN;
    if (chern_build_projector(sys) != 0) {
        chern_system_free(sys);
        return INT32_MIN;
    }
    /* 2x2 patch at the lattice center (kept small for L=10 -- larger
     * patches reach the boundary where local marker has compensating
     * surface contributions). */
    size_t lo = L / 2 - 1, hi = L / 2 + 1;
    double bulk_sum = chern_bulk_sum(sys, lo, hi);
    double mean = bulk_sum / 4.0;
    chern_system_free(sys);
    return (int)lround(mean);
}

int main(void) {
    fprintf(stdout, "=== QGT (momentum) vs chern_marker (real-space) on QWZ ===\n");

    struct { double m; int C; const char* label; } cases[] = {
        { -1.5, +1, "topological m=-1.5" },
        { -0.5, +1, "topological m=-0.5" },
        { +0.5, -1, "topological m=+0.5" },
        { +1.5, -1, "topological m=+1.5" },
        { -3.0,  0, "trivial m=-3.0"     },
        { +3.0,  0, "trivial m=+3.0"     },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        double m = cases[i].m;
        int expected = cases[i].C;
        int Cm = momentum_chern(m);
        int Cr = realspace_chern(m, /*L=*/10);
        fprintf(stdout, "\n-- %s --\n", cases[i].label);
        fprintf(stdout, "    momentum   Chern = %+d\n", Cm);
        fprintf(stdout, "    real-space Chern = %+d (L=10, 2x2 bulk patch)\n", Cr);
        CHECK(Cm == expected,
              "momentum = %+d (expected %+d)", Cm, expected);
        CHECK(Cr == expected,
              "real-space = %+d (expected %+d)", Cr, expected);
        CHECK(Cm == Cr,
              "momentum and real-space agree at C=%+d", Cm);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
