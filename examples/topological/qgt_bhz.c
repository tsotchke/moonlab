/**
 * @file qgt_bhz.c
 * @brief BHZ (Bernevig-Hughes-Zhang 2006) Z_2 invariant phase scan.
 *
 * Sweeps the band-inversion mass M for fixed A=B=1 and reports the
 * Z_2 invariant.  In the lattice regularisation the QSH window is
 * 0 < M/B < 8 (X-corner Dirac closings cancel; the M-corner closing
 * at 8B re-trivialises).  See docs/research/quantum_geometry_tensor.md
 * for the discussion of why this differs from the textbook continuum
 * 0 < M/B < 4 boundary.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <stdio.h>

int main(void) {
    const double A = 1.0;
    const double B = 1.0;

    printf("BHZ Z_2 invariant phase diagram\n");
    printf("  A=%.2f, B=%.2f\n", A, B);
    printf("  QSH window in lattice regularisation: 0 < M/B < 8\n\n");

    printf("  %-7s | %-3s | %s\n", "M", "Z2", "phase");
    printf("  --------+-----+----------------\n");
    for (double M = -2.0; M <= 10.0; M += 0.5) {
        qgt_system_n_t* sys = qgt_model_bhz(A, B, M);
        if (!sys) { fprintf(stderr, "alloc fail\n"); return 1; }
        int z2 = -1;
        if (qgt_z2_invariant(sys, /*N=*/32, &z2) != 0) {
            fprintf(stderr, "z2 fail at M=%.2f\n", M);
            qgt_free_nband(sys);
            return 1;
        }
        const char* phase = (z2 == 1) ? "QSH" : "trivial";
        printf("  %+7.2f | %3d | %s\n", M, z2, phase);
        qgt_free_nband(sys);
    }
    return 0;
}
