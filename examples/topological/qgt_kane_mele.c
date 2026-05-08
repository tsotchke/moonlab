/**
 * @file qgt_kane_mele.c
 * @brief Kane-Mele model Z_2 invariant via the v0.3 QGT module.
 *
 * Sweeps the sublattice-mass parameter `lambda_v` across the QSH /
 * trivial transition at lambda_v = 3*sqrt(3)*lambda_so for fixed
 * lambda_so = 0.06.  Reports the Z_2 invariant (1 = QSH with
 * helical edge modes; 0 = trivial) at every grid point.
 *
 * Build: example wired by cmake/examples.cmake.  Run:
 *   ./build/qgt_kane_mele
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const double t = 1.0;
    const double lambda_so = 0.06;
    const double boundary = 3.0 * sqrt(3.0) * lambda_so;

    printf("Kane-Mele Z_2 invariant phase diagram (Sz-conserving fast path)\n");
    printf("  t=%.2f, lambda_so=%.3f, lambda_r=0\n", t, lambda_so);
    printf("  Phase boundary at |lambda_v| = 3*sqrt(3)*lambda_so = %.4f\n\n",
           boundary);

    printf("  %-10s | %-3s | %s\n", "lambda_v", "Z2", "phase");
    printf("  -----------+-----+----------------\n");
    for (double lv = 0.0; lv <= 0.5; lv += 0.025) {
        qgt_system_n_t* sys = qgt_model_kane_mele(t, lambda_so, 0.0, lv);
        if (!sys) { fprintf(stderr, "alloc fail\n"); return 1; }
        int z2 = -1;
        if (qgt_z2_invariant(sys, /*N=*/48, &z2) != 0) {
            fprintf(stderr, "z2 fail at lv=%.3f\n", lv);
            qgt_free_nband(sys);
            return 1;
        }
        const char* phase = (z2 == 1) ? "QSH" : "trivial";
        printf("  %-10.3f | %3d | %s\n", lv, z2, phase);
        qgt_free_nband(sys);
    }

    printf("\nQSH for lambda_v < %.4f, trivial above.\n", boundary);
    return 0;
}
