/**
 * @file qgt_kitaev_z2.c
 * @brief Kitaev p-wave chain Z_2 invariant via the v0.3 1D-BdG path.
 *
 * The Kitaev (2001) chain is a 1D BdG topological superconductor whose
 * Z_2 invariant can be read off from the Pfaffian-sign product at the
 * two TR-invariant momenta k = 0 and k = pi:
 *   nu = (1 - sgn(M(0)) sgn(M(pi))) / 2
 * For our convention M(k) = -2t cos(k) - mu at TRIM, so the topological
 * regime is |mu| < 2|t| with Majorana zero modes at chain edges.
 *
 * This program is the momentum-space companion to the existing
 * real-space Jordan-Wigner Kitaev chain example
 * (`examples/topological/kitaev_chain.c`).
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <stdio.h>

int main(void) {
    const double t = 1.0;
    const double delta = 0.5;

    printf("Kitaev p-wave chain Z_2 invariant\n");
    printf("  t=%.2f, delta=%.2f, boundary at |mu|=2t=%.2f\n\n",
           t, delta, 2.0 * t);

    printf("  %-7s | %-3s | %s\n", "mu", "Z2", "phase");
    printf("  --------+-----+--------------------------------\n");
    for (double mu = -3.0; mu <= 3.0; mu += 0.25) {
        qgt_system_1d_t* sys = qgt_model_kitaev_chain(t, mu, delta);
        if (!sys) { fprintf(stderr, "alloc fail\n"); return 1; }
        int z2 = -1;
        qgt_z2_invariant_1d_bdg(sys, &z2);
        const char* phase = (z2 == 1) ? "topological (Majorana edges)"
                                       : "trivial";
        printf("  %+7.2f | %3d | %s\n", mu, z2, phase);
        qgt_free_1d(sys);
    }
    return 0;
}
