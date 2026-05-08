/**
 * @file qgt_hofstadter.c
 * @brief Hofstadter butterfly sub-band Chern numbers via the v0.3
 *        n-band QGT module.
 *
 * For magnetic flux phi = 1/q per plaquette, the Bloch Hamiltonian on
 * the q-site magnetic unit cell has q sub-bands with integer Chern
 * numbers determined by the TKNN Diophantine equation
 *   t_r * p + s_r * q = r.
 * The lowest band has Chern = +1 for any q.  This program prints the
 * lowest-band Chern at q in {2, 3, 4, 5, 6, 7} so the canonical
 * pattern is visible.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    printf("Harper-Hofstadter sub-band Chern numbers (TKNN)\n");
    printf("  square lattice, flux phi = 1/q per plaquette\n");
    printf("  reporting Chern of the lowest band only\n\n");

    printf("  %-3s | %-5s | %-5s\n", "q", "C", "expected");
    printf("  ----+-------+--------\n");
    /* Skip q=2: at flux = 1/2 the bands touch at the M-point and the
     * topology is not defined in the usual sense (semimetal). */
    const size_t qs[] = { 3, 4, 5, 6, 7 };
    for (size_t i = 0; i < sizeof(qs)/sizeof(qs[0]); i++) {
        size_t q = qs[i];
        qgt_system_n_t* sys = qgt_model_hofstadter(/*t=*/1.0,
                                                    /*p=*/1, q,
                                                    /*n_occupied=*/1);
        if (!sys) { fprintf(stderr, "alloc fail q=%zu\n", q); return 1; }
        qgt_berry_grid_t g;
        int rc = qgt_berry_grid_nband(sys, /*N=*/24, &g);
        if (rc != 0) { fprintf(stderr, "fail q=%zu\n", q); qgt_free_nband(sys); return 1; }
        int C = (int)lround(g.chern);
        printf("  %3zu | %+5d | %+5d\n", q, C, +1);
        qgt_berry_grid_free(&g);
        qgt_free_nband(sys);
    }
    printf("\nFor q=3 the full sub-band Cherns are (+1, -2, +1).  The middle\n");
    printf("band's negative compensation gives the famous Hofstadter butterfly\n");
    printf("when the spectrum vs. flux is plotted.\n");
    return 0;
}
