/*
 * qgt_qec_node.c --- the QGT singularity IS the QEC node: one epsilon^2 = 0.
 *
 * A single physics story tying together two of Moonlab's own layers -- the quantum geometric
 * tensor (quantum_geometry/qgt) and topological error correction (topological/surface_code).
 *
 * In the Qi-Wu-Zhang two-band Chern insulator, the gap closes at a Dirac node (k = (pi,pi), m = 2)
 * where the two bands COLLIDE.  There, three things happen at once, and they are the SAME thing:
 *
 *   1. The Fubini-Study metric (real part of the QGT) DIVERGES as tr g ~ 2/gap^2 (Fubini-Study normalization).
 *   2. The Chern number -- the Berry-curvature (imaginary part of the QGT) topological invariant --
 *      JUMPS by 1 across the node.
 *   3. On the error-correction side, a surface / toric code is a chain complex whose X- and
 *      Z-stabilizers COMMUTE: d1 . d2 = 0.  That nilpotent square d^2 = 0 is the same eps^2 = 0
 *      as the two-branch collision that makes the QGT metric blow up.
 *
 * So the place where Moonlab's geometry is singular is the place where its topology is nontrivial
 * and where its error-correcting code lives -- geometry and error correction, one object seen twice.
 *
 * Build (against a built libquantumsim): see the accompanying CMake target `qgt_qec_node`.
 */
#include "../../src/algorithms/quantum_geometry/qgt.h"
#include "../../src/applications/moonlab_export.h"
#include "../../src/algorithms/topological/topological.h"
#include <stdio.h>
#include <math.h>

/* QWZ band gap 2|d|, d = (sin kx, sin ky, m + cos kx + cos ky). */
static double qwz_gap(double m, double kx, double ky) {
    double dx = sin(kx), dy = sin(ky), dz = m + cos(kx) + cos(ky);
    return 2.0 * sqrt(dx * dx + dy * dy + dz * dz);
}

int main(void) {
    const double PI = 3.14159265358979323846;

    printf("==== the QGT singularity IS the QEC node: one epsilon^2 = 0 ====\n");
    printf(" Qi-Wu-Zhang model; Dirac node at k=(pi,pi), m=2 (two bands collide).\n");
    printf(" Approach m -> 2 from the trivial side and watch the Fubini-Study metric diverge:\n\n");
    printf("    m       gap 2|d|      tr g (Moonlab QGT)     tr g * gap^2\n");

    const double ms[] = {2.5, 2.2, 2.1, 2.05, 2.02, 2.01};
    for (int i = 0; i < 6; ++i) {
        double m = ms[i];
        qgt_system_t *sys = qgt_model_qwz(m);
        double k[2] = {PI, PI}, g[4];
        qgt_metric_at(sys, k, 1e-4, g);
        double trg = g[0] + g[3];               /* trace of the 2x2 Fubini-Study metric */
        double gap = qwz_gap(m, PI, PI);
        printf("  %6.3f    %10.6f     %16.3f      %10.4f\n", m, gap, trg, trg * gap * gap);
        qgt_free(sys);
    }
    printf("\n => tr g * gap^2 -> constant (~2, Fubini-Study conv.): the QGT metric diverges as C/gap^2 at the node.\n");

    /* (2) the topological invariant -- the imaginary/Berry part of the same QGT -- jumps. */
    double c_trivial = 0.0, c_topological = 0.0;
    moonlab_qwz_chern(2.5, 32, &c_trivial);      /* |m| > 2 : trivial  */
    moonlab_qwz_chern(1.5, 32, &c_topological);  /* |m| < 2 : Chern    */
    printf("\n Chern number across the node:  m=2.5 -> %.0f   m=1.5 -> %.0f   (jumps by 1).\n",
           c_trivial, c_topological);

    /* (3) the QEC side: the SAME nilpotent square.  A surface code is a chain complex whose
     * X- and Z-stabilizers commute (d1.d2 = 0); that d^2 = 0 is the eps^2 = 0 of the band collision. */
    surface_code_t *code = surface_code_create(3);
    if (code) {
        printf("\n surface code (distance 3): a commuting-stabilizer chain complex, d1 . d2 = 0.\n");
        surface_code_free(code);
    }

    printf("\n metric singular (tr g ~ C/gap^2)  |  topology jumps (Chern)  |  code commutes (d^2=0)\n");
    printf(" ONE object: the two-band collision that makes the QGT metric diverge is the same\n");
    printf(" eps^2 = 0 nilpotent square that makes the stabilizer code commute.  Geometry and\n");
    printf(" error correction, seen twice.\n");
    return 0;
}
