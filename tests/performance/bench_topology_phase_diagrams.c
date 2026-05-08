/**
 * @file bench_topology_phase_diagrams.c
 * @brief Phase-diagram sweep across all six topological models in the
 *        QGT module.
 *
 * For each model, sweeps the phase parameter through a grid and
 * records the topological invariant (Chern or Z_2) at every point.
 * Emits a JSON archive that downstream consumers (QGTL,
 * paper-figure scripts) can read for the canonical phase diagrams.
 *
 * Models swept:
 *   1D:
 *     - SSH (t1=1, t2 in [0.0, 2.0]):   winding W in {0, 1}
 *     - Kitaev chain (t=1, delta=0.5, mu in [-3, 3]):
 *                                        Z_2 in {0, 1}
 *
 *   2D 2-band:
 *     - QWZ (m in [-3, 3]):              Chern in {-1, 0, +1}
 *     - Haldane (t1=1, t2=0.06, phi=pi/2, M in [-1, 1]):
 *                                        Chern in {-1, 0, +1}
 *
 *   2D 4-band TRS:
 *     - Kane-Mele (t=1, lambda_so=0.06, lambda_r=0,
 *                  lambda_v in [0, 0.6]):  Z_2 in {0, 1}
 *     - BHZ (A=B=1, M in [-2, 10]):      Z_2 in {0, 1}
 *
 * Invoke: ./build_release/bench_topology_phase_diagrams [output.json]
 * Default JSON path: benchmarks/results/topology_phase_diagrams_TODAY.json
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    double param;
    int    invariant;
} sweep_record_t;

static int sweep_qwz(sweep_record_t* recs, size_t cap, size_t* n) {
    const double m_lo = -3.0, m_hi = 3.0;
    const size_t K = 31;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double m = m_lo + (m_hi - m_lo) * (double)i / (double)(K - 1);
        qgt_system_t* sys = qgt_model_qwz(m);
        qgt_berry_grid_t g;
        if (qgt_berry_grid(sys, 32, &g) != 0) { qgt_free(sys); continue; }
        recs[*n].name = "QWZ";
        recs[*n].param = m;
        recs[*n].invariant = (int)lround(g.chern);
        (*n)++;
        qgt_berry_grid_free(&g);
        qgt_free(sys);
    }
    return 0;
}

static int sweep_haldane(sweep_record_t* recs, size_t cap, size_t* n) {
    const double M_lo = -1.0, M_hi = 1.0;
    const size_t K = 21;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double M = M_lo + (M_hi - M_lo) * (double)i / (double)(K - 1);
        qgt_system_t* sys =
            qgt_model_haldane(1.0, 0.06, 0.5 * M_PI, M);
        qgt_berry_grid_t g;
        if (qgt_berry_grid(sys, 48, &g) != 0) { qgt_free(sys); continue; }
        recs[*n].name = "Haldane";
        recs[*n].param = M;
        recs[*n].invariant = (int)lround(g.chern);
        (*n)++;
        qgt_berry_grid_free(&g);
        qgt_free(sys);
    }
    return 0;
}

static int sweep_kane_mele(sweep_record_t* recs, size_t cap, size_t* n) {
    const double lv_lo = 0.0, lv_hi = 0.6;
    const size_t K = 13;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double lv = lv_lo + (lv_hi - lv_lo) * (double)i / (double)(K - 1);
        qgt_system_n_t* sys = qgt_model_kane_mele(1.0, 0.06, 0.0, lv);
        int z2 = -1;
        if (qgt_z2_invariant(sys, 48, &z2) != 0) { qgt_free_nband(sys); continue; }
        recs[*n].name = "Kane-Mele";
        recs[*n].param = lv;
        recs[*n].invariant = z2;
        (*n)++;
        qgt_free_nband(sys);
    }
    return 0;
}

static int sweep_bhz(sweep_record_t* recs, size_t cap, size_t* n) {
    const double M_lo = -2.0, M_hi = 10.0;
    const size_t K = 25;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double M = M_lo + (M_hi - M_lo) * (double)i / (double)(K - 1);
        qgt_system_n_t* sys = qgt_model_bhz(1.0, 1.0, M);
        int z2 = -1;
        if (qgt_z2_invariant(sys, 32, &z2) != 0) { qgt_free_nband(sys); continue; }
        recs[*n].name = "BHZ";
        recs[*n].param = M;
        recs[*n].invariant = z2;
        (*n)++;
        qgt_free_nband(sys);
    }
    return 0;
}

static int sweep_ssh(sweep_record_t* recs, size_t cap, size_t* n) {
    const double t2_lo = 0.0, t2_hi = 2.0;
    const size_t K = 21;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double t2 = t2_lo + (t2_hi - t2_lo) * (double)i / (double)(K - 1);
        qgt_system_1d_t* sys = qgt_model_ssh(1.0, t2);
        double w_raw = 0.0;
        int W = qgt_winding_1d(sys, 64, &w_raw);
        recs[*n].name = "SSH";
        recs[*n].param = t2;
        recs[*n].invariant = W;
        (*n)++;
        qgt_free_1d(sys);
    }
    return 0;
}

static int sweep_kitaev(sweep_record_t* recs, size_t cap, size_t* n) {
    const double mu_lo = -3.0, mu_hi = 3.0;
    const size_t K = 25;
    for (size_t i = 0; i < K && *n < cap; i++) {
        double mu = mu_lo + (mu_hi - mu_lo) * (double)i / (double)(K - 1);
        qgt_system_1d_t* sys = qgt_model_kitaev_chain(1.0, mu, 0.5);
        int z2 = -1;
        qgt_z2_invariant_1d_bdg(sys, &z2);
        recs[*n].name = "Kitaev";
        recs[*n].param = mu;
        recs[*n].invariant = z2;
        (*n)++;
        qgt_free_1d(sys);
    }
    return 0;
}

static void print_table(const sweep_record_t* recs, size_t n) {
    const char* current = NULL;
    int run_inv = INT32_MIN;
    double run_lo = 0.0, run_hi = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (current == NULL || strcmp(recs[i].name, current) != 0
            || recs[i].invariant != run_inv) {
            if (current != NULL) {
                printf("    [%-7.3f, %-7.3f]  invariant = %+d\n",
                       run_lo, run_hi, run_inv);
            }
            if (current == NULL || strcmp(recs[i].name, current) != 0) {
                printf("\n%s\n", recs[i].name);
                current = recs[i].name;
            }
            run_inv = recs[i].invariant;
            run_lo = recs[i].param;
        }
        run_hi = recs[i].param;
    }
    if (current != NULL) {
        printf("    [%-7.3f, %-7.3f]  invariant = %+d\n",
               run_lo, run_hi, run_inv);
    }
}

static void emit_json(FILE* f, const sweep_record_t* recs, size_t n) {
    fprintf(f, "{\n");
    fprintf(f, "  \"harness\": \"bench_topology_phase_diagrams\",\n");
    fprintf(f, "  \"description\": \"Topological-invariant sweeps for the QGT module's six built-in models\",\n");
    fprintf(f, "  \"records\": [\n");
    for (size_t i = 0; i < n; i++) {
        fprintf(f, "    {\"model\":\"%s\", \"param\":%.10g, \"invariant\":%d}%s\n",
                recs[i].name, recs[i].param, recs[i].invariant,
                (i + 1 < n) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
}

int main(int argc, char** argv) {
    sweep_record_t recs[256];
    size_t n_rec = 0;

    sweep_qwz      (recs, 256, &n_rec);
    sweep_haldane  (recs, 256, &n_rec);
    sweep_kane_mele(recs, 256, &n_rec);
    sweep_bhz      (recs, 256, &n_rec);
    sweep_ssh      (recs, 256, &n_rec);
    sweep_kitaev   (recs, 256, &n_rec);

    printf("Topology phase-diagram sweep summary (%zu records):\n", n_rec);
    print_table(recs, n_rec);

    if (argc >= 2) {
        FILE* f = fopen(argv[1], "w");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        emit_json(f, recs, n_rec);
        fclose(f);
        printf("\nJSON archive: %s (%zu records)\n", argv[1], n_rec);
    }
    return 0;
}
