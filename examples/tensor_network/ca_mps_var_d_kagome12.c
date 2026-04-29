/**
 * @file ca_mps_var_d_kagome12.c
 * @brief Variational-D CA-MPS on the kagome 12-site cluster.
 *
 * The hardest test of the var-D claim: kagome AFM Heisenberg has a
 * spin-liquid ground state where plain MPS bond requirements grow
 * fast (chi >= 256 needed for ~0.01 J convergence at PRB-reference
 * energy -5.444875).  Per the §6.4 Heisenberg negative result,
 * SU(2)-symmetric ground states resist var-D entropy reduction --
 * but kagome's frustration may inject enough Ising character to
 * make the dual Clifford useful.
 *
 * Hamiltonian: H = sum_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j) (Pauli
 * convention; PRB E_0 = -5.444875 corresponds to this normalisation).
 * 24 bonds on the 6x2 kagome torus.
 *
 * For each warmstart (I, H_all, dual, ferro): run var-D alternating,
 * report energy + entropy.  Compare to the D=I baseline (plain CA-MPS
 * with imag-time), which is essentially plain MPS.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/lattice_2d.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t collect_unique_bonds(const lattice_2d_t* lat, uint32_t (*bonds)[2]) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < lat->num_sites; i++) {
        for (uint32_t k = 0; k < lat->num_neighbors[i]; k++) {
            uint32_t j = lat->neighbors[i][k].site;
            if (j <= i) continue;
            int dup = 0;
            for (uint32_t m = 0; m < n; m++) {
                if (bonds[m][0] == i && bonds[m][1] == j) { dup = 1; break; }
            }
            if (dup) continue;
            bonds[n][0] = i; bonds[n][1] = j;
            n++;
        }
    }
    return n;
}

static void build_kagome_paulis(uint32_t n, const uint32_t (*bonds)[2],
                                  uint32_t num_bonds,
                                  uint8_t** out_paulis, double** out_coeffs,
                                  uint32_t* out_nterms) {
    uint32_t nt = 3 * num_bonds;
    uint8_t* paulis = (uint8_t*)calloc((size_t)nt * n, sizeof(uint8_t));
    double* coeffs = (double*)calloc(nt, sizeof(double));
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (uint8_t p = 1; p <= 3; p++) {
            uint32_t k = 3 * b + (p - 1);
            paulis[(size_t)k * n + i] = p;
            paulis[(size_t)k * n + j] = p;
            /* Pauli convention: J_spin=1 corresponds to coefficient 1
             * (PRB reference uses S.S = 0.25*(XX+YY+ZZ) so PRB J=1 maps
             * to our coefficient = 1; the resulting E_PRB = -5.4449 is
             * for the spin model, so our Pauli energy = 4 * E_PRB.
             * Keep coefficient = 1.0 for the Pauli sum, expect Pauli
             * GS energy = -21.7795). */
            coeffs[k] = 1.0;
        }
    }
    *out_paulis = paulis; *out_coeffs = coeffs; *out_nterms = nt;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "ca_mps_var_d_kagome12.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout, "=== variational-D CA-MPS on kagome 12-site cluster ===\n\n");

    lattice_2d_t* lat = lattice_2d_create(6, 2, LATTICE_KAGOME, BC_PERIODIC_XY);
    if (!lat || lat->num_sites != 12) {
        fprintf(stderr, "lattice setup failed\n");
        if (lat) lattice_2d_free(lat); return 1;
    }
    uint32_t (*bonds)[2] = calloc(36, sizeof(*bonds));
    uint32_t num_bonds = collect_unique_bonds(lat, bonds);
    fprintf(stdout, "  num_bonds = %u (expect 24)\n", num_bonds);

    uint8_t* paulis;
    double*  coeffs;
    uint32_t T;
    build_kagome_paulis(12, bonds, num_bonds, &paulis, &coeffs, &T);
    fprintf(stdout, "  Pauli terms = %u (24 bonds * 3 Pauli pairs)\n", T);
    fprintf(stdout, "  Reference: PRB 83, 212401 cluster 12 (Pauli) "
                    "E_GS = -21.7795 (Pauli) = -5.4449 (spin convention)\n\n");

    fprintf(stdout, "%-8s %14s %12s %10s %10s %12s %10s\n",
            "warm", "E_varD", "E_per_spin", "S(phi)", "iters", "gates", "wall_s");
    fprintf(stdout, "----------------------------------------------------------------------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_kagome12_var_d_v1\",\n  \"points\": [\n");
    int first = 1;

    const ca_mps_warmstart_t warms[4] = {
        CA_MPS_WARMSTART_IDENTITY, CA_MPS_WARMSTART_H_ALL,
        CA_MPS_WARMSTART_DUAL_TFIM, CA_MPS_WARMSTART_FERRO_TFIM,
    };
    const char* warm_names[4] = { "I", "H_all", "dual", "ferro" };

    /* Baseline: var-D with the IDENTITY warmstart is essentially plain
     * MPS imag-time (D never moves from I).  The S(phi) for this run
     * is the plain-MPS half-cut entropy of the converged kagome GS.
     * Other warmstarts modify D and may reduce S(phi).
     *
     * Use generous chi -- kagome GS at chi=64 is the realistic regime
     * (kagome at chi=32 is already a useful reduction over chi=256). */

    for (int w = 0; w < 4; w++) {
        moonlab_ca_mps_t* state = moonlab_ca_mps_create(12, /*max_bond=*/64);
        ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
        cfg.warmstart                   = warms[w];
        cfg.max_outer_iters             = 25;
        cfg.imag_time_dtau              = 0.10;
        cfg.imag_time_steps_per_outer   = 4;
        cfg.clifford_passes_per_outer   = 4;
        cfg.convergence_eps             = 1e-5;
        cfg.verbose                     = 0;

        ca_mps_var_d_alt_result_t res = {0};
        double t0 = now_s();
        moonlab_ca_mps_optimize_var_d_alternating(
            state, paulis, coeffs, T, &cfg, &res);
        double dt = now_s() - t0;

        double E_per_spin = res.final_energy / 4.0;  /* Pauli -> spin convention factor */

        fprintf(stdout, "%-8s %14.6f %12.6f %10.4f %10d %12d %10.2f\n",
                warm_names[w], res.final_energy, E_per_spin,
                res.final_phi_entropy, res.outer_iterations,
                res.total_gates_added, dt);
        fflush(stdout);

        if (!first) fprintf(json, ",\n");
        first = 0;
        fprintf(json, "    { \"warmstart\": \"%s\", \"E_varD\": %.6f, "
                      "\"E_per_spin_PRB_convention\": %.6f, "
                      "\"S_phi\": %.6f, \"iters\": %d, \"gates\": %d, "
                      "\"wall_s\": %.4f }",
                warm_names[w], res.final_energy, E_per_spin,
                res.final_phi_entropy, res.outer_iterations,
                res.total_gates_added, dt);

        moonlab_ca_mps_free(state);
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);

    free(paulis); free(coeffs);
    free(bonds);
    lattice_2d_free(lat);
    return 0;
}
