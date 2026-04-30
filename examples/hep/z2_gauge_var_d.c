/**
 * @file z2_gauge_var_d.c
 * @brief Variational-D CA-MPS on 1+1D Z2 lattice gauge theory.
 *
 * Demonstrates the headline mathematical claim of the var-D + LGT
 * bridge: the Gauss-law constraints G_x are stabilizer operators,
 * and var-D's Clifford prefactor D absorbs them, driving the MPS
 * factor |phi>'s effective entanglement well below the plain-MPS
 * value.
 *
 * Sweeps over the electric-field strength h at fixed t = 1, m = 0.5,
 * lambda = 5, comparing for each h:
 *   - Plain DMRG ground state |psi_GS> via dmrg_ground_state on the
 *     Pauli-sum -> MPO (constructed below).
 *   - var-D alternating optimiser on the same Pauli sum; reports
 *     S(|phi>) and the residual Gauss-law violation
 *     <psi|sum_x (I - G_x)|psi>.
 *
 * Headline: at all h, var-D's |phi> entropy should be substantially
 * below plain DMRG's |psi> entropy because the Gauss-law penalty
 * generates stabilizer-rich entanglement that D absorbs.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/applications/hep/lattice_z2_1d.h"

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double mps_max_half_cut_entropy(const tn_mps_state_t* s, uint32_t n) {
    double s_max = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        double e = tn_mps_entanglement_entropy(s, i);
        if (e > s_max) s_max = e;
    }
    return s_max;
}

/* Compute energy of a CA-MPS state under a Pauli sum (real coeffs). */
static double ca_mps_pauli_energy(const moonlab_ca_mps_t* s,
                                    const uint8_t* paulis,
                                    const double* coeffs,
                                    uint32_t num_terms,
                                    uint32_t n) {
    (void)n;
    double _Complex* cz = (double _Complex*)calloc(num_terms, sizeof(double _Complex));
    for (uint32_t i = 0; i < num_terms; i++) cz[i] = (double _Complex)coeffs[i];
    double _Complex out = 0.0;
    moonlab_ca_mps_expect_pauli_sum(s, paulis, cz, num_terms, &out);
    free(cz);
    return creal(out);
}

/* Compute mean Gauss-law violation <(N-2) - sum_{interior x} G_x> on a CA-MPS state. */
static double ca_mps_gauss_violation(const moonlab_ca_mps_t* s,
                                      const z2_lgt_config_t* cfg) {
    const uint32_t N = cfg->num_matter_sites;
    if (N < 3) return 0.0;
    const uint32_t nq = 2u * N - 1u;
    uint8_t* p = (uint8_t*)calloc(nq, 1);
    double total = 0.0;
    for (uint32_t x = 1; x + 1 < N; x++) {
        z2_lgt_1d_gauss_law_pauli(cfg, x, p);
        double _Complex e = 0;
        moonlab_ca_mps_expect_pauli(s, p, &e);
        total += 1.0 - creal(e);    /* (1 - <G_x>) */
    }
    free(p);
    return total;   /* total Gauss-law violation across interior sites */
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "z2_gauge_var_d.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    const uint32_t N         = 4;        /* matter sites; total qubits = 7 */
    const double  t_hop      = 1.0;
    const double  mass       = 0.5;
    const double  gauss_lam  = 5.0;
    const double  h_values[] = { 0.25, 0.5, 1.0, 1.5, 2.0 };

    fprintf(stdout, "=== variational-D CA-MPS on 1+1D Z2 lattice gauge theory ===\n");
    fprintf(stdout, "N = %u matter sites, total qubits = %u, lambda = %.1f\n",
            N, 2 * N - 1, gauss_lam);
    fprintf(stdout, "Gauss-law constraint G_x = X_{2x-1} X_{2x+1} Z_{2x} on interior x.\n\n");

    fprintf(stdout, "%6s %12s %10s %10s %12s %14s %10s\n",
            "h", "E_varD", "S_phi", "S_psi_max", "ratio", "Gauss_viol", "wall_s");
    fprintf(stdout, "----------------------------------------------------------------------------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/z2_lgt_var_d_v1\",\n");
    fprintf(json, "  \"N_matter\": %u, \"t_hop\": %.4f, \"mass\": %.4f, "
                  "\"gauss_lambda\": %.4f,\n",
            N, t_hop, mass, gauss_lam);
    fprintf(json, "  \"points\": [\n");
    int first = 1;

    for (size_t hi = 0; hi < sizeof(h_values)/sizeof(h_values[0]); hi++) {
        double h = h_values[hi];

        z2_lgt_config_t cfg = { .num_matter_sites = N,
                                  .t_hop = t_hop,
                                  .h_link = h,
                                  .mass = mass,
                                  .gauss_penalty = gauss_lam };

        uint8_t*  paulis;
        double*   coeffs;
        uint32_t  T, nq;
        if (z2_lgt_1d_build_pauli_sum(&cfg, &paulis, &coeffs, &T, &nq) != 0) {
            fprintf(stderr, "build_pauli_sum failed\n"); continue;
        }

        /* Plain MPS reference: run a CA-MPS with D=I for many imag-time
         * steps -- this is an MPS evolution under H since D never moves
         * from the identity. */
        moonlab_ca_mps_t* st_plain = moonlab_ca_mps_create(nq, /*max_bond=*/32);
        ca_mps_var_d_alt_config_t cfg_plain = ca_mps_var_d_alt_config_default();
        cfg_plain.warmstart                   = CA_MPS_WARMSTART_IDENTITY;
        cfg_plain.max_outer_iters             = 20;
        cfg_plain.imag_time_dtau              = 0.10;
        cfg_plain.imag_time_steps_per_outer   = 4;
        cfg_plain.clifford_passes_per_outer   = 0;   /* freeze D = I */
        cfg_plain.convergence_eps             = 1e-6;
        cfg_plain.verbose                     = 0;
        ca_mps_var_d_alt_result_t res_plain = {0};
        moonlab_ca_mps_optimize_var_d_alternating(
            st_plain, paulis, coeffs, T, &cfg_plain, &res_plain);
        double E_plain = res_plain.final_energy;
        double S_psi_max = res_plain.final_phi_entropy;   /* with D=I, this is S(|psi>) */

        /* var-D run: same Pauli sum, but allow Clifford prefactor to evolve. */
        moonlab_ca_mps_t* st_vd = moonlab_ca_mps_create(nq, /*max_bond=*/32);
        ca_mps_var_d_alt_config_t cfg_vd = ca_mps_var_d_alt_config_default();
        cfg_vd.warmstart                   = CA_MPS_WARMSTART_IDENTITY;
        cfg_vd.max_outer_iters             = 25;
        cfg_vd.imag_time_dtau              = 0.10;
        cfg_vd.imag_time_steps_per_outer   = 4;
        cfg_vd.clifford_passes_per_outer   = 8;
        cfg_vd.composite_2gate             = 1;   /* useful here -- gauge constraints
                                                      need 2-gate moves to absorb */
        cfg_vd.convergence_eps             = 1e-6;
        cfg_vd.verbose                     = 0;
        ca_mps_var_d_alt_result_t res_vd = {0};
        double t0 = now_s();
        moonlab_ca_mps_optimize_var_d_alternating(
            st_vd, paulis, coeffs, T, &cfg_vd, &res_vd);
        double dt = now_s() - t0;

        double E_vd = res_vd.final_energy;
        double S_phi = res_vd.final_phi_entropy;
        double ratio = (S_psi_max > 1e-12) ? (S_phi / S_psi_max) : 0.0;
        double gauss_viol = ca_mps_gauss_violation(st_vd, &cfg);

        fprintf(stdout, "%6.2f %12.6f %10.4f %10.4f %12.4f %14.3e %10.2f\n",
                h, E_vd, S_phi, S_psi_max, ratio, gauss_viol, dt);
        fflush(stdout);

        (void)E_plain;   /* available if a future row needs it */

        if (!first) fprintf(json, ",\n");
        first = 0;
        fprintf(json, "    { \"h\": %.4f, \"E_varD\": %.6f, \"S_phi\": %.6f, "
                      "\"S_psi_max\": %.6f, \"S_ratio\": %.6f, "
                      "\"gauss_violation\": %.6e, \"wall_s\": %.4f }",
                h, E_vd, S_phi, S_psi_max, ratio, gauss_viol, dt);

        moonlab_ca_mps_free(st_plain);
        moonlab_ca_mps_free(st_vd);
        free(paulis); free(coeffs);
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
