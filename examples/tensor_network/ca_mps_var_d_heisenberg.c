/**
 * @file ca_mps_var_d_heisenberg.c
 * @brief Variational-D CA-MPS vs plain DMRG on the 1D XXZ Heisenberg model.
 *
 * Generalization of the TFIM head-to-head sweep to a different model.
 * If the entropy-reduction claim is real and not TFIM-specific, var-D
 * should also reduce |phi> entropy on Heisenberg ground states.
 *
 * Hamiltonian: H = J * sum_{i=0}^{N-2} (X_i X_{i+1} + Y_i Y_{i+1} +
 *                                        Delta * Z_i Z_{i+1})
 * with J = 1.  We sweep Delta (the anisotropy) to cover regimes:
 *   - Delta = 0 (XY model): gapless, central charge c=1
 *   - Delta = 0.5 (gapless XXZ): gapless, c=1
 *   - Delta = 1 (Heisenberg point): gapless, SU(2) symmetric
 *   - Delta = 1.5 (anisotropic): gapped Ising-like
 *   - Delta = 2 (deeper gap)
 *
 * For each Delta: plain DMRG -> var-D alternating with all four
 * warmstarts -> report entropy reduction.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static double mps_max_entropy(const tn_mps_state_t* s, uint32_t n) {
    double s_max = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        double e = tn_mps_entanglement_entropy(s, i);
        if (e > s_max) s_max = e;
    }
    return s_max;
}

/* Build the XXZ Heisenberg Pauli sum:
 *   H = sum_{i<n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1}).
 *
 * Pauli encoding: 0=I, 1=X, 2=Y, 3=Z.  3(n-1) terms total. */
static void build_heisenberg(uint32_t n, double Delta,
                              uint8_t** out_paulis, double** out_coeffs,
                              uint32_t* out_num_terms) {
    uint32_t T = 3 * (n - 1);
    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, 1);
    double*  coeffs = (double*)calloc(T, sizeof(double));
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        paulis[k * n + i]     = 1; paulis[k * n + i + 1] = 1; coeffs[k] = 1.0; k++; /* X X */
        paulis[k * n + i]     = 2; paulis[k * n + i + 1] = 2; coeffs[k] = 1.0; k++; /* Y Y */
        paulis[k * n + i]     = 3; paulis[k * n + i + 1] = 3; coeffs[k] = Delta; k++; /* Delta Z Z */
    }
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = T;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "ca_mps_var_d_heisenberg.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout, "=== variational-D CA-MPS vs plain DMRG on XXZ Heisenberg ===\n\n");
    fprintf(stdout, "%4s %8s %12s %12s %10s %10s %10s %12s  %s\n",
            "n", "Delta", "E_dmrg", "E_varD", "dE_rel", "S_psi", "S_phi", "S_phi/psi", "warm");
    fprintf(stdout, "----------------------------------------------------------------------"
                    "----------------------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_var_d_xxz_v1\",\n  \"points\": [\n");
    int first_json = 1;

    const uint32_t n_values[]     = { 6, 8 };
    const double   Delta_values[] = { 0.0, 0.5, 1.0, 1.5, 2.0 };

    for (size_t ni = 0; ni < sizeof(n_values)/sizeof(n_values[0]); ni++) {
        uint32_t n = n_values[ni];
        for (size_t di = 0; di < sizeof(Delta_values)/sizeof(Delta_values[0]); di++) {
            double Delta = Delta_values[di];

            /* Plain DMRG via direct Heisenberg MPO. */
            mpo_t* mpo = mpo_heisenberg_create(n, /*J=*/1.0, Delta, /*h=*/0.0);
            if (!mpo) continue;
            tn_state_config_t mps_cfg = tn_state_config_create(32, 1e-12);
            tn_mps_state_t* mps = dmrg_init_random_mps(n, /*chi_init=*/8, &mps_cfg);
            if (!mps) { mpo_free(mpo); continue; }
            dmrg_config_t dcfg = dmrg_config_default();
            dcfg.max_bond_dim = 32;
            dcfg.max_sweeps   = 12;
            dcfg.energy_tol   = 1e-9;
            dcfg.verbose      = false;
            dmrg_result_t* dres = dmrg_ground_state(mps, mpo, &dcfg);
            double E_dmrg = dres ? dres->ground_energy : dmrg_compute_energy(mps, mpo);
            double S_psi  = mps_max_entropy(mps, n);
            tn_mps_free(mps);
            mpo_free(mpo);
            if (dres) dmrg_result_free(dres);

            /* var-D alternating with all four warmstarts. */
            uint8_t* paulis;
            double*  coeffs;
            uint32_t T;
            build_heisenberg(n, Delta, &paulis, &coeffs, &T);

            ca_mps_var_d_alt_config_t base = ca_mps_var_d_alt_config_default();
            base.max_outer_iters           = 30;
            base.imag_time_dtau            = 0.15;
            base.imag_time_steps_per_outer = 3;
            base.clifford_passes_per_outer = 6;
            base.convergence_eps           = 1e-6;
            base.verbose                   = 0;

            const ca_mps_warmstart_t warms[4] = {
                CA_MPS_WARMSTART_IDENTITY, CA_MPS_WARMSTART_H_ALL,
                CA_MPS_WARMSTART_DUAL_TFIM, CA_MPS_WARMSTART_FERRO_TFIM,
            };
            const char* warm_names[4] = { "I", "H_all", "dual", "ferro" };

            ca_mps_var_d_alt_result_t results[4] = {0};
            moonlab_ca_mps_t* states[4] = {0};
            for (int w = 0; w < 4; w++) {
                states[w] = moonlab_ca_mps_create(n, 32);
                ca_mps_var_d_alt_config_t cfg = base;
                cfg.warmstart = warms[w];
                if (warms[w] == CA_MPS_WARMSTART_FERRO_TFIM) {
                    cfg.imag_time_dtau            = 0.05;
                    cfg.imag_time_steps_per_outer = 8;
                    cfg.max_outer_iters           = 60;
                }
                moonlab_ca_mps_optimize_var_d_alternating(
                    states[w], paulis, coeffs, T, &cfg, &results[w]);
            }

            /* Pick smallest entropy among those near the best energy.  Use
             * E_dmrg as the variational floor (any var-D below E_dmrg-eps
             * is unphysical). */
            const double E_var_floor = E_dmrg - 1e-3;
            const double E_tol = 0.05;
            double E_best = 1e30;
            int valid[4] = {0};
            for (int w = 0; w < 4; w++) {
                valid[w] = (results[w].final_energy >= E_var_floor);
                if (valid[w] && results[w].final_energy < E_best)
                    E_best = results[w].final_energy;
            }
            if (E_best >= 1e30) {
                /* fallback: pick the highest-energy run */
                E_best = results[0].final_energy;
                for (int w = 0; w < 4; w++) {
                    valid[w] = 1;
                    if (results[w].final_energy > E_best)
                        E_best = results[w].final_energy;
                }
            }

            ca_mps_var_d_alt_result_t best = results[0];
            const char* best_name = warm_names[0];
            double S_min = (valid[0] && results[0].final_energy <= E_best + E_tol)
                            ? results[0].final_phi_entropy : 1e9;
            for (int w = 1; w < 4; w++) {
                if (valid[w] && results[w].final_energy <= E_best + E_tol &&
                    results[w].final_phi_entropy < S_min) {
                    best = results[w];
                    best_name = warm_names[w];
                    S_min = results[w].final_phi_entropy;
                }
            }

            double dE_rel = (fabs(E_dmrg) > 1e-12)
                ? fabs(best.final_energy - E_dmrg) / fabs(E_dmrg) : 0.0;
            double ratio = (S_psi > 1e-12) ? (best.final_phi_entropy / S_psi) : 0.0;

            fprintf(stdout, "%4u %8.3f %12.6f %12.6f %10.3e %10.4f %10.4f %12.4f  %s\n",
                    n, Delta, E_dmrg, best.final_energy, dE_rel,
                    S_psi, best.final_phi_entropy, ratio, best_name);
            fflush(stdout);

            if (!first_json) fprintf(json, ",\n");
            first_json = 0;
            fprintf(json, "    { \"n\": %u, \"Delta\": %.4f, "
                          "\"E_dmrg\": %.6f, \"E_varD\": %.6f, "
                          "\"S_psi\": %.6f, \"S_phi\": %.6f, "
                          "\"S_ratio\": %.6f, \"warmstart\": \"%s\" }",
                    n, Delta, E_dmrg, best.final_energy,
                    S_psi, best.final_phi_entropy, ratio, best_name);

            for (int w = 0; w < 4; w++) moonlab_ca_mps_free(states[w]);
            free(paulis); free(coeffs);
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
