/**
 * @file ca_mps_var_d_chi_scan.c
 * @brief Bond-dim cap scan: at each chi, compare plain-DMRG energy
 *        accuracy to variational-D CA-MPS energy accuracy on TFIM at
 *        criticality.
 *
 * The head-to-head sweep shows var-D's |phi> entropy is much smaller
 * than plain DMRG's |psi> entropy at fixed chi cap.  This experiment
 * asks the *practical* question: at what chi cap does each method
 * achieve a given accuracy threshold?  The headline claim is
 * "var-D needs lower chi than plain DMRG to reach the same energy."
 *
 * Setup:
 *   - TFIM at the quantum critical point (g = 1.0)
 *   - n = 8 (small enough that the largest chi tested gives near-exact
 *     answer for both methods)
 *   - chi cap sweep: 2, 4, 8, 16, 32
 *
 * For each chi:
 *   1. Run plain DMRG with max_bond = chi.  Record E_dmrg(chi).
 *   2. Run var-D alternating optimiser with max_bond = chi (for |phi>
 *      MPS factor).  Record E_varD(chi).
 *   3. Compare: |E - E_ref| where E_ref = plain DMRG at chi=64 (very
 *      well-converged baseline).
 *
 * The variational claim: E_varD(chi) reaches accuracy that
 * E_dmrg(chi') only reaches at chi' >> chi.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void build_tfim(uint32_t n, double g,
                        uint8_t** out_paulis, double** out_coeffs,
                        uint32_t* out_num_terms) {
    uint32_t T = (n - 1) + n;
    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, 1);
    double*  coeffs = (double*)calloc(T, sizeof(double));
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        paulis[k * n + i]     = 3;
        paulis[k * n + i + 1] = 3;
        coeffs[k] = -1.0;
        k++;
    }
    for (uint32_t i = 0; i < n; i++) {
        paulis[k * n + i] = 1;
        coeffs[k] = -g;
        k++;
    }
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = T;
}

static double mps_max_entropy(const tn_mps_state_t* s, uint32_t n) {
    double s_max = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        double e = tn_mps_entanglement_entropy(s, i);
        if (e > s_max) s_max = e;
    }
    return s_max;
}

/* Run plain DMRG at a specific chi cap, return ground energy and
 * |psi> half-cut entropy. */
static void run_plain_dmrg(uint32_t n, double g, uint32_t chi,
                            double* out_E, double* out_S) {
    dmrg_config_t cfg = dmrg_config_default();
    cfg.max_bond_dim = chi;
    cfg.max_sweeps   = 30;
    cfg.energy_tol   = 1e-10;
    cfg.verbose      = false;
    dmrg_result_t* res = NULL;
    tn_mps_state_t* psi = dmrg_tfim_ground_state(n, g, &cfg, &res);
    *out_E = res ? res->ground_energy : 0.0;
    *out_S = psi ? mps_max_entropy(psi, n) : 0.0;
    if (psi) tn_mps_free(psi);
    if (res) dmrg_result_free(res);
}

/* Run var-D CA-MPS at a specific chi cap, trying all four warmstarts
 * and keeping the result with smallest |phi> entropy among those
 * within E_tol of best energy. */
static void run_var_d(uint32_t n, double g, uint32_t chi,
                      const uint8_t* paulis, const double* coeffs,
                      uint32_t T, double* out_E, double* out_S) {
    ca_mps_var_d_alt_config_t base = ca_mps_var_d_alt_config_default();
    base.max_outer_iters           = 30;
    base.imag_time_dtau            = 0.15;
    base.imag_time_steps_per_outer = 3;
    base.clifford_passes_per_outer = 6;
    base.convergence_eps           = 1e-6;
    base.verbose                   = 0;

    /* Try all four warmstarts. */
    moonlab_ca_mps_t* states[4] = {0};
    ca_mps_var_d_alt_result_t results[4] = {0};
    const ca_mps_warmstart_t warms[4] = {
        CA_MPS_WARMSTART_IDENTITY,
        CA_MPS_WARMSTART_H_ALL,
        CA_MPS_WARMSTART_DUAL_TFIM,
        CA_MPS_WARMSTART_FERRO_TFIM,
    };
    for (int i = 0; i < 4; i++) {
        states[i] = moonlab_ca_mps_create(n, chi);
        ca_mps_var_d_alt_config_t cfg = base;
        cfg.warmstart = warms[i];
        if (warms[i] == CA_MPS_WARMSTART_FERRO_TFIM) {
            cfg.imag_time_dtau            = (n <= 8) ? 0.05 : 0.10;
            cfg.imag_time_steps_per_outer = (n <= 8) ? 8 : 4;
            cfg.max_outer_iters           = (n <= 8) ? 60 : 30;
        }
        moonlab_ca_mps_optimize_var_d_alternating(
            states[i], paulis, coeffs, T, &cfg, &results[i]);
    }

    /* Pick smallest entropy among those near the best energy.  Use
     * plain DMRG floor reasoning: any var-D below all DMRG values is
     * a numerical artifact -- but here we don't have that ref handy,
     * so just pick by energy then by entropy with a tolerance. */
    double E_best = results[0].final_energy;
    for (int i = 1; i < 4; i++) {
        if (results[i].final_energy < E_best) E_best = results[i].final_energy;
    }
    const double E_tol = 0.02;
    double S_best = 1e9;
    int idx_best = 0;
    for (int i = 0; i < 4; i++) {
        if (results[i].final_energy <= E_best + E_tol &&
            results[i].final_phi_entropy < S_best) {
            S_best = results[i].final_phi_entropy;
            idx_best = i;
        }
    }

    *out_E = results[idx_best].final_energy;
    *out_S = results[idx_best].final_phi_entropy;

    for (int i = 0; i < 4; i++) moonlab_ca_mps_free(states[i]);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "ca_mps_var_d_chi_scan.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    const uint32_t n = 8;
    const double g = 1.0;
    const uint32_t chi_values[] = { 2, 4, 8, 16, 32 };
    const uint32_t num_chi = sizeof(chi_values)/sizeof(chi_values[0]);

    fprintf(stdout, "=== Bond-dim cap scan: var-D vs plain DMRG ===\n");
    fprintf(stdout, "TFIM N=%u, g=%.2f (critical point)\n\n", n, g);

    /* Reference energy from a very well-converged plain DMRG run. */
    double E_ref, S_ref;
    run_plain_dmrg(n, g, /*chi=*/64, &E_ref, &S_ref);
    fprintf(stdout, "Reference: plain DMRG at chi=64 -> E_ref = %.10f, S = %.4f\n\n",
            E_ref, S_ref);

    fprintf(stdout, "%4s %12s %14s %12s %12s %14s %12s %12s\n",
            "chi", "E_dmrg", "|E_dmrg-E_ref|", "S_psi", "E_varD",
            "|E_varD-E_ref|", "S_phi", "S_phi/S_ref");
    fprintf(stdout, "----------------------------------------------------------------------"
                    "-------------------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_var_d_chi_scan_v1\",\n");
    fprintf(json, "  \"n\": %u, \"g\": %.4f, \"E_ref\": %.6f, \"S_ref\": %.6f,\n",
            n, g, E_ref, S_ref);
    fprintf(json, "  \"points\": [\n");
    int first = 1;

    /* Build TFIM Pauli sum once. */
    uint8_t* paulis;
    double*  coeffs;
    uint32_t T;
    build_tfim(n, g, &paulis, &coeffs, &T);

    for (uint32_t i = 0; i < num_chi; i++) {
        uint32_t chi = chi_values[i];

        double E_dmrg, S_psi;
        run_plain_dmrg(n, g, chi, &E_dmrg, &S_psi);

        double E_var, S_phi;
        run_var_d(n, g, chi, paulis, coeffs, T, &E_var, &S_phi);

        double dE_dmrg = fabs(E_dmrg - E_ref);
        double dE_var  = fabs(E_var  - E_ref);
        double S_ratio = (S_ref > 1e-12) ? (S_phi / S_ref) : 0.0;

        fprintf(stdout, "%4u %12.6f %14.3e %12.4f %12.6f %14.3e %12.4f %12.4f\n",
                chi, E_dmrg, dE_dmrg, S_psi, E_var, dE_var, S_phi, S_ratio);
        fflush(stdout);

        if (!first) fprintf(json, ",\n");
        first = 0;
        fprintf(json, "    { \"chi\": %u, \"E_dmrg\": %.6f, \"dE_dmrg\": %.6e, "
                      "\"S_psi\": %.6f, \"E_varD\": %.6f, \"dE_varD\": %.6e, "
                      "\"S_phi\": %.6f, \"S_ratio\": %.6f }",
                chi, E_dmrg, dE_dmrg, S_psi, E_var, dE_var, S_phi, S_ratio);
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    free(paulis); free(coeffs);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
