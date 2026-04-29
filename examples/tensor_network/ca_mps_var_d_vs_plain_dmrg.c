/**
 * @file ca_mps_var_d_vs_plain_dmrg.c
 * @brief Head-to-head: variational-D CA-MPS vs plain DMRG, TFIM phase
 *        sweep.
 *
 * For each (n, g), runs both
 *   - plain DMRG: dmrg_tfim_ground_state, reports E and S(|psi>);
 *   - variational-D CA-MPS: alternating optimiser starting from
 *     D=I, |phi>=|0...0>, reports E and S(|phi>).
 *
 * Headline metric: the entropy ratio S(|phi>)_varD / S(|psi>)_plain.
 * The variational claim is that this ratio is < 1 across the phase
 * diagram -- |phi> has less entanglement than |psi> by construction
 * (Clifford prefactor absorbs alignable entanglement), which means
 * a CA-MPS bond-bounded ansatz can fit a more entangled physical
 * state than plain MPS at the same chi.
 *
 * This is the paper figure: a phase-diagram sweep showing the
 * entropy advantage holds in the ferromagnetic, critical, and
 * paramagnetic regimes.
 *
 * Output: human-readable table to stdout + JSON to argv[1] (default
 * ca_mps_var_d_vs_plain_dmrg.json).
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double tfim_exact_energy_obc(int N, double g) {
    double E = 0.0;
    for (int j = 0; j < N; j++) {
        double k = M_PI * (2 * j + 1) / (double)(2 * N + 1);
        E -= sqrt(1.0 + g * g - 2.0 * g * cos(k));
    }
    return E;
}

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

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "ca_mps_var_d_vs_plain_dmrg.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout, "=== CA-MPS variational-D vs plain DMRG (TFIM phase sweep) ===\n");
    fprintf(stdout, "\nMetric: S_phi / S_psi -- ratio of |phi> half-cut entropy under\n");
    fprintf(stdout, "var-D CA-MPS to |psi> half-cut entropy under plain DMRG.  Lower\n");
    fprintf(stdout, "is better; ratio < 1 means CA-MPS represents the same physical\n");
    fprintf(stdout, "state with less entanglement, hence at lower bond dim.\n\n");

    fprintf(stdout, "%4s %6s %14s %14s %12s %10s %10s %10s\n",
            "n", "g", "E_exact", "E_varD", "rel_err",
            "S_psi", "S_phi", "S_phi/psi");
    fprintf(stdout, "------------------------------------------------------------------------"
                    "----------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_var_d_vs_plain_dmrg_v1\",\n");
    fprintf(json, "  \"points\": [\n");
    int first_json = 1;

    const uint32_t n_values[] = { 6, 8 };
    const double   g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.5 };

    for (size_t ni = 0; ni < sizeof(n_values)/sizeof(n_values[0]); ni++) {
        uint32_t n = n_values[ni];
        for (size_t gi = 0; gi < sizeof(g_values)/sizeof(g_values[0]); gi++) {
            double g = g_values[gi];

            /* Plain DMRG ground state. */
            dmrg_config_t cfg = dmrg_config_default();
            cfg.max_bond_dim = 32;
            cfg.max_sweeps   = 12;
            cfg.energy_tol   = 1e-8;
            cfg.verbose      = false;
            dmrg_result_t* dres = NULL;
            tn_mps_state_t* psi = dmrg_tfim_ground_state(n, g, &cfg, &dres);
            if (!psi) { fprintf(stderr, "DMRG failed at n=%u g=%.2f\n", n, g); continue; }
            double E_dmrg = dres ? dres->ground_energy : 0.0;
            double S_psi  = mps_max_entropy(psi, n);
            tn_mps_free(psi);
            if (dres) dmrg_result_free(dres);

            /* Variational-D CA-MPS. */
            uint8_t*  paulis;
            double*   coeffs;
            uint32_t  T;
            build_tfim(n, g, &paulis, &coeffs, &T);

            moonlab_ca_mps_t* state = moonlab_ca_mps_create(n, /*max_bond=*/32);
            ca_mps_var_d_alt_config_t acfg = ca_mps_var_d_alt_config_default();
            acfg.max_outer_iters           = 30;
            acfg.imag_time_dtau            = 0.15;
            acfg.imag_time_steps_per_outer = 3;
            acfg.clifford_passes_per_outer = 6;
            acfg.convergence_eps           = 1e-6;
            acfg.verbose                   = 0;

            ca_mps_var_d_alt_result_t ares = {0};
            moonlab_ca_mps_optimize_var_d_alternating(
                state, paulis, coeffs, T, &acfg, &ares);

            double E_exact = tfim_exact_energy_obc((int)n, g);
            double rel = fabs(ares.final_energy - E_exact) / fabs(E_exact);

            double ratio = (S_psi > 1e-12) ? (ares.final_phi_entropy / S_psi) : 0.0;
            fprintf(stdout, "%4u %6.3f %14.6f %14.6f %11.3e  %10.4f %10.4f %10.4f\n",
                    n, g, E_exact, ares.final_energy, rel,
                    S_psi, ares.final_phi_entropy, ratio);
            fflush(stdout);

            if (!first_json) fprintf(json, ",\n");
            first_json = 0;
            fprintf(json, "    { \"n\": %u, \"g\": %.4f, "
                          "\"E_exact\": %.6f, \"E_dmrg\": %.6f, "
                          "\"E_varD\": %.6f, "
                          "\"S_psi\": %.6f, \"S_phi\": %.6f, "
                          "\"S_ratio\": %.6f, "
                          "\"varD_outer_iters\": %d, "
                          "\"varD_gates\": %d }",
                    n, g, E_exact, E_dmrg, ares.final_energy,
                    S_psi, ares.final_phi_entropy, ratio,
                    ares.outer_iterations, ares.total_gates_added);

            moonlab_ca_mps_free(state);
            free(paulis);
            free(coeffs);
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
