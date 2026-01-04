/**
 * @file test_dmrg.c
 * @brief Test for DMRG ground state finding with exact energy comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src/algorithms/tensor_network/dmrg.h"
#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Compute exact TFIM ground state energy using Jordan-Wigner
 *
 * For OBC, the exact energy is:
 * E = -sum_k epsilon_k where epsilon_k = sqrt(1 + g^2 - 2g*cos(k))
 * and k = pi*(2n+1)/(2N+1) for n = 0, 1, ..., N-1
 */
double tfim_exact_energy_obc(int N, double g) {
    double E = 0.0;
    for (int n = 0; n < N; n++) {
        double k = M_PI * (2*n + 1) / (2*N + 1);
        double eps_k = sqrt(1 + g*g - 2*g*cos(k));
        E -= eps_k;
    }
    return E;
}

int main(int argc, char *argv[]) {
    int num_sites = 8;
    double g = 1.0;  // Critical point
    int max_bond = 64;

    if (argc > 1) num_sites = atoi(argv[1]);
    if (argc > 2) g = atof(argv[2]);
    if (argc > 3) max_bond = atoi(argv[3]);

    printf("DMRG Accuracy Test: %d sites, g = %.2f\n", num_sites, g);
    printf("=========================================\n\n");

    // Compute exact energy for comparison
    double E_exact = tfim_exact_energy_obc(num_sites, g);
    printf("Exact ground state energy (Jordan-Wigner): %.10f\n", E_exact);
    printf("Exact E/N: %.10f\n\n", E_exact / num_sites);

    // Configure DMRG with new production settings
    dmrg_config_t config = dmrg_config_default();
    config.max_bond_dim = max_bond;
    config.max_sweeps = 30;
    config.energy_tol = 1e-10;
    config.verbose = true;

    // Enable all accuracy improvements
    config.noise_strength = 1e-4;
    config.noise_decay = 0.5;
    config.dm_perturbation = 1e-6;
    config.warmup_sweeps = 3;
    config.warmup_noise = 1e-3;
    config.warmup_bond_dim = 16;

    printf("Running DMRG with max_bond_dim=%u, max_sweeps=%u...\n",
           config.max_bond_dim, config.max_sweeps);
    printf("  Subspace expansion: noise=%.1e, decay=%.1f\n",
           config.noise_strength, config.noise_decay);
    printf("  Density matrix perturbation: %.1e\n", config.dm_perturbation);
    printf("  Warmup: %u sweeps with noise=%.1e\n\n",
           config.warmup_sweeps, config.warmup_noise);

    // Find ground state
    dmrg_result_t *result = NULL;
    tn_mps_state_t *ground_state = dmrg_tfim_ground_state(num_sites, g, &config, &result);

    if (!ground_state) {
        fprintf(stderr, "DMRG failed!\n");
        return 1;
    }

    printf("\n=========================================\n");
    printf("DMRG Results:\n");
    printf("  Ground energy: %.10f\n", result->ground_energy);
    printf("  Sweeps: %u\n", result->num_sweeps);
    printf("  Converged: %s\n", result->converged ? "yes" : "no");
    printf("  Time: %.2f seconds\n", result->total_time);
    printf("\n");

    // Compute observables
    printf("Physical observables:\n");

    // Order parameter
    double order_param = 0.0;
    for (uint32_t i = 0; i < (uint32_t)num_sites; i++) {
        order_param += fabs(tn_expectation_z(ground_state, i));
    }
    order_param /= num_sites;
    printf("  Order parameter <|Z|>: %.6f\n", order_param);

    // ZZ correlation
    double zz = 0.0;
    for (uint32_t i = 0; i < (uint32_t)num_sites - 1; i++) {
        zz += tn_expectation_zz(ground_state, i, i + 1);
    }
    zz /= (num_sites - 1);
    printf("  NN ZZ correlation: %.6f\n", zz);

    // Entanglement entropy at center
    double S = tn_mps_entanglement_entropy(ground_state, num_sites / 2);
    printf("  S(L/2): %.6f\n", S);

    // Accuracy comparison
    double error = fabs(result->ground_energy - E_exact);
    double rel_error = error / fabs(E_exact) * 100.0;

    printf("\n--- ACCURACY COMPARISON ---\n");
    printf("  Exact energy:  %.10f\n", E_exact);
    printf("  DMRG energy:   %.10f\n", result->ground_energy);
    printf("  Absolute error: %.2e\n", error);
    printf("  Relative error: %.4f%%\n", rel_error);

    printf("\n=========================================\n");
    if (rel_error < 1.0) {
        printf("EXCELLENT: Error < 1%% - DMRG accuracy is production-ready!\n");
    } else if (rel_error < 5.0) {
        printf("GOOD: Error < 5%% - Acceptable for most applications.\n");
    } else {
        printf("NEEDS IMPROVEMENT: Error > 5%% - Consider higher bond dimension.\n");
    }

    // Cleanup
    dmrg_result_free(result);
    tn_mps_free(ground_state);

    return 0;
}
