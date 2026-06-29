/**
 * @file test_long_evolution.c
 * @brief Test longer evolution to see system-size dependence
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "src/algorithms/tensor_network/tensor.h"
#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_gates.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

static tn_gate_1q_t create_imag_time_x(double tau_h) {
    tn_gate_1q_t gate;
    double c = cosh(tau_h);
    double s = sinh(tau_h);
    gate.elements[0][0] = c;
    gate.elements[0][1] = s;
    gate.elements[1][0] = s;
    gate.elements[1][1] = c;
    return gate;
}

static tn_gate_2q_t create_imag_time_zz(double tau_J) {
    tn_gate_2q_t gate = {{{0}}};
    double ep = exp(tau_J);
    double em = exp(-tau_J);
    gate.elements[0][0] = ep;
    gate.elements[1][1] = em;
    gate.elements[2][2] = em;
    gate.elements[3][3] = ep;
    return gate;
}

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000) return fallback;
    return (int)parsed;
}

void test_size(uint32_t n_qubits, int n_steps) {
    printf("\n=== Testing N=%u qubits, %d steps ===\n", n_qubits, n_steps);
    fflush(stdout);

    const double J = 1.0;
    const double h = 0.5;
    const double tau = 0.1;
    const uint32_t max_bond = 64;

    tn_state_config_t config = {
        .max_bond_dim = max_bond,
        .svd_cutoff = 1e-12,
        .max_truncation_error = 1e-10,
        .track_truncation = true,
        .auto_canonicalize = false,
        .target_form = TN_CANONICAL_NONE
    };

    tn_mps_state_t *mps = tn_mps_create_zero(n_qubits, &config);
    if (!mps) {
        printf("  ERROR: Failed to create MPS\n");
        return;
    }

    double tilt = 0.05;
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    tn_gate_2q_t zz_gate = create_imag_time_zz(tau * J);
    tn_gate_1q_t x_gate = create_imag_time_x(tau * h);

    printf("  Step   Z_bulk   ZZ_bulk  max_chi\n");

    for (int step = 0; step <= n_steps; step++) {
        if (step > 0) {
            for (uint32_t i = 0; i < n_qubits - 1; i += 2) {
                double trunc_err = 0.0;
                tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            }
            for (uint32_t i = 1; i < n_qubits - 1; i += 2) {
                double trunc_err = 0.0;
                tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            }
            for (uint32_t i = 0; i < n_qubits; i++) {
                tn_apply_gate_1q(mps, i, &x_gate);
            }
        }

        if (step % 10 == 0 || step == n_steps) {
            double z_bulk = tn_expectation_z(mps, n_qubits / 2);
            uint32_t center = n_qubits / 2;
            double zz_bulk = tn_expectation_zz(mps, center, center + 1);

            uint32_t max_bond_dim = 1;
            for (uint32_t i = 0; i < n_qubits - 1; i++) {
                if (mps->bond_dims[i] > max_bond_dim) max_bond_dim = mps->bond_dims[i];
            }

            printf("  %4d   %.4f   %.4f     %u\n", step, z_bulk, zz_bulk, max_bond_dim);
            fflush(stdout);
        }
    }

    tn_mps_free(mps);
}

int main(void) {
    int full = env_enabled("MOONLAB_LONG_EVOLUTION_FULL");
    int n_steps = env_int("MOONLAB_LONG_EVOLUTION_STEPS", full ? 50 : 12);
    uint32_t max_qubits = (uint32_t)env_int("MOONLAB_LONG_EVOLUTION_MAX_QUBITS",
                                            full ? 64 : 32);

    printf("Testing longer evolution for system-size dependence\n");
    printf("Running %d Trotter steps (τ=0.1) for different system sizes\n", n_steps);
    printf("Finite-size effects should appear when correlation length > N/2\n");
    if (!full) {
        printf("Set MOONLAB_LONG_EVOLUTION_FULL=1 for the 64-qubit / 50-step sweep\n");
    }
    fflush(stdout);

    uint32_t sizes[] = {8, 16, 32, 64};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    for (size_t i = 0; i < num_sizes; i++) {
        if (sizes[i] <= max_qubits) {
            test_size(sizes[i], n_steps);
        }
    }

    printf("\nDone.\n");
    return 0;
}
