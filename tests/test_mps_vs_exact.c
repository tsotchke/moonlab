/**
 * @file test_mps_vs_exact.c
 * @brief Compare MPS measurements with exact state vector calculations
 *
 * Tests increasing system sizes to pinpoint where MPS calculations diverge from exact.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include "src/algorithms/tensor_network/tensor.h"
#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_gates.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

// ============================================================================
// EXACT CALCULATIONS
// ============================================================================

double exact_z(double complex *state, int n_qubits, int qubit) {
    uint64_t dim = 1ULL << n_qubits;
    double result = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        int bit = (i >> (n_qubits - 1 - qubit)) & 1;
        double z_val = (bit == 0) ? 1.0 : -1.0;
        double prob = cabs(state[i]) * cabs(state[i]);
        result += z_val * prob;
    }
    return result;
}

double exact_zz(double complex *state, int n_qubits, int q1, int q2) {
    uint64_t dim = 1ULL << n_qubits;
    double result = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        int bit1 = (i >> (n_qubits - 1 - q1)) & 1;
        int bit2 = (i >> (n_qubits - 1 - q2)) & 1;
        double z1 = (bit1 == 0) ? 1.0 : -1.0;
        double z2 = (bit2 == 0) ? 1.0 : -1.0;
        double prob = cabs(state[i]) * cabs(state[i]);
        result += z1 * z2 * prob;
    }
    return result;
}

// ============================================================================
// MPS TO STATE VECTOR
// ============================================================================

double complex *mps_to_statevector(tn_mps_state_t *mps) {
    uint32_t n = mps->num_qubits;
    uint64_t dim = 1ULL << n;

    double complex *state = calloc(dim, sizeof(double complex));
    if (!state) return NULL;

    for (uint64_t basis = 0; basis < dim; basis++) {
        double complex *current = malloc(sizeof(double complex));
        current[0] = 1.0;
        uint32_t current_dim = 1;

        for (uint32_t site = 0; site < n; site++) {
            int phys = (basis >> (n - 1 - site)) & 1;
            tensor_t *t = mps->tensors[site];
            uint32_t left_dim = t->dims[0];
            uint32_t right_dim = t->dims[2];

            double complex *next = calloc(right_dim, sizeof(double complex));
            for (uint32_t r = 0; r < right_dim; r++) {
                for (uint32_t l = 0; l < left_dim; l++) {
                    uint32_t idx[3] = {l, phys, r};
                    next[r] += current[l] * tensor_get(t, idx);
                }
            }
            free(current);
            current = next;
            current_dim = right_dim;
        }
        state[basis] = current[0];
        free(current);
    }

    // Normalize
    double norm_sq = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        norm_sq += cabs(state[i]) * cabs(state[i]);
    }
    double norm = sqrt(norm_sq);
    if (norm > 1e-30) {
        for (uint64_t i = 0; i < dim; i++) {
            state[i] /= norm;
        }
    }

    return state;
}

// ============================================================================
// GATES
// ============================================================================

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

// ============================================================================
// TEST ONE SIZE
// ============================================================================

void test_size(uint32_t n_qubits, int n_steps) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  Testing N = %u qubits\n", n_qubits);
    printf("══════════════════════════════════════════════════════════════════\n");

    const double J = 1.0;
    const double h = 0.5;
    const double tau = 0.1;
    const uint32_t max_bond = 128;

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

    // Initialize with small tilt
    double tilt = 0.05;
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    // Create gates
    tn_gate_2q_t zz_gate = create_imag_time_zz(tau * J);
    tn_gate_1q_t x_gate = create_imag_time_x(tau * h);

    // Can we do exact comparison? (limit to ~16 qubits due to memory)
    int can_exact = (n_qubits <= 16);

    printf("\n  Step  MPS<Z>   MPS<ZZ>  ");
    if (can_exact) printf("Exact<Z>  Exact<ZZ>  Z_err     ZZ_err");
    printf("\n  ────  ───────  ───────  ");
    if (can_exact) printf("────────  ─────────  ────────  ────────");
    printf("\n");

    uint32_t mid = n_qubits / 2;

    for (int step = 0; step <= n_steps; step++) {
        if (step > 0) {
            // Apply ZZ gates on even bonds
            for (uint32_t i = 0; i < n_qubits - 1; i += 2) {
                double trunc_err = 0.0;
                tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            }
            // Apply ZZ gates on odd bonds
            for (uint32_t i = 1; i < n_qubits - 1; i += 2) {
                double trunc_err = 0.0;
                tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            }
            // Apply X gates
            for (uint32_t i = 0; i < n_qubits; i++) {
                tn_apply_gate_1q(mps, i, &x_gate);
            }
        }

        // MPS measurements
        double mps_z = tn_expectation_z(mps, mid);
        double mps_zz = tn_expectation_zz(mps, 0, 1);

        printf("  %4d  %+.4f  %+.4f  ", step, mps_z, mps_zz);

        if (can_exact) {
            double complex *state = mps_to_statevector(mps);
            if (state) {
                double ex_z = exact_z(state, n_qubits, mid);
                double ex_zz = exact_zz(state, n_qubits, 0, 1);
                double z_err = fabs(mps_z - ex_z);
                double zz_err = fabs(mps_zz - ex_zz);

                printf("%+.4f   %+.4f    %.2e  %.2e", ex_z, ex_zz, z_err, zz_err);

                // Flag significant errors
                if (z_err > 0.01 || zz_err > 0.01) {
                    printf("  *** ERROR ***");
                }

                free(state);
            } else {
                printf("(exact failed)");
            }
        }
        printf("\n");
    }

    tn_mps_free(mps);
}

// ============================================================================
// MAIN
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     MPS vs EXACT: Pinpointing Calculation Errors                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("\nParameters: J=1.0, h=0.5, tau=0.1, max_bond=128\n");
    printf("Measuring <Z> at middle qubit, <ZZ> at boundary (qubits 0,1)\n");

    // Test sizes where we can compare with exact
    uint32_t sizes[] = {4, 6, 8, 10, 12, 14, 16};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        test_size(sizes[i], 5);
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY: Check Z_err and ZZ_err columns for discrepancies\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
