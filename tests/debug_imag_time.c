/**
 * @file debug_imag_time.c
 * @brief Minimal debug test for imaginary time evolution
 *
 * Small system (4 qubits) to verify correctness against known results.
 * Focus on accuracy, not performance.
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
#include "src/algorithms/tensor_network/svd_compress.h"

// ============================================================================
// SIMPLE EXACT CALCULATION FOR VERIFICATION
// ============================================================================

/**
 * For a 4-qubit system, compute exact ⟨ZZ⟩ correlation from state vector
 */
double exact_zz_correlation(double complex *state, int n_qubits, int q1, int q2) {
    int dim = 1 << n_qubits;
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        // Z eigenvalue is +1 for |0⟩, -1 for |1⟩
        int bit1 = (i >> (n_qubits - 1 - q1)) & 1;
        int bit2 = (i >> (n_qubits - 1 - q2)) & 1;
        double z1 = (bit1 == 0) ? 1.0 : -1.0;
        double z2 = (bit2 == 0) ? 1.0 : -1.0;

        double prob = cabs(state[i]) * cabs(state[i]);
        result += z1 * z2 * prob;
    }
    return result;
}

/**
 * Compute exact ⟨Z⟩ magnetization from state vector
 */
double exact_z_magnetization(double complex *state, int n_qubits) {
    int dim = 1 << n_qubits;
    double result = 0.0;

    for (int i = 0; i < dim; i++) {
        double prob = cabs(state[i]) * cabs(state[i]);

        // Sum Z eigenvalues over all qubits
        double z_sum = 0.0;
        for (int q = 0; q < n_qubits; q++) {
            int bit = (i >> (n_qubits - 1 - q)) & 1;
            z_sum += (bit == 0) ? 1.0 : -1.0;
        }
        result += (z_sum / n_qubits) * prob;
    }
    return result;
}

// ============================================================================
// MPS TO STATE VECTOR CONVERSION (for small systems)
// ============================================================================

/**
 * Convert MPS to full state vector for verification
 */
double complex *mps_to_statevector(tn_mps_state_t *mps) {
    uint32_t n = mps->num_qubits;
    uint64_t dim = 1ULL << n;

    double complex *state = calloc(dim, sizeof(double complex));
    if (!state) return NULL;

    // For each basis state |i⟩, compute coefficient by contracting MPS
    for (uint64_t basis = 0; basis < dim; basis++) {
        // Start with 1x1 "matrix" = 1
        double complex *current = malloc(sizeof(double complex));
        current[0] = 1.0;
        uint32_t current_dim = 1;

        for (uint32_t site = 0; site < n; site++) {
            // Get physical index for this site
            int phys = (basis >> (n - 1 - site)) & 1;

            tensor_t *t = mps->tensors[site];
            uint32_t left_dim = t->dims[0];
            uint32_t right_dim = t->dims[2];

            // New vector has dimension right_dim
            double complex *next = calloc(right_dim, sizeof(double complex));

            // Contract: next[r] = sum_l current[l] * T[l, phys, r]
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

        // Final result should be 1x1
        state[basis] = current[0];
        free(current);
    }

    return state;
}

// ============================================================================
// DEBUG GATE CREATION
// ============================================================================

/**
 * Create imaginary time ZZ gate: exp(tau * J * ZZ)
 * ZZ has eigenvalues +1 for |00⟩,|11⟩ and -1 for |01⟩,|10⟩
 */
tn_gate_2q_t create_imag_time_zz(double tau, double J) {
    tn_gate_2q_t gate;
    memset(&gate, 0, sizeof(gate));

    // exp(tau * J * ZZ) is diagonal:
    // |00⟩ -> exp(+tau*J) |00⟩
    // |01⟩ -> exp(-tau*J) |01⟩
    // |10⟩ -> exp(-tau*J) |10⟩
    // |11⟩ -> exp(+tau*J) |11⟩

    double exp_plus = exp(tau * J);
    double exp_minus = exp(-tau * J);

    gate.elements[0][0] = exp_plus;   // |00⟩ -> |00⟩
    gate.elements[1][1] = exp_minus;  // |01⟩ -> |01⟩
    gate.elements[2][2] = exp_minus;  // |10⟩ -> |10⟩
    gate.elements[3][3] = exp_plus;   // |11⟩ -> |11⟩

    printf("  ZZ gate (tau=%.3f, J=%.2f):\n", tau, J);
    printf("    exp(+tau*J) = %.6f\n", exp_plus);
    printf("    exp(-tau*J) = %.6f\n", exp_minus);

    return gate;
}

/**
 * Create imaginary time X gate: exp(tau * h * X)
 * X = [[0,1],[1,0]], eigenvalues ±1, eigenvectors |+⟩,|-⟩
 */
tn_gate_1q_t create_imag_time_x(double tau, double h) {
    tn_gate_1q_t gate;

    // exp(tau*h*X) = cosh(tau*h)*I + sinh(tau*h)*X
    double c = cosh(tau * h);
    double s = sinh(tau * h);

    gate.elements[0][0] = c;  // |0⟩ -> c|0⟩ + s|1⟩
    gate.elements[0][1] = s;
    gate.elements[1][0] = s;  // |1⟩ -> s|0⟩ + c|1⟩
    gate.elements[1][1] = c;

    printf("  X gate (tau=%.3f, h=%.2f):\n", tau, h);
    printf("    cosh(tau*h) = %.6f, sinh(tau*h) = %.6f\n", c, s);

    return gate;
}

// ============================================================================
// MAIN DEBUG TEST
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     DEBUG: Imaginary Time Evolution - Accuracy Test               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    // Test different system sizes to find where it breaks
    uint32_t test_sizes[] = {4, 8, 16, 32, 50, 100};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
    const uint32_t N_QUBITS = test_sizes[test_idx];
    const double J = 1.0;      // ZZ coupling
    const double h = 0.5;      // Transverse field (same as 100-qubit test)
    const double tau = 0.1;    // Imaginary time step
    const int N_STEPS = 5;     // Just 5 steps to match where 100-qubit fails
    const uint32_t MAX_BOND = 128;  // Same as 100-qubit test

    printf("Parameters:\n");
    printf("  N_QUBITS = %d\n", N_QUBITS);
    printf("  J = %.2f (ZZ coupling)\n", J);
    printf("  h = %.2f (transverse field)\n", h);
    printf("  J/h = %.2f (%s regime)\n", J/h, J/h > 1 ? "ferromagnetic" : "paramagnetic");
    printf("  tau = %.3f (imaginary time step)\n", tau);
    printf("  N_STEPS = %d\n", N_STEPS);
    printf("  MAX_BOND = %d\n\n", MAX_BOND);

    // Create MPS in |0000⟩ state with small tilt
    printf("Creating initial state...\n");

    tn_state_config_t config = {
        .max_bond_dim = MAX_BOND,
        .svd_cutoff = 1e-14,
        .max_truncation_error = 1e-12,
        .track_truncation = true,
        .auto_canonicalize = false,
        .target_form = TN_CANONICAL_NONE
    };

    tn_mps_state_t *mps = tn_mps_create_zero(N_QUBITS, &config);
    if (!mps) {
        printf("ERROR: Failed to create MPS\n");
        return 1;
    }

    // Initialize to |0000⟩ with small tilt toward |+⟩
    double tilt = 0.05;
    for (uint32_t i = 0; i < N_QUBITS; i++) {
        // |psi_i⟩ = cos(tilt)|0⟩ + sin(tilt)|1⟩
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    printf("  Initial MPS created\n\n");

    // Verify initial state
    printf("Converting MPS to state vector for verification...\n");
    double complex *state = mps_to_statevector(mps);
    if (!state) {
        printf("ERROR: Failed to convert MPS to state vector\n");
        tn_mps_free(mps);
        return 1;
    }

    // Normalize state vector
    double norm_sq = 0.0;
    uint64_t dim = 1ULL << N_QUBITS;
    for (uint64_t i = 0; i < dim; i++) {
        norm_sq += cabs(state[i]) * cabs(state[i]);
    }
    double norm = sqrt(norm_sq);
    for (uint64_t i = 0; i < dim; i++) {
        state[i] /= norm;
    }

    printf("\nInitial state coefficients (normalized):\n");
    for (uint64_t i = 0; i < dim && i < 8; i++) {
        if (cabs(state[i]) > 1e-10) {
            printf("  |");
            for (int q = N_QUBITS - 1; q >= 0; q--) {
                printf("%d", (int)((i >> q) & 1));
            }
            printf("⟩: %.6f + %.6fi\n", creal(state[i]), cimag(state[i]));
        }
    }

    double exact_z = exact_z_magnetization(state, N_QUBITS);
    double exact_zz = exact_zz_correlation(state, N_QUBITS, 0, 1);

    double mps_z = tn_expectation_z(mps, N_QUBITS / 2);
    double mps_zz = tn_expectation_zz(mps, 0, 1);

    printf("\nInitial observables:\n");
    printf("  Exact ⟨Z⟩   = %.6f\n", exact_z);
    printf("  MPS   ⟨Z⟩   = %.6f\n", mps_z);
    printf("  Exact ⟨ZZ⟩  = %.6f\n", exact_zz);
    printf("  MPS   ⟨ZZ⟩  = %.6f\n", mps_zz);

    if (fabs(mps_zz - exact_zz) > 0.01) {
        printf("\n  *** WARNING: MPS ⟨ZZ⟩ differs from exact! ***\n");
        printf("  This indicates a bug in tn_expectation_zz()\n");
    }

    free(state);

    // Create gates
    printf("\n\nCreating imaginary time gates...\n");
    tn_gate_2q_t zz_gate = create_imag_time_zz(tau, J);
    tn_gate_1q_t x_gate = create_imag_time_x(tau, h);

    // SVD config for truncation
    svd_compress_config_t svd_config = svd_compress_config_default();
    svd_config.max_bond_dim = MAX_BOND;
    svd_config.cutoff = 1e-14;

    printf("\n\nRunning imaginary time evolution...\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Step    MPS ⟨Z⟩   MPS ⟨ZZ⟩   Exact ⟨Z⟩  Exact ⟨ZZ⟩  MaxBond  Norm\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (int step = 0; step <= N_STEPS; step++) {
        // Convert to state vector for exact comparison
        state = mps_to_statevector(mps);

        // Normalize
        norm_sq = 0.0;
        for (uint64_t i = 0; i < dim; i++) {
            norm_sq += cabs(state[i]) * cabs(state[i]);
        }
        norm = sqrt(norm_sq);
        for (uint64_t i = 0; i < dim; i++) {
            state[i] /= norm;
        }

        // Compute observables
        exact_z = exact_z_magnetization(state, N_QUBITS);
        exact_zz = exact_zz_correlation(state, N_QUBITS, 0, 1);
        mps_z = tn_expectation_z(mps, N_QUBITS / 2);
        mps_zz = tn_expectation_zz(mps, 0, 1);

        // Get max bond
        uint32_t max_bond = 1;
        for (uint32_t i = 0; i < N_QUBITS - 1; i++) {
            if (mps->bond_dims[i] > max_bond) {
                max_bond = mps->bond_dims[i];
            }
        }

        // Print every 10 steps
        if (step % 10 == 0 || step == N_STEPS) {
            printf("%4d    %+.4f    %+.4f     %+.4f     %+.4f      %3d    %.4f\n",
                   step, mps_z, mps_zz, exact_z, exact_zz, max_bond, norm);
        }

        free(state);

        if (step == N_STEPS) break;

        // Apply one Trotter step: exp(-tau*H) ≈ exp(-tau*H_ZZ) * exp(-tau*H_X)

        // 1. Apply ZZ gates on even bonds (0-1, 2-3, ...)
        for (uint32_t i = 0; i < N_QUBITS - 1; i += 2) {
            double trunc_err = 0.0;
            tn_gate_error_t err = tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            if (err != TN_GATE_SUCCESS) {
                printf("ERROR: Gate application failed at bond %d: %d\n", i, err);
            }
        }

        // 2. Apply ZZ gates on odd bonds (1-2, 3-4, ...)
        for (uint32_t i = 1; i < N_QUBITS - 1; i += 2) {
            double trunc_err = 0.0;
            tn_gate_error_t err = tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
            if (err != TN_GATE_SUCCESS) {
                printf("ERROR: Gate application failed at bond %d: %d\n", i, err);
            }
        }

        // 3. Apply X gates on all sites
        for (uint32_t i = 0; i < N_QUBITS; i++) {
            tn_gate_error_t err = tn_apply_gate_1q(mps, i, &x_gate);
            if (err != TN_GATE_SUCCESS) {
                printf("ERROR: 1Q gate failed at site %d: %d\n", i, err);
            }
        }

        // 4. Normalize MPS (critical for imaginary time evolution)
        tn_mps_normalize(mps);
    }

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Final analysis
    printf("\nFinal state analysis:\n");

    state = mps_to_statevector(mps);
    norm_sq = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        norm_sq += cabs(state[i]) * cabs(state[i]);
    }
    norm = sqrt(norm_sq);
    for (uint64_t i = 0; i < dim; i++) {
        state[i] /= norm;
    }

    printf("\nFinal state coefficients (|coeff| > 0.01):\n");
    for (uint64_t i = 0; i < dim; i++) {
        if (cabs(state[i]) > 0.01) {
            printf("  |");
            for (int q = N_QUBITS - 1; q >= 0; q--) {
                printf("%d", (int)((i >> q) & 1));
            }
            printf("⟩: %.6f + %.6fi  (prob = %.4f)\n",
                   creal(state[i]), cimag(state[i]),
                   cabs(state[i]) * cabs(state[i]));
        }
    }

    printf("\nExpected for ferromagnetic ground state (J/h = 2):\n");
    printf("  |GHZ⟩ = (|0000⟩ + |1111⟩)/√2\n");
    printf("  ⟨Z⟩ = 0 (superposition of all up/all down)\n");
    printf("  ⟨ZZ⟩ = 1 (perfect nearest-neighbor correlation)\n");

    exact_zz = exact_zz_correlation(state, N_QUBITS, 0, 1);
    mps_zz = tn_expectation_zz(mps, 0, 1);

    printf("\nFinal comparison:\n");
    printf("  Exact ⟨ZZ⟩ = %.6f\n", exact_zz);
    printf("  MPS   ⟨ZZ⟩ = %.6f\n", mps_zz);

    if (fabs(exact_zz - mps_zz) > 0.01) {
        printf("\n*** BUG DETECTED: MPS measurement differs from exact ***\n");
        printf("The issue is in tn_expectation_zz(), not the evolution.\n");
    } else if (fabs(exact_zz - 1.0) > 0.1) {
        printf("\n*** Evolution may not have converged ***\n");
        printf("Try more steps or larger tau.\n");
    } else {
        printf("\n✓ Calculation appears correct!\n");
    }

    free(state);
    tn_mps_free(mps);

    printf("\n");
    return 0;
}
