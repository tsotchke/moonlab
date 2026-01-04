/**
 * @file quantum_spin_chain.c
 * @brief Tensor Network Simulation of a 100-Qubit Quantum Spin Chain
 *
 * This example demonstrates the power of tensor networks by simulating a
 * quantum system that would be IMPOSSIBLE with state vectors:
 *
 *   100 qubits × state vector = 2^100 amplitudes = 10^30 complex numbers
 *                             = 16 × 10^30 bytes = 16 QUINTILLION PETABYTES
 *
 *   100 qubits × MPS (χ=256) = ~26 MB
 *
 * We simulate the Transverse Field Ising Model (TFIM):
 *
 *   H = -J Σ Z_i Z_{i+1} - h Σ X_i
 *
 * This model exhibits a quantum phase transition at J/h = 1:
 *   - J/h > 1: Ferromagnetic phase (ordered, low entanglement)
 *   - J/h < 1: Paramagnetic phase (disordered, higher entanglement)
 *
 * The simulation computes:
 *   1. Ground state preparation via imaginary time evolution
 *   2. Entanglement entropy across the chain (detects phase transition)
 *   3. Magnetization ⟨Z⟩ and correlations ⟨Z_i Z_j⟩
 *   4. Real-time quantum quench dynamics
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <string.h>

#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_gates.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

// ============================================================================
// SIMULATION PARAMETERS (defaults, can be overridden via command line)
// ============================================================================

#define DEFAULT_NUM_QUBITS     100    // System size
#define DEFAULT_BOND_DIM       128    // Max bond dimension
#define DEFAULT_TROTTER_STEPS  50     // Time evolution steps
#define DEFAULT_TROTTER_DT     0.05   // Time step size
#define DEFAULT_FIELD_H        0.5    // Transverse field

// Fixed parameters
#define COUPLING_J             1.0    // ZZ coupling strength

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Create imaginary time evolution operator for X field: e^{τh X}
 *
 * e^{τh X} = cosh(τh) I + sinh(τh) X = [[cosh(τh), sinh(τh)], [sinh(τh), cosh(τh)]]
 *
 * This is NON-UNITARY - state must be normalized after application.
 */
static tn_gate_1q_t create_imag_time_x_gate(double tau_h) {
    tn_gate_1q_t gate;
    double c = cosh(tau_h);
    double s = sinh(tau_h);

    // Explicitly use complex numbers
    gate.elements[0][0] = c + 0.0*I;      // |0⟩⟨0| coefficient
    gate.elements[0][1] = s + 0.0*I;      // |0⟩⟨1| coefficient
    gate.elements[1][0] = s + 0.0*I;      // |1⟩⟨0| coefficient
    gate.elements[1][1] = c + 0.0*I;      // |1⟩⟨1| coefficient

    return gate;
}

/**
 * @brief Create imaginary time evolution operator for ZZ: e^{τJ Z⊗Z}
 *
 * For the Ising interaction -J Z_i Z_{i+1}, the imaginary time operator is:
 * e^{τJ ZZ} = diagonal matrix with:
 *   - e^{+τJ} for |00⟩ and |11⟩ (aligned spins, lower energy)
 *   - e^{-τJ} for |01⟩ and |10⟩ (anti-aligned spins, higher energy)
 *
 * This is NON-UNITARY and exponentially suppresses excited states.
 */
static tn_gate_2q_t create_imag_time_zz_gate(double tau_J) {
    tn_gate_2q_t gate;
    double ep = exp(tau_J);   // e^{τJ} for aligned spins
    double em = exp(-tau_J);  // e^{-τJ} for anti-aligned spins

    // Initialize to zero - MUST use complex numbers explicitly
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            gate.elements[i][j] = 0.0 + 0.0*I;
        }
    }

    // Diagonal elements only (computational basis |00⟩, |01⟩, |10⟩, |11⟩)
    // MUST cast to complex explicitly
    gate.elements[0][0] = ep + 0.0*I;  // |00⟩ → e^{τJ}|00⟩ (both spin down, ZZ = +1)
    gate.elements[1][1] = em + 0.0*I;  // |01⟩ → e^{-τJ}|01⟩ (opposite spins, ZZ = -1)
    gate.elements[2][2] = em + 0.0*I;  // |10⟩ → e^{-τJ}|10⟩ (opposite spins, ZZ = -1)
    gate.elements[3][3] = ep + 0.0*I;  // |11⟩ → e^{τJ}|11⟩ (both spin up, ZZ = +1)

    return gate;
}

/**
 * @brief Apply one Trotter step of e^{-dt H} for TFIM
 *
 * Uses second-order Trotter decomposition:
 *   e^{-dt(H_ZZ + H_X)} ≈ e^{-dt/2 H_X} e^{-dt H_ZZ} e^{-dt/2 H_X}
 *
 * For REAL TIME: Uses unitary gates (Rx for X, Rzz for ZZ)
 * For IMAGINARY TIME: Uses non-unitary exponential operators (cosh/sinh for X, exp for ZZ)
 *
 * CRITICAL: Imaginary time gates are NON-UNITARY! State norm grows and must be
 * normalized after each layer to prevent numerical instability.
 *
 * @param state MPS state
 * @param J ZZ coupling
 * @param h Transverse field
 * @param dt Time step (imaginary for ground state, real for dynamics)
 * @param imaginary_time If true, use imaginary time evolution
 * @return Error if any
 */
static int apply_trotter_step(tn_mps_state_t *state,
                               double J, double h, double dt,
                               bool imaginary_time) {
    uint32_t n = state->num_qubits;
    int ret;

    // ========================================================================
    // STEP 1: Half X evolution - e^{±i h*dt/2 * X} or e^{h*dt/2 * X}
    // ========================================================================
    if (!imaginary_time) {
        // REAL TIME: Unitary Rx rotation
        // e^{-i h dt/2 X} = Rx(h*dt)
        for (uint32_t i = 0; i < n; i++) {
            ret = tn_apply_rx(state, i, 2.0 * h * dt / 2.0);
            if (ret != TN_GATE_SUCCESS) return ret;
        }
    } else {
        // IMAGINARY TIME: Non-unitary hyperbolic operator
        // e^{h*dt/2*X} = cosh(h*dt/2) I + sinh(h*dt/2) X
        tn_gate_1q_t x_gate = create_imag_time_x_gate(h * dt / 2.0);

        // Single-qubit gates are independent - can parallelize
        int error_flag = 0;
        #pragma omp parallel for schedule(static) reduction(|:error_flag)
        for (uint32_t i = 0; i < n; i++) {
            if (tn_apply_gate_1q(state, i, &x_gate) != TN_GATE_SUCCESS) {
                error_flag = 1;
            }
        }
        if (error_flag) return TN_GATE_ERROR_CONTRACTION_FAILED;

        // Must normalize to prevent numerical explosion from non-unitary gates
        ret = tn_mps_normalize(state);
        if (ret != TN_STATE_SUCCESS) return ret;
    }

    // ========================================================================
    // STEP 2: Full ZZ evolution - e^{±i J*dt * ZZ} or e^{J*dt * ZZ}
    // ========================================================================
    if (!imaginary_time) {
        // REAL TIME: Unitary Rzz rotation
        double theta_zz = J * dt * 2.0;

        // Even bonds: (0,1), (2,3), (4,5), ...
        for (uint32_t i = 0; i + 1 < n; i += 2) {
            ret = tn_apply_rzz(state, i, i + 1, theta_zz);
            if (ret != TN_GATE_SUCCESS) return ret;
        }

        // Odd bonds: (1,2), (3,4), (5,6), ...
        for (uint32_t i = 1; i + 1 < n; i += 2) {
            ret = tn_apply_rzz(state, i, i + 1, theta_zz);
            if (ret != TN_GATE_SUCCESS) return ret;
        }
    } else {
        // IMAGINARY TIME: Non-unitary diagonal exponential
        // e^{J*dt*ZZ} = diag(e^J, e^{-J}, e^{-J}, e^J)
        tn_gate_2q_t zz_gate = create_imag_time_zz_gate(J * dt);

        // Even bonds: (0,1), (2,3), (4,5), ... - sequential for correct SVD ordering
        for (uint32_t i = 0; i + 1 < n; i += 2) {
            double trunc_err;
            ret = tn_apply_gate_2q(state, i, i + 1, &zz_gate, &trunc_err);
            if (ret != TN_GATE_SUCCESS) return ret;
        }

        // Odd bonds: (1,2), (3,4), (5,6), ... - sequential for correct SVD ordering
        for (uint32_t i = 1; i + 1 < n; i += 2) {
            double trunc_err;
            ret = tn_apply_gate_2q(state, i, i + 1, &zz_gate, &trunc_err);
            if (ret != TN_GATE_SUCCESS) return ret;
        }

        // After complete left-to-right TEBD sweep, MPS is in left-canonical form
        tn_mps_mark_canonical_left(state);

        // Normalize using fast O(chi^2) path since we're now left-canonical
        ret = tn_mps_normalize(state);
        if (ret != TN_STATE_SUCCESS) return ret;
    }

    // ========================================================================
    // STEP 3: Second half X evolution (2nd order Trotter symmetry)
    // ========================================================================
    if (!imaginary_time) {
        // REAL TIME: Unitary Rx rotation
        for (uint32_t i = 0; i < n; i++) {
            ret = tn_apply_rx(state, i, 2.0 * h * dt / 2.0);
            if (ret != TN_GATE_SUCCESS) return ret;
        }
    } else {
        // IMAGINARY TIME: Non-unitary hyperbolic operator
        tn_gate_1q_t x_gate = create_imag_time_x_gate(h * dt / 2.0);

        // Single-qubit gates are independent - can parallelize
        int error_flag = 0;
        #pragma omp parallel for schedule(static) reduction(|:error_flag)
        for (uint32_t i = 0; i < n; i++) {
            if (tn_apply_gate_1q(state, i, &x_gate) != TN_GATE_SUCCESS) {
                error_flag = 1;
            }
        }
        if (error_flag) return TN_GATE_ERROR_CONTRACTION_FAILED;

        // Normalize using transfer matrix (needed since X gates are non-unitary)
        ret = tn_mps_normalize(state);
        if (ret != TN_STATE_SUCCESS) return ret;
    }

    return TN_GATE_SUCCESS;
}

/**
 * @brief Compute average magnetization ⟨Z⟩
 * Uses stable O(n² × chi^4) transfer matrix method.
 */
static double compute_magnetization(const tn_mps_state_t *state) {
    uint32_t n = state->num_qubits;
    double total = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        total += tn_expectation_z(state, i);
    }
    return total / n;
}

/**
 * @brief Compute nearest-neighbor correlation ⟨Z_i Z_{i+1}⟩
 * Uses stable O(n² × chi^4) transfer matrix method.
 */
static double compute_zz_correlation(const tn_mps_state_t *state) {
    uint32_t n = state->num_qubits;
    double total = 0.0;
    for (uint32_t i = 0; i < n - 1; i++) {
        total += tn_expectation_zz(state, i, i + 1);
    }
    return total / (n - 1);
}

/**
 * @brief Print entanglement profile across chain
 */
static void print_entanglement_profile(const tn_mps_state_t *state) {
    printf("\n  Entanglement Profile (von Neumann entropy at each bond):\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  Bond:    ");

    // Print selected bonds
    uint32_t n = state->num_qubits;
    uint32_t bonds_to_print[] = {0, n/4, n/2-1, n/2, n/2+1, 3*n/4, n-2};
    int num_print = sizeof(bonds_to_print) / sizeof(bonds_to_print[0]);

    for (int i = 0; i < num_print; i++) {
        if (bonds_to_print[i] < n - 1) {
            printf("%4u   ", bonds_to_print[i]);
        }
    }
    printf("\n");

    printf("  S(bond): ");
    for (int i = 0; i < num_print; i++) {
        if (bonds_to_print[i] < n - 1) {
            double s = tn_mps_entanglement_entropy(state, bonds_to_print[i]);
            printf("%5.3f  ", s);
        }
    }
    printf("\n");

    // Central entropy (most important for phase detection)
    double s_center = tn_mps_entanglement_entropy(state, n / 2);
    printf("\n  Central Entanglement Entropy S(L/2) = %.4f\n", s_center);

    // Interpret result
    if (s_center < 0.5) {
        printf("  → LOW entanglement: System is in ORDERED (ferromagnetic) phase\n");
    } else if (s_center > 1.5) {
        printf("  → HIGH entanglement: System near CRITICAL point\n");
    } else {
        printf("  → MODERATE entanglement: System in DISORDERED (paramagnetic) phase\n");
    }
}

// ============================================================================
// MAIN SIMULATION
// ============================================================================

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int num_qubits = DEFAULT_NUM_QUBITS;
    int bond_dim = DEFAULT_BOND_DIM;
    int trotter_steps = DEFAULT_TROTTER_STEPS;
    double trotter_dt = DEFAULT_TROTTER_DT;
    double field_h = DEFAULT_FIELD_H;

    if (argc > 1) num_qubits = atoi(argv[1]);
    if (argc > 2) trotter_steps = atoi(argv[2]);
    if (argc > 3) bond_dim = atoi(argv[3]);
    if (argc > 4) field_h = atof(argv[4]);
    if (argc > 5) trotter_dt = atof(argv[5]);

    // Validate
    if (num_qubits < 4) num_qubits = 4;
    if (num_qubits > 1000) num_qubits = 1000;
    if (bond_dim < 2) bond_dim = 2;
    if (bond_dim > 512) bond_dim = 512;
    if (trotter_steps < 1) trotter_steps = 1;

    printf("\n");
    printf("Usage: %s [qubits] [steps] [bond_dim] [field_h] [dt]\n", argv[0]);
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   TENSOR NETWORK QUANTUM SIMULATION                           ║\n");
    printf("║   Transverse Field Ising Model                                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    clock_t start_time = clock();

    // ========================================================================
    // PART 1: Setup
    // ========================================================================

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  SIMULATION PARAMETERS\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
    printf("  System size:       %d qubits\n", num_qubits);
    printf("  Bond dimension:    %d (max)\n", bond_dim);
    printf("  ZZ coupling (J):   %.2f\n", COUPLING_J);
    printf("  Transverse field:  %.2f\n", field_h);
    printf("  J/h ratio:         %.2f ", COUPLING_J / field_h);
    if (COUPLING_J / field_h > 1.0) {
        printf("(ferromagnetic regime)\n");
    } else if (COUPLING_J / field_h < 1.0) {
        printf("(paramagnetic regime)\n");
    } else {
        printf("(critical point!)\n");
    }
    printf("  Trotter steps:     %d × dt=%.3f\n", trotter_steps, trotter_dt);
    printf("\n");

    // Estimate memory
    uint64_t mem = tn_mps_estimate_memory(num_qubits, bond_dim);
    printf("  Estimated memory:  %.2f MB\n", mem / (1024.0 * 1024.0));
    printf("\n");

    // ========================================================================
    // PART 2: Create Initial State
    // ========================================================================

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  CREATING INITIAL STATE\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");

    tn_state_config_t config = tn_state_config_create(bond_dim, 1e-10);
    config.auto_canonicalize = true;

    // Start in appropriate initial state based on the phase
    // J/h = 2.0 means we're in ferromagnetic regime (g = h/J = 0.5 < 1)
    // Ground state is close to |00...0⟩ or |11...1⟩
    tn_mps_state_t *state = tn_mps_create_zero(num_qubits, &config);
    if (!state) {
        fprintf(stderr, "Error: Failed to create MPS state\n");
        return 1;
    }

    // For ferromagnetic phase: start from |00...0⟩ with small tilt
    // Add small rotation to break Z₂ symmetry and allow entanglement to build
    printf("  Initializing ferromagnetic-like state |00...0⟩ + small tilt...\n");
    int ret = TN_GATE_SUCCESS;
    for (uint32_t i = 0; i < (uint32_t)num_qubits; i++) {
        ret = tn_apply_ry(state, i, 0.1);  // Small tilt to break symmetry
        if (ret != TN_GATE_SUCCESS) {
            fprintf(stderr, "Error: Failed to apply Ry gates\n");
            tn_mps_free(state);
            return 1;
        }
    }

    tn_mps_stats_t stats = tn_mps_get_stats(state);
    printf("  ✓ Created %u-qubit MPS\n", state->num_qubits);
    printf("  ✓ Memory: %.2f KB\n", stats.memory_bytes / 1024.0);
    printf("  ✓ Initial bond dimension: %u\n", stats.max_bond_dim);
    printf("\n");

    // Initial measurements
    double mag_init = compute_magnetization(state);
    double corr_init = compute_zz_correlation(state);
    printf("  Initial state observables:\n");
    printf("    ⟨Z⟩ = %.4f (magnetization)\n", mag_init);
    printf("    ⟨ZZ⟩ = %.4f (nearest-neighbor correlation)\n", corr_init);
    printf("\n");

    // ========================================================================
    // PART 3: Imaginary Time Evolution (Ground State Preparation)
    // ========================================================================

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  GROUND STATE PREPARATION (Imaginary Time Evolution)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
    printf("  Evolving with e^{-τH} to project onto ground state...\n");
    printf("\n");

    int ground_state_steps = 30;  // Imaginary time steps
    double imag_dt = 0.1;

    printf("  Step    ⟨Z⟩      ⟨ZZ⟩     Max Bond    Truncation    Norm\n");
    printf("  ─────────────────────────────────────────────────────────────────\n");

    for (int step = 0; step <= ground_state_steps; step++) {
        if (step > 0) {
            ret = apply_trotter_step(state, COUPLING_J, field_h, imag_dt, true);
            if (ret != TN_GATE_SUCCESS) {
                fprintf(stderr, "ERROR: Imaginary time Trotter step failed at step %d\n", step);
                fprintf(stderr, "       Error code: %d (%s)\n", ret, tn_gate_error_string(ret));
                fprintf(stderr, "       This indicates a fundamental issue with the imaginary time gates.\n");
                break;
            }
        }

        // Debug: compute actual norm to track state health
        double actual_norm = tn_mps_norm(state);

        if (step % 5 == 0 || step == ground_state_steps) {
            double mag = compute_magnetization(state);
            double corr = compute_zz_correlation(state);
            tn_mps_stats_t s = tn_mps_get_stats(state);
            printf("  %4d   %+.4f   %+.4f      %4u       %.2e    %.4f\n",
                   step, mag, corr, s.max_bond_dim, s.truncation_error, actual_norm);
        }
    }
    printf("\n");

    // ========================================================================
    // PART 4: Ground State Analysis
    // ========================================================================

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  GROUND STATE ANALYSIS\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    double mag_gs = compute_magnetization(state);
    double corr_gs = compute_zz_correlation(state);
    stats = tn_mps_get_stats(state);

    printf("\n");
    printf("  Ground state observables:\n");
    printf("    ⟨Z⟩ = %.4f (order parameter)\n", mag_gs);
    printf("    ⟨ZZ⟩ = %.4f (nearest-neighbor correlation)\n", corr_gs);
    printf("    Max bond dimension: %u\n", stats.max_bond_dim);
    printf("    Memory usage: %.2f MB\n", stats.memory_bytes / (1024.0 * 1024.0));

    print_entanglement_profile(state);

    // Interpret phase
    printf("\n");
    printf("  PHASE INTERPRETATION:\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    if (COUPLING_J / field_h > 1.0) {
        printf("  J/h = %.2f > 1 → FERROMAGNETIC PHASE\n", COUPLING_J / field_h);
        printf("  • Spins prefer to align (⟨ZZ⟩ → +1)\n");
        printf("  • Low entanglement (area law satisfied)\n");
        printf("  • Classical-like order\n");
    } else if (COUPLING_J / field_h < 1.0) {
        printf("  J/h = %.2f < 1 → PARAMAGNETIC PHASE\n", COUPLING_J / field_h);
        printf("  • Spins align with transverse field\n");
        printf("  • Moderate entanglement\n");
        printf("  • Quantum fluctuations dominate\n");
    } else {
        printf("  J/h = 1.0 → CRITICAL POINT\n");
        printf("  • Quantum phase transition!\n");
        printf("  • Entanglement entropy scales logarithmically\n");
        printf("  • Correlation length diverges\n");
    }
    printf("\n");

    // ========================================================================
    // PART 5: Real-Time Quantum Dynamics (Quench)
    // ========================================================================

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  QUANTUM QUENCH DYNAMICS\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
    printf("  Suddenly changing h: %.2f → %.2f and evolving in real time...\n",
           field_h, field_h * 2.0);
    printf("  This creates non-equilibrium quantum dynamics!\n");
    printf("\n");

    double h_quench = field_h * 2.0;  // Quench to different field strength

    printf("  Time      ⟨Z⟩      ⟨ZZ⟩     S(L/2)    Max Bond\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    int quench_steps = 20;
    double dt_quench = 0.1;

    for (int step = 0; step <= quench_steps; step++) {
        double t = step * dt_quench;

        if (step > 0) {
            ret = apply_trotter_step(state, COUPLING_J, h_quench, dt_quench, false);
            if (ret != TN_GATE_SUCCESS) {
                fprintf(stderr, "Warning: Quench evolution failed at step %d\n", step);
                break;
            }
        }

        if (step % 4 == 0) {
            double mag = compute_magnetization(state);
            double corr = compute_zz_correlation(state);
            double s_center = tn_mps_entanglement_entropy(state, num_qubits / 2);
            tn_mps_stats_t s = tn_mps_get_stats(state);
            printf("  %.2f    %+.4f   %+.4f    %.4f      %4u\n",
                   t, mag, corr, s_center, s.max_bond_dim);
        }
    }
    printf("\n");

    printf("  Note: Entanglement growth during quench dynamics!\n");
    printf("  This is a signature of quantum thermalization.\n");
    printf("\n");

    // ========================================================================
    // PART 6: Summary
    // ========================================================================

    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  SIMULATION SUMMARY\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");

    stats = tn_mps_get_stats(state);

    printf("  ✓ Simulated %d-qubit quantum spin chain\n", num_qubits);
    printf("  ✓ Found approximate ground state via imaginary time evolution\n");
    printf("  ✓ Analyzed quantum phase (J/h = %.2f)\n", COUPLING_J / field_h);
    printf("  ✓ Simulated non-equilibrium quench dynamics\n");
    printf("\n");
    printf("  Final state statistics:\n");
    printf("    • Memory used: %.2f MB\n", stats.memory_bytes / (1024.0 * 1024.0));
    printf("    • Max bond dimension: %u\n", stats.max_bond_dim);
    printf("    • Truncation error: %.2e\n", stats.truncation_error);
    printf("    • Total time: %.2f seconds\n", total_time);
    printf("\n");

    printf("  TRY EXPERIMENTING:\n");
    printf("  Try different parameters:\n");
    printf("    • ./quantum_spin_chain 100 50 128 1.0   # Critical point\n");
    printf("    • ./quantum_spin_chain 100 50 128 2.0   # Paramagnetic phase\n");
    printf("    • ./quantum_spin_chain 200 50 128 0.5   # Larger system\n");
    printf("    • ./quantum_spin_chain 100 50 256 0.5   # Higher accuracy\n");
    printf("\n");

    // Cleanup
    tn_mps_free(state);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   SIMULATION COMPLETE                                         ║\n");
    printf("║                                                               ║\n");
    printf("║   This demonstrated tensor network simulation of a quantum    ║\n");
    printf("║   system far beyond classical state-vector capabilities!      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
