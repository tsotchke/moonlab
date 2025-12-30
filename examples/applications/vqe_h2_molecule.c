/**
 * @file vqe_h2_molecule.c
 * @brief VQE demonstration: Compute H₂ ground state energy
 * 
 * This example demonstrates molecular simulation using the
 * Variational Quantum Eigensolver (VQE) algorithm.
 * 
 * Application: Drug discovery, materials science, chemistry
 * 
 * Expected output:
 * - Ground state energy: -1.137 Ha (chemical accuracy)
 * - Convergence in <100 iterations
 * - Demonstrates quantum chemistry on 2-qubit system
 */

#include "../../src/algorithms/vqe.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/entropy_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Entropy callback for quantum operations
static entropy_pool_ctx_t *global_entropy_pool = NULL;

static int entropy_callback(void *user_data, uint8_t *buffer, size_t size) {
    (void)user_data;
    return entropy_pool_get_bytes(global_entropy_pool, buffer, size);
}

int main(int argc, char **argv) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                                                            ║\n");
    printf("║      VQE MOLECULAR SIMULATION: H₂ MOLECULE                 ║\n");
    printf("║                                                            ║\n");
    printf("║  Demonstrates quantum chemistry calculation using          ║\n");
    printf("║  Variational Quantum Eigensolver on 2-qubit system         ║\n");
    printf("║                                                            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    // Parse command line (optional bond distance)
    double bond_distance = 0.7414;  // Equilibrium distance (Angstroms)
    
    if (argc > 1) {
        bond_distance = atof(argv[1]);
        if (bond_distance < 0.4 || bond_distance > 2.5) {
            fprintf(stderr, "Error: Bond distance must be in range [0.4, 2.5] Angstroms\n");
            return 1;
        }
    }
    
    printf("Configuration:\n");
    printf("  Molecule:          H₂ (Hydrogen)\n");
    printf("  Bond distance:     %.4f Angstroms\n", bond_distance);
    printf("  Basis set:         STO-3G (minimal basis)\n");
    printf("  Active space:      2 electrons, 2 orbitals → 2 qubits\n");
    printf("  Reference method:  FCI (Full Configuration Interaction)\n");
    printf("\n");
    
    // Initialize entropy source
    if (entropy_pool_init(&global_entropy_pool) != 0) {
        fprintf(stderr, "Error: Failed to initialize entropy pool\n");
        return 1;
    }
    
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, entropy_callback, NULL);
    
    // Step 1: Create molecular Hamiltonian
    printf("Step 1: Constructing molecular Hamiltonian...\n");
    molecular_hamiltonian_t *hamiltonian = vqe_create_h2_hamiltonian(bond_distance);
    if (!hamiltonian) {
        fprintf(stderr, "Error: Failed to create H2 Hamiltonian\n");
        entropy_pool_free(global_entropy_pool);
        return 1;
    }
    
    vqe_print_hamiltonian(hamiltonian);
    
    // Step 2: Create variational ansatz
    printf("Step 2: Creating variational ansatz...\n");
    size_t num_layers = 2;  // Sufficient for H₂
    vqe_ansatz_t *ansatz = vqe_create_hardware_efficient_ansatz(
        hamiltonian->num_qubits,
        num_layers
    );
    if (!ansatz) {
        fprintf(stderr, "Error: Failed to create ansatz\n");
        molecular_hamiltonian_free(hamiltonian);
        entropy_pool_free(global_entropy_pool);
        return 1;
    }
    
    printf("  Ansatz type:       Hardware-Efficient\n");
    printf("  Layers:            %zu\n", num_layers);
    printf("  Parameters:        %zu\n", ansatz->num_parameters);
    printf("\n");
    
    // Step 3: Create optimizer
    printf("Step 3: Configuring optimizer...\n");
    vqe_optimizer_t *optimizer = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    if (!optimizer) {
        fprintf(stderr, "Error: Failed to create optimizer\n");
        vqe_ansatz_free(ansatz);
        molecular_hamiltonian_free(hamiltonian);
        entropy_pool_free(global_entropy_pool);
        return 1;
    }
    
    optimizer->max_iterations = 200;
    optimizer->learning_rate = 0.1;
    optimizer->tolerance = 1e-7;
    optimizer->verbose = 1;
    
    printf("  Method:            ADAM (Adaptive Moment Estimation)\n");
    printf("  Max iterations:    %zu\n", optimizer->max_iterations);
    printf("  Learning rate:     %.3f\n", optimizer->learning_rate);
    printf("  Tolerance:         %.2e\n", optimizer->tolerance);
    printf("\n");
    
    // Step 4: Create VQE solver
    printf("Step 4: Initializing VQE solver...\n");
    vqe_solver_t *solver = vqe_solver_create(
        hamiltonian,
        ansatz,
        optimizer,
        &entropy
    );
    if (!solver) {
        fprintf(stderr, "Error: Failed to create VQE solver\n");
        vqe_optimizer_free(optimizer);
        vqe_ansatz_free(ansatz);
        molecular_hamiltonian_free(hamiltonian);
        entropy_pool_free(global_entropy_pool);
        return 1;
    }
    
    printf("  Solver initialized successfully\n\n");
    
    // Step 5: Run VQE optimization
    printf("Step 5: Running VQE optimization...\n\n");
    
    clock_t start = clock();
    vqe_result_t result = vqe_solve(solver);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Step 6: Validate results
    printf("Step 6: Validating results...\n\n");
    
    // Reference energies for H₂ at equilibrium (r = 0.7414 A)
    double fci_reference = -1.137283834488;  // Exact FCI
    double hf_reference = -1.116685;          // Hartree-Fock
    
    result.fci_energy = fci_reference;
    result.hf_energy = hf_reference;
    
    // Recalculate chemical accuracy
    double error_ha = fabs(result.ground_state_energy - fci_reference);
    double error_kcal = vqe_hartree_to_kcalmol(error_ha);
    result.chemical_accuracy = error_kcal;
    
    // Print final results
    vqe_print_result(&result);
    
    printf("Performance Metrics:\n");
    printf("  Total time:        %.3f seconds\n", elapsed);
    printf("  Time per iteration: %.3f ms\n", 1000.0 * elapsed / result.iterations);
    printf("  Measurements:      %zu\n", solver->total_measurements);
    printf("\n");
    
    // Validation
    printf("Validation:\n");
    if (error_kcal < 1.0) {
        printf("  ✓ CHEMICAL ACCURACY ACHIEVED!\n");
        printf("  ✓ Error < 1 kcal/mol (publication quality)\n");
    } else if (error_kcal < 1.6) {
        printf("  ⚠ Near chemical accuracy (acceptable for research)\n");
    } else {
        printf("  ✗ Below chemical accuracy (increase layers or iterations)\n");
    }
    
    if (result.ground_state_energy < hf_reference) {
        printf("  ✓ Captures electron correlation\n");
        printf("  ✓ Beyond Hartree-Fock accuracy\n");
    }
    
    printf("\n");
    
    // Cleanup
    free(result.optimal_parameters);
    vqe_solver_free(solver);
    vqe_optimizer_free(optimizer);
    vqe_ansatz_free(ansatz);
    molecular_hamiltonian_free(hamiltonian);
    entropy_pool_free(global_entropy_pool);
    
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                                                            ║\n");
    printf("║  VQE DEMONSTRATION COMPLETE                                ║\n");
    printf("║                                                            ║\n");
    printf("║  This 2-qubit quantum chemistry calculation demonstrates   ║\n");
    printf("║  how VQE can compute molecular properties with chemical    ║\n");
    printf("║  accuracy - essential for drug discovery and materials     ║\n");
    printf("║  science applications.                                     ║\n");
    printf("║                                                            ║\n");
    printf("║  Try different bond distances:                             ║\n");
    printf("║    ./vqe_h2_molecule 0.5    (compressed)                   ║\n");
    printf("║    ./vqe_h2_molecule 0.74   (equilibrium)                  ║\n");
    printf("║    ./vqe_h2_molecule 1.5    (stretched)                    ║\n");
    printf("║                                                            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}