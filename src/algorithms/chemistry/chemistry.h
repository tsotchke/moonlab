/**
 * @file chemistry.h
 * @brief Quantum chemistry algorithms for molecular simulation
 *
 * This module provides quantum algorithms for chemistry:
 * - Jordan-Wigner transformation (fermion to qubit mapping)
 * - Molecular Hamiltonian construction
 * - UCCSD ansatz for VQE
 * - Hartree-Fock state preparation
 *
 * @stability stable
 * @since v1.0.0
 */

#ifndef CHEMISTRY_H
#define CHEMISTRY_H

#include "../../quantum/state.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

// ============================================================================
// JORDAN-WIGNER TRANSFORMATION
// ============================================================================

/**
 * @brief Fermionic operator type
 */
typedef enum {
    FERMION_CREATE,     // Creation operator a†
    FERMION_ANNIHILATE  // Annihilation operator a
} fermion_op_type_t;

/**
 * @brief Single fermionic operator
 */
typedef struct {
    fermion_op_type_t type;  // Create or annihilate
    uint32_t orbital;        // Orbital index
} fermion_op_t;

/**
 * @brief Pauli operator type for Jordan-Wigner result
 */
typedef enum {
    PAULI_I = 0,  // Identity
    PAULI_X = 1,  // Pauli X
    PAULI_Y = 2,  // Pauli Y
    PAULI_Z = 3   // Pauli Z
} pauli_type_t;

/**
 * @brief Pauli string (tensor product of Pauli operators)
 */
typedef struct {
    pauli_type_t *ops;     // Array of Pauli operators
    uint32_t num_qubits;   // Number of qubits
    double complex coeff;  // Coefficient
} pauli_string_t;

/**
 * @brief Jordan-Wigner transformed operator (sum of Pauli strings)
 */
typedef struct {
    pauli_string_t *terms;  // Array of Pauli strings
    uint32_t num_terms;     // Number of terms
} jw_operator_t;

/**
 * @brief Transform a single fermionic operator using Jordan-Wigner
 *
 * Maps fermionic operators to qubit operators:
 * a†_j = (1/2)(X_j - iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
 * a_j  = (1/2)(X_j + iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
 *
 * @param op Fermionic operator to transform
 * @param num_orbitals Total number of orbitals
 * @return Jordan-Wigner transformed operator
 */
jw_operator_t jw_transform_single(fermion_op_t op, uint32_t num_orbitals);

/**
 * @brief Transform a product of fermionic operators
 *
 * @param ops Array of fermionic operators
 * @param num_ops Number of operators
 * @param num_orbitals Total number of orbitals
 * @return Jordan-Wigner transformed operator
 */
jw_operator_t jw_transform_product(const fermion_op_t *ops, uint32_t num_ops,
                                   uint32_t num_orbitals);

/**
 * @brief Free Jordan-Wigner operator memory
 */
void jw_operator_free(jw_operator_t *op);

/**
 * @brief Free Pauli string memory
 */
void pauli_string_free(pauli_string_t *ps);

// ============================================================================
// MOLECULAR HAMILTONIAN
// ============================================================================

/**
 * @brief One-electron integral (h_pq)
 */
typedef struct {
    uint32_t p, q;       // Orbital indices
    double value;        // Integral value
} one_electron_integral_t;

/**
 * @brief Two-electron integral (h_pqrs) in chemist notation
 */
typedef struct {
    uint32_t p, q, r, s;  // Orbital indices
    double value;         // Integral value
} two_electron_integral_t;

/**
 * @brief Molecular Hamiltonian in second quantization
 *
 * H = Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs h_pqrs a†_p a†_q a_r a_s
 */
typedef struct {
    uint32_t num_orbitals;              // Number of spatial orbitals
    uint32_t num_electrons;             // Number of electrons
    double nuclear_repulsion;           // Nuclear repulsion energy
    one_electron_integral_t *h1;        // One-electron integrals
    uint32_t num_h1;                    // Number of one-electron terms
    two_electron_integral_t *h2;        // Two-electron integrals
    uint32_t num_h2;                    // Number of two-electron terms
} molecular_hamiltonian_t;

/**
 * @brief Qubit Hamiltonian (sum of Pauli strings)
 */
typedef struct {
    pauli_string_t *terms;  // Pauli string terms
    uint32_t num_terms;     // Number of terms
    uint32_t num_qubits;    // Number of qubits
} qubit_hamiltonian_t;

/**
 * @brief Create molecular Hamiltonian from integrals
 *
 * @param num_orbitals Number of spatial orbitals
 * @param num_electrons Number of electrons
 * @param nuclear_repulsion Nuclear repulsion energy
 * @return Empty Hamiltonian structure to populate
 */
molecular_hamiltonian_t *molecular_hamiltonian_create(uint32_t num_orbitals,
                                                       uint32_t num_electrons,
                                                       double nuclear_repulsion);

/**
 * @brief Add one-electron integral to Hamiltonian
 */
void molecular_hamiltonian_add_h1(molecular_hamiltonian_t *h,
                                   uint32_t p, uint32_t q, double value);

/**
 * @brief Add two-electron integral to Hamiltonian
 */
void molecular_hamiltonian_add_h2(molecular_hamiltonian_t *h,
                                   uint32_t p, uint32_t q,
                                   uint32_t r, uint32_t s, double value);

/**
 * @brief Convert molecular Hamiltonian to qubit Hamiltonian
 *
 * Uses Jordan-Wigner transformation to map fermionic operators
 * to Pauli operators.
 *
 * @param mol_h Molecular Hamiltonian
 * @return Qubit Hamiltonian
 */
qubit_hamiltonian_t *molecular_to_qubit_hamiltonian(const molecular_hamiltonian_t *mol_h);

/**
 * @brief Free molecular Hamiltonian
 */
void molecular_hamiltonian_free(molecular_hamiltonian_t *h);

/**
 * @brief Free qubit Hamiltonian
 */
void qubit_hamiltonian_free(qubit_hamiltonian_t *h);

/**
 * @brief Compute expectation value of qubit Hamiltonian
 *
 * @param h Qubit Hamiltonian
 * @param state Quantum state
 * @return Expectation value <H>
 */
double qubit_hamiltonian_expectation(const qubit_hamiltonian_t *h,
                                      const quantum_state_t *state);

// ============================================================================
// UCCSD ANSATZ
// ============================================================================

/**
 * @brief UCCSD (Unitary Coupled Cluster Singles and Doubles) configuration
 */
typedef struct {
    uint32_t num_orbitals;     // Number of spatial orbitals
    uint32_t num_electrons;    // Number of electrons
    uint32_t num_singles;      // Number of single excitations
    uint32_t num_doubles;      // Number of double excitations
    double *amplitudes;        // Variational parameters
    uint32_t num_amplitudes;   // Total number of parameters
} uccsd_config_t;

/**
 * @brief Single excitation indices
 */
typedef struct {
    uint32_t i;  // Occupied orbital
    uint32_t a;  // Virtual orbital
} single_excitation_t;

/**
 * @brief Double excitation indices
 */
typedef struct {
    uint32_t i, j;  // Occupied orbitals
    uint32_t a, b;  // Virtual orbitals
} double_excitation_t;

/**
 * @brief Create UCCSD configuration
 *
 * @param num_orbitals Number of spatial orbitals
 * @param num_electrons Number of electrons
 * @return UCCSD configuration with zero amplitudes
 */
uccsd_config_t *uccsd_config_create(uint32_t num_orbitals, uint32_t num_electrons);

/**
 * @brief Free UCCSD configuration
 */
void uccsd_config_free(uccsd_config_t *config);

/**
 * @brief Apply UCCSD ansatz to Hartree-Fock state
 *
 * Applies exp(T - T†) where T = T1 + T2:
 * T1 = Σ_ia t_ia a†_a a_i (singles)
 * T2 = Σ_ijab t_ijab a†_a a†_b a_j a_i (doubles)
 *
 * @param state Quantum state (initialized to HF reference)
 * @param config UCCSD configuration with amplitudes
 * @return Success or error code
 */
qs_error_t uccsd_apply(quantum_state_t *state, const uccsd_config_t *config);

/**
 * @brief Apply single excitation operator exp(t(a†_a a_i - a†_i a_a))
 *
 * @param state Quantum state
 * @param i Occupied orbital
 * @param a Virtual orbital
 * @param t Amplitude
 * @param num_orbitals Total orbitals
 * @return Success or error code
 */
qs_error_t uccsd_apply_single(quantum_state_t *state,
                               uint32_t i, uint32_t a, double t,
                               uint32_t num_orbitals);

/**
 * @brief Apply double excitation operator
 *
 * @param state Quantum state
 * @param i, j Occupied orbitals
 * @param a, b Virtual orbitals
 * @param t Amplitude
 * @param num_orbitals Total orbitals
 * @return Success or error code
 */
qs_error_t uccsd_apply_double(quantum_state_t *state,
                               uint32_t i, uint32_t j,
                               uint32_t a, uint32_t b, double t,
                               uint32_t num_orbitals);

// ============================================================================
// HARTREE-FOCK STATE PREPARATION
// ============================================================================

/**
 * @brief Prepare Hartree-Fock reference state
 *
 * Creates |1111...0000> with n_electrons ones in lowest orbitals.
 * Uses Jordan-Wigner mapping where orbital j maps to qubit j.
 *
 * @param state Quantum state to initialize
 * @param num_electrons Number of electrons
 * @param num_orbitals Number of orbitals
 * @return Success or error code
 */
qs_error_t hartree_fock_state(quantum_state_t *state,
                               uint32_t num_electrons,
                               uint32_t num_orbitals);

// ============================================================================
// MOLECULAR GEOMETRY
// ============================================================================

/**
 * @brief Atom specification
 */
typedef struct {
    char symbol[4];   // Element symbol (e.g., "H", "He", "Li")
    double x, y, z;   // Cartesian coordinates in Angstroms
} atom_t;

/**
 * @brief Molecular geometry
 */
typedef struct {
    atom_t *atoms;      // Array of atoms
    uint32_t num_atoms; // Number of atoms
    int charge;         // Molecular charge
    int multiplicity;   // Spin multiplicity (2S+1)
} molecule_t;

/**
 * @brief Create molecule from atom list
 */
molecule_t *molecule_create(const atom_t *atoms, uint32_t num_atoms,
                             int charge, int multiplicity);

/**
 * @brief Free molecule
 */
void molecule_free(molecule_t *mol);

/**
 * @brief Calculate nuclear repulsion energy
 */
double molecule_nuclear_repulsion(const molecule_t *mol);

/**
 * @brief Create H2 molecule at given bond length
 *
 * @param bond_length H-H bond length in Angstroms
 * @return H2 molecule
 */
molecule_t *molecule_h2(double bond_length);

/**
 * @brief Create LiH molecule at given bond length
 */
molecule_t *molecule_lih(double bond_length);

/**
 * @brief Create H2O molecule with given geometry
 *
 * @param oh_length O-H bond length in Angstroms
 * @param angle H-O-H angle in degrees
 * @return H2O molecule
 */
molecule_t *molecule_h2o(double oh_length, double angle);

#endif /* CHEMISTRY_H */
