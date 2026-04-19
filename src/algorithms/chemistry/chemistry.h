/**
 * @file chemistry.h
 * @brief Quantum chemistry primitives: fermion-to-qubit mapping, HF
 *        reference state, unitary coupled cluster ansatz.
 *
 * OVERVIEW
 * --------
 * Second-quantised electronic-structure problems live in the
 * antisymmetric Fock space of fermionic creation / annihilation
 * operators @f$\{a_p, a_p^{\dagger}\}@f$ obeying
 * @f$\{a_p, a_q^{\dagger}\} = \delta_{pq}@f$.  To run them on a
 * quantum computer they must first be mapped to qubit operators; the
 * *Jordan-Wigner transformation* (Jordan-Wigner 1928) is the
 * canonical mapping:
 * @f[
 *   a_p \;=\; \tfrac12\,Z_0 \cdots Z_{p-1}\,(X_p + i Y_p),
 *   \qquad
 *   a_p^{\dagger} \;=\; \tfrac12\,Z_0 \cdots Z_{p-1}\,(X_p - i Y_p),
 * @f]
 * with a nonlocal Z-string ("parity") that preserves anticommutation
 * at the cost of @f$O(N)@f$ single-qubit operators per fermion
 * operator.  Bravyi-Kitaev and its variants trade locality against
 * tree depth; this module currently implements only Jordan-Wigner.
 * The molecular electronic Hamiltonian
 * @f[
 *   H \;=\; \sum_{pq} h_{pq}\,a_p^{\dagger} a_q
 *     + \tfrac12 \sum_{pqrs} h_{pqrs}\,
 *       a_p^{\dagger} a_q^{\dagger} a_r a_s
 * @f]
 * with integrals @f$h_{pq}, h_{pqrs}@f$ from a classical
 * Hartree-Fock or CASSCF precomputation is transformed via
 * Jordan-Wigner into a weighted sum of Pauli strings, consumed by
 * the VQE engine (`vqe.h`).
 *
 * The *unitary coupled-cluster* ansatz with singles and doubles
 * (UCCSD) approximates the exact ground state by
 * @f[
 *   |\psi_{\mathrm{UCCSD}}\rangle \;=\;
 *     e^{T - T^{\dagger}}\,|\mathrm{HF}\rangle,
 *   \qquad
 *   T \;=\; \sum_{ia} t_i^a a_a^{\dagger} a_i
 *         + \tfrac14 \sum_{ijab} t_{ij}^{ab} a_a^{\dagger}
 *           a_b^{\dagger} a_j a_i,
 * @f]
 * where @f$i, j@f$ range over occupied Hartree-Fock orbitals and
 * @f$a, b@f$ over virtuals.  Jordan-Wigner maps @f$T - T^{\dagger}@f$
 * to a sum of anti-Hermitian Pauli strings; a first-order Trotter
 * split produces a parameterised quantum circuit whose parameters
 * @f$\{t\}@f$ are optimised classically by the variational loop.
 * Peruzzo et al. (2014) introduced VQE with UCCSD on photonic
 * hardware; O'Malley et al. (2016) and Kandala et al. (2017) then
 * scaled it to superconducting qubits (H2, LiH).  McArdle et al. and
 * Cao et al. give modern reviews of the landscape that this module
 * fits into.
 *
 * BUILT-IN MOLECULES
 * ------------------
 * Pre-computed Pauli Hamiltonians for H2, LiH, and H2O are provided
 * along with a Hartree-Fock reference bitmask used by
 * `vqe_compute_energy` to prepare the reference state before the
 * ansatz runs.  All three match the integrals used in the O'Malley
 * and Kandala papers at the equilibrium bond length.
 *
 * REFERENCES
 * ----------
 *  - P. Jordan and E. Wigner, "Ueber das Paulische Aequivalenzverbot",
 *    Z. Physik 47, 631 (1928), doi:10.1007/BF01331938.  Original
 *    fermion-to-qubit mapping.
 *  - A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou,
 *    P. J. Love, A. Aspuru-Guzik and J. L. O'Brien, "A variational
 *    eigenvalue solver on a quantum processor", Nat. Commun. 5, 4213
 *    (2014), arXiv:1304.3061.  VQE with UCCSD.
 *  - P. J. J. O'Malley et al., "Scalable Quantum Simulation of
 *    Molecular Energies", Phys. Rev. X 6, 031007 (2016),
 *    arXiv:1512.06860.  H2 on superconducting hardware; our H2
 *    Hamiltonian matches.
 *  - A. Kandala et al., "Hardware-efficient variational quantum
 *    eigensolver for small molecules and quantum magnets",
 *    Nature 549, 242 (2017), arXiv:1704.05018.  HEA ansatz; origin
 *    of the alternative ansatz surface in `vqe.h`.
 *  - S. McArdle, S. Endo, A. Aspuru-Guzik, S. C. Benjamin and X. Yuan,
 *    "Quantum computational chemistry", Rev. Mod. Phys. 92, 015003
 *    (2020), arXiv:1808.10402.  Canonical review.
 *  - Y. Cao et al., "Quantum Chemistry in the Age of Quantum
 *    Computing", Chem. Rev. 119, 10856 (2019), arXiv:1812.09976.
 *
 * @stability evolving
 * @since v0.1.2
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
 * @brief Prepare the Hartree-Fock Slater-determinant reference state
 *        |1...10...0⟩ (num_electrons qubits set, rest zero).
 *        Not an SCF calculation — feed SCF orbitals into the Hamiltonian.
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
