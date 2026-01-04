/**
 * @file topological.h
 * @brief Topological quantum computing simulation
 *
 * This module implements topological quantum computing primitives:
 * - Anyon models (Fibonacci, Ising)
 * - Braiding operations and F/R matrices
 * - Fusion trees and fusion rules
 * - Surface code and toric code
 * - Topological entanglement entropy
 * - Anyonic gates via braiding
 *
 * Topological quantum computing encodes information in non-local,
 * topologically protected degrees of freedom, providing inherent
 * fault tolerance against local perturbations.
 *
 * @stability stable
 * @since v1.0.0
 */

#ifndef TOPOLOGICAL_H
#define TOPOLOGICAL_H

#include "../../quantum/state.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

// ============================================================================
// ANYON MODELS
// ============================================================================

/**
 * @brief Anyon type enumeration
 */
typedef enum {
    ANYON_MODEL_FIBONACCI,  // Fibonacci anyons (τ×τ = 1+τ)
    ANYON_MODEL_ISING,      // Ising anyons (σ×σ = 1+ψ)
    ANYON_MODEL_SU2_K       // SU(2)_k anyons
} anyon_model_t;

/**
 * @brief Anyon charge labels
 * For Fibonacci: 1 (vacuum), τ (tau)
 * For Ising: 1 (vacuum), σ (sigma), ψ (psi)
 */
typedef uint32_t anyon_charge_t;

// Fibonacci anyon charges
#define FIB_VACUUM 0
#define FIB_TAU    1

// Ising anyon charges
#define ISING_VACUUM 0
#define ISING_SIGMA  1
#define ISING_PSI    2

/**
 * @brief Anyon model specification
 */
typedef struct {
    anyon_model_t type;
    uint32_t num_charges;           // Number of distinct charges
    uint32_t level;                 // Level k for SU(2)_k
    double complex **F_matrices;    // F-symbols (6j-symbols)
    double complex **R_matrices;    // R-symbols (braiding phases)
    uint32_t ***fusion_rules;       // N^c_{ab} fusion multiplicities
} anyon_system_t;

/**
 * @brief Initialize Fibonacci anyon system
 *
 * Fibonacci anyons have fusion rule τ×τ = 1+τ and are universal
 * for quantum computation via braiding alone.
 *
 * @return Fibonacci anyon system
 */
anyon_system_t *anyon_system_fibonacci(void);

/**
 * @brief Initialize Ising anyon system
 *
 * Ising anyons have fusion rules:
 * σ×σ = 1+ψ, σ×ψ = σ, ψ×ψ = 1
 *
 * @return Ising anyon system
 */
anyon_system_t *anyon_system_ising(void);

/**
 * @brief Initialize SU(2)_k anyon system
 *
 * @param k Level parameter (k=2 gives Ising, k=3 gives Fibonacci)
 * @return SU(2)_k anyon system
 */
anyon_system_t *anyon_system_su2k(uint32_t k);

/**
 * @brief Free anyon system
 */
void anyon_system_free(anyon_system_t *sys);

/**
 * @brief Get quantum dimension of anyon charge
 *
 * For Fibonacci: d_1 = 1, d_τ = φ (golden ratio)
 * For Ising: d_1 = 1, d_σ = √2, d_ψ = 1
 *
 * @param sys Anyon system
 * @param charge Anyon charge
 * @return Quantum dimension
 */
double anyon_quantum_dimension(const anyon_system_t *sys, anyon_charge_t charge);

/**
 * @brief Get total quantum dimension D = √(Σ d_a²)
 */
double anyon_total_dimension(const anyon_system_t *sys);

// ============================================================================
// FUSION TREES
// ============================================================================

/**
 * @brief Fusion tree node
 *
 * Represents the fusion of two anyons into a third:
 * a × b → c (with multiplicity N^c_{ab})
 */
typedef struct fusion_node {
    anyon_charge_t left;         // Left incoming charge
    anyon_charge_t right;        // Right incoming charge
    anyon_charge_t result;       // Outgoing fused charge
    struct fusion_node *parent;  // Parent in tree
    struct fusion_node *left_child;
    struct fusion_node *right_child;
} fusion_node_t;

/**
 * @brief Fusion tree state
 *
 * A fusion tree represents a specific way of fusing n anyons
 * to obtain a total charge. The state is a superposition over
 * valid intermediate fusion outcomes.
 */
typedef struct {
    anyon_system_t *anyon_sys;     // Anyon model
    anyon_charge_t *external;       // External (physical) anyon charges
    uint32_t num_anyons;            // Number of external anyons
    anyon_charge_t total_charge;    // Total fused charge
    fusion_node_t *root;            // Root of fusion tree
    double complex *amplitudes;     // Amplitudes for each fusion path
    uint32_t num_paths;             // Number of valid fusion paths
} fusion_tree_t;

/**
 * @brief Create fusion tree from external charges
 *
 * Enumerates all valid fusion paths and initializes amplitudes.
 *
 * @param sys Anyon system
 * @param charges External anyon charges
 * @param num_anyons Number of anyons
 * @param total_charge Required total charge
 * @return Fusion tree state
 */
fusion_tree_t *fusion_tree_create(anyon_system_t *sys,
                                   const anyon_charge_t *charges,
                                   uint32_t num_anyons,
                                   anyon_charge_t total_charge);

/**
 * @brief Free fusion tree
 */
void fusion_tree_free(fusion_tree_t *tree);

/**
 * @brief Count fusion paths
 *
 * Returns the dimension of the fusion space for given charges.
 *
 * @param sys Anyon system
 * @param charges External charges
 * @param num_anyons Number of anyons
 * @param total_charge Total charge
 * @return Number of distinct fusion paths
 */
uint32_t fusion_count_paths(const anyon_system_t *sys,
                            const anyon_charge_t *charges,
                            uint32_t num_anyons,
                            anyon_charge_t total_charge);

// ============================================================================
// BRAIDING OPERATIONS
// ============================================================================

/**
 * @brief Braid two adjacent anyons
 *
 * Exchanges anyons at positions i and i+1, applying the
 * appropriate R-matrix phase and F-matrix basis change.
 *
 * @param tree Fusion tree (modified in place)
 * @param position Position of left anyon to braid
 * @param clockwise Direction of braid (true = σ, false = σ⁻¹)
 * @return QS_SUCCESS or error
 */
qs_error_t braid_anyons(fusion_tree_t *tree, uint32_t position, bool clockwise);

/**
 * @brief Apply F-move (basis change)
 *
 * Changes the fusion order at a vertex using F-matrix.
 * (a×b)×c ↔ a×(b×c)
 *
 * @param tree Fusion tree
 * @param vertex Vertex to apply F-move
 * @return QS_SUCCESS or error
 */
qs_error_t apply_F_move(fusion_tree_t *tree, uint32_t vertex);

/**
 * @brief Get F-matrix element
 *
 * F^{abc}_d[e,f] relates different fusion orderings:
 * (a×b→e)×c→d ↔ a×(b×c→f)→d
 *
 * @param sys Anyon system
 * @param a,b,c,d External charges
 * @param e,f Intermediate channels
 * @return F-matrix element
 */
double complex get_F_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c, anyon_charge_t d,
                            anyon_charge_t e, anyon_charge_t f);

/**
 * @brief Get R-matrix element
 *
 * R^{ab}_c is the phase acquired when exchanging a and b
 * that fuse to c.
 *
 * @param sys Anyon system
 * @param a,b Exchanged charges
 * @param c Fusion outcome
 * @return R-matrix element (phase)
 */
double complex get_R_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c);

// ============================================================================
// ANYONIC QUANTUM GATES
// ============================================================================

/**
 * @brief Anyonic qubit encoding
 *
 * For Fibonacci anyons, a qubit is encoded in 4 anyons
 * with total charge 1: |0⟩ ~ (τ,τ)→1, |1⟩ ~ (τ,τ)→τ
 */
typedef struct {
    fusion_tree_t *tree;
    anyon_system_t *sys;
    uint32_t num_logical_qubits;
} anyonic_register_t;

/**
 * @brief Create anyonic qubit register
 *
 * @param sys Anyon system
 * @param num_qubits Number of logical qubits
 * @return Anyonic register
 */
anyonic_register_t *anyonic_register_create(anyon_system_t *sys,
                                             uint32_t num_qubits);

/**
 * @brief Free anyonic register
 */
void anyonic_register_free(anyonic_register_t *reg);

/**
 * @brief Apply NOT gate via braiding
 *
 * For Fibonacci qubits, NOT is achieved by braiding
 * the middle two anyons.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @return QS_SUCCESS or error
 */
qs_error_t anyonic_not(anyonic_register_t *reg, uint32_t qubit);

/**
 * @brief Apply Hadamard-like gate via braiding
 *
 * Fibonacci anyons can approximate H to arbitrary precision
 * using appropriate braid sequences.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @return QS_SUCCESS or error
 */
qs_error_t anyonic_hadamard(anyonic_register_t *reg, uint32_t qubit);

/**
 * @brief Apply T gate approximation via braiding
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @param precision Approximation precision
 * @return QS_SUCCESS or error
 */
qs_error_t anyonic_T_gate(anyonic_register_t *reg, uint32_t qubit,
                          double precision);

/**
 * @brief Apply two-qubit entangling gate
 *
 * @param reg Anyonic register
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return QS_SUCCESS or error
 */
qs_error_t anyonic_entangle(anyonic_register_t *reg,
                            uint32_t qubit1, uint32_t qubit2);

// ============================================================================
// SURFACE CODE
// ============================================================================

/**
 * @brief Surface code lattice
 *
 * Implements the 2D surface code on a square lattice with
 * distance d (d×d data qubits, (d-1)² syndrome qubits).
 */
typedef struct {
    uint32_t distance;           // Code distance
    uint32_t num_data_qubits;    // d²
    uint32_t num_ancilla_qubits; // (d-1)² for each type (X and Z)
    quantum_state_t *state;      // Full quantum state
    uint8_t *x_syndrome;         // X-type syndrome measurements
    uint8_t *z_syndrome;         // Z-type syndrome measurements
} surface_code_t;

/**
 * @brief Create surface code
 *
 * @param distance Code distance (odd, ≥3)
 * @return Surface code structure
 */
surface_code_t *surface_code_create(uint32_t distance);

/**
 * @brief Free surface code
 */
void surface_code_free(surface_code_t *code);

/**
 * @brief Initialize surface code in logical |0⟩
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_init_logical_zero(surface_code_t *code);

/**
 * @brief Initialize surface code in logical |+⟩
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_init_logical_plus(surface_code_t *code);

/**
 * @brief Apply logical X gate
 *
 * String of X operators along a path from left to right edge.
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_logical_X(surface_code_t *code);

/**
 * @brief Apply logical Z gate
 *
 * String of Z operators along a path from top to bottom edge.
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_logical_Z(surface_code_t *code);

/**
 * @brief Measure X-type stabilizers
 *
 * Measures all face (plaquette) stabilizers.
 *
 * @param code Surface code (syndrome updated)
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_measure_X_stabilizers(surface_code_t *code);

/**
 * @brief Measure Z-type stabilizers
 *
 * Measures all vertex (star) stabilizers.
 *
 * @param code Surface code (syndrome updated)
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_measure_Z_stabilizers(surface_code_t *code);

/**
 * @brief Apply single-qubit error
 *
 * @param code Surface code
 * @param qubit Data qubit index
 * @param error_type 'X', 'Y', or 'Z'
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_apply_error(surface_code_t *code,
                                     uint32_t qubit, char error_type);

/**
 * @brief Decode syndrome and apply correction
 *
 * Uses minimum weight perfect matching decoder.
 *
 * @param code Surface code (corrected in place)
 * @return QS_SUCCESS or error
 */
qs_error_t surface_code_decode_correct(surface_code_t *code);

// ============================================================================
// TORIC CODE
// ============================================================================

/**
 * @brief Toric code on a torus
 *
 * Similar to surface code but on periodic boundary conditions,
 * encoding 2 logical qubits.
 */
typedef struct {
    uint32_t L;                  // Linear size (L×L torus)
    uint32_t num_qubits;         // 2L² edge qubits
    quantum_state_t *state;      // Full quantum state
    uint8_t *vertex_syndrome;    // A_v eigenvalues
    uint8_t *plaquette_syndrome; // B_p eigenvalues
} toric_code_t;

/**
 * @brief Create toric code
 *
 * @param L Linear size
 * @return Toric code structure
 */
toric_code_t *toric_code_create(uint32_t L);

/**
 * @brief Free toric code
 */
void toric_code_free(toric_code_t *code);

/**
 * @brief Initialize toric code ground state
 *
 * Projects onto the +1 eigenspace of all stabilizers.
 *
 * @param code Toric code
 * @return QS_SUCCESS or error
 */
qs_error_t toric_code_init_ground_state(toric_code_t *code);

/**
 * @brief Create anyon pair
 *
 * Creates an e-anyon (vertex) or m-anyon (plaquette) pair
 * by applying a string of Pauli operators.
 *
 * @param code Toric code
 * @param type 'e' for electric (Z-string), 'm' for magnetic (X-string)
 * @param x1,y1 Start position
 * @param x2,y2 End position
 * @return QS_SUCCESS or error
 */
qs_error_t toric_code_create_anyon_pair(toric_code_t *code,
                                         char type,
                                         uint32_t x1, uint32_t y1,
                                         uint32_t x2, uint32_t y2);

/**
 * @brief Move anyon
 *
 * @param code Toric code
 * @param type Anyon type
 * @param from_x,from_y Current position
 * @param to_x,to_y New position
 * @return QS_SUCCESS or error
 */
qs_error_t toric_code_move_anyon(toric_code_t *code, char type,
                                  uint32_t from_x, uint32_t from_y,
                                  uint32_t to_x, uint32_t to_y);

/**
 * @brief Braid anyons in toric code
 *
 * @param code Toric code
 * @param anyon1_x,anyon1_y First anyon position
 * @param anyon2_x,anyon2_y Second anyon position
 * @return QS_SUCCESS or error
 */
qs_error_t toric_code_braid(toric_code_t *code,
                            uint32_t anyon1_x, uint32_t anyon1_y,
                            uint32_t anyon2_x, uint32_t anyon2_y);

// ============================================================================
// TOPOLOGICAL ENTANGLEMENT ENTROPY
// ============================================================================

/**
 * @brief Compute topological entanglement entropy
 *
 * S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
 * where A, B, C are regions forming an annulus.
 *
 * For topologically ordered states, S_topo = log(D) where
 * D is the total quantum dimension.
 *
 * @param state Quantum state
 * @param region_A Qubits in region A
 * @param num_A Size of region A
 * @param region_B Qubits in region B
 * @param num_B Size of region B
 * @param region_C Qubits in region C
 * @param num_C Size of region C
 * @return Topological entanglement entropy
 */
double topological_entanglement_entropy(const quantum_state_t *state,
                                         const uint32_t *region_A, uint32_t num_A,
                                         const uint32_t *region_B, uint32_t num_B,
                                         const uint32_t *region_C, uint32_t num_C);

/**
 * @brief Compute Kitaev-Preskill topological entropy
 *
 * Alternative formula using disk and annulus regions.
 *
 * @param state Quantum state
 * @param center_qubits Central disk qubits
 * @param num_center Number of center qubits
 * @param ring_qubits Surrounding ring qubits
 * @param num_ring Number of ring qubits
 * @return Topological entropy γ = log(D)
 */
double kitaev_preskill_entropy(const quantum_state_t *state,
                                const uint32_t *center_qubits, uint32_t num_center,
                                const uint32_t *ring_qubits, uint32_t num_ring);

// ============================================================================
// MODULAR S AND T MATRICES
// ============================================================================

/**
 * @brief Compute modular S-matrix
 *
 * S_{ab} = (1/D) Σ_c N^c_{ab} d_c e^{2πi θ_c}
 * where θ_c is the topological spin.
 *
 * @param sys Anyon system
 * @param S_matrix Output S-matrix (num_charges × num_charges)
 */
void compute_modular_S_matrix(const anyon_system_t *sys,
                               double complex *S_matrix);

/**
 * @brief Compute modular T-matrix
 *
 * T_{ab} = δ_{ab} e^{2πi θ_a}
 *
 * @param sys Anyon system
 * @param T_matrix Output T-matrix (num_charges × num_charges)
 */
void compute_modular_T_matrix(const anyon_system_t *sys,
                               double complex *T_matrix);

/**
 * @brief Compute topological spin
 *
 * θ_a = e^{2πi h_a} where h_a is the conformal weight.
 *
 * @param sys Anyon system
 * @param charge Anyon charge
 * @return Topological spin e^{2πi θ}
 */
double complex topological_spin(const anyon_system_t *sys,
                                 anyon_charge_t charge);

#endif /* TOPOLOGICAL_H */
