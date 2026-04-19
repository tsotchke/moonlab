#ifndef QPE_H
#define QPE_H

#include "../quantum/state.h"
#include "../utils/quantum_entropy.h"
#include <stddef.h>

// Use standard complex type from state.h
typedef double _Complex complex_t;

/**
 * @file qpe.h
 * @brief Quantum Phase Estimation.
 *
 * OVERVIEW
 * --------
 * Given a unitary @f$U@f$ and an eigenstate @f$|\psi\rangle@f$ with
 * @f$U|\psi\rangle = e^{2\pi i\varphi}|\psi\rangle@f$, quantum phase
 * estimation returns an @f$m@f$-bit approximation of @f$\varphi \in
 * [0, 1)@f$ with polynomial resources.  The canonical circuit, due
 * to Kitaev (1995) in eigenvalue-sampling form and Cleve, Ekert,
 * Macchiavello & Mosca (1998) in inverse-QFT form, proceeds as:
 *
 *   1. Prepare the @f$m@f$-qubit ancilla register in
 *      @f$H^{\otimes m}|0\rangle^{\otimes m}@f$.
 *   2. Apply controlled-@f$U^{2^k}@f$ with the @f$k@f$-th ancilla
 *      as the control, for @f$k = 0,\ldots,m-1@f$.
 *   3. Apply the inverse quantum Fourier transform to the ancilla.
 *   4. Measure the ancilla in the computational basis; the outcome
 *      @f$y \in \{0,\ldots,2^m - 1\}@f$ gives
 *      @f$\tilde\varphi = y/2^m@f$, within @f$2^{-m}@f$ of
 *      @f$\varphi@f$ with probability @f$\ge 4/\pi^2@f$.
 *
 * QPE is the backbone of Shor's factoring algorithm (the order-
 * finding subroutine is QPE on @f$U_a: |y\rangle \mapsto |a y \bmod
 * N\rangle@f$), HHL-class linear-system solvers, several quantum-
 * chemistry excited-state methods, and the variational quantum
 * eigensolver's "QPE-polished" family.  The bit-ordering convention
 * between the ancilla and the fractional representation of
 * @f$\varphi@f$ is easy to get wrong: our QFT uses the convention
 * where the most significant bit is qubit 0, and the inverse QFT
 * therefore emits a trailing bit-reversal swap so the measurement
 * outcome @f$y@f$ corresponds to @f$\varphi = y/2^m@f$ with no
 * further post-processing (see `qpe_bitstring_to_phase`).
 *
 * PRECISION / RESOURCES
 * ---------------------
 *  - @f$m@f$ ancilla qubits give @f$\varphi@f$ within @f$2^{-m}@f$
 *    of the true eigenphase.
 *  - Moonlab's 32-qubit state vector supports up to @f$m = 16@f$
 *    precision qubits alongside a @f$16@f$-qubit system register, or
 *    any other sum @f$m + n_{\mathrm{sys}} \le 32@f$.
 *  - Iterative variants (Kitaev 1995 style; Svore et al.) trade
 *    circuit depth for shot count but are not implemented here.
 *
 * REFERENCES
 * ----------
 *  - A. Yu. Kitaev, "Quantum measurements and the Abelian Stabilizer
 *    Problem", arXiv:quant-ph/9511026 (1995).  Original
 *    eigenvalue-sampling form.
 *  - R. Cleve, A. Ekert, C. Macchiavello and M. Mosca, "Quantum
 *    algorithms revisited", Proc. R. Soc. Lond. A 454, 339 (1998),
 *    arXiv:quant-ph/9708016.  Inverse-QFT formulation used here.
 */

// ============================================================================
// UNITARY OPERATOR
// ============================================================================

/**
 * @brief Unitary operator for QPE
 */
typedef struct {
    size_t num_qubits;           // Qubits operator acts on
    void *operator_data;         // Opaque operator specification
    
    // Apply U to state
    qs_error_t (*apply)(quantum_state_t *state, void *data);
    
    // Apply U^k to state (for controlled-U^(2^k))
    qs_error_t (*apply_power)(quantum_state_t *state, void *data, uint64_t power);
    
    // Optional: eigenvalue for validation
    complex_t eigenvalue;
} unitary_operator_t;

/**
 * @brief Create unitary operator
 * @param num_qubits Number of qubits
 * @return Initialized operator
 */
unitary_operator_t* unitary_operator_create(size_t num_qubits);

/**
 * @brief Free unitary operator
 * @param op Operator to free
 */
void unitary_operator_free(unitary_operator_t *op);

// ============================================================================
// EIGENSTATE
// ============================================================================

/**
 * @brief Eigenstate of unitary operator
 */
typedef struct {
    quantum_state_t *state;      // Eigenstate |ψ⟩
    complex_t eigenvalue;        // e^(2πiφ)
    double phase;                // φ ∈ [0,1)
} eigenstate_t;

/**
 * @brief Create eigenstate
 * @param num_qubits Number of qubits
 * @return Initialized eigenstate
 */
eigenstate_t* eigenstate_create(size_t num_qubits);

/**
 * @brief Free eigenstate
 * @param es Eigenstate to free
 */
void eigenstate_free(eigenstate_t *es);

// ============================================================================
// QPE ALGORITHM
// ============================================================================

/**
 * @brief QPE configuration
 */
typedef struct {
    size_t precision_qubits;     // m qubits for phase estimation
    size_t system_qubits;        // n qubits for system
    double phase_accuracy;       // Target accuracy (2^(-m))
} qpe_config_t;

/**
 * @brief QPE result
 */
typedef struct {
    double estimated_phase;      // φ estimate in [0, 1)
    complex_t estimated_eigenvalue;  // e^(2πiφ)
    uint64_t phase_bitstring;   // m-bit measurement outcome
    double confidence;           // Probability of correct estimation
    size_t precision_bits;       // Number of precision bits used
    
    // Reference values (if known)
    double true_phase;           // Known phase (for validation)
    double phase_error;          // |estimated - true|
} qpe_result_t;

/**
 * @brief Execute QPE algorithm
 * 
 * Estimates phase φ where U|ψ⟩ = e^(2πiφ)|ψ⟩
 * 
 * @param unitary Unitary operator U
 * @param eigenstate Eigenstate |ψ⟩ of U
 * @param precision_qubits Number of bits of precision (m)
 * @param entropy Entropy source
 * @return QPE result with phase estimate
 */
qpe_result_t qpe_estimate_phase(
    const unitary_operator_t *unitary,
    const eigenstate_t *eigenstate,
    size_t precision_qubits,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Apply controlled-U^k operation
 * 
 * Applies U^k to target if control qubit is |1⟩
 * 
 * @param state Full quantum state
 * @param control Control qubit index
 * @param target_start First target qubit
 * @param unitary Unitary operator
 * @param power k (apply U^k)
 * @return QS_SUCCESS or error
 */
qs_error_t qpe_apply_controlled_unitary_power(
    quantum_state_t *state,
    int control,
    int target_start,
    const unitary_operator_t *unitary,
    uint64_t power
);

/**
 * @brief Convert measured bitstring to phase estimate
 * 
 * @param bitstring Measured m-bit string
 * @param precision_bits Number of bits
 * @return Phase φ ∈ [0, 1)
 */
double qpe_bitstring_to_phase(uint64_t bitstring, size_t precision_bits);

/**
 * @brief Print QPE result
 * @param result QPE result
 */
void qpe_print_result(const qpe_result_t *result);

// ============================================================================
// PRE-BUILT UNITARIES (for testing/demonstration)
// ============================================================================

/**
 * @brief Create phase gate unitary: U|ψ⟩ = e^(iθ)|ψ⟩
 * @param theta Phase angle
 * @return Unitary operator
 */
unitary_operator_t* qpe_create_phase_gate(double theta);

/**
 * @brief Create T gate unitary: U|1⟩ = e^(iπ/4)|1⟩
 * @return Unitary operator
 */
unitary_operator_t* qpe_create_t_gate(void);

/**
 * @brief Create rotation unitary: RZ(θ)
 * @param theta Rotation angle
 * @return Unitary operator
 */
unitary_operator_t* qpe_create_rz_gate(double theta);

#endif /* QPE_H */
