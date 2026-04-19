#ifndef GROVER_H
#define GROVER_H

#include "../quantum/state.h"
#include "../utils/quantum_entropy.h"
#include <stddef.h>

/**
 * @file grover.h
 * @brief Grover amplitude-amplification search.
 *
 * OVERVIEW
 * --------
 * Grover's algorithm (Grover 1996) finds a marked element in an
 * unstructured search space of size @f$N = 2^{n}@f$ in
 * @f$O(\sqrt{N})@f$ oracle calls, a provably optimal quadratic
 * speedup over the best classical algorithm
 * @f$O(N)@f$ when the oracle is the only operational primitive
 * (Bennett-Bernstein-Brassard-Vazirani lower bound).  The algorithm
 * consists of a single preparation of the uniform superposition
 * followed by @f$k = \lfloor \pi\sqrt{N/M}/4 \rfloor@f$ iterations of
 * the *Grover operator*
 * @f[
 *   \hat G \;=\; H^{\otimes n}\,(2|0\rangle\langle 0| - \mathbb{1})\,H^{\otimes n}\,\hat O ,
 * @f]
 * where @f$\hat O@f$ is the oracle that flips the phase of the
 * marked state(s), @f$M@f$ is the number of marked states, and the
 * middle factor is the diffusion (reflection about the uniform
 * state).  Each iteration rotates the state by angle
 * @f$\theta = 2\arcsin\sqrt{M/N}@f$ in the 2D subspace spanned by
 * the marked and un-marked uniform states, so success probability
 * after @f$k@f$ iterations is @f$\sin^{2}((2k+1)\theta/2)@f$.
 *
 * Brassard, Hoyer, Mosca and Tapp generalised this to the *amplitude
 * amplification* framework: given any algorithm @f$A@f$ that prepares
 * a state with marked-component amplitude @f$a@f$, @f$O(1/a)@f$
 * applications of a "Grover-like" operator built from @f$A@f$ and the
 * oracle drive the marked amplitude to unity.  Grover's original
 * algorithm is the special case @f$A = H^{\otimes n}@f$ acting on
 * @f$|0\rangle^{\otimes n}@f$.
 *
 * USE CASES IN MOONLAB
 * --------------------
 * - Oracle-based search demonstrations (unit tests, `example_grover_*`).
 * - The Grover-mode QRNG (`qrng.h`) amplifies a marked subset of the
 *   computational basis to produce biased-random bitstreams with a
 *   quantum provenance that is not reproducible by a PRNG.
 * - Cryptographic-grade pre-image / collision search at qubit counts
 *   small enough to be instructive rather than production.
 *
 * REFERENCES
 * ----------
 *  - L. K. Grover, "A fast quantum mechanical algorithm for database
 *    search", STOC 1996, arXiv:quant-ph/9605043.  Original paper.
 *  - G. Brassard, P. Hoyer, M. Mosca and A. Tapp, "Quantum amplitude
 *    amplification and estimation", Contemp. Math. 305, 53 (2002),
 *    arXiv:quant-ph/0005055.  Generalisation and the amplitude
 *    estimation primitive.
 */

/**
 * @brief Grover algorithm configuration
 */
typedef struct {
    size_t num_qubits;          // Number of qubits (search space = 2^n)
    uint64_t marked_state;      // State to search for
    size_t num_iterations;      // Number of Grover iterations (≈π√N/4)
    int use_optimal_iterations; // Auto-calculate optimal iterations
} grover_config_t;

/**
 * @brief Grover algorithm result
 */
typedef struct {
    uint64_t found_state;        // Measured state
    double success_probability;  // P(measuring marked state)
    size_t oracle_calls;         // Number of oracle queries
    size_t iterations_performed; // Actual iterations
    double fidelity;            // |⟨target|final⟩|²
    int found_marked_state;      // 1 if found, 0 if not
} grover_result_t;

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

/**
 * @brief Execute Grover's search algorithm
 *
 * Searches for marked state in O(√N) queries (classical requires O(N)).
 *
 * Algorithm:
 * 1. Initialize: |s⟩ = H⊗ⁿ|0⟩ⁿ (equal superposition)
 * 2. Repeat √N times:
 *    a. Oracle: O|x⟩ = (-1)^f(x)|x⟩ where f(marked) = 1
 *    b. Diffusion: D = 2|s⟩⟨s| - I (amplitude amplification)
 * 3. Measure
 *
 * SECURITY: Uses cryptographically secure entropy for final measurement.
 *
 * @param state Quantum state (will be modified)
 * @param config Algorithm configuration
 * @param entropy Secure entropy source for measurement
 * @return Result including found state and statistics
 */
grover_result_t grover_search(quantum_state_t *state, const grover_config_t *config, quantum_entropy_ctx_t *entropy);

/**
 * @brief Calculate optimal number of Grover iterations
 * 
 * For N items, optimal = ⌊π√N/4⌋
 * Gives maximum success probability ≈ 1 - 1/N
 * 
 * @param num_qubits Number of qubits
 * @return Optimal number of iterations
 */
size_t grover_optimal_iterations(size_t num_qubits);

/**
 * @brief Oracle operator for Grover's algorithm
 * 
 * Marks target state with phase flip: |target⟩ → -|target⟩
 * Implements: O|x⟩ = (-1)^f(x)|x⟩ where f(marked) = 1, else 0
 * 
 * @param state Quantum state
 * @param marked_state Index of state to mark
 * @return QS_SUCCESS or error
 */
qs_error_t grover_oracle(quantum_state_t *state, uint64_t marked_state);

/**
 * @brief Diffusion operator (inversion about average)
 * 
 * Implements: D = 2|s⟩⟨s| - I where |s⟩ = H⊗ⁿ|0⟩ⁿ
 * This amplifies amplitude of marked state.
 * 
 * @param state Quantum state
 * @return QS_SUCCESS or error
 */
qs_error_t grover_diffusion(quantum_state_t *state);

/**
 * @brief Single Grover iteration (oracle + diffusion)
 * 
 * One complete Grover iteration: G = D·O
 * 
 * @param state Quantum state
 * @param marked_state State to amplify
 * @return QS_SUCCESS or error
 */
qs_error_t grover_iteration(quantum_state_t *state, uint64_t marked_state);

// ============================================================================
// RANDOM SAMPLING USING GROVER
// ============================================================================

/**
 * @brief Generate random number using Grover's algorithm
 *
 * Uses quantum superposition and Grover iterations to sample
 * from quantum distribution. Demonstrates quantum enhancement.
 *
 * @param state Quantum state
 * @param num_qubits Number of qubits (determines range: 0 to 2^n-1)
 * @param entropy Secure entropy source
 * @return Sampled random number
 */
uint64_t grover_random_sample(quantum_state_t *state, size_t num_qubits, quantum_entropy_ctx_t *entropy);

/**
 * @brief Generate multiple random samples using Grover
 *
 * @param state Quantum state
 * @param num_qubits Number of qubits
 * @param samples Output array
 * @param num_samples Number of samples to generate
 * @param entropy Secure entropy source
 * @return QS_SUCCESS or error
 */
qs_error_t grover_random_samples(
    quantum_state_t *state,
    size_t num_qubits,
    uint64_t *samples,
    size_t num_samples,
    quantum_entropy_ctx_t *entropy
);

// ============================================================================
// ANALYSIS & VERIFICATION
// ============================================================================

/**
 * @brief Analyze Grover's algorithm performance
 *
 * Runs multiple Grover searches and analyzes:
 * - Success rate
 * - Average iterations to success
 * - Quantum speedup vs classical
 *
 * @param num_qubits Number of qubits
 * @param num_trials Number of trials to run
 * @param entropy Secure entropy source for measurements
 */
typedef struct {
    double success_rate;         // Fraction of successful searches
    double avg_iterations;       // Average iterations used
    double quantum_speedup;      // √N (theoretical)
    double measured_speedup;     // Observed speedup
    size_t total_oracle_calls;   // Total oracle queries
} grover_analysis_t;

grover_analysis_t grover_analyze_performance(size_t num_qubits, size_t num_trials, quantum_entropy_ctx_t *entropy);

/**
 * @brief Print Grover result
 * @param result Grover result
 */
void grover_print_result(const grover_result_t *result, const grover_config_t *config);

// ============================================================================
// ADVANCED GROVER OPERATIONS (V3.0 ENHANCEMENTS)
// ============================================================================

/**
 * @brief Adaptive Grover search with automatic iteration optimization
 *
 * Automatically determines optimal number of iterations and adjusts
 * based on success probability feedback.
 *
 * @param state Quantum state
 * @param marked_state Target to find
 * @param entropy Secure entropy source
 * @return Grover result with adaptive optimization stats
 */
grover_result_t grover_adaptive_search(
    quantum_state_t *state,
    uint64_t marked_state,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Multi-oracle Grover with individual phase controls
 *
 * Advanced oracle that marks multiple states with configurable phases.
 * Allows fine-grained control over amplitude distribution.
 *
 * @param state Quantum state
 * @param marked_states Array of states to mark
 * @param phases Phase angle for each marked state (in radians)
 * @param num_marked Number of marked states
 * @return QS_SUCCESS or error
 */
qs_error_t grover_oracle_multi_phase(
    quantum_state_t *state,
    const uint64_t *marked_states,
    const double *phases,
    size_t num_marked
);

/**
 * @brief Amplitude amplification for arbitrary target state
 *
 * Generalized amplitude amplification that can amplify any
 * set of basis states with custom amplitude targets.
 *
 * @param state Quantum state
 * @param target_amplitudes Desired amplitude for each basis state
 * @param num_iterations Number of amplification iterations
 * @return QS_SUCCESS or error
 */
qs_error_t grover_amplitude_amplification(
    quantum_state_t *state,
    const double *target_amplitudes,
    size_t num_iterations
);

/**
 * @brief Quantum sampling with importance sampling
 *
 * Uses Grover's algorithm to implement quantum importance sampling,
 * which provides quadratic speedup for certain distribution classes.
 *
 * @param state Quantum state
 * @param importance_function Weight function for importance sampling
 * @param num_samples Number of samples to generate
 * @param samples Output array for samples
 * @param entropy Secure entropy source
 * @return QS_SUCCESS or error
 */
qs_error_t grover_importance_sampling(
    quantum_state_t *state,
    double (*importance_function)(uint64_t),
    size_t num_samples,
    uint64_t *samples,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Quantum Metropolis-Hastings using Grover
 *
 * Implements quantum-enhanced MCMC sampling using Grover's algorithm
 * for proposal distribution. Provides speedup for peaked target distributions.
 *
 * @param state Quantum state
 * @param target_distribution Target distribution to sample from
 * @param current_state Current MCMC state
 * @param entropy Secure entropy source
 * @return Next MCMC state
 */
uint64_t grover_mcmc_step(
    quantum_state_t *state,
    double (*target_distribution)(uint64_t),
    uint64_t current_state,
    quantum_entropy_ctx_t *entropy
);

#endif /* GROVER_H */
