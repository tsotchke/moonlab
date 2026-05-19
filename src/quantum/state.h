#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>      /* for FILE in *_fprint signatures */
#include <complex.h>
#include <math.h>
#ifdef __cplusplus
extern "C" {
#endif


// Use C99 complex types
typedef double _Complex complex_t;

/**
 * @file quantum_state.h
 * @brief Advanced quantum state vector simulation engine
 *
 * Implements full quantum circuit simulation with:
 * - Up to 32 qubits (4.3B dimensional state space) - Phase 4 scaling
 * - Universal gate set (Pauli, Hadamard, Phase, CNOT, Toffoli)
 * - Wavefunction collapse on measurement
 * - Quantum entanglement and superposition
 * - Bell inequality violation capability
 *
 * M2 Ultra with 192GB RAM can handle:
 * - 32 qubits: 4.3B states = 68.7GB (comfortable)
 * - 30 qubits: 1.07B states = 17.2GB (easy)
 * - 28 qubits: 268M states = 4.3GB (trivial)
 */

/* Properly namespaced.  The unprefixed `MAX_QUBITS` was prone to
 * collision with vendored Qiskit-Aer / cuStateVec headers that use the
 * same name, so we rename to `MOONLAB_MAX_QUBITS` and leave a
 * deprecated alias for one cycle.  The deprecated alias will be
 * removed in v0.3.0; new code should use the prefixed name. */
/* On wasm32 (emscripten default), size_t is 32 bits.  Bound the
 * compile-time cap so `(size_t)1 << MOONLAB_MAX_QUBITS` cannot
 * overflow size_t on that target.  Native 64-bit hosts keep the
 * historical 32-qubit cap (4.3B amps x 16 B = 68.7 GB). */
#if defined(__wasm32__) || (defined(__EMSCRIPTEN__) && !defined(__wasm64__))
#define MOONLAB_MAX_QUBITS         30  /* 1.07B amps x 16 B = 17.2 GB; size_t = 32-bit. */
#else
#define MOONLAB_MAX_QUBITS         32  /* 4.3B amps × 16 B = 68.7 GB. */
#endif
#define MOONLAB_MAX_STATE_DIM      ((size_t)1 << MOONLAB_MAX_QUBITS)
#define MOONLAB_RECOMMENDED_QUBITS 28  /* 268M amps = 4.3 GB. */

/* Static guards against silent shift-overflow if a future patch
 * raises MOONLAB_MAX_QUBITS past the available size_t width on the
 * current target. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(MOONLAB_MAX_QUBITS < (int)(sizeof(size_t) * 8),
               "MOONLAB_MAX_QUBITS must fit in size_t on this target.");
_Static_assert(MOONLAB_MAX_QUBITS <= 63,
               "MOONLAB_MAX_QUBITS must be <= 63 to avoid 1ULL shift overflow.");
#endif

/* Deprecated unprefixed aliases (removal scheduled for v0.3.0). */
#define MAX_QUBITS              MOONLAB_MAX_QUBITS
#define MAX_STATE_DIM           MOONLAB_MAX_STATE_DIM
#define RECOMMENDED_MAX_QUBITS  MOONLAB_RECOMMENDED_QUBITS

/* Error codes specific to quantum simulation.  These live in their
 * own enum for historical reasons (predates the centralised
 * `moonlab_status_t` registry in `src/utils/moonlab_status.h`).  All
 * values are int-compatible with `moonlab_status_t`, so a function
 * declared `moonlab_status_t f()` may return any QS_* literal and a
 * function declared `qs_error_t f()` may be assigned to a
 * `moonlab_status_t`.  New modules should prefer `moonlab_status_t`
 * directly; the QS_* names remain stable for the v0.x series. */
#include "../utils/moonlab_status.h"

#include "../applications/moonlab_api.h"

typedef enum {
    QS_SUCCESS = 0,
    QS_ERROR_INVALID_QUBIT = -1,
    QS_ERROR_INVALID_STATE = -2,
    QS_ERROR_NOT_NORMALIZED = -3,
    QS_ERROR_OUT_OF_MEMORY = -4,
    QS_ERROR_INVALID_DIMENSION = -5
} qs_error_t;

/**
 * @brief Quantum state representation
 *
 * Represents a pure quantum state |ψ⟩ = Σ αᵢ|i⟩ where:
 * - αᵢ are complex amplitudes
 * - Σ|αᵢ|² = 1 (normalization)
 * - |i⟩ are computational basis states
 *
 * @thread-safety NOT thread-safe. Concurrent callers must serialize
 *                all mutating operations (gates, measurement, reset).
 *                Read-only accessors on a stable state are safe.
 */
typedef struct {
    size_t num_qubits;              /* Number of qubits */
    size_t state_dim;               /* 2^num_qubits */
    complex_t *amplitudes;          /* State vector coefficients */

    /* Measurement history for verification (circular buffer cap; see
     * @c quantum_state_record_measurement and
     * @c quantum_state_get_measurement_history). */
    uint64_t *measurement_outcomes;
    size_t num_measurements;
    size_t max_measurements;

    /* Memory management. */
    int owns_memory;                /* 1 if we allocated amplitudes */
} quantum_state_t;

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/**
 * @brief Initialize quantum state in |0...0⟩
 *
 * Phase 4: Scales to 32 qubits (4.3B states, 68.7GB with 192GB RAM)
 * Uses Accelerate framework with AMX-aligned memory (64-byte boundaries)
 * for optimal M2 Ultra performance.
 *
 * Memory Requirements:
 * - 20 qubits: 1M states = 16MB
 * - 22 qubits: 4.2M states = 67MB
 * - 24 qubits: 16.8M states = 268MB
 * - 26 qubits: 67M states = 1.1GB
 * - 28 qubits: 268M states = 4.3GB (recommended max for speed)
 * - 30 qubits: 1.07B states = 17.2GB
 * - 32 qubits: 4.3B states = 68.7GB (feasible with 192GB RAM!)
 *
 * @param state Pointer to quantum state structure
 * @param num_qubits Number of qubits (1-32, recommended 28 for speed/memory balance)
 * @return QS_SUCCESS or error code
 */
MOONLAB_API qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits);

/**
 * @brief Free quantum state resources
 * @param state Quantum state to free
 */
MOONLAB_API void quantum_state_free(quantum_state_t *state);

/**
 * @brief Create arbitrary quantum state from amplitudes
 * @param state Quantum state to initialize
 * @param amplitudes Array of complex amplitudes
 * @param dim Dimension of state space
 * @return QS_SUCCESS or error code
 */
MOONLAB_API qs_error_t quantum_state_from_amplitudes(
    quantum_state_t *state,
    const complex_t *amplitudes,
    size_t dim
);

/**
 * @brief Clone quantum state (deep copy)
 * @param dest Destination state
 * @param src Source state
 * @return QS_SUCCESS or error code
 */
MOONLAB_API qs_error_t quantum_state_clone(quantum_state_t *dest, const quantum_state_t *src);

/**
 * @brief Reset state to |0...0⟩
 * @param state Quantum state to reset
 */
MOONLAB_API void quantum_state_reset(quantum_state_t *state);

// ============================================================================
// STATE PROPERTIES
// ============================================================================

/**
 * @brief Check if state is normalized (Σ|αᵢ|² = 1)
 * @param state Quantum state
 * @param tolerance Tolerance for normalization check
 * @return 1 if normalized, 0 otherwise
 */
int quantum_state_is_normalized(const quantum_state_t *state, double tolerance);

/**
 * @brief Normalize quantum state
 * @param state Quantum state to normalize
 * @return QS_SUCCESS or error code
 */
MOONLAB_API qs_error_t quantum_state_normalize(quantum_state_t *state);

/**
 * @brief Calculate von Neumann entropy S = -Tr(ρ log₂ ρ)
 * @param state Quantum state
 * @return Entropy in bits
 */
double quantum_state_entropy(const quantum_state_t *state);

/**
 * @brief ‖ψ‖⁴ (= Tr(ρ²) for a pure state). Always 1 for normalized |ψ⟩.
 *        For subsystem purity use entanglement_purity() (entanglement.h).
 */
double quantum_state_purity(const quantum_state_t *state);

/**
 * @brief Calculate fidelity F = |⟨ψ|φ⟩|²
 * @param state1 First quantum state
 * @param state2 Second quantum state
 * @return Fidelity between 0 and 1
 */
double quantum_state_fidelity(const quantum_state_t *state1, const quantum_state_t *state2);

/**
 * @brief Get probability amplitude for basis state |i⟩
 * @param state Quantum state
 * @param basis_index Index of basis state
 * @return Complex amplitude αᵢ
 */
complex_t quantum_state_get_amplitude(const quantum_state_t *state, uint64_t basis_index);

/**
 * @brief Get probability for measuring basis state |i⟩
 * @param state Quantum state
 * @param basis_index Index of basis state
 * @return Probability |αᵢ|²
 */
MOONLAB_API double quantum_state_get_probability(const quantum_state_t *state, uint64_t basis_index);

// ============================================================================
// ENTANGLEMENT MEASURES
// ============================================================================

/**
 * @brief Calculate entanglement entropy between subsystems
 * 
 * Computes von Neumann entropy of reduced density matrix.
 * For pure bipartite states: S(ρ_A) = S(ρ_B) quantifies entanglement.
 * 
 * @param state Full quantum state
 * @param qubits_subsystem_a Qubits in subsystem A
 * @param num_qubits_a Number of qubits in A
 * @return Entanglement entropy in bits
 */
double quantum_state_entanglement_entropy(
    const quantum_state_t *state,
    const int *qubits_subsystem_a,
    size_t num_qubits_a
);

/**
 * @brief Compute reduced density matrix for subsystem
 * 
 * Partial trace: ρ_A = Tr_B(|ψ⟩⟨ψ|)
 * 
 * @param state Full quantum state
 * @param qubits_to_trace Qubits to trace out
 * @param num_traced Number of qubits to trace
 * @param reduced_density Output: reduced density matrix
 * @return QS_SUCCESS or error code
 */
qs_error_t quantum_state_partial_trace(
    const quantum_state_t *state,
    const int *qubits_to_trace,
    size_t num_traced,
    complex_t *reduced_density
);

// ============================================================================
// MEASUREMENT HISTORY
// ============================================================================

/**
 * @brief Record measurement outcome
 * @param state Quantum state
 * @param outcome Measurement outcome (basis index)
 */
void quantum_state_record_measurement(quantum_state_t *state, uint64_t outcome);

/**
 * @brief Get measurement history
 * @param state Quantum state
 * @param outcomes Output array for outcomes
 * @param max_outcomes Maximum outcomes to retrieve
 * @return Number of measurements retrieved
 */
size_t quantum_state_get_measurement_history(
    const quantum_state_t *state,
    uint64_t *outcomes,
    size_t max_outcomes
);

/**
 * @brief Clear measurement history
 * @param state Quantum state
 */
void quantum_state_clear_measurements(quantum_state_t *state);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Print quantum state to a chosen FILE stream (debug).
 *
 * Lets callers redirect to a file, pipe, or a memory-backed FILE; the
 * original ::quantum_state_print is kept as a thin wrapper that always
 * writes to stdout.
 *
 * @param out       Destination stream (must be non-NULL)
 * @param state     Quantum state
 * @param max_terms Maximum number of terms to print (0 = all)
 */
void quantum_state_fprint(FILE *out, const quantum_state_t *state, size_t max_terms);

/**
 * @brief Print quantum state to stdout (debug).
 *
 * Equivalent to `quantum_state_fprint(stdout, state, max_terms)`.  Kept
 * as the legacy entry point; new code should prefer the explicit-FILE
 * variant so that quiet test runs and structured pipelines can redirect.
 *
 * @param state     Quantum state
 * @param max_terms Maximum number of terms to print (0 = all)
 */
void quantum_state_print(const quantum_state_t *state, size_t max_terms);

/**
 * @brief Get string representation of basis state
 * @param basis_index Basis state index
 * @param num_qubits Number of qubits
 * @param buffer Output buffer
 * @param buffer_size Size of buffer
 */
void quantum_basis_state_string(
    uint64_t basis_index,
    size_t num_qubits,
    char *buffer,
    size_t buffer_size
);

// ============================================================================
// CONVENIENCE FUNCTIONS (POINTER-BASED API)
// ============================================================================

/**
 * @brief Allocate and initialize quantum state
 *
 * Allocates a quantum_state_t on heap and initializes to |0...0⟩.
 * Caller must call quantum_state_destroy() when done.
 *
 * @param num_qubits Number of qubits
 * @return Pointer to new state or NULL on error
 */
quantum_state_t* quantum_state_create(int num_qubits);

/**
 * @brief Destroy heap-allocated quantum state
 *
 * Frees resources and the state structure itself.
 *
 * @param state State to destroy (safe to pass NULL)
 */
void quantum_state_destroy(quantum_state_t* state);

/**
 * @brief Reset state to |0...0⟩
 *
 * Alias for quantum_state_reset for API consistency.
 *
 * @param state State to reset
 */
static inline void quantum_state_init_zero(quantum_state_t* state) {
    quantum_state_reset(state);
}

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_STATE_H */
