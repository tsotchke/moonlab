/**
 * @file noise.h
 * @brief Kraus-operator quantum channels for noisy simulation.
 *
 * OVERVIEW
 * --------
 * Physical quantum devices are never unitary; their evolution is a
 * trace-preserving completely positive (CPTP) map on the density
 * matrix.  By the Stinespring-Kraus representation theorem, every
 * CPTP map admits a decomposition
 * @f[
 *   \rho \;\mapsto\; \mathcal E(\rho) \;=\;
 *     \sum_k K_k \,\rho\, K_k^{\dagger},
 *   \qquad
 *   \sum_k K_k^{\dagger} K_k \;=\; \mathbb{1},
 * @f]
 * which is the computational form we implement.  The Kraus operators
 * @f$\{K_k\}@f$ are drawn from the canonical list of device error
 * channels:
 *  - *Depolarising* (isotropic random Pauli)
 *    @f$\mathcal E_p(\rho) = (1-p)\rho +
 *    (p/3)(X\rho X + Y\rho Y + Z\rho Z)@f$.
 *    At @f$p = 3/4@f$ the channel saturates the maximally mixed
 *    state; see `test_depolarizing_uniform` for the verification.
 *  - *Amplitude damping* with rate @f$\gamma@f$ models @f$T_1@f$
 *    relaxation @f$|1\rangle \to |0\rangle@f$.
 *  - *Phase damping* with rate @f$\lambda@f$ models pure @f$T_2@f$
 *    dephasing.
 *  - *Bit-flip*, *phase-flip*, *bit-phase-flip* are the elementary
 *    Pauli channels.
 *
 * The simulator supports both direct (Kraus-applied) and trajectory
 * (stochastic unravelling) modes.  A completeness validator
 * (`noise_kraus_completeness_deviation`) numerically verifies
 * @f$\lVert\sum_k K_k^{\dagger} K_k - \mathbb{1}\rVert_{\max} \le
 * 2.2\times 10^{-16}@f$ across the six built-in channels at five
 * parameter values, serving as a quantitative sanity check that new
 * channels do not violate CPTP by construction.
 *
 * Composite channels (sequential or convex combinations), correlated
 * two-qubit channels, and temporal-correlation (coloured-noise)
 * channels are queued for the 0.3 noise-suite expansion.  The
 * primitives here are the minimum needed for NISQ-era VQE / QAOA
 * noise studies in the sense of Preskill's NISQ review.
 *
 * REFERENCES
 * ----------
 *  - M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum
 *    Information", Cambridge University Press (10th anniversary ed.,
 *    2010).  Chapter 8 is the textbook reference for the Kraus
 *    representation, CPTP maps, and the canonical error channels.
 *  - J. Preskill, "Quantum Computing in the NISQ era and beyond",
 *    Quantum 2, 79 (2018), arXiv:1801.00862.  The context that makes
 *    honest noise simulation (rather than ideal-unitary simulation)
 *    the relevant research object at current-generation qubit counts.
 *  - R. Horodecki, P. Horodecki, M. Horodecki and K. Horodecki,
 *    "Quantum entanglement", Rev. Mod. Phys. 81, 865 (2009),
 *    arXiv:quant-ph/0702225.  Noisy-state entanglement theory that
 *    the measures in entanglement.h evaluate on output of these
 *    channels.
 *
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef QUANTUM_NOISE_H
#define QUANTUM_NOISE_H

#include "state.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NOISE MODEL STRUCTURE
// ============================================================================

/**
 * @brief Quantum noise model configuration
 */
typedef struct {
    int enabled;                    /**< Whether noise is active */

    // Single-qubit error rates
    double depolarizing_rate;       /**< Depolarizing probability */
    double amplitude_damping_rate;  /**< Amplitude damping (T1) */
    double phase_damping_rate;      /**< Phase damping (T2 pure) */

    // Thermal relaxation
    double t1;                      /**< T1 time (energy relaxation) */
    double t2;                      /**< T2 time (dephasing) */
    double gate_time;               /**< Gate duration for thermal calc */

    // Two-qubit error rates
    double two_qubit_depolarizing_rate;

    // Readout errors
    double readout_error_0;         /**< P(1|0) - measure 1 when state is 0 */
    double readout_error_1;         /**< P(0|1) - measure 0 when state is 1 */

} noise_model_t;

// ============================================================================
// NOISE MODEL MANAGEMENT
// ============================================================================

/**
 * @brief Create a noise model
 */
noise_model_t* noise_model_create(void);

/**
 * @brief Destroy noise model
 */
void noise_model_destroy(noise_model_t* model);

/**
 * @brief Copy noise model
 */
noise_model_t* noise_model_copy(const noise_model_t* model);

/**
 * @brief Create realistic noise model from hardware specs
 */
noise_model_t* noise_model_create_realistic(double t1_us, double t2_us,
                                            double gate_error,
                                            double readout_error);

// ============================================================================
// NOISE CHANNELS
// ============================================================================

/**
 * @brief Apply depolarizing channel to single qubit
 */
void noise_depolarizing_single(quantum_state_t* state, int qubit,
                               double probability, double random_value);

/**
 * @brief Apply depolarizing channel to two qubits
 */
void noise_depolarizing_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                  double probability, double random_value);

/**
 * @brief Apply amplitude damping channel
 */
void noise_amplitude_damping(quantum_state_t* state, int qubit,
                             double gamma, double random_value);

/**
 * @brief Apply phase damping channel
 */
void noise_phase_damping(quantum_state_t* state, int qubit,
                         double gamma, double random_value);

/**
 * @brief Apply pure dephasing
 */
void noise_pure_dephasing(quantum_state_t* state, int qubit,
                          double sigma, double random_phase);

/**
 * @brief Apply bit flip channel
 */
void noise_bit_flip(quantum_state_t* state, int qubit,
                    double probability, double random_value);

/**
 * @brief Apply phase flip channel
 */
void noise_phase_flip(quantum_state_t* state, int qubit,
                      double probability, double random_value);

/**
 * @brief Apply bit-phase flip channel
 */
void noise_bit_phase_flip(quantum_state_t* state, int qubit,
                          double probability, double random_value);

/**
 * @brief Apply thermal relaxation
 */
void noise_thermal_relaxation(quantum_state_t* state, int qubit,
                              double t1, double t2, double time,
                              const double* random_values);

/**
 * @brief Simulate readout error
 */
int noise_readout_error(int outcome, double error_0_to_1, double error_1_to_0,
                        double random_value);

// ============================================================================
// APPLY NOISE MODEL
// ============================================================================

/**
 * @brief Apply noise model to qubit
 */
void noise_apply_model(quantum_state_t* state, int qubit,
                       const noise_model_t* model,
                       const double* random_values);

/**
 * @brief Apply noise model to two-qubit gate
 */
void noise_apply_model_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                 const noise_model_t* model,
                                 const double* random_values);

// ============================================================================
// CONFIGURATION
// ============================================================================

void noise_model_set_depolarizing(noise_model_t* model, double rate);
void noise_model_set_amplitude_damping(noise_model_t* model, double rate);
void noise_model_set_phase_damping(noise_model_t* model, double rate);
void noise_model_set_thermal(noise_model_t* model, double t1, double t2);
void noise_model_set_gate_time(noise_model_t* model, double time);
void noise_model_set_readout_error(noise_model_t* model,
                                   double error_0, double error_1);
void noise_model_set_enabled(noise_model_t* model, int enabled);

// ============================================================================
// KRAUS-CHANNEL VALIDATION
// ============================================================================

/**
 * @brief Identifier for the single-qubit channel to validate.
 */
typedef enum {
    NOISE_CHANNEL_DEPOLARIZING,    /**< One parameter: probability p. */
    NOISE_CHANNEL_AMPLITUDE_DAMPING,/**< One parameter: gamma. */
    NOISE_CHANNEL_PHASE_DAMPING,   /**< One parameter: lambda. */
    NOISE_CHANNEL_BIT_FLIP,        /**< One parameter: probability p. */
    NOISE_CHANNEL_PHASE_FLIP,      /**< One parameter: probability p. */
    NOISE_CHANNEL_BIT_PHASE_FLIP,  /**< One parameter: probability p. */
} noise_channel_id_t;

/**
 * @brief Check the Kraus completeness relation Σ_i K_i† K_i = I for a
 *        single-qubit channel at parameter @p p. Returns the max
 *        absolute deviation from the identity. A valid CPTP channel
 *        returns 0 to within floating-point tolerance (~1e-15).
 *
 * @param channel Channel identifier.
 * @param p Channel parameter (gamma, lambda, or probability).
 * @return Max |Σ K†K - I| element-wise; negative on invalid input.
 */
double noise_kraus_completeness_deviation(noise_channel_id_t channel, double p);

// ============================================================================
// COMPOSITE / CORRELATED CHANNELS  (v0.2, Plan 2D)
// ============================================================================

/**
 * @brief Correlated two-qubit Pauli channel.
 *
 * Applies one of the 16 two-qubit Pauli operators (I, X, Y, Z)^{⊗2}
 * with supplied probabilities.  @p probs is a length-16 array
 * indexed by (p_a * 4 + p_b) with p_a, p_b ∈ {0=I, 1=X, 2=Y, 3=Z};
 * it must sum to 1 within 1e-9 (otherwise the function returns
 * without mutating the state).
 *
 * This is the right model for correlated gate errors on a 2q gate --
 * e.g. a bias toward XX errors on a CNOT, which a product of two
 * depolarizing channels cannot reproduce.
 *
 * @param state         mutable state.
 * @param qubit_a       first qubit.
 * @param qubit_b       second qubit (must differ from qubit_a).
 * @param probs         length-16 probability table, sum to 1.
 * @param uniform       a uniform [0, 1) sample.
 */
void noise_correlated_two_qubit_pauli(quantum_state_t* state,
                                       int qubit_a, int qubit_b,
                                       const double *probs,
                                       double uniform);

/**
 * @brief Convex mixture of two single-qubit channels.
 *
 * With probability @p mixture_prob, apply channel A (selected by
 * @p channel_a at parameter @p param_a).  Otherwise apply channel B.
 * @p uniform_pick decides which branch; @p random_channel is passed
 * into the selected channel's own sampling input.
 *
 * Useful for time-varying noise (pick branch per gate) or for
 * biased-basis error models.
 */
void noise_convex_mixture_single(quantum_state_t* state, int qubit,
                                  noise_channel_id_t channel_a, double param_a,
                                  noise_channel_id_t channel_b, double param_b,
                                  double mixture_prob,
                                  double uniform_pick,
                                  double random_channel);

/**
 * @brief Sequential composition of two single-qubit channels on the
 *        same qubit.  Equivalent to applying them in succession.
 */
void noise_composite_sequential_single(quantum_state_t* state, int qubit,
                                        noise_channel_id_t channel_a, double param_a,
                                        noise_channel_id_t channel_b, double param_b,
                                        double random_a,
                                        double random_b);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_NOISE_H */
