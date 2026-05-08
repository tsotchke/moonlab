/**
 * @file noise_mpdo.h
 * @brief Matrix-product density operator (MPDO) noise simulator.
 *
 * The MPDO representation stores a noisy n-qubit density matrix
 * @f$\rho \in \mathcal{B}(\mathcal H_2^{\otimes n})@f$ as an MPS-of-
 * superoperators.  Each site tensor carries a 4-dimensional physical
 * leg corresponding to the vectorisation of the local 2x2 density-
 * matrix block (Liouville representation):
 * @f[
 *   \mathrm{vec}(\rho_i) \;=\;
 *     (\rho_{00},\, \rho_{01},\, \rho_{10},\, \rho_{11})^{\!\top},
 * @f]
 * and a left/right virtual bond of dimension @f$\chi@f$ that
 * captures correlations across sites.  A single-qubit Kraus channel
 * acts as a 4x4 superoperator applied to the local physical leg of
 * site @f$i@f$ in O(chi^2) time; coherent gates act identically to
 * the pure-state MPS but with the doubled physical dimension.
 *
 * For local noise the bond dimension chi grows polynomially in
 * circuit depth; experimental setups with quasi-1D layouts at
 * single-qubit error rates of 1e-3 are tractable up to ~100 qubits
 * (the v0.3 release scope per the moonlab arc plan §2A).
 *
 * STATUS: scaffolding for the v0.3 release.  The first cut implements:
 *   - Lifecycle (create / free / clone)
 *   - Direct product-state init from a pure n-qubit |0> state
 *   - Single-qubit Kraus-channel application by 4x4 superoperator
 *   - Trace and Hermiticity check
 *   - Single-site Pauli expectation
 *
 * Two-site channels, MPDO contraction with arbitrary observables,
 * SVD-based bond truncation, and integration with the existing
 * noise.h channel definitions (depolarizing, amplitude_damping,
 * etc.) land in subsequent v0.3 commits.
 *
 * REFERENCES
 * ----------
 *  - F. Verstraete, J.J. Garcia-Ripoll, J.I. Cirac,
 *    "Matrix product density operators: Simulation of finite-T and
 *     dissipative systems", Phys. Rev. Lett. 93, 207204 (2004).
 *    arXiv:cond-mat/0406426.  Original MPDO construction.
 *  - M. Zwolak, G. Vidal, "Mixed-state dynamics in 1D quantum
 *    lattice systems: a TEBD approach", Phys. Rev. Lett. 93, 207205 (2004).
 *  - For the v0.3 application target see the matrix-product noise
 *    simulator overview in the moonlab arc plan §2A.
 *
 * @since v0.3.0
 * @stability evolving
 */

#ifndef MOONLAB_NOISE_MPDO_H
#define MOONLAB_NOISE_MPDO_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

#include "../applications/moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef double _Complex mpdo_complex_t;

/** Forward-declared opaque handle. */
typedef struct moonlab_mpdo_t moonlab_mpdo_t;

/** Error codes for MPDO operations. */
typedef enum {
    MPDO_SUCCESS    = 0,
    MPDO_ERR_INVALID = -1,
    MPDO_ERR_QUBIT   = -2,
    MPDO_ERR_OOM     = -3,
    MPDO_ERR_BACKEND = -4,
} mpdo_error_t;

/* ================================================================== */
/*  Lifecycle                                                         */
/* ================================================================== */

/**
 * @brief Create an n-qubit MPDO in the pure |0...0><0...0| product state.
 *
 * Bond dimensions start at chi = 1 (product state); they grow as
 * channels and gates are applied, capped at @p max_bond_dim.
 *
 * @param num_qubits     Number of qubits, 1..1000.
 * @param max_bond_dim   MPS truncation cap.  32 is reasonable for
 *                       ~50-qubit local-noise circuits at 1e-3 error.
 * @return New handle or NULL on bad args / alloc failure.
 */
MOONLAB_API moonlab_mpdo_t* moonlab_mpdo_create(uint32_t num_qubits,
                                                  uint32_t max_bond_dim);

/** Release all memory.  No-op on NULL. */
MOONLAB_API void moonlab_mpdo_free(moonlab_mpdo_t* m);

/** Deep copy of the MPDO. */
MOONLAB_API moonlab_mpdo_t* moonlab_mpdo_clone(const moonlab_mpdo_t* m);

/* ================================================================== */
/*  Introspection                                                     */
/* ================================================================== */

MOONLAB_API uint32_t moonlab_mpdo_num_qubits(const moonlab_mpdo_t* m);
MOONLAB_API uint32_t moonlab_mpdo_max_bond_dim(const moonlab_mpdo_t* m);
MOONLAB_API uint32_t moonlab_mpdo_current_bond_dim(const moonlab_mpdo_t* m);

/**
 * @brief Trace of the represented density matrix Tr(rho).
 *
 * For a properly normalised quantum state Tr(rho) = 1; deviations
 * indicate truncation error (small) or implementation bugs (large).
 */
MOONLAB_API double moonlab_mpdo_trace(const moonlab_mpdo_t* m);

/* ================================================================== */
/*  Channel application                                               */
/* ================================================================== */

/**
 * @brief Apply a single-qubit Kraus channel given by a list of 2x2 Kraus operators.
 *
 * The channel is @f$\rho \mapsto \sum_a K_a \rho K_a^\dagger@f$.
 * The MPDO implementation translates this to a 4x4 superoperator
 * acting on the vectorised physical index of qubit @p qubit:
 * @f$S_{\alpha\beta,\gamma\delta} = \sum_a (K_a)_{\alpha\gamma}
 * \overline{(K_a)_{\beta\delta}}@f$ (row-major
 * @f$\alpha = 2 r_{\rm out} + r'_{\rm out}@f$,
 * @f$\beta  = 2 c_{\rm in} + c'_{\rm in}@f$).
 *
 * @param[in,out] state  MPDO handle.  Mutated in place.
 * @param[in]     qubit  Target qubit index.
 * @param[in]     kraus  Flat array of @p num_kraus 2x2 row-major
 *                       complex Kraus operators
 *                       (kraus[k * 4 + r * 2 + c]).
 * @param[in]     num_kraus Number of Kraus operators.
 *
 * Channel completeness @f$\sum_a K_a^\dagger K_a = \mathbb{1}@f$ is
 * NOT verified by this routine; the caller is responsible.  The
 * existing noise_kraus_completeness_deviation helper in noise.h
 * provides a numerical check.
 *
 * @return MPDO_SUCCESS or an error code.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_kraus_1q(moonlab_mpdo_t* state,
                                                      uint32_t qubit,
                                                      const mpdo_complex_t* kraus,
                                                      uint32_t num_kraus);

/* ---- Named single-qubit channels ----------------------------------
 *
 * Convenience wrappers around the standard textbook channels.  Each
 * builds the appropriate Kraus operators internally and calls
 * moonlab_mpdo_apply_kraus_1q.  Channel parameter conventions match
 * the existing src/quantum/noise.h definitions so the noise model
 * stays consistent across MPDO and pure-state simulation paths.
 */

/**
 * @brief Single-qubit depolarising channel
 *        @f$\rho \mapsto (1-p)\rho + (p/3)(X\rho X + Y\rho Y + Z\rho Z)@f$.
 *
 * @param p  Depolarising probability in [0, 1].  At p=0 the channel
 *           is identity; at p=3/4 it sends every state to the maximally
 *           mixed state I/2.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_depolarizing_1q(moonlab_mpdo_t* state,
                                                            uint32_t qubit,
                                                            double p);

/**
 * @brief Single-qubit amplitude damping (T1 relaxation)
 *        @f$\rho \mapsto K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger@f$
 *        with @f$K_0 = \mathrm{diag}(1, \sqrt{1-\gamma})@f$,
 *        @f$K_1 = \sqrt{\gamma}\,|0\rangle\langle 1|@f$.
 *
 * @param gamma  Damping parameter in [0, 1].  At gamma=0 the channel
 *               is identity; at gamma=1 it deterministically resets
 *               every state to |0><0|.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_amplitude_damping_1q(moonlab_mpdo_t* state,
                                                                  uint32_t qubit,
                                                                  double gamma);

/**
 * @brief Single-qubit phase damping (pure dephasing, T2 only)
 *        @f$K_0 = \mathrm{diag}(1, \sqrt{1-\lambda})@f$,
 *        @f$K_1 = \sqrt{\lambda}\,|1\rangle\langle 1|@f$.
 *
 * @param lambda  Dephasing parameter in [0, 1].  Reduces off-diagonal
 *                density-matrix elements by factor sqrt(1-lambda).
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_phase_damping_1q(moonlab_mpdo_t* state,
                                                              uint32_t qubit,
                                                              double lambda);

/**
 * @brief Bit-flip channel
 *        @f$\rho \mapsto (1-p)\rho + p\,X\rho X@f$.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_bit_flip_1q(moonlab_mpdo_t* state,
                                                        uint32_t qubit,
                                                        double p);

/**
 * @brief Phase-flip channel
 *        @f$\rho \mapsto (1-p)\rho + p\,Z\rho Z@f$.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_phase_flip_1q(moonlab_mpdo_t* state,
                                                          uint32_t qubit,
                                                          double p);

/**
 * @brief Bit-phase-flip channel
 *        @f$\rho \mapsto (1-p)\rho + p\,Y\rho Y@f$.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_apply_bit_phase_flip_1q(moonlab_mpdo_t* state,
                                                               uint32_t qubit,
                                                               double p);

/* ================================================================== */
/*  Observables                                                       */
/* ================================================================== */

/**
 * @brief Single-site Pauli expectation value Tr(rho * P_q) where P
 *        is one of {I, X, Y, Z} on qubit @p qubit and identity
 *        elsewhere.
 *
 * @param[in]  state       MPDO handle.
 * @param[in]  qubit       Target qubit.
 * @param[in]  pauli_code  0=I, 1=X, 2=Y, 3=Z.
 * @param[out] out_expval  Real expectation value (Pauli observables
 *                         are Hermitian so the imaginary part is
 *                         numerically zero up to roundoff).
 * @return MPDO_SUCCESS or an error code.
 */
MOONLAB_API mpdo_error_t moonlab_mpdo_expect_pauli_1q(const moonlab_mpdo_t* state,
                                                      uint32_t qubit,
                                                      uint8_t  pauli_code,
                                                      double*  out_expval);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_NOISE_MPDO_H */
