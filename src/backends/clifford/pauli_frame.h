/**
 * @file pauli_frame.h
 * @brief Pauli-frame sampler for Clifford-circuit + Pauli-noise simulation.
 *
 * Stim-style frame tracking: rather than updating an O(n)-cost tableau
 * per Clifford gate per shot, each shot tracks a 2n-bit Pauli frame
 * @f$F = X^{x_0} Z^{z_0} \otimes \cdots \otimes X^{x_{n-1}} Z^{z_{n-1}}@f$
 * representing the deviation of this shot's state from the ideal
 * (deterministic) trajectory.  Clifford gates propagate the frame by
 * the standard commutation rules (H swaps x and z; S takes z ^= x;
 * CNOT propagates x from control to target and z from target to
 * control).  Pauli errors are injected by flipping frame bits with
 * the per-channel probability.  Z-basis measurements read the frame's
 * x-bit at the measured qubit (under the convention that the frame's
 * X-component flips the ideal measurement outcome).
 *
 * The phase of F is not tracked: measurement outcomes on Pauli
 * strings only depend on commutation pattern, not on the ±1 / ±i
 * factor of the frame.  This is the core simplification that makes
 * Pauli frames O(1) per gate vs the tableau's O(n).
 *
 * For paper §3.6 / §4.4 surface-code threshold sweeps, the bench
 * harness applies a depth-d circuit (with i.i.d. Pauli noise) to a
 * batch of N independent frames and reports shots / second.  At
 * d ~ 50 and N = 10^4 shots Stim's published baseline is on the
 * order of 10^9 operations / second; the Moonlab harness targets
 * the same regime.
 *
 * @since v0.3.0
 */
#ifndef MOONLAB_PAULI_FRAME_H
#define MOONLAB_PAULI_FRAME_H
#include "applications/moonlab_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pauli_frame_t pauli_frame_t;

/* ================================================================== */
/*  Lifecycle                                                          */
/* ================================================================== */

/**
 * @brief Allocate a frame of n qubits, initialised to identity (all
 *        x and z bits zero).
 */
MOONLAB_API pauli_frame_t* pauli_frame_create(size_t num_qubits);

/** @brief Release all memory.  Safe on NULL. */
MOONLAB_API void pauli_frame_free(pauli_frame_t* f);

/** @brief Reset to identity (all bits zero). */
MOONLAB_API void pauli_frame_clear(pauli_frame_t* f);

/** @brief Number of qubits this frame was allocated for. */
size_t pauli_frame_num_qubits(const pauli_frame_t* f);

/** @brief Read the (x_q, z_q) bits at qubit @p q.  Out parameters
 *  receive 0 or 1.  No-op on NULL inputs. */
MOONLAB_API void pauli_frame_read(const pauli_frame_t* f, size_t q,
                       uint8_t* out_x, uint8_t* out_z);

/* ================================================================== */
/*  Single-qubit Clifford propagation                                  */
/* ================================================================== */

/* H: X <-> Z (swap x_q and z_q). */
MOONLAB_API void pauli_frame_h(pauli_frame_t* f, size_t q);
/* S: Z stays, X picks up Z (z_q ^= x_q). */
MOONLAB_API void pauli_frame_s(pauli_frame_t* f, size_t q);
/* S^dagger: same bit transform as S (the difference is phase, untracked). */
void pauli_frame_s_dag(pauli_frame_t* f, size_t q);

/* X, Y, Z gates on the *circuit*: they commute or anti-commute with
 * frame bits but don't change the bit pattern (only the global phase,
 * which we don't track).  Provided as no-ops for API parity. */
void pauli_frame_x(pauli_frame_t* f, size_t q);
void pauli_frame_y(pauli_frame_t* f, size_t q);
void pauli_frame_z(pauli_frame_t* f, size_t q);

/* ================================================================== */
/*  Two-qubit Clifford propagation                                     */
/* ================================================================== */

/* CNOT(c, t): propagates X-error on control to target (x_t ^= x_c) and
 *             Z-error on target to control (z_c ^= z_t). */
MOONLAB_API void pauli_frame_cnot(pauli_frame_t* f, size_t control, size_t target);

/* CZ(a, b): propagates X-error on a to Z-error on b (z_b ^= x_a) and
 *           X-error on b to Z-error on a (z_a ^= x_b). */
MOONLAB_API void pauli_frame_cz(pauli_frame_t* f, size_t a, size_t b);

/* SWAP(a, b): exchanges (x_a, z_a) and (x_b, z_b). */
void pauli_frame_swap(pauli_frame_t* f, size_t a, size_t b);

/* ================================================================== */
/*  Pauli error injection                                              */
/* ================================================================== */

/* Flip the X-component on qubit @p q (inject an X error). */
MOONLAB_API void pauli_frame_inject_x(pauli_frame_t* f, size_t q);
/* Flip the Z-component on qubit @p q. */
MOONLAB_API void pauli_frame_inject_z(pauli_frame_t* f, size_t q);
/* Inject a Y error = X * Z (up to phase) -> flip both x and z bits. */
void pauli_frame_inject_y(pauli_frame_t* f, size_t q);

/* ================================================================== */
/*  Measurements                                                       */
/* ================================================================== */

/**
 * @brief Z-basis measurement outcome contribution from the frame.
 *
 * Returns the frame's X-bit at qubit @p q, which is the parity by
 * which the actual measurement outcome differs from the ideal
 * (deterministic) trajectory's outcome.  The caller XORs this with
 * the deterministic outcome to get the noisy outcome.
 *
 * Reading the frame does not collapse it -- multiple measurements on
 * the same qubit return the same value until the frame is mutated.
 */
uint8_t pauli_frame_measure_z(const pauli_frame_t* f, size_t q);

/**
 * @brief X-basis measurement outcome contribution.  Returns z_q.
 */
uint8_t pauli_frame_measure_x(const pauli_frame_t* f, size_t q);

/* ================================================================== */
/*  Batched-shot helpers                                               */
/* ================================================================== */

/**
 * @brief Allocate an array of @p n_shots independent identity frames.
 *
 * Returns a pointer to a contiguous bit-packed array; treat the return
 * as opaque and use the helper functions below to advance / read.
 */
typedef struct pauli_frame_batch_t pauli_frame_batch_t;
MOONLAB_API pauli_frame_batch_t* pauli_frame_batch_create(size_t num_qubits, size_t num_shots);
MOONLAB_API void pauli_frame_batch_free(pauli_frame_batch_t* b);
size_t pauli_frame_batch_num_shots(const pauli_frame_batch_t* b);
size_t pauli_frame_batch_num_qubits(const pauli_frame_batch_t* b);

/* Apply a Clifford gate to every frame in the batch.  All frames in
 * the batch see the same gate -- per-shot variation comes from noise
 * injection. */
void pauli_frame_batch_h(pauli_frame_batch_t* b, size_t q);
void pauli_frame_batch_s(pauli_frame_batch_t* b, size_t q);
MOONLAB_API void pauli_frame_batch_cnot(pauli_frame_batch_t* b, size_t c, size_t t);
void pauli_frame_batch_cz(pauli_frame_batch_t* b, size_t a, size_t b_q);
void pauli_frame_batch_swap(pauli_frame_batch_t* b, size_t a, size_t b_q);

/* Inject an i.i.d. Pauli error per shot: each shot independently
 * draws from the depolarising channel { I (1-p), X (p/3), Y (p/3),
 * Z (p/3) } using the supplied splitmix64 RNG state. */
MOONLAB_API void pauli_frame_batch_depolarising(pauli_frame_batch_t* b, size_t q,
                                     double p, uint64_t* rng_state);

/* Inject an i.i.d. X-only bit-flip channel (X with prob p, I otherwise). */
MOONLAB_API void pauli_frame_batch_bit_flip(pauli_frame_batch_t* b, size_t q,
                                 double p, uint64_t* rng_state);

/* Read each shot's measure-Z outcome contribution at qubit q.  The
 * output buffer must hold @p num_shots bytes (0 or 1 each). */
MOONLAB_API void pauli_frame_batch_measure_z(const pauli_frame_batch_t* b, size_t q,
                                  uint8_t* out);

/* Noisy Z-basis measurement: reads x_q for each shot, flips the
 * recorded bit with probability @p p_flip, and resets the ancilla's
 * (x_q, z_q) frame components to zero (treating the measurement as a
 * destructive readout followed by a re-prepared |0> ancilla).  The
 * out buffer holds @p num_shots bytes.  This is the primitive needed
 * for noisy-syndrome QEC simulation -- the "M_meas" of the standard
 * surface-code stabilizer round. */
MOONLAB_API void pauli_frame_batch_measure_z_noisy(pauli_frame_batch_t* b, size_t q,
                                        double p_flip, uint64_t* rng_state,
                                        uint8_t* out);

/* Destructive reset on qubit q: clears the (x_q, z_q) bits across all
 * shots without recording a measurement.  Used between rounds when an
 * ancilla is reused without referencing its outcome. */
void pauli_frame_batch_reset_zero(pauli_frame_batch_t* b, size_t q);

/* Reset every shot's frame to identity. */
MOONLAB_API void pauli_frame_batch_clear(pauli_frame_batch_t* b);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_PAULI_FRAME_H */
