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
#include "../../applications/moonlab_api.h"

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
 * @stability evolving
 */
MOONLAB_API pauli_frame_t* pauli_frame_create(size_t num_qubits);

/**
 * @brief Release all memory.  Safe on NULL.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_free(pauli_frame_t* f);

/**
 * @brief Reset to identity (all bits zero).
 * @stability evolving
 */
MOONLAB_API void pauli_frame_clear(pauli_frame_t* f);

/** @brief Number of qubits this frame was allocated for. */
size_t pauli_frame_num_qubits(const pauli_frame_t* f);

/** @brief Read the (x_q, z_q) bits at qubit @p q.  Out parameters
 * @stability evolving
 *  receive 0 or 1.  No-op on NULL inputs. */
MOONLAB_API void pauli_frame_read(const pauli_frame_t* f, size_t q,
                       uint8_t* out_x, uint8_t* out_z);

/* ================================================================== */
/*  Single-qubit Clifford propagation                                  */
/* ================================================================== */

/**
 * @brief Propagate a Hadamard on qubit @p q through the frame.
 *
 * H conjugates @f$X \leftrightarrow Z@f$, so the frame bits are
 * exchanged: @f$(x_q, z_q) \mapsto (z_q, x_q)@f$.
 *
 * @param f Frame, mutated in place.
 * @param q Qubit index; out-of-range indices and a NULL @p f are
 *          silently ignored.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_h(pauli_frame_t* f, size_t q);

/**
 * @brief Propagate a phase gate S on qubit @p q through the frame.
 *
 * @f$S X S^\dagger = Y@f$ and @f$S Z S^\dagger = Z@f$, so an
 * X-component picks up a Z-component: @f$z_q \mathrel{\hat=} x_q@f$.
 * The accompanying factor of @f$i@f$ is phase-only and is not tracked.
 *
 * @param f Frame, mutated in place.
 * @param q Qubit index; out-of-range indices and a NULL @p f are
 *          silently ignored.
 * @stability evolving
 */
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

/**
 * @brief Propagate a CNOT through the frame.
 *
 * An X-error on the control copies to the target and a Z-error on the
 * target copies back to the control:
 * @f$x_t \mathrel{\hat=} x_c@f$, @f$z_c \mathrel{\hat=} z_t@f$.
 *
 * @param f       Frame, mutated in place.
 * @param control Control qubit index.
 * @param target  Target qubit index, distinct from @p control.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_cnot(pauli_frame_t* f, size_t control, size_t target);

/**
 * @brief Propagate a CZ through the frame.
 *
 * CZ is symmetric and maps each X-component to a Z-component on the
 * partner: @f$z_b \mathrel{\hat=} x_a@f$ and
 * @f$z_a \mathrel{\hat=} x_b@f$ (both read before either is written).
 *
 * @param f Frame, mutated in place.
 * @param a First qubit index.
 * @param b Second qubit index, distinct from @p a.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_cz(pauli_frame_t* f, size_t a, size_t b);

/* SWAP(a, b): exchanges (x_a, z_a) and (x_b, z_b). */
void pauli_frame_swap(pauli_frame_t* f, size_t a, size_t b);

/* ================================================================== */
/*  Pauli error injection                                              */
/* ================================================================== */

/**
 * @brief Inject an X error on qubit @p q.
 *
 * Toggles the frame's X-component, @f$x_q \mathrel{\hat=} 1@f$.
 * Injecting twice cancels.
 *
 * @param f Frame, mutated in place.
 * @param q Qubit index; out-of-range indices and a NULL @p f are
 *          silently ignored.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_inject_x(pauli_frame_t* f, size_t q);

/**
 * @brief Inject a Z error on qubit @p q.
 *
 * Toggles the frame's Z-component, @f$z_q \mathrel{\hat=} 1@f$.
 * Injecting twice cancels.
 *
 * @param f Frame, mutated in place.
 * @param q Qubit index; out-of-range indices and a NULL @p f are
 *          silently ignored.
 * @stability evolving
 */
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

/**
 * @brief Allocate a shot-bit-packed batch of identity frames.
 *
 * Stores one X row and one Z row per qubit, each
 * @f$\lceil \mathrm{num\_shots}/64 \rceil@f$ 64-bit words wide, with
 * shot @f$s@f$ living in bit @f$s \bmod 64@f$ of word
 * @f$\lfloor s/64 \rfloor@f$.  All bits start zero (identity frame on
 * every shot).
 *
 * @param num_qubits Qubits per frame; must be nonzero.
 * @param num_shots  Independent shots in the batch; must be nonzero.
 * @return Owned batch handle, or NULL if either count is zero or an
 *         allocation fails.
 * @stability evolving
 */
MOONLAB_API pauli_frame_batch_t* pauli_frame_batch_create(size_t num_qubits, size_t num_shots);

/**
 * @brief Release a batch created by ::pauli_frame_batch_create.
 *
 * Frees the X and Z bit planes and the handle.
 *
 * @param b Batch handle to release; NULL is a no-op.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_free(pauli_frame_batch_t* b);
size_t pauli_frame_batch_num_shots(const pauli_frame_batch_t* b);
size_t pauli_frame_batch_num_qubits(const pauli_frame_batch_t* b);

/* Apply a Clifford gate to every frame in the batch.  All frames in
 * the batch see the same gate -- per-shot variation comes from noise
 * injection. */
void pauli_frame_batch_h(pauli_frame_batch_t* b, size_t q);
void pauli_frame_batch_s(pauli_frame_batch_t* b, size_t q);
/**
 * @brief Propagate a CNOT through every frame in the batch.
 *
 * Two word-parallel row XORs over the packed shot planes,
 * @f$x_t \mathrel{\hat=} x_c@f$ then @f$z_c \mathrel{\hat=} z_t@f$,
 * widened to the host SIMD lane count (see
 * ::pauli_frame_simd_backend).
 *
 * @param b Batch, mutated in place.
 * @param c Control qubit index.
 * @param t Target qubit index, distinct from @p c.  A NULL batch, an
 *          out-of-range index, or @p c == @p t is silently ignored.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_cnot(pauli_frame_batch_t* b, size_t c, size_t t);
void pauli_frame_batch_cz(pauli_frame_batch_t* b, size_t a, size_t b_q);
void pauli_frame_batch_swap(pauli_frame_batch_t* b, size_t a, size_t b_q);

/**
 * @brief Inject an i.i.d. single-qubit depolarising error per shot.
 *
 * Each shot draws one splitmix64 word and independently selects from
 * @f$\{I\,(1-p),\; X\,(p/3),\; Y\,(p/3),\; Z\,(p/3)\}@f$.  The
 * selections for the 64 shots in a word are accumulated into an X mask
 * and a Z mask and XORed into the frame in one write per word, so the
 * inner loop stays branch-light and single-threaded (OpenMP was
 * measured net-negative at typical 10^4-10^5 shot counts).
 *
 * @param b         Batch, mutated in place.
 * @param q         Qubit index.
 * @param p         Total error probability; @f$p \le 0@f$ returns
 *                  immediately without consuming RNG.
 * @param rng_state splitmix64 state, advanced once per shot.  Must be
 *                  non-NULL.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_depolarising(pauli_frame_batch_t* b, size_t q,
                                     double p, uint64_t* rng_state);

/**
 * @brief Inject an i.i.d. X-only bit-flip channel per shot.
 *
 * Flips @f$x_q@f$ with probability @p p independently in each shot,
 * accumulating a per-word mask exactly as the depolarising kernel does.
 * @f$p \ge 1@f$ is special-cased to an unconditional flip of every
 * shot: the naive threshold @f$(\mathrm{uint64\_t})(2^{64} p)@f$ is an
 * out-of-range conversion at @f$p = 1@f$ and yields 0 on current
 * toolchains, which would flip nothing.  The special case still draws
 * the same number of RNG words so interleaved callers stay
 * deterministic.
 *
 * @param b         Batch, mutated in place.
 * @param q         Qubit index.
 * @param p         Flip probability; @f$p \le 0@f$ returns immediately
 *                  without consuming RNG.
 * @param rng_state splitmix64 state, advanced once per shot.  Must be
 *                  non-NULL.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_bit_flip(pauli_frame_batch_t* b, size_t q,
                                 double p, uint64_t* rng_state);

/**
 * @brief Read every shot's Z-basis outcome contribution at qubit @p q.
 *
 * Unpacks the X bit plane of qubit @p q into one byte per shot.  Each
 * byte is the parity by which that shot's outcome differs from the
 * noiseless trajectory; the caller XORs it with the deterministic
 * outcome.  Non-destructive: the frame is left untouched.
 *
 * @param b   Batch to read.
 * @param q   Qubit index.
 * @param out Receives @c num_shots bytes, each 0 or 1, in shot order.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_measure_z(const pauli_frame_batch_t* b, size_t q,
                                  uint8_t* out);

/**
 * @brief Destructive noisy Z-basis measurement across the batch.
 *
 * Reads @f$x_q@f$ per shot, flips the reported bit with probability
 * @p p_flip, then zeroes @f$(x_q, z_q)@f$ for every shot -- modelling a
 * destructive readout followed by a re-prepared @f$|0\rangle@f$
 * ancilla.  This is the "M_meas" primitive of a noisy surface-code
 * stabilizer round.  The frame is cleared whether or not readout noise
 * is active.
 *
 * @param b         Batch, mutated in place (ancilla frame cleared).
 * @param q         Qubit index.
 * @param p_flip    Readout flip probability; @f$\le 0@f$ (or a NULL
 *                  @p rng_state) gives a noiseless read that still
 *                  clears the frame.
 * @param rng_state splitmix64 state, advanced once per shot when
 *                  readout noise is active.
 * @param out       Receives @c num_shots bytes, each 0 or 1.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_measure_z_noisy(pauli_frame_batch_t* b, size_t q,
                                        double p_flip, uint64_t* rng_state,
                                        uint8_t* out);

/* Destructive reset on qubit q: clears the (x_q, z_q) bits across all
 * shots without recording a measurement.  Used between rounds when an
 * ancilla is reused without referencing its outcome. */
void pauli_frame_batch_reset_zero(pauli_frame_batch_t* b, size_t q);

/**
 * @brief Reset every shot's frame to identity.
 *
 * Zeroes both bit planes in full, so the batch can be reused for
 * another circuit without reallocating.  Qubit and shot counts are
 * unchanged.
 *
 * @param b Batch, mutated in place; NULL is a no-op.
 * @stability evolving
 */
MOONLAB_API void pauli_frame_batch_clear(pauli_frame_batch_t* b);

/* ================================================================== */
/*  Circuit-level batch shot sampler                                   */
/* ================================================================== */

/**
 * @brief Op kinds accepted by the batch circuit sampler.
 *
 * X, Y, Z are frame no-ops (their deterministic sign contribution is
 * folded into the reference sample); RESET prepares |0> with a fresh
 * random Z-frame per shot; MEASURE performs a destructive-free Z-basis
 * read whose result is (reference_bit XOR frame_x).
 */
typedef enum {
    PF_OP_H = 0, PF_OP_S, PF_OP_S_DAG,
    PF_OP_X, PF_OP_Y, PF_OP_Z,
    PF_OP_CNOT, PF_OP_CZ, PF_OP_SWAP,
    PF_OP_RESET, PF_OP_MEASURE,
    /* Noise channels.  These carry a probability in pf_circuit_op_t::p and
     * are per-shot: they act only on the frame, never on the reference
     * trajectory, so the reference pass skips them.  Names and semantics
     * match the corresponding stim instructions. */
    PF_OP_X_ERROR,       /* X with probability p (stim X_ERROR)          */
    PF_OP_Z_ERROR,       /* Z with probability p (stim Z_ERROR)          */
    PF_OP_Y_ERROR,       /* Y with probability p (stim Y_ERROR)          */
    PF_OP_DEPOLARIZE1,   /* uniform X/Y/Z each p/3 (stim DEPOLARIZE1)    */
    PF_OP_DEPOLARIZE2,   /* uniform over 15 two-qubit Paulis, p/15 each  */
    PF_OP_MEASURE_NOISY, /* measure q0, reported outcome flipped w.p. p  */

    /* ---- Additive general Pauli channels -------------------------------
     * These need more than the single probability pf_circuit_op_t carries,
     * and pf_circuit_op_t is ABI-frozen.  So they store the BASE INDEX of
     * their probability block in a separate caller-supplied `chan_args`
     * table, encoded as a double in pf_circuit_op_t::p (exact for every
     * index below 2^53).  Only the _ex sampler entry points take that
     * table; the original entry points reject a circuit containing these
     * kinds rather than skipping them, because skipping a noise channel
     * silently changes the physics being sampled.                        */

    PF_OP_PAULI_CHANNEL_1 = 17, /**< q0 = qubit, p = base index.
                                 *   chan_args[base+0..2] = px, py, pz
                                 *   (stim PAULI_CHANNEL_1 order).  I is
                                 *   implicit with probability
                                 *   1 - (px + py + pz).                  */

    PF_OP_PAULI_CHANNEL_2 = 18  /**< q0, q1 = qubits, p = base index.
                                 *   chan_args[base+0..14] = the 15 stim
                                 *   PAULI_CHANNEL_2 probabilities in
                                 *   stim's argument order
                                 *   (`stim.gate_data('PAULI_CHANNEL_2')`):
                                 *
                                 *     0:IX  1:IY  2:IZ  3:XI  4:XX
                                 *     5:XY  6:XZ  7:YI  8:YX  9:YY
                                 *    10:YZ 11:ZI 12:ZX 13:ZY 14:ZZ
                                 *
                                 *   The FIRST letter acts on q0 and the
                                 *   second on q1.  II is implicit with
                                 *   probability 1 - sum(args).           */
} pf_op_kind_t;

/** @brief One circuit instruction.  @p q1 is used only by two-qubit ops;
 *  @p p only by the noise channels and PF_OP_MEASURE_NOISY. */
typedef struct {
    uint8_t  kind;   /* pf_op_kind_t */
    uint32_t q0;
    uint32_t q1;
    double   p;
} pf_circuit_op_t;

/**
 * @brief Number of MEASURE ops in an op list (buffer sizing helper).
 * @stability evolving
 */
MOONLAB_API size_t pauli_frame_circuit_num_measurements(
    const pf_circuit_op_t* ops, size_t num_ops);

/**
 * @brief Batch-sample a Clifford + measurement circuit over @p num_shots
 *        independent shots, reproducing Stim's compile_sampler().sample()
 *        output distribution.
 *
 * A single reference tableau pass fixes, per measurement, the deterministic
 * bit and whether the outcome is random; the shot-bit-packed Pauli frame
 * then supplies each shot's flip (with fresh Z-frame randomness injected at
 * resets and random measurements).  The frame word ops are SIMD-widened to
 * the host ISA and the shot batch is split into @p num_threads OpenMP
 * blocks, each with an independent RNG stream.
 *
 * Output layout is measurement-major: out[m * num_shots + shot] is the
 * m-th measurement's outcome (0/1 byte) for the given shot.  The buffer
 * must hold (#MEASURE ops) * num_shots bytes.
 *
 * @param num_qubits  qubit count
 * @param ops,num_ops circuit instruction list
 * @param num_shots   number of independent shots
 * @param seed        base RNG seed (0 -> internal default)
 * @param num_threads OpenMP block count; <=0 selects omp_get_max_threads()
 * @param out         measurement-major output buffer
 * @return number of measurements written, or -1 on error.
 * @stability evolving
 */
MOONLAB_API long pauli_frame_batch_sample_circuit(
    size_t num_qubits,
    const pf_circuit_op_t* ops, size_t num_ops,
    size_t num_shots, uint64_t seed,
    int num_threads, uint8_t* out);

/**
 * @brief Batch-sample DETECTORS of a Clifford + measurement circuit.
 *
 * A detector is the parity of a set of measurement records, reported as the
 * deviation from the noiseless trajectory (so it reads 0 on a noiseless
 * run).  This is what a decoder consumes, and it mirrors stim's
 * compile_detector_sampler().sample().
 *
 * Detectors are supplied in CSR form: detector d covers the measurement
 * indices det_indices[det_offsets[d] .. det_offsets[d+1]).  @p det_offsets
 * therefore holds @p num_detectors + 1 entries.  Measurement indices count
 * MEASURE and MEASURE_NOISY ops in circuit order from 0.
 *
 * The measurement record is reduced to detectors inside each shot block, so
 * the full num_measurements x num_shots record is never materialised.
 *
 * @param out receives num_detectors * num_shots bytes, detector-major
 *            (detector d, shot s at out[d * num_shots + s]).
 * @return num_detectors on success, negative on error.
 * @stability evolving
 */
MOONLAB_API long pauli_frame_batch_sample_detectors(
    size_t num_qubits, const pf_circuit_op_t* ops, size_t num_ops,
    const size_t* det_offsets, const uint32_t* det_indices,
    size_t num_detectors, size_t num_shots, uint64_t seed,
    int num_threads, uint8_t* out);

/* ================================================================== */
/*  Channel-table sampler entry points                                 */
/* ================================================================== */

/**
 * @brief pauli_frame_batch_sample_circuit() plus a general-Pauli-channel
 *        argument table.
 *
 * PF_OP_PAULI_CHANNEL_1 / PF_OP_PAULI_CHANNEL_2 ops read their
 * probabilities from @p chan_args at the base index stored in
 * pf_circuit_op_t::p.  Every other op kind behaves exactly as in
 * pauli_frame_batch_sample_circuit(), which is this function with
 * `chan_args = NULL, num_chan_args = 0`.
 *
 * @return number of measurements written, or -1 on error.  A channel op
 *         whose probability block does not fit inside @p chan_args (in
 *         particular a NULL table) is an error, never a skipped op.
 * @stability beta
 */
MOONLAB_API long pauli_frame_batch_sample_circuit_ex(
    size_t num_qubits, const pf_circuit_op_t* ops, size_t num_ops,
    const double* chan_args, size_t num_chan_args,
    size_t num_shots, uint64_t seed, int num_threads, uint8_t* out);

/**
 * @brief pauli_frame_batch_sample_detectors() plus a general-Pauli-channel
 *        argument table.  @see pauli_frame_batch_sample_circuit_ex.
 *
 * @return num_detectors on success, negative on error.
 * @stability beta
 */
MOONLAB_API long pauli_frame_batch_sample_detectors_ex(
    size_t num_qubits, const pf_circuit_op_t* ops, size_t num_ops,
    const double* chan_args, size_t num_chan_args,
    const size_t* det_offsets, const uint32_t* det_indices,
    size_t num_detectors, size_t num_shots, uint64_t seed,
    int num_threads, uint8_t* out);

/**
 * @brief Name of the SIMD kernel this translation unit was compiled to.
 *
 * Reports which of the batched frame kernels the compiler selected at
 * build time from the host ISA macros: "neon" on AArch64/NEON,
 * "avx512" when __AVX512F__ is defined, "avx2" when __AVX2__ is, and
 * "scalar" otherwise.  Build with -DQSIM_NATIVE_ARCH=ON to get the
 * widest kernel the host supports.
 *
 * @return Static string, never NULL; one of "neon", "avx512", "avx2",
 *         "scalar".
 * @stability evolving
 */
MOONLAB_API const char* pauli_frame_simd_backend(void);
/**
 * @brief SIMD lane width in 64-bit words (1 = scalar fallback).
 * @stability evolving
 */
MOONLAB_API int pauli_frame_simd_lanes(void);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_PAULI_FRAME_H */
