/**
 * @file stim_circuit.h
 * @brief Import and export of the Stim `.stim` circuit file format.
 *
 * Stim is the de-facto standard stabiliser simulator for quantum error
 * correction, and its circuit format is what the QEC community publishes
 * benchmarks in.  This module reads and writes that format directly, so a
 * moonlab build can be dropped into an existing Stim/Sinter/PyMatching
 * harness without a translation layer, and so moonlab's sampler can be
 * differentially tested against Stim's on the community's own circuits.
 *
 * The format reference is Stim's `doc/file_format_stim_circuit.md`:
 *
 * @verbatim
 * <CIRCUIT>     ::= <LINE>*
 * <LINE>        ::= <INDENT> (<INSTRUCTION> | <BLOCK_START> | <BLOCK_END>)?
 *                   <COMMENT>? '\n'
 * <INSTRUCTION> ::= <NAME> <TAG>? <PARENS_ARGUMENTS>? <TARGETS>
 * <NAME>        ::= /[a-zA-Z][a-zA-Z0-9_]*!/   (case insensitive)
 * <TAG>         ::= '[' /[^\r\]\n]/ * ']'
 * <TARG>        ::= '!'? <uint> | "rec[-" <uint> "]" | "sweep[" <uint> "]"
 *                 | '!'? [XYZ] <uint> | '*'
 * <BLOCK_START> ::= <INSTRUCTION> /[ \t]* / '{'
 * <BLOCK_END>   ::= '}'
 * @endverbatim
 *
 * SUPPORTED SURFACE.  Every Clifford gate Stim defines on one or two qubits
 * is accepted and lowered onto moonlab's primitive set
 * (H, S, S_DAG, X, Y, Z, CNOT, CZ, SWAP, RESET, MEASURE); the decompositions
 * are exhaustively checked against `stim.Tableau.from_named_gate` by
 * `bindings/python/tests/test_qec_stim_gates.py`.  Measurement and reset in
 * all three bases, the Pauli noise channels
 * (X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1, DEPOLARIZE2, PAULI_CHANNEL_1,
 * PAULI_CHANNEL_2), the QEC annotations
 * (DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS, SHIFT_COORDS, TICK, MPAD) and
 * REPEAT blocks are all handled.
 *
 * ANYTHING ELSE IS REJECTED, never skipped: an unsupported instruction, a
 * sweep-bit target, a combiner, a Pauli target where one is not allowed, or
 * an inverted target on a non-measurement all fail the parse with
 * MOONLAB_STIM_ERR_UNSUPPORTED, a 1-based line number, and a message naming
 * the offending token.  Silently dropping an instruction would silently
 * change the physics being benchmarked.
 */
#ifndef MOONLAB_STIM_CIRCUIT_H
#define MOONLAB_STIM_CIRCUIT_H

#include <stddef.h>
#include <stdint.h>

#include "../applications/moonlab_api.h"
#include "../backends/clifford/pauli_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes shared by the Stim circuit and DEM readers. */
typedef enum {
    MOONLAB_STIM_OK              = 0,
    MOONLAB_STIM_ERR_SYNTAX      = -701, /**< malformed token or structure    */
    MOONLAB_STIM_ERR_UNSUPPORTED = -702, /**< well formed but not implemented */
    MOONLAB_STIM_ERR_BAD_ARG     = -703, /**< bad argument from the caller    */
    MOONLAB_STIM_ERR_OOM         = -704, /**< allocation failure              */
    MOONLAB_STIM_ERR_IO          = -705, /**< file could not be read          */
    MOONLAB_STIM_ERR_OVERFLOW    = -706  /**< caller buffer too small         */
} moonlab_stim_status_t;

/**
 * @brief Where a parse failed and why.
 *
 * Every entry point taking an @p err out-parameter accepts NULL for it.  On
 * success @p code is MOONLAB_STIM_OK and @p message is empty.
 */
typedef struct {
    int    code;         /**< a moonlab_stim_status_t value            */
    size_t line;         /**< 1-based source line, 0 if not localised  */
    char   message[256]; /**< NUL-terminated, names the offending token */
} moonlab_stim_error_t;

/** @brief Parsed Stim circuit.  Opaque. */
typedef struct moonlab_stim_circuit moonlab_stim_circuit_t;

/* ================================================================== */
/*  Parsing and serialisation                                          */
/* ================================================================== */

/**
 * @brief Parse `.stim` circuit text.
 * @param text NUL-terminated circuit source.
 * @param err  optional failure detail.
 * @return circuit handle, or NULL on failure (see @p err).
 */
MOONLAB_API moonlab_stim_circuit_t* moonlab_stim_circuit_parse(
    const char* text, moonlab_stim_error_t* err);

/** @brief Parse a `.stim` file from disk.  @see moonlab_stim_circuit_parse. */
MOONLAB_API moonlab_stim_circuit_t* moonlab_stim_circuit_parse_file(
    const char* path, moonlab_stim_error_t* err);

MOONLAB_API void moonlab_stim_circuit_free(moonlab_stim_circuit_t* c);

/**
 * @brief Serialise back to `.stim` text in Stim's canonical spelling.
 *
 * @param flatten 0 preserves REPEAT blocks as they were parsed, 1 expands
 *                them.  Both forms re-parse to a semantically identical
 *                circuit.
 * @return malloc'd NUL-terminated text to release with
 *         moonlab_stim_text_free(), or NULL on allocation failure.
 */
MOONLAB_API char* moonlab_stim_circuit_to_text(
    const moonlab_stim_circuit_t* c, int flatten);

/** @brief Release text returned by any moonlab stim serialiser. */
MOONLAB_API void moonlab_stim_text_free(char* text);

/* ================================================================== */
/*  Introspection                                                      */
/* ================================================================== */

/** @brief One past the largest qubit index the circuit touches. */
MOONLAB_API size_t moonlab_stim_circuit_num_qubits(const moonlab_stim_circuit_t* c);
/** @brief Measurement record length, counting MPAD entries. */
MOONLAB_API size_t moonlab_stim_circuit_num_measurements(const moonlab_stim_circuit_t* c);
MOONLAB_API size_t moonlab_stim_circuit_num_detectors(const moonlab_stim_circuit_t* c);
/** @brief One past the largest OBSERVABLE_INCLUDE index. */
MOONLAB_API size_t moonlab_stim_circuit_num_observables(const moonlab_stim_circuit_t* c);
MOONLAB_API size_t moonlab_stim_circuit_num_ticks(const moonlab_stim_circuit_t* c);

/**
 * @brief Read declared coordinates.
 *
 * Writes min(n, @p cap) values, where n is the number of coordinates
 * declared for that index (0 when none were).  SHIFT_COORDS offsets in force
 * at the declaration are already folded in.
 *
 * @return n, or -1 when the index is out of range.
 */
MOONLAB_API long moonlab_stim_circuit_qubit_coords(
    const moonlab_stim_circuit_t* c, size_t qubit, double* out, size_t cap);
MOONLAB_API long moonlab_stim_circuit_detector_coords(
    const moonlab_stim_circuit_t* c, size_t detector, double* out, size_t cap);

/* ================================================================== */
/*  Lowering onto the Pauli-frame sampler                              */
/* ================================================================== */

/** @brief Number of pf_circuit_op_t entries the lowering produces. */
MOONLAB_API long moonlab_stim_circuit_num_ops(const moonlab_stim_circuit_t* c);
/** @brief Number of doubles the lowering's channel-argument table needs. */
MOONLAB_API long moonlab_stim_circuit_num_channel_args(const moonlab_stim_circuit_t* c);

/**
 * @brief Lower to a flat op list for pauli_frame_batch_sample_*_ex().
 *
 * REPEAT blocks are expanded.  PF_OP_PAULI_CHANNEL_1 / _2 ops carry the base
 * index of their probabilities in the channel-argument table in
 * pf_circuit_op_t::p.
 *
 * @return the number of ops written, or MOONLAB_STIM_ERR_OVERFLOW when a cap
 *         is too small.
 */
MOONLAB_API long moonlab_stim_circuit_lower(
    const moonlab_stim_circuit_t* c,
    pf_circuit_op_t* ops_out, size_t ops_cap,
    double* chan_args_out, size_t chan_cap,
    moonlab_stim_error_t* err);

/**
 * @brief Detector parity sets in CSR form over measurement-record indices.
 *
 * Detector d covers indices[offsets[d] .. offsets[d+1]).  @p offsets_out
 * needs num_detectors + 1 entries.  Either buffer may be NULL to size only.
 *
 * @return the number of index entries, or MOONLAB_STIM_ERR_OVERFLOW.
 */
MOONLAB_API long moonlab_stim_circuit_detector_csr(
    const moonlab_stim_circuit_t* c,
    size_t* offsets_out, size_t off_cap,
    uint32_t* indices_out, size_t idx_cap);

/** @brief As moonlab_stim_circuit_detector_csr(), for logical observables. */
MOONLAB_API long moonlab_stim_circuit_observable_csr(
    const moonlab_stim_circuit_t* c,
    size_t* offsets_out, size_t off_cap,
    uint32_t* indices_out, size_t idx_cap);

/**
 * @brief One byte per measurement record: 1 when the record is inverted.
 *
 * Stim's `!` target prefix inverts the value written to the record.  This is
 * a constant per record, so it cancels out of every detector and observable
 * (both are parities measured as a deviation from the noiseless reference
 * trajectory, and the constant enters that reference identically).  It
 * therefore only affects raw measurement sampling, which is where
 * moonlab_stim_circuit_sample_measurements() applies it.
 *
 * @return the number of records described, or MOONLAB_STIM_ERR_OVERFLOW.
 */
MOONLAB_API long moonlab_stim_circuit_measurement_inversions(
    const moonlab_stim_circuit_t* c, uint8_t* out, size_t cap);

/* ================================================================== */
/*  Sampling                                                           */
/* ================================================================== */

/**
 * @brief Sample raw measurement records.
 *
 * @param out receives num_measurements * num_shots bytes, measurement-major
 *            (record m, shot s at out[m * num_shots + s]).
 * @return the number of records written, or negative on error.
 */
MOONLAB_API long moonlab_stim_circuit_sample_measurements(
    const moonlab_stim_circuit_t* c, size_t num_shots, uint64_t seed,
    int num_threads, uint8_t* out);

/**
 * @brief Sample detectors and logical observables.
 *
 * Mirrors Stim's `compile_detector_sampler().sample(..., separate_observables=True)`.
 *
 * @param det_out num_detectors * num_shots bytes, detector-major; may be NULL.
 * @param obs_out num_observables * num_shots bytes, observable-major; may be NULL.
 * @return num_detectors on success, negative on error.
 */
MOONLAB_API long moonlab_stim_circuit_sample_detectors(
    const moonlab_stim_circuit_t* c, size_t num_shots, uint64_t seed,
    int num_threads, uint8_t* det_out, uint8_t* obs_out);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_STIM_CIRCUIT_H */
