/**
 * @file    moonlab_qgtl_backend.h
 * @brief   QGTL-shaped circuit-ingestion surface for moonlab.
 *
 * QGTL (`~/Desktop/quantum_geometric_tensor`) is the hardware-bridge
 * sibling of the moonlab + libirrep + SbNN platform.  It already
 * ships IBM Quantum / Rigetti / IonQ / D-Wave backends plus its own
 * `sim_*` simulator API.  This header is the **moonlab-side
 * ingestion contract** QGTL plugs into when it wants to route a
 * circuit through moonlab's simulator backend before paying for a
 * real-QPU shot.
 *
 * Design constraints:
 *
 * 1. **Numeric gate-type compatibility with QGTL.**
 *    `moonlab_qgtl_gate_t` is a verbatim copy of QGTL's
 *    `gate_type_t` numbering (see
 *    `quantum_geometric/core/quantum_base_types.h`).  QGTL's
 *    backend wrapper can just cast: `(moonlab_qgtl_gate_t)qgtl.type`.
 *    The enum values were locked at the v0.6.6 release; if QGTL
 *    renumbers `gate_type_t`, we mirror that here.
 *
 * 2. **Minimal contract.**  Four entry points: create circuit, add
 *    gate, execute, free.  Optional `_results_free` for the output
 *    block.  Nothing about backends, shots, transpilation, or
 *    error mitigation -- QGTL handles all of that upstream and
 *    just hands moonlab the canonicalised gate list.
 *
 * 3. **One-call execution.**  `moonlab_qgtl_execute` runs the full
 *    circuit through libquantumsim's quantum_state_t + gate_*
 *    surface in one call.  The caller does not need to know that
 *    moonlab even uses a state-vector backend underneath -- a
 *    future switch to MPS / Clifford is internal.
 *
 * 4. **Stable ABI.**  Every entry point carries MOONLAB_API and
 *    appears in the v0.6.6 stable surface.  QGTL can dlopen
 *    libquantumsim and dlsym these directly.
 *
 * @since v0.6.6
 */

#ifndef MOONLAB_QGTL_BACKEND_H
#define MOONLAB_QGTL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Gate-type tag, numerically matching QGTL's
 *  `quantum_geometric/core/quantum_base_types.h::gate_type_t` so
 *  QGTL's backend wrapper can cast directly.  Locked at v0.6.6. */
typedef enum {
    MOONLAB_QGTL_GATE_I    = 0,
    MOONLAB_QGTL_GATE_X    = 1,
    MOONLAB_QGTL_GATE_Y    = 2,
    MOONLAB_QGTL_GATE_Z    = 3,
    MOONLAB_QGTL_GATE_H    = 4,
    MOONLAB_QGTL_GATE_S    = 5,
    MOONLAB_QGTL_GATE_T    = 6,
    MOONLAB_QGTL_GATE_RX   = 7,  /**< params[0] = theta */
    MOONLAB_QGTL_GATE_RY   = 8,  /**< params[0] = theta */
    MOONLAB_QGTL_GATE_RZ   = 9,  /**< params[0] = theta */
    MOONLAB_QGTL_GATE_CNOT = 10, /**< control = source, target = sink */
    MOONLAB_QGTL_GATE_CY   = 11,
    MOONLAB_QGTL_GATE_CZ   = 12,
    MOONLAB_QGTL_GATE_SWAP = 13  /**< control = qubit1, target = qubit2 */
} moonlab_qgtl_gate_t;

/** @brief Status codes for the QGTL-ingestion surface.  Drawn from
 *  the `MOONLAB_STATUS_ERR_MODULE_BASE - 300..` band so they don't
 *  collide with the libirrep bridge or other modules. */
#define MOONLAB_QGTL_OK              ( 0)
#define MOONLAB_QGTL_BAD_ARG         (-301) /**< NULL pointer or out-of-range value. */
#define MOONLAB_QGTL_OOM             (-302) /**< Internal allocation failure. */
#define MOONLAB_QGTL_UNSUPPORTED     (-303) /**< Gate not implemented for this backend. */
#define MOONLAB_QGTL_INTERNAL        (-304) /**< Bug -- a moonlab core call failed. */

/** @brief Opaque circuit record.  Builds up a gate list, then
 *  `moonlab_qgtl_execute` runs the whole thing in one shot. */
typedef struct moonlab_qgtl_circuit moonlab_qgtl_circuit_t;

/** @brief Execution options.  All fields are caller-owned and
 *  caller-initialised.  Pass `{0}` for a single-shot run with a
 *  randomised RNG seed. */
typedef struct {
    int      num_shots; /**< 0 = no measurement loop, just compute prob distribution. */
    uint64_t rng_seed;  /**< 0 = use a deterministic-from-clock seed. */
    int      return_probabilities; /**< 1 = fill `probabilities`; 0 = leave NULL. */
} moonlab_qgtl_exec_options_t;

/** @brief Output buffer for `moonlab_qgtl_execute`.
 *
 *  Caller passes a zero-initialised `moonlab_qgtl_results_t`; the
 *  call allocates `outcomes` (length `num_shots`) and optionally
 *  `probabilities` (length `2^num_qubits`).  Release with
 *  `moonlab_qgtl_results_free`.  At `num_qubits > 24` the
 *  probability vector is 256 MB+ and the caller should pass
 *  `return_probabilities = 0`. */
typedef struct {
    int       num_qubits;
    int       num_shots;
    uint64_t *outcomes;       /**< length `num_shots`; each entry is a bitstring. */
    double   *probabilities;  /**< length `1ULL << num_qubits` or NULL. */
} moonlab_qgtl_results_t;

/**
 * @brief Allocate a fresh circuit record for `num_qubits` qubits.
 *
 * The internal gate list grows on demand; no `add_gate` quota.
 *
 * @return Owned handle, or NULL on OOM / `num_qubits` out of range.
 *         The current build caps `num_qubits` at `MOONLAB_MAX_QUBITS`
 *         (30 in WASM, 32 native).
 */
MOONLAB_API moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_create(int num_qubits);

/** @brief Release a handle returned by @ref moonlab_qgtl_circuit_create. */
MOONLAB_API void moonlab_qgtl_circuit_free(moonlab_qgtl_circuit_t *c);

/**
 * @brief Append a gate to the circuit's gate list.
 *
 * @param[in]  c       Circuit handle.
 * @param[in]  type    Gate tag; see @ref moonlab_qgtl_gate_t.
 * @param[in]  target  Target qubit (or qubit2 for SWAP).
 * @param[in]  control Control qubit for CNOT / CY / CZ / SWAP;
 *                     ignored for single-qubit gates.
 * @param[in]  params  Parameter array.  RX / RY / RZ read
 *                     `params[0]` as `theta`.  Single-qubit gates
 *                     without parameters tolerate NULL.
 * @return MOONLAB_QGTL_OK on success or a negative status code.
 */
MOONLAB_API int
moonlab_qgtl_add_gate(moonlab_qgtl_circuit_t *c,
                      moonlab_qgtl_gate_t    type,
                      int                    target,
                      int                    control,
                      const double          *params);

/**
 * @brief Run the circuit through moonlab's state-vector backend.
 *
 * Initialises a `quantum_state_t` of size `2^num_qubits`, applies
 * every gate in the recorded list (in order), then either dumps
 * the probability distribution into `out->probabilities` or
 * samples `opts->num_shots` measurement outcomes into
 * `out->outcomes` (or both, if `return_probabilities = 1` and
 * `num_shots > 0`).
 *
 * @return MOONLAB_QGTL_OK on success or a negative status code.
 */
MOONLAB_API int
moonlab_qgtl_execute(moonlab_qgtl_circuit_t           *c,
                     const moonlab_qgtl_exec_options_t *opts,
                     moonlab_qgtl_results_t           *out);

/** @brief Release any buffers attached to a results record by
 *  @ref moonlab_qgtl_execute.  Safe on a zero-initialised record. */
MOONLAB_API void moonlab_qgtl_results_free(moonlab_qgtl_results_t *r);

/* ------------------------------------------------------------------
 * Introspection (handy for QGTL's circuit-validation step).
 * ------------------------------------------------------------------ */

/** @brief Number of qubits the circuit was created with. */
MOONLAB_API int moonlab_qgtl_circuit_num_qubits(const moonlab_qgtl_circuit_t *c);

/** @brief Number of gates recorded in the circuit. */
MOONLAB_API int moonlab_qgtl_circuit_num_gates(const moonlab_qgtl_circuit_t *c);

/* ------------------------------------------------------------------
 * Serialization surface (since v0.8.3).
 *
 * Portable line-oriented text format so the same circuit can move
 * between moonlab, QGTL, libirrep, SbNN, and any language binding
 * without dragging in a JSON dependency.  Format:
 *
 *     # moonlab-circuit v1
 *     NUM_QUBITS <n>
 *     <GATE> <target> [control] [theta]
 *     ...
 *
 * Gate names match the enum: I X Y Z H S T RX RY RZ CNOT CY CZ SWAP.
 * Lines starting with '#' or blank are ignored.  Theta uses '%.17g'
 * so doubles roundtrip exactly.
 * ------------------------------------------------------------------ */

/**
 * @brief Serialize a circuit to a portable text buffer.
 *
 * @param[in]   c            Circuit handle.
 * @param[out]  buf          Output buffer.  May be NULL only when
 *                           `buf_size == 0` (size-query mode).
 * @param[in]   buf_size     Bytes available at `buf`.
 * @param[out]  out_written  Optional.  On success, set to the byte
 *                           count written (excluding terminating NUL,
 *                           which is always emitted when `buf_size > 0`).
 *                           In size-query mode this is the size you
 *                           would have to pass for a successful call.
 *
 * @return MOONLAB_QGTL_OK on success, MOONLAB_QGTL_BAD_ARG on NULL
 *         circuit, or MOONLAB_QGTL_OOM if `buf_size > 0 && buf_size`
 *         is insufficient.  In the OOM case `*out_written` reports
 *         the required size.
 */
MOONLAB_API int
moonlab_qgtl_circuit_serialize(const moonlab_qgtl_circuit_t *c,
                               char  *buf,
                               size_t buf_size,
                               size_t *out_written);

/**
 * @brief Parse a circuit from a text buffer produced by
 *        @ref moonlab_qgtl_circuit_serialize (or written by hand).
 *
 * @param[in]   buf        Input buffer.
 * @param[in]   buf_size   Bytes to read at `buf`.  Pass 0 for a
 *                         NUL-terminated string.  With an explicit non-zero
 *                         length, a NUL anywhere inside the declared extent
 *                         is invalid input.
 * @param[out]  out_status Optional.  Set to MOONLAB_QGTL_OK on
 *                         success or a negative code on failure.
 *
 * @return Owned circuit handle on success, NULL on failure.  Free
 *         via @ref moonlab_qgtl_circuit_free.
 */
MOONLAB_API moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_deserialize(const char *buf,
                                 size_t      buf_size,
                                 int        *out_status);

/**
 * @brief Save a circuit to a file in the same text format as
 *        @ref moonlab_qgtl_circuit_serialize.
 *
 * @return MOONLAB_QGTL_OK on success, MOONLAB_QGTL_BAD_ARG / OOM
 *         / INTERNAL (= I/O) on failure.
 */
MOONLAB_API int
moonlab_qgtl_circuit_save(const moonlab_qgtl_circuit_t *c,
                          const char *path);

/**
 * @brief Load a circuit previously written by
 *        @ref moonlab_qgtl_circuit_save.
 *
 * @return Owned handle, or NULL on failure.
 */
MOONLAB_API moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_load(const char *path, int *out_status);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QGTL_BACKEND_H */
