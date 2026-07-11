/**
 * @file    decoder_bench.h
 * @brief   Multi-decoder bench harness scaffold (v0.6.7).
 *
 * Unified slot dispatcher so moonlab benchmarks can compare:
 *
 *   - **GREEDY**   : built-in nearest-pair matching, always available.
 *   - **MWPM_EXACT**: exact MWPM by enumeration (k <= 14 defects),
 *                    falls back to greedy + 2-opt otherwise.  Built-in
 *                    via the same code path as
 *                    `examples/applications/surface_code_threshold.c`.
 *   - **SBNN**     : SbNN's learned `qec_decoder_*` API
 *                    (`~/Desktop/spin_based_neural_network/include/qec_decoder/`).
 *                    Returns `MOONLAB_DECODER_NOT_BUILT` until
 *                    `-DQSIM_ENABLE_SBNN=ON` lands in v0.6.8.
 *   - **LIBIRREP_SS**: libirrep's `single_shot.h` decoder.  Returns
 *                    `MOONLAB_DECODER_NOT_BUILT` until v0.6.8 wires
 *                    it behind `QSIM_ENABLE_LIBIRREP`.
 *   - **PYMATCHING**: optional Stim-pymatching reference via the
 *                    POSIX Python bridge.  Slot reserved on Windows/Web
 *                    builds until a native subprocess transport lands.
 *
 * This release ships the slot enum + dispatcher + GREEDY decoder
 * working in-tree.  v0.6.8 fills the SBNN, LIBIRREP_SS, and
 * PYMATCHING slots with real linkage.
 *
 * @since v0.6.7
 */

#ifndef MOONLAB_DECODER_BENCH_H
#define MOONLAB_DECODER_BENCH_H

#include <stddef.h>
#include <stdint.h>

#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Slot tag for the decoder dispatcher.  Numerically stable
 *  across releases so JSON output can dispatch by ID. */
typedef enum {
    MOONLAB_DECODER_GREEDY     = 0, /**< Built-in nearest-pair matching. */
    MOONLAB_DECODER_MWPM_EXACT = 1, /**< Built-in exact + 2-opt fallback. */
    MOONLAB_DECODER_SBNN       = 2, /**< SbNN learned decoder (v0.6.8). */
    MOONLAB_DECODER_LIBIRREP_SS = 3, /**< libirrep single-shot (v0.6.8). */
    MOONLAB_DECODER_PYMATCHING = 4  /**< Stim-pymatching reference (v0.6.8). */
} moonlab_decoder_kind_t;

/** @brief Status codes for the decoder dispatcher. */
#define MOONLAB_DECODER_OK              ( 0)
#define MOONLAB_DECODER_NOT_BUILT       (-401) /**< Slot needs an external library not linked. */
#define MOONLAB_DECODER_BAD_ARG         (-402) /**< NULL pointer or out-of-range value. */
#define MOONLAB_DECODER_INFEASIBLE      (-403) /**< Odd number of defects, no matching exists. */
#define MOONLAB_DECODER_OOM             (-404) /**< Allocation failure. */

/** @brief Code geometry for the dispatcher.  Toric / surface codes
 *  are described by a single distance + lattice kind; richer codes
 *  pass their own structure separately (this header stays
 *  scaffold-lean). */
typedef struct {
    int distance;     /**< Code distance d. */
    int num_qubits;   /**< Total qubits.  Toric d^2 -> 2 d^2; rotated surface -> d^2. */
    int is_toric;     /**< 1 = periodic, 0 = open-boundary (rotated surface). */
} moonlab_decoder_code_t;

/** @brief Input syndrome buffer + output correction.
 *
 *  `syndromes[i]` is non-zero when stabiliser `i` is flagged.  The
 *  decoder writes `corrections[q]` = 1 if data qubit `q` should be
 *  flipped to neutralise the syndrome.  Both arrays are
 *  caller-allocated, length `num_qubits` (or larger -- decoder
 *  reads `num_qubits` entries).
 *
 *  The dispatcher does not modify any moonlab state.  This is a
 *  pure-function interface: same syndrome in -> same correction
 *  out (modulo RNG seed for stochastic decoders). */
typedef struct {
    const moonlab_decoder_code_t *code;
    const unsigned char          *syndromes;
    unsigned char                *corrections;
    int                           num_stabilisers; /**< length of `syndromes`. */
    uint64_t                      rng_seed;        /**< 0 = deterministic clock-derived. */
} moonlab_decoder_input_t;

/**
 * @brief Decode a syndrome with the requested slot.
 *
 * @param[in]  slot   Which decoder to use.
 * @param[in]  in     Syndrome + code + output buffers.
 * @return MOONLAB_DECODER_OK on success, MOONLAB_DECODER_NOT_BUILT
 *         when the requested slot needs an external library, or a
 *         negative status code on failure.
 */
MOONLAB_API int
moonlab_decoder_decode(moonlab_decoder_kind_t          slot,
                       const moonlab_decoder_input_t  *in);

/** @brief Indicates whether a slot has its external dependency
 *  linked at build time.  Always returns 1 for GREEDY / MWPM_EXACT.
 *  Returns 0 for SBNN / LIBIRREP_SS / PYMATCHING until v0.6.8+. */
MOONLAB_API int moonlab_decoder_slot_available(moonlab_decoder_kind_t slot);

/** @brief Human-readable name for a slot, useful for JSON output. */
MOONLAB_API const char *moonlab_decoder_slot_name(moonlab_decoder_kind_t slot);

/* ------------------------------------------------------------------
 * Runtime decoder registry (since v1.0.3)
 *
 * Parallel to moonlab_register_backend in the scheduler.  Lets
 * private overlays plug new decoders in at runtime under a name,
 * without touching the moonlab_decoder_kind_t enum (which stays the
 * stable wire-level slot tag).  Five baked-in decoders auto-register
 * at first use: "greedy", "mwpm_exact", "sbnn", "libirrep_single_shot",
 * "pymatching".  Their availability is governed by the same build
 * flags as the enum dispatcher (CPU-only decoders always available;
 * SBNN / LIBIRREP_SS gated on link-time presence; PYMATCHING gated
 * on the bridge script path and POSIX subprocess support).
 *
 * Use cases for the registry:
 *
 *   - Private overlay registers a proprietary decoder ("tsotchke-bp-osd")
 *     without code changes in the public moonlab tree.
 *   - Benchmarks can compare across registered names instead of
 *     enumerating slots, so adding a decoder is a one-line registry call.
 *   - QGTL or any external project can swap in a hardware-decoded path
 *     by registering a thin wrapper that submits to its decoder service.
 * ------------------------------------------------------------------ */

/** @brief Decoder function signature.  Same semantics as the enum
 *  dispatcher: read `in->syndromes`, write `in->corrections`, return
 *  MOONLAB_DECODER_OK on success or a negative status code on
 *  failure.  `ctx` is the opaque pointer passed at registration. */
typedef int (*moonlab_decoder_fn)(const moonlab_decoder_input_t *in,
                                  void                          *ctx);

/** @brief Registered decoder record.  Names are stored as a strdup'd
 *  copy so callers may free their string after register; descriptions
 *  are likewise copied. */
typedef struct {
    const char         *name;        /**< stable, registry-owned */
    moonlab_decoder_fn  fn;          /**< implementation pointer */
    void               *ctx;         /**< user context, passed to fn */
    const char         *description; /**< human-readable, registry-owned */
} moonlab_decoder_entry_t;

/** @brief Register (or replace) a decoder under `name`.  Returns 0
 *  on success, MOONLAB_DECODER_BAD_ARG on NULL input or registry-full. */
MOONLAB_API int moonlab_register_decoder(const char         *name,
                                         moonlab_decoder_fn  fn,
                                         void               *ctx,
                                         const char         *description);

/** @brief Remove a decoder from the registry.  Returns 0 on success,
 *  negative on not-found. */
MOONLAB_API int moonlab_unregister_decoder(const char *name);

/** @brief Look up a decoder by name.  Returns NULL if not registered. */
MOONLAB_API const moonlab_decoder_entry_t *
moonlab_lookup_decoder(const char *name);

/** @brief Dispatch decode-by-name.  Equivalent to looking up `name`
 *  and calling its fn.  Returns MOONLAB_DECODER_BAD_ARG if `name` is
 *  not registered (which is the runtime equivalent of an unknown enum). */
MOONLAB_API int
moonlab_decoder_decode_by_name(const char                     *name,
                               const moonlab_decoder_input_t  *in);

/** @brief Number of decoders currently registered. */
MOONLAB_API int moonlab_num_decoders(void);

/** @brief Copy up to `max` registered decoder names into `out_names`.
 *  Returns the number copied. */
MOONLAB_API int moonlab_list_decoders(const char **out_names, int max);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_DECODER_BENCH_H */
