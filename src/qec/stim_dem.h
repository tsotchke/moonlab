/**
 * @file stim_dem.h
 * @brief Import and export of Stim's detector error model (`.dem`) format.
 *
 * A detector error model is the decoding problem stripped of the circuit: a
 * list of independent error mechanisms, each with a probability, the
 * detectors it lights, and the logical observables it flips.  It is the
 * interchange format the QEC community decodes against, and what
 * `pymatching.Matching.from_detector_error_model` consumes.
 *
 * This module is the bridge in both directions:
 *
 * - IMPORT.  Stim-produced DEM text becomes the edge list
 *   `src/qec/uf_decoder.h` consumes, so moonlab's union-find and correlated
 *   decoders can decode any circuit Stim can analyse.
 * - EXPORT.  A moonlab edge list becomes standard DEM text that Stim parses
 *   and PyMatching loads, so a moonlab noise model can be decoded by anyone
 *   else's decoder.
 *
 * The format reference is Stim's
 * `doc/file_format_dem_detector_error_model.md`:
 *
 * @verbatim
 * <INSTRUCTION> ::= <NAME> <TAG>? <PARENS_ARGUMENTS>? <TARGETS>
 * <TARG>        ::= 'D' <uint> | 'L' <uint> | <uint> | '^'
 * @endverbatim
 *
 * with instructions `error`, `detector`, `logical_observable`,
 * `shift_detectors` and `repeat N { ... }`.
 *
 * DECOMPOSED MECHANISMS.  Stim emits `error(p) D1 D2 ^ D3 D4` for one
 * physical fault whose graphlike components are perfectly correlated.  A
 * matching decoder throws that correlation away.  On import each component
 * becomes an edge and every pair of components of the same mechanism becomes
 * a correlation link, which is exactly what
 * moonlab_uf_decoder_new_correlated() consumes -- so the correlation Stim
 * took the trouble to record is the correlation moonlab decodes with.
 * Export is the exact inverse, so an edge list survives a round trip through
 * DEM text.
 */
#ifndef MOONLAB_STIM_DEM_H
#define MOONLAB_STIM_DEM_H

#include <stddef.h>
#include <stdint.h>

#include "../applications/moonlab_api.h"
#include "stim_circuit.h" /* moonlab_stim_error_t, moonlab_stim_status_t */
#include "uf_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Parsed detector error model in moonlab edge-list form.  Opaque. */
typedef struct moonlab_dem moonlab_dem_t;

/* ================================================================== */
/*  Parsing and serialisation                                          */
/* ================================================================== */

/**
 * @brief Parse DEM text into moonlab's edge list.
 *
 * `repeat` blocks and `shift_detectors` are resolved to absolute detector
 * indices.  Parallel mechanisms on the same (detector pair, observable mask)
 * are merged with p = p1(1-p2) + p2(1-p1), which is what PyMatching does
 * internally, so both decoders see one model.
 *
 * @return model handle, or NULL on failure (see @p err).
 */
MOONLAB_API moonlab_dem_t* moonlab_dem_parse(
    const char* text, moonlab_stim_error_t* err);

/** @brief Parse a `.dem` file from disk.  @see moonlab_dem_parse. */
MOONLAB_API moonlab_dem_t* moonlab_dem_parse_file(
    const char* path, moonlab_stim_error_t* err);

MOONLAB_API void moonlab_dem_free(moonlab_dem_t* d);

/**
 * @brief Serialise the model back to DEM text.
 *
 * @return malloc'd NUL-terminated text to release with
 *         moonlab_stim_text_free(), or NULL on allocation failure.
 */
MOONLAB_API char* moonlab_dem_to_text(const moonlab_dem_t* d);

/**
 * @brief Serialise an edge list directly to DEM text.
 *
 * The inverse of the import merge.  Each correlation link (u, v, q) is
 * emitted as one decomposed mechanism `error(q) <u> ^ <v>`, and each edge
 * then carries the residual probability left after peeling off every joint
 * probability touching it, so that
 * `edges -> text -> moonlab_dem_parse -> edges` is a fixed point.
 *
 * @param edge_prob   per-edge merged flip probability in (0, 1).
 * @param edge_obs    per-edge observable bitmask (at most 64 observables).
 * @param corr_a,corr_b,corr_joint_p correlation links; may be NULL when
 *                    @p num_corr is 0.
 * @return malloc'd DEM text to release with moonlab_stim_text_free(), or
 *         NULL on failure (see @p err).  Inconsistent inputs -- a residual
 *         probability driven negative by the links -- fail with
 *         MOONLAB_STIM_ERR_BAD_ARG rather than silently clamping.
 */
MOONLAB_API char* moonlab_dem_text_from_edges(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_prob, const uint64_t* edge_obs, size_t num_edges,
    const uint32_t* corr_a, const uint32_t* corr_b,
    const double* corr_joint_p, size_t num_corr,
    moonlab_stim_error_t* err);

/* ================================================================== */
/*  Edge-list view                                                     */
/* ================================================================== */

MOONLAB_API size_t moonlab_dem_num_detectors(const moonlab_dem_t* d);
MOONLAB_API size_t moonlab_dem_num_observables(const moonlab_dem_t* d);
MOONLAB_API size_t moonlab_dem_num_edges(const moonlab_dem_t* d);
MOONLAB_API size_t moonlab_dem_num_correlations(const moonlab_dem_t* d);

/**
 * @brief Components with more than two detectors, which are not graphlike.
 *
 * These cannot be edges in a matching graph.  They are counted here rather
 * than silently dropped, so a caller can tell whether the model it is
 * decoding was fully representable.  A well decomposed Stim DEM
 * (`decompose_errors=True`) reports 0.
 */
MOONLAB_API size_t moonlab_dem_num_hyperedges(const moonlab_dem_t* d);

/**
 * @brief Components with no detectors at all, which flip observables only.
 *
 * Counted separately from hyperedges: they are representable in principle
 * but carry no syndrome information, so no decoder can act on them.
 */
MOONLAB_API size_t moonlab_dem_num_detectorless(const moonlab_dem_t* d);

/**
 * @brief Copy out the edge list in moonlab_uf_decoder_new() layout.
 *
 * @param edge_b       MOONLAB_UF_BOUNDARY for an edge running to the boundary.
 * @param edge_weight  log-likelihood ratio ln((1-p)/p), what the decoder matches on.
 * @param edge_prob    the raw probability p.
 * Any out-pointer may be NULL to fetch only some columns.
 * @return the number of edges, or MOONLAB_STIM_ERR_OVERFLOW when @p cap is
 *         below moonlab_dem_num_edges().
 */
MOONLAB_API long moonlab_dem_edges(
    const moonlab_dem_t* d,
    uint32_t* edge_a, uint32_t* edge_b,
    double* edge_weight, uint64_t* edge_obs,
    double* edge_prob, size_t cap);

/**
 * @brief Copy out the correlation links in
 *        moonlab_uf_decoder_new_correlated() layout.
 * @return the number of links, or MOONLAB_STIM_ERR_OVERFLOW.
 */
MOONLAB_API long moonlab_dem_correlations(
    const moonlab_dem_t* d,
    uint32_t* corr_a, uint32_t* corr_b,
    double* corr_joint_p, size_t cap);

/** @brief Coordinates from a `detector(...)` instruction.
 *  @see moonlab_stim_circuit_detector_coords. */
MOONLAB_API long moonlab_dem_detector_coords(
    const moonlab_dem_t* d, size_t detector, double* out, size_t cap);

/**
 * @brief Build a decoder straight from the model.
 * @param correlated non-zero selects the two-pass correlated decoder, which
 *                   consumes the links Stim's `^` decompositions carry.
 * @return decoder handle to release with moonlab_uf_decoder_free(), or NULL.
 */
MOONLAB_API moonlab_uf_decoder_t* moonlab_dem_make_uf_decoder(
    const moonlab_dem_t* d, int correlated);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_STIM_DEM_H */
