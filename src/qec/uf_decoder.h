/**
 * @file uf_decoder.h
 * @brief Union-find decoder over a detector error model.
 *
 * Consumes the same input a matching decoder does: a graph whose nodes are
 * detectors plus one virtual boundary node, and whose edges are the error
 * mechanisms of a detector error model.  Each edge carries the set of
 * logical observables it flips, so decoding a shot reduces to choosing a
 * subset of edges whose boundary is the observed syndrome and XORing their
 * observable masks.
 *
 * The algorithm is Delfosse-Nickerson union-find: clusters are grown around
 * lit detectors until every cluster has even parity or touches the
 * boundary, then the spanning forest of the grown region is peeled to give
 * a correction.  Growth is near-linear in the number of defects, which is
 * what makes it competitive with sparse blossom while staying far simpler.
 *
 * Decoding is per-shot independent, so a batch is split across threads.
 */
#ifndef MOONLAB_UF_DECODER_H
#define MOONLAB_UF_DECODER_H

#include <stddef.h>
#include <stdint.h>

#include "../applications/moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moonlab_uf_decoder moonlab_uf_decoder_t;

/** Sentinel destination marking an edge that runs to the boundary. */
#define MOONLAB_UF_BOUNDARY UINT32_MAX

/**
 * @brief Build a decoder from a detector error model in edge-list form.
 *
 * @param num_detectors   number of detector nodes.
 * @param num_observables number of logical observables (<= 64).
 * @param edge_a          first detector of each edge.
 * @param edge_b          second detector, or MOONLAB_UF_BOUNDARY.
 * @param edge_weight     edge weight; larger means less likely. Growth is
 *                        quantised from these, so relative size is what
 *                        matters.  Pass NULL for unweighted growth.
 * @param edge_obs        bitmask of observables this edge flips.
 * @param num_edges       number of edges.
 * @return decoder handle, or NULL on allocation failure / invalid input.
 */
MOONLAB_API moonlab_uf_decoder_t* moonlab_uf_decoder_new(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges);

MOONLAB_API void moonlab_uf_decoder_free(moonlab_uf_decoder_t* d);

/**
 * @brief Decode a batch of shots.
 *
 * @param det   detector data, detector-major: detector i of shot s at
 *              det[i * num_shots + s], matching the layout
 *              pauli_frame_batch_sample_detectors writes.
 * @param obs_out receives num_observables * num_shots bytes, observable-major.
 * @param num_threads <=0 selects all cores.
 * @return num_shots on success, negative on error.
 */
MOONLAB_API long moonlab_uf_decode_batch(
    moonlab_uf_decoder_t* d, const uint8_t* det, size_t num_shots,
    int num_threads, uint8_t* obs_out);

/** @brief Number of edges the decoder was built with. */
MOONLAB_API size_t moonlab_uf_decoder_num_edges(const moonlab_uf_decoder_t* d);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_UF_DECODER_H */
