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
 *
 * CORRELATED TWO-PASS DECODING.  A decomposed detector error model emits
 * mechanisms like "error(p) D1 D2 ^ D3 D4": ONE physical fault whose
 * graphlike components are perfectly correlated.  Matching decoders treat
 * the components as independent edges and discard the correlation.  A
 * decoder built with moonlab_uf_decoder_new_correlated() decodes each shot
 * twice: pass 1 as usual, then, for every mechanism one of whose components
 * lies on the pass-1 correction, the partner components' probabilities are
 * replaced by their conditional probabilities given the used component, and
 * pass 2 re-matches under the updated weights.  Pass 2's answer is emitted.
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

/**
 * @brief Build a CORRELATED decoder: two-pass decoding over mechanism links.
 *
 * In addition to the plain edge list, takes the per-edge merged flip
 * probability and a list of correlation links.  A link (u, v, q) states
 * that edges u and v are components of one physical mechanism (or several)
 * whose combined probability of firing is q; firing flips BOTH edges.  A
 * mechanism with C components contributes all C*(C-1)/2 pairwise links;
 * links repeated across mechanisms must be pre-combined by the caller with
 * q = q1(1-q2) + q2(1-q1).
 *
 * The conditional update applied between the passes is exact within the
 * independent-mechanism model.  Split edge v's sources into joint
 * mechanisms (fire both u and v, combined probability q) and v-only
 * mechanisms, q_v = (p_v - q)/(1 - 2q); u-only likewise.  Given that pass 1
 * decided u flipped, the flip came from a joint mechanism with probability
 *   r = q(1-q_u) / (q(1-q_u) + q_u(1-q)),
 * and v is then flipped iff an odd number of its remaining sources fired:
 *   P(v | u) = r(1-q_v) + (1-r) q_v.
 * Edge v's pass-2 weight is ln((1-P(v|u))/P(v|u)), floored at a small
 * positive value so shortest paths stay well defined, and never above the
 * static weight.  All link weights are precomputed at construction.
 *
 * Decoding through moonlab_uf_decode_batch() is unchanged in signature;
 * a decoder built here runs both passes.  Pass 2 re-matches the pass-1
 * clusters exactly: boosted shortest distances are computed per shot on a
 * mini-graph over the shot's defects and the boosted edges' endpoints,
 * whose edges are the static all-pairs distances plus the boosted edges,
 * so the per-cluster matching stays exact under the updated weights.
 *
 * @param edge_prob    merged flip probability per edge, in (0, 1).
 * @param corr_a       first edge index of each link.
 * @param corr_b       second edge index of each link.
 * @param corr_joint_p combined joint probability q per link, 0 < q < 0.5.
 * @param num_corr     number of links; 0 degrades to the plain decoder.
 * @return decoder handle, or NULL on allocation failure / invalid input.
 */
MOONLAB_API moonlab_uf_decoder_t* moonlab_uf_decoder_new_correlated(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges,
    const double* edge_prob,
    const uint32_t* corr_a, const uint32_t* corr_b,
    const double* corr_joint_p, size_t num_corr);

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
