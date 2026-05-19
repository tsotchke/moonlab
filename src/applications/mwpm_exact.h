/**
 * @file    mwpm_exact.h
 * @brief   Exact minimum-weight perfect matching for toric-code syndromes.
 *
 * Lifted from `examples/applications/surface_code_threshold.c` so the
 * decoder-bench `MOONLAB_DECODER_MWPM_EXACT` slot (and any external
 * caller) can hit it through the stable ABI.
 *
 * Algorithm:
 *  - Up to `MWPM_BRUTE_FORCE_MAX` (= 10) defects: enumerate all
 *    `(n - 1)!!` perfect matchings via recursion, pick the minimum.
 *  - Larger sets: greedy nearest-pair seed + 2-opt local search.
 *    Worst-case O(n^3) seed + O(n^2) per 2-opt iteration; converges
 *    within a hundred iterations for any practical defect count.
 *
 * The toric distance is L1 modulo `d`; the correction path is the
 * geodesic along the shorter wrap in each axis, picking up exactly
 * one edge per step.
 *
 * @since v0.7.2
 */

#ifndef MOONLAB_MWPM_EXACT_H
#define MOONLAB_MWPM_EXACT_H

#include <stddef.h>

#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOONLAB_MWPM_OK             ( 0)
#define MOONLAB_MWPM_BAD_ARG        (-601)
#define MOONLAB_MWPM_OOM            (-602)
#define MOONLAB_MWPM_INFEASIBLE     (-603) /**< Odd defect count on closed surface. */

/** @brief Cutoff between brute-force enumeration + greedy + 2-opt.
 *  At n = 10 there are 9!! = 945 matchings to enumerate; n = 12 is
 *  10395, n = 14 is 135135.  The 2-opt path takes over past 14. */
#define MOONLAB_MWPM_BRUTE_FORCE_MAX 10

/**
 * @brief  Decode an X-error pattern on a `d x d` toric code by
 *         minimum-weight matching of the Z-vertex defects.
 *
 * @param[in]  distance     Toric distance `d`, `d >= 2`.
 * @param[in]  syndromes    Length `d*d`; non-zero entries flag
 *                          defective Z-vertices in row-major order
 *                          (vertex `(a, b)` at index `a*d + b`).
 * @param[in]  num_stabs    Length of `syndromes`; must be `d*d`.
 * @param[out] corrections  Length `2*d*d`; receives the flipped-edge
 *                          mask in the same row-major layout used by
 *                          `decoder_bench.c::torus_edge_between`
 *                          (horizontal edges `[0, d*d)`, vertical
 *                          edges `[d*d, 2*d*d)`).  Caller must
 *                          zero-init.
 * @return  `MOONLAB_MWPM_OK` on success; `MOONLAB_MWPM_INFEASIBLE`
 *          when the defect count is odd (no perfect matching exists
 *          on the closed surface).  Negative on other failures.
 */
MOONLAB_API int
moonlab_mwpm_exact_decode_toric(int distance,
                                const unsigned char *syndromes,
                                int num_stabs,
                                unsigned char *corrections);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MWPM_EXACT_H */
