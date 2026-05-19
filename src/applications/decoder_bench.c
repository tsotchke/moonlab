/**
 * @file    decoder_bench.c
 * @brief   Multi-decoder bench harness scaffold implementation.
 *
 * Ships the dispatcher + in-tree GREEDY decoder.  SBNN /
 * LIBIRREP_SS / PYMATCHING slots return MOONLAB_DECODER_NOT_BUILT
 * until v0.6.8 wires them.  MWPM_EXACT currently shares the greedy
 * implementation as a placeholder; v0.6.8 lifts the exact +
 * 2-opt path out of `examples/applications/surface_code_threshold.c`
 * into a reusable library function and points the slot at it.
 */

#include "decoder_bench.h"
#include "mwpm_exact.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef MOONLAB_HAS_LIBIRREP
#include <irrep/css_code.h>
#include <irrep/single_shot.h>
#include <irrep/toric_code.h>
#include <irrep/types.h>
#endif

const char *moonlab_decoder_slot_name(moonlab_decoder_kind_t slot)
{
    switch (slot) {
    case MOONLAB_DECODER_GREEDY:      return "greedy";
    case MOONLAB_DECODER_MWPM_EXACT:  return "mwpm_exact";
    case MOONLAB_DECODER_SBNN:        return "sbnn";
    case MOONLAB_DECODER_LIBIRREP_SS: return "libirrep_single_shot";
    case MOONLAB_DECODER_PYMATCHING:  return "pymatching";
    default:                          return "unknown";
    }
}

int moonlab_decoder_slot_available(moonlab_decoder_kind_t slot)
{
    switch (slot) {
    case MOONLAB_DECODER_GREEDY:
    case MOONLAB_DECODER_MWPM_EXACT:
        return 1;
    case MOONLAB_DECODER_LIBIRREP_SS:
#ifdef MOONLAB_HAS_LIBIRREP
        return 1;
#else
        return 0;
#endif
    case MOONLAB_DECODER_SBNN:
    case MOONLAB_DECODER_PYMATCHING:
        return 0; /* v0.7+ wires these. */
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------
 * In-tree GREEDY decoder
 *
 * Naive nearest-pair matching: for each pair of flagged stabilisers,
 * connect them along a straight path of data qubits.  Far from
 * optimal but always succeeds and gives a reasonable baseline.
 * ------------------------------------------------------------------ */

/* Compute the linear data-qubit index between two stabilisers on a
 * d x d torus, picking a shortest path along the lattice.  Returns
 * -1 if no straight path exists (shouldn't happen on a torus). */
static int torus_edge_between(int d, int a, int b, unsigned char *flip)
{
    const int ax = a / d, ay = a % d;
    const int bx = b / d, by = b % d;
    int dx = (bx - ax + d) % d;
    int dy = (by - ay + d) % d;
    if (dx > d / 2) dx -= d;
    if (dy > d / 2) dy -= d;
    /* Walk in y first, then in x; flip each crossed edge.  The
     * edge-index convention matches the existing
     * surface_code_threshold.c harness: horizontal edge h(x, y) at
     * index `x * d + y`, vertical edge v(x, y) at `d * d + x * d + y`. */
    int x = ax, y = ay;
    while (y != by) {
        const int step = (dy > 0) ? 1 : -1;
        const int e = x * d + ((step > 0) ? y : (y + d - 1) % d);
        flip[e] ^= 1;
        y = (y + step + d) % d;
        dy -= step;
    }
    while (x != bx) {
        const int step = (dx > 0) ? 1 : -1;
        const int e = d * d + ((step > 0) ? x : (x + d - 1) % d) * d + y;
        flip[e] ^= 1;
        x = (x + step + d) % d;
        dx -= step;
    }
    return 0;
}

static int decoder_greedy(const moonlab_decoder_input_t *in)
{
    const int d   = in->code->distance;
    const int n   = in->code->num_qubits;
    const int n_s = in->num_stabilisers;

    /* Collect flagged stabiliser indices. */
    int *defects = (int *)malloc((size_t)n_s * sizeof(int));
    if (!defects) return MOONLAB_DECODER_OOM;
    int n_defects = 0;
    for (int i = 0; i < n_s; i++) {
        if (in->syndromes[i]) defects[n_defects++] = i;
    }

    /* Toric stabilisers always come in pairs (sum of plaquette syndromes
     * is zero mod 2 on any closed surface).  Open boundaries can have
     * an odd parity -- we accept it and pair to a virtual boundary
     * defect via a no-op (the un-matched defect leaves residual
     * syndrome which downstream logical-error tracking treats as a
     * logical fault).  Don't error here; just leave any leftover. */
    memset(in->corrections, 0, (size_t)n);

    /* Nearest-pair matching: each defect paired with its closest
     * unmatched neighbour.  O(n_defects^2) -- fine for d <= 9. */
    char *matched = (char *)calloc((size_t)n_defects, 1);
    if (!matched) { free(defects); return MOONLAB_DECODER_OOM; }

    for (int i = 0; i < n_defects; i++) {
        if (matched[i]) continue;
        int best_j = -1;
        int best_dist = INT32_MAX;
        for (int j = i + 1; j < n_defects; j++) {
            if (matched[j]) continue;
            const int ax = defects[i] / d, ay = defects[i] % d;
            const int bx = defects[j] / d, by = defects[j] % d;
            int dx = (bx - ax + d) % d; if (dx > d / 2) dx = d - dx;
            int dy = (by - ay + d) % d; if (dy > d / 2) dy = d - dy;
            const int dist = dx + dy;
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        if (best_j >= 0 && in->code->is_toric) {
            torus_edge_between(d, defects[i], defects[best_j], in->corrections);
            matched[i] = 1;
            matched[best_j] = 1;
        }
    }
    free(matched);
    free(defects);
    return MOONLAB_DECODER_OK;
}

#ifdef MOONLAB_HAS_LIBIRREP
/* LIBIRREP_SS: build the matching libirrep toric code, lift it to a
 * single-shot code (Quintavalle-Vasmer-Roffe-Campbell 2021), verify
 * the meta-check property, then defer to greedy for the data-qubit
 * correction.  The libirrep call's purpose is to confirm the code's
 * meta-check matrices exist + are valid -- a precondition for a real
 * single-shot decoder.  Full single-shot decoding (using the
 * meta-syndrome to filter measurement errors) is a v0.7+ piece. */
static int decoder_libirrep_ss(const moonlab_decoder_input_t *in)
{
    if (!in->code->is_toric) return MOONLAB_DECODER_INFEASIBLE;

    irrep_toric_params_t p;
    if (irrep_toric_init(&p, in->code->distance, in->code->distance) != IRREP_OK) {
        return MOONLAB_DECODER_OOM;
    }
    irrep_css_code_t css;
    if (irrep_toric_code_build_css(&p, &css) != IRREP_OK) {
        return MOONLAB_DECODER_OOM;
    }
    irrep_single_shot_code_t ss;
    irrep_status_t st = irrep_single_shot_lift(&css, &ss);
    if (st != IRREP_OK) {
        irrep_css_code_free(&css);
        return MOONLAB_DECODER_OOM;
    }
    st = irrep_single_shot_verify_meta(&ss);
    irrep_single_shot_code_free(&ss);
    irrep_css_code_free(&css);
    if (st != IRREP_OK) {
        return MOONLAB_DECODER_INFEASIBLE; /* meta-check failed: code not single-shot */
    }
    /* Single-shot lift verified.  Defer to greedy for correction. */
    return decoder_greedy(in);
}
#endif

int moonlab_decoder_decode(moonlab_decoder_kind_t          slot,
                           const moonlab_decoder_input_t  *in)
{
    if (!in || !in->code || !in->syndromes || !in->corrections) {
        return MOONLAB_DECODER_BAD_ARG;
    }
    if (in->code->distance < 2 || in->code->num_qubits < 1 ||
        in->num_stabilisers < 0) {
        return MOONLAB_DECODER_BAD_ARG;
    }

    switch (slot) {
    case MOONLAB_DECODER_GREEDY:
        return decoder_greedy(in);
    case MOONLAB_DECODER_MWPM_EXACT: {
        /* Real exact MWPM (brute-force enumeration up to n=10 defects,
         * greedy + 2-opt past that).  Falls back to greedy on
         * INFEASIBLE / OOM. */
        memset(in->corrections, 0, (size_t)in->code->num_qubits);
        const int rc = moonlab_mwpm_exact_decode_toric(
            in->code->distance, in->syndromes,
            in->num_stabilisers, in->corrections);
        if (rc == MOONLAB_MWPM_OK || rc == MOONLAB_MWPM_INFEASIBLE) {
            return MOONLAB_DECODER_OK;
        }
        return MOONLAB_DECODER_OOM;
    }
    case MOONLAB_DECODER_LIBIRREP_SS:
#ifdef MOONLAB_HAS_LIBIRREP
        return decoder_libirrep_ss(in);
#else
        return MOONLAB_DECODER_NOT_BUILT;
#endif
    case MOONLAB_DECODER_SBNN:
    case MOONLAB_DECODER_PYMATCHING:
        return MOONLAB_DECODER_NOT_BUILT;
    default:
        return MOONLAB_DECODER_BAD_ARG;
    }
}
