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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
    case MOONLAB_DECODER_SBNN:
    case MOONLAB_DECODER_LIBIRREP_SS:
    case MOONLAB_DECODER_PYMATCHING:
        return 0; /* v0.6.8 flips these. */
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
    case MOONLAB_DECODER_MWPM_EXACT: /* shares GREEDY for now; v0.6.8 separates */
        return decoder_greedy(in);
    case MOONLAB_DECODER_SBNN:
    case MOONLAB_DECODER_LIBIRREP_SS:
    case MOONLAB_DECODER_PYMATCHING:
        return MOONLAB_DECODER_NOT_BUILT;
    default:
        return MOONLAB_DECODER_BAD_ARG;
    }
}
