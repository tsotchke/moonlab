/**
 * @file    mwpm_exact.c
 * @brief   Exact MWPM decoder for the toric code.
 *
 * Lifted from `examples/applications/surface_code_threshold.c`
 * (the `bf_match` + `greedy_2opt_match` + `apply_correction_path`
 * functions) into a stable library entry point.
 */

#include "mwpm_exact.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int a, b; } point_t;

static int torus_dist(point_t p, point_t q, int d)
{
    int da = p.a - q.a; if (da < 0) da = -da;
    int db = p.b - q.b; if (db < 0) db = -db;
    if (d - da < da) da = d - da;
    if (d - db < db) db = d - db;
    return da + db;
}

static int collect_defects(int d, const unsigned char *syndromes, point_t *out)
{
    int n = 0;
    for (int a = 0; a < d; a++) {
        for (int b = 0; b < d; b++) {
            if (syndromes[a * d + b]) {
                out[n].a = a; out[n].b = b; n++;
            }
        }
    }
    return n;
}

static inline int h_idx(int d, int a, int b) { return a * d + b; }
static inline int v_idx(int d, int a, int b) { return d * d + a * d + b; }

/* Apply the geodesic correction path between two paired defects.
 *
 * Edge convention (matches compute_syndrome in the toric harness):
 *   h(a, b) at index `a*d + b`           connects vertex (a, b) to (a+1, b)
 *                                         -- horizontal-edge / along +a axis
 *   v(a, b) at index `d*d + a*d + b`     connects vertex (a, b) to (a, b+1)
 *                                         -- vertical-edge   / along +b axis
 *
 * Walking the defect from p to q: a-steps cross h-edges, b-steps
 * cross v-edges.  An earlier version of this function had the index
 * families swapped (h-edges flipped during b-steps), which left
 * residual syndrome and inflated logical-error rates across the
 * shoot-out.  Fix paired with the same-shape correction in
 * decoder_bench.c torus_edge_between. */
static void apply_path(int d, point_t p, point_t q, unsigned char *corr)
{
    int db_fwd = ((q.b - p.b) % d + d) % d;
    int db_bwd = ((p.b - q.b) % d + d) % d;
    int db_steps, db_dir;
    if (db_fwd <= db_bwd) { db_steps = db_fwd; db_dir = +1; }
    else                  { db_steps = db_bwd; db_dir = -1; }

    int da_fwd = ((q.a - p.a) % d + d) % d;
    int da_bwd = ((p.a - q.a) % d + d) % d;
    int da_steps, da_dir;
    if (da_fwd <= da_bwd) { da_steps = da_fwd; da_dir = +1; }
    else                  { da_steps = da_bwd; da_dir = -1; }

    int b = p.b;
    for (int s = 0; s < db_steps; s++) {
        int b_edge = (db_dir == +1) ? b : ((b + d - 1) % d);
        corr[v_idx(d, p.a, b_edge)] ^= 1;   /* b-step crosses v-edge */
        b = (b + db_dir + d) % d;
    }
    int a = p.a;
    for (int s = 0; s < da_steps; s++) {
        int a_edge = (da_dir == +1) ? a : ((a + d - 1) % d);
        corr[h_idx(d, a_edge, q.b)] ^= 1;   /* a-step crosses h-edge */
        a = (a + da_dir + d) % d;
    }
}

/* Brute-force MWPM by recursion: pair the smallest-index unused
 * defect with each higher unused index, recurse. */
typedef struct {
    int      n;
    point_t *defs;
    int      d;
    int     *used;
    int     *curr;
    int     *best;
    int      best_w;
} bf_ctx_t;

static void bf_recurse(bf_ctx_t *ctx, int depth, int total_w)
{
    if (depth * 2 == ctx->n) {
        if (total_w < ctx->best_w) {
            ctx->best_w = total_w;
            memcpy(ctx->best, ctx->curr, sizeof(int) * (size_t)ctx->n);
        }
        return;
    }
    int pi = -1;
    for (int k = 0; k < ctx->n; k++) {
        if (!ctx->used[k]) { pi = k; break; }
    }
    if (pi < 0) return;
    ctx->used[pi] = 1;
    for (int pj = pi + 1; pj < ctx->n; pj++) {
        if (ctx->used[pj]) continue;
        ctx->used[pj] = 1;
        const int w = torus_dist(ctx->defs[pi], ctx->defs[pj], ctx->d);
        ctx->curr[2 * depth]     = pi;
        ctx->curr[2 * depth + 1] = pj;
        if (total_w + w < ctx->best_w) {
            bf_recurse(ctx, depth + 1, total_w + w);
        }
        ctx->used[pj] = 0;
    }
    ctx->used[pi] = 0;
}

static int bf_match(int n, point_t *defs, int d, int *pairs)
{
    if (n == 0) return 0;
    bf_ctx_t ctx;
    ctx.n = n; ctx.defs = defs; ctx.d = d;
    ctx.used = (int *)calloc((size_t)n, sizeof(int));
    ctx.curr = (int *)calloc((size_t)n, sizeof(int));
    ctx.best = (int *)calloc((size_t)n, sizeof(int));
    if (!ctx.used || !ctx.curr || !ctx.best) {
        free(ctx.used); free(ctx.curr); free(ctx.best);
        return -1;
    }
    ctx.best_w = INT32_MAX;
    bf_recurse(&ctx, 0, 0);
    if (ctx.best_w == INT32_MAX) {
        memset(pairs, 0, sizeof(int) * (size_t)n);
        ctx.best_w = 0;
    } else {
        memcpy(pairs, ctx.best, sizeof(int) * (size_t)n);
    }
    const int rv = ctx.best_w;
    free(ctx.used); free(ctx.curr); free(ctx.best);
    return rv;
}

static int greedy_2opt_match(int n, point_t *defs, int d, int *pairs)
{
    int *used = (int *)calloc((size_t)n, sizeof(int));
    if (!used) return -1;
    int total = 0;
    int filled = 0;
    while (filled < n) {
        int best_i = -1, best_j = -1, best_w = INT32_MAX;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (used[j]) continue;
                const int w = torus_dist(defs[i], defs[j], d);
                if (w < best_w) { best_w = w; best_i = i; best_j = j; }
            }
        }
        if (best_i < 0) break;
        used[best_i] = 1; used[best_j] = 1;
        pairs[filled]     = best_i;
        pairs[filled + 1] = best_j;
        total += best_w;
        filled += 2;
    }
    free(used);

    /* 2-opt. */
    int max_iters = 100;
    int improved = 1;
    while (improved && max_iters-- > 0) {
        improved = 0;
        for (int ip = 0; ip < n; ip += 2) {
            for (int jp = ip + 2; jp < n; jp += 2) {
                const int a0 = pairs[ip],     a1 = pairs[ip + 1];
                const int b0 = pairs[jp],     b1 = pairs[jp + 1];
                const int w_orig =
                    torus_dist(defs[a0], defs[a1], d) +
                    torus_dist(defs[b0], defs[b1], d);
                const int w_swap1 =
                    torus_dist(defs[a0], defs[b0], d) +
                    torus_dist(defs[a1], defs[b1], d);
                const int w_swap2 =
                    torus_dist(defs[a0], defs[b1], d) +
                    torus_dist(defs[a1], defs[b0], d);
                const int best_alt = (w_swap1 <= w_swap2) ? 1 : 2;
                const int best_alt_w = (w_swap1 <= w_swap2) ? w_swap1 : w_swap2;
                if (best_alt_w < w_orig) {
                    if (best_alt == 1) {
                        pairs[ip + 1] = b0;
                        pairs[jp]     = a1;
                    } else {
                        pairs[ip + 1] = b1;
                        pairs[jp]     = a1;
                        pairs[jp + 1] = b0;
                    }
                    total += (best_alt_w - w_orig);
                    improved = 1;
                }
            }
        }
    }
    return total;
}

int moonlab_mwpm_exact_decode_toric(int distance,
                                    const unsigned char *syndromes,
                                    int num_stabs,
                                    unsigned char *corrections)
{
    if (distance < 2 || !syndromes || !corrections)
        return MOONLAB_MWPM_BAD_ARG;
    if (num_stabs != distance * distance)
        return MOONLAB_MWPM_BAD_ARG;

    const int dd = distance * distance;
    point_t *defects = (point_t *)malloc((size_t)dd * sizeof(point_t));
    if (!defects) return MOONLAB_MWPM_OOM;

    const int n = collect_defects(distance, syndromes, defects);
    /* Closed surface: defect count must be even. */
    if ((n & 1) != 0) { free(defects); return MOONLAB_MWPM_INFEASIBLE; }

    if (n == 0) { free(defects); return MOONLAB_MWPM_OK; }

    int *pairs = (int *)malloc((size_t)n * sizeof(int));
    if (!pairs) { free(defects); return MOONLAB_MWPM_OOM; }

    int rc = MOONLAB_MWPM_OK;
    if (n <= MOONLAB_MWPM_BRUTE_FORCE_MAX) {
        if (bf_match(n, defects, distance, pairs) < 0) rc = MOONLAB_MWPM_OOM;
    } else {
        if (greedy_2opt_match(n, defects, distance, pairs) < 0) rc = MOONLAB_MWPM_OOM;
    }

    if (rc == MOONLAB_MWPM_OK) {
        for (int k = 0; k < n; k += 2) {
            apply_path(distance, defects[pairs[k]], defects[pairs[k + 1]],
                       corrections);
        }
    }
    free(pairs);
    free(defects);
    return rc;
}
