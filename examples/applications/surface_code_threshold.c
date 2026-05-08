/**
 * @file surface_code_threshold.c
 * @brief Code-capacity threshold sweep harness for the toric code.
 *
 * Implements the toric code at distances d ∈ {3, 5, 7, 9} under
 * code-capacity X-only bit-flip noise.  Threshold for this model is
 * p_th ≈ 0.103 (Dennis-Kitaev-Landahl-Preskill, 2002,
 * arXiv:quant-ph/0110143).  This is the appropriate literature anchor
 * for the harness's noise model; the v1 file used a stripped-down
 * planar code with interior-only plaquettes whose boundary structure
 * broke the matcher and produced anti-threshold curves.  v2 (this
 * file) restores threshold behaviour by using the toric code (no
 * boundaries) plus correct minimum-weight perfect matching.
 *
 * Geometry: d × d torus of vertices, 2d² data qubits on edges.
 * Horizontal edge h(a,b) connects (a,b) → (a,(b+1) mod d); vertical
 * edge v(a,b) connects (a,b) → ((a+1) mod d, b).  Z-stars at every
 * vertex have weight 4 on the four incident edges.  Logical Z₁ = Z on
 * h(0, ·) (horizontal-row loop); logical Z₂ = Z on v(·, 0) (vertical-
 * column loop); together with X₁ = X on v(·, 0) and X₂ = X on h(0, ·)
 * they form the two logical qubits of the toric code.  We track only
 * X-errors and only the Z-syndrome.
 *
 * Two decoders are emitted to JSON:
 *   - "no_correction": no decoding; logical-Z parity of the raw
 *     X-error pattern is the failure indicator.
 *   - "mwpm_optimal": minimum-weight perfect matching by exact
 *     enumeration for ≤ 10 defects, greedy + 2-opt local search for
 *     larger.  On the torus, defects always come in pairs so the
 *     matching is always perfect.
 *
 * Output: JSON, schema "moonlab/surface_code_threshold_v2".
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  PCG-XSH-RR 64 -> 32 PRNG.                                          */
/* ------------------------------------------------------------------ */

typedef struct { uint64_t state; uint64_t inc; } pcg32_t;

static uint32_t pcg32_next(pcg32_t* r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + (r->inc | 1);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_uniform(pcg32_t* r) {
    return (pcg32_next(r) >> 8) / (double)(1u << 24);
}

static void pcg32_seed(pcg32_t* r, uint64_t seed, uint64_t stream) {
    r->state = 0;
    r->inc = (stream << 1u) | 1u;
    pcg32_next(r);
    r->state += seed;
    pcg32_next(r);
}

/* ------------------------------------------------------------------ */
/*  Toric-code geometry                                                */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t d; uint32_t n_edges; uint32_t n_verts; } toric_t;

static void toric_init(toric_t* t, uint32_t d) {
    t->d       = d;
    t->n_edges = 2u * d * d;
    t->n_verts = d * d;
}

static uint32_t h_idx(const toric_t* t, uint32_t a, uint32_t b) {
    return a * t->d + b;
}

static uint32_t v_idx(const toric_t* t, uint32_t a, uint32_t b) {
    return t->d * t->d + a * t->d + b;
}

static uint32_t vert_idx(const toric_t* t, uint32_t a, uint32_t b) {
    return a * t->d + b;
}

/* Z-syndrome at vertex (a, b) is the parity of X-errors on the four
 * incident edges: h(a, b-1), h(a, b), v(a-1, b), v(a, b)  (indices
 * mod d).  An X-error on edge h(a, b) flips Z-stars at endpoints
 * (a, b) and (a, b+1); an X-error on edge v(a, b) flips Z-stars at
 * (a, b) and (a+1, b).  Each X-error therefore flips exactly two
 * stars, so the total syndrome weight is always even. */
static void compute_z_syndrome(const toric_t* t, const uint8_t* x_err,
                                uint8_t* z_synd) {
    const uint32_t d = t->d;
    for (uint32_t a = 0; a < d; a++) {
        for (uint32_t b = 0; b < d; b++) {
            uint32_t bm = (b == 0) ? d - 1 : b - 1;
            uint32_t am = (a == 0) ? d - 1 : a - 1;
            uint8_t parity = 0;
            parity ^= x_err[h_idx(t, a, bm)];
            parity ^= x_err[h_idx(t, a, b)];
            parity ^= x_err[v_idx(t, am, b)];
            parity ^= x_err[v_idx(t, a, b)];
            z_synd[vert_idx(t, a, b)] = parity & 1;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Torus distance + geodesic correction                               */
/* ------------------------------------------------------------------ */

typedef struct { int a, b; } point_t;

static int torus_dist(point_t p, point_t q, int d) {
    int da = p.a - q.a; if (da < 0) da = -da;
    int db = p.b - q.b; if (db < 0) db = -db;
    if (d - da < da) da = d - da;
    if (d - db < db) db = d - db;
    return da + db;
}

/* Apply the geodesic correction string between two paired vertices on
 * the torus.  Move along the shorter wrap in both coordinates, first
 * along columns (b-direction, flipping horizontal edges), then along
 * rows (a-direction, flipping vertical edges).  Each step flips one
 * data qubit. */
static void apply_correction_path(const toric_t* t, point_t p, point_t q,
                                   uint8_t* err) {
    const int d = (int)t->d;
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
        err[h_idx(t, (uint32_t)p.a, (uint32_t)b_edge)] ^= 1;
        b = (b + db_dir + d) % d;
    }
    int a = p.a;
    for (int s = 0; s < da_steps; s++) {
        int a_edge = (da_dir == +1) ? a : ((a + d - 1) % d);
        err[v_idx(t, (uint32_t)a_edge, (uint32_t)q.b)] ^= 1;
        a = (a + da_dir + d) % d;
    }
}

/* ------------------------------------------------------------------ */
/*  Defect collection                                                  */
/* ------------------------------------------------------------------ */

static int collect_defects(const toric_t* t, const uint8_t* synd,
                            point_t* out) {
    int n = 0;
    for (uint32_t a = 0; a < t->d; a++) {
        for (uint32_t b = 0; b < t->d; b++) {
            if (synd[vert_idx(t, a, b)]) {
                out[n].a = (int)a; out[n].b = (int)b; n++;
            }
        }
    }
    return n;
}

/* ------------------------------------------------------------------ */
/*  Brute-force optimal MWPM via plain recursion                       */
/*  Pair smallest-index unused defect with each higher unused index;   */
/*  recurse on the rest.  (n-1)!! enumerations.  n ≤ 10 → ≤ 945.       */
/* ------------------------------------------------------------------ */

typedef struct {
    int n;
    point_t* defs;
    int d;
    int* used;
    int* curr;     /* size n; pair (curr[2i], curr[2i+1]) for i in [0, n/2) */
    int* best;
    int  best_w;
} bf_ctx_t;

static void bf_recurse(bf_ctx_t* ctx, int depth, int total_w) {
    if (depth * 2 == ctx->n) {
        if (total_w < ctx->best_w) {
            ctx->best_w = total_w;
            memcpy(ctx->best, ctx->curr, sizeof(int) * (size_t)ctx->n);
        }
        return;
    }
    /* Pick smallest unused index. */
    int pi = -1;
    for (int k = 0; k < ctx->n; k++) {
        if (!ctx->used[k]) { pi = k; break; }
    }
    if (pi < 0) return;
    ctx->used[pi] = 1;
    /* Try every higher unused index as the partner. */
    for (int pj = pi + 1; pj < ctx->n; pj++) {
        if (ctx->used[pj]) continue;
        ctx->used[pj] = 1;
        int w = torus_dist(ctx->defs[pi], ctx->defs[pj], ctx->d);
        ctx->curr[2 * depth]     = pi;
        ctx->curr[2 * depth + 1] = pj;
        if (total_w + w < ctx->best_w) {  /* simple pruning */
            bf_recurse(ctx, depth + 1, total_w + w);
        }
        ctx->used[pj] = 0;
    }
    ctx->used[pi] = 0;
}

static int bf_match(int n, point_t* defs, int d, int* pairs) {
    if (n == 0) return 0;
    bf_ctx_t ctx;
    ctx.n     = n;
    ctx.defs  = defs;
    ctx.d     = d;
    ctx.used  = (int*)calloc((size_t)n, sizeof(int));
    ctx.curr  = (int*)calloc((size_t)n, sizeof(int));
    ctx.best  = (int*)calloc((size_t)n, sizeof(int));
    ctx.best_w = INT32_MAX;
    bf_recurse(&ctx, 0, 0);
    if (ctx.best_w == INT32_MAX) {
        memset(pairs, 0, sizeof(int) * (size_t)n);
        ctx.best_w = 0;
    } else {
        memcpy(pairs, ctx.best, sizeof(int) * (size_t)n);
    }
    int rv = ctx.best_w;
    free(ctx.used); free(ctx.curr); free(ctx.best);
    return rv;
}

/* ------------------------------------------------------------------ */
/*  Greedy initial pairing + 2-opt local search                        */
/* ------------------------------------------------------------------ */

/* Greedy seed: repeatedly find the closest unmatched pair; match them.
 * O(n²) per match, O(n³) overall for the initial pairing.  Then
 * 2-opt: for every two pairs ((a, b), (c, d)) try the three possible
 * pairings and keep the minimum.  Iterate until no improvement. */
static int greedy_2opt_match(int n, point_t* defs, int d, int* pairs) {
    int* used = (int*)calloc((size_t)n, sizeof(int));
    int total = 0;
    int filled = 0;
    while (filled < n) {
        int best_i = -1, best_j = -1, best_w = INT32_MAX;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (used[j]) continue;
                int w = torus_dist(defs[i], defs[j], d);
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
                int a0 = pairs[ip],     a1 = pairs[ip + 1];
                int b0 = pairs[jp],     b1 = pairs[jp + 1];
                int w_orig =
                    torus_dist(defs[a0], defs[a1], d) +
                    torus_dist(defs[b0], defs[b1], d);
                int w_swap1 =                       /* (a0,b0)(a1,b1) */
                    torus_dist(defs[a0], defs[b0], d) +
                    torus_dist(defs[a1], defs[b1], d);
                int w_swap2 =                       /* (a0,b1)(a1,b0) */
                    torus_dist(defs[a0], defs[b1], d) +
                    torus_dist(defs[a1], defs[b0], d);
                int best_alt = (w_swap1 <= w_swap2) ? 1 : 2;
                int best_alt_w = (w_swap1 <= w_swap2) ? w_swap1 : w_swap2;
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

/* ------------------------------------------------------------------ */
/*  Logical-Z parity check                                             */
/* ------------------------------------------------------------------ */

/* The toric code has two logical qubits.  In our edge labelling:
 *   - logical X_1 = X on the b-direction lattice loop = X on
 *     {h(0, b) : b in [0, d)} (a row of horizontal edges).
 *   - logical Z_1 = Z on the dual a-direction loop = Z on
 *     {h(a, 0) : a in [0, d)} (a column of horizontal edges).
 *   - logical X_2 = X on {v(a, 0)}; logical Z_2 = Z on {v(0, b)}.
 * An X-error pattern E has a logical-X_1 component iff it
 * anticommutes with logical Z_1, i.e. iff its support has odd
 * intersection with {h(a, 0)}.  Symmetrically for X_2 / Z_2.
 * We report a logical failure if either qubit's check fails. */
static int logical_failure_X1(const toric_t* t, const uint8_t* x_err) {
    int parity = 0;
    for (uint32_t a = 0; a < t->d; a++) {
        parity ^= x_err[h_idx(t, a, 0)];
    }
    return parity & 1;
}

static int logical_failure_X2(const toric_t* t, const uint8_t* x_err) {
    int parity = 0;
    for (uint32_t b = 0; b < t->d; b++) {
        parity ^= x_err[v_idx(t, 0, b)];
    }
    return parity & 1;
}

/* ------------------------------------------------------------------ */
/*  One Monte-Carlo trial                                              */
/* ------------------------------------------------------------------ */

typedef struct { int fail_no_correction; int fail_mwpm; } trial_result_t;

static trial_result_t mc_trial(const toric_t* t, double p, pcg32_t* rng,
                                uint8_t* x_err, uint8_t* x_err_g,
                                uint8_t* z_synd, point_t* defs,
                                int* pairs) {
    for (uint32_t e = 0; e < t->n_edges; e++) {
        x_err[e] = (pcg32_uniform(rng) < p) ? 1 : 0;
    }

    trial_result_t out;
    out.fail_no_correction =
        logical_failure_X1(t, x_err) | logical_failure_X2(t, x_err);

    memcpy(x_err_g, x_err, t->n_edges);
    compute_z_syndrome(t, x_err_g, z_synd);
    int n_def = collect_defects(t, z_synd, defs);

    if (n_def > 0) {
        if (n_def <= 10) {
            (void)bf_match(n_def, defs, (int)t->d, pairs);
        } else {
            (void)greedy_2opt_match(n_def, defs, (int)t->d, pairs);
        }
        for (int k = 0; k < n_def; k += 2) {
            apply_correction_path(t, defs[pairs[k]], defs[pairs[k + 1]],
                                   x_err_g);
        }
    }

    out.fail_mwpm =
        logical_failure_X1(t, x_err_g) | logical_failure_X2(t, x_err_g);
    return out;
}

/* ------------------------------------------------------------------ */
/*  Sweep + JSON                                                       */
/* ------------------------------------------------------------------ */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "surface_code_threshold.json";
    const uint32_t trials = (argc >= 3) ? (uint32_t)strtoul(argv[2], NULL, 10)
                                         : 5000;

    fprintf(stdout,
            "=== Toric-code threshold sweep (X-only bit-flip) ===\n"
            "  schema: moonlab/surface_code_threshold_v2\n"
            "  literature anchor: p_th ≈ 0.103 (Dennis et al., 2002)\n"
            "  trials per (d, p): %u\n\n", trials);

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/surface_code_threshold_v2\",\n");
    fprintf(f, "  \"description\": \"Toric-code code-capacity logical-error "
               "rate vs i.i.d. X-only bit-flip noise p, distances "
               "{3, 5, 7, 9}.  Optimal MWPM via brute-force pairing "
               "(<=10 defects) or greedy + 2-opt (>10 defects).  Logical "
               "failure = parity of residual X-error pattern on either "
               "of the two logical-Z chains (row 0 horizontal edges or "
               "column 0 vertical edges).  Literature threshold for this "
               "model is p_th ~ 0.103 (Dennis-Kitaev-Landahl-Preskill, "
               "2002; arXiv:quant-ph/0110143).\",\n");
    fprintf(f, "  \"trials_per_point\": %u,\n", trials);
    fprintf(f, "  \"distances\": [3, 5, 7, 9],\n");
    fprintf(f, "  \"sweeps\": [");

    /* Sub-threshold to mildly super-threshold sweep. */
    const double p_values[] = {
        0.030, 0.050, 0.070, 0.085, 0.095, 0.103, 0.110, 0.125
    };
    const size_t n_p = sizeof(p_values) / sizeof(p_values[0]);
    const uint32_t distances[] = { 3, 5, 7, 9 };
    const size_t n_d = sizeof(distances) / sizeof(distances[0]);

    int first = 1;
    for (size_t di = 0; di < n_d; di++) {
        const uint32_t d = distances[di];
        toric_t t; toric_init(&t, d);
        fprintf(stdout, "  d=%u  (%u edges, %u stars)\n",
                d, t.n_edges, t.n_verts);
        fprintf(stdout, "  %-8s %-15s %-15s %-12s\n",
                "p", "p_log_uncorr", "p_log_mwpm", "wall_s");

        uint8_t* x_err   = (uint8_t*)calloc(t.n_edges, 1);
        uint8_t* x_err_g = (uint8_t*)calloc(t.n_edges, 1);
        uint8_t* z_synd  = (uint8_t*)calloc(t.n_verts, 1);
        point_t* defs    = (point_t*)calloc(t.n_verts, sizeof(point_t));
        int* pairs       = (int*)calloc(t.n_verts, sizeof(int));

        for (size_t pi = 0; pi < n_p; pi++) {
            const double p = p_values[pi];
            pcg32_t rng;
            pcg32_seed(&rng, /*seed*/ 0x4d6f6f6e6c6162ULL ^ (uint64_t)d,
                       /*stream*/ (uint64_t)(pi + 1));

            const double t0 = now_s();
            uint32_t fails_uncorr = 0, fails_mwpm = 0;
            for (uint32_t k = 0; k < trials; k++) {
                trial_result_t tr = mc_trial(&t, p, &rng,
                                              x_err, x_err_g,
                                              z_synd, defs, pairs);
                fails_uncorr += (uint32_t)tr.fail_no_correction;
                fails_mwpm   += (uint32_t)tr.fail_mwpm;
            }
            const double dt = now_s() - t0;
            const double pl_uncorr = (double)fails_uncorr / (double)trials;
            const double pl_mwpm   = (double)fails_mwpm   / (double)trials;
            const double se_uncorr =
                sqrt(pl_uncorr * (1.0 - pl_uncorr) / (double)trials);
            const double se_mwpm =
                sqrt(pl_mwpm * (1.0 - pl_mwpm) / (double)trials);

            fprintf(stdout, "  %-8.4f %-15.6f %-15.6f %-10.2f\n",
                    p, pl_uncorr, pl_mwpm, dt);

            fprintf(f, "%s\n    {\"d\": %u, \"p\": %.6f, \"trials\": %u, "
                       "\"fails_uncorr\": %u, \"fails_mwpm\": %u, "
                       "\"p_log_uncorr\": %.10g, \"p_log_mwpm\": %.10g, "
                       "\"se_uncorr\": %.6g, \"se_mwpm\": %.6g, "
                       "\"wall_s\": %.4f}",
                    first ? "" : ",", d, p, trials,
                    fails_uncorr, fails_mwpm, pl_uncorr, pl_mwpm,
                    se_uncorr, se_mwpm, dt);
            first = 0;
        }
        free(x_err); free(x_err_g); free(z_synd); free(defs); free(pairs);
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stdout, "wrote %s\n", out_path);
    fprintf(stdout, "\nThreshold: read JSON; the crossing of p_log(d, p)\n"
                    "curves between consecutive distances brackets p_th.\n"
                    "Literature anchor for this model: p_th ~ 0.103.\n");
    return 0;
}
