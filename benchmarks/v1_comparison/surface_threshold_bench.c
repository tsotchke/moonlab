/**
 * @file surface_threshold_bench.c
 * @brief v1.0 head-to-head: Moonlab surface-code threshold sweep vs Stim.
 *
 * Toric-code (d x d torus, 2*d*d data qubits on edges) under code-capacity
 * X-only bit-flip noise.  Each shot:
 *   1. Sample i.i.d. X-error pattern at rate p.
 *   2. Compute the Z-star syndrome.
 *   3. Decode via minimum-weight perfect matching: brute-force enumeration
 *      for <=10 defects, greedy + 2-opt for larger defect populations.
 *   4. Apply the geodesic correction; declare a logical failure if the
 *      residual X pattern has odd parity on either of the two toric
 *      logical-Z chains.
 *
 * Sweep:  d in {3, 5, 7}, p in {0.001, 0.003, 0.01, 0.03, 0.1},
 *         n_shots = 10000 (override with --shots).
 *
 * Output: JSON, schema "moonlab/v1_comparison/surface_threshold".  Records
 * carry { d, p, shots, fails, p_logical, std_error, wall_clock_s,
 * peak_rss_bytes }.  The competitor (Stim + PyMatching) side runs
 * separately; see docs/benchmarks/v1_comparison.md for the exact
 * `python bench/stim_threshold.py` invocation.
 *
 * The core decoder and geometry routines are lifted verbatim from
 * examples/applications/surface_code_threshold.c so the moonlab number
 * here is identical to the production example's at the matched (d, p)
 * grid points -- this file only changes the sweep grid and the JSON
 * envelope.
 *
 * @since v1.0
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <limits.h>

/* ------------------------------------------------------------------ */
/*  PCG-XSH-RR 64 -> 32 PRNG (deterministic per (d, p) seed).         */
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

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    uint64_t r = (uint64_t)ru.ru_maxrss;
#if defined(__APPLE__)
    return r;
#else
    return r * 1024ULL;
#endif
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

typedef struct { int a, b; } point_t;

static int torus_dist(point_t p, point_t q, int d) {
    int da = p.a - q.a; if (da < 0) da = -da;
    int db = p.b - q.b; if (db < 0) db = -db;
    if (d - da < da) da = d - da;
    if (d - db < db) db = d - db;
    return da + db;
}

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

typedef struct {
    int n; point_t* defs; int d;
    int* used; int* curr; int* best; int  best_w;
} bf_ctx_t;

static void bf_recurse(bf_ctx_t* ctx, int depth, int total_w) {
    if (depth * 2 == ctx->n) {
        if (total_w < ctx->best_w) {
            ctx->best_w = total_w;
            memcpy(ctx->best, ctx->curr, sizeof(int) * (size_t)ctx->n);
        }
        return;
    }
    int pi = -1;
    for (int k = 0; k < ctx->n; k++) if (!ctx->used[k]) { pi = k; break; }
    if (pi < 0) return;
    ctx->used[pi] = 1;
    for (int pj = pi + 1; pj < ctx->n; pj++) {
        if (ctx->used[pj]) continue;
        ctx->used[pj] = 1;
        int w = torus_dist(ctx->defs[pi], ctx->defs[pj], ctx->d);
        ctx->curr[2 * depth]     = pi;
        ctx->curr[2 * depth + 1] = pj;
        if (total_w + w < ctx->best_w) bf_recurse(ctx, depth + 1, total_w + w);
        ctx->used[pj] = 0;
    }
    ctx->used[pi] = 0;
}

static int bf_match(int n, point_t* defs, int d, int* pairs) {
    if (n == 0) return 0;
    bf_ctx_t ctx;
    ctx.n = n; ctx.defs = defs; ctx.d = d;
    ctx.used = (int*)calloc((size_t)n, sizeof(int));
    ctx.curr = (int*)calloc((size_t)n, sizeof(int));
    ctx.best = (int*)calloc((size_t)n, sizeof(int));
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

static int greedy_2opt_match(int n, point_t* defs, int d, int* pairs) {
    int* used = (int*)calloc((size_t)n, sizeof(int));
    int total = 0, filled = 0;
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
        pairs[filled] = best_i; pairs[filled + 1] = best_j;
        total += best_w; filled += 2;
    }
    free(used);
    int max_iters = 100, improved = 1;
    while (improved && max_iters-- > 0) {
        improved = 0;
        for (int ip = 0; ip < n; ip += 2) {
            for (int jp = ip + 2; jp < n; jp += 2) {
                int a0 = pairs[ip], a1 = pairs[ip + 1];
                int b0 = pairs[jp], b1 = pairs[jp + 1];
                int w_orig =
                    torus_dist(defs[a0], defs[a1], d) +
                    torus_dist(defs[b0], defs[b1], d);
                int w_s1 = torus_dist(defs[a0], defs[b0], d) +
                           torus_dist(defs[a1], defs[b1], d);
                int w_s2 = torus_dist(defs[a0], defs[b1], d) +
                           torus_dist(defs[a1], defs[b0], d);
                int alt   = (w_s1 <= w_s2) ? 1 : 2;
                int alt_w = (w_s1 <= w_s2) ? w_s1 : w_s2;
                if (alt_w < w_orig) {
                    if (alt == 1) {
                        pairs[ip + 1] = b0; pairs[jp] = a1;
                    } else {
                        pairs[ip + 1] = b1; pairs[jp] = a1; pairs[jp + 1] = b0;
                    }
                    total += (alt_w - w_orig);
                    improved = 1;
                }
            }
        }
    }
    return total;
}

static int logical_failure_X1(const toric_t* t, const uint8_t* x_err) {
    int parity = 0;
    for (uint32_t a = 0; a < t->d; a++) parity ^= x_err[h_idx(t, a, 0)];
    return parity & 1;
}

static int logical_failure_X2(const toric_t* t, const uint8_t* x_err) {
    int parity = 0;
    for (uint32_t b = 0; b < t->d; b++) parity ^= x_err[v_idx(t, 0, b)];
    return parity & 1;
}

static int mc_trial(const toric_t* t, double p, pcg32_t* rng,
                     uint8_t* x_err, uint8_t* z_synd,
                     point_t* defs, int* pairs) {
    for (uint32_t e = 0; e < t->n_edges; e++)
        x_err[e] = (pcg32_uniform(rng) < p) ? 1 : 0;
    compute_z_syndrome(t, x_err, z_synd);
    int n_def = collect_defects(t, z_synd, defs);
    if (n_def > 0) {
        if (n_def <= 10) (void)bf_match(n_def, defs, (int)t->d, pairs);
        else             (void)greedy_2opt_match(n_def, defs, (int)t->d, pairs);
        for (int k = 0; k < n_def; k += 2)
            apply_correction_path(t, defs[pairs[k]], defs[pairs[k + 1]], x_err);
    }
    return logical_failure_X1(t, x_err) | logical_failure_X2(t, x_err);
}

/* ------------------------------------------------------------------ */
/*  Sweep + JSON                                                      */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    const char* out_path = "surface_threshold.json";
    uint32_t shots = 10000;
    /* Per the v1 protocol. */
    const uint32_t distances[] = {3, 5, 7};
    const double   p_values[]  = {0.001, 0.003, 0.01, 0.03, 0.1};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--shots") == 0 && i + 1 < argc) {
            long v = strtol(argv[i + 1], NULL, 10);
            if (v < 1 || v > 100000000) {
                fprintf(stderr, "shots must be in [1, 1e8]\n"); return 2;
            }
            shots = (uint32_t)v;
            i++;
        } else if (argv[i][0] != '-') {
            out_path = argv[i];
        } else {
            fprintf(stderr,
                    "usage: %s [out.json] [--shots 10000]\n", argv[0]);
            return 2;
        }
    }

    const size_t n_d = sizeof(distances) / sizeof(distances[0]);
    const size_t n_p = sizeof(p_values)  / sizeof(p_values[0]);

    fprintf(stdout,
            "=== v1.0 head-to-head: surface-code threshold vs Stim ===\n"
            "    schema: moonlab/v1_comparison/surface_threshold\n"
            "    model: toric code, X-only bit-flip, MWPM decoder\n"
            "    shots per (d, p): %u\n\n", shots);

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/v1_comparison/surface_threshold\",\n");
    fprintf(f, "  \"code\": \"toric\",\n");
    fprintf(f, "  \"noise\": \"X-only bit-flip, i.i.d.\",\n");
    fprintf(f, "  \"decoder\": \"MWPM (brute-force <=10 defects, greedy+2opt otherwise)\",\n");
    fprintf(f, "  \"distances\": [");
    for (size_t i = 0; i < n_d; i++) fprintf(f, "%s%u", i ? ", " : "", distances[i]);
    fprintf(f, "],\n  \"p_values\": [");
    for (size_t i = 0; i < n_p; i++) fprintf(f, "%s%.4f", i ? ", " : "", p_values[i]);
    fprintf(f, "],\n  \"shots_per_point\": %u,\n", shots);
    fprintf(f, "  \"reference_threshold\": 0.103,\n");
    fprintf(f, "  \"reference_source\": \"Dennis-Kitaev-Landahl-Preskill, "
               "arXiv:quant-ph/0110143 (2002)\",\n");
    fprintf(f, "  \"runs\": [");

    fprintf(stdout, "  %-4s %-8s %-12s %-12s %-12s %-12s\n",
            "d", "p", "p_logical", "std_err", "wall_s", "peak_rss_MB");

    int first = 1;
    for (size_t di = 0; di < n_d; di++) {
        uint32_t d = distances[di];
        toric_t t; toric_init(&t, d);
        uint8_t* x_err  = (uint8_t*)calloc(t.n_edges, 1);
        uint8_t* z_synd = (uint8_t*)calloc(t.n_verts, 1);
        point_t* defs   = (point_t*)calloc(t.n_verts, sizeof(point_t));
        int* pairs      = (int*)calloc(t.n_verts, sizeof(int));

        for (size_t pi = 0; pi < n_p; pi++) {
            double p = p_values[pi];
            pcg32_t rng;
            pcg32_seed(&rng, 0x4d6f6f6e6c6162ULL ^ (uint64_t)d,
                       (uint64_t)(pi + 1));
            double t0 = now_s();
            uint32_t fails = 0;
            for (uint32_t k = 0; k < shots; k++)
                fails += (uint32_t)mc_trial(&t, p, &rng, x_err, z_synd,
                                             defs, pairs);
            double dt = now_s() - t0;
            double pl  = (double)fails / (double)shots;
            double se  = sqrt(pl * (1.0 - pl) / (double)shots);
            uint64_t rss = peak_rss_bytes();

            fprintf(stdout, "  %-4u %-8.4f %-12.6f %-12.6f %-12.3f %-12.2f\n",
                    d, p, pl, se, dt, (double)rss / (1024.0 * 1024.0));
            fprintf(f, "%s\n    {\"d\": %u, \"p\": %.6f, \"shots\": %u, "
                       "\"fails\": %u, \"p_logical\": %.10g, "
                       "\"std_error\": %.6g, \"wall_clock_s\": %.4f, "
                       "\"peak_rss_bytes\": %llu}",
                    first ? "" : ",", d, p, shots, fails, pl, se, dt,
                    (unsigned long long)rss);
            first = 0;
        }
        free(x_err); free(z_synd); free(defs); free(pairs);
        fputc('\n', stdout);
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stdout, "wrote %s\n", out_path);
    return 0;
}
