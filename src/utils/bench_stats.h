/**
 * @file bench_stats.h
 * @brief Shared helpers for benchmark-timing distributions.
 *
 * Benches in @c tests/performance/ historically reported a single
 * mean over a small inner loop.  That is not enough for release
 * claims: a reviewer asking "what's the variance on this number"
 * has no answer.  This header lets a bench collect N outer-loop
 * samples and report mean, stddev, min, max in one place, and
 * splice the distribution into the reproducibility manifest.
 *
 * Usage:
 *   const int N = bench_stats_n_runs(5);       // MOONLAB_BENCH_N or 5
 *   double samples[64];
 *   for (int i = 0; i < N; i++) {
 *       double t0 = now_us();
 *       // ... run one replica ...
 *       samples[i] = now_us() - t0;
 *   }
 *   bench_stats_t s = bench_stats_compute(samples, N);
 *   printf("%.1f us +/- %.1f  (min %.1f, max %.1f, n=%d)\n",
 *          s.mean_us, s.stddev_us, s.min_us, s.max_us, s.n_runs);
 *
 * The function is header-only so benches don't need a libquantumsim
 * re-link to pick it up.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_BENCH_STATS_H
#define MOONLAB_BENCH_STATS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double mean_us;
    double stddev_us;
    double min_us;
    double max_us;
    double rel_stddev;  /* stddev / mean; 0 on zero mean */
    int    n_runs;
} bench_stats_t;

static inline bench_stats_t bench_stats_compute(const double *samples,
                                                 int n) {
    bench_stats_t s = (bench_stats_t){0};
    if (n <= 0 || !samples) return s;
    double sum = 0.0, sum_sq = 0.0;
    double mn = samples[0], mx = samples[0];
    for (int i = 0; i < n; i++) {
        const double v = samples[i];
        sum += v;
        sum_sq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const double mean = sum / (double)n;
    double var = (sum_sq / (double)n) - mean * mean;
    if (var < 0.0) var = 0.0;
    const double sd = sqrt(var);
    s.mean_us    = mean;
    s.stddev_us  = sd;
    s.min_us     = mn;
    s.max_us     = mx;
    s.rel_stddev = (mean > 0.0) ? (sd / mean) : 0.0;
    s.n_runs     = n;
    return s;
}

/**
 * @brief Return the caller-requested number of replicas, respecting
 *        the MOONLAB_BENCH_N env var override.  Clamps to [1, 1024].
 */
static inline int bench_stats_n_runs(int default_n) {
    const char *env = getenv("MOONLAB_BENCH_N");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0 && v <= 1024) return v;
    }
    if (default_n < 1)    default_n = 1;
    if (default_n > 1024) default_n = 1024;
    return default_n;
}

/**
 * @brief Pretty-print a single bench_stats_t line.  Does NOT add a
 *        trailing newline.
 */
static inline void bench_stats_print(const bench_stats_t *s, FILE *out) {
    if (!s || !out) return;
    fprintf(out,
            "%.2f us +/- %.2f (%.1f%%)  [min %.2f  max %.2f  n=%d]",
            s->mean_us, s->stddev_us, s->rel_stddev * 100.0,
            s->min_us, s->max_us, s->n_runs);
}

/**
 * @brief Format a bench_stats_t as a JSON object suitable to splice
 *        into a reproducibility-manifest metrics fragment.  Writes
 *        at most @p cap bytes into @p buf; returns number of bytes
 *        that *would* have been written (snprintf semantics).
 */
static inline int bench_stats_to_json(const bench_stats_t *s,
                                       char *buf, size_t cap) {
    if (!s || !buf || cap == 0) return 0;
    return snprintf(buf, cap,
        "{\"n\":%d,\"mean_us\":%.6f,\"stddev_us\":%.6f,"
        "\"rel_stddev\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f}",
        s->n_runs, s->mean_us, s->stddev_us,
        s->rel_stddev, s->min_us, s->max_us);
}

#endif /* MOONLAB_BENCH_STATS_H */
