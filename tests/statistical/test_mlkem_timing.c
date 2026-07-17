/**
 * @file test_mlkem_timing.c
 * @brief Statistical timing-variance harness for ML-KEM-512 decaps.
 *
 * A dudect-style class-separation measurement over the FO decapsulation path:
 *
 *   - class 0: decaps of a VALID ciphertext   (FO re-encrypt comparison hits)
 *   - class 1: decaps of an INVALID ciphertext (FO re-encrypt comparison misses,
 *              implicit-rejection branch taken)
 *
 * The two classes exercise the two sides of the constant-time FO ciphertext
 * comparison inside decaps.  We interleave the two measurements rep-by-rep to
 * cancel slow clock drift, trim scheduling outliers, then compute Welch's t
 * between the trimmed per-class timing distributions.  |t| >~ 10 is the
 * conventional dudect boundary for a detectable class difference.
 *
 * This is a best-effort measurement.  On noisy shared CI hardware it is
 * REPORTED but NON-GATING (gating=0): a hosted runner's jitter easily
 * manufactures a spurious t.  It becomes a hard gate only when
 * QSIM_TIMING_STRICT=1 is set for a controlled, quiet host.  That distinction
 * is emitted in the JSONL so nobody mistakes an informational run for a
 * constant-time proof.
 */

#include "stat_common.h"
#include "../../src/crypto/mlkem/mlkem.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Trimmed mean and (sample) variance over [lo, hi) of a sorted array. */
static void trimmed_stats(const double *sorted, size_t lo, size_t hi,
                          double *mean, double *var) {
    size_t n = hi - lo;
    double s = 0.0;
    for (size_t i = lo; i < hi; i++) s += sorted[i];
    double m = s / (double)n;
    double v = 0.0;
    for (size_t i = lo; i < hi; i++) { double d = sorted[i] - m; v += d * d; }
    v /= (double)(n > 1 ? n - 1 : 1);
    *mean = m;
    *var = v;
}

static size_t env_size(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || v == 0) return def;
    return (size_t)v;
}

int main(void) {
    int strict = 0;
    {
        const char *s = getenv("QSIM_TIMING_STRICT");
        strict = (s && s[0] && s[0] != '0');
    }
    size_t reps = env_size("QSIM_TIMING_REPS", 20000);
    const double T_THRESHOLD = 10.0;   /* dudect leakage boundary */

    printf("ML-KEM-512 constant-time variance harness\n");
    printf("  reps=%zu strict=%d (gating=%d)\n", reps, strict, strict);

    uint8_t d[32], z[32], m[32];
    memset(d, 0x44, sizeof(d));
    memset(z, 0x55, sizeof(z));
    memset(m, 0x66, sizeof(m));

    uint8_t ek[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk[MLKEM512_SECRETKEYBYTES];
    uint8_t c_valid[MLKEM512_CIPHERTEXTBYTES];
    uint8_t c_invalid[MLKEM512_CIPHERTEXTBYTES];
    uint8_t K[32], Kss[32];

    moonlab_mlkem512_keygen(ek, dk, d, z);
    moonlab_mlkem512_encaps(c_valid, K, ek, m);
    memcpy(c_invalid, c_valid, sizeof(c_valid));
    c_invalid[0] ^= 0x01;   /* one-bit-invalid ciphertext */

    double *t0 = (double *)malloc(reps * sizeof(double));
    double *t1 = (double *)malloc(reps * sizeof(double));
    if (!t0 || !t1) { fprintf(stderr, "alloc failure\n"); free(t0); free(t1); return 2; }

    /* Warmup so caches / branch predictors reach steady state. */
    for (int w = 0; w < 2000; w++) {
        moonlab_mlkem512_decaps(Kss, c_valid, dk);
        moonlab_mlkem512_decaps(Kss, c_invalid, dk);
    }

    /* Interleaved measurement cancels slow drift between classes. */
    for (size_t i = 0; i < reps; i++) {
        double a = now_ns();
        moonlab_mlkem512_decaps(Kss, c_valid, dk);
        double b = now_ns();
        moonlab_mlkem512_decaps(Kss, c_invalid, dk);
        double c = now_ns();
        t0[i] = b - a;
        t1[i] = c - b;
    }

    qsort(t0, reps, sizeof(double), cmp_double);
    qsort(t1, reps, sizeof(double), cmp_double);

    /* Drop the fastest 5% and slowest 10% (OS scheduling / migration). */
    size_t lo = (size_t)(reps * 0.05);
    size_t hi = (size_t)(reps * 0.90);
    if (hi <= lo) { lo = 0; hi = reps; }

    double m0, v0, m1, v1;
    trimmed_stats(t0, lo, hi, &m0, &v0);
    trimmed_stats(t1, lo, hi, &m1, &v1);
    size_t n = hi - lo;

    double se = sqrt(v0 / (double)n + v1 / (double)n);
    double t = (se > 0.0) ? (m0 - m1) / se : 0.0;
    double abst = fabs(t);

    printf("  class0 valid   mean=%.1f ns var=%.1f\n", m0, v0);
    printf("  class1 invalid mean=%.1f ns var=%.1f\n", m1, v1);
    printf("  Welch t=%.3f (|t| threshold=%.1f, n_per_class=%zu)\n",
           t, T_THRESHOLD, n);

    int pass = (abst < T_THRESHOLD);

    char stats[512];
    snprintf(stats, sizeof(stats),
        "{\"welch_t\":%.4f,\"abs_t\":%.4f,\"threshold\":%.1f,"
        "\"mean_valid_ns\":%.2f,\"mean_invalid_ns\":%.2f,"
        "\"n_per_class\":%zu,\"strict\":%d,\"note\":\"%s\"}",
        t, abst, T_THRESHOLD, m0, m1, n, strict,
        strict ? "gating on controlled host"
               : "informational on shared CI (non-gating)");
    stat_emit_result("constant_time_variance", pass, strict ? 1 : 0, stats);

    free(t0);
    free(t1);

    if (strict && !pass) {
        fprintf(stderr, "FAIL: constant-time class separation |t|=%.2f >= %.1f\n",
                abst, T_THRESHOLD);
        return 1;
    }
    return 0;
}
