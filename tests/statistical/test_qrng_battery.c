/**
 * @file test_qrng_battery.c
 * @brief Adversarial statistical-quality battery for the Moonlab QRNG.
 *
 * Independent, from-scratch SP 800-22-style suite run over two surfaces:
 *
 *   1. The stable conditioned delivery path @c moonlab_qrng_bytes (the bytes
 *      real consumers get).  This MUST pass every test with wide margin.
 *
 *   2. The raw v3 GROVER-mode path (@c qrng_v3_bytes on a GROVER context),
 *      used as a POSITIVE CONTROL: Grover amplitude amplification biases the
 *      measured distribution toward a per-block target value, so a
 *      peak-symbol-frequency statistic must detect strong concentration in
 *      the raw stream and NOT in the conditioned stream.  If it does not, the
 *      battery has no teeth and that is a finding to hand to the QRNG owner.
 *
 * Plus a bounded PractRand-lite streaming check: a multi-MB stream is consumed
 * in fixed chunks while accumulating a global monobit z and a byte-frequency
 * chi-square, so memory stays flat regardless of the deep byte budget.
 *
 * Thresholds are deliberately loose (p < 1e-6 fails) so that a real CSPRNG
 * drawing fresh entropy every run passes essentially always, and only an
 * egregious defect trips a gate.  All statistics are reported regardless.
 *
 * No external dependencies; every statistic is implemented in stat_common.h
 * or here.  We intentionally do NOT link Moonlab's own nist_sp800_22 module:
 * the tester must be independent of the code under test.
 */

#include "stat_common.h"
#include "../../src/applications/moonlab_export.h"
#include "../../src/applications/qrng.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* p-value below this counts as an egregious failure.  A correct CSPRNG
 * produces p ~ U(0,1); the chance any single test dips below 1e-6 by luck is
 * negligible, so a trip means a real defect, not variance. */
#define FAIL_P 1e-6

/* Positive-control peak-symbol thresholds (mean max byte count per 256-byte
 * block).  Uniform bytes give ~4 with tight variance; the GROVER raw path
 * concentrates ~1 target value ~12x per block. */
#define PEAK_BLOCK   256
#define PEAK_THRESH  8.0
#define STREAM_ALLOC_FAILURE (-1000)

/* ------------------------------------------------------------------ */
/*  Byte-source fillers                                                 */
/* ------------------------------------------------------------------ */

static int fill_conditioned(uint8_t *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        size_t take = n - off;
        if (take > 8192) take = 8192;   /* internal path prefers chunks */
        int rc = moonlab_qrng_bytes(buf + off, take);
        if (rc != 0) return rc;
        off += take;
    }
    return 0;
}

static int fill_grover_raw(uint8_t *buf, size_t n) {
    qrng_v3_config_t cfg;
    qrng_v3_get_default_config(&cfg);
    cfg.mode = QRNG_V3_MODE_GROVER;
    cfg.enable_bell_monitoring = 0;   /* raw path: no conditioning, no gate */
    cfg.num_qubits = 8;               /* 1 byte per measurement */

    qrng_v3_ctx_t *ctx = NULL;
    if (qrng_v3_init_with_config(&ctx, &cfg) != QRNG_V3_SUCCESS || !ctx) {
        return -1;
    }
    size_t off = 0;
    while (off < n) {
        size_t take = n - off;
        if (take > 16384) take = 16384;
        if (qrng_v3_bytes(ctx, buf + off, take) != QRNG_V3_SUCCESS) {
            qrng_v3_free(ctx);
            return -1;
        }
        off += take;
    }
    qrng_v3_free(ctx);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  SP 800-22-style tests (bit-level, MSB-first)                        */
/* ------------------------------------------------------------------ */

static double test_monobit(const uint8_t *buf, size_t nbits) {
    long s = 0;
    for (size_t i = 0; i < nbits; i++) s += stat_bit(buf, i) ? 1 : -1;
    double s_obs = fabs((double)s) / sqrt((double)nbits);
    return erfc(s_obs / sqrt(2.0));
}

static double test_block_frequency(const uint8_t *buf, size_t nbits, size_t M) {
    size_t N = nbits / M;
    if (N == 0) return -1.0;
    double chi2 = 0.0;
    for (size_t b = 0; b < N; b++) {
        size_t ones = 0;
        for (size_t i = 0; i < M; i++) ones += (size_t)stat_bit(buf, b * M + i);
        double pi = (double)ones / (double)M;
        chi2 += (pi - 0.5) * (pi - 0.5);
    }
    chi2 *= 4.0 * (double)M;
    return stat_chi2_sf(chi2, (double)N);
}

/* Returns -1.0 if the monobit prerequisite is not met. */
static double test_runs(const uint8_t *buf, size_t nbits) {
    size_t ones = 0;
    for (size_t i = 0; i < nbits; i++) ones += (size_t)stat_bit(buf, i);
    double pi = (double)ones / (double)nbits;
    double tau = 2.0 / sqrt((double)nbits);
    if (fabs(pi - 0.5) >= tau) return -1.0;   /* prereq fail */
    size_t V = 1;
    for (size_t i = 1; i < nbits; i++)
        if (stat_bit(buf, i) != stat_bit(buf, i - 1)) V++;
    double num = fabs((double)V - 2.0 * (double)nbits * pi * (1.0 - pi));
    double den = 2.0 * sqrt(2.0 * (double)nbits) * pi * (1.0 - pi);
    return erfc(num / den);
}

/* Longest run of ones in 8-bit blocks (NIST M=8 classes). */
static double test_longest_run(const uint8_t *buf, size_t nbits) {
    const size_t M = 8;
    size_t N = nbits / M;
    if (N < 16) return -1.0;
    /* classes: <=1, 2, 3, >=4 */
    const double pi[4] = {0.2148, 0.3672, 0.2305, 0.1875};
    long v[4] = {0, 0, 0, 0};
    for (size_t b = 0; b < N; b++) {
        size_t longest = 0, cur = 0;
        for (size_t i = 0; i < M; i++) {
            if (stat_bit(buf, b * M + i)) { cur++; if (cur > longest) longest = cur; }
            else cur = 0;
        }
        if (longest <= 1) v[0]++;
        else if (longest == 2) v[1]++;
        else if (longest == 3) v[2]++;
        else v[3]++;
    }
    double chi2 = 0.0;
    for (int k = 0; k < 4; k++) {
        double exp = (double)N * pi[k];
        chi2 += (v[k] - exp) * (v[k] - exp) / exp;
    }
    return stat_chi2_sf(chi2, 3.0);
}

/* Cumulative-sums test, forward mode. */
static double test_cusum(const uint8_t *buf, size_t nbits) {
    long sum = 0, zmax = 0;
    for (size_t i = 0; i < nbits; i++) {
        sum += stat_bit(buf, i) ? 1 : -1;
        long a = sum < 0 ? -sum : sum;
        if (a > zmax) zmax = a;
    }
    if (zmax == 0) return 1.0;
    double n = (double)nbits;
    double z = (double)zmax;
    double sqn = sqrt(n);
    double p = 1.0;
    long kstart, kend;
    kstart = (long)floor((-n / z + 1.0) / 4.0);
    kend = (long)floor((n / z - 1.0) / 4.0);
    for (long k = kstart; k <= kend; k++)
        p -= stat_normal_cdf(((4 * k + 1) * z) / sqn) -
             stat_normal_cdf(((4 * k - 1) * z) / sqn);
    kstart = (long)floor((-n / z - 3.0) / 4.0);
    kend = (long)floor((n / z - 1.0) / 4.0);
    for (long k = kstart; k <= kend; k++)
        p += stat_normal_cdf(((4 * k + 3) * z) / sqn) -
             stat_normal_cdf(((4 * k + 1) * z) / sqn);
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

/* phi(m) helper for approximate entropy (circular). */
static double ap_phi(const uint8_t *buf, size_t n, int m) {
    if (m <= 0) return 0.0;
    size_t P = (size_t)1 << m;
    unsigned long *cnt = (unsigned long *)calloc(P, sizeof(unsigned long));
    if (!cnt) return NAN;
    for (size_t i = 0; i < n; i++) {
        size_t idx = 0;
        for (int j = 0; j < m; j++)
            idx = (idx << 1) | (size_t)stat_bit(buf, (i + (size_t)j) % n);
        cnt[idx]++;
    }
    double phi = 0.0;
    for (size_t k = 0; k < P; k++)
        if (cnt[k]) {
            double c = (double)cnt[k] / (double)n;
            phi += c * log(c);
        }
    free(cnt);
    return phi;
}

static double test_approx_entropy(const uint8_t *buf, size_t nbits, int m) {
    double apen = ap_phi(buf, nbits, m) - ap_phi(buf, nbits, m + 1);
    double chi2 = 2.0 * (double)nbits * (log(2.0) - apen);
    return stat_chi2_sf(chi2, (double)((size_t)1 << m)); /* df = 2^m */
}

/* psi^2(m) for the serial test (circular). */
static double serial_psi2(const uint8_t *buf, size_t n, int m) {
    if (m <= 0) return 0.0;
    size_t P = (size_t)1 << m;
    unsigned long *cnt = (unsigned long *)calloc(P, sizeof(unsigned long));
    if (!cnt) return NAN;
    for (size_t i = 0; i < n; i++) {
        size_t idx = 0;
        for (int j = 0; j < m; j++)
            idx = (idx << 1) | (size_t)stat_bit(buf, (i + (size_t)j) % n);
        cnt[idx]++;
    }
    double s = 0.0;
    for (size_t k = 0; k < P; k++) s += (double)cnt[k] * (double)cnt[k];
    free(cnt);
    return (double)P / (double)n * s - (double)n;
}

/* Serial test with block length m; writes both p-values. */
static void test_serial(const uint8_t *buf, size_t nbits, int m,
                        double *p1, double *p2) {
    double psim = serial_psi2(buf, nbits, m);
    double psim1 = serial_psi2(buf, nbits, m - 1);
    double psim2 = serial_psi2(buf, nbits, m - 2);
    double del1 = psim - psim1;
    double del2 = psim - 2.0 * psim1 + psim2;
    *p1 = stat_igamc((double)((size_t)1 << (m - 2)), del1 / 2.0);
    *p2 = stat_igamc((double)((size_t)1 << (m - 3)), del2 / 2.0);
}

/* Collision (birthday) test over 24-bit words drawn from the byte stream. */
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static double test_collision(const uint8_t *buf, size_t nbytes) {
    const int WBYTES = 3;                 /* 24-bit words */
    const double M = 16777216.0;          /* 2^24 bins */
    size_t m = nbytes / (size_t)WBYTES;
    if (m < 4096) return -1.0;
    if (m > 40000) m = 40000;             /* keep expected collisions moderate */
    uint32_t *w = (uint32_t *)malloc(m * sizeof(uint32_t));
    if (!w) return -1.0;
    for (size_t i = 0; i < m; i++) {
        const uint8_t *p = buf + i * (size_t)WBYTES;
        w[i] = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    }
    qsort(w, m, sizeof(uint32_t), cmp_u32);
    long collisions = 0;
    for (size_t i = 1; i < m; i++)
        if (w[i] == w[i - 1]) collisions++;
    free(w);
    double mu = (double)m * ((double)m - 1.0) / (2.0 * M);
    /* Poisson two-sided p from the regularized gamma. */
    double p_le = stat_igamc((double)collisions + 1.0, mu);        /* P(X<=C) */
    double p_ge = (collisions == 0) ? 1.0
                                    : stat_igam((double)collisions, mu); /* P(X>=C) */
    double p = 2.0 * (p_le < p_ge ? p_le : p_ge);
    if (p > 1.0) p = 1.0;
    return p;
}

/* ------------------------------------------------------------------ */
/*  Positive-control concentration statistics (byte level)             */
/* ------------------------------------------------------------------ */

static double peak_symbol_freq(const uint8_t *buf, size_t n, size_t block) {
    size_t nb = n / block;
    if (nb == 0) return 0.0;
    double acc = 0.0;
    unsigned hist[256];
    for (size_t b = 0; b < nb; b++) {
        memset(hist, 0, sizeof(hist));
        const uint8_t *p = buf + b * block;
        unsigned mx = 0;
        for (size_t i = 0; i < block; i++) {
            unsigned c = ++hist[p[i]];
            if (c > mx) mx = c;
        }
        acc += (double)mx;
    }
    return acc / (double)nb;
}

static double lag1_equal_rate(const uint8_t *buf, size_t n) {
    if (n < 2) return 0.0;
    size_t eq = 0;
    for (size_t i = 1; i < n; i++)
        if (buf[i] == buf[i - 1]) eq++;
    return (double)eq / (double)(n - 1);
}

/* ------------------------------------------------------------------ */
/*  PractRand-lite streaming anomaly check (bounded memory)            */
/* ------------------------------------------------------------------ */

static int streaming_check(size_t total_bytes,
                           double *out_monobit_p, double *out_bytechi_p,
                           size_t *out_completed_bytes) {
    const size_t CHUNK = 65536;
    uint8_t *chunk = (uint8_t *)malloc(CHUNK);
    if (!chunk) return STREAM_ALLOC_FAILURE;

    *out_completed_bytes = 0;

    unsigned long long ones = 0, bits = 0;
    unsigned long long hist[256];
    memset(hist, 0, sizeof(hist));

    size_t done = 0;
    while (done < total_bytes) {
        size_t take = total_bytes - done;
        if (take > CHUNK) take = CHUNK;
        int source_rc = fill_conditioned(chunk, take);
        if (source_rc != 0) {
            *out_completed_bytes = done;
            free(chunk);
            return source_rc;
        }
        for (size_t i = 0; i < take; i++) {
            uint8_t v = chunk[i];
            hist[v]++;
            /* popcount */
            unsigned b = v;
            b = b - ((b >> 1) & 0x55u);
            b = (b & 0x33u) + ((b >> 2) & 0x33u);
            b = (b + (b >> 4)) & 0x0Fu;
            ones += b;
            bits += 8;
        }
        done += take;
    }
    free(chunk);
    *out_completed_bytes = done;

    double s_obs = fabs((double)ones - (double)bits * 0.5) /
                   sqrt((double)bits * 0.25);
    *out_monobit_p = erfc(s_obs / sqrt(2.0));

    double nbytes = (double)total_bytes;
    double expv = nbytes / 256.0;
    double chi2 = 0.0;
    for (int k = 0; k < 256; k++) {
        double d = (double)hist[k] - expv;
        chi2 += d * d / expv;
    }
    *out_bytechi_p = stat_chi2_sf(chi2, 255.0);
    return 0;
}

/* ------------------------------------------------------------------ */

static size_t env_bytes(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || v == 0) return def;
    return (size_t)v;
}

int main(void) {
    size_t battery_bytes = env_bytes("QSIM_BATTERY_BYTES", 262144);      /* 256 KiB */
    size_t grover_bytes  = env_bytes("QSIM_GROVER_BYTES", 65536);        /* 64 KiB */
    size_t stream_bytes  = env_bytes("QSIM_STREAM_BYTES",
                                     (size_t)4 * 1024 * 1024);           /* 4 MiB */

    printf("QRNG statistical battery\n");
    printf("  battery_bytes=%zu grover_bytes=%zu stream_bytes=%zu\n",
           battery_bytes, grover_bytes, stream_bytes);

    uint8_t *cond = (uint8_t *)malloc(battery_bytes);
    uint8_t *grov = (uint8_t *)malloc(grover_bytes);
    if (!cond || !grov) {
        fprintf(stderr, "allocation failure\n");
        free(cond); free(grov);
        return 2;
    }

    if (fill_conditioned(cond, battery_bytes) != 0) {
        fprintf(stderr, "moonlab_qrng_bytes failed\n");
        free(cond); free(grov);
        return 2;
    }

    size_t nbits = battery_bytes * 8;

    /* -------- SP 800-22 subset over the conditioned path -------- */
    double p_mono   = test_monobit(cond, nbits);
    double p_block  = test_block_frequency(cond, nbits, 128);
    double p_runs   = test_runs(cond, nbits);
    double p_lrun   = test_longest_run(cond, nbits);
    double p_cusum  = test_cusum(cond, nbits);
    double p_apen   = test_approx_entropy(cond, nbits, 2);
    double p_ser1 = 0.0, p_ser2 = 0.0;
    test_serial(cond, nbits, 3, &p_ser1, &p_ser2);
    double p_coll   = test_collision(cond, battery_bytes);

    printf("  monobit        p=%.6g\n", p_mono);
    printf("  block_freq     p=%.6g\n", p_block);
    printf("  runs           p=%.6g%s\n", p_runs, p_runs < 0 ? " (prereq skip)" : "");
    printf("  longest_run    p=%.6g\n", p_lrun);
    printf("  cusum          p=%.6g\n", p_cusum);
    printf("  approx_entropy p=%.6g\n", p_apen);
    printf("  serial1        p=%.6g\n", p_ser1);
    printf("  serial2        p=%.6g\n", p_ser2);
    printf("  collision      p=%.6g%s\n", p_coll, p_coll < 0 ? " (skip)" : "");

    /* -------- streaming multi-MB check -------- */
    double p_smono = -1.0, p_sbyte = -1.0;
    size_t stream_completed = 0;
    int stream_rc = streaming_check(stream_bytes, &p_smono, &p_sbyte,
                                    &stream_completed);
    int stream_ok = (stream_rc == 0);
    printf("  stream_monobit p=%.6g\n", p_smono);
    printf("  stream_bytechi p=%.6g\n", p_sbyte);
    printf("  stream_source  rc=%d completed=%zu/%zu\n",
           stream_rc, stream_completed, stream_bytes);

    int battery_fail = 0;
    #define GATE(p) do { if ((p) >= 0.0 && (p) < FAIL_P) battery_fail = 1; } while (0)
    GATE(p_mono); GATE(p_block); GATE(p_runs); GATE(p_lrun); GATE(p_cusum);
    GATE(p_apen); GATE(p_ser1); GATE(p_ser2); GATE(p_coll);
    GATE(p_smono); GATE(p_sbyte);
    #undef GATE
    if (!stream_ok) battery_fail = 1;

    char stats[1024];
    snprintf(stats, sizeof(stats),
        "{\"monobit_p\":%.6g,\"block_freq_p\":%.6g,\"runs_p\":%.6g,"
        "\"longest_run_p\":%.6g,\"cusum_p\":%.6g,\"approx_entropy_p\":%.6g,"
        "\"serial1_p\":%.6g,\"serial2_p\":%.6g,\"collision_p\":%.6g,"
        "\"stream_monobit_p\":%.6g,\"stream_bytechi_p\":%.6g,"
        "\"stream_source_rc\":%d,\"stream_bytes_completed\":%zu,"
        "\"battery_bytes\":%zu,\"stream_bytes\":%zu,\"fail_threshold\":%g}",
        p_mono, p_block, p_runs, p_lrun, p_cusum, p_apen, p_ser1, p_ser2,
        p_coll, p_smono, p_sbyte, stream_rc, stream_completed,
        battery_bytes, stream_bytes, (double)FAIL_P);
    stat_emit_result("qrng_statistical_battery", !battery_fail, 1, stats);

    /* -------- positive control: GROVER bias must be detectable -------- */
    int control_fail = 0;
    double grov_peak = 0.0, cond_peak = 0.0, grov_lag1 = 0.0, cond_lag1 = 0.0;
    if (fill_grover_raw(grov, grover_bytes) != 0) {
        fprintf(stderr, "raw GROVER path unavailable\n");
        control_fail = 1;
    } else {
        grov_peak = peak_symbol_freq(grov, grover_bytes, PEAK_BLOCK);
        cond_peak = peak_symbol_freq(cond, battery_bytes, PEAK_BLOCK);
        grov_lag1 = lag1_equal_rate(grov, grover_bytes);
        cond_lag1 = lag1_equal_rate(cond, battery_bytes);
        printf("  [control] grover_peak=%.3f cond_peak=%.3f (thresh=%.1f)\n",
               grov_peak, cond_peak, PEAK_THRESH);
        printf("  [control] grover_lag1=%.5f cond_lag1=%.5f\n",
               grov_lag1, cond_lag1);
        /* Teeth: bias present in raw stream, absent in conditioned stream. */
        if (!(grov_peak > PEAK_THRESH))  control_fail = 1; /* bias not detected */
        if (!(cond_peak < PEAK_THRESH))  control_fail = 1; /* false positive   */
    }

    snprintf(stats, sizeof(stats),
        "{\"grover_peak_symbol\":%.4f,\"conditioned_peak_symbol\":%.4f,"
        "\"peak_threshold\":%.1f,\"grover_lag1_equal\":%.6f,"
        "\"conditioned_lag1_equal\":%.6f,\"grover_bytes\":%zu}",
        grov_peak, cond_peak, (double)PEAK_THRESH, grov_lag1, cond_lag1,
        grover_bytes);
    stat_emit_result("qrng_bias_positive_control", !control_fail, 1, stats);

    free(cond);
    free(grov);

    if (battery_fail) fprintf(stderr, "FAIL: conditioned battery gate tripped\n");
    if (control_fail) fprintf(stderr, "FAIL: GROVER positive control did not fire\n");
    return (battery_fail || control_fail) ? 1 : 0;
}
