/**
 * @file stat_common.h
 * @brief Self-contained statistical primitives for the Moonlab adversarial
 *        statistical battery.
 *
 * Header-only (all functions `static`) so the four test translation units in
 * this directory can each pull in exactly what they need with no extra link
 * objects and no risk of duplicate symbols.  Nothing here depends on any
 * library outside libm; the whole point of this lane is to test the RNG and
 * crypto with an *independent* implementation of the statistics, so we do NOT
 * reuse Moonlab's own nist_sp800_22.* module.
 *
 * Contains:
 *   - Cephes-style regularized incomplete gamma (igam / igamc) for chi-square
 *     and gamma tail p-values, implemented from scratch.
 *   - Standard-normal CDF built on libm erfc.
 *   - A machine-readable RESULT line emitter consumed by run_statistical.sh.
 *
 * References:
 *   - L. E. Bassham et al., NIST SP 800-22 rev 1a (2010).
 *   - W. J. Cody, "Rational Chebyshev approximation for the error function".
 *   - Cephes math library (igam.c), public-domain S. L. Moshier.
 */

#ifndef MOONLAB_STAT_COMMON_H
#define MOONLAB_STAT_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Incomplete gamma (Cephes, public domain, transcribed from scratch) */
/* ------------------------------------------------------------------ */

#define STAT_MACHEP 1.11022302462515654042e-16
#define STAT_MAXLOG 7.09782712893383996732e2
#define STAT_BIG    4.503599627370496e15
#define STAT_BIGINV 2.22044604925031308085e-16

static double stat_igam(double a, double x);
static double stat_igamc(double a, double x);

/* Regularized lower incomplete gamma P(a, x) via power series. */
static double stat_igam(double a, double x) {
    if (x <= 0.0 || a <= 0.0) return 0.0;
    if (x > 1.0 && x > a) return 1.0 - stat_igamc(a, x);

    double ax = a * log(x) - x - lgamma(a);
    if (ax < -STAT_MAXLOG) return 0.0;
    ax = exp(ax);

    double r = a;
    double c = 1.0;
    double ans = 1.0;
    do {
        r += 1.0;
        c *= x / r;
        ans += c;
    } while (c / ans > STAT_MACHEP);

    return ans * ax / a;
}

/* Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x) via
 * continued fraction (Lentz-style, as in Cephes). */
static double stat_igamc(double a, double x) {
    if (x <= 0.0 || a <= 0.0) return 1.0;
    if (x < 1.0 || x < a) return 1.0 - stat_igam(a, x);

    double ax = a * log(x) - x - lgamma(a);
    if (ax < -STAT_MAXLOG) return 0.0;
    ax = exp(ax);

    double y = 1.0 - a;
    double z = x + y + 1.0;
    double c = 0.0;
    double pkm2 = 1.0;
    double qkm2 = x;
    double pkm1 = x + 1.0;
    double qkm1 = z * x;
    double ans = pkm1 / qkm1;
    double t;
    do {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        double yc = y * c;
        double pk = pkm1 * z - pkm2 * yc;
        double qk = qkm1 * z - qkm2 * yc;
        if (qk != 0.0) {
            double r = pk / qk;
            t = fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (fabs(pk) > STAT_BIG) {
            pkm2 *= STAT_BIGINV;
            pkm1 *= STAT_BIGINV;
            qkm2 *= STAT_BIGINV;
            qkm1 *= STAT_BIGINV;
        }
    } while (t > STAT_MACHEP);

    return ans * ax;
}

/* Upper tail of a chi-square with `df` degrees of freedom evaluated at
 * `chi2`; this is the standard SP 800-22 p-value transform. */
static double stat_chi2_sf(double chi2, double df) {
    if (df <= 0.0) return 1.0;
    if (chi2 < 0.0) chi2 = 0.0;
    return stat_igamc(df / 2.0, chi2 / 2.0);
}

/* Standard-normal CDF via libm erfc. */
static double stat_normal_cdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

/* ------------------------------------------------------------------ */
/*  Bit access over a byte buffer (MSB-first, matching NIST convention)*/
/* ------------------------------------------------------------------ */

static inline int stat_bit(const uint8_t *buf, size_t i) {
    return (buf[i >> 3] >> (7 - (i & 7))) & 1;
}

/* ------------------------------------------------------------------ */
/*  RESULT line emitter -- the contract with run_statistical.sh        */
/*                                                                     */
/*  Exactly one line per named JSONL result:                           */
/*     RESULT name=<name> value=<PASS|FAIL> gating=<0|1> stats={...}    */
/*  `stats` is a compact JSON object and MUST be the final token so the */
/*  shell parser can grab the remainder of the line verbatim.          */
/* ------------------------------------------------------------------ */

static void stat_emit_result(const char *name, int pass, int gating,
                             const char *stats_json) {
    printf("RESULT name=%s value=%s gating=%d stats=%s\n",
           name, pass ? "PASS" : "FAIL", gating,
           (stats_json && stats_json[0]) ? stats_json : "{}");
    fflush(stdout);
}

#endif /* MOONLAB_STAT_COMMON_H */
