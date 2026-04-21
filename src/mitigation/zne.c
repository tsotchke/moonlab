/**
 * @file zne.c
 * @brief ZNE estimators (linear, Richardson, exponential).
 */

#include "zne.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- */
/* Linear least-squares                                          */
/* ------------------------------------------------------------- */
/* Fit y = a + b x by OLS; return (a, b) and residual stddev. */
static int fit_linear(const double *x, const double *y, size_t n,
                       double *a_out, double *b_out, double *sd_out) {
    if (n < 2) return -1;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; i++) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    const double dn = (double)n;
    const double denom = dn * sxx - sx * sx;
    if (fabs(denom) < 1e-300) return -2;
    const double b = (dn * sxy - sx * sy) / denom;
    const double a = (sy - b * sx) / dn;
    *a_out = a;
    *b_out = b;
    if (sd_out) {
        double ssq = 0.0;
        for (size_t i = 0; i < n; i++) {
            double r = y[i] - (a + b * x[i]);
            ssq += r * r;
        }
        *sd_out = (n > 2) ? sqrt(ssq / (double)(n - 2)) : 0.0;
    }
    return 0;
}

/* ------------------------------------------------------------- */
/* Richardson: Lagrange interpolation at x = 0                    */
/* ------------------------------------------------------------- */
/* E_mitigated = sum_i y_i * prod_{j != i} (-x_j) / (x_i - x_j)
 * (evaluation of the Lagrange interpolating polynomial at x = 0) */
static double richardson_at_zero(const double *x, const double *y,
                                  size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double num = 1.0, den = 1.0;
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            num *= -x[j];
            den *= (x[i] - x[j]);
        }
        if (fabs(den) < 1e-300) return 0.0;  /* degenerate scales */
        acc += y[i] * (num / den);
    }
    return acc;
}

/* ------------------------------------------------------------- */
/* Exponential: fit y = a + b * exp(-c * x)                       */
/* ------------------------------------------------------------- */
/* Three-parameter nonlinear fit with a pathological residual
 * landscape (a-b degeneracy for small c).  Strategy: grid-and-refine
 * on c (1D), since for fixed c the model is linear in (a, b) and the
 * optimum is closed-form.  This is stable regardless of the flatness
 * of the data, which Gauss-Newton is not. */
static int fit_ab_fixed_c(const double *x, const double *y, size_t n,
                           double c, double *a_out, double *b_out,
                           double *ssq_out) {
    double sEE = 0, sE = 0, sY = 0, sYE = 0;
    for (size_t i = 0; i < n; i++) {
        double e = exp(-c * x[i]);
        sEE += e * e;
        sE  += e;
        sY  += y[i];
        sYE += y[i] * e;
    }
    const double dn = (double)n;
    const double denom = dn * sEE - sE * sE;
    if (fabs(denom) < 1e-300) return -1;
    const double b = (dn * sYE - sE * sY) / denom;
    const double a = (sY - b * sE) / dn;
    *a_out = a; *b_out = b;
    double ssq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = y[i] - (a + b * exp(-c * x[i]));
        ssq += r * r;
    }
    *ssq_out = ssq;
    return 0;
}

static int fit_exponential(const double *x, const double *y, size_t n,
                            double *a_out, double *b_out, double *c_out,
                            double *sd_out) {
    if (n < 3) return -1;

    /* Coarse log-scan over c in [0.01, 10]. */
    double best_c = 1.0, best_a = 0.0, best_b = 0.0, best_ssq = 1e300;
    const int NGRID = 200;
    for (int i = 0; i <= NGRID; i++) {
        double logc = -2.0 + (3.0 * (double)i / (double)NGRID);  /* -2..1 */
        double c = pow(10.0, logc);
        double a, b, ssq;
        if (fit_ab_fixed_c(x, y, n, c, &a, &b, &ssq) == 0 && ssq < best_ssq) {
            best_c = c; best_a = a; best_b = b; best_ssq = ssq;
        }
    }
    /* Local golden-section refine around best_c. */
    double lo = best_c * 0.5, hi = best_c * 2.0;
    const double GR = (sqrt(5.0) - 1.0) / 2.0;
    for (int it = 0; it < 100; it++) {
        double c1 = hi - GR * (hi - lo);
        double c2 = lo + GR * (hi - lo);
        double a1, b1, s1, a2, b2, s2;
        fit_ab_fixed_c(x, y, n, c1, &a1, &b1, &s1);
        fit_ab_fixed_c(x, y, n, c2, &a2, &b2, &s2);
        if (s1 < s2) {
            hi = c2;
            if (s1 < best_ssq) { best_c = c1; best_a = a1; best_b = b1; best_ssq = s1; }
        } else {
            lo = c1;
            if (s2 < best_ssq) { best_c = c2; best_a = a2; best_b = b2; best_ssq = s2; }
        }
        if (hi - lo < 1e-10) break;
    }
    if (!isfinite(best_a) || !isfinite(best_b) || !isfinite(best_c)) return -3;
    *a_out = best_a; *b_out = best_b; *c_out = best_c;
    if (sd_out) {
        *sd_out = (n > 3) ? sqrt(best_ssq / (double)(n - 3)) : 0.0;
    }
    return 0;
}

/* ------------------------------------------------------------- */
/* Public API                                                     */
/* ------------------------------------------------------------- */

double zne_extrapolate(const double *scales,
                       const double *expectations,
                       size_t n,
                       zne_method_t method,
                       double *stderr_out) {
    if (!scales || !expectations || n < 2) {
        if (stderr_out) *stderr_out = -1.0;
        return 0.0;
    }

    switch (method) {
        case ZNE_LINEAR: {
            double a = 0, b = 0, sd = 0;
            if (fit_linear(scales, expectations, n, &a, &b, &sd) != 0) {
                if (stderr_out) *stderr_out = -1.0;
                return 0.0;
            }
            if (stderr_out) *stderr_out = sd;
            return a;
        }
        case ZNE_RICHARDSON: {
            if (stderr_out) *stderr_out = 0.0;
            return richardson_at_zero(scales, expectations, n);
        }
        case ZNE_EXPONENTIAL: {
            double a = 0, b = 0, c = 0, sd = 0;
            if (n < 3 ||
                fit_exponential(scales, expectations, n, &a, &b, &c, &sd) != 0) {
                /* Fallback to linear. */
                double a2 = 0, b2 = 0, sd2 = 0;
                if (fit_linear(scales, expectations, n, &a2, &b2, &sd2) != 0) {
                    if (stderr_out) *stderr_out = -1.0;
                    return 0.0;
                }
                if (stderr_out) *stderr_out = sd2;
                return a2;
            }
            if (stderr_out) *stderr_out = sd;
            return a + b;  /* x = 0 -> exp(0) = 1 */
        }
    }
    if (stderr_out) *stderr_out = -1.0;
    return 0.0;
}

double zne_mitigate(zne_expectation_fn fn,
                    void *user,
                    const double *scales,
                    size_t n,
                    zne_method_t method,
                    double *stderr_out) {
    if (!fn || !scales || n < 2) {
        if (stderr_out) *stderr_out = -1.0;
        return 0.0;
    }
    double *E = calloc(n, sizeof(double));
    if (!E) {
        if (stderr_out) *stderr_out = -1.0;
        return 0.0;
    }
    for (size_t i = 0; i < n; i++) {
        E[i] = fn(scales[i], user);
    }
    double out = zne_extrapolate(scales, E, n, method, stderr_out);
    free(E);
    return out;
}
