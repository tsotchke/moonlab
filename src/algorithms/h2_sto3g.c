/**
 * @file h2_sto3g.c
 * @brief First-principles STO-3G H2 Pauli coefficients (see h2_sto3g.h).
 *
 * All integrals are s-type Gaussian primitives on the internuclear (z) axis;
 * the molecular-orbital reduction uses the g/u symmetry of homonuclear H2.
 * Ported from the rsh-differentiable-quantum witness (validated to the STO-3G
 * FCI energy at machine precision).
 */
#include "h2_sto3g.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define ANG2BOHR (1.0 / 0.52917721067)

/* STO-3G hydrogen 1s: tabulated exponents already encode the zeta scaling
 * (a second zeta^2 double-scales). */
static const double A_[3] = {3.42525091, 0.62391373, 0.16885540};
static const double D_[3] = {0.15432897, 0.53532814, 0.44463454};

static double nprim(double a) { return pow(2.0 * a / M_PI, 0.75); }

/* robust erf: Maclaurin series for |x|<3, erfc continued fraction beyond. */
static double erf_series(double x) {
    double sum = x, term = x, x2 = x * x;
    for (int n = 1; n < 200; n++) {
        term *= (-x2) * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
        sum += term;
        if (fabs(term) < 1e-18 * fabs(sum)) break;
    }
    return (2.0 / sqrt(M_PI)) * sum;
}
static double erfc_cf(double x) {
    double frac = 0.0;
    for (int k = 40; k >= 1; k--) frac = (0.5 * k) / (x + frac);
    return (exp(-x * x) / sqrt(M_PI)) * (1.0 / (x + frac));
}
static double erf_r(double x) {
    double s = (x < 0) ? -1.0 : 1.0, ax = fabs(x);
    return s * (ax < 3.0 ? erf_series(ax) : 1.0 - erfc_cf(ax));
}
/* Boys F0(u) = (sqrt(pi)/2) erf(sqrt u)/sqrt u ; F0(0)=1 (same-center). */
static double boys0(double u) {
    if (u < 1e-13) return 1.0;
    double su = sqrt(u);
    return 0.5 * sqrt(M_PI) * erf_r(su) / su;
}

/* z of center 0 (A, at origin) or 1 (B, at R bohr); squared center separation. */
static double zc(int c, double R) { return (c == 0) ? 0.0 : R; }
static double d2c(int ci, int cj, double R) { return (ci == cj) ? 0.0 : R * R; }

/* primitive geometric factors (normalization folded in at contraction). */
static double p_overlap(double a, double b, int ci, int cj, double R) {
    double p = a + b, mu = a * b / p;
    return pow(M_PI / p, 1.5) * exp(-mu * d2c(ci, cj, R));
}
static double p_kinetic(double a, double b, int ci, int cj, double R) {
    double p = a + b, mu = a * b / p, d2 = d2c(ci, cj, R);
    return pow(M_PI / p, 1.5) * exp(-mu * d2) * mu * (3.0 - 2.0 * mu * d2);
}
static double p_nuclear(double a, double b, int ci, int cj, int cc, double R) {
    double p = a + b, mu = a * b / p;
    double pz = (a * zc(ci, R) + b * zc(cj, R)) / p;
    double dd = pz - zc(cc, R), pc2 = dd * dd;
    return -(2.0 * M_PI / p) * exp(-mu * d2c(ci, cj, R)) * boys0(p * pc2);
}
static double p_eri(double a, double b, double c, double d,
                    int ca, int cb, int cc, int cd, double R) {
    double p = a + b, q = c + d, muab = a * b / p, mucd = c * d / q;
    double pz = (a * zc(ca, R) + b * zc(cb, R)) / p;
    double qz = (c * zc(cc, R) + d * zc(cd, R)) / q;
    double dd = pz - qz, pq2 = dd * dd;
    return (2.0 * pow(M_PI, 2.5) / (p * q * sqrt(p + q)))
         * exp(-muab * d2c(ca, cb, R)) * exp(-mucd * d2c(cc, cd, R))
         * boys0((p * q / (p + q)) * pq2);
}

/* contract STO-3G primitives -> AO integrals at bond length R (bohr). */
static double ao_1e(double (*fn)(double, double, int, int, double),
                    int ci, int cj, double R, const double *dn) {
    double acc = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            acc += dn[i] * dn[j] * fn(A_[i], A_[j], ci, cj, R);
    return acc;
}
static double ao_nuclear(int ci, int cj, double R, const double *dn) {
    double acc = 0.0;
    for (int cc = 0; cc < 2; cc++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                acc += dn[i] * dn[j] * p_nuclear(A_[i], A_[j], ci, cj, cc, R);
    return acc;
}
static double ao_eri(int ca, int cb, int cc, int cd, double R, const double *dn) {
    double acc = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    acc += dn[i] * dn[j] * dn[k] * dn[l]
                         * p_eri(A_[i], A_[j], A_[k], A_[l], ca, cb, cc, cd, R);
    return acc;
}

void h2_sto3g_pauli_coeffs(double r_angstrom, double g[5]) {
    double R = r_angstrom * ANG2BOHR;
    double dn[3];
    for (int i = 0; i < 3; i++) dn[i] = D_[i] * nprim(A_[i]);

    double s   = ao_1e(p_overlap, 0, 1, R, dn);
    double hAA = ao_1e(p_kinetic, 0, 0, R, dn) + ao_nuclear(0, 0, R, dn);
    double hAB = ao_1e(p_kinetic, 0, 1, R, dn) + ao_nuclear(0, 1, R, dn);
    double hgg = (hAA + hAB) / (1.0 + s);
    double huu = (hAA - hAB) / (1.0 - s);
    double cg  = 1.0 / sqrt(2.0 * (1.0 + s));
    double cu  = 1.0 / sqrt(2.0 * (1.0 - s));
    double gv[2] = {cg, cg}, uv[2] = {cu, -cu};   /* sigma_u = (A - B) */

    double gggg = 0, uuuu = 0, gguu = 0, gugu = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++) {
                    double e = ao_eri(i, j, k, l, R, dn);
                    gggg += gv[i] * gv[j] * gv[k] * gv[l] * e;
                    uuuu += uv[i] * uv[j] * uv[k] * uv[l] * e;
                    gguu += gv[i] * gv[j] * uv[k] * uv[l] * e;
                    gugu += gv[i] * uv[j] * gv[k] * uv[l] * e;
                }

    double h11 = 2.0 * hgg + gggg;   /* E[(sigma_g)^2] */
    double h22 = 2.0 * huu + uuuu;   /* E[(sigma_u)^2] */
    double h12 = gugu;               /* exchange coupling */
    double eopen = hgg + huu + gguu; /* E[g^up u^dn] = J_gu */

    g[0] = 0.5 * (eopen + 0.5 * (h11 + h22));  /* II */
    g[1] = 0.25 * (h22 - h11);                 /* IZ */
    g[2] = 0.25 * (h11 - h22);                 /* ZI */
    g[3] = 0.5 * (eopen - 0.5 * (h11 + h22));  /* ZZ */
    g[4] = h12;                                /* XX */
}
