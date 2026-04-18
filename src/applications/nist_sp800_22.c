/**
 * @file nist_sp800_22.c
 * @brief NIST SP 800-22 rev 1a statistical tests.
 *
 * Implementations follow the algorithmic descriptions in NIST SP 800-22
 * Rev 1a (2010), Sections 2.1-2.15. Chi-square and gamma-function-based
 * p-values use Lanczos gamma + regularised incomplete gamma. All tests
 * operate on a bit array (uint8_t, 0 or 1).
 *
 * Not a certified implementation — use for regression / quality
 * monitoring of the Moonlab QRNG, not for FIPS cert.
 */

#include "nist_sp800_22.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --------------------------------------------------------------------- */
/* Special functions: erfc, lanczos gamma, regularized incomplete gamma. */
/* --------------------------------------------------------------------- */

/* Regularised upper incomplete gamma: Q(a, x) = Γ(a, x) / Γ(a).
 * Power-series + continued-fraction per Numerical Recipes §6.2. */
static double lngamma(double x) {
    /* Lanczos g=5, n=7. */
    static const double c[] = {
         76.18009172947146,    -86.50532032941677,
         24.01409824083091,     -1.231739572450155,
          0.1208650973866179e-2,-0.5395239384953e-5
    };
    double y = x, t = x + 5.5;
    t -= (x + 0.5) * log(t);
    double s = 1.000000000190015;
    for (int i = 0; i < 6; i++) { y += 1.0; s += c[i] / y; }
    return -t + log(2.5066282746310005 * s / x);
}

static double gammp_series(double a, double x) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 1; n < 500; n++) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (fabs(del) < fabs(sum) * 1e-15) break;
    }
    return sum * exp(-x + a * log(x) - lngamma(a));
}

static double gammq_cf(double a, double x) {
    const double FPMIN = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / FPMIN;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 500; i++) {
        double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < 1e-15) break;
    }
    return exp(-x + a * log(x) - lngamma(a)) * h;
}

/* Q(a, x) = 1 - P(a, x) */
static double igamc(double a, double x) {
    if (x < 0.0 || a <= 0.0) return -1.0;
    if (x == 0.0) return 1.0;
    if (x < a + 1.0) return 1.0 - gammp_series(a, x);
    return gammq_cf(a, x);
}

static double erfc_tail(double x) {
    /* erfc via incomplete gamma: erfc(x) = Q(1/2, x^2) for x >= 0. */
    if (x < 0.0) return 2.0 - erfc_tail(-x);
    return igamc(0.5, x * x);
}

/* ---------------------------- Helpers ------------------------------- */

static int64_t sum_plus_minus(const uint8_t *bits, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++) s += bits[i] ? 1 : -1;
    return s;
}

/* --------------------------- Tests 1..15 ---------------------------- */

double sp800_22_monobit(const uint8_t *b, size_t n) {
    if (n < 100) return -1.0;
    int64_t s = sum_plus_minus(b, n);
    double s_obs = fabs((double)s) / sqrt((double)n);
    return erfc_tail(s_obs / sqrt(2.0));
}

double sp800_22_block_frequency(const uint8_t *b, size_t n, size_t M) {
    if (n < 100 || M < 20 || M > n) return -1.0;
    size_t N = n / M;
    double chi2 = 0.0;
    for (size_t i = 0; i < N; i++) {
        size_t ones = 0;
        for (size_t j = 0; j < M; j++) ones += b[i * M + j];
        double pi = (double)ones / (double)M;
        double d = pi - 0.5;
        chi2 += d * d;
    }
    chi2 *= 4.0 * (double)M;
    return igamc((double)N / 2.0, chi2 / 2.0);
}

double sp800_22_runs(const uint8_t *b, size_t n) {
    if (n < 100) return -1.0;
    size_t ones = 0;
    for (size_t i = 0; i < n; i++) ones += b[i];
    double pi = (double)ones / (double)n;
    if (fabs(pi - 0.5) >= 2.0 / sqrt((double)n)) return 0.0;
    size_t V = 1;
    for (size_t i = 1; i < n; i++) if (b[i] != b[i-1]) V++;
    double num = fabs((double)V - 2.0 * (double)n * pi * (1.0 - pi));
    double den = 2.0 * sqrt(2.0 * (double)n) * pi * (1.0 - pi);
    return erfc_tail(num / den);
}

/* Longest run: SP 800-22 prescribes table for M=128 (n>=6272). */
double sp800_22_longest_run(const uint8_t *b, size_t n) {
    size_t M, N;
    const double *pi;
    int v_lo;
    static const double pi128[6] = {0.1174035788, 0.2430464095, 0.2493673718,
                                    0.1752489878, 0.1027377066, 0.1122159454};
    static const double pi512[6] = {0.1170, 0.2460, 0.2523, 0.1755, 0.1015, 0.1077};
    static const double pi1000[7]= {0.0882, 0.2092, 0.2483, 0.1933, 0.1208,
                                    0.0675, 0.0727};
    int K_use;
    if (n >= 750000)   { M = 10000; K_use = 6; N = n / M; pi = pi1000; v_lo = 10; }
    else if (n >= 6272){ M = 128;   K_use = 5; N = n / M; pi = pi128; v_lo = 4; }
    else if (n >= 128) { M = 8;     K_use = 3; N = n / M; pi = pi512; v_lo = 1;
                         (void)pi512; }
    else return -1.0;

    /* Simplified: use (K_use+1)-class binning with prescribed pi[] */
    long *nu = calloc((size_t)K_use + 1, sizeof(long));
    if (!nu) return -1.0;
    for (size_t i = 0; i < N; i++) {
        size_t max_run = 0, cur = 0;
        for (size_t j = 0; j < M; j++) {
            if (b[i * M + j]) { cur++; if (cur > max_run) max_run = cur; }
            else cur = 0;
        }
        int idx = (int)max_run - v_lo;
        if (idx < 0) idx = 0;
        if (idx > K_use) idx = K_use;
        nu[idx]++;
    }
    double chi2 = 0.0;
    for (int i = 0; i <= K_use; i++) {
        double exp_i = (double)N * pi[i];
        double d = (double)nu[i] - exp_i;
        chi2 += d * d / exp_i;
    }
    free(nu);
    return igamc((double)K_use / 2.0, chi2 / 2.0);
}

/* Binary matrix rank over GF(2). Used by sp800_22_rank. */
static int gf2_rank(uint8_t *M, int rows, int cols) {
    int r = 0;
    for (int c = 0; c < cols && r < rows; c++) {
        int pivot = -1;
        for (int i = r; i < rows; i++) {
            if (M[i * cols + c]) { pivot = i; break; }
        }
        if (pivot == -1) continue;
        if (pivot != r) {
            for (int k = 0; k < cols; k++) {
                uint8_t t = M[r * cols + k];
                M[r * cols + k] = M[pivot * cols + k];
                M[pivot * cols + k] = t;
            }
        }
        for (int i = 0; i < rows; i++) {
            if (i != r && M[i * cols + c]) {
                for (int k = 0; k < cols; k++)
                    M[i * cols + k] ^= M[r * cols + k];
            }
        }
        r++;
    }
    return r;
}

double sp800_22_rank(const uint8_t *b, size_t n) {
    const int Q = 32, M = 32;
    size_t block = (size_t)(Q * M);
    if (n < 38 * block) return -1.0;
    size_t N = n / block;
    long F_M = 0, F_M1 = 0, F_rest = 0;
    uint8_t *mat = malloc(block);
    if (!mat) return -1.0;
    for (size_t k = 0; k < N; k++) {
        memcpy(mat, b + k * block, block);
        int r = gf2_rank(mat, M, Q);
        if (r == M) F_M++;
        else if (r == M - 1) F_M1++;
        else F_rest++;
    }
    free(mat);
    /* Theoretical probabilities (Kim 1988):
     *   P(full rank)  ≈ 0.2888
     *   P(rank M-1)   ≈ 0.5776
     *   P(rank <=M-2) ≈ 0.1336 */
    double e_FM = 0.2888 * (double)N;
    double e_F1 = 0.5776 * (double)N;
    double e_R  = 0.1336 * (double)N;
    double chi2 = ((double)F_M  - e_FM) * ((double)F_M  - e_FM) / e_FM
                + ((double)F_M1 - e_F1) * ((double)F_M1 - e_F1) / e_F1
                + ((double)F_rest - e_R) * ((double)F_rest - e_R) / e_R;
    return exp(-chi2 / 2.0);  /* chi^2 with 2 dof upper-tail */
}

/* Spectral (DFT) test. */
double sp800_22_dft(const uint8_t *b, size_t n) {
    if (n < 1000) return -1.0;
    double *X = malloc(n * sizeof(double));
    double *M = malloc((n / 2) * sizeof(double));
    if (!X || !M) { free(X); free(M); return -1.0; }
    for (size_t i = 0; i < n; i++) X[i] = b[i] ? 1.0 : -1.0;
    /* Use a direct DFT magnitude of the first half. O(n^2); for n=1e6
     * this is slow — acceptable for smoke but not for production. */
    for (size_t k = 0; k < n / 2; k++) {
        double re = 0.0, im = 0.0;
        for (size_t j = 0; j < n; j++) {
            double angle = 2.0 * M_PI * (double)k * (double)j / (double)n;
            re += X[j] * cos(angle);
            im -= X[j] * sin(angle);
        }
        M[k] = sqrt(re * re + im * im);
    }
    double T = sqrt(log(1.0 / 0.05) * (double)n);
    double N0 = 0.95 * (double)(n / 2);
    size_t N1 = 0;
    for (size_t k = 0; k < n / 2; k++) if (M[k] < T) N1++;
    double d = ((double)N1 - N0) / sqrt((double)n * 0.95 * 0.05 / 4.0);
    free(X); free(M);
    return erfc_tail(fabs(d) / sqrt(2.0));
}

/* Non-overlapping template test (m=9, default template = 000000001). */
double sp800_22_non_overlapping_template(const uint8_t *b, size_t n) {
    const int m = 9;
    const uint8_t T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    size_t N = 8;
    if (n < (size_t)(N * 1000 * m)) return -1.0;
    size_t M = n / N;
    double mu = (double)(M - m + 1) / (double)(1 << m);
    double var = (double)M * (1.0 / (1 << m) -
                              (double)(2 * m - 1) / (double)(1ULL << (2 * m)));
    double chi2 = 0.0;
    for (size_t blk = 0; blk < N; blk++) {
        long W = 0;
        size_t i = 0;
        while (i + m <= M) {
            int match = 1;
            for (int k = 0; k < m; k++) {
                if (b[blk * M + i + k] != T[k]) { match = 0; break; }
            }
            if (match) { W++; i += m; } else i++;
        }
        double d = (double)W - mu;
        chi2 += d * d / var;
    }
    return igamc((double)N / 2.0, chi2 / 2.0);
}

/* Overlapping template: same pattern, uses Pr(W=k) distribution. */
double sp800_22_overlapping_template(const uint8_t *b, size_t n) {
    const int m = 9;
    enum { K = 5 };
    const size_t M = 1032;
    if (n < (8 * M)) return -1.0;
    size_t N = n / M;
    /* Reference probabilities from SP 800-22 Table 2.6. */
    static const double p[6] = {
        0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865
    };
    long v[6] = {0};
    for (size_t blk = 0; blk < N; blk++) {
        long W = 0;
        for (size_t i = 0; i + m <= M; i++) {
            int match = 1;
            for (int k = 0; k < m; k++) {
                if (b[blk * M + i + k] != ((k == m - 1) ? 1 : 0)) { match = 0; break; }
            }
            if (match) W++;
        }
        if (W >= (long)K) v[K]++;
        else v[W]++;
    }
    double chi2 = 0.0;
    for (size_t i = 0; i <= K; i++) {
        double e = (double)N * p[i];
        double d = (double)v[i] - e;
        chi2 += d * d / e;
    }
    return igamc((double)K / 2.0, chi2 / 2.0);
}

/* Maurer's universal statistical test. Minimal impl following SP 800-22 §2.9. */
double sp800_22_universal(const uint8_t *b, size_t n) {
    /* Recommended L=7, Q=1280, K = floor(n/L) - Q; require n >= 387840. */
    const int L = 7;
    const size_t Q = 1280;
    size_t K = n / L;
    if (K <= Q) return -1.0;
    K -= Q;
    size_t n_used = (Q + K) * L;
    if (n_used > n) return -1.0;
    static const double expectedValue[16] = {
        0, 0, 0, 0, 0, 0,
        5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
        10.170032, 11.168765, 12.168070, 13.167693, 14.167488
    };
    size_t ntbl = (size_t)1 << L;
    size_t *T = calloc(ntbl, sizeof(size_t));
    if (!T) return -1.0;
    for (size_t i = 0; i < Q; i++) {
        size_t x = 0;
        for (int k = 0; k < L; k++) x = (x << 1) | b[i * L + k];
        T[x] = i + 1;
    }
    double sum = 0.0;
    for (size_t i = Q; i < Q + K; i++) {
        size_t x = 0;
        for (int k = 0; k < L; k++) x = (x << 1) | b[i * L + k];
        sum += log2((double)(i + 1 - T[x]));
        T[x] = i + 1;
    }
    free(T);
    double fn = sum / (double)K;
    double sigma = 0.7 - 0.8 / L + (4.0 + 32.0 / L) * pow((double)K, -3.0 / L) / 15.0;
    double c = sigma * sqrt(expectedValue[L] * (1.0 + 1.0 / K));  /* rough */
    double arg = fabs(fn - expectedValue[L]) / (sqrt(2.0) * c);
    return erfc_tail(arg);
}

/* Linear Complexity: minimal impl using Berlekamp-Massey. */
static int berlekamp_massey(const uint8_t *s, size_t n) {
    int *C = calloc(n, sizeof(int));
    int *B = calloc(n, sizeof(int));
    if (!C || !B) { free(C); free(B); return -1; }
    C[0] = 1; B[0] = 1;
    int L = 0, m = -1;
    for (size_t N = 0; N < n; N++) {
        int d = s[N];
        for (int i = 1; i <= L; i++) d ^= C[i] & s[N - i];
        if (d == 1) {
            int *T = malloc(n * sizeof(int));
            memcpy(T, C, n * sizeof(int));
            for (size_t j = 0; j + (N - (size_t)m) < n; j++)
                C[j + N - (size_t)m] ^= B[j];
            if (2 * L <= (int)N) {
                L = (int)N + 1 - L;
                m = (int)N;
                memcpy(B, T, n * sizeof(int));
            }
            free(T);
        }
    }
    free(C); free(B);
    return L;
}

double sp800_22_linear_complexity(const uint8_t *b, size_t n, size_t M) {
    if (M < 500 || n < M * 10) return -1.0;
    const int K = 6;
    /* SP 800-22 Table 2.10 probabilities (M=500-5000). */
    static const double pi[7] = {
        0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833
    };
    size_t N = n / M;
    long nu[7] = {0};
    double mu = M / 2.0 + (9.0 + pow(-1.0, (double)(M + 1))) / 36.0 -
                (M / 3.0 + 2.0 / 9.0) / pow(2.0, (double)M);
    for (size_t blk = 0; blk < N; blk++) {
        int L = berlekamp_massey(b + blk * M, M);
        if (L < 0) return -1.0;
        double T = pow(-1.0, (double)M) * ((double)L - mu) + 2.0 / 9.0;
        int idx;
        if (T <= -2.5) idx = 0;
        else if (T <= -1.5) idx = 1;
        else if (T <= -0.5) idx = 2;
        else if (T <=  0.5) idx = 3;
        else if (T <=  1.5) idx = 4;
        else if (T <=  2.5) idx = 5;
        else idx = 6;
        nu[idx]++;
    }
    double chi2 = 0.0;
    for (int i = 0; i <= K; i++) {
        double e = (double)N * pi[i];
        double d = (double)nu[i] - e;
        chi2 += d * d / e;
    }
    return igamc((double)K / 2.0, chi2 / 2.0);
}

/* Serial test: overlapping m-bit pattern frequencies. */
static double serial_psi2(const uint8_t *b, size_t n, size_t m) {
    if (m == 0) return 0.0;
    size_t k = (size_t)1 << m;
    size_t *v = calloc(k, sizeof(size_t));
    if (!v) return -1.0;
    for (size_t i = 0; i < n; i++) {
        size_t x = 0;
        for (size_t j = 0; j < m; j++) {
            size_t idx = (i + j) % n;   /* circular */
            x = (x << 1) | b[idx];
        }
        v[x]++;
    }
    double sum = 0.0;
    for (size_t i = 0; i < k; i++) sum += (double)v[i] * (double)v[i];
    free(v);
    return sum * ((double)k / (double)n) - (double)n;
}

double sp800_22_serial(const uint8_t *b, size_t n, size_t m) {
    if (m < 2 || n < (size_t)(1u << (m + 2))) return -1.0;
    double psi_m = serial_psi2(b, n, m);
    double psi_m1 = serial_psi2(b, n, m - 1);
    double psi_m2 = (m >= 2) ? serial_psi2(b, n, m - 2) : 0.0;
    double d1 = psi_m - psi_m1;
    double d2 = psi_m - 2.0 * psi_m1 + psi_m2;
    double p1 = igamc(pow(2.0, (double)(m - 1)) / 2.0, d1 / 2.0);
    double p2 = igamc(pow(2.0, (double)(m - 2)) / 2.0, d2 / 2.0);
    return (p1 < p2) ? p1 : p2;
}

double sp800_22_approximate_entropy(const uint8_t *b, size_t n, size_t m) {
    if (m < 2 || n < (size_t)(1u << (m + 5))) return -1.0;
    double phi[2] = {0, 0};
    for (int r = 0; r < 2; r++) {
        size_t mm = m + (size_t)r;
        size_t k = (size_t)1 << mm;
        size_t *v = calloc(k, sizeof(size_t));
        if (!v) return -1.0;
        for (size_t i = 0; i < n; i++) {
            size_t x = 0;
            for (size_t j = 0; j < mm; j++) {
                size_t idx = (i + j) % n;
                x = (x << 1) | b[idx];
            }
            v[x]++;
        }
        double sum = 0.0;
        for (size_t i = 0; i < k; i++) {
            if (v[i] > 0) {
                double p = (double)v[i] / (double)n;
                sum += p * log(p);
            }
        }
        free(v);
        phi[r] = sum;
    }
    double apen = phi[0] - phi[1];
    double chi2 = 2.0 * (double)n * (log(2.0) - apen);
    double df = pow(2.0, (double)(m - 1));
    return igamc(df / 2.0, chi2 / 2.0);
}

/* Cumulative Sums (forward). */
double sp800_22_cusum_forward(const uint8_t *b, size_t n) {
    if (n < 100) return -1.0;
    long s = 0, max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        s += b[i] ? 1 : -1;
        if (labs(s) > max_abs) max_abs = labs(s);
    }
    double z = (double)max_abs;
    double sum1 = 0.0;
    for (long k = (long)((-n / z - 1) / 4); k <= (long)((n / z - 1) / 4); k++) {
        double a = (4.0 * k + 1.0) * z / sqrt((double)n);
        double c = (4.0 * k - 1.0) * z / sqrt((double)n);
        sum1 += 0.5 * (erfc_tail(-a / sqrt(2.0)) - erfc_tail(-c / sqrt(2.0)));
    }
    double sum2 = 0.0;
    for (long k = (long)((-n / z - 3) / 4); k <= (long)((n / z - 1) / 4); k++) {
        double a = (4.0 * k + 3.0) * z / sqrt((double)n);
        double c = (4.0 * k + 1.0) * z / sqrt((double)n);
        sum2 += 0.5 * (erfc_tail(-a / sqrt(2.0)) - erfc_tail(-c / sqrt(2.0)));
    }
    double p = 1.0 - sum1 + sum2;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

double sp800_22_cusum_reverse(const uint8_t *b, size_t n) {
    /* Run the forward test on the reversed sequence. */
    if (n < 100) return -1.0;
    uint8_t *r = malloc(n);
    if (!r) return -1.0;
    for (size_t i = 0; i < n; i++) r[i] = b[n - 1 - i];
    double p = sp800_22_cusum_forward(r, n);
    free(r);
    return p;
}

/* Random Excursions (state x=+1). Minimal single-state-at-a-time impl.
 * Full SP 800-22 runs the test at 8 states [-4..-1, 1..4]. We report the
 * minimum p-value across the 4 positive states as a conservative summary. */
double sp800_22_random_excursions(const uint8_t *b, size_t n) {
    if (n < 1000000) return -1.0;
    /* Walk S_k = Σ (2b_i - 1). Count visits to state x in cycles bounded
     * by zero crossings. Full table-based chi-square is involved; we
     * approximate with a bounded chi-square against expected frequencies. */
    long *S = malloc((n + 2) * sizeof(long));
    if (!S) return -1.0;
    S[0] = 0;
    for (size_t i = 0; i < n; i++) S[i + 1] = S[i] + (b[i] ? 1 : -1);
    /* Cycle boundaries (zero crossings with S=0). */
    size_t J = 0;
    for (size_t i = 1; i <= n; i++) if (S[i] == 0) J++;
    if (J < 500) { free(S); return -1.0; }
    /* For state x=1, count visits per cycle; chi^2 against Pr table. */
    double min_p = 1.0;
    for (int xi = 1; xi <= 4; xi++) {
        long cyc_counts[6] = {0};
        size_t start = 0;
        for (size_t i = 1; i <= n; i++) {
            if (S[i] == 0) {
                long visits = 0;
                for (size_t j = start; j < i; j++)
                    if (S[j + 1] == xi) visits++;
                long idx = visits;
                if (idx > 5) idx = 5;
                cyc_counts[idx]++;
                start = i;
            }
        }
        static const double pi1[6] = {0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0313};
        double chi2 = 0.0;
        for (int k = 0; k < 6; k++) {
            double e = (double)J * pi1[k];
            if (e > 0) {
                double d = (double)cyc_counts[k] - e;
                chi2 += d * d / e;
            }
        }
        double p = igamc(2.5, chi2 / 2.0);
        if (p < min_p) min_p = p;
    }
    free(S);
    return min_p;
}

double sp800_22_random_excursions_variant(const uint8_t *b, size_t n) {
    if (n < 1000000) return -1.0;
    long *S = malloc((n + 2) * sizeof(long));
    if (!S) return -1.0;
    S[0] = 0;
    for (size_t i = 0; i < n; i++) S[i + 1] = S[i] + (b[i] ? 1 : -1);
    size_t J = 0;
    for (size_t i = 1; i <= n; i++) if (S[i] == 0) J++;
    if (J < 500) { free(S); return -1.0; }
    double min_p = 1.0;
    for (int xi = 1; xi <= 9; xi++) {
        long visits = 0;
        for (size_t i = 1; i <= n; i++) if (S[i] == xi) visits++;
        double arg = fabs((double)(visits - (long)J))
                   / sqrt(2.0 * (double)J * (4.0 * fabs((double)xi) - 2.0));
        double p = erfc_tail(arg);
        if (p < min_p) min_p = p;
    }
    free(S);
    return min_p;
}

int sp800_22_run_all(const uint8_t *b, size_t n, double p[15]) {
    if (n < 100000) return -1;
    p[0]  = sp800_22_monobit(b, n);
    p[1]  = sp800_22_block_frequency(b, n, 128);
    p[2]  = sp800_22_runs(b, n);
    p[3]  = sp800_22_longest_run(b, n);
    p[4]  = sp800_22_rank(b, n);
    p[5]  = sp800_22_dft(b, n);
    p[6]  = sp800_22_non_overlapping_template(b, n);
    p[7]  = sp800_22_overlapping_template(b, n);
    p[8]  = sp800_22_universal(b, n);
    p[9]  = sp800_22_linear_complexity(b, n, 500);
    p[10] = sp800_22_serial(b, n, 8);
    p[11] = sp800_22_approximate_entropy(b, n, 8);
    double cf = sp800_22_cusum_forward(b, n);
    double cr = sp800_22_cusum_reverse(b, n);
    p[12] = (cf >= 0 && cr >= 0) ? ((cf < cr) ? cf : cr) : -1.0;
    p[13] = sp800_22_random_excursions(b, n);
    p[14] = sp800_22_random_excursions_variant(b, n);

    int passed = 0;
    for (int i = 0; i < 15; i++) {
        if (p[i] >= 0.01) passed++;
    }
    return passed;
}
