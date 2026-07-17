/*
 * tests/numerical/numerical_common.h
 *
 * Shared scaffolding for the numerical-edge and uninitialised-memory
 * bug-hunting harnesses.  Every harness:
 *   - counts checks and failures,
 *   - prints the seed / inputs / expected-vs-got on any miss,
 *   - emits a single machine-readable summary line the run script parses:
 *       NUMERIC_SUMMARY harness=<name> checks=<n> fails=<m>
 *   - returns the failure count as its process exit code (0 == clean).
 *
 * No external dependencies beyond libc + the library headers each
 * harness pulls in via source-relative paths.
 */
#ifndef MOONLAB_NUMERICAL_COMMON_H
#define MOONLAB_NUMERICAL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <string.h>

/* The library headers also declare this identical typedef; a repeated
 * identical typedef is legal in C11, so including both is fine.  Declaring
 * it here lets the helpers below be used before any library header. */
typedef double _Complex complex_t;

/* ---- global accounting ------------------------------------------------- */
static int    g_checks = 0;
static int    g_fails  = 0;
static const char *g_harness = "unknown";

static inline void nc_begin(const char *name) {
    g_harness = name;
    g_checks = 0;
    g_fails = 0;
    printf("=== numerical harness: %s ===\n", name);
    fflush(stdout);
}

/* Emit summary + return exit code. */
static inline int nc_end(void) {
    printf("NUMERIC_SUMMARY harness=%s checks=%d fails=%d\n",
           g_harness, g_checks, g_fails);
    fflush(stdout);
    return g_fails;
}

/* ---- miss reporting ---------------------------------------------------- */
#define NC_MISS(fmt, ...)                                                     \
    do {                                                                      \
        g_fails++;                                                            \
        printf("  MISS [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__);\
        fflush(stdout);                                                       \
    } while (0)

#define NC_INFO(fmt, ...)                                                     \
    do { printf("  info: " fmt "\n", ##__VA_ARGS__); fflush(stdout); } while (0)

/* ---- finite / value checks -------------------------------------------- */
static inline int nc_is_bad(double x) { return isnan(x) || isinf(x); }
static inline int nc_is_bad_c(complex_t z) {
    return nc_is_bad(creal(z)) || nc_is_bad(cimag(z));
}

/* Assert |got - exp| <= tol (absolute). Returns 1 on pass, 0 on fail. */
static inline int nc_close(const char *what, double got, double exp, double tol) {
    g_checks++;
    if (nc_is_bad(got)) {
        NC_MISS("%s: got=%.17g (non-finite) exp=%.17g", what, got, exp);
        return 0;
    }
    double err = fabs(got - exp);
    if (err > tol) {
        NC_MISS("%s: got=%.17g exp=%.17g abserr=%.3e tol=%.3e",
                what, got, exp, err, tol);
        return 0;
    }
    return 1;
}

static inline int nc_close_c(const char *what, complex_t got, complex_t exp, double tol) {
    g_checks++;
    if (nc_is_bad_c(got)) {
        NC_MISS("%s: got=(%.17g,%.17g) non-finite", what, creal(got), cimag(got));
        return 0;
    }
    double err = cabs(got - exp);
    if (err > tol) {
        NC_MISS("%s: got=(%.17g,%.17g) exp=(%.17g,%.17g) abserr=%.3e tol=%.3e",
                what, creal(got), cimag(got), creal(exp), cimag(exp), err, tol);
        return 0;
    }
    return 1;
}

/* Assert a plain finiteness / no-NaN condition. */
static inline int nc_finite(const char *what, double x) {
    g_checks++;
    if (nc_is_bad(x)) { NC_MISS("%s: non-finite value %.17g", what, x); return 0; }
    return 1;
}

/* Assert a boolean condition holds. */
static inline int nc_expect(const char *what, int cond) {
    g_checks++;
    if (!cond) { NC_MISS("%s: condition false", what); return 0; }
    return 1;
}

/* ---- deterministic splitmix64 RNG (seed printed by caller) ------------- */
static inline uint64_t nc_splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
/* uniform double in [-1, 1] */
static inline double nc_urand(uint64_t *s) {
    return 2.0 * ((double)(nc_splitmix64(s) >> 11) / 9007199254740992.0) - 1.0;
}

#endif /* MOONLAB_NUMERICAL_COMMON_H */
