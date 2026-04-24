#ifndef QSIM_COMPLEX_COMPAT_H
#define QSIM_COMPLEX_COMPAT_H

#if defined(_WIN32) && defined(__clang__) && !defined(__cplusplus)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifdef complex
#undef complex
#endif
#define complex _Complex
#define _Complex_I (__extension__ 1.0i)
#define I _Complex_I

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double creal(double complex z) {
    return __real__ z;
}

static inline double cimag(double complex z) {
    return __imag__ z;
}

static inline double cabs(double complex z) {
    return hypot(creal(z), cimag(z));
}

static inline double carg(double complex z) {
    return atan2(cimag(z), creal(z));
}

static inline double complex conj(double complex z) {
    return creal(z) - I * cimag(z);
}

static inline double complex cexp(double complex z) {
    const double scale = exp(creal(z));
    const double angle = cimag(z);
    return scale * (cos(angle) + I * sin(angle));
}

static inline double complex csqrt(double complex z) {
    const double r = cabs(z);
    const double a = creal(z);
    const double b = cimag(z);
    const double real_part = sqrt((r + a) / 2.0);
    const double imag_part = copysign(sqrt((r - a) / 2.0), b);
    return real_part + I * imag_part;
}

static inline double complex cpow(double complex x, double complex y) {
    const double log_abs = log(cabs(x));
    const double arg = carg(x);
    const double a = creal(y);
    const double b = cimag(y);
    const double magnitude = exp(a * log_abs - b * arg);
    const double angle = a * arg + b * log_abs;
    return magnitude * (cos(angle) + I * sin(angle));
}

#elif defined(_MSC_VER) && !defined(__clang__) && !defined(__cplusplus)

#error "MSVC C does not support C99 complex arithmetic; build Windows with clang-cl or GCC."

#else

#include_next <complex.h>

#endif

#endif /* QSIM_COMPLEX_COMPAT_H */
