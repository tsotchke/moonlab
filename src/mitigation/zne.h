/**
 * @file zne.h
 * @brief Zero-noise extrapolation (ZNE) for error mitigation.
 *
 * ZNE is the simplest and most widely deployed error-mitigation
 * technique on current NISQ hardware.  The idea: evaluate an
 * observable at several noise scales lambda >= 1 (lambda = 1 is the
 * hardware noise level; lambda > 1 amplifies the noise), then
 * extrapolate the measurement data back to lambda = 0, the zero-noise
 * limit.
 *
 * This header ships three extrapolation estimators:
 *   - ZNE_LINEAR       linear fit E(lambda) = a + b*lambda; E_mitigated = a
 *   - ZNE_RICHARDSON   exact Lagrange interpolation at lambda = 0 using
 *                      the supplied scales (deg = n - 1 polynomial)
 *   - ZNE_EXPONENTIAL  nonlinear fit E = a + b*exp(-c*lambda); fallback
 *                      to linear if the fit fails or n < 3
 *
 * The caller is responsible for actually producing the measurements
 * at the requested scales -- typically by applying digital gate
 * folding (U U^dag U etc.) or by multiplying rates in a Kraus-channel
 * noise model and re-simulating.  Convenience @ref zne_mitigate
 * wraps a user callback to automate the sweep-and-extrapolate loop.
 *
 * References:
 *   - Temme et al., Phys. Rev. Lett. 119, 180509 (2017)
 *   - Li & Benjamin, Phys. Rev. X 7, 021050 (2017)
 *   - Giurgica-Tiron et al., arXiv:2005.10921 (digital folding)
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_MITIGATION_ZNE_H
#define MOONLAB_MITIGATION_ZNE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ZNE_LINEAR      = 0,
    ZNE_RICHARDSON  = 1,
    ZNE_EXPONENTIAL = 2,
} zne_method_t;

/**
 * @brief Extrapolate a set of (lambda, E) measurements to lambda = 0.
 *
 * @param scales        array of noise scales lambda_i (must be length n,
 *                      all distinct, all > 0; lambda = 1 convention is
 *                      the hardware level, but any positive scaling
 *                      works so long as they are distinct)
 * @param expectations  array of measured expectations E(lambda_i)
 * @param n             number of samples; must be >= 2
 * @param method        extrapolation estimator
 * @param stderr_out    optional; set to the fit residual standard
 *                      deviation if non-NULL.  For the Richardson
 *                      estimator this is always 0 (exact interpolation).
 *
 * @return extrapolated value E(0); returns 0 and sets *stderr_out to
 *         a negative sentinel on argument error.
 */
double zne_extrapolate(
    const double *scales,
    const double *expectations,
    size_t n,
    zne_method_t method,
    double *stderr_out);

/**
 * @brief Prototype for a user-provided expectation-value oracle.
 *
 * The callback should run the circuit at the requested noise scale
 * and return the measured expectation value.  Typical
 * implementations: multiply noise-model rates by @p lambda then run a
 * full forward-and-measure, or apply digital gate folding before the
 * forward path.
 */
typedef double (*zne_expectation_fn)(double lambda, void *user);

/**
 * @brief Sweep @p fn across the provided noise scales and extrapolate.
 *
 * Convenience driver equivalent to:
 *     for each lambda in scales:   E_i = fn(lambda, user)
 *     return zne_extrapolate(scales, E, n, method, stderr_out)
 *
 * @return extrapolated E(0); 0 on argument error.
 */
double zne_mitigate(
    zne_expectation_fn fn,
    void *user,
    const double *scales,
    size_t n,
    zne_method_t method,
    double *stderr_out);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MITIGATION_ZNE_H */
