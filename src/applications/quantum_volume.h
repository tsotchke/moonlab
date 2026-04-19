/**
 * @file quantum_volume.h
 * @brief IBM Quantum Volume benchmark (Cross et al. 2019).
 *
 * OVERVIEW
 * --------
 * Quantum Volume is a device-level benchmark that rewards coherent,
 * fully-connected, high-fidelity operation rather than isolated gate
 * fidelity.  A random "model circuit" of width and depth @f$d@f$ is
 * constructed by stacking @f$d@f$ layers of @f$\lfloor d/2 \rfloor@f$
 * independent Haar-random @f$SU(4)@f$ gates, each layer preceded by a
 * uniformly random permutation of the qubits into pairs.  The
 * *heavy-output probability* of a given circuit is the total
 * probability mass on outcomes whose ideal probability exceeds the
 * median of the @f$2^d@f$ ideal output probabilities; classical
 * Porter-Thomas expectation for Haar-random circuits gives an
 * asymptotic mean of @f$(1 + \ln 2)/2 \approx 0.847@f$, while a
 * classical depolarising baseline saturates at @f$1/2@f$.
 *
 * The device passes width @f$d@f$ iff the mean heavy-output
 * probability estimated from many independent circuits exceeds
 * @f$2/3@f$ with one-sided 97.5 %% confidence.  The reported Quantum
 * Volume is @f$2^{d_{\max}}@f$ for the largest passing width.
 *
 * IMPLEMENTATION
 * --------------
 * The Haar-random 4x4 unitaries are drawn by the Mezzadri algorithm:
 * sample a 4x4 matrix of i.i.d. standard complex Gaussians, take the
 * modified Gram-Schmidt QR, then rotate each column so the
 * corresponding diagonal of @f$R@f$ is real and positive.  Randomising
 * the triangular-factor phase removes the bias of the naive
 * construction and yields a genuine Haar sample (see Mezzadri 2007).
 * Qubit permutations are drawn by the Fisher-Yates shuffle.  Heavy
 * outputs are computed exactly from the full statevector (no shot
 * noise).
 *
 * The parameter space accessible on the 32-qubit state-vector core is
 * @f$d \in [2, 16]@f$; @f$d = 10@f$ with 100 circuits takes about a
 * second on a single core.
 *
 * STATISTICAL REPORTING
 * ---------------------
 * The harness returns the sample mean, sample standard deviation and
 * the normal-approximation one-sided 97.5 %% lower confidence bound
 * @f$\bar{\mathrm{hop}} - 1.96\,s/\sqrt{N_{\mathrm{trials}}}@f$.  At
 * the trial counts used in testing (40-100) the normal approximation
 * is conservative relative to a bootstrap; the simulator passes by a
 * very comfortable margin because shot noise is absent.
 *
 * REFERENCES
 * ----------
 *  - A. W. Cross, L. S. Bishop, S. Sheldon, P. D. Nation and J. M.
 *    Gambetta, "Validating quantum computers using randomized model
 *    circuits", Phys. Rev. A 100, 032328 (2019), arXiv:1811.12926.
 *    Protocol definition.
 *  - F. Mezzadri, "How to generate random matrices from the classical
 *    compact groups", Notices of the AMS 54, 592 (2007),
 *    arXiv:math-ph/0609050.  Haar-sampling algorithm for @f$U(N)@f$.
 *
 * @since  v0.2.0
 * @stability evolving
 */

#ifndef MOONLAB_QUANTUM_VOLUME_H
#define MOONLAB_QUANTUM_VOLUME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t width;         /**< qubit count == model-circuit depth       */
    size_t num_trials;    /**< independent circuits drawn                */
    double mean_hop;      /**< sample mean heavy-output probability      */
    double stddev_hop;    /**< sample standard deviation                 */
    double lower_ci_97p5; /**< one-sided 97.5%% CI lower bound on mean   */
    int    passed;        /**< lower_ci_97p5 > 2/3 iff non-zero          */
} qv_result_t;

/**
 * @brief Run the IBM Quantum Volume protocol at width @p width.
 *
 * Draws @p num_trials independent circuits, evaluates the exact heavy
 * output probability of each, and returns the aggregate statistics in
 * @p out.  Ideal (noiseless) state-vector simulation; shot noise is
 * absent by construction.
 *
 * @param width      qubit count / circuit depth; 2 <= width <= 16
 * @param num_trials number of independent circuits; must be >= 10
 * @param rng_seed   64-bit seed for the internal xorshift RNG
 * @param out        result sink (non-NULL)
 * @return 0 on success, non-zero on invalid arguments or OOM
 */
int quantum_volume_run(size_t width,
                       size_t num_trials,
                       uint64_t rng_seed,
                       qv_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QUANTUM_VOLUME_H */
