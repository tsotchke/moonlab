/**
 * @file test_tdvp_adaptive_energy_conservation.c
 * @brief Acceptance test #2 from
 *        `docs/research/adaptive_bond_tdvp.md`: real-time TDVP under
 *        the entropy-feedback PID controller must preserve `<H>`.
 *
 * Real-time TDVP is a symplectic integrator on the MPS manifold;
 * for a time-independent Hamiltonian it conserves energy up to
 * O(dt^3) per step.  The adaptive-bond controller only changes
 * *which* bond dimensions are kept after each two-site update; it
 * never modifies the Hamiltonian or the integration step.  If the
 * PID is wired correctly the energy drift over a short evolution
 * should therefore stay within the symplectic-error envelope, which
 * is what we assert here.
 *
 * Scope: 8-site Heisenberg, random initial MPS at chi=8, five
 * real-time steps at dt=0.02 with `target_entropy_error = 1e-3` and
 * `chi_ceiling = 32`.  The 8-site / short-time bound is loose
 * compared to the design-note 24-site / t=10 / 1e-5 target -- it is
 * specifically calibrated to run cheaply under ctest while still
 * catching gross PID misbehaviour.  Larger / longer validation
 * sweeps live in `benchmarks/`.
 *
 * Reference: arXiv:2604.03960 (entropy-feedback bond control for
 * 2TDVP) for the algorithm; Haegeman et al. PRB 94, 165116 (2016)
 * for the symplectic-conservation property of two-site TDVP.
 */

#include "../../src/algorithms/tensor_network/tdvp.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

int main(void) {
    fprintf(stdout, "=== TDVP adaptive-bond: real-time energy conservation ===\n");

    const uint32_t n = 8;
    mpo_t *mpo = mpo_heisenberg_create(n, /*J=*/1.0, /*Delta=*/1.0, /*h=*/0.0);
    if (!mpo) {
        fprintf(stderr, "SKIP: mpo_heisenberg_create failed\n");
        return 0;
    }

    tn_state_config_t mcfg = tn_state_config_create(64, 1e-12);
    tn_mps_state_t *mps = dmrg_init_random_mps(n, /*chi_init=*/8, &mcfg);
    if (!mps) {
        mpo_free(mpo);
        fprintf(stderr, "SKIP: dmrg_init_random_mps failed\n");
        return 0;
    }

    /* Reference energy of the initial state, computed against the
     * same MPO + MPS instance the engine will see. */
    const double E_initial = dmrg_compute_energy(mps, mpo);

    tdvp_config_t cfg = tdvp_config_adaptive(/*target_entropy_error=*/1e-3);
    cfg.evolution_type            = TDVP_REAL_TIME;
    cfg.dt                        = 0.02;
    cfg.adaptive_bond.chi_ceiling = 32;
    cfg.max_bond_dim              = 32;
    cfg.normalize                 = true;   /* renormalise each step to defeat roundoff */

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);
    CHECK(engine != NULL, "engine create");
    if (!engine) {
        tn_mps_free(mps);
        mpo_free(mpo);
        return 1;
    }

    /* Run 5 real-time steps and pin |E(t) - E(0)| / |E(0)| at each. */
    const int num_steps = 5;
    const double tol = 5e-3;   /* loose bound for the 8-site / short-time scope */

    double E_max_drift = 0.0;
    uint64_t chi_sum = 0;
    uint32_t chi_samples = 0;

    tdvp_result_t result = {0};
    for (int step = 0; step < num_steps; step++) {
        int rc = tdvp_step(engine, &result);
        CHECK(rc == 0, "tdvp_step rc=%d on step %d", rc, step);
        if (rc != 0) break;

        const double E_now = result.energy;
        const double drift = fabs(E_now - E_initial) /
                             (fabs(E_initial) > 1e-12 ? fabs(E_initial) : 1.0);
        if (drift > E_max_drift) E_max_drift = drift;

        CHECK(drift < tol,
              "step %d: |E - E0| / |E0| = %.3e > tol %.0e "
              "(E0=%.6f, E=%.6f)",
              step, drift, tol, E_initial, E_now);

        /* Aggregate the per-bond chi for the median-chi check. */
        if (result.bond_chi_distribution) {
            for (uint32_t b = 0; b < result.n_bonds; b++) {
                chi_sum += result.bond_chi_distribution[b];
                chi_samples++;
            }
        }
    }

    /* Bonus: the controller should actually be trimming.  On an
     * 8-site Heisenberg with chi_ceiling=32 we expect the mean
     * accepted chi across (bonds, steps) to be safely below the
     * ceiling.  Use a generous bound (< 80% of ceiling) so the
     * test pins the controller behaviour without being brittle to
     * implementation tweaks. */
    if (chi_samples > 0) {
        double mean_chi = (double)chi_sum / (double)chi_samples;
        CHECK(mean_chi <
                  0.80 * (double)cfg.adaptive_bond.chi_ceiling,
              "mean chi = %.2f is at or above 80%% of ceiling %u "
              "(controller is not trimming)",
              mean_chi, cfg.adaptive_bond.chi_ceiling);
        fprintf(stdout,
                "  mean accepted chi = %.2f, max relative |dE| = %.3e\n",
                mean_chi, E_max_drift);
    }

    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    tn_mps_free(mps);
    mpo_free(mpo);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
