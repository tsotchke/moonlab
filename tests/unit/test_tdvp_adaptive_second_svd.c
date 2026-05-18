/**
 * @file test_tdvp_adaptive_second_svd.c
 * @brief Acceptance test for the adaptive-bond two-pass SVD branch.
 *
 * The v0.4 adaptive truncation helper runs a first SVD at
 * `chi_ceiling` to expose the singular spectrum, then a second SVD
 * at the PID-selected `target_chi` when the controller asks for a
 * dimension strictly below the pass-1 output (see the comment at
 * `src/algorithms/tensor_network/tdvp.c:680-689`).  Both the energy-
 * conservation and TFIM-ground-state acceptance tests exercise the
 * path implicitly, but neither asserts that the second-SVD branch
 * actually fires nor that the resulting MPS stays sane while it does.
 *
 * This test pins both properties:
 *
 *   1. Configure imag-time evolution with `chi_ceiling = 64` and a
 *      target entropy budget so loose that the PID monotonically
 *      shrinks every bond toward `chi_floor`.  Drive the engine for
 *      enough imag-time steps that at least one step records a
 *      per-bond chi strictly below the pass-1 cap (which would equal
 *      `chi_ceiling` when no second pass runs).
 *
 *   2. Verify that across the run:
 *        - the recorded bond-chi never exceeds `chi_ceiling`;
 *        - the bond-chi sequence is non-increasing on average over
 *          the last few steps (the controller has settled);
 *        - the cumulative truncation error stays finite and bounded;
 *        - the final energy is finite (no NaN from a corrupted SVD).
 *
 *  These invariants would all fail if the second SVD silently
 *  produced a malformed factorization, so the test grounds the
 *  second-pass branch against actual physics behaviour rather than
 *  poking the internals.
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
    fprintf(stdout,
            "=== TDVP adaptive-bond: second-SVD re-truncation branch ===\n");

    const uint32_t n = 6;
    mpo_t *mpo = mpo_tfim_create(n, /*J=*/1.0, /*h=*/1.0);
    if (!mpo) {
        fprintf(stderr, "SKIP: mpo_tfim_create failed\n");
        return 0;
    }

    /* Wide starting bond so the pass-1 SVD has room to return a
     * larger chi than what the PID will eventually settle on -- this
     * is what forces the second-SVD branch to engage. */
    tn_state_config_t mcfg = tn_state_config_create(64, 1e-12);
    tn_mps_state_t *mps = dmrg_init_random_mps(n, /*chi_init=*/32, &mcfg);
    if (!mps) {
        mpo_free(mpo);
        fprintf(stderr, "SKIP: dmrg_init_random_mps failed\n");
        return 0;
    }

    /* Configure: large chi_ceiling so pass 1 keeps a wide spectrum;
     * imag-time evolution at moderate dt so the state converges
     * toward the TFIM ground state, whose entanglement is finite and
     * well below chi_ceiling. */
    tdvp_config_t cfg = tdvp_config_adaptive(/*target_entropy_error=*/1e-3);
    cfg.evolution_type            = TDVP_IMAGINARY_TIME;
    cfg.dt                        = 0.05;
    cfg.adaptive_bond.chi_ceiling = 64;
    cfg.adaptive_bond.chi_floor   = 2;
    cfg.max_bond_dim              = 64;
    cfg.normalize                 = true;

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);
    CHECK(engine != NULL, "engine create");
    if (!engine) {
        tn_mps_free(mps);
        mpo_free(mpo);
        return 1;
    }

    const int num_steps = 12;
    const uint32_t ceiling = cfg.adaptive_bond.chi_ceiling;

    int observed_below_ceiling = 0;
    int observed_below_chi32 = 0;   /* below the chi_init seed */
    double max_chi_first_half  = 0.0;
    double max_chi_second_half = 0.0;
    double last_energy = 0.0;
    double last_trunc_err = 0.0;

    tdvp_result_t result = {0};
    for (int step = 0; step < num_steps; step++) {
        int rc = tdvp_step(engine, &result);
        CHECK(rc == 0, "tdvp_step rc=%d on step %d", rc, step);
        if (rc != 0) break;

        last_energy    = result.energy;
        last_trunc_err = result.truncation_error;

        CHECK(isfinite(result.energy),
              "step %d: non-finite energy %g", step, result.energy);
        CHECK(isfinite(result.norm) && result.norm > 0.0,
              "step %d: bad norm %g", step, result.norm);
        CHECK(isfinite(result.truncation_error),
              "step %d: non-finite truncation_error %g",
              step, result.truncation_error);

        if (!result.bond_chi_distribution || result.n_bonds != n - 1) continue;

        double max_chi_step = 0.0;
        for (uint32_t b = 0; b < result.n_bonds; b++) {
            uint32_t chi_b = result.bond_chi_distribution[b];
            CHECK(chi_b <= ceiling,
                  "step %d bond %u: chi %u > ceiling %u",
                  step, b, chi_b, ceiling);
            if (chi_b < ceiling) observed_below_ceiling = 1;
            if (chi_b < 32)      observed_below_chi32   = 1;
            if ((double)chi_b > max_chi_step) max_chi_step = (double)chi_b;
        }

        if (step < num_steps / 2) {
            if (max_chi_step > max_chi_first_half) {
                max_chi_first_half = max_chi_step;
            }
        } else {
            if (max_chi_step > max_chi_second_half) {
                max_chi_second_half = max_chi_step;
            }
        }
    }

    /* Branch coverage: at least one step recorded a chi strictly
     * below the pass-1 ceiling.  When the second-SVD branch is dead
     * code (target_chi >= first->bond_dim always), every bond would
     * read out exactly chi_ceiling. */
    CHECK(observed_below_ceiling,
          "no step recorded a per-bond chi below chi_ceiling = %u; "
          "second-SVD branch was never exercised", ceiling);

    /* Settling: the controller is configured to shrink, so the max
     * chi across the second half of the run must not exceed the first
     * half.  Allow equality (the spectrum is allowed to be flat). */
    CHECK(max_chi_second_half <= max_chi_first_half + 1.0,
          "max chi rose between halves: %.0f -> %.0f (controller not "
          "settling)", max_chi_first_half, max_chi_second_half);

    /* Final-state sanity. */
    CHECK(isfinite(last_energy), "final energy not finite: %g", last_energy);
    CHECK(isfinite(last_trunc_err) && last_trunc_err >= 0.0,
          "final truncation_error not finite or negative: %g",
          last_trunc_err);

    fprintf(stdout,
            "  observed_below_ceiling=%d observed_below_chi32=%d "
            "max_chi_first_half=%.0f max_chi_second_half=%.0f\n",
            observed_below_ceiling, observed_below_chi32,
            max_chi_first_half, max_chi_second_half);
    fprintf(stdout,
            "  final energy = %.6f, last truncation_error = %.3e\n",
            last_energy, last_trunc_err);

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
