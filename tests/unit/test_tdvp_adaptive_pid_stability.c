/**
 * @file test_tdvp_adaptive_pid_stability.c
 * @brief Acceptance test #4 from
 *        `docs/research/adaptive_bond_tdvp.md`: PID stability sweep
 *        over the (kp, ki, kd) gain space.
 *
 * The adaptive-bond controller is a feedback loop with two
 * independent time scales -- the imag/real-time step duration and
 * the implicit "entropy update" cadence of the TDVP sweep.  Poorly
 * chosen gains can drive the per-bond chi into oscillations that
 * defeat the wall-time savings the controller is meant to deliver.
 * This test sweeps a 3 x 3 x 3 grid of gain triplets around the
 * reference-paper defaults (kp = 0.5, ki = 0.05, kd = 0.1) and
 * asserts that the per-bond chi changes by at most `MAX_OSC` between
 * consecutive steps for at least 80 % of the grid (22 / 27 points).
 *
 * Scope: 8-site Heisenberg chain (J = 1, Delta = 1), 5 real-time
 * TDVP steps per gain triplet at `dt = 0.02`, `chi_ceiling = 32`.
 * Real-time evolution is used because it conserves norm by
 * unitarity and therefore isolates the controller's stability from
 * any imag-time renormalisation dynamics.
 *
 * Reference: arXiv:2604.03960 (entropy-feedback bond control for
 * 2TDVP, section 4 on gain calibration).
 */

#include "../../src/algorithms/tensor_network/tdvp.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

/* Maximum per-step bond-dim change tolerated by the design note. */
#define MAX_OSC 4

/* Result of running one gain triplet. */
typedef struct {
    double kp, ki, kd;
    uint32_t max_osc;          /* Largest |chi(t+1) - chi(t)| we saw. */
    int run_ok;                /* 1 if all 5 steps returned rc=0. */
} sweep_point_t;

static int run_one(double kp, double ki, double kd,
                   sweep_point_t *out) {
    const uint32_t n = 8;
    const int num_steps = 5;

    mpo_t *mpo = mpo_heisenberg_create(n, /*J=*/1.0, /*Delta=*/1.0, /*h=*/0.0);
    if (!mpo) return -1;

    tn_state_config_t mcfg = tn_state_config_create(32, 1e-12);
    tn_mps_state_t *mps = dmrg_init_random_mps(n, /*chi_init=*/8, &mcfg);
    if (!mps) { mpo_free(mpo); return -1; }

    tdvp_config_t cfg = tdvp_config_adaptive(/*target_entropy_error=*/1e-3);
    cfg.evolution_type            = TDVP_REAL_TIME;
    cfg.dt                        = 0.02;
    cfg.adaptive_bond.chi_ceiling = 32;
    cfg.adaptive_bond.kp          = kp;
    cfg.adaptive_bond.ki          = ki;
    cfg.adaptive_bond.kd          = kd;
    cfg.max_bond_dim              = 32;
    cfg.normalize                 = true;

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);
    if (!engine) {
        tn_mps_free(mps);
        mpo_free(mpo);
        return -1;
    }

    /* Per-bond chi history: shape [num_steps, n - 1]. */
    uint32_t *history = (uint32_t *)calloc(
        (size_t)num_steps * (n - 1), sizeof(uint32_t));
    if (!history) {
        tdvp_engine_free(engine);
        tn_mps_free(mps);
        mpo_free(mpo);
        return -1;
    }

    int ok = 1;
    tdvp_result_t result = {0};
    for (int step = 0; step < num_steps; step++) {
        int rc = tdvp_step(engine, &result);
        if (rc != 0) { ok = 0; break; }
        if (result.bond_chi_distribution && result.n_bonds == n - 1) {
            for (uint32_t b = 0; b < n - 1; b++) {
                history[step * (n - 1) + b] =
                    result.bond_chi_distribution[b];
            }
        }
    }

    /* Max oscillation = max over bonds + step pairs of |chi(t+1) - chi(t)|. */
    uint32_t max_osc = 0;
    if (ok) {
        for (int step = 1; step < num_steps; step++) {
            for (uint32_t b = 0; b < n - 1; b++) {
                uint32_t a = history[(step - 1) * (n - 1) + b];
                uint32_t c = history[step * (n - 1) + b];
                uint32_t d = (a > c) ? (a - c) : (c - a);
                if (d > max_osc) max_osc = d;
            }
        }
    }

    out->kp      = kp;
    out->ki      = ki;
    out->kd      = kd;
    out->max_osc = max_osc;
    out->run_ok  = ok;

    free(history);
    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    tn_mps_free(mps);
    mpo_free(mpo);
    return 0;
}

int main(void) {
    fprintf(stdout, "=== TDVP adaptive-bond: PID stability sweep ===\n");

    /* 3 x 3 x 3 grid around the reference defaults. */
    const double kp_grid[3] = {0.25, 0.50, 1.00};
    const double ki_grid[3] = {0.025, 0.05, 0.10};
    const double kd_grid[3] = {0.05, 0.10, 0.20};

    int total = 0;
    int passed = 0;
    int run_failures = 0;

    fprintf(stdout,
            "  %-6s %-6s %-6s  %-7s %-3s\n",
            "kp", "ki", "kd", "max_osc", "ok");

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            sweep_point_t pt = {0};
            int rc = run_one(kp_grid[i], ki_grid[j], kd_grid[k], &pt);
            CHECK(rc == 0, "run failed at (kp=%.3f, ki=%.3f, kd=%.3f)",
                  kp_grid[i], ki_grid[j], kd_grid[k]);
            if (rc != 0) continue;

            total++;
            if (!pt.run_ok) {
                run_failures++;
            } else if (pt.max_osc <= MAX_OSC) {
                passed++;
            }
            int stable = pt.run_ok && pt.max_osc <= MAX_OSC;
            fprintf(stdout,
                    "  %-6.3f %-6.3f %-6.3f  %-7u %s\n",
                    pt.kp, pt.ki, pt.kd, pt.max_osc,
                    stable ? "yes" : "no");
        }
      }
    }

    fprintf(stdout,
            "Summary: %d / %d gain triplets stable, %d hard-failed runs.\n",
            passed, total, run_failures);

    /* Design-note acceptance: >= 80 % of grid stable (22 / 27). */
    const int threshold = (total * 80) / 100;
    CHECK(passed >= threshold,
          "only %d / %d points are stable; threshold %d (80%%)",
          passed, total, threshold);
    CHECK(run_failures == 0,
          "%d run(s) hard-failed (engine couldn't complete sweep)",
          run_failures);

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
