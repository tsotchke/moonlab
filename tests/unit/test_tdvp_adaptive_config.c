/**
 * @file test_tdvp_adaptive_config.c
 * @brief Backwards-compatibility regression for the v0.4 adaptive-bond
 *        TDVP config (Phase 3B, step 1 of the implementation roadmap).
 *
 * Acceptance criterion #1 from `docs/research/adaptive_bond_tdvp.md`:
 *
 *   With `adaptive_bond.enabled = false`, every existing test in
 *   `tests/unit/test_tdvp*.c` passes bit-identically against the
 *   v0.3.1 baseline.
 *
 * This file is the structural component of that criterion -- it pins
 * the layout, defaults, and helpers introduced by step 1 so future
 * adaptive-bond patches cannot regress the config surface without
 * the test failing.  Numerical TDVP-evolution parity belongs to a
 * separate test in step 2 once the PID inner helper exists.
 */

#include "../../src/algorithms/tensor_network/tdvp.h"

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
    fprintf(stdout, "=== TDVP adaptive-bond config regression ===\n");

    /* ---- 1. Default config preserves the v0.3 legacy path ---------- */
    {
        tdvp_config_t cfg = tdvp_config_default();
        CHECK(cfg.evolution_type == TDVP_REAL_TIME, "default evolution");
        CHECK(cfg.variant == TDVP_TWO_SITE, "default variant");
        CHECK(cfg.integrator == INTEGRATOR_LANCZOS, "default integrator");
        CHECK(cfg.dt == 0.01, "default dt");
        CHECK(cfg.max_bond_dim == 128, "default max_bond_dim");
        CHECK(cfg.svd_cutoff == 1e-10, "default svd_cutoff");
        CHECK(cfg.normalize == true, "default normalize");
        CHECK(cfg.verbose == false, "default verbose");

        /* Adaptive-bond must be disabled in the default config so
         * v0.3 callers see no behaviour change. */
        CHECK(cfg.adaptive_bond.enabled == false,
              "adaptive_bond.enabled is false by default");
        CHECK(cfg.adaptive_bond.target_entropy_error == 0.0,
              "disabled config has zero target_entropy_error");
        CHECK(cfg.adaptive_bond.kp == 0.0, "disabled kp=0");
        CHECK(cfg.adaptive_bond.ki == 0.0, "disabled ki=0");
        CHECK(cfg.adaptive_bond.kd == 0.0, "disabled kd=0");
        CHECK(cfg.adaptive_bond.chi_floor == 0, "disabled chi_floor=0");
        CHECK(cfg.adaptive_bond.chi_ceiling == 0, "disabled chi_ceiling=0");
        CHECK(cfg.adaptive_bond.alpha == 0.0, "disabled alpha=0");
    }

    /* ---- 2. tdvp_adaptive_bond_config_disabled matches default ---- */
    {
        tdvp_adaptive_bond_config_t ab = tdvp_adaptive_bond_config_disabled();
        CHECK(!ab.enabled, "disabled helper has enabled=false");
        CHECK(ab.target_entropy_error == 0.0, "disabled helper eps=0");
        CHECK(ab.kp == 0.0 && ab.ki == 0.0 && ab.kd == 0.0,
              "disabled helper gains zeroed");
        CHECK(ab.chi_floor == 0 && ab.chi_ceiling == 0,
              "disabled helper bounds zeroed");
        CHECK(ab.alpha == 0.0, "disabled helper alpha=0");
    }

    /* ---- 3. tdvp_adaptive_bond_config_default has reference gains - */
    {
        const double eps = 1e-3;
        tdvp_adaptive_bond_config_t ab =
            tdvp_adaptive_bond_config_default(eps);
        CHECK(ab.enabled, "reference helper enabled=true");
        CHECK(ab.target_entropy_error == eps,
              "reference helper threads eps_S");
        CHECK(ab.kp == 0.5, "reference kp=0.5");
        CHECK(ab.ki == 0.05, "reference ki=0.05");
        CHECK(ab.kd == 0.1, "reference kd=0.1");
        CHECK(ab.chi_floor == 4, "reference chi_floor=4");
        CHECK(ab.chi_ceiling == 4096, "reference chi_ceiling=4096");
        CHECK(ab.alpha == 8.0, "reference alpha=8");
    }

    /* ---- 4. tdvp_config_adaptive routes the helper into a full
     *        TDVP config and lifts max_bond_dim to the ceiling -------- */
    {
        const double eps = 5e-4;
        tdvp_config_t cfg = tdvp_config_adaptive(eps);
        CHECK(cfg.adaptive_bond.enabled, "adaptive cfg enabled");
        CHECK(cfg.adaptive_bond.target_entropy_error == eps,
              "adaptive cfg eps threaded");
        CHECK(cfg.max_bond_dim == cfg.adaptive_bond.chi_ceiling,
              "adaptive cfg max_bond_dim raised to chi_ceiling");
        CHECK(cfg.variant == TDVP_TWO_SITE,
              "adaptive cfg defaults to two-site");
        CHECK(cfg.svd_cutoff == 1e-10,
              "adaptive cfg keeps the legacy SVD floor as outer safety");
    }

    /* ---- 5. Struct sizes are sensible (catches accidental padding
     *        explosions if someone reorders the fields later) -------- */
    {
        const size_t cfg_size = sizeof(tdvp_config_t);
        const size_t ab_size  = sizeof(tdvp_adaptive_bond_config_t);
        /* Adaptive-bond struct: 1 bool + 5 doubles + 2 uint32_t.
         * Worst-case packing on 64-bit ABIs is 8 + 5*8 + 2*4 + 4 = 60,
         * rounded up to 64.  Allow a generous 16-byte slack. */
        CHECK(ab_size <= 80, "tdvp_adaptive_bond_config_t size sane (%zu)",
              ab_size);
        /* TDVP config: legacy fields + adaptive_bond.  Allow up to
         * 256 bytes to absorb future padding without breaking the
         * regression. */
        CHECK(cfg_size <= 256, "tdvp_config_t size sane (%zu)", cfg_size);
    }

    if (failures == 0) {
        fprintf(stdout, "PASS: 5 cases\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
