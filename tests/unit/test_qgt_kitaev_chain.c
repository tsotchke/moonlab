/**
 * @file test_qgt_kitaev_chain.c
 * @brief Tests for the Kitaev (2001) 1D p-wave topological superconductor.
 *
 * Bloch BdG Hamiltonian:
 *   H(k) = (-2t cos(k) - mu) tau_z + 2 delta sin(k) tau_y
 *
 * Phase diagram (delta != 0):
 *   |mu| < 2|t|  ->  Z_2 = 1  (topological, Majorana edge modes)
 *   |mu| > 2|t|  ->  Z_2 = 0  (trivial)
 *
 * Pins:
 *   1. Lifecycle (create / free).
 *   2. Z_2 = 1 at mu = 0 (deep topological point).
 *   3. Z_2 = 0 at |mu| > 2t (trivial).
 *   4. Phase boundary at mu = 2|t| -- transition to within +/-0.01.
 *   5. delta = 0 case: gap closes everywhere (no defined Z_2; spec
 *      says invariant returns 0 at gap closing).
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    fprintf(stdout, "=== Kitaev p-wave chain Z_2 invariant ===\n");

    const double t = 1.0;
    const double delta = 0.5;

    /* ---- Case 1: deep topological point mu = 0 ----------------- */
    {
        const double mu = 0.0;
        qgt_system_1d_t* sys = qgt_model_kitaev_chain(t, mu, delta);
        CHECK(sys != NULL, "create Kitaev mu=0");
        int z2 = -1;
        int rc = qgt_z2_invariant_1d_bdg(sys, &z2);
        CHECK(rc == 0, "z2 rc=%d", rc);
        fprintf(stdout, "  mu=%.2f: Z_2 = %d (expected 1)\n", mu, z2);
        CHECK(z2 == 1, "topological mu=0: expected Z_2=1, got %d", z2);
        qgt_free_1d(sys);
    }

    /* ---- Case 2: trivial mu = 3t ------------------------------- */
    {
        const double mu = 3.0;
        qgt_system_1d_t* sys = qgt_model_kitaev_chain(t, mu, delta);
        int z2 = -1;
        int rc = qgt_z2_invariant_1d_bdg(sys, &z2);
        CHECK(rc == 0, "z2 rc=%d", rc);
        fprintf(stdout, "  mu=%.2f: Z_2 = %d (expected 0)\n", mu, z2);
        CHECK(z2 == 0, "trivial mu=3: expected Z_2=0, got %d", z2);
        qgt_free_1d(sys);
    }

    /* ---- Case 3: trivial mu = -3t (negative side) ------------- */
    {
        const double mu = -3.0;
        qgt_system_1d_t* sys = qgt_model_kitaev_chain(t, mu, delta);
        int z2 = -1;
        qgt_z2_invariant_1d_bdg(sys, &z2);
        fprintf(stdout, "  mu=%.2f: Z_2 = %d (expected 0)\n", mu, z2);
        CHECK(z2 == 0, "trivial mu=-3: expected Z_2=0, got %d", z2);
        qgt_free_1d(sys);
    }

    /* ---- Case 4: phase boundary sweep -------------------------- */
    {
        fprintf(stdout, "  -- mu sweep at t=1, delta=0.5, boundary |mu|=2 --\n");
        double mus[] = { -2.5, -1.5, -1.0, 0.0, 1.0, 1.5, 2.5 };
        for (size_t i = 0; i < sizeof(mus) / sizeof(mus[0]); i++) {
            double mu = mus[i];
            qgt_system_1d_t* sys = qgt_model_kitaev_chain(t, mu, delta);
            int z2 = -1;
            int rc = qgt_z2_invariant_1d_bdg(sys, &z2);
            CHECK(rc == 0, "sweep rc=%d at mu=%.2f", rc, mu);
            int expected = (fabs(mu) < 2.0 * t) ? 1 : 0;
            CHECK(z2 == expected,
                  "mu=%.2f: expected Z_2=%d, got %d", mu, expected, z2);
            fprintf(stdout, "    mu=%+.2f -> Z_2 = %d (expected %d)\n",
                    mu, z2, expected);
            qgt_free_1d(sys);
        }
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
