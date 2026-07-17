/**
 * @file test_tn_mps_interior_gate.c
 * @brief Regression: a 2q gate on an interior bond must preserve the norm.
 *
 * apply_gate_2q_adjacent rescales the two-site block's singular values by
 * ||theta||_F and parks log(||theta||_F) in log_norm_factor -- correct only when
 * the gate bond is the orthogonality center (||theta||_F == the physical Schmidt
 * norm == 1).  The gate path leaves the MPS in TN_CANONICAL_NONE, so a gate that
 * lands on an interior bond whose BOTH outer bonds are already entangled had
 * ||theta||_F != 1 and silently moved physical norm into log_norm_factor,
 * halving every probability.
 *
 * Minimal pure-Clifford, all-adjacent, CPU repro (no Metal): the state
 * H(0) CX(0,1) H(2) CX(2,3) CZ(1,2) is a valid stabilizer state with unit norm;
 * the CZ lands on the interior bond (1,2) whose outer bonds are both non-trivial.
 * Before the fix the read-out norm collapsed to 0.5.
 */

#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <complex.h>
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
    fprintf(stdout, "=== 2q gate on interior bond preserves norm ===\n");

    const uint32_t n = 4;
    tn_state_config_t cfg = tn_state_config_create(64, 1e-14);
    tn_mps_state_t *s = tn_mps_create_zero(n, &cfg);

    tn_apply_h(s, 0);
    tn_apply_cnot(s, 0, 1);
    tn_apply_h(s, 2);
    tn_apply_cnot(s, 2, 3);
    tn_apply_cz(s, 1, 2);      /* interior bond, both outer bonds entangled */

    uint64_t dim = 1ULL << n;
    double complex *sv = calloc(dim, sizeof(double complex));
    tn_mps_to_statevector(s, sv);

    double norm_sq = 0.0;
    for (uint64_t b = 0; b < dim; b++)
        norm_sq += creal(sv[b]) * creal(sv[b]) + cimag(sv[b]) * cimag(sv[b]);

    fprintf(stdout, "  read-out norm^2 = %.12f (expect 1)\n", norm_sq);
    CHECK(fabs(norm_sq - 1.0) < 1e-9,
          "state-vector norm^2 %.9f != 1 (interior-bond gate corrupted the norm)", norm_sq);

    /* Two Bell pairs (0,1) and (2,3) give |0000>+|0011>+|1100>+|1111> (4 equal
     * amplitudes); CZ(1,2) only adds a phase to |1111>, so the physical state
     * has exactly 4 populated basis states each with probability 1/4.  Before
     * the fix each was halved to 1/8 (total norm 0.5). */
    int populated = 0;
    double maxdev = 0.0;
    for (uint64_t b = 0; b < dim; b++) {
        double p = creal(sv[b]) * creal(sv[b]) + cimag(sv[b]) * cimag(sv[b]);
        if (p > 1e-6) {
            populated++;
            double dev = fabs(p - 0.25);
            if (dev > maxdev) maxdev = dev;
        }
    }
    fprintf(stdout, "  populated basis states = %d (expect 4), max |p - 1/4| = %.3e\n",
            populated, maxdev);
    CHECK(populated == 4, "expected 4 populated basis states, got %d", populated);
    CHECK(maxdev < 1e-9, "populated probabilities deviate from 1/4 by %.3e", maxdev);

    free(sv);
    tn_mps_free(s);

    if (failures == 0) { fprintf(stdout, "PASS\n"); return 0; }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
