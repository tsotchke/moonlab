/**
 * @file test_metal_mps_parity.c
 * @brief MPS (Metal-backed) vs exact statevector parity for an entangling
 *        circuit that drives the bond dimension past the Metal engagement
 *        threshold (>= 32 on macOS), so the Metal 2q-gate SVD and the Metal
 *        <Z> expectation paths are actually exercised.
 *
 * Runs the identical circuit on a dense double-precision statevector and on an
 * MPS, then compares <Z_q> per qubit. With the honest hybrid (GPU/float gate
 * contraction, CPU/double SVD + double norm-divided expectation) the two must
 * agree to single-precision tolerance. The old float2 Jacobi SVD and the
 * float2 unnormalized expectation produced garbage that this catches.
 *
 * The float32 Metal 2q-gate path is off by default (it breaks the exact
 * simulator's double-precision contract -- see tn_metal_2q_lossy_enabled in
 * tn_gates.c), so this test explicitly opts in via MOONLAB_TN_GPU_LOSSY=1 to
 * actually drive the Metal kernel it is here to validate.
 *
 * Skips cleanly (success) when Metal is unavailable at runtime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#if defined(__APPLE__) && defined(HAS_METAL)
#  include "../../src/optimization/gpu_metal.h"
#  include "../../src/quantum/state.h"
#  include "../../src/quantum/gates.h"
#  include "../../src/quantum/measurement.h"
#  include "../../src/algorithms/tensor_network/tn_state.h"
#  include "../../src/algorithms/tensor_network/tn_gates.h"
#  include "../../src/algorithms/tensor_network/tn_measurement.h"
#  define METAL_PRESENT 1
#else
#  define METAL_PRESENT 0
#endif

static int failures = 0;
#define CHECK(cond, ...) do {                                   \
    if (!(cond)) { fprintf(stderr, "  FAIL  " __VA_ARGS__);     \
                   fprintf(stderr, "\n"); failures++; }         \
    else { fprintf(stdout, "  OK    " __VA_ARGS__);             \
           fprintf(stdout, "\n"); }                             \
} while (0)

#if METAL_PRESENT

#define NQ 12
#define DEPTH 12

/* Fixed, irrational-ish angles so bond growth is deterministic and maximal. */
static double angle_for(int layer, int qubit, int which) {
    return 0.37 + 0.113 * layer + 0.071 * qubit + 0.041 * which;
}

int main(void) {
    if (!metal_is_available()) {
        printf("test_metal_mps_parity: Metal unavailable at runtime -- SKIP\n");
        return 0;
    }

    /* Opt into the lossy float32 Metal 2q-gate path (off by default) so the
     * circuit below actually runs on the Metal kernel this test validates. */
    setenv("MOONLAB_TN_GPU_LOSSY", "1", 1);

    /* Dense reference state. */
    quantum_state_t dense;
    if (quantum_state_init(&dense, NQ) != QS_SUCCESS) {
        fprintf(stderr, "dense init failed\n");
        return 1;
    }

    /* MPS state with a generous bond cap so entanglement is not truncated. */
    tn_state_config_t cfg = tn_state_config_create(64, 1e-12);
    tn_mps_state_t *mps = tn_mps_create_zero(NQ, &cfg);
    if (!mps) { fprintf(stderr, "mps create failed\n"); quantum_state_free(&dense); return 1; }

    /* Identical scrambling brickwork on both: generic single-qubit rotations
     * (RX then RY) followed by CZ on the brickwork adjacent pairs. RX/RY/CZ
     * are unambiguous across the dense and MPS APIs, and this kicked-Ising-
     * style circuit drives the bond dimension into the Metal regime. */
    for (int layer = 0; layer < DEPTH; layer++) {
        for (int q = 0; q < NQ; q++) {
            double tx = angle_for(layer, q, 0);
            double ty = angle_for(layer, q, 1);
            gate_rx(&dense, q, tx);  tn_apply_rx(mps, (uint32_t)q, tx);
            gate_ry(&dense, q, ty);  tn_apply_ry(mps, (uint32_t)q, ty);
        }
        int start = layer & 1;   /* alternate even/odd bonds (brickwork) */
        for (int q = start; q + 1 < NQ; q += 2) {
            gate_cz(&dense, q, q + 1);
            tn_apply_cz(mps, (uint32_t)q, (uint32_t)(q + 1));
        }
    }

    /* Confirm the bond dimension actually entered the Metal regime. */
    uint32_t maxb = tn_mps_max_bond_dim(mps);
    printf("  info  MPS max bond dimension reached: %u\n", (unsigned)maxb);
    CHECK(maxb >= 32, "bond dimension reached the Metal engagement threshold (>= 32)");

    /* Per-qubit <Z> parity. */
    double max_diff = 0.0;
    for (int q = 0; q < NQ; q++) {
        double zdense = measurement_expectation_z(&dense, q);
        double zmps = tn_expectation_z(mps, (uint32_t)q);
        double d = fabs(zdense - zmps);
        if (d > max_diff) max_diff = d;
    }
    printf("  info  max |<Z>_dense - <Z>_mps| = %.3e\n", max_diff);
    CHECK(max_diff < 5e-3, "MPS <Z> matches exact statevector to single precision (max diff %.3e)", max_diff);

    tn_mps_free(mps);
    quantum_state_free(&dense);

    if (failures == 0) { printf("test_metal_mps_parity: ALL PASSED\n"); return 0; }
    fprintf(stderr, "test_metal_mps_parity: %d FAILED\n", failures);
    return 1;
}

#else  /* !METAL_PRESENT */

int main(void) {
    printf("test_metal_mps_parity: built without Metal -- SKIP\n");
    return 0;
}

#endif
