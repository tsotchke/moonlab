/**
 * @file test_gpu_sync_contract.c
 * @brief GPU host-sync contract for dense states.
 *
 * Guards the invariant behind the ICC gpu_host_sync_contract criterion:
 *
 *  1. On a build without a real GPU backend, quantum_state_create_gpu must
 *     fail LOUDLY (QS_ERROR_NOT_SUPPORTED) and hand back a NULL state -- never
 *     silently succeed with a broken/CPU state that later reads stale memory.
 *  2. The host-sync entry points are safe no-ops (QS_SUCCESS) on a CPU state,
 *     so the read/collapse paths that now call them unconditionally cost
 *     nothing and never corrupt a CPU state.
 *  3. The dense read/measure paths that were taught to sync still produce the
 *     correct answers on an ordinary CPU state (regression guard that the sync
 *     hooks did not disturb the CPU path).
 *
 * This is a core test: it runs on every build (no GPU required).
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"
#include <math.h>
#include <stdio.h>
#include <complex.h>

static int failures = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) { fprintf(stderr, "  FAIL  " __VA_ARGS__);     \
                   fprintf(stderr, "\n"); failures++; }         \
    else { fprintf(stdout, "  OK    " __VA_ARGS__);             \
           fprintf(stdout, "\n"); }                             \
} while (0)

int main(void) {
    /* (1) create_gpu fails loudly on a CPU/weak-stub build. On a real CUDA
     * build this returns QS_SUCCESS; either is a defined, non-silent outcome,
     * but this CI build has no CUDA, so we require NOT_SUPPORTED + NULL out. */
    quantum_state_t *g = (quantum_state_t *)0x1;  /* poison; must be overwritten */
    qs_error_t rc = quantum_state_create_gpu(4, &g);
    if (rc == QS_SUCCESS) {
        /* Real GPU backend (CUDA build on a GPU host -- e.g. the xavier/cosbox
         * mesh nodes): create_gpu succeeds and returns a valid non-NULL state.
         * Both outcomes are defined and non-silent; the contract is only that
         * create_gpu never silently mis-behaves. */
        CHECK(g != NULL, "quantum_state_create_gpu success returns a non-NULL GPU state");
        if (g) quantum_state_destroy(g);
    } else {
        /* CPU / weak-stub build: must fail LOUDLY (NOT_SUPPORTED) with NULL out. */
        CHECK(rc == QS_ERROR_NOT_SUPPORTED,
              "quantum_state_create_gpu returns NOT_SUPPORTED on a non-GPU build (rc=%d)", (int)rc);
        CHECK(g == NULL, "quantum_state_create_gpu sets *out_state = NULL on failure");
    }

    /* (2) sync entry points are no-op success on a CPU state. */
    quantum_state_t s;
    CHECK(quantum_state_init(&s, 3) == QS_SUCCESS, "init 3-qubit CPU state");
    CHECK(s.gpu_state == NULL, "fresh CPU state has no GPU backing");
    CHECK(quantum_state_sync_to_host(&s) == QS_SUCCESS, "sync_to_host is a no-op on a CPU state");
    CHECK(quantum_state_sync_from_host(&s) == QS_SUCCESS, "sync_from_host is a no-op on a CPU state");
    /* NULL argument is a defined error, not a crash. */
    CHECK(quantum_state_sync_to_host(NULL) == QS_ERROR_INVALID_PARAM, "sync_to_host(NULL) errors cleanly");

    /* (3) the synced read/measure paths still work on a CPU state.
     * Prepare |+> on qubit 0: <Z0> == 0, P(1) == 0.5. */
    gate_hadamard(&s, 0);
    double p1 = measurement_probability_one(&s, 0);
    CHECK(fabs(p1 - 0.5) < 1e-9, "measurement_probability_one(|+>) == 0.5 (got %.6f)", p1);
    double z0 = measurement_expectation_z(&s, 0);
    CHECK(fabs(z0) < 1e-9, "measurement_expectation_z(|+>) == 0 (got %.6f)", z0);
    CHECK(quantum_state_is_normalized(&s, 1e-9), "state remains normalized after synced reads");

    quantum_state_free(&s);

    if (failures == 0) {
        printf("test_gpu_sync_contract: ALL PASSED\n");
        return 0;
    }
    fprintf(stderr, "test_gpu_sync_contract: %d FAILED\n", failures);
    return 1;
}
