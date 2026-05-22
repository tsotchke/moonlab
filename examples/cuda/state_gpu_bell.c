/**
 * @file  state_gpu_bell.c
 * @brief Proof that v1.1 step 9 "transparent GPU dispatch" works.
 *
 * Constructs a Bell state |Phi+> using the SAME `gate_hadamard()` +
 * `gate_cnot()` API that 350+ existing CPU tests use.  The only
 * difference is the constructor: quantum_state_create_gpu() instead
 * of quantum_state_create().  After the circuit runs, we copy the
 * GPU amplitudes back to host and assert exactly the same invariants
 * the CPU Bell circuit would.
 *
 * This validates that *the existing algorithm surface* is now
 * GPU-accelerated -- not a new GPU-specific API.  Any code that
 * uses gate_hadamard / gate_cnot automatically gets the GPU when
 * the state was created via the new constructor.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 16;
    printf("=== moonlab state_gpu Bell at N=%d ===\n", N);
    printf("    GPU kind: %s\n",
        moonlab_gpu_probe_kind_str(moonlab_gpu_probe_kind()));

    quantum_state_t *s = NULL;
    qs_error_t err = quantum_state_create_gpu((size_t)N, &s);
    if (err != QS_SUCCESS) {
        fprintf(stderr, "quantum_state_create_gpu failed: %d\n", (int)err);
        return 1;
    }
    if (!s->gpu_state) {
        fprintf(stderr, "state has no GPU backing\n");
        return 2;
    }

    /* Use the SAME API as the CPU tests.  The dispatch inside
     * gate_hadamard() / gate_cnot() routes to CUDA because
     * s->gpu_state is non-NULL.  No "use_gpu" flag, no special
     * call -- it just happens. */
    err = gate_hadamard(s, 0);
    if (err != QS_SUCCESS) { fprintf(stderr, "gate_hadamard failed: %d\n", err); return 3; }
    err = gate_cnot(s, 0, 1);
    if (err != QS_SUCCESS) { fprintf(stderr, "gate_cnot failed: %d\n", err); return 4; }

    /* Pull GPU state back into host buffer so we can inspect. */
    err = quantum_state_sync_to_host(s);
    if (err != QS_SUCCESS) { fprintf(stderr, "sync_to_host failed: %d\n", err); return 5; }

    /* Bell |Phi+> = (|00> + |11>)/sqrt(2).  After sync:
     *   amplitudes[0]               = 1/sqrt(2)
     *   amplitudes[1] ... [dim-2]   = 0
     *   amplitudes[1 | (1<<1)]      = 1/sqrt(2)
     */
    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    double a0 = cabs(s->amplitudes[0]);
    double a3 = cabs(s->amplitudes[(size_t)1 | ((size_t)1 << 1)]);
    /* Norm: sum |amps|^2 over the whole buffer. */
    double norm = 0.0;
    size_t dim = (size_t)1 << N;
    for (size_t k = 0; k < dim; k++) {
        double x = creal(s->amplitudes[k]);
        double y = cimag(s->amplitudes[k]);
        norm += x*x + y*y;
    }

    printf("    |amp[00]| = %.6f  (expected %.6f)\n", a0, inv_sqrt2);
    printf("    |amp[11]| = %.6f  (expected %.6f)\n", a3, inv_sqrt2);
    printf("    norm      = %.12f  (expected 1.0)\n", norm);

    int ok = fabs(a0   - inv_sqrt2) < 1e-9
          && fabs(a3   - inv_sqrt2) < 1e-9
          && fabs(norm - 1.0)       < 1e-12;
    printf("    state_gpu Bell via gate_hadamard+gate_cnot: %s\n",
        ok ? "PASS" : "FAIL");

    quantum_state_destroy(s);
    return ok ? 0 : 6;
}
