/**
 * @file  state_gpu_multiq.c
 * @brief v1.1 follow-up #4 verification: Toffoli / Fredkin / MCX / MCZ
 *        dispatch via the new multi-control CUDA kernels.
 *
 * Runs the same multi-qubit circuit on a CPU state and a GPU state,
 * syncs the GPU state to host, and asserts amplitudes agree at
 * machine precision.  Tests:
 *   - Toffoli (CCX) at multiple (c1, c2, t) orderings
 *   - Fredkin (CSWAP) at multiple control / target orderings
 *   - MCX with 3 controls
 *   - MCZ with 4 controls
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void run_circuit(quantum_state_t *s) {
    /* Seed with some non-trivial state: H on every qubit + a phase
     * on q0 to break symmetry, so the dispatched gates have actual
     * amplitude to act on. */
    for (int q = 0; q < (int)s->num_qubits; q++) gate_hadamard(s, q);
    gate_t(s, 0);

    /* Toffoli sweep: various orderings of (c1, c2, t). */
    gate_toffoli(s, 0, 1, 2);
    gate_toffoli(s, 2, 3, 0);
    gate_toffoli(s, 5, 0, 3);   /* mixed ordering */

    /* Fredkin sweep. */
    gate_fredkin(s, 0, 1, 4);
    gate_fredkin(s, 3, 5, 2);

    /* MCX with 3 controls. */
    const int mcx_ctrls[3] = { 1, 3, 5 };
    gate_mcx(s, mcx_ctrls, 3, 0);

    /* MCZ with 4 controls -- target 5. */
    const int mcz_ctrls[4] = { 0, 1, 2, 3 };
    gate_mcz(s, mcz_ctrls, 4, 5);
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 8;
    if (N < 6) { fprintf(stderr, "need N>=6\n"); return 1; }

    printf("=== moonlab step-9d multi-qubit GPU vs CPU parity at N=%d ===\n", N);
    printf("    GPU kind: %s\n",
        moonlab_gpu_probe_kind_str(moonlab_gpu_probe_kind()));

    quantum_state_t *cpu = quantum_state_create(N);
    run_circuit(cpu);

    quantum_state_t *gpu = NULL;
    qs_error_t err = quantum_state_create_gpu((size_t)N, &gpu);
    if (err != QS_SUCCESS) {
        fprintf(stderr, "gpu create failed: %d\n", (int)err);
        return 2;
    }
    run_circuit(gpu);
    quantum_state_sync_to_host(gpu);

    size_t dim = (size_t)1 << N;
    double max_err = 0.0, cpu_norm = 0.0, gpu_norm = 0.0;
    for (size_t k = 0; k < dim; k++) {
        double e = cabs(cpu->amplitudes[k] - gpu->amplitudes[k]);
        if (e > max_err) max_err = e;
        cpu_norm += cabs(cpu->amplitudes[k]) * cabs(cpu->amplitudes[k]);
        gpu_norm += cabs(gpu->amplitudes[k]) * cabs(gpu->amplitudes[k]);
    }
    printf("    cpu_norm = %.12f\n", cpu_norm);
    printf("    gpu_norm = %.12f\n", gpu_norm);
    printf("    max |amp_cpu - amp_gpu| = %.3e\n", max_err);

    const double tol = 1e-10;
    int ok = max_err < tol
          && fabs(cpu_norm - 1.0) < 1e-12
          && fabs(gpu_norm - 1.0) < 1e-12;
    printf("    Toffoli+Fredkin+MCX+MCZ via GPU kernels: %s\n",
        ok ? "PASS" : "FAIL");

    quantum_state_destroy(gpu);
    quantum_state_destroy(cpu);
    return ok ? 0 : 4;
}
