/**
 * @file  state_gpu_circuit.c
 * @brief v1.1 step 9b validation: full single+two-qubit gate
 *        surface dispatches transparently to GPU.
 *
 * Runs the same multi-gate circuit twice, once on a CPU-backed
 * quantum_state_t and once on a GPU-backed one, and confirms the
 * amplitudes agree to 1e-10.  If the dispatch is broken (or any
 * gate goes through the CPU path on the GPU state, leaving the
 * GPU buffer untouched), the comparison fails.
 *
 * Circuit (covers the new GPU-routed gates):
 *   H 0
 *   X 1
 *   Y 2
 *   Z 3
 *   S 0     ; S^dagger 1
 *   T 2     ; T^dagger 3
 *   RX(0.7) 0 ; RY(1.1) 1 ; RZ(0.5) 2
 *   PHASE(0.3) 3
 *   CNOT 0 1
 *   CZ 1 2
 *   SWAP 2 3
 *   CPHASE(0.4) 0 3
 *
 * That's 15 gates, hits H + X + Y + Z + S + S^dag + T + T^dag +
 * RX + RY + RZ + phase + CNOT + CZ + SWAP + cphase = the entire
 * step-9b dispatch set.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void run_circuit(quantum_state_t *s) {
    gate_hadamard (s, 0);
    gate_pauli_x  (s, 1);
    gate_pauli_y  (s, 2);
    gate_pauli_z  (s, 3);
    gate_s        (s, 0);
    gate_s_dagger (s, 1);
    gate_t        (s, 2);
    gate_t_dagger (s, 3);
    gate_rx       (s, 0, 0.7);
    gate_ry       (s, 1, 1.1);
    gate_rz       (s, 2, 0.5);
    gate_phase    (s, 3, 0.3);
    gate_cnot     (s, 0, 1);
    gate_cz       (s, 1, 2);
    gate_swap     (s, 2, 3);
    gate_cphase   (s, 0, 3, 0.4);
    /* Step 9c additions: controlled-rotation surface via decomposition. */
    gate_cy       (s, 0, 2);
    gate_crx      (s, 1, 3, 0.6);
    gate_cry      (s, 2, 0, 0.9);
    gate_crz      (s, 3, 1, 1.2);
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 8;
    if (N < 4) { fprintf(stderr, "need N>=4\n"); return 1; }

    printf("=== moonlab step-9b GPU vs CPU parity at N=%d ===\n", N);
    printf("    GPU kind: %s\n",
        moonlab_gpu_probe_kind_str(moonlab_gpu_probe_kind()));

    /* CPU run. */
    quantum_state_t *cpu = quantum_state_create(N);
    if (!cpu) { fprintf(stderr, "cpu create failed\n"); return 2; }
    run_circuit(cpu);

    /* GPU run. */
    quantum_state_t *gpu = NULL;
    qs_error_t err = quantum_state_create_gpu((size_t)N, &gpu);
    if (err != QS_SUCCESS) {
        fprintf(stderr, "gpu create failed: %d\n", (int)err);
        quantum_state_destroy(cpu);
        return 3;
    }
    run_circuit(gpu);
    quantum_state_sync_to_host(gpu);

    /* Compare. */
    size_t dim = (size_t)1 << N;
    double max_err = 0.0;
    double cpu_norm = 0.0, gpu_norm = 0.0;
    for (size_t k = 0; k < dim; k++) {
        complex_t diff = cpu->amplitudes[k] - gpu->amplitudes[k];
        double e = cabs(diff);
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
    printf("    step-9b 16-gate circuit: %s (tol=%.0e)\n",
        ok ? "PASS" : "FAIL", tol);

    quantum_state_destroy(gpu);
    quantum_state_destroy(cpu);
    return ok ? 0 : 4;
}
