/**
 * @file test_ca_mps_prob.c
 * @brief Smoke test for the convenience CA-MPS API additions
 *        (T-dagger, phase, CRZ, CRX, CRY, prob_z) against the dense
 *        state-vector backend.  Runs a deterministic mixed circuit on
 *        n=4 and verifies every single-qubit Z probability agrees to
 *        <1e-12 -- catches sign-convention bugs in the CR* decomps.
 */
#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static double sv_prob_z(const quantum_state_t* sv, int q) {
    /* P(Z_q = +1) = sum_{bit q = 0} |a_s|^2 */
    double p0 = 0.0;
    for (size_t s = 0; s < sv->state_dim; s++) {
        if (((s >> q) & 1u) == 0u) {
            double re = creal(sv->amplitudes[s]);
            double im = cimag(sv->amplitudes[s]);
            p0 += re * re + im * im;
        }
    }
    return p0;
}

int main(void) {
    const uint32_t n = 4;
    moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, 16);
    quantum_state_t*  sv = quantum_state_create((int)n);
    if (!ca || !sv) return 2;

    /* Build a deterministic mixed circuit covering every new primitive.
     * Goal: non-trivial entanglement + non-Clifford phase that exercises
     * the CR* phase-kickback decomps. */
    moonlab_ca_mps_h(ca, 0);                   gate_hadamard(sv, 0);
    moonlab_ca_mps_h(ca, 1);                   gate_hadamard(sv, 1);
    moonlab_ca_mps_t_gate(ca, 0);              gate_t(sv, 0);
    moonlab_ca_mps_t_dagger(ca, 1);            gate_t_dagger(sv, 1);
    moonlab_ca_mps_phase(ca, 2, 0.7);          gate_phase(sv, 2, 0.7);
    moonlab_ca_mps_crz(ca, 0, 2, 0.4);         gate_crz(sv, 0, 2, 0.4);
    moonlab_ca_mps_crx(ca, 1, 3, 0.9);         gate_crx(sv, 1, 3, 0.9);
    moonlab_ca_mps_cry(ca, 2, 3, -0.6);        gate_cry(sv, 2, 3, -0.6);
    moonlab_ca_mps_cnot(ca, 0, 1);             gate_cnot(sv, 0, 1);
    moonlab_ca_mps_t_gate(ca, 3);              gate_t(sv, 3);
    moonlab_ca_mps_t_dagger(ca, 3);            gate_t_dagger(sv, 3);
    moonlab_ca_mps_phase(ca, 0, -0.3);         gate_phase(sv, 0, -0.3);
    moonlab_ca_mps_u3(ca, 1, 0.8, -1.2, 0.5);  gate_u3(sv, 1, 0.8, -1.2, 0.5);
    moonlab_ca_mps_u3(ca, 2, 1.5, 0.0, -0.7);  gate_u3(sv, 2, 1.5, 0.0, -0.7);
    /* Exercise the new 3-qubit gates. */
    moonlab_ca_mps_toffoli(ca, 0, 1, 2);       gate_toffoli(sv, 0, 1, 2);
    moonlab_ca_mps_fredkin(ca, 3, 0, 1);       gate_fredkin(sv, 3, 0, 1);

    for (uint32_t q = 0; q < n; q++) {
        double pca, psv = sv_prob_z(sv, (int)q);
        ca_mps_error_t e = moonlab_ca_mps_prob_z(ca, q, &pca);
        if (e != CA_MPS_SUCCESS) {
            fprintf(stderr, "FAIL prob_z(q=%u) returned err %d\n", q, e);
            failures++; continue;
        }
        double err = fabs(pca - psv);
        if (err > 1e-12) {
            fprintf(stderr, "FAIL  q=%u  CA=%.12f  SV=%.12f  err=%.2e\n",
                    q, pca, psv, err);
            failures++;
        } else {
            fprintf(stdout, "  OK  q=%u  P(Z=+1)=%.6f  err=%.2e\n", q, pca, err);
        }
    }
    moonlab_ca_mps_free(ca);
    quantum_state_free(sv);
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
