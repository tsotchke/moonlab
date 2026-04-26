/**
 * @file test_ca_mps_prob.c
 * @brief Smoke test for moonlab_ca_mps_prob_z and moonlab_ca_mps_t_dagger
 *        against the dense state-vector backend.  Runs a deterministic
 *        H + T + T-dagger + S + CNOT circuit on n=4 and verifies every
 *        single-qubit Z probability agrees to <1e-12.
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

    /* Build a deterministic circuit:
     *   H q0; H q1; T q0; T_dag q1; S q2; CNOT q0 q2; H q3; T q3; T_dag q3 */
    moonlab_ca_mps_h(ca, 0);            gate_hadamard(sv, 0);
    moonlab_ca_mps_h(ca, 1);            gate_hadamard(sv, 1);
    moonlab_ca_mps_t_gate(ca, 0);       gate_t(sv, 0);
    moonlab_ca_mps_t_dagger(ca, 1);     gate_t_dagger(sv, 1);
    moonlab_ca_mps_s(ca, 2);            gate_s(sv, 2);
    moonlab_ca_mps_cnot(ca, 0, 2);      gate_cnot(sv, 0, 2);
    moonlab_ca_mps_h(ca, 3);            gate_hadamard(sv, 3);
    moonlab_ca_mps_t_gate(ca, 3);       gate_t(sv, 3);
    moonlab_ca_mps_t_dagger(ca, 3);     gate_t_dagger(sv, 3);
    /* T then T-dagger = identity, so q3 should be back to |+> after H. */

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
    /* Specifically check q3: H then T then T-dag is exactly H|0> = |+>,
     * so P(Z=+1) = 0.5. */
    double pca3;
    moonlab_ca_mps_prob_z(ca, 3, &pca3);
    if (fabs(pca3 - 0.5) > 1e-12) {
        fprintf(stderr, "FAIL  q=3 should be exactly 0.5 (H|0>=|+>); got %.12f\n", pca3);
        failures++;
    }

    moonlab_ca_mps_free(ca);
    quantum_state_free(sv);
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
