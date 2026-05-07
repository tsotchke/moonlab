/**
 * @file bench_apply_conjugated_imag.c
 * @brief Profile the imag-time sweep at chi=128.
 *
 * Open v0.3 audit finding. apply_conjugated_imag (in ca_mps.c) is the hot
 * path for var-D's |phi>-update: it builds an MPO of bond dim <=2 and runs
 * tn_apply_mpo with the caller's chi cap. For Hamiltonians dominated by
 * weight-1 (single-site fields) and weight-2 (two-site bonds) Pauli strings
 * the MPO machinery is unnecessary overhead — those are just 1q/2q gates.
 *
 * This bench measures the actual per-call cost on a realistic workload
 * (n=18 XXZ chain, 51 Pauli terms / sweep) so we can quantify the win
 * before/after a fast path.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* XXZ on n sites: 3*(n-1) Pauli terms (XX, YY, Delta*ZZ per bond). */
static void build_xxz(uint32_t n, double Delta,
                       uint8_t** out_paulis, double** out_coeffs,
                       uint32_t* out_num_terms) {
    uint32_t T = 3 * (n - 1);
    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, 1);
    double*  coeffs = (double*)calloc(T, sizeof(double));
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        paulis[k * n + i]     = 1; paulis[k * n + i + 1] = 1; coeffs[k] = 1.0;   k++;
        paulis[k * n + i]     = 2; paulis[k * n + i + 1] = 2; coeffs[k] = 1.0;   k++;
        paulis[k * n + i]     = 3; paulis[k * n + i + 1] = 3; coeffs[k] = Delta; k++;
    }
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = T;
}

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : 18u;
    const uint32_t chi = (argc > 2) ? (uint32_t)atoi(argv[2]) : 128u;
    const int outer_iters = (argc > 3) ? atoi(argv[3]) : 6;
    const double Delta = 1.0;

    fprintf(stdout, "bench_apply_conjugated_imag: n=%u chi=%u outer=%d Delta=%.2f\n",
            n, chi, outer_iters, Delta);

    uint8_t* paulis;
    double*  coeffs;
    uint32_t T;
    build_xxz(n, Delta, &paulis, &coeffs, &T);
    fprintf(stdout, "  num_terms=%u (%u XX + %u YY + %u ZZ bonds)\n",
            T, n - 1, n - 1, n - 1);

    moonlab_ca_mps_t* state = moonlab_ca_mps_create(n, chi);
    if (!state) { fprintf(stderr, "create failed\n"); return 1; }

    /* Trotter step + outer-iter count match the var-D defaults. */
    ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
    cfg.verbose                     = 1;
    cfg.max_outer_iters             = outer_iters;
    cfg.imag_time_dtau              = 0.1;
    cfg.imag_time_steps_per_outer   = 5;
    cfg.clifford_passes_per_outer   = 0;  /* isolate the imag-time path */
    cfg.convergence_eps             = 0.0;
    cfg.warmstart                   = CA_MPS_WARMSTART_IDENTITY;

    ca_mps_var_d_alt_result_t res = {0};
    double t0 = now_ms();
    ca_mps_error_t e = moonlab_ca_mps_optimize_var_d_alternating(
        state, paulis, coeffs, T, &cfg, &res);
    double dt = now_ms() - t0;
    if (e != CA_MPS_SUCCESS) {
        fprintf(stderr, "optimize failed: %d\n", (int)e);
        return 1;
    }

    long total_calls = (long)outer_iters * cfg.imag_time_steps_per_outer * T;
    fprintf(stdout, "  total wall: %.2f ms\n", dt);
    fprintf(stdout, "  imag-time calls: %ld (outer * trotter * terms)\n", total_calls);
    fprintf(stdout, "  per-call avg: %.4f ms\n", dt / (double)total_calls);
    fprintf(stdout, "  E_initial=%.6f  E_final=%.6f\n",
            res.initial_energy, res.final_energy);
    fprintf(stdout, "  S(phi) initial=%.4f final=%.4f\n",
            res.initial_phi_entropy, res.final_phi_entropy);

    moonlab_ca_mps_free(state);
    free(paulis);
    free(coeffs);
    return 0;
}
