/**
 * @file bench_warmstart_empirical_entropy.c
 * @brief Empirical S(|phi_0>) on physical Z2 LGT ground states.
 *
 * Theorem 1's bound min(|A|-k_A, |B|-k_B) log 2 is an *upper* bound
 * over the worst-case state in the warmstart-stabilised subspace
 * V_phi_0.  Physical ground states (e.g. confining-phase Z2 LGT)
 * typically come in well below the worst case.  This harness
 * measures, for each N:
 *
 *   - S_warmstart  -- the |phi> entropy right after the warmstart
 *                     (before any imag-time evolution).  Should be 0
 *                     since |phi> = |0^n> is a product state.
 *   - S_converged  -- the |phi> entropy at imag-time convergence
 *                     against the Z2 LGT Hamiltonian.
 *   - S_upper      -- the Theorem 1 upper bound from the
 *                     pivot-distribution scaling sweep.
 *
 * The "tightness" of the bound is S_converged / S_upper.  When this
 * is < 1 the warmstart provides headroom; when ~ 1 the bound is
 * being saturated by the workload.
 *
 * Code paths:
 *   moonlab_ca_mps_optimize_var_d_alternating
 *     with warmstart = STABILIZER_SUBGROUP and
 *     imag_time_steps_per_outer chosen large enough to converge.
 *
 * Output: per-N row to stdout + JSON archive at argv[1].
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/applications/hep/lattice_z2_1d.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t N;
    uint32_t n;
    uint32_t k;
    /* run config */
    double mass;
    double t_hop;
    double h_link;
    double gauss_penalty;
    /* outputs */
    double initial_phi_entropy;
    double final_phi_entropy;
    double final_energy;
    int    outer_iterations;
    double walltime_s;
    /* upper bound from Theorem 1 */
    int upper_bound_log2;
} empirical_record_t;

static int run_one(uint32_t N, double mass, double t_hop,
                   double h_link, double gauss_penalty,
                   empirical_record_t* out) {
    memset(out, 0, sizeof(*out));
    out->N = N;
    out->mass = mass;
    out->t_hop = t_hop;
    out->h_link = h_link;
    out->gauss_penalty = gauss_penalty;

    z2_lgt_config_t cfg = {0};
    cfg.num_matter_sites = N;
    cfg.t_hop = t_hop; cfg.h_link = h_link; cfg.mass = mass;
    cfg.gauss_penalty = gauss_penalty;

    /* Build the Hamiltonian Pauli sum. */
    uint8_t* paulis_h = NULL;
    double*  coeffs   = NULL;
    uint32_t T = 0, nq = 0;
    if (z2_lgt_1d_build_pauli_sum(&cfg, &paulis_h, &coeffs, &T, &nq) != 0) {
        return -1;
    }
    out->n = nq;

    /* Build the gauge-law generators (k = N-2 interior). */
    uint32_t k = (N >= 2) ? (N - 2) : 0;
    if (k == 0) { free(paulis_h); free(coeffs); return -1; }
    out->k = k;
    uint8_t* gens = (uint8_t*)calloc((size_t)k * nq, 1);
    for (uint32_t i = 0; i < k; i++) {
        z2_lgt_1d_gauss_law_pauli(&cfg, i + 1, &gens[(size_t)i * nq]);
    }

    /* Configure var-D. */
    moonlab_ca_mps_t* s = moonlab_ca_mps_create(nq, 32);
    ca_mps_var_d_alt_config_t acfg = ca_mps_var_d_alt_config_default();
    acfg.warmstart                 = CA_MPS_WARMSTART_STABILIZER_SUBGROUP;
    acfg.warmstart_stab_paulis     = gens;
    acfg.warmstart_stab_num_gens   = k;
    acfg.max_outer_iters           = 2;
    acfg.imag_time_steps_per_outer = 30;
    acfg.imag_time_dtau            = 0.05;
    acfg.clifford_passes_per_outer = 0;   /* no Clifford updates: pin
                                           * the warmstart's D, just
                                           * evolve |phi> under H */
    acfg.convergence_eps           = 1e-5;
    acfg.verbose                   = 0;

    ca_mps_var_d_alt_result_t res = {0};
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ca_mps_error_t err = moonlab_ca_mps_optimize_var_d_alternating(
        s, paulis_h, coeffs, T, &acfg, &res);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    out->walltime_s =
        (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    if (err != CA_MPS_SUCCESS) {
        fprintf(stderr, "[N=%u] var-D failed: %d\n", (unsigned)N, (int)err);
        moonlab_ca_mps_free(s);
        free(gens); free(paulis_h); free(coeffs);
        return -1;
    }

    out->initial_phi_entropy = res.initial_phi_entropy;
    out->final_phi_entropy   = res.final_phi_entropy;
    out->final_energy        = res.final_energy;
    out->outer_iterations    = res.outer_iterations;

    /* Theorem 1 upper bound: min(|A|-k_A, |B|-k_B) for half-cut.  For
     * Z2 LGT N: pivots are at link qubits, distributing roughly
     * uniformly, half-cut split (k_A, k_B) ~ (k/2, k/2). */
    uint32_t cut = nq / 2;
    uint32_t kA = k / 2, kB = k - kA;
    uint32_t free_l = cut    - kA;
    uint32_t free_r = (nq - cut) - kB;
    out->upper_bound_log2 = (int)((free_l < free_r) ? free_l : free_r);

    moonlab_ca_mps_free(s);
    free(gens); free(paulis_h); free(coeffs);
    return 0;
}

int main(int argc, char** argv) {
    empirical_record_t recs[16];
    size_t n_rec = 0;

    /* Z2 LGT confining-phase parameters: large gauss penalty, dominant
     * mass + electric field, suppressed hopping.  The ground state has
     * the matter qubits roughly fixed and the gauge qubits near
     * vacuum. */
    const double mass = 0.5;
    const double t_hop = 0.5;
    const double h_link = 1.5;
    const double gauss_penalty = 5.0;

    printf("\n%-3s | %-3s | %-3s | %-8s | %-10s | %-10s | %-9s | %-7s\n",
           "N", "n", "k", "S(unrot)", "S(upper)", "S(converged)",
           "S/S_upper", "wall_s");
    printf("----+-----+-----+----------+------------+------------+"
           "-----------+--------\n");

    const uint32_t Ns[] = {4, 6, 8};
    for (size_t i = 0; i < sizeof(Ns) / sizeof(Ns[0]); i++) {
        if (run_one(Ns[i], mass, t_hop, h_link, gauss_penalty,
                    &recs[n_rec]) != 0) continue;
        empirical_record_t* r = &recs[n_rec];
        double tightness = (r->upper_bound_log2 > 0)
            ? (r->final_phi_entropy) / (double)r->upper_bound_log2
            : 0.0;
        double unrot = (r->n / 2 < r->n - r->n / 2) ? r->n / 2 : r->n - r->n / 2;
        printf("%3u | %3u | %3u | %8.2f | %10d | %10.4f | %9.3f | %7.2f\n",
               (unsigned)r->N, (unsigned)r->n, (unsigned)r->k,
               unrot, r->upper_bound_log2,
               r->final_phi_entropy, tightness,
               r->walltime_s);
        n_rec++;
    }

    if (argc >= 2) {
        FILE* f = fopen(argv[1], "w");
        if (!f) { fprintf(stderr, "could not open %s\n", argv[1]); return 1; }
        fprintf(f, "{\n");
        fprintf(f,
            "  \"harness\": \"bench_warmstart_empirical_entropy\",\n"
            "  \"description\": \"Empirical S(|phi_0>) on Z2 LGT confining-phase ground state\",\n"
            "  \"parameters\": {\"mass\": %g, \"t_hop\": %g, \"h_link\": %g, "
            "\"gauss_penalty\": %g},\n"
            "  \"records\": [\n",
            mass, t_hop, h_link, gauss_penalty);
        for (size_t i = 0; i < n_rec; i++) {
            empirical_record_t* r = &recs[i];
            fprintf(f, "    {\"N\":%u, \"n\":%u, \"k\":%u, ",
                    (unsigned)r->N, (unsigned)r->n, (unsigned)r->k);
            fprintf(f, "\"final_energy\":%.10g, ", r->final_energy);
            fprintf(f, "\"initial_phi_entropy_log2\":%.10g, ",
                    r->initial_phi_entropy);
            fprintf(f, "\"final_phi_entropy_log2\":%.10g, ",
                    r->final_phi_entropy);
            fprintf(f, "\"upper_bound_log2\":%d, ", r->upper_bound_log2);
            fprintf(f, "\"outer_iterations\":%d, ", r->outer_iterations);
            fprintf(f, "\"walltime_s\":%.6f}%s\n",
                    r->walltime_s, (i + 1 < n_rec) ? "," : "");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
        printf("\nwrote JSON archive: %s (%zu records)\n", argv[1], n_rec);
    }

    return 0;
}
