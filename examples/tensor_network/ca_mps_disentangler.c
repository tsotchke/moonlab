/**
 * @file ca_mps_disentangler.c
 * @brief Greedy Clifford disentangler: post-process a plain-MPS state to
 *        minimise half-cut entanglement entropy via Clifford gates.
 *
 * Different angle on the variational-D claim.  The existing alternating
 * optimiser drives an initial |phi>=|0...0> toward the ground state under
 * imaginary-time + Clifford search; energy is the objective and entropy
 * comes along as a side-effect.  But for a fixed target state |psi> (e.g.
 * the converged plain-DMRG ground state), we can target ENTROPY directly:
 * apply Clifford G to |psi>, measure S(G|psi>), accept if smaller, repeat.
 * The accepted gate sequence implicitly defines a Clifford D such that
 * |psi> = D|phi> with low S(|phi>).  No imag-time, no warmstart selection,
 * no per-model heuristics -- just direct entropy minimisation.
 *
 * For TFIM this should hit the oracle-proof level (5-50x reduction at all
 * g, including criticality) without the basin-hopping issues that the
 * alternating optimiser had at large N.
 *
 * Setup: TFIM phase sweep, n in {6, 8, 10}, g in {0.25, 0.5, 1.0, 1.5, 2.5}.
 * For each (n, g): plain DMRG -> direct Clifford disentangler -> report
 * S_psi (input entropy) vs S_phi (post-disentangle entropy) plus the
 * accepted gate count.
 */

#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double mps_max_half_cut_entropy(const tn_mps_state_t* s, uint32_t n) {
    double s_max = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        double e = tn_mps_entanglement_entropy(s, i);
        if (e > s_max) s_max = e;
    }
    return s_max;
}

/* Candidate Clifford gates.  All are self-inverse except S/S^dag. */
enum cliff_gate {
    G_H, G_S, G_SDAG, G_X, G_Y, G_Z,
    G_CNOT, G_CZ, G_SWAP,
    G__N
};

static const char* g_name(int g) {
    switch (g) {
        case G_H: return "H"; case G_S: return "S"; case G_SDAG: return "Sdag";
        case G_X: return "X"; case G_Y: return "Y"; case G_Z: return "Z";
        case G_CNOT: return "CNOT"; case G_CZ: return "CZ"; case G_SWAP: return "SWAP";
    }
    return "?";
}

static int is_2q(int g) {
    return g == G_CNOT || g == G_CZ || g == G_SWAP;
}

static tn_gate_error_t apply_gate(tn_mps_state_t* psi, int g, uint32_t q1, uint32_t q2) {
    switch (g) {
        case G_H:    return tn_apply_h(psi, q1);
        case G_S:    return tn_apply_s(psi, q1);
        case G_SDAG: {
            /* tn_apply_s applies S; S^dag = S^3 = S after S after S */
            tn_gate_error_t e = tn_apply_s(psi, q1);
            if (e == TN_GATE_SUCCESS) e = tn_apply_s(psi, q1);
            if (e == TN_GATE_SUCCESS) e = tn_apply_s(psi, q1);
            return e;
        }
        case G_X:    return tn_apply_x(psi, q1);
        case G_Y:    return tn_apply_y(psi, q1);
        case G_Z:    return tn_apply_z(psi, q1);
        case G_CNOT: return tn_apply_cnot(psi, q1, q2);
        case G_CZ:   return tn_apply_cz(psi, q1, q2);
        case G_SWAP: return tn_apply_swap(psi, q1, q2);
    }
    return TN_GATE_ERROR_INVALID_GATE;
}

static tn_gate_error_t undo_gate(tn_mps_state_t* psi, int g, uint32_t q1, uint32_t q2) {
    /* H, X, Y, Z, CNOT, CZ, SWAP are self-inverse.  S and S^dag are mutual
     * inverses.  We just call apply_gate of the inverse. */
    switch (g) {
        case G_S:    return apply_gate(psi, G_SDAG, q1, q2);
        case G_SDAG: return apply_gate(psi, G_S, q1, q2);
        default:     return apply_gate(psi, g, q1, q2);
    }
}

/* Greedy single-pass Clifford disentangler.  Tests each candidate on
 * a clone of @p psi (so apply/undo SVD non-idempotency doesn't drift
 * the state), then mutates @p psi only with the best accepted gate. */
static int disentangle_pass(tn_mps_state_t* psi, uint32_t n,
                              double improvement_eps, int verbose) {
    double S_curr = mps_max_half_cut_entropy(psi, n);

    /* Find the best gate over all candidates. */
    double best_dS = -improvement_eps;
    int best_g = -1;
    uint32_t best_q1 = 0, best_q2 = 0;

    const int gates_1q[] = { G_H, G_S, G_SDAG };
    const int n1q = sizeof(gates_1q)/sizeof(gates_1q[0]);
    const int gates_2q[] = { G_CNOT, G_CZ, G_SWAP };
    const int n2q = sizeof(gates_2q)/sizeof(gates_2q[0]);

    for (uint32_t q = 0; q < n; q++) {
        for (int gi = 0; gi < n1q; gi++) {
            int g = gates_1q[gi];
            tn_mps_state_t* clone = tn_mps_copy(psi);
            if (!clone) continue;
            if (apply_gate(clone, g, q, 0) != TN_GATE_SUCCESS) {
                tn_mps_free(clone); continue;
            }
            double S_test = mps_max_half_cut_entropy(clone, n);
            tn_mps_free(clone);
            double dS = S_test - S_curr;
            if (dS < best_dS) {
                best_dS = dS; best_g = g; best_q1 = q; best_q2 = 0;
            }
        }
    }
    for (uint32_t q = 0; q + 1 < n; q++) {
        for (int gi = 0; gi < n2q; gi++) {
            int g = gates_2q[gi];
            tn_mps_state_t* clone = tn_mps_copy(psi);
            if (!clone) continue;
            if (apply_gate(clone, g, q, q + 1) != TN_GATE_SUCCESS) {
                tn_mps_free(clone); continue;
            }
            double S_test = mps_max_half_cut_entropy(clone, n);
            tn_mps_free(clone);
            double dS = S_test - S_curr;
            if (dS < best_dS) {
                best_dS = dS; best_g = g; best_q1 = q; best_q2 = q + 1;
            }
        }
    }

    if (best_g < 0) return 0;  /* no improvement */

    apply_gate(psi, best_g, best_q1, best_q2);
    if (verbose) {
        if (is_2q(best_g)) {
            fprintf(stdout, "  [disentangle] accept %s(%u,%u)  S: %.4f -> %.4f\n",
                    g_name(best_g), best_q1, best_q2, S_curr, S_curr + best_dS);
        } else {
            fprintf(stdout, "  [disentangle] accept %s(%u)    S: %.4f -> %.4f\n",
                    g_name(best_g), best_q1, S_curr, S_curr + best_dS);
        }
        fflush(stdout);
    }
    return 1;  /* accepted */
}

/* Apply the TFIM "dual" structured Clifford warmstart to @p psi:
 * H on every qubit, then CNOT(0,1) CNOT(1,2) ... CNOT(n-2, n-1).
 * The 2026-04-28 oracle proof showed this Clifford disentangles the
 * TFIM ground state by 5-50x across the entire phase diagram.  Use
 * it as the initial structured move; greedy refinement then
 * polishes from there. */
static int apply_dual_warmstart(tn_mps_state_t* psi, uint32_t n) {
    int gates = 0;
    for (uint32_t q = 0; q < n; q++) {
        if (tn_apply_h(psi, q) != TN_GATE_SUCCESS) return gates;
        gates++;
    }
    for (uint32_t q = 0; q + 1 < n; q++) {
        if (tn_apply_cnot(psi, q, q + 1) != TN_GATE_SUCCESS) return gates;
        gates++;
    }
    return gates;
}

static double disentangle(tn_mps_state_t* psi, uint32_t n,
                           int max_passes, double improvement_eps,
                           int verbose, int* out_gate_count,
                           int use_warmstart) {
    int gate_count = 0;
    if (use_warmstart) {
        int wg = apply_dual_warmstart(psi, n);
        if (verbose) {
            fprintf(stdout, "  [disentangle] warmstart applied %d gates, S=%.4f\n",
                    wg, mps_max_half_cut_entropy(psi, n));
            fflush(stdout);
        }
        gate_count += wg;
    }
    for (int pass = 0; pass < max_passes; pass++) {
        int accepted = disentangle_pass(psi, n, improvement_eps, verbose);
        if (!accepted) break;
        gate_count++;
    }
    if (out_gate_count) *out_gate_count = gate_count;
    return mps_max_half_cut_entropy(psi, n);
}

static void build_tfim_dmrg_gs(uint32_t n, double g, uint32_t chi,
                                tn_mps_state_t** out_psi, double* out_E,
                                double* out_S) {
    dmrg_config_t cfg = dmrg_config_default();
    cfg.max_bond_dim = chi;
    cfg.max_sweeps   = 20;
    cfg.energy_tol   = 1e-9;
    cfg.verbose      = false;
    dmrg_result_t* res = NULL;
    *out_psi = dmrg_tfim_ground_state(n, g, &cfg, &res);
    *out_E = res ? res->ground_energy : 0.0;
    *out_S = (*out_psi) ? mps_max_half_cut_entropy(*out_psi, n) : 0.0;
    if (res) dmrg_result_free(res);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "ca_mps_disentangler.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout, "=== Greedy Clifford disentangler on TFIM ground state ===\n\n");
    fprintf(stdout, "Plain DMRG converges to |psi_GS>; we then greedy-apply Clifford\n");
    fprintf(stdout, "gates directly to that MPS to minimise half-cut entropy.  The\n");
    fprintf(stdout, "accepted gate sequence is the Clifford D such that |psi_GS> =\n");
    fprintf(stdout, "D |phi> with smaller S(|phi>) than S(|psi_GS>).\n\n");

    fprintf(stdout, "%4s %6s %12s %10s %10s %10s %8s %10s\n",
            "n", "g", "E_dmrg", "S_psi", "S_phi", "ratio", "gates", "wall_s");
    fprintf(stdout, "----------------------------------------------------------------------\n");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_disentangler_v1\",\n  \"points\": [\n");
    int first = 1;

    const uint32_t n_values[] = { 6, 8, 10 };
    const double   g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.5 };

    for (size_t ni = 0; ni < sizeof(n_values)/sizeof(n_values[0]); ni++) {
        uint32_t n = n_values[ni];
        for (size_t gi = 0; gi < sizeof(g_values)/sizeof(g_values[0]); gi++) {
            double g = g_values[gi];

            /* Use a larger chi cap for the MPS than we need for the GS
             * itself; intermediate states during Clifford application
             * can have higher bond dim than the final |phi>, and a
             * tight cap during the Clifford pass corrupts the result.
             * For n=10, scaling chi to 96 keeps the disentangler stable
             * (the 2^(n/2) entropy ceiling is at chi=32 only for the
             * physical state -- intermediate Clifford-rotated states
             * can saturate higher). */
            uint32_t chi_dmrg = (n <= 8) ? 32 : 96;
            tn_mps_state_t* psi = NULL;
            double E, S_psi;
            build_tfim_dmrg_gs(n, g, chi_dmrg, &psi, &E, &S_psi);
            if (!psi) continue;

            /* Try both: greedy-only (no warmstart) and warmstart+greedy.
             * Keep whichever gives smaller |phi> entropy. */
            tn_mps_state_t* psi_greedy = tn_mps_copy(psi);
            int gates_greedy = 0;
            double t0 = now_s();
            double S_greedy = disentangle(psi_greedy, n, /*max_passes=*/100,
                                           /*eps=*/1e-6, /*verbose=*/0,
                                           &gates_greedy, /*warmstart=*/0);

            tn_mps_state_t* psi_warm = tn_mps_copy(psi);
            int gates_warm = 0;
            double S_warm = disentangle(psi_warm, n, /*max_passes=*/100,
                                         /*eps=*/1e-6, /*verbose=*/0,
                                         &gates_warm, /*warmstart=*/1);
            double dt = now_s() - t0;

            int gate_count;
            double S_phi;
            const char* mode;
            if (S_warm < S_greedy) {
                S_phi = S_warm; gate_count = gates_warm; mode = "warm";
            } else {
                S_phi = S_greedy; gate_count = gates_greedy; mode = "greedy";
            }
            (void)mode;
            tn_mps_free(psi_greedy);
            tn_mps_free(psi_warm);

            double ratio = (S_psi > 1e-12) ? (S_phi / S_psi) : 0.0;
            fprintf(stdout, "%4u %6.3f %12.6f %10.4f %10.4f %10.4f %8d %10.2f\n",
                    n, g, E, S_psi, S_phi, ratio, gate_count, dt);
            fflush(stdout);

            if (!first) fprintf(json, ",\n");
            first = 0;
            fprintf(json, "    { \"n\": %u, \"g\": %.4f, \"E_dmrg\": %.6f, "
                          "\"S_psi\": %.6f, \"S_phi\": %.6f, "
                          "\"ratio\": %.6f, \"gates\": %d, \"wall_s\": %.4f }",
                    n, g, E, S_psi, S_phi, ratio, gate_count, dt);

            tn_mps_free(psi);
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
