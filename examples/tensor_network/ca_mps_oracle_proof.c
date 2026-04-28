/**
 * @file ca_mps_oracle_proof.c
 * @brief Oracle-proof for variational-D mode: does a hand-supplied Clifford
 *        reduce |phi> bond-dim for the TFIM ground state?
 *
 * Per docs/research/ca_mps.md §5.3 minimum-viable validating experiment.
 *
 * The fixed-D CA-MPS in v0.3.0 fails on VQE/QAOA because the workload-supplied
 * Clifford prefactor delocalizes non-Clifford rotations.  Variational-D mode
 * proposes to *search* for a better Clifford D such that |phi> = D^dag |psi>
 * has lower MPS bond dimension than |psi> itself.
 *
 * Before sinking 2 weeks into the variational search, this experiment asks
 * the oracle question: does there *exist* a non-trivial Clifford that reduces
 * chi(|phi>) for a known-hard target state?  We pick TFIM near criticality
 * (where chi(|psi>) grows polynomially with N) and test two candidate
 * oracles:
 *
 *  1. D = CNOT-chain.  Maps the cat-state component of the GS at small g
 *     to a product state.  Should win at g << 1, lose at g >> 1.
 *  2. D = (CNOT-chain) (H on all qubits).  Implements a basic Kramers-Wannier
 *     dual transformation; should mirror the small-g win at large g.
 *
 * What we measure for each (n, g, D-choice):
 *  - chi(|psi>)   peak bond of the plain TFIM ground state
 *  - chi(|phi>)   peak bond after applying D^dag to the same state
 *  - <psi|H|psi>  energy preserved as a sanity check
 *
 * Outcome interpretation:
 *  - If chi(|phi>) << chi(|psi>) for SOME oracle Clifford at SOME g, the
 *    variational-D thesis has a target the search could plausibly reach.
 *  - If no oracle does better than D=I across the parameter sweep, even a
 *    perfect search wouldn't help and we should reconsider the entire
 *    fixed-D-vs-variational-D paradigm before committing implementation
 *    effort.
 */

#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t mps_max_bond(const tn_mps_state_t* s, uint32_t n) {
    uint32_t b = 1;
    for (uint32_t i = 0; i + 1 < n; i++) {
        uint32_t bb = tn_mps_bond_dim(s, i);
        if (bb > b) b = bb;
    }
    return b;
}

/* Maximum half-cut von Neumann entropy across all bipartitions.  This is
 * the representation-independent measure of "how entangled |psi> is" --
 * the bond-dim metric is unreliable when DMRG keeps allocated bonds at
 * chi_max even when actual rank is small. */
static double mps_max_half_cut_entropy(const tn_mps_state_t* s, uint32_t n) {
    double s_max = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        double e = tn_mps_entanglement_entropy(s, i);
        if (e > s_max) s_max = e;
    }
    return s_max;
}

/* Apply D^dag = D^{-1} to an MPS state; D is a Clifford specified by the
 * choice integer.  All Clifford gates we use here are self-inverse (H, CNOT)
 * so D^dag is the same gate sequence in REVERSE order. */
enum oracle_choice {
    ORACLE_IDENTITY,        /* D = I (control case). chi unchanged. */
    ORACLE_CNOT_CHAIN,      /* D = CNOT(0,1) CNOT(1,2) ... CNOT(n-2, n-1) */
    ORACLE_H_THEN_CNOT,     /* D = H_all then CNOT_chain (basic KW-style) */
};

static const char* oracle_name(int o) {
    switch (o) {
        case ORACLE_IDENTITY:       return "identity";
        case ORACLE_CNOT_CHAIN:     return "cnot_chain";
        case ORACLE_H_THEN_CNOT:    return "h_then_cnot";
    }
    return "?";
}

/* Apply D to |psi>.  The result represents D|psi>, which equals |phi> if we
 * interpret CA-MPS as |psi> = D^dag |phi> -- i.e., D|psi> = |phi>.  All gates
 * here are self-inverse so this is also D^dag. */
static void apply_oracle(tn_mps_state_t* s, int oracle, uint32_t n) {
    switch (oracle) {
        case ORACLE_IDENTITY:
            return;
        case ORACLE_CNOT_CHAIN:
            for (uint32_t i = 0; i + 1 < n; i++) {
                tn_apply_cnot(s, i, i + 1);
            }
            return;
        case ORACLE_H_THEN_CNOT:
            for (uint32_t q = 0; q < n; q++) {
                tn_apply_h(s, q);
            }
            for (uint32_t i = 0; i + 1 < n; i++) {
                tn_apply_cnot(s, i, i + 1);
            }
            return;
    }
}

int main(int argc, char** argv) {
    const uint32_t n_values[] = { 8, 12 };
    const double   g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.5 };
    const int oracles[] = { ORACLE_IDENTITY, ORACLE_CNOT_CHAIN, ORACLE_H_THEN_CNOT };
    const char* out_path = (argc >= 2) ? argv[1] : "ca_mps_oracle_proof.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout, "=== CA-MPS oracle-proof: chi(|phi>) under fixed Cliffords ===\n\n");
    fprintf(stdout, "%4s %6s %-12s %10s %10s %10s %10s %14s\n",
            "n", "g", "oracle", "chi(psi)", "chi(phi)", "S(psi)", "S(phi)", "E_GS");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_oracle_proof_v2\",\n  \"points\": [\n");
    int first_json = 1;

    for (size_t ni = 0; ni < sizeof(n_values)/sizeof(n_values[0]); ni++) {
        uint32_t n = n_values[ni];
        for (size_t gi = 0; gi < sizeof(g_values)/sizeof(g_values[0]); gi++) {
            double g = g_values[gi];
            /* Get the TFIM ground state via plain DMRG -- this is |psi>. */
            dmrg_config_t cfg = dmrg_config_default();
            cfg.max_bond_dim = 32;
            cfg.max_sweeps   = 12;
            cfg.energy_tol   = 1e-8;
            cfg.verbose      = false;
            fprintf(stdout, "[run] n=%u g=%.3f ...\n", n, g);
            fflush(stdout);
            dmrg_result_t* res = NULL;
            tn_mps_state_t* psi = dmrg_tfim_ground_state(n, g, &cfg, &res);
            if (!psi) { fprintf(stderr, "DMRG failed at n=%u, g=%.2f\n", n, g); continue; }
            double E = res ? res->ground_energy : 0.0;
            uint32_t chi_psi = mps_max_bond(psi, n);
            double S_psi = mps_max_half_cut_entropy(psi, n);

            for (size_t oi = 0; oi < sizeof(oracles)/sizeof(oracles[0]); oi++) {
                /* Work on a deep copy of psi so we don't mutate it across
                 * oracles. */
                tn_mps_state_t* phi = tn_mps_copy(psi);
                if (!phi) {
                    fprintf(stderr, "tn_mps_copy failed\n");
                    continue;
                }
                apply_oracle(phi, oracles[oi], n);
                uint32_t chi_phi = mps_max_bond(phi, n);
                double S_phi = mps_max_half_cut_entropy(phi, n);

                fprintf(stdout, "%4u %6.3f %-12s %10u %10u %10.4f %10.4f %14.6f\n",
                        n, g, oracle_name(oracles[oi]), chi_psi, chi_phi,
                        S_psi, S_phi, E);
                fflush(stdout);

                if (!first_json) fprintf(json, ",\n");
                first_json = 0;
                fprintf(json, "    { \"n\": %u, \"g\": %.4f, \"oracle\": \"%s\", "
                              "\"chi_psi\": %u, \"chi_phi\": %u, "
                              "\"S_psi\": %.6f, \"S_phi\": %.6f, "
                              "\"E_GS\": %.6f }",
                        n, g, oracle_name(oracles[oi]), chi_psi, chi_phi,
                        S_psi, S_phi, E);

                tn_mps_free(phi);
            }
            tn_mps_free(psi);
            if (res) dmrg_result_free(res);
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
