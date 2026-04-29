/**
 * @file ca_mps_var_d.c
 * @brief Variational-D Clifford-only search for CA-MPS.
 *
 * Implements the greedy local-Clifford search described in §5.3 of
 * docs/research/ca_mps.md.  Each pass enumerates candidate Cliffords
 * (single-qubit H/S/S^dag on every qubit, optionally CNOT/CZ/SWAP on
 * every nearest-neighbour pair), evaluates the resulting energy at
 * fixed |phi>, and accepts the largest decrease.  Passes repeat until
 * no candidate improves on the current energy by more than
 * @c improvement_eps.
 *
 * Energy evaluation reuses ::moonlab_ca_mps_expect_pauli_sum, so this
 * file is mostly orchestration -- it owns the gate-enumeration loop
 * and the apply/undo discipline that keeps state->D consistent if the
 * candidate is rejected.
 */

#include "ca_mps_var_d.h"
#include "ca_mps.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

ca_mps_var_d_config_t ca_mps_var_d_config_default(void) {
    ca_mps_var_d_config_t c;
    c.max_passes = 50;
    c.improvement_eps = 1e-8;
    c.include_2q_gates = 1;
    c.verbose = 0;
    return c;
}

/* Apply a candidate Clifford gate to @p s->D, indexed by an enum.  All
 * gates here are self-inverse with one exception (S, whose inverse is
 * S^dag); we list S^dag separately so the apply/undo path is uniform.
 *
 * Two-qubit gates use a single qubit pair (q1, q2).  Single-qubit gates
 * ignore q2.
 */
enum cand_gate {
    CAND_H, CAND_S, CAND_SDAG, CAND_X, CAND_Y, CAND_Z,
    CAND_CNOT, CAND_CZ, CAND_SWAP,
    CAND__N
};

static const char* cand_name(int g) {
    switch (g) {
        case CAND_H:    return "H";
        case CAND_S:    return "S";
        case CAND_SDAG: return "Sdag";
        case CAND_X:    return "X";
        case CAND_Y:    return "Y";
        case CAND_Z:    return "Z";
        case CAND_CNOT: return "CNOT";
        case CAND_CZ:   return "CZ";
        case CAND_SWAP: return "SWAP";
    }
    return "?";
}

static int cand_is_2q(int g) {
    return g == CAND_CNOT || g == CAND_CZ || g == CAND_SWAP;
}

/* Apply a candidate gate to the tableau.  Returns the success status
 * from the underlying CA-MPS Clifford call. */
static ca_mps_error_t apply_cand(moonlab_ca_mps_t* s, int gate,
                                  uint32_t q1, uint32_t q2) {
    switch (gate) {
        case CAND_H:    return moonlab_ca_mps_h(s, q1);
        case CAND_S:    return moonlab_ca_mps_s(s, q1);
        case CAND_SDAG: return moonlab_ca_mps_sdag(s, q1);
        case CAND_X:    return moonlab_ca_mps_x(s, q1);
        case CAND_Y:    return moonlab_ca_mps_y(s, q1);
        case CAND_Z:    return moonlab_ca_mps_z(s, q1);
        case CAND_CNOT: return moonlab_ca_mps_cnot(s, q1, q2);
        case CAND_CZ:   return moonlab_ca_mps_cz(s, q1, q2);
        case CAND_SWAP: return moonlab_ca_mps_swap(s, q1, q2);
    }
    return CA_MPS_ERR_INVALID;
}

/* Inverse of a candidate gate.  H, X, Y, Z, CNOT, CZ, SWAP are self-inverse;
 * S/S^dag are each other's inverse. */
static ca_mps_error_t undo_cand(moonlab_ca_mps_t* s, int gate,
                                 uint32_t q1, uint32_t q2) {
    switch (gate) {
        case CAND_S:    return moonlab_ca_mps_sdag(s, q1);
        case CAND_SDAG: return moonlab_ca_mps_s(s, q1);
        default:        return apply_cand(s, gate, q1, q2);
    }
}

/* Compute <psi|H|psi> = sum_k coeffs[k] * <psi|paulis[k]|psi> using the
 * existing CA-MPS Pauli-sum expectation primitive. */
static double evaluate_energy(const moonlab_ca_mps_t* s,
                               const uint8_t* paulis,
                               const double* coeffs,
                               uint32_t num_terms) {
    /* Convert the real coefficient array to complex once (small alloc). */
    double _Complex* cz = (double _Complex*)calloc(num_terms, sizeof(double _Complex));
    if (!cz) return 0.0;
    for (uint32_t i = 0; i < num_terms; i++) cz[i] = (double _Complex)coeffs[i];

    double _Complex out = 0.0;
    moonlab_ca_mps_expect_pauli_sum(s, paulis, cz, num_terms, &out);
    free(cz);
    return creal(out);
}

ca_mps_error_t moonlab_ca_mps_optimize_var_d_clifford_only(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    const double* coeffs,
    uint32_t num_terms,
    const ca_mps_var_d_config_t* config,
    ca_mps_var_d_result_t* result) {
    if (!state || !paulis || !coeffs || num_terms == 0) return CA_MPS_ERR_INVALID;

    ca_mps_var_d_config_t cfg =
        config ? *config : ca_mps_var_d_config_default();

    uint32_t n = moonlab_ca_mps_num_qubits(state);

    double E_curr = evaluate_energy(state, paulis, coeffs, num_terms);
    double S0 = moonlab_ca_mps_max_half_cut_entropy(state);
    if (result) {
        result->initial_energy = E_curr;
        result->initial_phi_entropy = S0;
        result->final_energy = E_curr;
        result->final_phi_entropy = S0;
        result->gates_added = 0;
        result->passes = 0;
        result->converged = 0;
    }

    if (cfg.verbose) {
        fprintf(stdout, "[var-D] initial E = %.10f, S(phi) = %.6f\n", E_curr, S0);
    }

    int gates_added = 0;
    int passes = 0;
    int converged = 0;

    for (passes = 0; passes < cfg.max_passes; passes++) {
        /* Best candidate found so far in this pass. */
        double best_dE = -cfg.improvement_eps;   /* must beat eps to accept */
        int best_gate = -1;
        uint32_t best_q1 = 0, best_q2 = 0;

        const int gate_list_1q[] = { CAND_H, CAND_S, CAND_SDAG };
        const int n1q = sizeof(gate_list_1q) / sizeof(gate_list_1q[0]);
        const int gate_list_2q[] = { CAND_CNOT, CAND_CZ, CAND_SWAP };
        const int n2q = sizeof(gate_list_2q) / sizeof(gate_list_2q[0]);

        /* Single-qubit candidates. */
        for (uint32_t q = 0; q < n; q++) {
            for (int gi = 0; gi < n1q; gi++) {
                int g = gate_list_1q[gi];
                if (apply_cand(state, g, q, 0) != CA_MPS_SUCCESS) continue;
                double E_test = evaluate_energy(state, paulis, coeffs, num_terms);
                undo_cand(state, g, q, 0);
                double dE = E_test - E_curr;
                if (dE < best_dE) {
                    best_dE = dE;
                    best_gate = g;
                    best_q1 = q;
                    best_q2 = 0;
                }
            }
        }

        /* Two-qubit candidates on every nearest-neighbour pair. */
        if (cfg.include_2q_gates) {
            for (uint32_t q = 0; q + 1 < n; q++) {
                for (int gi = 0; gi < n2q; gi++) {
                    int g = gate_list_2q[gi];
                    if (apply_cand(state, g, q, q + 1) != CA_MPS_SUCCESS) continue;
                    double E_test = evaluate_energy(state, paulis, coeffs, num_terms);
                    undo_cand(state, g, q, q + 1);
                    double dE = E_test - E_curr;
                    if (dE < best_dE) {
                        best_dE = dE;
                        best_gate = g;
                        best_q1 = q;
                        best_q2 = q + 1;
                    }
                }
            }
        }

        if (best_gate < 0) {
            /* No candidate beat the threshold -- local minimum reached. */
            converged = 1;
            passes++;
            break;
        }

        /* Accept the best candidate permanently. */
        apply_cand(state, best_gate, best_q1, best_q2);
        E_curr += best_dE;
        gates_added++;
        if (cfg.verbose) {
            if (cand_is_2q(best_gate)) {
                fprintf(stdout, "[var-D] accept %s(%u,%u)  dE=%+.6e  E=%.10f\n",
                        cand_name(best_gate), best_q1, best_q2, best_dE, E_curr);
            } else {
                fprintf(stdout, "[var-D] accept %s(%u)    dE=%+.6e  E=%.10f\n",
                        cand_name(best_gate), best_q1, best_dE, E_curr);
            }
        }
    }

    double S_final = moonlab_ca_mps_max_half_cut_entropy(state);
    if (result) {
        result->final_energy = E_curr;
        result->final_phi_entropy = S_final;
        result->gates_added = gates_added;
        result->passes = passes;
        result->converged = converged;
    }
    if (cfg.verbose) {
        fprintf(stdout, "[var-D] final   E = %.10f, S(phi) = %.6f, "
                        "gates=%d, passes=%d, converged=%d\n",
                E_curr, S_final, gates_added, passes, converged);
    }

    return CA_MPS_SUCCESS;
}

/* ================================================================== */
/*  Alternating optimization                                           */
/* ================================================================== */

ca_mps_var_d_alt_config_t ca_mps_var_d_alt_config_default(void) {
    ca_mps_var_d_alt_config_t c;
    c.max_outer_iters             = 30;
    c.imag_time_dtau              = 0.1;
    c.imag_time_steps_per_outer   = 5;
    c.clifford_passes_per_outer   = 10;
    c.convergence_eps             = 1e-7;
    c.include_2q_gates            = 1;
    c.verbose                     = 0;
    return c;
}

/* One Trotter cycle of e^(-dtau * sum_k c_k P_k) on |psi>.  First-order
 * Trotter; for our purposes (variational descent) higher-order Trotter
 * isn't needed -- the goal is energy descent, and any consistent
 * approximation of the imag-time evolution achieves that. */
static ca_mps_error_t imag_time_sweep(moonlab_ca_mps_t* s,
                                       const uint8_t* paulis,
                                       const double* coeffs,
                                       uint32_t num_terms,
                                       double dtau) {
    uint32_t n = moonlab_ca_mps_num_qubits(s);
    for (uint32_t k = 0; k < num_terms; k++) {
        const uint8_t* P_k = &paulis[(size_t)k * n];
        double tau_k = dtau * coeffs[k];
        if (tau_k == 0.0) continue;
        ca_mps_error_t e = moonlab_ca_mps_imag_pauli_rotation(s, P_k, tau_k);
        if (e != CA_MPS_SUCCESS) return e;
    }
    /* Renormalise: imag-time evolution is non-unitary; norm decays as
     * e^(-2 * dtau * E).  Without renormalisation the MPS state vector
     * shrinks toward zero. */
    return moonlab_ca_mps_normalize(s);
}

ca_mps_error_t moonlab_ca_mps_optimize_var_d_alternating(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    const double* coeffs,
    uint32_t num_terms,
    const ca_mps_var_d_alt_config_t* config,
    ca_mps_var_d_alt_result_t* result) {
    if (!state || !paulis || !coeffs || num_terms == 0) return CA_MPS_ERR_INVALID;

    ca_mps_var_d_alt_config_t cfg =
        config ? *config : ca_mps_var_d_alt_config_default();

    double E_curr = evaluate_energy(state, paulis, coeffs, num_terms);
    double S0 = moonlab_ca_mps_max_half_cut_entropy(state);
    if (result) {
        result->initial_energy = E_curr;
        result->initial_phi_entropy = S0;
        result->final_energy = E_curr;
        result->final_phi_entropy = S0;
        result->total_gates_added = 0;
        result->outer_iterations = 0;
        result->converged = 0;
    }

    if (cfg.verbose) {
        fprintf(stdout, "[var-D-alt] start  E = %.10f, S(phi) = %.6f\n",
                E_curr, S0);
    }

    int total_gates = 0;
    int converged = 0;
    int iter = 0;

    for (iter = 0; iter < cfg.max_outer_iters; iter++) {
        /* 1. |phi>-update: imag-time evolution toward GS of H. */
        for (int t = 0; t < cfg.imag_time_steps_per_outer; t++) {
            ca_mps_error_t e = imag_time_sweep(state, paulis, coeffs,
                                                num_terms, cfg.imag_time_dtau);
            if (e != CA_MPS_SUCCESS) return e;
        }
        double E_after_imag = evaluate_energy(state, paulis, coeffs, num_terms);

        /* 2. D-update: greedy Clifford search at current |phi>. */
        ca_mps_var_d_config_t inner = ca_mps_var_d_config_default();
        inner.max_passes       = cfg.clifford_passes_per_outer;
        inner.include_2q_gates = cfg.include_2q_gates;
        inner.improvement_eps  = cfg.convergence_eps * 1e-2;
        inner.verbose          = 0;
        ca_mps_var_d_result_t inner_res = {0};
        ca_mps_error_t e = moonlab_ca_mps_optimize_var_d_clifford_only(
            state, paulis, coeffs, num_terms, &inner, &inner_res);
        if (e != CA_MPS_SUCCESS) return e;
        total_gates += inner_res.gates_added;

        double E_new = inner_res.final_energy;
        double S_phi = moonlab_ca_mps_max_half_cut_entropy(state);
        double dE = E_new - E_curr;

        if (cfg.verbose) {
            fprintf(stdout,
                    "[var-D-alt] iter %2d  E = %.10f  (after-imag %.10f)  "
                    "S(phi)=%.4f  +%d gates  dE=%+.3e\n",
                    iter, E_new, E_after_imag, S_phi,
                    inner_res.gates_added, dE);
        }

        E_curr = E_new;
        if (-dE < cfg.convergence_eps) {
            converged = 1;
            iter++;
            break;
        }
    }

    double S_final = moonlab_ca_mps_max_half_cut_entropy(state);
    if (result) {
        result->final_energy = E_curr;
        result->final_phi_entropy = S_final;
        result->total_gates_added = total_gates;
        result->outer_iterations = iter;
        result->converged = converged;
    }
    if (cfg.verbose) {
        fprintf(stdout,
                "[var-D-alt] final  E = %.10f, S(phi) = %.6f, "
                "iters=%d, total_gates=%d, converged=%d\n",
                E_curr, S_final, iter, total_gates, converged);
    }
    return CA_MPS_SUCCESS;
}
