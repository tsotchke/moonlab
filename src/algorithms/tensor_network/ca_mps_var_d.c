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
#include "ca_mps_var_d_stab_warmstart.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

ca_mps_var_d_config_t ca_mps_var_d_config_default(void) {
    ca_mps_var_d_config_t c;
    c.max_passes = 50;
    c.improvement_eps = 1e-8;
    c.include_2q_gates = 1;
    c.composite_2gate = 0;
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

/* One candidate move: a primary gate plus optionally a composite second
 * gate.  gate2 < 0 means single-gate candidate. */
typedef struct {
    int gate;       uint32_t q1;   uint32_t q2;
    int gate2;      uint32_t q1_2; uint32_t q2_2;
} cand_t;

/* Apply a candidate's primary (and optional composite) gate to @p s.
 * Returns 0 on success, -1 if any step failed; on -1 the caller must
 * still call cand_undo to roll back any partial progress. */
static int cand_apply(moonlab_ca_mps_t* s, const cand_t* c) {
    if (apply_cand(s, c->gate, c->q1, c->q2) != CA_MPS_SUCCESS) return -1;
    if (c->gate2 >= 0) {
        if (apply_cand(s, c->gate2, c->q1_2, c->q2_2) != CA_MPS_SUCCESS) {
            undo_cand(s, c->gate, c->q1, c->q2);
            return -1;
        }
    }
    return 0;
}

static void cand_undo(moonlab_ca_mps_t* s, const cand_t* c) {
    if (c->gate2 >= 0) undo_cand(s, c->gate2, c->q1_2, c->q2_2);
    undo_cand(s, c->gate, c->q1, c->q2);
}

/* Enumerate all single-gate candidates: 1q on every qubit, optionally
 * 2q on every nearest-neighbour pair.  Caller frees out[]. */
static cand_t* enumerate_single_cands(uint32_t n, int include_2q,
                                       size_t* out_count) {
    const int gate_list_1q[] = { CAND_H, CAND_S, CAND_SDAG };
    const int n1q = sizeof(gate_list_1q) / sizeof(gate_list_1q[0]);
    const int gate_list_2q[] = { CAND_CNOT, CAND_CZ, CAND_SWAP };
    const int n2q = sizeof(gate_list_2q) / sizeof(gate_list_2q[0]);

    size_t cap = (size_t)n * n1q + (include_2q ? (size_t)(n - 1) * n2q : 0);
    cand_t* cands = (cand_t*)calloc(cap, sizeof(cand_t));
    if (!cands) { *out_count = 0; return NULL; }

    size_t k = 0;
    for (uint32_t q = 0; q < n; q++) {
        for (int gi = 0; gi < n1q; gi++) {
            cands[k] = (cand_t){ gate_list_1q[gi], q, 0, -1, 0, 0 };
            k++;
        }
    }
    if (include_2q) {
        for (uint32_t q = 0; q + 1 < n; q++) {
            for (int gi = 0; gi < n2q; gi++) {
                cands[k] = (cand_t){ gate_list_2q[gi], q, q + 1, -1, 0, 0 };
                k++;
            }
        }
    }
    *out_count = k;
    return cands;
}

/* Enumerate composite single+single candidates: gA on qa, gB on qb,
 * with qb != qa.  Helps escape 1-gate local minima. */
static cand_t* enumerate_composite_cands(uint32_t n, size_t* out_count) {
    const int gate_list_1q[] = { CAND_H, CAND_S, CAND_SDAG };
    const int n1q = sizeof(gate_list_1q) / sizeof(gate_list_1q[0]);

    size_t cap = (size_t)n * n1q * (size_t)(n - 1) * n1q;
    cand_t* cands = (cand_t*)calloc(cap, sizeof(cand_t));
    if (!cands) { *out_count = 0; return NULL; }

    size_t k = 0;
    for (uint32_t qa = 0; qa < n; qa++) {
        for (int ga = 0; ga < n1q; ga++) {
            for (uint32_t qb = 0; qb < n; qb++) {
                if (qb == qa) continue;
                for (int gb = 0; gb < n1q; gb++) {
                    cands[k] = (cand_t){
                        gate_list_1q[ga], qa, 0,
                        gate_list_1q[gb], qb, 0
                    };
                    k++;
                }
            }
        }
    }
    *out_count = k;
    return cands;
}

/* Evaluate every candidate's resulting energy in parallel.  Returns
 * the energies in @p out_energies (must be length n_cands).  Each
 * thread keeps a private CA-MPS clone and a private complex-coeffs
 * buffer; the original @p state is not mutated.  INFINITY is written
 * for any candidate whose apply() failed (out-of-range, etc.). */
static void evaluate_candidates_parallel(const moonlab_ca_mps_t* state,
                                          const uint8_t* paulis,
                                          const double* coeffs,
                                          uint32_t num_terms,
                                          const cand_t* cands,
                                          size_t n_cands,
                                          double* out_energies) {
    /* Pre-convert coefficients to complex once (read-only across threads). */
    double _Complex* cz_master =
        (double _Complex*)calloc(num_terms, sizeof(double _Complex));
    if (!cz_master) {
        for (size_t i = 0; i < n_cands; i++) out_energies[i] = INFINITY;
        return;
    }
    for (uint32_t i = 0; i < num_terms; i++) cz_master[i] = (double _Complex)coeffs[i];

    #pragma omp parallel
    {
        moonlab_ca_mps_t* my = moonlab_ca_mps_clone(state);

        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < n_cands; i++) {
            if (!my) {
                out_energies[i] = INFINITY;
                continue;
            }
            const cand_t* c = &cands[i];
            if (cand_apply(my, c) != 0) {
                out_energies[i] = INFINITY;
                continue;
            }
            double _Complex out = 0.0;
            moonlab_ca_mps_expect_pauli_sum(my, paulis, cz_master, num_terms, &out);
            out_energies[i] = creal(out);
            cand_undo(my, c);
        }

        if (my) moonlab_ca_mps_free(my);
    }

    free(cz_master);
}

/* Sequential reduction: pick the candidate with the most-negative
 * energy delta vs E_curr that exceeds the improvement threshold.
 * Returns the index, or -1 if none beat the threshold. */
static long pick_best_candidate(const double* energies, size_t n_cands,
                                 double E_curr, double threshold,
                                 double* out_dE) {
    long best = -1;
    double best_dE = -threshold;
    for (size_t i = 0; i < n_cands; i++) {
        if (!isfinite(energies[i])) continue;
        const double dE = energies[i] - E_curr;
        if (dE < best_dE) { best_dE = dE; best = (long)i; }
    }
    *out_dE = best_dE;
    return best;
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
        /* Phase A: enumerate single-gate candidates (1q + optional 2q),
         * evaluate them all in parallel, pick the best.  Each thread
         * keeps a private CA-MPS clone so candidate apply/undo doesn't
         * race on the shared tableau. */
        size_t n_single = 0;
        cand_t* single_cands =
            enumerate_single_cands(n, cfg.include_2q_gates, &n_single);
        cand_t best;
        best.gate = -1;
        double best_dE = 0.0;

        if (single_cands && n_single > 0) {
            double* energies = (double*)calloc(n_single, sizeof(double));
            if (energies) {
                evaluate_candidates_parallel(state, paulis, coeffs, num_terms,
                                              single_cands, n_single, energies);
                long pick = pick_best_candidate(energies, n_single, E_curr,
                                                 cfg.improvement_eps, &best_dE);
                if (pick >= 0) best = single_cands[pick];
                free(energies);
            }
        }
        free(single_cands);

        /* Phase B: composite 2-gate move search runs only when phase A
         * found nothing.  Same parallel pattern. */
        if (best.gate < 0 && cfg.composite_2gate) {
            size_t n_composite = 0;
            cand_t* composite_cands = enumerate_composite_cands(n, &n_composite);
            if (composite_cands && n_composite > 0) {
                double* energies = (double*)calloc(n_composite, sizeof(double));
                if (energies) {
                    evaluate_candidates_parallel(state, paulis, coeffs, num_terms,
                                                  composite_cands, n_composite,
                                                  energies);
                    long pick = pick_best_candidate(energies, n_composite, E_curr,
                                                     cfg.improvement_eps, &best_dE);
                    if (pick >= 0) best = composite_cands[pick];
                    free(energies);
                }
            }
            free(composite_cands);
        }

        if (best.gate < 0) {
            /* No candidate beat the threshold -- local minimum reached. */
            converged = 1;
            passes++;
            break;
        }

        /* Accept the best candidate permanently on the original state. */
        cand_apply(state, &best);
        E_curr += best_dE;
        gates_added++;
        if (best.gate2 >= 0) gates_added++;

        if (cfg.verbose) {
            if (cand_is_2q(best.gate)) {
                fprintf(stdout, "[var-D] accept %s(%u,%u)  dE=%+.6e  E=%.10f\n",
                        cand_name(best.gate), best.q1, best.q2, best_dE, E_curr);
            } else {
                fprintf(stdout, "[var-D] accept %s(%u)    dE=%+.6e  E=%.10f\n",
                        cand_name(best.gate), best.q1, best_dE, E_curr);
            }
            if (best.gate2 >= 0) {
                fprintf(stdout, "[var-D] accept %s(%u)    (composite partner)\n",
                        cand_name(best.gate2), best.q1_2);
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
    c.composite_2gate             = 0;
    c.warmstart                   = CA_MPS_WARMSTART_IDENTITY;
    c.warmstart_stab_paulis       = NULL;
    c.warmstart_stab_num_gens     = 0;
    c.verbose                     = 0;
    return c;
}

/* One Trotter cycle of e^(-dtau * sum_k c_k P_k) on |psi> using the
 * symmetric (Strang) splitting:
 *
 *   exp(-dtau sum_k c_k P_k)
 *     ~ [prod_{k=0..T-2} exp(-dtau/2 c_k P_k)]
 *       . exp(-dtau c_{T-1} P_{T-1})
 *       . [prod_{k=T-2..0} exp(-dtau/2 c_k P_k)]
 *
 * Per-step Trotter error is O(dtau^3) vs the O(dtau^2) of first-order
 * Trotter.  Cost is ~2x per cycle, but at typical var-D dtau values the
 * accuracy gain dominates (the v0.2.1 paper-grade var-D-vs-ED parity
 * benchmark observed ~30-100x lower energy error at the same wall-clock
 * budget once Strang replaced first-order).
 *
 * Renormalises every RENORM_INTERVAL applications to keep the MPS norm
 * close to 1 throughout; without intermediate renormalisation the state
 * accumulates O(num_terms) compounded norm shifts in a single sweep --
 * when the conjugated Paulis are high-weight (which happens after a
 * non-trivial D warmstart) the resulting numerical drift can drive
 * evaluate_energy below the variational lower bound. */
static ca_mps_error_t imag_time_sweep(moonlab_ca_mps_t* s,
                                       const uint8_t* paulis,
                                       const double* coeffs,
                                       uint32_t num_terms,
                                       double dtau) {
    if (num_terms == 0) return CA_MPS_SUCCESS;
    const uint32_t n = moonlab_ca_mps_num_qubits(s);
    const uint32_t RENORM_INTERVAL = 4;

    /* Forward half-sweep, k = 0..T-2 with dtau/2. */
    for (uint32_t k = 0; k + 1 < num_terms; k++) {
        const uint8_t* P_k = &paulis[(size_t)k * n];
        const double tau_k = 0.5 * dtau * coeffs[k];
        if (tau_k == 0.0) continue;
        ca_mps_error_t e = moonlab_ca_mps_imag_pauli_rotation(s, P_k, tau_k);
        if (e != CA_MPS_SUCCESS) return e;
        if ((k + 1) % RENORM_INTERVAL == 0) {
            e = moonlab_ca_mps_normalize(s);
            if (e != CA_MPS_SUCCESS) return e;
        }
    }
    /* Centre term: full dtau on k = T-1. */
    {
        const uint8_t* P_last = &paulis[(size_t)(num_terms - 1) * n];
        const double tau_last = dtau * coeffs[num_terms - 1];
        if (tau_last != 0.0) {
            ca_mps_error_t e = moonlab_ca_mps_imag_pauli_rotation(s, P_last, tau_last);
            if (e != CA_MPS_SUCCESS) return e;
        }
    }
    /* Backward half-sweep, k = T-2..0 with dtau/2. */
    for (uint32_t k = num_terms - 1; k-- > 0; ) {
        const uint8_t* P_k = &paulis[(size_t)k * n];
        const double tau_k = 0.5 * dtau * coeffs[k];
        if (tau_k == 0.0) continue;
        ca_mps_error_t e = moonlab_ca_mps_imag_pauli_rotation(s, P_k, tau_k);
        if (e != CA_MPS_SUCCESS) return e;
        if (((num_terms - 1 - k) + 1) % RENORM_INTERVAL == 0) {
            e = moonlab_ca_mps_normalize(s);
            if (e != CA_MPS_SUCCESS) return e;
        }
    }
    /* Final renormalise.  Imag-time evolution is non-unitary; norm
     * decays as e^(-2 * dtau * E).  Without renormalisation the MPS
     * state vector shrinks toward zero. */
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

    /* Optional warm-start: apply a structured Clifford to D before
     * starting the alternating loop.  Useful at criticality where
     * the right Clifford involves H on every qubit (and possibly a
     * CNOT chain); the greedy local search starting from D=I has no
     * single-gate descent toward it because reaching the right
     * basin requires ~n consecutive accepts that look bad in step 1.
     * Warmstart options bias the search toward known-productive
     * basins for specific model classes. */
    if (cfg.warmstart == CA_MPS_WARMSTART_H_ALL ||
        cfg.warmstart == CA_MPS_WARMSTART_DUAL_TFIM) {
        uint32_t n = moonlab_ca_mps_num_qubits(state);
        for (uint32_t q = 0; q < n; q++) {
            ca_mps_error_t e = moonlab_ca_mps_h(state, q);
            if (e != CA_MPS_SUCCESS) return e;
        }
        if (cfg.warmstart == CA_MPS_WARMSTART_DUAL_TFIM) {
            /* CNOT chain: CNOT(0,1) CNOT(1,2) ... CNOT(n-2,n-1).
             * Combined with H_all this is the dual basis transform
             * for 1D TFIM that the 2026-04-28 oracle proof showed
             * reduces |phi> entropy by 5-50x across the phase
             * diagram, including the critical point. */
            for (uint32_t q = 0; q + 1 < n; q++) {
                ca_mps_error_t e = moonlab_ca_mps_cnot(state, q, q + 1);
                if (e != CA_MPS_SUCCESS) return e;
            }
        }
    } else if (cfg.warmstart == CA_MPS_WARMSTART_FERRO_TFIM) {
        /* Cat-state encoder: H on qubit 0, then CNOT chain.  Applied
         * to |phi> = |0...0>, this Clifford produces D|0...0> =
         * (|0..0> + |1..1>) / sqrt(2) -- the symmetric cat state
         * that is the GS of TFIM in the deep ferromagnetic regime
         * (g << 1).  Without this warmstart, var-D converges to a
         * low-entropy product-state |phi> but the corresponding D
         * doesn't reach the cat-state image, leaving an O(0.1)
         * energy gap to the exact GS. */
        uint32_t n = moonlab_ca_mps_num_qubits(state);
        ca_mps_error_t e = moonlab_ca_mps_h(state, 0);
        if (e != CA_MPS_SUCCESS) return e;
        for (uint32_t q = 0; q + 1 < n; q++) {
            e = moonlab_ca_mps_cnot(state, q, q + 1);
            if (e != CA_MPS_SUCCESS) return e;
        }
    } else if (cfg.warmstart == CA_MPS_WARMSTART_STABILIZER_SUBGROUP) {
        /* Gauge-aware warmstart: build a Clifford D that places
         * D|0^n> in the +1 eigenspace of every supplied stabilizer
         * generator (the Gauss-law operators of an LGT, or the
         * stabilizers of a quantum error-correcting code).  See
         * ca_mps_var_d_stab_warmstart.{c,h} for the symplectic
         * Gauss-Jordan construction. */
        if (!cfg.warmstart_stab_paulis || cfg.warmstart_stab_num_gens == 0) {
            return CA_MPS_ERR_INVALID;
        }
        ca_mps_error_t e = moonlab_ca_mps_apply_stab_subgroup_warmstart(
            state,
            cfg.warmstart_stab_paulis,
            cfg.warmstart_stab_num_gens);
        if (e != CA_MPS_SUCCESS) return e;
    }

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
        inner.composite_2gate  = cfg.composite_2gate;
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
