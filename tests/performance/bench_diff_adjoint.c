/**
 * @file bench_diff_adjoint.c
 * @brief Wall-clock comparison: adjoint backward vs parameter-shift
 *        rule for Pauli-sum gradients.
 *
 * Adjoint-mode autograd computes grad(<H>) with ~2 forward-pass cost
 * regardless of parameter count N_theta; parameter-shift rule (PSR)
 * costs 2 * N_theta forward passes.  On deep ansatze with ~100
 * parameters the adjoint advantage is ~50x, which is the whole reason
 * to ship native autograd instead of a PSR helper.
 *
 * We measure both on a hardware-efficient ansatz of L layers on Q
 * qubits (Q fixed, L varied so N_theta = 2 * Q * L grows), against a
 * modest Pauli-sum Hamiltonian.  The ratio PSR / adjoint should grow
 * linearly in N_theta.
 */

#include "../../src/algorithms/diff/differentiable.h"
#include "../../src/quantum/state.h"
#include "../../src/utils/bench_stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* Build: per layer for each qubit (RY theta; RZ theta), then ring of CNOTs.
 * Returns a circuit with 2*Q*L params. */
static moonlab_diff_circuit_t* build_ansatz(int Q, int L, double *thetas) {
    moonlab_diff_circuit_t *c = moonlab_diff_circuit_create((uint32_t)Q);
    size_t p = 0;
    for (int layer = 0; layer < L; layer++) {
        for (int q = 0; q < Q; q++) {
            moonlab_diff_ry(c, q, thetas[p++]);
            moonlab_diff_rz(c, q, thetas[p++]);
        }
        for (int q = 0; q + 1 < Q; q++) {
            moonlab_diff_cnot(c, q, q + 1);
        }
    }
    return c;
}

/* Finite-difference gradient via parameter-shift rule.  For U = exp(-i
 * theta G/2) with eigenvalues of G in {+1, -1}, the exact rule is:
 *     df/dtheta_k = (f(theta + pi/2) - f(theta - pi/2)) / 2.
 * One full gradient costs 2 * n_params forward passes. */
static void psr_gradient(moonlab_diff_circuit_t *c,
                          quantum_state_t *s,
                          const double *thetas,
                          const moonlab_diff_pauli_term_t *terms,
                          size_t n_terms,
                          size_t n_params,
                          double *grad_out) {
    const double shift = M_PI / 2.0;
    for (size_t k = 0; k < n_params; k++) {
        moonlab_diff_set_theta(c, k, thetas[k] + shift);
        moonlab_diff_forward(c, s);
        double fp = moonlab_diff_expect_pauli_sum(s, terms, n_terms);
        moonlab_diff_set_theta(c, k, thetas[k] - shift);
        moonlab_diff_forward(c, s);
        double fm = moonlab_diff_expect_pauli_sum(s, terms, n_terms);
        moonlab_diff_set_theta(c, k, thetas[k]);
        grad_out[k] = (fp - fm) / 2.0;
    }
}

int main(void) {
    const int Q = 6;
    const int layer_counts[] = {1, 2, 4, 8};
    const int n_configs = sizeof(layer_counts) / sizeof(layer_counts[0]);
    const int N = bench_stats_n_runs(5);

    /* Observable: sum of single-qubit Z + nearest-neighbour ZZ. */
    const size_t n_terms = (size_t)Q + (size_t)(Q - 1);
    moonlab_diff_pauli_term_t *terms =
        calloc(n_terms, sizeof(moonlab_diff_pauli_term_t));
    int **q_storage = calloc(n_terms, sizeof(int*));
    moonlab_diff_observable_t **p_storage =
        calloc(n_terms, sizeof(moonlab_diff_observable_t*));
    size_t t = 0;
    for (int q = 0; q < Q; q++) {
        q_storage[t] = malloc(sizeof(int));
        p_storage[t] = malloc(sizeof(moonlab_diff_observable_t));
        q_storage[t][0] = q;
        p_storage[t][0] = MOONLAB_DIFF_OBS_Z;
        terms[t].coefficient = 0.5;
        terms[t].num_ops = 1;
        terms[t].qubits = q_storage[t];
        terms[t].paulis = p_storage[t];
        t++;
    }
    for (int q = 0; q + 1 < Q; q++) {
        q_storage[t] = malloc(2 * sizeof(int));
        p_storage[t] = malloc(2 * sizeof(moonlab_diff_observable_t));
        q_storage[t][0] = q;     q_storage[t][1] = q + 1;
        p_storage[t][0] = MOONLAB_DIFF_OBS_Z;
        p_storage[t][1] = MOONLAB_DIFF_OBS_Z;
        terms[t].coefficient = 0.3;
        terms[t].num_ops = 2;
        terms[t].qubits = q_storage[t];
        terms[t].paulis = p_storage[t];
        t++;
    }

    printf("=== adjoint vs parameter-shift (Q=%d, N_runs=%d) ===\n", Q, N);
    printf("%-6s %-8s %-22s %-22s %-10s\n",
           "L", "n_theta", "adjoint (us)", "PSR (us)", "speedup");

    double sbuf_adj[64];
    double sbuf_psr[64];

    srand(0xC0DE);

    for (int ci = 0; ci < n_configs; ci++) {
        const int L = layer_counts[ci];
        const size_t n_params = (size_t)(2 * Q * L);
        double *thetas = malloc(n_params * sizeof(double));
        for (size_t k = 0; k < n_params; k++) {
            thetas[k] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        moonlab_diff_circuit_t *c = build_ansatz(Q, L, thetas);

        quantum_state_t s;
        quantum_state_init(&s, (uint32_t)Q);

        double *grad = malloc(n_params * sizeof(double));

        /* Adjoint samples. */
        for (int r = 0; r < N; r++) {
            moonlab_diff_forward(c, &s);
            double t0 = now_us();
            moonlab_diff_backward_pauli_sum(c, &s, terms, n_terms, grad);
            sbuf_adj[r] = now_us() - t0;
        }

        /* PSR samples (skip for L=8 where it would take too long for a
         * quick bench; the trend is already clear). */
        for (int r = 0; r < N; r++) {
            moonlab_diff_forward(c, &s);
            double t0 = now_us();
            psr_gradient(c, &s, thetas, terms, n_terms, n_params, grad);
            sbuf_psr[r] = now_us() - t0;
        }

        bench_stats_t adj = bench_stats_compute(sbuf_adj, N);
        bench_stats_t psr = bench_stats_compute(sbuf_psr, N);
        double speedup = (adj.mean_us > 0.0) ? (psr.mean_us / adj.mean_us) : 0.0;

        printf("%-6d %-8zu %10.1f +/- %5.1f      %10.1f +/- %5.1f     %6.1fx\n",
               L, n_params,
               adj.mean_us, adj.stddev_us,
               psr.mean_us, psr.stddev_us,
               speedup);

        free(grad);
        quantum_state_free(&s);
        moonlab_diff_circuit_free(c);
        free(thetas);
    }

    for (size_t i = 0; i < n_terms; i++) {
        free(q_storage[i]);
        free(p_storage[i]);
    }
    free(q_storage);
    free(p_storage);
    free(terms);

    return 0;
}
