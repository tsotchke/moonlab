/**
 * @file test_grover.c
 * @brief End-to-end smoke test for Grover's search algorithm.
 *
 * Grover's algorithm on an N = 2^n search space over a single marked
 * state finds the target with probability approaching 1 as long as we
 * use ceil(pi/4 * sqrt(N)) iterations (and slightly lower otherwise).
 * This test pins that behaviour at n in {3, 4, 5}.
 */

#include "../../src/algorithms/grover.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void one_case(int num_qubits, uint64_t target) {
    quantum_state_t state;
    quantum_state_init(&state, num_qubits);

    entropy_ctx_t hw_ctx;
    entropy_init(&hw_ctx);
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy,
        (quantum_entropy_fn)entropy_get_bytes, &hw_ctx);

    grover_config_t cfg = {
        .num_qubits = (size_t)num_qubits,
        .marked_state = target,
        .num_iterations = 0,          /* ignored when auto-optimal is set */
        .use_optimal_iterations = 1,
    };

    grover_result_t res = grover_search(&state, &cfg, &entropy);

    /* The "success probability" reported by the algorithm is the
     * amplitude^2 on the marked state just before measurement. For
     * any reasonable iteration count, that should exceed 0.8 for
     * small N and approach 1 for larger N. */
    CHECK(res.success_probability >= 0.8,
          "n=%d target=0x%llx success_prob=%.4f (>= 0.8)",
          num_qubits, (unsigned long long)target,
          res.success_probability);

    /* The measurement outcome is stochastic but for success_prob > 0.8
     * it should almost always match the target. We cannot make it
     * deterministic without a controlled RNG, so only require
     * success_probability bound here. */

    quantum_state_free(&state);
}

/* Multi-marked search: with k marked states in N = 2^n, Grover's
 * optimal iteration count is floor(pi/4 * sqrt(N/k)); after that the
 * total probability on the marked set approaches 1. Here k=3 markers
 * on n=4 (N=16) gives opt r ≈ 1.8 -> use 2 iterations; Σ P_marked
 * should comfortably exceed 0.9. */
static void test_multi_marked(void) {
    fprintf(stdout, "\n-- Grover: multi-marked search (k=3 on n=4) --\n");
    const int n = 4;
    const uint64_t marked[3] = { 0b0011, 0b0110, 0b1100 };
    const size_t k = sizeof(marked) / sizeof(marked[0]);
    const double phases[3] = { M_PI, M_PI, M_PI };  /* full phase flip */

    quantum_state_t state;
    quantum_state_init(&state, n);
    for (int q = 0; q < n; q++) gate_hadamard(&state, q);

    size_t N = (size_t)1 << n;
    size_t iters = (size_t)floor(M_PI / 4.0 * sqrt((double)N / (double)k));
    if (iters == 0) iters = 1;

    for (size_t i = 0; i < iters; i++) {
        qs_error_t oerr = grover_oracle_multi_phase(&state, marked, phases, k);
        CHECK(oerr == QS_SUCCESS, "oracle iter %zu", i);
        qs_error_t derr = grover_diffusion(&state);
        CHECK(derr == QS_SUCCESS, "diffusion iter %zu", i);
    }

    double p_marked = 0.0;
    for (size_t i = 0; i < k; i++) {
        double a = cabs(state.amplitudes[marked[i]]);
        p_marked += a * a;
    }
    fprintf(stdout, "    iters=%zu sum P(marked) = %.4f\n",
            iters, p_marked);
    CHECK(p_marked >= 0.9,
          "multi-marked search concentrates mass on marked set (got %.4f)",
          p_marked);

    quantum_state_free(&state);
}

int main(void) {
    fprintf(stdout, "=== Grover smoke tests ===\n");
    one_case(3, 0b101);
    one_case(4, 0b1010);
    one_case(5, 0b01101);
    test_multi_marked();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
