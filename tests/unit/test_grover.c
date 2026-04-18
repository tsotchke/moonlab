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
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
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

int main(void) {
    fprintf(stdout, "=== Grover smoke tests ===\n");
    one_case(3, 0b101);
    one_case(4, 0b1010);
    one_case(5, 0b01101);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
