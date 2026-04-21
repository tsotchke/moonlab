/**
 * @file test_mutual_info.c
 * @brief Quantum mutual information I(A:B) sanity.
 */

#include "../../src/quantum/entanglement.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

int main(void) {
    fprintf(stdout, "=== quantum mutual information tests ===\n");

    /* 1. Product state |00>: I(0:1) = 0. */
    {
        quantum_state_t s;
        quantum_state_init(&s, 2);
        int A[1] = {0}, B[1] = {1};
        double mi = entanglement_mutual_information(&s, A, 1, B, 1);
        fprintf(stdout, "  |00>:        mi = %.6f  (expect 0)\n", mi);
        CHECK(mi < 1e-9, "product state has zero mutual information");
        quantum_state_free(&s);
    }

    /* 2. Bell state |Phi+>: I(0:1) = 2 * S_A = 2 * 1 = 2. */
    {
        quantum_state_t s;
        quantum_state_init(&s, 2);
        gate_hadamard(&s, 0); gate_cnot(&s, 0, 1);
        int A[1] = {0}, B[1] = {1};
        double mi = entanglement_mutual_information(&s, A, 1, B, 1);
        fprintf(stdout, "  |Phi+>:      mi = %.6f  (expect 2)\n", mi);
        CHECK(fabs(mi - 2.0) < 1e-9, "Bell state has mi = 2 bits");
        quantum_state_free(&s);
    }

    /* 3. GHZ_3 with A={0}, B={1} (qubit 2 traced out):
     *    rho_AB = (|00><00| + |11><11|)/2, so
     *    S(AB) = S(A) = S(B) = 1, I(A:B) = 1 bit (classical GHZ correlations). */
    {
        quantum_state_t s;
        quantum_state_init(&s, 3);
        gate_hadamard(&s, 0); gate_cnot(&s, 0, 1); gate_cnot(&s, 1, 2);
        int A[1] = {0}, B[1] = {1};
        double mi = entanglement_mutual_information(&s, A, 1, B, 1);
        fprintf(stdout, "  GHZ_3 (0,1): mi = %.6f  (expect 1)\n", mi);
        CHECK(fabs(mi - 1.0) < 1e-9, "GHZ_3 reduced pair has I = 1 bit");
        quantum_state_free(&s);
    }

    /* 4. GHZ_3 with A = {0}, B = {1, 2}: A u B is everything, pure ->
     *    S_AB = 0, S_A = 1, S_B = 1, so mi = 2 bits. */
    {
        quantum_state_t s;
        quantum_state_init(&s, 3);
        gate_hadamard(&s, 0); gate_cnot(&s, 0, 1); gate_cnot(&s, 1, 2);
        int A[1] = {0}, B[2] = {1, 2};
        double mi = entanglement_mutual_information(&s, A, 1, B, 2);
        fprintf(stdout, "  GHZ_3 (0|12):mi = %.6f  (expect 2)\n", mi);
        CHECK(fabs(mi - 2.0) < 1e-9, "bipartite I across GHZ cut is 2 bits");
        quantum_state_free(&s);
    }

    /* 5. Disjointness check: overlapping A/B returns 0. */
    {
        quantum_state_t s;
        quantum_state_init(&s, 2);
        int A[1] = {0}, B[2] = {0, 1};
        double mi = entanglement_mutual_information(&s, A, 1, B, 2);
        CHECK(mi == 0.0, "overlapping partitions rejected");
        quantum_state_free(&s);
    }

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
