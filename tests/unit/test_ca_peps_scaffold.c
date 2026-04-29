/**
 * @file test_ca_peps_scaffold.c
 * @brief Validates that the CA-PEPS scaffold compiles, links, and
 *        returns NOT_IMPLEMENTED at runtime as documented.
 *
 * The CA-PEPS module ships as scaffolding only -- every entry point
 * returns CA_PEPS_ERR_NOT_IMPLEMENTED.  This test verifies:
 *   1. moonlab_ca_peps_create / free work (struct lifecycle is real);
 *   2. introspection accessors return the values passed to create;
 *   3. clone produces a distinct handle;
 *   4. every gate / measurement entry point returns NOT_IMPLEMENTED.
 *
 * The test ensures consumers fail loudly (and immediately) when they
 * try to use CA-PEPS before the real implementation lands, rather
 * than getting silently-wrong results.
 */

#include "../../src/algorithms/tensor_network/ca_peps.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    fprintf(stdout, "=== CA-PEPS scaffold smoke test ===\n");

    moonlab_ca_peps_t* s = moonlab_ca_peps_create(4, 3, 16);
    CHECK(s != NULL, "moonlab_ca_peps_create returned NULL");
    if (!s) return 1;

    CHECK(moonlab_ca_peps_lx(s) == 4, "Lx mismatch: %u", moonlab_ca_peps_lx(s));
    CHECK(moonlab_ca_peps_ly(s) == 3, "Ly mismatch: %u", moonlab_ca_peps_ly(s));
    CHECK(moonlab_ca_peps_num_qubits(s) == 12,
          "num_qubits mismatch: %u", moonlab_ca_peps_num_qubits(s));
    CHECK(moonlab_ca_peps_max_bond_dim(s) == 16,
          "max_bond_dim mismatch: %u", moonlab_ca_peps_max_bond_dim(s));

    moonlab_ca_peps_t* clone = moonlab_ca_peps_clone(s);
    CHECK(clone != NULL, "clone returned NULL");
    CHECK(clone != s, "clone returned the same handle as source");

    /* Every gate entry point returns NOT_IMPLEMENTED. */
    CHECK(moonlab_ca_peps_h(s, 0) == CA_PEPS_ERR_NOT_IMPLEMENTED,
          "h didn't return NOT_IMPLEMENTED");
    CHECK(moonlab_ca_peps_s(s, 1) == CA_PEPS_ERR_NOT_IMPLEMENTED,
          "s didn't return NOT_IMPLEMENTED");
    CHECK(moonlab_ca_peps_cnot(s, 0, 1) == CA_PEPS_ERR_NOT_IMPLEMENTED,
          "cnot didn't return NOT_IMPLEMENTED");
    CHECK(moonlab_ca_peps_rx(s, 0, 0.5) == CA_PEPS_ERR_NOT_IMPLEMENTED,
          "rx didn't return NOT_IMPLEMENTED");

    uint8_t pauli[12] = {0};
    pauli[0] = 3;  /* Z on qubit 0 */
    double _Complex e = 0;
    CHECK(moonlab_ca_peps_expect_pauli(s, pauli, &e) == CA_PEPS_ERR_NOT_IMPLEMENTED,
          "expect_pauli didn't return NOT_IMPLEMENTED");

    moonlab_ca_peps_free(clone);
    moonlab_ca_peps_free(s);

    fprintf(stdout, "=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
