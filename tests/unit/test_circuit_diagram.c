/**
 * @file test_circuit_diagram.c
 * @brief Circuit diagram builder smoke.
 */

#include "../../src/visualization/circuit_diagram.h"
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

static void test_bell_circuit(void) {
    fprintf(stdout, "\n-- circuit_diagram: build a Bell-state circuit --\n");
    circuit_diagram_t* c = circuit_create(2);
    CHECK(c != NULL, "circuit_create(2) returned non-NULL");
    if (!c) return;

    circuit_set_title(c, "Bell |Phi+>");
    CHECK(circuit_add_gate(c, 0, "H") >= 0, "add Hadamard on qubit 0");
    CHECK(circuit_add_controlled(c, 0, 1, "X") >= 0, "add CNOT(0,1)");
    CHECK(circuit_add_measurement(c, 0, 0) >= 0, "add measure(q0)");
    CHECK(circuit_add_measurement(c, 1, 1) >= 0, "add measure(q1)");

    int depth = circuit_get_depth(c);
    CHECK(depth > 0, "circuit_get_depth > 0 (got %d)", depth);

    circuit_free(c);
}

static void test_with_classical_register(void) {
    fprintf(stdout, "\n-- circuit_diagram: classical register --\n");
    circuit_diagram_t* c = circuit_create_with_classical(3, 3);
    CHECK(c != NULL, "create 3 qubits + 3 classical bits");
    if (!c) return;
    CHECK(circuit_add_toffoli(c, 0, 1, 2) >= 0, "add Toffoli(0,1,2)");
    CHECK(circuit_add_swap(c, 0, 2) >= 0, "add SWAP(0,2)");
    circuit_free(c);
}

int main(void) {
    fprintf(stdout, "=== circuit diagram builder smoke ===\n");
    test_bell_circuit();
    test_with_classical_register();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
