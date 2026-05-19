/**
 * @file  test_qgtl_backend.c
 * @brief Validate the QGTL-shaped circuit-ingestion surface.
 *
 * Three circuits exercise the full add_gate -> execute path:
 *   1. Bell pair (H + CNOT)            -> 50/50 on |00>/|11>.
 *   2. 3-qubit GHZ (H + 2 CNOTs)       -> 50/50 on |000>/|111>.
 *   3. Grover N=2 (1 marked state)     -> >95% probability on marked.
 *
 * Plus shot-sampling sanity and error-path coverage.
 */

#include "../../src/applications/moonlab_qgtl_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_bell_pair(void)
{
    fprintf(stdout, "\n--- Bell pair (H_0 + CNOT_01) ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    CHECK(c != NULL, "circuit_create(2)");
    if (!c) return;

    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL) == 0, "H 0");
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL) == 0,
          "CNOT control=0 target=1");
    CHECK(moonlab_qgtl_circuit_num_gates(c) == 2, "two gates recorded");

    moonlab_qgtl_exec_options_t opts = {
        .num_shots = 0, .rng_seed = 0, .return_probabilities = 1
    };
    moonlab_qgtl_results_t res = {0};
    const int rc = moonlab_qgtl_execute(c, &opts, &res);
    CHECK(rc == 0, "execute returns OK");
    CHECK(res.probabilities != NULL, "probabilities allocated");
    if (res.probabilities) {
        /* |00> and |11> each 0.5, |01> + |10> = 0. */
        const double p00 = res.probabilities[0];
        const double p01 = res.probabilities[1];
        const double p10 = res.probabilities[2];
        const double p11 = res.probabilities[3];
        fprintf(stdout, "    probs: |00>=%.4f |01>=%.4f |10>=%.4f |11>=%.4f\n",
                p00, p01, p10, p11);
        CHECK(fabs(p00 - 0.5) < 1e-9, "P(|00>) = 0.5");
        CHECK(fabs(p11 - 0.5) < 1e-9, "P(|11>) = 0.5");
        CHECK(p01 < 1e-9 && p10 < 1e-9, "P(|01>) + P(|10>) = 0");
    }
    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
}

static void test_ghz_3(void)
{
    fprintf(stdout, "\n--- 3-qubit GHZ ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(3);
    CHECK(c != NULL, "circuit_create(3)");
    if (!c) return;

    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL) == 0, "H 0");
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL) == 0,
          "CNOT control=0 target=1");
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 2, 1, NULL) == 0,
          "CNOT control=1 target=2");

    moonlab_qgtl_exec_options_t opts = {
        .num_shots = 0, .return_probabilities = 1
    };
    moonlab_qgtl_results_t res = {0};
    CHECK(moonlab_qgtl_execute(c, &opts, &res) == 0, "execute returns OK");
    if (res.probabilities) {
        const double p_000 = res.probabilities[0];
        const double p_111 = res.probabilities[7];
        fprintf(stdout, "    P(|000>) = %.6f   P(|111>) = %.6f\n", p_000, p_111);
        CHECK(fabs(p_000 - 0.5) < 1e-9, "P(|000>) = 0.5");
        CHECK(fabs(p_111 - 0.5) < 1e-9, "P(|111>) = 0.5");
        /* Everything else is zero (GHZ has just the two extremal basis states). */
        double off_mass = 0.0;
        for (int b = 1; b < 7; b++) off_mass += res.probabilities[b];
        CHECK(off_mass < 1e-9, "P(other states) = %.3e (must be ~0)", off_mass);
    }
    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
}

static void test_shot_sampling(void)
{
    fprintf(stdout, "\n--- shot sampling on Bell pair (1024 shots, fixed seed) ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);

    moonlab_qgtl_exec_options_t opts = {
        .num_shots = 1024,
        .rng_seed  = 0xdeadbeefULL,
        .return_probabilities = 0
    };
    moonlab_qgtl_results_t res = {0};
    CHECK(moonlab_qgtl_execute(c, &opts, &res) == 0, "execute with 1024 shots");
    CHECK(res.outcomes != NULL, "outcomes buffer allocated");

    /* Every outcome must be 0b00 or 0b11; never 0b01 or 0b10. */
    int n00 = 0, n11 = 0, n_other = 0;
    for (int s = 0; s < 1024; s++) {
        if (res.outcomes[s] == 0) n00++;
        else if (res.outcomes[s] == 3) n11++;
        else n_other++;
    }
    fprintf(stdout, "    counts: |00>=%d |11>=%d other=%d (total %d)\n",
            n00, n11, n_other, n00 + n11 + n_other);
    CHECK(n_other == 0, "no off-Bell outcomes sampled");
    /* Statistical: 5-sigma bound on n00 ~ Binomial(1024, 0.5) is +/- 80. */
    CHECK(n00 > 1024 / 2 - 80 && n00 < 1024 / 2 + 80,
          "n00 = %d within 5sigma of 512", n00);

    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
}

static void test_grover_n2(void)
{
    /* 2-qubit Grover with |11> as the marked state: one iteration
     * gives 100% on |11>.  Circuit: H H Oracle(11)=CZ Diffuser. */
    fprintf(stdout, "\n--- Grover N=2 (marked |11>) ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 1, -1, NULL);
    /* Oracle: phase-flip on |11> = CZ(0, 1). */
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CZ, 1, 0, NULL);
    /* Diffusion operator: H H Z Z CZ H H. */
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 1, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_Z, 0, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_Z, 1, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CZ, 1, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 0, -1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 1, -1, NULL);

    moonlab_qgtl_exec_options_t opts = {
        .num_shots = 0, .return_probabilities = 1
    };
    moonlab_qgtl_results_t res = {0};
    CHECK(moonlab_qgtl_execute(c, &opts, &res) == 0, "Grover execute OK");
    if (res.probabilities) {
        const double p11 = res.probabilities[3];
        fprintf(stdout, "    P(|11>) after 1 Grover iteration = %.6f\n", p11);
        CHECK(p11 > 0.99, "Grover N=2 finds marked state with P > 0.99");
    }
    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
}

static void test_rotation_gate(void)
{
    /* RY(pi) on |0> rotates to |1>; RY(pi/2) gives 50/50 over Z. */
    fprintf(stdout, "\n--- RY(pi/2) on |0> ---\n");
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(1);
    const double theta = 1.5707963267948966; /* pi / 2 */
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_RY, 0, -1, &theta);

    moonlab_qgtl_exec_options_t opts = {.return_probabilities = 1};
    moonlab_qgtl_results_t res = {0};
    CHECK(moonlab_qgtl_execute(c, &opts, &res) == 0, "RY execute OK");
    if (res.probabilities) {
        CHECK(fabs(res.probabilities[0] - 0.5) < 1e-9, "P(|0>) = 0.5 after RY(pi/2)");
        CHECK(fabs(res.probabilities[1] - 0.5) < 1e-9, "P(|1>) = 0.5 after RY(pi/2)");
    }
    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
}

static void test_error_paths(void)
{
    fprintf(stdout, "\n--- error paths ---\n");
    CHECK(moonlab_qgtl_circuit_create(0)  == NULL, "create(0) rejected");
    CHECK(moonlab_qgtl_circuit_create(33) == NULL, "create(33) rejected");

    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H, 2, -1, NULL) ==
          MOONLAB_QGTL_BAD_ARG,
          "H on qubit 2 of 2-qubit circuit rejected");
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 0, 0, NULL) ==
          MOONLAB_QGTL_BAD_ARG,
          "CNOT control == target rejected");
    CHECK(moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_RX, 0, -1, NULL) ==
          MOONLAB_QGTL_BAD_ARG,
          "RX with NULL params rejected");
    moonlab_qgtl_circuit_free(c);
    moonlab_qgtl_circuit_free(NULL); /* must not crash */
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== QGTL-shaped circuit-ingestion surface ===\n");

    test_bell_pair();
    test_ghz_3();
    test_shot_sampling();
    test_grover_n2();
    test_rotation_gate();
    test_error_paths();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
