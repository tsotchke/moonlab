/**
 * @file test_clifford_pauli_api.c
 * @brief Unit tests for the CA-MPS-facing Pauli-string introspection API:
 *        clifford_row_pauli, clifford_conjugate_pauli, clifford_tableau_clone.
 *
 * The Heisenberg-picture action of D is: after applying Clifford G to the
 * tableau, row i destabilizer stores D X_i D^dagger and row n+i stabilizer
 * stores D Z_i D^dagger.
 *
 * Convention reminder (Pauli codes):  0=I, 1=X, 2=Y, 3=Z.
 * Convention reminder (phase codes):  0=+1, 1=+i, 2=-1, 3=-i.
 */

#include "../../src/backends/clifford/clifford.h"

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

static const char* pauli_char[] = { "I", "X", "Y", "Z" };

static void print_pauli(const char* label, const uint8_t* p, size_t n, int phase) {
    const char* sign_str[] = { "+", "+i", "-", "-i" };
    fprintf(stdout, "    %s = %s", label, sign_str[phase & 3]);
    for (size_t q = 0; q < n; q++) fprintf(stdout, "%s", pauli_char[p[q]]);
    fprintf(stdout, "\n");
}

static int pauli_eq(const uint8_t* a, const uint8_t* b, size_t n, int pa, int pb) {
    if ((pa & 3) != (pb & 3)) return 0;
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    fprintf(stdout, "=== clifford_row_pauli / conjugate_pauli / clone ===\n\n");

    const size_t n = 4;

    /* 1. Fresh tableau: D = I. Every destabilizer is X_i, every stabilizer is Z_i. */
    fprintf(stdout, "Test 1: |0000> tableau is identity (rows are single-qubit Paulis).\n");
    clifford_tableau_t* t = clifford_tableau_create(n);
    for (size_t q = 0; q < n; q++) {
        uint8_t destab[4], stab[4];
        int dphase = -1, sphase = -1;
        clifford_row_pauli(t, q, destab, &dphase);
        clifford_row_pauli(t, n + q, stab, &sphase);

        uint8_t expect_destab[4] = {0, 0, 0, 0}; expect_destab[q] = 1;
        uint8_t expect_stab[4]   = {0, 0, 0, 0}; expect_stab[q] = 3;
        CHECK(pauli_eq(destab, expect_destab, n, dphase, 0),
              "destab[%zu] = X_%zu  (got phase=%d)", q, q, dphase);
        CHECK(pauli_eq(stab, expect_stab, n, sphase, 0),
              "stab[%zu]   = Z_%zu  (got phase=%d)", q, q, sphase);
    }

    /* 2. Apply H_0. Now D = H_0. Heisenberg action: H Z_0 H = X_0, H X_0 H = Z_0.
     *    Other qubits unchanged. */
    fprintf(stdout, "\nTest 2: after H_0, stab[0] = X_0 and destab[0] = Z_0.\n");
    clifford_h(t, 0);
    {
        uint8_t destab0[4], stab0[4];
        int dp, sp;
        clifford_row_pauli(t, 0, destab0, &dp);
        clifford_row_pauli(t, n + 0, stab0, &sp);
        print_pauli("destab[0]", destab0, n, dp);
        print_pauli("stab[0]  ", stab0, n, sp);
        uint8_t expect_destab[4] = {3, 0, 0, 0};  /* Z on qubit 0 */
        uint8_t expect_stab[4]   = {1, 0, 0, 0};  /* X on qubit 0 */
        CHECK(pauli_eq(destab0, expect_destab, n, dp, 0), "H: destab[0] = Z_0");
        CHECK(pauli_eq(stab0, expect_stab, n, sp, 0),   "H: stab[0]   = X_0");
    }

    /* 3. Apply S_0.  Now D = S_0 H_0.  Heisenberg: S Z = Z, S X = Y.
     *    Starting from D = H_0 (stab[0] = X), applying S gives S X S^\dagger = Y,
     *    so stab[0] should now be Y_0. */
    fprintf(stdout, "\nTest 3: after S_0 (combined D = S_0 H_0), stab[0] = Y_0.\n");
    clifford_s(t, 0);
    {
        uint8_t stab0[4]; int sp;
        clifford_row_pauli(t, n + 0, stab0, &sp);
        print_pauli("stab[0]", stab0, n, sp);
        uint8_t expect[4] = {2, 0, 0, 0};  /* Y on qubit 0 */
        CHECK(pauli_eq(stab0, expect, n, sp, 0), "stab[0] = Y_0");
    }

    clifford_tableau_free(t);

    /* 4. CNOT_{0,1}: H Z_0 H = X_0 -> CNOT absorbs: CNOT X_0 CNOT = X_0 X_1.
     *    Fresh tableau + CNOT(0,1): stab[0] = Z_0 * Z_1? Wrong.
     *
     *    CNOT_{0,1} conjugation rules (Aaronson-Gottesman):
     *       X_0 -> X_0 X_1     (destab row 0 of output = X on both)
     *       X_1 -> X_1         (destab row 1 unchanged)
     *       Z_0 -> Z_0         (stab row 0 unchanged)
     *       Z_1 -> Z_0 Z_1     (stab row 1 = Z on 0 and 1)
     */
    fprintf(stdout, "\nTest 4: fresh + CNOT(0,1): destab[0]=X_0X_1, stab[1]=Z_0Z_1.\n");
    clifford_tableau_t* t2 = clifford_tableau_create(n);
    clifford_cnot(t2, 0, 1);
    {
        uint8_t destab0[4], stab1[4];
        int dp, sp;
        clifford_row_pauli(t2, 0, destab0, &dp);
        clifford_row_pauli(t2, n + 1, stab1, &sp);
        print_pauli("destab[0]", destab0, n, dp);
        print_pauli("stab[1]  ", stab1, n, sp);
        uint8_t expect_d0[4] = {1, 1, 0, 0};  /* X on 0 and 1 */
        uint8_t expect_s1[4] = {3, 3, 0, 0};  /* Z on 0 and 1 */
        CHECK(pauli_eq(destab0, expect_d0, n, dp, 0), "CNOT: destab[0] = X_0 X_1");
        CHECK(pauli_eq(stab1,   expect_s1, n, sp, 0), "CNOT: stab[1]   = Z_0 Z_1");
    }

    /* 5. Conjugation round-trip:
     *    clifford_conjugate_pauli(t, P, 0, out, &ph) should give D P D^dagger.
     *    On fresh |0000>: D = I so output == input, phase == 0. */
    fprintf(stdout, "\nTest 5: conjugate through identity Clifford is a no-op.\n");
    clifford_tableau_t* t3 = clifford_tableau_create(n);
    {
        uint8_t in[4]  = {1, 3, 2, 0};  /* X Z Y I */
        uint8_t out[4] = {0, 0, 0, 0};
        int out_ph = -1;
        clifford_conjugate_pauli(t3, in, 0, out, &out_ph);
        print_pauli("in ", in,  n, 0);
        print_pauli("out", out, n, out_ph);
        CHECK(pauli_eq(in, out, n, 0, out_ph), "identity conjugation preserves Pauli + phase");
    }
    clifford_tableau_free(t3);

    /* 6. Conjugation via H_0: D X_0 D^\dagger = H X_0 H = Z_0, D Z_0 D^\dagger = X_0.
     *    Input X_0 should come out Z_0; input Z_0 should come out X_0. */
    fprintf(stdout, "\nTest 6: conjugation through D = H_0.\n");
    clifford_tableau_t* t4 = clifford_tableau_create(n);
    clifford_h(t4, 0);
    {
        uint8_t in_x[4]  = {1, 0, 0, 0};  /* X_0 */
        uint8_t out[4];
        int out_ph = -1;
        clifford_conjugate_pauli(t4, in_x, 0, out, &out_ph);
        print_pauli("H X_0 H", out, n, out_ph);
        uint8_t expect[4] = {3, 0, 0, 0};  /* Z_0 */
        CHECK(pauli_eq(out, expect, n, out_ph, 0), "H X_0 H = Z_0");

        uint8_t in_z[4]  = {3, 0, 0, 0};
        clifford_conjugate_pauli(t4, in_z, 0, out, &out_ph);
        print_pauli("H Z_0 H", out, n, out_ph);
        uint8_t expect_x[4] = {1, 0, 0, 0};
        CHECK(pauli_eq(out, expect_x, n, out_ph, 0), "H Z_0 H = X_0");
    }
    clifford_tableau_free(t4);

    /* 7. Conjugation of two-site X_0 X_1 through CNOT_{0,1}:
     *    D X_0 D^\dagger = X_0 X_1, D X_1 D^\dagger = X_1, so product is X_0 X_1 X_1 = X_0. */
    fprintf(stdout, "\nTest 7: conjugation of X_0 X_1 through CNOT(0,1).\n");
    clifford_tableau_t* t5 = clifford_tableau_create(n);
    clifford_cnot(t5, 0, 1);
    {
        uint8_t in[4]  = {1, 1, 0, 0};  /* X_0 X_1 */
        uint8_t out[4];
        int out_ph = -1;
        clifford_conjugate_pauli(t5, in, 0, out, &out_ph);
        print_pauli("CNOT (X0 X1) CNOT", out, n, out_ph);
        uint8_t expect[4] = {1, 0, 0, 0};
        CHECK(pauli_eq(out, expect, n, out_ph, 0), "CNOT (X_0 X_1) CNOT = X_0");
    }
    clifford_tableau_free(t5);

    /* 7b. Round-trip invariant: for any P and tableau t representing C,
     *     clifford_conjugate_pauli_inverse(t, clifford_conjugate_pauli(t, P))
     *     should return P (up to a trivial sign round-trip). */
    fprintf(stdout, "\nTest 7b: round-trip C then C^dagger returns input Pauli.\n");
    clifford_tableau_t* t6 = clifford_tableau_create(n);
    clifford_h(t6, 0);          /* C = H_0 */
    clifford_cnot(t6, 0, 1);    /* C = CNOT(0,1) H_0 */
    clifford_s(t6, 2);          /* C = S_2 CNOT(0,1) H_0 */
    {
        uint8_t in_paulis[][4] = {
            {1, 0, 0, 0},  /* X_0 */
            {0, 1, 0, 0},  /* X_1 */
            {3, 3, 0, 0},  /* Z_0 Z_1 */
            {1, 1, 0, 0},  /* X_0 X_1 */
            {2, 2, 0, 0},  /* Y_0 Y_1 */
            {1, 3, 2, 0},  /* X Z Y I */
        };
        for (size_t k = 0; k < sizeof(in_paulis) / sizeof(in_paulis[0]); k++) {
            uint8_t forward[4], back[4];
            int fph = 0, bph = 0;
            clifford_conjugate_pauli(t6, in_paulis[k], 0, forward, &fph);
            clifford_conjugate_pauli_inverse(t6, forward, fph, back, &bph);
            print_pauli("  in     ", in_paulis[k], n, 0);
            print_pauli("  forward", forward, n, fph);
            print_pauli("  back   ", back, n, bph);
            char label[64];
            snprintf(label, sizeof(label),
                     "round-trip input %s%s%s%s",
                     pauli_char[in_paulis[k][0]], pauli_char[in_paulis[k][1]],
                     pauli_char[in_paulis[k][2]], pauli_char[in_paulis[k][3]]);
            CHECK(pauli_eq(in_paulis[k], back, n, 0, bph), "%s", label);
        }
    }
    clifford_tableau_free(t6);

    /* 8. Clone and mutate independently. */
    fprintf(stdout, "\nTest 8: clifford_tableau_clone isolation.\n");
    clifford_tableau_t* orig = clifford_tableau_create(n);
    clifford_h(orig, 0);
    clifford_tableau_t* copy = clifford_tableau_clone(orig);
    clifford_h(orig, 1);  /* mutate original only */
    {
        uint8_t s_orig[4], s_copy[4];
        int po, pc;
        clifford_row_pauli(orig, n + 1, s_orig, &po);
        clifford_row_pauli(copy, n + 1, s_copy, &pc);
        /* Original has H on qubit 1 now: stab[1] = X_1. Clone didn't. */
        uint8_t expect_orig[4] = {0, 1, 0, 0};
        uint8_t expect_copy[4] = {0, 0, 0, 3}; (void)expect_copy;
        uint8_t expect_copy_qubit1[4] = {0, 3, 0, 0};  /* unchanged Z_1 */
        CHECK(pauli_eq(s_orig, expect_orig, n, po, 0), "after H_1 on orig: stab[1] = X_1");
        CHECK(pauli_eq(s_copy, expect_copy_qubit1, n, pc, 0), "clone unchanged: stab[1] = Z_1");
    }
    clifford_tableau_free(orig);
    clifford_tableau_free(copy);

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
