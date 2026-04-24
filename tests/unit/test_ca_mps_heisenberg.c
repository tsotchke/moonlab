/**
 * @file test_ca_mps_heisenberg.c
 * @brief CA-MPS Heisenberg-Hamiltonian energy check on small clusters.
 *
 * Verifies that moonlab_ca_mps_expect_pauli_sum correctly computes
 * <psi|H|psi> for the Heisenberg antiferromagnet
 *
 *   H = J sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j)
 *
 * on (i) a 2-site dimer (N=2, 1 bond; ground-state singlet energy -3J),
 * (ii) a 4-site open chain (exact ED comparison via dense state vector),
 * (iii) an 8-site open chain (the smallest system where accumulated
 * Clifford-pre-rotation stresses the conjugation path).
 *
 * In each case we build the Hamiltonian as a Pauli sum, prepare a
 * specific test state (Neel basis, singlet product, or a Clifford-
 * evolved basis state), and check the CA-MPS expectation matches a
 * direct dense-state-vector evaluation to 1e-9.
 *
 * This is the integration test for the primitive that kagome ground-
 * state search (and any VQE-on-CA-MPS driver) will build on.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <complex.h>
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

/* Dense-SV Pauli expectation for cross-check. */
static double _Complex sv_expect_pauli(const quantum_state_t* sv,
                                       const uint8_t* pauli) {
    size_t n = sv->num_qubits;
    size_t dim = sv->state_dim;
    size_t flip_mask = 0;
    for (size_t q = 0; q < n; q++) {
        if (pauli[q] == 1 || pauli[q] == 2) flip_mask |= ((size_t)1 << q);
    }
    double _Complex phase[] = { 1.0, 1.0*I, -1.0, -1.0*I };
    double _Complex acc = 0.0;
    for (size_t s = 0; s < dim; s++) {
        int i_pow = 0;
        for (size_t q = 0; q < n; q++) {
            uint8_t p = pauli[q];
            int bit = (int)((s >> q) & 1u);
            if (p == 2) i_pow = (i_pow + (bit ? 3 : 1)) & 3;
            else if (p == 3) { if (bit) i_pow = (i_pow + 2) & 3; }
        }
        acc += conj(sv->amplitudes[s ^ flip_mask]) * phase[i_pow] * sv->amplitudes[s];
    }
    return acc;
}

static double _Complex sv_expect_pauli_sum(const quantum_state_t* sv,
                                           const uint8_t* paulis,
                                           const double _Complex* coeffs,
                                           uint32_t num_terms) {
    size_t n = sv->num_qubits;
    double _Complex acc = 0.0;
    for (uint32_t k = 0; k < num_terms; k++) {
        acc += coeffs[k] * sv_expect_pauli(sv, paulis + (size_t)k * n);
    }
    return acc;
}

/* Build Heisenberg H = sum_<ij> (XX + YY + ZZ) for the given bond list. */
static void build_heisenberg_pauli_sum(uint32_t n,
                                        const uint32_t (*bonds)[2], uint32_t num_bonds,
                                        uint8_t** out_paulis,
                                        double _Complex** out_coeffs,
                                        uint32_t* out_num_terms) {
    uint32_t num_terms = 3 * num_bonds;
    uint8_t* paulis = (uint8_t*)calloc((size_t)num_terms * n, sizeof(uint8_t));
    double _Complex* coeffs = (double _Complex*)calloc(num_terms, sizeof(double _Complex));
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (uint8_t p = 1; p <= 3; p++) {
            uint32_t term = 3 * b + (p - 1);
            uint8_t* row = paulis + (size_t)term * n;
            row[i] = p;
            row[j] = p;
            coeffs[term] = 1.0;
        }
    }
    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_num_terms = num_terms;
}

/* Apply identical gate sequence to both CA-MPS and dense SV. */
static void apply_gate_pair(moonlab_ca_mps_t* ca, quantum_state_t* sv,
                            int op, uint32_t q, uint32_t q2) {
    switch (op) {
        case 0: moonlab_ca_mps_h(ca, q);    gate_hadamard(sv, (int)q); break;
        case 1: moonlab_ca_mps_x(ca, q);    gate_pauli_x(sv, (int)q); break;
        case 2: moonlab_ca_mps_z(ca, q);    gate_pauli_z(sv, (int)q); break;
        case 3: moonlab_ca_mps_cnot(ca, q, q2); gate_cnot(sv, (int)q, (int)q2); break;
    }
}

/* Test 1: 2-site dimer. */
static void test_dimer(void) {
    fprintf(stdout, "\n=== Test 1: N=2 Heisenberg dimer ===\n");
    const uint32_t n = 2;
    uint32_t bonds[1][2] = { {0, 1} };

    uint8_t* paulis; double _Complex* coeffs; uint32_t nterms;
    build_heisenberg_pauli_sum(n, bonds, 1, &paulis, &coeffs, &nterms);

    /* On |00>: <ZZ>=1, <XX>=<YY>=0 ==> <H> = +1. */
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, 16);
        double _Complex e;
        moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, nterms, &e);
        CHECK(cabs(e - 1.0) < 1e-9, "|00>: <H> = %+.9f + %+.9f i (expect +1)",
              creal(e), cimag(e));
        moonlab_ca_mps_free(s);
    }

    /* Singlet (|01> - |10>)/sqrt(2): <H> = -3. */
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, 16);
        moonlab_ca_mps_h(s, 0);
        moonlab_ca_mps_x(s, 1);
        moonlab_ca_mps_cnot(s, 0, 1);
        moonlab_ca_mps_z(s, 0);
        double _Complex e;
        moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, nterms, &e);
        CHECK(cabs(e - (-3.0)) < 1e-9,
              "singlet: <H> = %+.9f + %+.9f i (expect -3)",
              creal(e), cimag(e));
        moonlab_ca_mps_free(s);
    }

    free(paulis); free(coeffs);
}

/* Test 2: CA-MPS vs dense-SV on Heisenberg Hamiltonian after random Clifford. */
static void test_random_clifford_heisenberg(uint32_t n, uint32_t depth, uint32_t seed) {
    fprintf(stdout, "\n=== Test 2: N=%u chain + depth-%u Clifford + Heisenberg expectation ===\n",
            n, depth);

    /* Open chain: bonds (0,1), (1,2), ..., (n-2, n-1). */
    uint32_t num_bonds = n - 1;
    uint32_t (*bonds)[2] = calloc(num_bonds, sizeof(*bonds));
    for (uint32_t i = 0; i < num_bonds; i++) { bonds[i][0] = i; bonds[i][1] = i + 1; }

    uint8_t* paulis; double _Complex* coeffs; uint32_t nterms;
    build_heisenberg_pauli_sum(n, bonds, num_bonds, &paulis, &coeffs, &nterms);

    moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, 16);
    quantum_state_t* sv = quantum_state_create((int)n);

    uint64_t rng = 0xCAFECAFECAFECAFEULL ^ ((uint64_t)seed * 0x9E3779B9ULL);
    for (uint32_t step = 0; step < depth; step++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t op = (uint32_t)((rng >> 32) % 4);
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t q = (uint32_t)((rng >> 32) % n);
        uint32_t q2 = (q + 1) % n;
        apply_gate_pair(ca, sv, (int)op, q, q2);
    }

    double _Complex e_ca, e_sv;
    moonlab_ca_mps_expect_pauli_sum(ca, paulis, coeffs, nterms, &e_ca);
    e_sv = sv_expect_pauli_sum(sv, paulis, coeffs, nterms);

    CHECK(cabs(e_ca - e_sv) < 1e-9,
          "<H>_CA = %+.9f  <H>_SV = %+.9f   err = %.2e",
          creal(e_ca), creal(e_sv), cabs(e_ca - e_sv));

    moonlab_ca_mps_free(ca);
    quantum_state_free(sv);
    free(paulis); free(coeffs); free(bonds);
}

int main(void) {
    fprintf(stdout, "=== CA-MPS Heisenberg-Hamiltonian expectation cross-check ===\n");

    test_dimer();
    test_random_clifford_heisenberg(4,  5, 1);
    test_random_clifford_heisenberg(4, 10, 2);
    test_random_clifford_heisenberg(6,  5, 3);
    test_random_clifford_heisenberg(6, 10, 4);
    test_random_clifford_heisenberg(8,  5, 5);
    test_random_clifford_heisenberg(8, 10, 6);

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
