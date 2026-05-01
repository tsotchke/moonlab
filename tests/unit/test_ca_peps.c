/**
 * @file test_ca_peps.c
 * @brief Validate the CA-PEPS row-major-MPS implementation.
 *
 * The v0.2.1 CA-PEPS embeds the physical factor as a row-major MPS
 * over Lx * Ly qubits, so by construction it must reproduce CA-MPS
 * results when fed the same gate sequence with the same linear-index
 * mapping.  This test runs a small mixed Clifford + rotation circuit
 * on a 3x3 lattice through both APIs and checks that every Pauli-
 * string expectation agrees to <1e-10.  It also exercises the
 * adjacency check on 2-qubit Cliffords and the create/clone/free
 * lifecycle.
 */

#include "../../src/algorithms/tensor_network/ca_peps.h"
#include "../../src/algorithms/tensor_network/ca_mps.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERT FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        return 1; \
    } \
} while (0)

#define ASSERT_OK(call) do { \
    int _e = (int)(call); \
    if (_e != 0) { \
        fprintf(stderr, "ASSERT_OK failed (rc=%d) at %s:%d: %s\n", \
                _e, __FILE__, __LINE__, #call); \
        return 1; \
    } \
} while (0)

/* Apply a small mixed Clifford + rotation circuit on the linear-index
 * view of a 3x3 lattice.  The same gate sequence runs through both
 * CA-PEPS and CA-MPS; results must match. */
static int apply_circuit_peps(moonlab_ca_peps_t* p) {
    const double t1 = 0.37, t2 = -0.91, t3 = 1.23;
    /* Clifford layer: H on every site, CNOTs along the first row,
     * CZs on the leftmost column. */
    for (uint32_t q = 0; q < 9; q++) ASSERT_OK(moonlab_ca_peps_h(p, q));
    ASSERT_OK(moonlab_ca_peps_cnot(p, 0, 1));   /* (0,0)-(1,0) horizontal */
    ASSERT_OK(moonlab_ca_peps_cnot(p, 1, 2));   /* (1,0)-(2,0) horizontal */
    ASSERT_OK(moonlab_ca_peps_cz(p, 0, 3));     /* (0,0)-(0,1) vertical */
    ASSERT_OK(moonlab_ca_peps_cz(p, 3, 6));     /* (0,1)-(0,2) vertical */
    /* Non-Clifford rotations on the centre site. */
    ASSERT_OK(moonlab_ca_peps_rx(p, 4, t1));
    ASSERT_OK(moonlab_ca_peps_ry(p, 4, t2));
    ASSERT_OK(moonlab_ca_peps_rz(p, 4, t3));
    /* Another Clifford layer to push the rotation through. */
    ASSERT_OK(moonlab_ca_peps_s(p, 4));
    ASSERT_OK(moonlab_ca_peps_sdag(p, 5));
    ASSERT_OK(moonlab_ca_peps_x(p, 8));
    return 0;
}

static int apply_circuit_mps(moonlab_ca_mps_t* m) {
    const double t1 = 0.37, t2 = -0.91, t3 = 1.23;
    for (uint32_t q = 0; q < 9; q++) ASSERT_OK(moonlab_ca_mps_h(m, q));
    ASSERT_OK(moonlab_ca_mps_cnot(m, 0, 1));
    ASSERT_OK(moonlab_ca_mps_cnot(m, 1, 2));
    ASSERT_OK(moonlab_ca_mps_cz(m, 0, 3));
    ASSERT_OK(moonlab_ca_mps_cz(m, 3, 6));
    ASSERT_OK(moonlab_ca_mps_rx(m, 4, t1));
    ASSERT_OK(moonlab_ca_mps_ry(m, 4, t2));
    ASSERT_OK(moonlab_ca_mps_rz(m, 4, t3));
    ASSERT_OK(moonlab_ca_mps_s(m, 4));
    ASSERT_OK(moonlab_ca_mps_sdag(m, 5));
    ASSERT_OK(moonlab_ca_mps_x(m, 8));
    return 0;
}

static int test_lifecycle(void) {
    moonlab_ca_peps_t* p = moonlab_ca_peps_create(3, 3, 16);
    ASSERT(p != NULL, "create returned NULL");
    ASSERT(moonlab_ca_peps_lx(p) == 3, "Lx wrong");
    ASSERT(moonlab_ca_peps_ly(p) == 3, "Ly wrong");
    ASSERT(moonlab_ca_peps_num_qubits(p) == 9, "num_qubits wrong");
    ASSERT(moonlab_ca_peps_max_bond_dim(p) == 16, "max_bond wrong");

    moonlab_ca_peps_t* c = moonlab_ca_peps_clone(p);
    ASSERT(c != NULL, "clone returned NULL");
    ASSERT(moonlab_ca_peps_num_qubits(c) == 9, "clone num_qubits wrong");

    moonlab_ca_peps_free(c);
    moonlab_ca_peps_free(p);
    /* Free on NULL must be a no-op. */
    moonlab_ca_peps_free(NULL);
    return 0;
}

static int test_adjacency_validation(void) {
    moonlab_ca_peps_t* p = moonlab_ca_peps_create(3, 3, 8);
    ASSERT(p != NULL, "create");
    /* Horizontal adjacency: (0,0)-(1,0). */
    ASSERT(moonlab_ca_peps_cnot(p, 0, 1) == CA_PEPS_SUCCESS, "horiz adjacent CNOT");
    /* Vertical adjacency: (0,0)-(0,1). */
    ASSERT(moonlab_ca_peps_cnot(p, 0, 3) == CA_PEPS_SUCCESS, "vert adjacent CNOT");
    /* Diagonal is *not* adjacent on the square lattice. */
    ASSERT(moonlab_ca_peps_cnot(p, 0, 4) == CA_PEPS_ERR_QUBIT, "diagonal rejected");
    /* Same site is not adjacent. */
    ASSERT(moonlab_ca_peps_cz(p, 1, 1) == CA_PEPS_ERR_QUBIT, "same-site rejected");
    /* Out of range. */
    ASSERT(moonlab_ca_peps_h(p, 9) == CA_PEPS_ERR_QUBIT, "out-of-range rejected");
    moonlab_ca_peps_free(p);
    return 0;
}

static int test_matches_ca_mps(void) {
    moonlab_ca_peps_t* p = moonlab_ca_peps_create(3, 3, 32);
    moonlab_ca_mps_t*  m = moonlab_ca_mps_create(9, 32);
    ASSERT(p != NULL && m != NULL, "create both");

    if (apply_circuit_peps(p) != 0) return 1;
    if (apply_circuit_mps(m)  != 0) return 1;

    /* Compare a mixed bag of Pauli-string expectations.  Pauli code
     * convention matches CA-MPS: 0=I, 1=X, 2=Y, 3=Z. */
    const uint8_t cases[][9] = {
        {0,0,0,0,3,0,0,0,0},          /* Z on centre */
        {1,0,0,0,0,0,0,0,1},          /* X on (0,0) and (2,2) */
        {3,0,0,3,0,0,3,0,0},          /* Z on left column */
        {0,3,0,0,3,0,0,3,0},          /* Z on middle column */
        {2,1,3,1,2,1,3,1,2},          /* dense XYZ checkerboard */
        {0,0,0,0,1,0,0,0,0},          /* X on centre */
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);

    for (size_t k = 0; k < n_cases; k++) {
        double _Complex zp = 0.0, zm = 0.0;
        ASSERT_OK(moonlab_ca_peps_expect_pauli(p, cases[k], &zp));
        ASSERT_OK(moonlab_ca_mps_expect_pauli(m, cases[k], &zm));
        const double err = cabs(zp - zm);
        if (err > 1e-10) {
            fprintf(stderr,
                    "case %zu: PEPS=%.12g+%.12gi  MPS=%.12g+%.12gi  err=%.3e\n",
                    k, creal(zp), cimag(zp), creal(zm), cimag(zm), err);
            moonlab_ca_peps_free(p);
            moonlab_ca_mps_free(m);
            return 1;
        }
    }

    moonlab_ca_peps_free(p);
    moonlab_ca_mps_free(m);
    return 0;
}

int main(void) {
    if (test_lifecycle() != 0) return 1;
    fprintf(stderr, "PASS test_lifecycle\n");
    if (test_adjacency_validation() != 0) return 1;
    fprintf(stderr, "PASS test_adjacency_validation\n");
    if (test_matches_ca_mps() != 0) return 1;
    fprintf(stderr, "PASS test_matches_ca_mps\n");
    return 0;
}
