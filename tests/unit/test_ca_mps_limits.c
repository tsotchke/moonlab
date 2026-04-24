/**
 * @file test_ca_mps_limits.c
 * @brief Unit tests for CA-MPS reduction limits.
 *
 * The CA-MPS factorization |psi> = C|phi> must reduce correctly to
 * (i) the bare Clifford tableau when no non-Clifford gates are applied,
 * (ii) the bare MPS when no Clifford gates are applied, and
 * (iii) textbook stabilizer expectations for small Bell / GHZ states.
 *
 * These are regression tests.  Any future change to the tableau, the MPS,
 * or the CA-MPS glue code must leave these invariants undisturbed.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"

#include <complex.h>
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

#define CLOSE(a, b) (fabs((a) - (b)) < 1e-9)
#define CLOSEC(a, b) (cabs((a) - (b)) < 1e-9)

int main(void) {
    fprintf(stdout, "=== CA-MPS reduction-limit tests ===\n\n");

    /* ------------------------------------------------------------ */
    fprintf(stdout, "Test 1: |0> on 1 qubit.  <Z> = +1.\n");
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        uint8_t p_z[1] = {3};
        double _Complex ez;
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        CHECK(CLOSEC(ez, 1.0), "<Z> on |0> = %.9f + %.9f i", creal(ez), cimag(ez));
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 2: H|0> = |+>.  <X> = +1, <Z> = 0.\n");
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        moonlab_ca_mps_h(s, 0);
        uint8_t p_x[1] = {1};
        uint8_t p_z[1] = {3};
        double _Complex ex, ez;
        moonlab_ca_mps_expect_pauli(s, p_x, &ex);
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        CHECK(CLOSEC(ex, 1.0), "<X> on |+> = %.9f", creal(ex));
        CHECK(CLOSEC(ez, 0.0), "<Z> on |+> = %.9f", creal(ez));
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 3: Bell state (H_0, CNOT(0,1)). <XX>=<ZZ>=1, <YY>=-1.\n");
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(2, 32);
        moonlab_ca_mps_h(s, 0);
        moonlab_ca_mps_cnot(s, 0, 1);
        uint8_t p_xx[2] = {1, 1};
        uint8_t p_yy[2] = {2, 2};
        uint8_t p_zz[2] = {3, 3};
        double _Complex exx, eyy, ezz;
        moonlab_ca_mps_expect_pauli(s, p_xx, &exx);
        moonlab_ca_mps_expect_pauli(s, p_yy, &eyy);
        moonlab_ca_mps_expect_pauli(s, p_zz, &ezz);
        CHECK(CLOSEC(exx,  1.0), "<XX> = %.9f + %.9f i", creal(exx), cimag(exx));
        CHECK(CLOSEC(eyy, -1.0), "<YY> = %.9f + %.9f i", creal(eyy), cimag(eyy));
        CHECK(CLOSEC(ezz,  1.0), "<ZZ> = %.9f + %.9f i", creal(ezz), cimag(ezz));
        CHECK(moonlab_ca_mps_current_bond_dim(s) == 1,
              "Bell state has bond dim 1 in CA-MPS (MPS stays |00>)");
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 4: 8-qubit GHZ (H_0 + CNOT cascade).\n");
    {
        const uint32_t n = 8;
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, 32);
        moonlab_ca_mps_h(s, 0);
        for (uint32_t q = 0; q + 1 < n; q++) moonlab_ca_mps_cnot(s, q, q + 1);

        /* <Z_0 Z_1> = 1, ..., <Z_0 Z_{n-1}> = 1. */
        for (uint32_t q = 1; q < n; q++) {
            uint8_t p[n]; memset(p, 0, n);
            p[0] = 3; p[q] = 3;
            double _Complex e;
            moonlab_ca_mps_expect_pauli(s, p, &e);
            CHECK(CLOSEC(e, 1.0), "<Z_0 Z_%u> = %.9f", q, creal(e));
        }

        /* <X_0 X_1 ... X_{n-1}> = +1 for GHZ = (|0..0> + |1..1>)/sqrt(2). */
        uint8_t pxall[n]; for (uint32_t k = 0; k < n; k++) pxall[k] = 1;
        double _Complex exall;
        moonlab_ca_mps_expect_pauli(s, pxall, &exall);
        CHECK(CLOSEC(exall, 1.0), "<X_0...X_%u> GHZ product = %.9f",
              n - 1, creal(exall));

        CHECK(moonlab_ca_mps_current_bond_dim(s) == 1,
              "GHZ on n=8 has MPS bond dim 1 in CA-MPS");
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 5: Pure-MPS limit: rx(theta) with no Clifford.\n");
    {
        /* On |0>, rx(theta) produces cos(theta/2)|0> - i sin(theta/2)|1>. */
        double theta = 0.7;
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        moonlab_ca_mps_rx(s, 0, theta);

        /* exp(i theta X) = cos(theta) I + i sin(theta) X.
         * On |0>:  cos(theta) |0> + i sin(theta) |1>.
         * <Z> = |<0|...|0>|^2 - |<1|...|1>|^2 = cos^2 - sin^2 = cos(2 theta). */
        uint8_t p_z[1] = {3};
        double _Complex ez;
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        double expected = cos(2.0 * theta);
        CHECK(CLOSEC(ez, expected),
              "rx(%.2f): <Z> = %.9f (expect %.9f)", theta, creal(ez), expected);
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 6: Mixed: H + rz(theta).  Rz on |+> rotates in XY plane.\n");
    {
        /* |+> under rz(theta) = exp(i theta Z) = cos(theta) I + i sin(theta) Z:
         *   exp(i theta Z) |+> = cos(theta)|+> + i sin(theta) |-> .
         *   <X> on this state = cos^2(theta) - sin^2(theta) = cos(2 theta).
         *   <Y> on this state = -2 sin(theta) cos(theta) = -sin(2 theta).
         *     (Derivation: Y|+> = -i|->, Y|-> = i|+>; all cross terms give
         *     -sin cos and -cos sin, summing to -2 sin cos.) */
        double theta = 0.3;
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        moonlab_ca_mps_h(s, 0);
        moonlab_ca_mps_rz(s, 0, theta);

        uint8_t p_x[1] = {1}, p_y[1] = {2}, p_z[1] = {3};
        double _Complex ex, ey, ez;
        moonlab_ca_mps_expect_pauli(s, p_x, &ex);
        moonlab_ca_mps_expect_pauli(s, p_y, &ey);
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        double expX = cos(2.0 * theta);
        double expY = -sin(2.0 * theta);
        CHECK(CLOSEC(ex, expX), "<X> = %.9f (expect %.9f = cos(2 theta))",
              creal(ex), expX);
        CHECK(CLOSEC(ey, expY), "<Y> = %.9f (expect %.9f = -sin(2 theta))",
              creal(ey), expY);
        CHECK(CLOSEC(ez, 0.0), "<Z> = %.9f (expect 0)", creal(ez));
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 7: T|0> = |0> (T on basis state is diagonal, measurement unchanged).\n");
    {
        /* T|0> = |0>, T|+> phases to cos(pi/8)|0> + e^{i pi/4} sin(pi/8) ... actually
         * more straightforward: on |0>, T gives |0>, so Z expectation stays +1 and
         * X, Y expectations stay 0. */
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        moonlab_ca_mps_t_gate(s, 0);
        uint8_t p_x[1] = {1}, p_y[1] = {2}, p_z[1] = {3};
        double _Complex ex, ey, ez;
        moonlab_ca_mps_expect_pauli(s, p_x, &ex);
        moonlab_ca_mps_expect_pauli(s, p_y, &ey);
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        CHECK(CLOSEC(ez, 1.0), "T|0>: <Z> = %.9f", creal(ez));
        CHECK(CLOSEC(ex, 0.0), "T|0>: <X> = %.9f", creal(ex));
        CHECK(CLOSEC(ey, 0.0), "T|0>: <Y> = %.9f", creal(ey));
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 8: H T H |0>.  H T H = exp(i pi/8 X) under our t_gate convention.\n");
    {
        /* Moonlab's t_gate is exp(i (pi/8) Z) up to global phase.
         * So H T H = H exp(i pi/8 Z) H = exp(i pi/8 X)
         *         = cos(pi/8) I + i sin(pi/8) X
         * applied to |0>: cos(pi/8)|0> + i sin(pi/8)|1>.
         *   <Z> = cos^2(pi/8) - sin^2(pi/8) = cos(pi/4) = 1/sqrt(2) ≈ 0.70711
         *   <X> = 0 (state has 0 real amplitude in +/- basis)
         *   <Y> = +2 sin(pi/8) cos(pi/8) = +sin(pi/4) = +1/sqrt(2) ≈ +0.70711
         *     (Bloch view: rotate +Z by -pi/4 around X axis -> into +Y, +Z) */
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 32);
        moonlab_ca_mps_h(s, 0);
        moonlab_ca_mps_t_gate(s, 0);
        moonlab_ca_mps_h(s, 0);
        uint8_t p_x[1] = {1}, p_y[1] = {2}, p_z[1] = {3};
        double _Complex ex, ey, ez;
        moonlab_ca_mps_expect_pauli(s, p_x, &ex);
        moonlab_ca_mps_expect_pauli(s, p_y, &ey);
        moonlab_ca_mps_expect_pauli(s, p_z, &ez);
        CHECK(CLOSEC(ex,  0.0),          "H T H |0>: <X> = %.9f (expect 0)", creal(ex));
        CHECK(CLOSEC(ez,  M_SQRT1_2),    "H T H |0>: <Z> = %.9f (expect 1/sqrt(2))", creal(ez));
        CHECK(CLOSEC(ey,  M_SQRT1_2),    "H T H |0>: <Y> = %.9f (expect +1/sqrt(2))", creal(ey));
        moonlab_ca_mps_free(s);
    }

    /* ------------------------------------------------------------ */
    fprintf(stdout, "\nTest 9: Clone is independent.\n");
    {
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(3, 32);
        moonlab_ca_mps_h(s, 0);
        moonlab_ca_mps_t* c = moonlab_ca_mps_clone(s);
        moonlab_ca_mps_x(s, 1);  /* mutate original */
        uint8_t p_z_1[3] = {0, 3, 0};
        double _Complex eo, ec;
        moonlab_ca_mps_expect_pauli(s, p_z_1, &eo);
        moonlab_ca_mps_expect_pauli(c, p_z_1, &ec);
        CHECK(CLOSEC(eo, -1.0), "orig after X_1: <Z_1> = %.9f", creal(eo));
        CHECK(CLOSEC(ec,  1.0), "clone no X_1: <Z_1> = %.9f", creal(ec));
        moonlab_ca_mps_free(s);
        moonlab_ca_mps_free(c);
    }

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
