/**
 * @file test_constants.c
 * @brief Regression test for hex-encoded constants in src/utils/constants.h.
 *
 * Every numerical constant in constants.h is stored as an IEEE 754 hex
 * pattern for bit-exact cross-platform reproducibility. A historical bug
 * in QC_2SQRT2_HEX (stored 3.4142 instead of 2.8284) broke the Bell-test
 * pass/fail logic for an entire release cycle because there was no
 * machine-checked ground truth for these values.
 *
 * This test pins every constant to the value its comment claims, to a
 * relative tolerance of 1e-14 (about 50 ULPs for doubles around 1.0).
 * Any future edit that breaks this test is almost certainly a typo.
 */

#include "../../src/utils/constants.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void check(const char* name, double got, double want) {
    const double tol = 1e-14;
    double rel = fabs(want) < 1e-300 ? fabs(got - want) : fabs(got - want) / fabs(want);
    if (rel > tol) {
        fprintf(stderr,
                "  FAIL  %-22s got=%.17g  want=%.17g  rel_err=%.3e\n",
                name, got, want, rel);
        failures++;
    } else {
        fprintf(stdout, "  OK    %-22s = %.17g\n", name, got);
    }
}

int main(void) {
    fprintf(stdout, "=== constants.h regression check ===\n");

    /* Mathematical constants. */
    check("QC_PI",            QC_PI,            3.141592653589793);
    check("QC_PI_2",          QC_PI_2,          1.5707963267948966);
    check("QC_PI_4",          QC_PI_4,          0.7853981633974483);
    check("QC_2PI",           QC_2PI,           6.283185307179586);
    check("QC_3PI_4",         QC_3PI_4,         2.356194490192345);

    check("QC_SQRT2",         QC_SQRT2,         1.4142135623730951);
    check("QC_SQRT2_INV",     QC_SQRT2_INV,     0.7071067811865476);
    check("QC_2SQRT2",        QC_2SQRT2,        2.8284271247461903);
    check("QC_TSIRELSON_BOUND", QC_TSIRELSON_BOUND, 2.8284271247461903);

    check("QC_SQRT3",         QC_SQRT3,         1.7320508075688772);
    check("QC_SQRT3_INV",     QC_SQRT3_INV,     0.5773502691896258);

    check("QC_E",             QC_E,             2.718281828459045);
    check("QC_E_INV",         QC_E_INV,         0.36787944117144233);
    check("QC_LN2",           QC_LN2,           0.6931471805599453);

    /* Physical constants (real values — NOT the bit-mixing constants
     * further down in the header, which are not physical). */
    check("QC_FINE_STRUCTURE", QC_FINE_STRUCTURE, 7.2973525693e-3);
    check("QC_PLANCK",         QC_PLANCK,         1.0);
    check("QC_RYDBERG",        QC_RYDBERG,        0.5);
    check("QC_RYDBERG_EV",     QC_RYDBERG_EV,     13.605693122994);
    check("QC_ELECTRON_G",     QC_ELECTRON_G,     2.00231930436256);
    check("QC_GOLDEN_RATIO",   QC_GOLDEN_RATIO,   1.6180339887498949);

    /* Cross-consistency checks. */
    check("QC_PI ~ pi",        QC_PI,    (double)M_PI);
    check("QC_E  ~ e",         QC_E,     exp(1.0));
    check("QC_LN2 ~ log(2)",   QC_LN2,   log(2.0));
    check("QC_GOLDEN ~ (1+sqrt5)/2", QC_GOLDEN_RATIO,
          (1.0 + sqrt(5.0)) / 2.0);
    check("QC_2SQRT2 ~ 2*sqrt(2)", QC_2SQRT2, 2.0 * sqrt(2.0));

    fprintf(stdout, "=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
