/**
 * @file test_simd_dispatch_portability.c
 * @brief Forced-baseline regression for compiled AVX-512 deployments.
 *
 * This test is deliberately not skipped on AVX-512 hosts.  It forces the
 * baseline runtime decision, then exercises a real dispatched primitive and
 * verifies that AVX-512 is not selected.  The same executable therefore
 * represents a downstream non-AVX-512 runner even when the build host has the
 * optional AVX-512 translation unit.
 */

#include "../../src/optimization/simd_dispatch.h"
#include "../../src/optimization/simd_avx512.h"
#include "../../src/optimization/simd_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int force_baseline(void)
{
#ifdef _WIN32
    return _putenv_s("MOONLAB_SIMD_FORCE_BASELINE", "1");
#else
    return setenv("MOONLAB_SIMD_FORCE_BASELINE", "1", 1);
#endif
}

int main(void)
{
    if (force_baseline() != 0) {
        fprintf(stderr, "failed to set baseline override\n");
        return EXIT_FAILURE;
    }

    if (simd_runtime_has_avx512() != 0) {
        fprintf(stderr, "forced baseline still reports AVX-512\n");
        return EXIT_FAILURE;
    }
    if (avx512_is_available() != 0) {
        fprintf(stderr, "direct AVX-512 availability probe ignored baseline override\n");
        return EXIT_FAILURE;
    }
    if (strcmp(avx512_get_features(), "AVX-512 unavailable") != 0) {
        fprintf(stderr, "direct AVX-512 feature probe was not baseline-safe\n");
        return EXIT_FAILURE;
    }
    if (simd_get_backend(SIMD_OP_SUM_SQUARED_MAG) == SIMD_BACKEND_AVX512) {
        fprintf(stderr, "forced baseline selected AVX-512\n");
        return EXIT_FAILURE;
    }

    complex_t values[4] = {
        1.0 + 2.0 * I, -3.0 + 0.5 * I,
        0.25 - 1.5 * I, 2.0 - 0.25 * I,
    };
    const double expected = 1.0 * 1.0 + 2.0 * 2.0 +
                            3.0 * 3.0 + 0.5 * 0.5 +
                            0.25 * 0.25 + 1.5 * 1.5 +
                            2.0 * 2.0 + 0.25 * 0.25;
    const double got = simd_sum_squared_magnitudes(values, 4);
    if (fabs(got - expected) > 1e-12) {
        fprintf(stderr, "baseline SIMD result mismatch: %.17g != %.17g\n",
                got, expected);
        return EXIT_FAILURE;
    }

    puts("forced-baseline SIMD portability PASS");
    return EXIT_SUCCESS;
}
