/** Forced-baseline regression for compiled AVX-512 deployments. */
#include "../../src/optimization/simd_dispatch.h"
#include "../../src/optimization/simd_avx512.h"
#include "../../src/optimization/simd_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    if (simd_runtime_has_avx512() != 0 ||
        simd_get_backend(SIMD_OP_SUM_SQUARED_MAG) == SIMD_BACKEND_AVX512) {
        fprintf(stderr, "forced baseline selected AVX-512\n");
        return EXIT_FAILURE;
    }
    if (avx512_is_available() != 0 ||
        strcmp(avx512_get_features(), "AVX-512 unavailable") != 0) {
        fprintf(stderr, "direct AVX-512 probe ignored baseline override\n");
        return EXIT_FAILURE;
    }
#ifdef SIMD_ARCH_X86
    if (simd_get_vector_width() > 32 || simd_get_unroll_factor() > 4) {
        fprintf(stderr, "forced baseline retained AVX-512 execution metadata\n");
        return EXIT_FAILURE;
    }
#endif
    complex_t values[4] = {1.0 + 2.0 * I, -3.0 + 0.5 * I,
                            0.25 - 1.5 * I, 2.0 - 0.25 * I};
    const double expected = 1.0 + 4.0 + 9.0 + .25 + .0625 + 2.25 + 4.0 + .0625;
    if (fabs(simd_sum_squared_magnitudes(values, 4) - expected) > 1e-12) {
        fprintf(stderr, "baseline SIMD result mismatch\n");
        return EXIT_FAILURE;
    }
    puts("forced-baseline SIMD portability PASS");
    return EXIT_SUCCESS;
}
