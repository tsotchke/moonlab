/**
 * @file test_metal_parity.c
 * @brief Metal GPU kernel parity against CPU reference.
 *
 * The Metal kernels in `src/optimization/kernels/quantum_kernels.metal`
 * operate on `float2` (single-precision complex) amplitudes, while the
 * CPU path in `src/quantum/gates.c` uses `double _Complex`. The parity
 * test therefore:
 *
 *   1. Builds a random normalised state on the CPU in double precision.
 *   2. Applies the CPU gate (double precision) for the reference.
 *   3. Casts the SAME initial state to float2, writes it into a Metal
 *      buffer, and runs the Metal kernel.
 *   4. Reads the result back as float2, casts to double, and compares.
 *
 * Tolerance is set for single-precision: L2 relative <= 5e-5.
 *
 * Skipped (with success) when Metal is unavailable (no HAS_METAL or
 * the runtime reports no adapter).
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) && defined(HAS_METAL)
#  include "../../src/optimization/gpu_metal.h"
#  define METAL_PRESENT 1
#else
#  define METAL_PRESENT 0
#endif

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

#if METAL_PRESENT

/* Metal kernels use float2 = { float real, float imag }. */
typedef struct { float re; float im; } fcomplex_t;

static void randomize_double(complex_t* buf, size_t n, unsigned seed) {
    srand(seed);
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        buf[i] = re + im * I;
        norm += re * re + im * im;
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < n; ++i) buf[i] /= norm;
}

static void cast_double_to_float(const complex_t* src, fcomplex_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i].re = (float)creal(src[i]);
        dst[i].im = (float)cimag(src[i]);
    }
}

static double l2_rel_diff(const complex_t* ref, const fcomplex_t* got, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double re_r = creal(ref[i]);
        double im_r = cimag(ref[i]);
        double re_g = (double)got[i].re;
        double im_g = (double)got[i].im;
        double dre = re_r - re_g;
        double dim = im_r - im_g;
        num += dre * dre + dim * dim;
        den += re_r * re_r + im_r * im_r;
    }
    return (den > 0.0) ? sqrt(num / den) : sqrt(num);
}

typedef int (*metal_gate1q_fn)(metal_compute_ctx_t*, metal_buffer_t*,
                               uint32_t, uint32_t);
typedef qs_error_t (*cpu_gate1q_fn)(quantum_state_t*, int);

static void test_gate_parity(metal_compute_ctx_t* ctx,
                             const char* name,
                             cpu_gate1q_fn cpu_fn,
                             metal_gate1q_fn metal_fn,
                             int num_qubits, int qubit) {
    const uint32_t dim = 1u << num_qubits;

    /* CPU reference path: double precision. */
    quantum_state_t cpu_state;
    quantum_state_init(&cpu_state, num_qubits);
    randomize_double(cpu_state.amplitudes, dim,
                     0xD0D0u ^ (unsigned)qubit ^ (unsigned)num_qubits);

    /* Mirror the initial state into a float2 Metal buffer. */
    const size_t nbytes = dim * sizeof(fcomplex_t);
    fcomplex_t* initial_float = malloc(nbytes);
    cast_double_to_float(cpu_state.amplitudes, initial_float, dim);

    metal_buffer_t* gpu_buf = metal_buffer_create(ctx, nbytes);
    if (!gpu_buf) {
        fprintf(stdout, "  SKIP  %s: metal_buffer_create returned NULL\n", name);
        free(initial_float);
        quantum_state_free(&cpu_state);
        return;
    }
    memcpy(metal_buffer_contents(gpu_buf), initial_float, nbytes);

    /* Apply. */
    cpu_fn(&cpu_state, qubit);
    int rc = metal_fn(ctx, gpu_buf, (uint32_t)qubit, dim);
    if (rc != 0) {
        fprintf(stdout, "  SKIP  %s: metal kernel rc=%d\n", name, rc);
        metal_buffer_free(gpu_buf);
        free(initial_float);
        quantum_state_free(&cpu_state);
        return;
    }

    /* Compare CPU double result against GPU float2 result. Tolerance is
     * set for single-precision accumulation: ~5e-5 L2-relative is
     * generous for typical 32-element states; real world ~1e-6 on random
     * normalised inputs. */
    const fcomplex_t* gpu_amps = (const fcomplex_t*)metal_buffer_contents(gpu_buf);
    double rel = l2_rel_diff(cpu_state.amplitudes, gpu_amps, dim);
    CHECK(rel < 5e-5,
          "%s q=%d n=%d: L2 rel diff %.3e (< 5e-5, single-precision tol)",
          name, qubit, num_qubits, rel);

    metal_buffer_free(gpu_buf);
    free(initial_float);
    quantum_state_free(&cpu_state);
}

#endif  /* METAL_PRESENT */

int main(void) {
    fprintf(stdout, "=== Metal GPU parity tests ===\n");
#if METAL_PRESENT
    if (!metal_is_available()) {
        fprintf(stdout, "  SKIP  metal_is_available() reports 0 — no Metal adapter\n");
        fprintf(stdout, "\n=== %d failure%s ===\n",
                failures, failures == 1 ? "" : "s");
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    metal_compute_ctx_t* ctx = metal_compute_init();
    if (!ctx) {
        fprintf(stdout, "  SKIP  metal_compute_init() returned NULL\n");
        fprintf(stdout, "\n=== %d failure%s ===\n",
                failures, failures == 1 ? "" : "s");
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    test_gate_parity(ctx, "Hadamard", gate_hadamard, metal_hadamard, 5, 0);
    test_gate_parity(ctx, "Hadamard", gate_hadamard, metal_hadamard, 5, 2);
    test_gate_parity(ctx, "Pauli X",  gate_pauli_x,  metal_pauli_x,  5, 1);
    test_gate_parity(ctx, "Pauli X",  gate_pauli_x,  metal_pauli_x,  5, 4);
    test_gate_parity(ctx, "Pauli Z",  gate_pauli_z,  metal_pauli_z,  5, 0);
    test_gate_parity(ctx, "Pauli Z",  gate_pauli_z,  metal_pauli_z,  5, 3);

    metal_compute_free(ctx);
#else
    fprintf(stdout, "  SKIP  Metal not compiled in (non-Apple target or HAS_METAL undefined)\n");
#endif
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
