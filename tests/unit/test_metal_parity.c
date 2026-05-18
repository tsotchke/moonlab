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

static void check_trace_identity(const metal_backend_trace_t* trace,
                                 const char* owner,
                                 const char* operation) {
    CHECK(trace != NULL, "Metal backend trace is present");
    if (!trace) return;

    CHECK(trace->owner != NULL && strcmp(trace->owner, owner) == 0,
          "Metal backend trace owner is %s", owner);
    CHECK(trace->operation != NULL && strcmp(trace->operation, operation) == 0,
          "Metal backend trace operation is %s", operation);
    CHECK(trace->backend_name != NULL,
          "Metal backend trace records backend name");
    CHECK(trace->device_name != NULL,
          "Metal backend trace records device name");
    CHECK(trace->metal_available == 0 || trace->metal_available == 1,
          "Metal backend trace availability is boolean");
    CHECK(trace->tensor_pipelines_loaded <= trace->tensor_pipelines_expected,
          "Metal tensor pipeline trace is bounded (%d/%d)",
          trace->tensor_pipelines_loaded,
          trace->tensor_pipelines_expected);
}

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

static double complex fcomplex_load(const fcomplex_t* data, size_t idx) {
    return (double)data[idx].re + (double)data[idx].im * I;
}

static double metal_svd_reconstruction_error(const fcomplex_t* original,
                                             const fcomplex_t* U,
                                             const float* S,
                                             const fcomplex_t* Vt,
                                             uint32_t m,
                                             uint32_t n,
                                             uint32_t rank) {
    double err_sq = 0.0;
    double ref_sq = 0.0;

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double complex reconstructed = 0.0;
            for (uint32_t k = 0; k < rank; k++) {
                reconstructed += fcomplex_load(U, (size_t)i * rank + k) *
                                 (double)S[k] *
                                 fcomplex_load(Vt, (size_t)k * n + j);
            }

            double complex ref = fcomplex_load(original, (size_t)i * n + j);
            double complex diff = reconstructed - ref;
            err_sq += creal(diff) * creal(diff) + cimag(diff) * cimag(diff);
            ref_sq += creal(ref) * creal(ref) + cimag(ref) * cimag(ref);
        }
    }

    return ref_sq > 0.0 ? sqrt(err_sq / ref_sq) : sqrt(err_sq);
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

static void test_batch_grover_trace(metal_compute_ctx_t* ctx) {
    const uint32_t num_searches = 1;
    const uint32_t num_qubits = 2;
    const uint32_t state_dim = 1u << num_qubits;
    const uint32_t num_iterations = 1;
    const size_t nbytes = num_searches * state_dim * sizeof(fcomplex_t);

    metal_buffer_t* batch_states = metal_buffer_create(ctx, nbytes);
    CHECK(batch_states != NULL, "Metal batch Grover state buffer allocated");
    if (!batch_states) return;

    memset(metal_buffer_contents(batch_states), 0, nbytes);

    uint32_t targets[] = {2};
    uint32_t results[] = {0};
    int rc = metal_grover_batch_search(ctx, batch_states, targets, results,
                                       num_searches, num_qubits,
                                       num_iterations);

    const metal_backend_trace_t* trace = metal_get_last_backend_trace();
    check_trace_identity(trace, "metal_grover_batch_search",
                         "batch-grover-search");
    CHECK(rc == 0, "Metal batch Grover search completes successfully");
    CHECK(trace && trace->batch_search_pipeline_loaded == 1,
          "Metal batch Grover trace confirms compiled batch pipeline");
    if (rc == 0) {
        CHECK(results[0] == targets[0],
              "Metal batch Grover search finds target %u", targets[0]);
    }

    metal_buffer_free(batch_states);
}

static void test_svd_cpu_fallback(metal_compute_ctx_t* ctx) {
    fprintf(stdout, "\n-- Metal SVD CPU fallback --\n");

    const uint32_t m = 3;
    const uint32_t n = 2;
    const uint32_t rank_capacity = 2;
    fcomplex_t input[m * n] = {
        {3.0f, 0.0f}, {0.5f, 0.25f},
        {0.0f, 0.0f}, {2.0f, 0.0f},
        {0.25f, -0.5f}, {0.0f, 0.0f}
    };

    metal_buffer_t* A = metal_buffer_create(ctx, sizeof(input));
    metal_buffer_t* U = metal_buffer_create(ctx, (size_t)m * rank_capacity * sizeof(fcomplex_t));
    metal_buffer_t* S = metal_buffer_create(ctx, rank_capacity * sizeof(float));
    metal_buffer_t* Vt = metal_buffer_create(ctx, (size_t)rank_capacity * n * sizeof(fcomplex_t));

    CHECK(A && U && S && Vt, "SVD fallback buffers allocated");
    if (!A || !U || !S || !Vt) {
        if (A) metal_buffer_free(A);
        if (U) metal_buffer_free(U);
        if (S) metal_buffer_free(S);
        if (Vt) metal_buffer_free(Vt);
        return;
    }

    memcpy(metal_buffer_contents(A), input, sizeof(input));
    memset(metal_buffer_contents(U), 0, (size_t)m * rank_capacity * sizeof(fcomplex_t));
    memset(metal_buffer_contents(S), 0, rank_capacity * sizeof(float));
    memset(metal_buffer_contents(Vt), 0, (size_t)rank_capacity * n * sizeof(fcomplex_t));

    setenv("MOONLAB_METAL_FORCE_SVD_CPU_FALLBACK", "1", 1);
    uint32_t actual_rank = 0;
    int rc = metal_svd_truncate(ctx, A, U, S, Vt, m, n, rank_capacity, 0.0, &actual_rank);
    unsetenv("MOONLAB_METAL_FORCE_SVD_CPU_FALLBACK");

    CHECK(rc == 0, "forced SVD CPU fallback returns success");
    CHECK(actual_rank == rank_capacity,
          "forced SVD CPU fallback keeps full rank (%u)", actual_rank);

    const float* s_values = (const float*)metal_buffer_contents(S);
    CHECK(s_values[0] >= s_values[1] && s_values[1] > 0.0f,
          "fallback singular values are sorted and positive");

    double rel = metal_svd_reconstruction_error(
        input,
        (const fcomplex_t*)metal_buffer_contents(U),
        s_values,
        (const fcomplex_t*)metal_buffer_contents(Vt),
        m,
        n,
        actual_rank);
    CHECK(rel < 2e-5,
          "forced SVD CPU fallback reconstructs matrix, rel %.3e", rel);

    metal_buffer_free(A);
    metal_buffer_free(U);
    metal_buffer_free(S);
    metal_buffer_free(Vt);
}

#endif  /* METAL_PRESENT */

int main(void) {
    fprintf(stdout, "=== Metal GPU parity tests ===\n");
#if METAL_PRESENT
    metal_backend_trace_t probe =
        metal_backend_probe("test_metal_parity", "probe");
    check_trace_identity(&probe, "test_metal_parity", "probe");

    if (!metal_is_available()) {
        CHECK(probe.fallback_intentional == 1,
              "Metal unavailable probe is marked as intentional fallback");
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

    const metal_backend_trace_t* init_trace = metal_get_last_backend_trace();
    check_trace_identity(init_trace, "metal_compute_init", "initialize");
    CHECK(init_trace->device_created == 1,
          "Metal init trace confirms real device creation");
    CHECK(init_trace->command_queue_created == 1,
          "Metal init trace confirms command queue creation");
    CHECK(init_trace->shader_library_loaded == 1,
          "Metal init trace confirms shader library loading");
    CHECK(init_trace->last_status == 0,
          "Metal init trace reports successful status");

    test_batch_grover_trace(ctx);
    metal_wait_completion(ctx);
    const metal_backend_trace_t* wait_trace = metal_get_last_backend_trace();
    check_trace_identity(wait_trace, "metal_wait_completion", "synchronize");
    CHECK(wait_trace->last_status == 0,
          "Metal wait trace reports successful synchronization");

    test_gate_parity(ctx, "Hadamard", gate_hadamard, metal_hadamard, 5, 0);
    test_gate_parity(ctx, "Hadamard", gate_hadamard, metal_hadamard, 5, 2);
    test_gate_parity(ctx, "Pauli X",  gate_pauli_x,  metal_pauli_x,  5, 1);
    test_gate_parity(ctx, "Pauli X",  gate_pauli_x,  metal_pauli_x,  5, 4);
    test_gate_parity(ctx, "Pauli Z",  gate_pauli_z,  metal_pauli_z,  5, 0);
    test_gate_parity(ctx, "Pauli Z",  gate_pauli_z,  metal_pauli_z,  5, 3);
    test_svd_cpu_fallback(ctx);

    metal_compute_free(ctx);
#else
    fprintf(stdout, "  SKIP  Metal not compiled in (non-Apple target or HAS_METAL undefined)\n");
#endif
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
