/**
 * @file test_gpu_backend_discovery.c
 * @brief Exercise the GPU-backend discovery path without requiring
 *        an actual GPU.
 *
 * The CUDA / OpenCL / Vulkan / cuQuantum backends are gated at build
 * time by QSIM_ENABLE_* options that are OFF by default, and at
 * runtime by whether a dispatchable device is present.  On a typical
 * CI machine there is no GPU available at all, which is exactly the
 * path we need to exercise: `gpu_compute_init` must degrade cleanly
 * to "CPU only" rather than abort or segfault.
 *
 * This test is deliberately tolerant: it verifies the discovery API
 * returns without crashing, then checks that whatever backend comes
 * back is one the switch statement in `gpu_backend_name` recognises.
 */

#include "../../src/optimization/gpu/gpu_backend.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void expect_close(const char* label, double got, double expected, int* failures) {
    if (fabs(got - expected) > 1e-12) {
        fprintf(stderr, "FAIL %s got %.17g, expected %.17g\n", label, got, expected);
        (*failures)++;
    }
}

int main(void) {
    int failures = 0;

    /* Feature probe: must not crash, must return 0 or 1. */
    int avail = gpu_is_available();
    if (avail != 0 && avail != 1) {
        fprintf(stderr, "FAIL gpu_is_available returned %d, expected 0/1\n",
                avail);
        failures++;
    }
    printf("gpu_is_available = %d\n", avail);

    /* Auto-discover path: gpu_compute_init(AUTO) should either return
     * a usable context OR return NULL (no GPU configured).  Both are
     * fine; a NULL is how CPU-only hosts behave. */
    gpu_context_t* ctx = gpu_compute_init(GPU_BACKEND_AUTO);
    if (ctx) {
        gpu_backend_type_t t = gpu_get_backend_type(ctx);
        const char* nm = gpu_backend_name(t);
        if (!nm) {
            fprintf(stderr, "FAIL gpu_backend_name(%d) returned NULL\n", (int)t);
            failures++;
        }
        printf("gpu_compute_init succeeded: backend=%s (type=%d)\n",
               nm ? nm : "(null)", (int)t);
        gpu_compute_free(ctx);
    } else {
        printf("gpu_compute_init returned NULL (no GPU available) -- OK\n");
    }

    /* Explicit "unsupported" backend requests must not crash. */
    gpu_context_t* none = gpu_compute_init(GPU_BACKEND_NONE);
    if (none) {
        /* Legal: some builds treat NONE as a no-op context. */
        complex_t amplitudes[4] = {
            1.0 + 0.0 * I,
            0.5 + 0.5 * I,
            -0.25 + 0.25 * I,
            0.0 + 0.0 * I,
        };
        const double expected_norm2 = 1.625;

        gpu_buffer_t* amp_buffer = gpu_buffer_create_from_data(
            none, amplitudes, sizeof(amplitudes));
        gpu_buffer_t* probability_buffer = gpu_buffer_create(
            none, 4 * sizeof(double));
        if (!amp_buffer || !probability_buffer) {
            fprintf(stderr, "FAIL GPU_BACKEND_NONE host buffers were not created\n");
            failures++;
        } else {
            double norm2 = 0.0;
            gpu_error_t rc = gpu_sum_squared_magnitudes(
                none, amp_buffer, 4, &norm2);
            if (rc != GPU_SUCCESS) {
                fprintf(stderr, "FAIL gpu_sum_squared_magnitudes(NONE) rc=%d\n", (int)rc);
                failures++;
            } else {
                expect_close("none norm2", norm2, expected_norm2, &failures);
            }

            rc = gpu_compute_probabilities(none, amp_buffer, probability_buffer, 4);
            if (rc != GPU_SUCCESS) {
                fprintf(stderr, "FAIL gpu_compute_probabilities(NONE) rc=%d\n", (int)rc);
                failures++;
            } else {
                double probabilities[4] = {0};
                rc = gpu_buffer_read(probability_buffer, probabilities,
                                     sizeof(probabilities), 0);
                if (rc != GPU_SUCCESS) {
                    fprintf(stderr, "FAIL gpu_buffer_read probabilities rc=%d\n", (int)rc);
                    failures++;
                } else {
                    expect_close("prob[0]", probabilities[0], 1.0, &failures);
                    expect_close("prob[1]", probabilities[1], 0.5, &failures);
                    expect_close("prob[2]", probabilities[2], 0.125, &failures);
                    expect_close("prob[3]", probabilities[3], 0.0, &failures);
                }
            }

            rc = gpu_normalize(none, amp_buffer, sqrt(expected_norm2), 4);
            if (rc != GPU_SUCCESS) {
                fprintf(stderr, "FAIL gpu_normalize(NONE) rc=%d\n", (int)rc);
                failures++;
            } else {
                norm2 = 0.0;
                rc = gpu_sum_squared_magnitudes(none, amp_buffer, 4, &norm2);
                if (rc != GPU_SUCCESS) {
                    fprintf(stderr, "FAIL normalized sum rc=%d\n", (int)rc);
                    failures++;
                } else {
                    expect_close("normalized norm2", norm2, 1.0, &failures);
                }
            }

            rc = gpu_hadamard(none, amp_buffer, 0, 4);
            if (rc != GPU_ERROR_NOT_SUPPORTED) {
                fprintf(stderr, "FAIL gpu_hadamard(NONE) rc=%d, expected %d\n",
                        (int)rc, (int)GPU_ERROR_NOT_SUPPORTED);
                failures++;
            }
        }

        gpu_buffer_free(probability_buffer);
        gpu_buffer_free(amp_buffer);
        gpu_compute_free(none);
    }
    printf("gpu_compute_init(NONE) did not crash -- OK\n");

    /* Every enum value we advertise should have a human-readable name. */
    for (int t = 0; t < 10; t++) {
        const char* nm = gpu_backend_name((gpu_backend_type_t)t);
        if (!nm) {
            fprintf(stderr, "FAIL backend type %d has NULL name\n", t);
            failures++;
        }
    }

    printf("\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
