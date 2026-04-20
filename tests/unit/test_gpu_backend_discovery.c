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

#include <stdio.h>
#include <string.h>

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
