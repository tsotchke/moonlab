/**
 * @file test_qrng_thread_safety.c
 * @brief Concurrent regression test for the stable QRNG byte API.
 *
 * Run this target under ThreadSanitizer to verify that the process-lifetime
 * v3 context is never entered by more than one caller at a time.
 */

#include "../../src/applications/moonlab_export.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define THREAD_COUNT 8
#define DRAWS_PER_THREAD 8
#define BYTES_PER_DRAW 256

typedef struct {
    uint8_t output[BYTES_PER_DRAW];
    int rc;
    int nonzero;
} worker_result_t;

static void *draw_worker(void *opaque) {
    worker_result_t *result = (worker_result_t *)opaque;
    result->rc = 0;
    result->nonzero = 0;

    for (int draw = 0; draw < DRAWS_PER_THREAD; ++draw) {
        result->rc = moonlab_qrng_bytes(result->output, sizeof(result->output));
        if (result->rc != 0) return NULL;
        for (size_t i = 0; i < sizeof(result->output); ++i) {
            if (result->output[i] != 0) result->nonzero = 1;
        }
    }
    return NULL;
}

int main(void) {
    pthread_t threads[THREAD_COUNT];
    worker_result_t results[THREAD_COUNT];
    memset(results, 0, sizeof(results));

    for (int i = 0; i < THREAD_COUNT; ++i) {
        if (pthread_create(&threads[i], NULL, draw_worker, &results[i]) != 0) {
            fprintf(stderr, "pthread_create failed for worker %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < THREAD_COUNT; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "pthread_join failed for worker %d\n", i);
            return 1;
        }
        if (results[i].rc != 0 || !results[i].nonzero) {
            fprintf(stderr, "worker %d failed: rc=%d nonzero=%d\n",
                    i, results[i].rc, results[i].nonzero);
            return 1;
        }
    }

    for (int i = 1; i < THREAD_COUNT; ++i) {
        if (memcmp(results[0].output, results[i].output, BYTES_PER_DRAW) == 0) {
            fprintf(stderr, "workers 0 and %d returned identical final draws\n", i);
            return 1;
        }
    }

    puts("concurrent moonlab_qrng_bytes draws passed");
    return 0;
}
