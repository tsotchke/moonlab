/**
 * @file moonlab_qrng_export.c
 * @brief Stable C export for the Moonlab v3 quantum RNG + ABI version probe.
 *
 * Implements the functions declared in `moonlab_export.h` — the committed,
 * versioned ABI surface for downstream Moonlab consumers. Downstream
 * libraries (notably QGTL) locate these symbols via dlsym at runtime; the
 * contract in `moonlab_export.h` states that they will remain name- and
 * signature-stable across the entire 0.x release series.
 *
 * For `moonlab_qrng_bytes`, a process-lifetime v3 QRNG context is lazily
 * constructed on first call and freed at atexit. The wrapper is
 * thread-safe: concurrent callers are serialised via an internal mutex
 * and the underlying v3 engine maintains its own locking for byte
 * generation.
 */

#include "moonlab_export.h"
#include "qrng.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

static qrng_v3_ctx_t* g_qrng_v3 = NULL;
static pthread_mutex_t g_qrng_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_atexit_once = PTHREAD_ONCE_INIT;

static void qrng_v3_atexit(void) {
    pthread_mutex_lock(&g_qrng_mutex);
    if (g_qrng_v3) {
        qrng_v3_free(g_qrng_v3);
        g_qrng_v3 = NULL;
    }
    pthread_mutex_unlock(&g_qrng_mutex);
}

static void register_atexit_impl(void) {
    atexit(qrng_v3_atexit);
}

/**
 * @brief Fill buffer with quantum-random bytes from Moonlab's v3 engine.
 *
 * @param buf  Output buffer (must be non-NULL when size > 0)
 * @param size Number of bytes to generate
 * @return 0 on success, negative on error.
 */
int moonlab_qrng_bytes(uint8_t* buf, size_t size) {
    if (size == 0) return 0;
    if (!buf) return -1;

    pthread_mutex_lock(&g_qrng_mutex);
    if (!g_qrng_v3) {
        qrng_v3_error_t e = qrng_v3_init(&g_qrng_v3);
        if (e != QRNG_V3_SUCCESS || !g_qrng_v3) {
            pthread_mutex_unlock(&g_qrng_mutex);
            return -2;
        }
        pthread_once(&g_atexit_once, register_atexit_impl);
    }
    qrng_v3_ctx_t* ctx = g_qrng_v3;
    pthread_mutex_unlock(&g_qrng_mutex);

    qrng_v3_error_t e = qrng_v3_bytes(ctx, buf, size);
    return (e == QRNG_V3_SUCCESS) ? 0 : -3;
}

void moonlab_abi_version(int* major, int* minor, int* patch) {
    if (major) *major = MOONLAB_ABI_VERSION_MAJOR;
    if (minor) *minor = MOONLAB_ABI_VERSION_MINOR;
    if (patch) *patch = MOONLAB_ABI_VERSION_PATCH;
}
