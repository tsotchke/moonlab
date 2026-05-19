/**
 * @file moonlab_qrng_export.c
 * @brief Quantum-RNG half of the v0.2.x stable C export surface.
 *
 * Implements only @c moonlab_qrng_bytes from @c moonlab_export.h;
 * the rest of the ABI surface (abi_version, qwz_chern, dmrg_*,
 * ca_mps_var_d_*, ca_mps_gauge_warmstart, z2_lgt_1d_*,
 * status_string) lives in the sibling file
 * @c moonlab_export_lean.c which has no qrng / hardware_entropy
 * dependency and can therefore be included in the emscripten WASM
 * build without dragging in the heavy native-only RNG stack.
 *
 * A process-lifetime v3 QRNG context is lazily constructed on the
 * first call and freed at atexit.  The wrapper is thread-safe:
 * concurrent callers are serialised via an internal mutex and the
 * underlying v3 engine maintains its own locking for byte
 * generation.
 *
 * @since v0.5.0 (file split from moonlab_qrng_export.c).
 */

#include "moonlab_export.h"
#include "qrng.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

/* ================================================================== */
/*  v3 QRNG bytes -- lazy process-lifetime context.                    */
/* ================================================================== */

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
