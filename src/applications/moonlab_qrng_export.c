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
 * A process-lifetime v3 QRNG context is lazily constructed in
 * BELL_VERIFIED mode and freed at atexit.  The wrapper serialises the
 * stateful v3 context, absorbs fresh health-tested entropy plus the v3
 * stream into a domain-separated SHAKE256 conditioner, and releases
 * output only after the current simulated Bell epoch has passed.
 *
 * @since v0.5.0 (file split from moonlab_qrng_export.c).
 */

#include "moonlab_export.h"
#include "qrng.h"
#include "../crypto/sha3/sha3.h"
#include "../utils/secure_memory.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

/* ================================================================== */
/*  v3 QRNG bytes -- lazy process-lifetime context.                    */
/* ================================================================== */

static qrng_v3_ctx_t* g_qrng_v3 = NULL;
static pthread_mutex_t g_qrng_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_atexit_once = PTHREAD_ONCE_INIT;
static uint64_t g_conditioned_requests = 0;

#define QRNG_CONDITIONER_CHUNK 4096u
#define QRNG_CONDITIONER_SEED_BYTES 64u

static const uint8_t g_conditioner_domain[] =
    "Moonlab conditioned hybrid QRNG v1 / SHAKE256 / ABI 0.5";

static void encode_u64_le(uint8_t out[8], uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[i] = (uint8_t)(value >> (8u * i));
    }
}

static void qrng_v3_reset_locked(void) {
    if (g_qrng_v3) {
        qrng_v3_free(g_qrng_v3);
        g_qrng_v3 = NULL;
    }
}

static void qrng_v3_atexit(void) {
    pthread_mutex_lock(&g_qrng_mutex);
    qrng_v3_reset_locked();
    pthread_mutex_unlock(&g_qrng_mutex);
}

static void register_atexit_impl(void) {
    atexit(qrng_v3_atexit);
}

static int qrng_v3_init_locked(void) {
    if (g_qrng_v3) return 0;

    qrng_v3_config_t config;
    qrng_v3_get_default_config(&config);
    config.mode = QRNG_V3_MODE_BELL_VERIFIED;
    config.enable_bell_monitoring = 1;
    config.bell_test_interval = 1024u * 1024u;

    qrng_v3_error_t e = qrng_v3_init_with_config(&g_qrng_v3, &config);
    if (e != QRNG_V3_SUCCESS || !g_qrng_v3) {
        qrng_v3_reset_locked();
        return -2;
    }
    pthread_once(&g_atexit_once, register_atexit_impl);
    return 0;
}

/**
 * @brief Fill buffer with conditioned hybrid random bytes.
 *
 * @param buf  Output buffer (must be non-NULL when size > 0)
 * @param size Number of bytes to generate
 * @return 0 on success, negative on error.
 */
int moonlab_qrng_bytes(uint8_t* buf, size_t size) {
    if (size == 0) return 0;
    if (!buf) return -1;

    pthread_mutex_lock(&g_qrng_mutex);
    int init_rc = qrng_v3_init_locked();
    if (init_rc != 0) {
        pthread_mutex_unlock(&g_qrng_mutex);
        return init_rc;
    }

    if (g_conditioned_requests == UINT64_MAX) {
        secure_memzero(buf, size);
        pthread_mutex_unlock(&g_qrng_mutex);
        return -5;
    }

    uint8_t conditioner_seed[QRNG_CONDITIONER_SEED_BYTES];
    uint8_t raw[QRNG_CONDITIONER_CHUNK];
    uint8_t encoded[16];
    sha3_ctx_t conditioner;

    if (entropy_pool_get_bytes(g_qrng_v3->entropy_pool,
                               conditioner_seed,
                               sizeof(conditioner_seed)) != 0) {
        secure_memzero(buf, size);
        qrng_v3_reset_locked();
        pthread_mutex_unlock(&g_qrng_mutex);
        return -4;
    }

    encode_u64_le(encoded, g_conditioned_requests);
    encode_u64_le(encoded + 8, (uint64_t)size);
    shake256_init(&conditioner);
    sha3_update(&conditioner, g_conditioner_domain,
                sizeof(g_conditioner_domain) - 1);
    sha3_update(&conditioner, encoded, sizeof(encoded));
    sha3_update(&conditioner, conditioner_seed, sizeof(conditioner_seed));

    size_t remaining = size;
    while (remaining > 0) {
        size_t take = remaining < sizeof(raw) ? remaining : sizeof(raw);
        qrng_v3_error_t e = qrng_v3_bytes(g_qrng_v3, raw, take);
        if (e != QRNG_V3_SUCCESS) {
            secure_memzero(buf, size);
            secure_memzero(raw, sizeof(raw));
            secure_memzero(conditioner_seed, sizeof(conditioner_seed));
            secure_memzero(&conditioner, sizeof(conditioner));
            qrng_v3_reset_locked();
            pthread_mutex_unlock(&g_qrng_mutex);
            return -3;
        }
        sha3_update(&conditioner, raw, take);
        remaining -= take;
    }

    shake_squeeze(&conditioner, buf, size);
    g_conditioned_requests++;

    secure_memzero(raw, sizeof(raw));
    secure_memzero(conditioner_seed, sizeof(conditioner_seed));
    secure_memzero(encoded, sizeof(encoded));
    secure_memzero(&conditioner, sizeof(conditioner));
    pthread_mutex_unlock(&g_qrng_mutex);
    return 0;
}

int moonlab_qrng_get_status(moonlab_qrng_status_t* status) {
    if (!status) return -1;

    pthread_mutex_lock(&g_qrng_mutex);
    int init_rc = qrng_v3_init_locked();
    if (init_rc != 0) {
        pthread_mutex_unlock(&g_qrng_mutex);
        return init_rc;
    }

    qrng_v3_stats_t stats;
    if (qrng_v3_get_stats(g_qrng_v3, &stats) != QRNG_V3_SUCCESS) {
        pthread_mutex_unlock(&g_qrng_mutex);
        return -2;
    }

    uint64_t capabilities =
        MOONLAB_QRNG_CAP_HARDWARE_OS_ENTROPY |
        MOONLAB_QRNG_CAP_CONTINUOUS_HEALTH_TESTS |
        MOONLAB_QRNG_CAP_SHAKE256_CONDITIONED |
        MOONLAB_QRNG_CAP_BELL_SIMULATION_GATED |
        MOONLAB_QRNG_CAP_THREAD_SAFE;
    if (stats.bell_tests_performed > 0 &&
        stats.bell_tests_passed == stats.bell_tests_performed) {
        capabilities |= MOONLAB_QRNG_CAP_BELL_EPOCH_CERTIFIED;
    }

    moonlab_qrng_status_t value = {
        .struct_size = (uint32_t)sizeof(moonlab_qrng_status_t),
        .api_version = 1,
        .capabilities = capabilities,
        .conditioned_requests = g_conditioned_requests,
        .raw_bytes_generated = stats.bytes_generated,
        .bell_tests_performed = stats.bell_tests_performed,
        .bell_tests_passed = stats.bell_tests_passed,
        .average_chsh = stats.average_chsh,
        .minimum_chsh = stats.min_chsh
    };
    *status = value;
    pthread_mutex_unlock(&g_qrng_mutex);
    return 0;
}
