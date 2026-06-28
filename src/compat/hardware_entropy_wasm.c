/**
 * @file hardware_entropy_wasm.c
 * @brief WASM-only implementation of the `entropy_*` surface declared
 *        in `src/applications/hardware_entropy.h`.
 *
 * The native `hardware_entropy.c` impl pulls in RDRAND / RDSEED /
 * `/dev/random` / jitter-timing -- none of which make sense or
 * compile cleanly under emscripten / wasm32.  The WASM build
 * therefore swaps in this file instead, which delegates to
 * @c getentropy(3) (a small libc syscall emscripten emulates by
 * calling out to the host's `crypto.getRandomValues()` via JS).
 *
 * Functions implemented here:
 *  - @c entropy_init    : zero the ctx and mark `GETRANDOM` as the
 *                         preferred source.
 *  - @c entropy_free    : nothing to free; no fd's were opened.
 *  - @c entropy_get_bytes : call `getentropy()` in 256-byte chunks
 *                          (the POSIX cap).
 *
 * The rest of the `hardware_entropy.h` surface (capability probe,
 * source enumeration, etc.) is intentionally absent -- the v0.2.x
 * stable ABI only requires the three functions above to support the
 * `quantum_entropy_ctx_create_hw` shim that VQE / QAOA / Bell /
 * Grover call through.
 *
 * @since v0.5.4
 */

#ifdef __EMSCRIPTEN__

#include "../applications/hardware_entropy.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>      /* getentropy on emscripten */
#include <errno.h>

entropy_error_t entropy_init(entropy_ctx_t *ctx) {
    if (!ctx) return ENTROPY_ERROR_INVALID_PARAM;
    memset(ctx, 0, sizeof(*ctx));
    ctx->caps.has_getrandom    = 1;  /* WASM emscripten always provides getentropy. */
    ctx->caps.preferred_source = ENTROPY_SOURCE_GETRANDOM;
    ctx->dev_random_fd         = -1;
    ctx->dev_urandom_fd        = -1;
    ctx->last_source           = ENTROPY_SOURCE_GETRANDOM;
    return ENTROPY_SUCCESS;
}

void entropy_free(entropy_ctx_t *ctx) {
    /* No fd's, no allocations -- this is a no-op on WASM. */
    (void)ctx;
}

entropy_error_t entropy_get_bytes(entropy_ctx_t *ctx,
                                    uint8_t *buffer,
                                    size_t size) {
    if (!ctx || !buffer) return ENTROPY_ERROR_INVALID_PARAM;
    if (size == 0) return ENTROPY_SUCCESS;

    /* getentropy() takes at most 256 bytes per call per POSIX. */
    size_t off = 0;
    while (off < size) {
        size_t chunk = (size - off) > 256u ? 256u : (size - off);
        if (getentropy(buffer + off, chunk) != 0) {
            /* Defensive: on the off-chance the emscripten polyfill
             * returns -1 (host with no crypto.getRandomValues), bail
             * loudly rather than silently feed predictable bytes. */
            return ENTROPY_ERROR_SYSCALL;
        }
        off += chunk;
        ctx->total_bytes += chunk;
    }
    ctx->last_source = ENTROPY_SOURCE_GETRANDOM;
    return ENTROPY_SUCCESS;
}

#else  /* !__EMSCRIPTEN__ */

/* Native builds use the full hardware_entropy.c.  This file is
 * still compiled on those platforms (the WASM CMake includes it
 * unconditionally for the WASM target only); the empty-translation-
 * unit warning is silenced by a single static declaration. */
typedef int moonlab_hardware_entropy_wasm_empty_tu;

#endif /* __EMSCRIPTEN__ */
