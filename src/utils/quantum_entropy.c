/**
 * @file quantum_entropy.c
 * @brief Non-inline helpers for @c quantum_entropy_ctx_t lifetime
 *        management.
 *
 * The header ships the cheap path as @c static @c inline, which is
 * fine for in-tree callers but invisible to dlsym / ctypes-based
 * bindings.  This file provides a pair of plain extern symbols that
 * allocate and release a fully-wired hardware-backed context.
 *
 * @since 0.2.0
 */

#include "quantum_entropy.h"
#include "../applications/hardware_entropy.h"

#include <stdlib.h>

quantum_entropy_ctx_t *quantum_entropy_ctx_create_hw(void) {
    entropy_ctx_t *hw = (entropy_ctx_t *)calloc(1, sizeof(*hw));
    if (!hw) return NULL;
    if (entropy_init(hw) != ENTROPY_SUCCESS) {
        free(hw);
        return NULL;
    }

    quantum_entropy_ctx_t *ctx =
        (quantum_entropy_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        entropy_free(hw);
        free(hw);
        return NULL;
    }

    quantum_entropy_init(ctx,
                         (quantum_entropy_fn)entropy_get_bytes,
                         hw);
    return ctx;
}

void quantum_entropy_ctx_destroy(quantum_entropy_ctx_t *ctx) {
    if (!ctx) return;
    entropy_ctx_t *hw = (entropy_ctx_t *)ctx->user_data;
    if (hw) {
        entropy_free(hw);
        free(hw);
    }
    free(ctx);
}
