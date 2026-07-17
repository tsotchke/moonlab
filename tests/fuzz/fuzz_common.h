/**
 * @file    fuzz_common.h
 * @brief   Shared declarations for the Moonlab coverage-guided fuzz lane.
 *
 * Every fuzz target in this directory is a single translation unit that
 * defines the standard libFuzzer entry point
 *
 *     int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
 *
 * The same TU builds three ways without modification:
 *
 *   1. libFuzzer  -- compiled with `-fsanitize=fuzzer,address,undefined`;
 *                    the fuzzer runtime supplies `main` and drives the
 *                    entry point with mutated inputs.
 *   2. AFL++      -- compiled with `afl-clang-fast` and linked against
 *                    `fuzz_driver.c`, whose shared shim uses the
 *                    `__AFL_FUZZ_TESTCASE_*` persistent-mode intrinsics.
 *   3. replay     -- compiled with `-fsanitize=address,undefined` (no
 *                    fuzzer) and linked against `fuzz_driver.c`, which
 *                    falls back to a plain stdin / argv file driver so a
 *                    seed corpus can be replayed deterministically under
 *                    the sanitizers in CI and ctest.
 *
 * No target may leak across a single `LLVMFuzzerTestOneInput` call:
 * LeakSanitizer runs at process exit and the persistent AFL loop reuses
 * one process for the whole campaign, so each invocation must free every
 * allocation and close every descriptor it opens.
 */

#ifndef MOONLAB_FUZZ_COMMON_H
#define MOONLAB_FUZZ_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The libFuzzer contract entry point every target implements. */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* ------------------------------------------------------------------
 * Small helpers shared by the targets for consuming fuzzer bytes.
 * Header-static so each TU gets its own copy with no link deps.
 * ------------------------------------------------------------------ */

/** Consume one byte from the front of the buffer, advancing the cursor.
 *  Returns 0 once the buffer is exhausted (deterministic tail). */
static inline uint8_t fuzz_u8(const uint8_t **p, const uint8_t *end)
{
    if (*p >= end) return 0;
    return *(*p)++;
}

/** Consume a little-endian uint32 (or as many bytes as remain). */
static inline uint32_t fuzz_u32(const uint8_t **p, const uint8_t *end)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)fuzz_u8(p, end) << (8 * i);
    return v;
}

/** Copy up to `cap` bytes of the remaining buffer into `dst`, zero-pad
 *  the rest.  Advances the cursor past what it consumed.  Returns the
 *  number of real (non-pad) bytes copied. */
static inline size_t fuzz_fill(const uint8_t **p, const uint8_t *end,
                               uint8_t *dst, size_t cap)
{
    size_t avail = (size_t)(end - *p);
    size_t n = avail < cap ? avail : cap;
    if (n) memcpy(dst, *p, n);
    if (n < cap) memset(dst + n, 0, cap - n);
    *p += n;
    return n;
}

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_FUZZ_COMMON_H */
