/**
 * @file    fuzz_target_config_parse.c
 * @brief   Surface: qsim configuration string / JSON parsing.
 *
 * Drives the untrusted-string entry points in `src/utils/config.c`:
 *
 *   - `qsim_config_from_json`   -- the `strstr` + `sscanf` "simplified
 *                                  JSON" loader used by `qsim_config_load`
 *                                  for on-disk config files.
 *   - `qsim_backend_from_string`,
 *     `qsim_simd_from_string`,
 *     `qsim_log_level_from_string` -- the `strcasecmp` enum decoders fed
 *                                  from environment variables and the
 *                                  JSON values above.
 *
 * These consume attacker-influenced config (env, file), so the contract
 * is: any NUL-terminated string yields a valid config or NULL, never an
 * out-of-bounds read across the `%63[^\"]` / `strstr` scans, and never a
 * leak (the returned config is destroyed here).
 */

#include "fuzz_common.h"

#include "utils/config.h"

#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* All of these take C strings; make a NUL-terminated copy. */
    char *s = (char *)malloc(size + 1);
    if (!s) return 0;
    if (size) memcpy(s, data, size);
    s[size] = '\0';

    /* Scalar enum decoders: fed directly from QSIM_* env vars. */
    (void)qsim_backend_from_string(s);
    (void)qsim_simd_from_string(s);
    (void)qsim_log_level_from_string(s);

    /* The JSON document loader used by qsim_config_load(). */
    qsim_config_t *cfg = qsim_config_from_json(s);
    if (cfg) {
        /* Round-trip through validation + JSON emit to touch the
         * fields the parser populated. */
        char err[128];
        (void)qsim_config_validate(cfg, err, sizeof(err));
        char *json = qsim_config_to_json(cfg);
        free(json);
        qsim_config_destroy(cfg);
    }

    free(s);
    return 0;
}
