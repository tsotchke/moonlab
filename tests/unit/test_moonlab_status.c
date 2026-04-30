/**
 * @file test_moonlab_status.c
 * @brief Unit test for the centralised moonlab_status_t registry.
 *
 * Pins:
 *   - Canonical zero / -1 / -2 / -3 / -4 codes stringify to the
 *     expected names regardless of which module is queried.
 *   - Module-specific extensions (CA_PEPS_ERR_NOT_IMPLEMENTED at
 *     -100) stringify to their per-module name.
 *   - moonlab_status_ok zero <-> 1, every error code <-> 0.
 *   - The fallback path returns a non-NULL string for an unknown
 *     code so logging shims never have to handle NULL.
 *
 * Closes audit task #73.
 */

#include "../../src/utils/moonlab_status.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    fprintf(stdout, "=== moonlab_status_t registry test ===\n");

    /* Canonical codes -- shared across modules. */
    CHECK(strcmp(moonlab_status_to_string(MOONLAB_MODULE_CA_MPS,
                                            MOONLAB_STATUS_SUCCESS),
                  "SUCCESS") == 0,
          "SUCCESS should stringify to \"SUCCESS\"");
    CHECK(strcmp(moonlab_status_to_string(MOONLAB_MODULE_CA_MPS,
                                            MOONLAB_STATUS_ERR_INVALID),
                  "ERR_INVALID") == 0,
          "INVALID should stringify to \"ERR_INVALID\"");
    CHECK(strcmp(moonlab_status_to_string(MOONLAB_MODULE_TN_STATE,
                                            MOONLAB_STATUS_ERR_OOM),
                  "ERR_OOM") == 0,
          "OOM should stringify to \"ERR_OOM\" regardless of module");

    /* Module-specific extension. */
    CHECK(strcmp(moonlab_status_to_string(MOONLAB_MODULE_CA_PEPS, -100),
                  "CA_PEPS_ERR_NOT_IMPLEMENTED") == 0,
          "CA-PEPS -100 should stringify to CA_PEPS_ERR_NOT_IMPLEMENTED");

    /* Convenience predicate. */
    CHECK(moonlab_status_ok(MOONLAB_STATUS_SUCCESS) == 1,
          "ok(SUCCESS) should be 1");
    CHECK(moonlab_status_ok(MOONLAB_STATUS_ERR_INVALID) == 0,
          "ok(INVALID) should be 0");
    CHECK(moonlab_status_ok(MOONLAB_STATUS_ERR_BACKEND) == 0,
          "ok(BACKEND) should be 0");

    /* Fallback for an unknown code -- must be non-NULL so loggers
     * can always print something.  We don't pin the exact string
     * (it includes a numeric code) but we pin that it's non-NULL
     * and contains the module prefix. */
    const char* fallback =
        moonlab_status_to_string(MOONLAB_MODULE_CLIFFORD, -42);
    CHECK(fallback != NULL, "fallback must not return NULL");
    CHECK(strstr(fallback, "CLIFFORD") != NULL,
          "fallback should mention the module prefix CLIFFORD: got \"%s\"",
          fallback);

    /* Aliasing: var-D and stab-warmstart share CA_MPS's enum
     * conceptually.  All three modules should produce the same
     * stringification for the canonical codes. */
    const char* a = moonlab_status_to_string(MOONLAB_MODULE_CA_MPS,
                                                MOONLAB_STATUS_ERR_INVALID);
    const char* b = moonlab_status_to_string(MOONLAB_MODULE_CA_MPS_VAR_D,
                                                MOONLAB_STATUS_ERR_INVALID);
    const char* c =
        moonlab_status_to_string(MOONLAB_MODULE_CA_MPS_STAB_WARMSTART,
                                   MOONLAB_STATUS_ERR_INVALID);
    CHECK(strcmp(a, b) == 0 && strcmp(b, c) == 0,
          "var-D and stab-warmstart aliases should match CA_MPS for canonical codes");

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
