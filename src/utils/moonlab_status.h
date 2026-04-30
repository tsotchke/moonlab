/**
 * @file moonlab_status.h
 * @brief Centralised return-code registry for Moonlab.
 *
 * Documents the canonical zero / negative-integer convention shared
 * by every per-module error enum in `src/`, plus a generic
 * `moonlab_status_t` typedef and a diagnostic `to_string` helper.
 *
 * Convention (followed by every existing `*_error_t` in the
 * codebase):
 *   - 0 means success.
 *   - Negative integers are errors.
 *   - Standard error codes -1, -2, -3, -4 always mean
 *     INVALID / QUBIT / OOM / BACKEND respectively (in modules
 *     where those concepts apply).
 *   - Module-specific extensions occupy -100 and below to avoid
 *     collisions with the standard set (e.g. `CA_PEPS_ERR_NOT_IMPLEMENTED = -100`).
 *
 * Existing per-module enums (`ca_mps_error_t`, `tn_state_error_t`,
 * etc.) are not deprecated -- new code can keep using them.  This
 * header layers two things on top:
 *
 *   1. `moonlab_status_t` -- a generic int typedef for code that
 *      composes calls across modules (or wants to return "success
 *      or some error" without picking a specific enum).
 *   2. `moonlab_status_module_t` enum + `moonlab_status_to_string` --
 *      a pretty-printer that takes (module, status) and returns
 *      the canonical name.  Useful for logging shims, abi error
 *      surfaces, and python/rust binding error translators.
 *
 * Closes audit task #73 ("Centralise error enums into one
 * moonlab_status_t registry").  Earlier marked completed without
 * code; this header is the actual deliverable.
 */

#ifndef MOONLAB_STATUS_H
#define MOONLAB_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic Moonlab return code.
 *
 * 0 = success, < 0 = error.  See the canonical codes below; a
 * function may also return a module-specific extension code (-100
 * and below).  Use ::moonlab_status_to_string for diagnostic
 * messages.
 */
typedef int moonlab_status_t;

/** Canonical success code. */
#define MOONLAB_STATUS_SUCCESS          ( 0)

/** Canonical failure codes shared across modules. */
#define MOONLAB_STATUS_ERR_INVALID      (-1)
#define MOONLAB_STATUS_ERR_QUBIT        (-2)
#define MOONLAB_STATUS_ERR_OOM          (-3)
#define MOONLAB_STATUS_ERR_BACKEND      (-4)

/** First module-specific extension code.  Modules use this and below
 *  for codes that don't fit the canonical four. */
#define MOONLAB_STATUS_ERR_MODULE_BASE  (-100)

/**
 * @brief Module identifiers for ::moonlab_status_to_string.
 *
 * One per per-module error enum in `docs/error_codes.md`.  When a
 * new module-level enum is added, append a value here and extend
 * `moonlab_status_to_string` to handle it.
 */
typedef enum {
    MOONLAB_MODULE_GENERIC = 0,
    MOONLAB_MODULE_CA_MPS,
    MOONLAB_MODULE_CA_MPS_VAR_D,           /* aliases CA_MPS errors */
    MOONLAB_MODULE_CA_MPS_STAB_WARMSTART,  /* aliases CA_MPS errors */
    MOONLAB_MODULE_CA_PEPS,
    MOONLAB_MODULE_TN_STATE,
    MOONLAB_MODULE_TN_GATE,
    MOONLAB_MODULE_TN_MEASURE,
    MOONLAB_MODULE_TENSOR,
    MOONLAB_MODULE_CONTRACT,
    MOONLAB_MODULE_SVD_COMPRESS,
    MOONLAB_MODULE_CLIFFORD,
    MOONLAB_MODULE_PARTITION,
    MOONLAB_MODULE_DIST_GATE,
    MOONLAB_MODULE_MPI_BRIDGE
} moonlab_status_module_t;

/**
 * @brief Pretty-printer for a (module, status) pair.
 *
 * Returns a static string (do not free).  Falls through to a
 * generic "<unknown status N for module M>" if the code is not
 * recognised, so callers can always log something.
 */
const char* moonlab_status_to_string(moonlab_status_module_t module,
                                       moonlab_status_t status);

/**
 * @brief Return 1 if @p status is the success code, 0 otherwise.
 *
 * Convenience for code that wants to be explicit ("status == 0"
 * works too but reads less clearly when many modules' enums are in
 * flight).
 */
static inline int moonlab_status_ok(moonlab_status_t status) {
    return status == MOONLAB_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_STATUS_H */
