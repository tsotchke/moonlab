/**
 * @file moonlab_status.c
 * @brief Centralised status-code stringifier.
 */

#include "moonlab_status.h"

#include <stdio.h>

/* Per-module extension code names.  Only those modules that have
 * non-canonical extension codes need an entry here; modules that
 * use only the standard -1, -2, -3, -4 set fall through to the
 * common stringifier. */

static const char* extension_string(moonlab_status_module_t module,
                                      moonlab_status_t status) {
    switch (module) {
        case MOONLAB_MODULE_CA_PEPS:
            switch (status) {
                case -100: return "CA_PEPS_ERR_NOT_IMPLEMENTED";
                default: return NULL;
            }
        default:
            return NULL;
    }
}

static const char* module_prefix(moonlab_status_module_t module) {
    switch (module) {
        case MOONLAB_MODULE_GENERIC:                return "MOONLAB_STATUS";
        case MOONLAB_MODULE_CA_MPS:                 return "CA_MPS";
        case MOONLAB_MODULE_CA_MPS_VAR_D:           return "CA_MPS";  /* shares enum */
        case MOONLAB_MODULE_CA_MPS_STAB_WARMSTART:  return "CA_MPS";  /* shares enum */
        case MOONLAB_MODULE_CA_PEPS:                return "CA_PEPS";
        case MOONLAB_MODULE_TN_STATE:               return "TN_STATE";
        case MOONLAB_MODULE_TN_GATE:                return "TN_GATE";
        case MOONLAB_MODULE_TN_MEASURE:             return "TN_MEASURE";
        case MOONLAB_MODULE_TENSOR:                 return "TENSOR";
        case MOONLAB_MODULE_CONTRACT:               return "CONTRACT";
        case MOONLAB_MODULE_SVD_COMPRESS:           return "SVD_COMPRESS";
        case MOONLAB_MODULE_CLIFFORD:               return "CLIFFORD";
        case MOONLAB_MODULE_PARTITION:              return "PARTITION";
        case MOONLAB_MODULE_DIST_GATE:              return "DIST_GATE";
        case MOONLAB_MODULE_MPI_BRIDGE:             return "MPI_BRIDGE";
    }
    return "UNKNOWN_MODULE";
}

const char* moonlab_status_to_string(moonlab_status_module_t module,
                                       moonlab_status_t status) {
    /* Canonical codes -- shared by every module that uses the
     * standard convention. */
    switch (status) {
        case MOONLAB_STATUS_SUCCESS:        return "SUCCESS";
        case MOONLAB_STATUS_ERR_INVALID:    return "ERR_INVALID";
        case MOONLAB_STATUS_ERR_QUBIT:      return "ERR_QUBIT";
        case MOONLAB_STATUS_ERR_OOM:        return "ERR_OOM";
        case MOONLAB_STATUS_ERR_BACKEND:    return "ERR_BACKEND";
    }

    const char* ext = extension_string(module, status);
    if (ext) return ext;

    /* Fallback: print module + numeric code into a thread-local
     * buffer so callers can log something without segfaulting.
     * Note this is racy under multithreaded use; callers that need
     * stable strings under threading should map status to their
     * module's enum constants directly. */
    static __thread char buf[80];
    (void)snprintf(buf, sizeof buf, "<%s status %d>",
                   module_prefix(module), status);
    return buf;
}
