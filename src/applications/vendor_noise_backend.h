/**
 * @file    vendor_noise_backend.h
 * @brief   Stochastic-Pauli vendor-noise emulator backends (since v1.1).
 *
 * Pre-flight emulators for the three major gate-model QPU vendor
 * families.  Each backend takes the ideal moonlab_qgtl_execute output
 * and post-processes every shot with stochastic Pauli + readout
 * noise drawn from a vendor-typical calibration profile.  This is
 * the same noise model used by Stim and by most published QPU
 * pre-flight benchmark suites: it is accurate when the dominant
 * noise is Pauli-stabilised (T1 / T2 / depolarising / readout
 * error) and is a fast Monte-Carlo approximation that does not
 * carry the full density matrix.
 *
 * Three backends register themselves under the scheduler's
 * registry on @ref moonlab_register_vendor_noise_backends:
 *
 *   - "ibm-falcon"     IBM Falcon r5.11 heavy-hex typical calibration
 *   - "rigetti-aspen"  Rigetti Aspen-M-3 octagon-tile typical
 *   - "ionq-forte"     IonQ Forte all-to-all ion-trap typical
 *
 * D-Wave (quantum annealer) does not consume QGTL gate-model
 * circuits; its emulator lives elsewhere.
 *
 * The numbers are public-data typical values; for a specific
 * device's current calibration the caller should fetch the actual
 * coupling map / error table from the vendor and use the lower-
 * level @ref moonlab_register_backend with a custom profile
 * struct.
 *
 * @since v1.1.0
 */

#ifndef MOONLAB_VENDOR_NOISE_BACKEND_H
#define MOONLAB_VENDOR_NOISE_BACKEND_H

#include <stdint.h>

#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Stochastic Pauli noise profile.
 *
 * Every probability is per-application; values stack
 * multiplicatively over circuit depth.  Range [0, 1].
 */
typedef struct {
    /** Per single-qubit gate depolarising-error probability. */
    double p_gate_1q;
    /** Per two-qubit gate depolarising-error probability. */
    double p_gate_2q;
    /** Per qubit at measurement time bit-flip probability. */
    double p_readout;
    /** Optional human-readable label, e.g. "IBM Falcon r5.11 typical". */
    const char *description;
} moonlab_vendor_noise_profile_t;

/**
 * @brief Register the three pre-baked vendor profiles
 *        (ibm-falcon, rigetti-aspen, ionq-forte) into the
 *        scheduler's backend registry.
 *
 *        Idempotent: safe to call multiple times.  Each call
 *        re-installs the profile (so if the caller overrode one
 *        and wants to revert to the defaults, calling this again
 *        works).
 */
MOONLAB_API int moonlab_register_vendor_noise_backends(void);

/**
 * @brief Register a single vendor-noise backend under @p name with
 *        the caller's custom profile.  Lets the user pin a current
 *        IBM device's actual calibration table rather than the
 *        typical baked-in defaults.
 */
MOONLAB_API int moonlab_register_vendor_noise_backend_with_profile(
        const char *name,
        const moonlab_vendor_noise_profile_t *profile);

/**
 * @brief Read back a baked-in profile.  Returns NULL if @p name is
 *        not one of the three pre-baked profiles.
 */
MOONLAB_API const moonlab_vendor_noise_profile_t *
moonlab_lookup_vendor_noise_profile(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_VENDOR_NOISE_BACKEND_H */
