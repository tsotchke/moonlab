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
 * The canonical names are the "-emu" suffix to leave the bare
 * vendor names free for live-hardware backends that sibling
 * libraries (QGTL) register through @ref moonlab_register_backend
 * at runtime:
 *
 *   - "ibm-falcon-emu"    IBM Falcon r5.11 heavy-hex typical calibration
 *   - "rigetti-aspen-emu" Rigetti Aspen-M-3 octagon-tile typical
 *   - "ionq-forte-emu"    IonQ Forte all-to-all ion-trap typical
 *
 * The legacy bare names ("ibm-falcon", "rigetti-aspen", "ionq-forte")
 * still resolve via aliases for one release cycle.  New code should
 * use the explicit "-emu" suffix; "ionq-forte" without the suffix
 * is reserved for the live-hardware backend that QGTL registers.
 *
 * D-Wave (quantum annealer) does not consume QGTL gate-model
 * circuits; its emulator lives elsewhere.
 *
 * @section v1_1_registry Profile registry (since v1.1.0 line)
 *
 * Profiles can be registered at runtime via @ref
 * moonlab_register_vendor_noise_profile so private overlays (live
 * calibration scrapers, customer-specific tunings) can install
 * their own profiles under arbitrary names alongside the pre-baked
 * typical profiles.  Backends installed via @ref
 * moonlab_register_vendor_noise_backend_with_profile consume those
 * profiles by reference.
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
 *        (ibm-falcon-emu, rigetti-aspen-emu, ionq-forte-emu) plus
 *        their legacy-name aliases (ibm-falcon, rigetti-aspen,
 *        ionq-forte) into the scheduler's backend registry.
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
 *
 *        The profile struct is copied into the registry, so the
 *        caller's storage can be freed after the call returns.
 */
MOONLAB_API int moonlab_register_vendor_noise_backend_with_profile(
        const char *name,
        const moonlab_vendor_noise_profile_t *profile);

/* ------------------------------------------------------------------
 * Profile registry (since v1.1.0)
 *
 * Decouples profile data from backend registration.  Sibling
 * libraries register calibration data under arbitrary names; the
 * profile is then either consumed via an explicit backend
 * registration call or looked up directly by code that wants the
 * raw calibration values.
 *
 * Use cases:
 *   - moonlab-private live-calibration scraper installs
 *     "ibm-falcon-2026-05-20" under the same registry as the
 *     typical-data fallback.
 *   - QGTL installs customer-specific tuning profiles under names
 *     like "customer-acme-tuned-fintech".
 * ------------------------------------------------------------------ */

/**
 * @brief Register a noise profile under @p name without installing
 *        a backend.  Useful for live-calibration scrapers that want
 *        to keep the data and the backend separate.
 *
 *        Profile struct is copied; caller may free their storage.
 *        If @p name is already registered the new profile replaces
 *        the old.
 *
 * @return MOONLAB_SCHED_OK / MOONLAB_SCHED_BAD_ARG / MOONLAB_SCHED_OOM.
 */
MOONLAB_API int moonlab_register_vendor_noise_profile(
        const char *name,
        const moonlab_vendor_noise_profile_t *profile);

/**
 * @brief Unregister a profile by name.  Returns
 *        MOONLAB_SCHED_OK or MOONLAB_SCHED_BACKEND_NOT_FOUND
 *        (reused for "profile not found" — same status family).
 */
MOONLAB_API int moonlab_unregister_vendor_noise_profile(const char *name);

/**
 * @brief Look up a profile by name.  Returns the pre-baked profile
 *        for the canonical names ("ibm-falcon-emu",
 *        "rigetti-aspen-emu", "ionq-forte-emu") and the legacy
 *        aliases ("ibm-falcon", "rigetti-aspen", "ionq-forte"), or
 *        any profile registered via
 *        @ref moonlab_register_vendor_noise_profile.  NULL if
 *        @p name is not in the registry.
 */
MOONLAB_API const moonlab_vendor_noise_profile_t *
moonlab_lookup_vendor_noise_profile(const char *name);

/**
 * @brief Number of currently-registered profiles (including the
 *        pre-baked + alias entries).
 */
MOONLAB_API int moonlab_num_vendor_noise_profiles(void);

/**
 * @brief Populate @p out_names with up to @p max profile names.
 *        Pointers are owned by the registry, valid until the
 *        respective profile is unregistered.  Returns count
 *        written (<= max).
 */
MOONLAB_API int moonlab_list_vendor_noise_profiles(
        const char **out_names, int max);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_VENDOR_NOISE_BACKEND_H */
