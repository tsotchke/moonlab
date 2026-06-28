/**
 * @file    vendor_noise_backend.c
 * @brief   Stochastic-Pauli vendor-noise emulators.
 *
 * See vendor_noise_backend.h for the contract.  Implementation:
 *
 *   1. Run the ideal circuit through moonlab_qgtl_execute to get
 *      the noiseless shots.
 *   2. For each shot, walk the circuit again counting gates per
 *      qubit.  For every single-qubit gate, with probability
 *      p_gate_1q draw a uniformly-random Pauli {X, Y, Z} and
 *      XOR-flip the corresponding outcome bit (Y is X+Z; Z does
 *      nothing in the computational basis but is included so the
 *      total rate matches a depolarising channel).
 *   3. Same for two-qubit gates with p_gate_2q.
 *   4. Apply per-qubit readout bit-flip with p_readout.
 *
 * Step 1 reuses the existing simulator; steps 2-4 are pure
 * post-processing on uint64 outcomes.  This is exactly the Pauli
 * frame technique used by Stim and most published QPU pre-flight
 * benchmark suites; it is statistically correct for circuits whose
 * dominant noise is Pauli-stabilised and is a fast Monte-Carlo
 * approximation that does not carry the full density matrix.
 */

#include "vendor_noise_backend.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "moonlab_qgtl_backend.h"
#include "../distributed/scheduler.h"

/* ---- Pre-baked profiles ---- */

static const moonlab_vendor_noise_profile_t IBM_FALCON_TYPICAL = {
    /* IBM Falcon r5.11 heavy-hex typical (2024 public data):
     *   single-qubit gate error ~ 1e-3
     *   two-qubit (CX/ECR) gate error ~ 1e-2
     *   readout error ~ 2.0% */
    .p_gate_1q   = 0.001,
    .p_gate_2q   = 0.010,
    .p_readout   = 0.020,
    .description = "IBM Falcon r5.11 heavy-hex typical (public calibration)"
};

static const moonlab_vendor_noise_profile_t RIGETTI_ASPEN_TYPICAL = {
    /* Rigetti Aspen-M-3 octagon typical (public reports):
     *   1q gate error ~ 5e-4
     *   2q (CZ) gate error ~ 1.5e-2 (slightly worse than Falcon)
     *   readout error ~ 2.5% */
    .p_gate_1q   = 0.0005,
    .p_gate_2q   = 0.015,
    .p_readout   = 0.025,
    .description = "Rigetti Aspen-M-3 octagon-tile typical (public reports)"
};

static const moonlab_vendor_noise_profile_t IONQ_FORTE_TYPICAL = {
    /* IonQ Forte all-to-all (ion-trap) typical:
     *   1q gate error ~ 2e-4   (trapped-ion gates are very clean)
     *   2q gate error ~ 4e-3   (also better than superconducting)
     *   readout error ~ 0.5%   (PMT-based readout is very accurate) */
    .p_gate_1q   = 0.0002,
    .p_gate_2q   = 0.004,
    .p_readout   = 0.005,
    .description = "IonQ Forte all-to-all ion-trap typical (public reports)"
};

/* ============================================================ */
/* Profile registry (since v1.1.0).                              */
/*                                                                */
/* The three pre-baked typical profiles are auto-registered on   */
/* first access under both their canonical "-emu" names and the  */
/* legacy bare names (one release cycle of compatibility).       */
/* Sibling libraries can register additional profiles under      */
/* arbitrary names via moonlab_register_vendor_noise_profile.    */
/* Up to 32 slots; the registry is mutex-protected.              */
/* ============================================================ */

#define MOONLAB_VENDOR_NOISE_MAX_PROFILES 32

typedef struct {
    char                              *name;     /* heap-owned copy */
    moonlab_vendor_noise_profile_t     profile;  /* by-value copy */
    char                              *description_copy; /* if profile.description was non-NULL */
    int                                in_use;
} profile_slot_t;

static profile_slot_t g_profiles[MOONLAB_VENDOR_NOISE_MAX_PROFILES];
static int            g_num_profiles = 0;
static pthread_mutex_t g_profile_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_profile_init_once = PTHREAD_ONCE_INIT;

/* Internal: install a slot.  Caller holds the lock.  Replaces an
 * existing entry with the same name.  Returns the slot pointer or
 * NULL on OOM/full. */
static profile_slot_t *install_profile_locked(
    const char *name, const moonlab_vendor_noise_profile_t *profile)
{
    /* Replace if name already exists. */
    for (int i = 0; i < g_num_profiles; i++) {
        if (g_profiles[i].in_use && strcmp(g_profiles[i].name, name) == 0) {
            free(g_profiles[i].description_copy);
            g_profiles[i].description_copy = NULL;
            g_profiles[i].profile = *profile;
            if (profile->description) {
                g_profiles[i].description_copy = strdup(profile->description);
                g_profiles[i].profile.description = g_profiles[i].description_copy;
            }
            return &g_profiles[i];
        }
    }
    if (g_num_profiles >= MOONLAB_VENDOR_NOISE_MAX_PROFILES) return NULL;
    profile_slot_t *s = &g_profiles[g_num_profiles];
    s->name = strdup(name);
    if (!s->name) return NULL;
    s->profile = *profile;
    s->description_copy = NULL;
    if (profile->description) {
        s->description_copy = strdup(profile->description);
        s->profile.description = s->description_copy;
    }
    s->in_use = 1;
    g_num_profiles++;
    return s;
}

static void register_baked_profiles(void)
{
    /* No lock needed during pthread_once initialiser. */
    install_profile_locked("ibm-falcon-emu",    &IBM_FALCON_TYPICAL);
    install_profile_locked("rigetti-aspen-emu", &RIGETTI_ASPEN_TYPICAL);
    install_profile_locked("ionq-forte-emu",    &IONQ_FORTE_TYPICAL);
    /* Legacy aliases (one release of compatibility -- new code
     * should use the -emu suffix; the bare "ionq-forte" name is
     * reserved for the live-hardware backend that QGTL registers). */
    install_profile_locked("ibm-falcon",    &IBM_FALCON_TYPICAL);
    install_profile_locked("rigetti-aspen", &RIGETTI_ASPEN_TYPICAL);
    install_profile_locked("ionq-forte",    &IONQ_FORTE_TYPICAL);
}

static void ensure_profile_registry(void)
{
    pthread_once(&g_profile_init_once, register_baked_profiles);
}

int moonlab_register_vendor_noise_profile(
    const char *name, const moonlab_vendor_noise_profile_t *profile)
{
    if (!name || !profile) return MOONLAB_SCHED_BAD_ARG;
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    profile_slot_t *s = install_profile_locked(name, profile);
    pthread_mutex_unlock(&g_profile_lock);
    return s ? MOONLAB_SCHED_OK : MOONLAB_SCHED_OOM;
}

int moonlab_unregister_vendor_noise_profile(const char *name)
{
    if (!name) return MOONLAB_SCHED_BAD_ARG;
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    for (int i = 0; i < g_num_profiles; i++) {
        if (g_profiles[i].in_use && strcmp(g_profiles[i].name, name) == 0) {
            free(g_profiles[i].name);
            free(g_profiles[i].description_copy);
            /* Shift the tail down. */
            for (int j = i; j < g_num_profiles - 1; j++) {
                g_profiles[j] = g_profiles[j + 1];
            }
            memset(&g_profiles[g_num_profiles - 1], 0, sizeof(profile_slot_t));
            g_num_profiles--;
            pthread_mutex_unlock(&g_profile_lock);
            return MOONLAB_SCHED_OK;
        }
    }
    pthread_mutex_unlock(&g_profile_lock);
    return MOONLAB_SCHED_BACKEND_NOT_FOUND;
}

const moonlab_vendor_noise_profile_t *
moonlab_lookup_vendor_noise_profile(const char *name)
{
    if (!name) return NULL;
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    const moonlab_vendor_noise_profile_t *hit = NULL;
    for (int i = 0; i < g_num_profiles; i++) {
        if (g_profiles[i].in_use && strcmp(g_profiles[i].name, name) == 0) {
            hit = &g_profiles[i].profile;
            break;
        }
    }
    pthread_mutex_unlock(&g_profile_lock);
    return hit;
}

int moonlab_num_vendor_noise_profiles(void)
{
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    const int n = g_num_profiles;
    pthread_mutex_unlock(&g_profile_lock);
    return n;
}

int moonlab_list_vendor_noise_profiles(const char **out_names, int max)
{
    if (!out_names || max <= 0) return 0;
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    const int n = g_num_profiles < max ? g_num_profiles : max;
    for (int i = 0; i < n; i++) {
        out_names[i] = g_profiles[i].name;
    }
    pthread_mutex_unlock(&g_profile_lock);
    return n;
}

/* ---- Backend implementation ---- */

/* Read the job's gate list.  Scheduler's struct moonlab_job is
 * opaque to us; we re-serialise via the JSON path to get the gate
 * stream out and parse it back.  For a v1 emulator this is fine; a
 * future version can punch through with a `moonlab_job_walk_gates`
 * iterator. */

/* A simple Bob-style xorshift64* for fast stochastic noise sampling.
 * Each shot uses its own state derived from the run-level seed so
 * shots are independent. */
static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Internal: parse the gate stream from the job's JSON dump.  Returns
 * a heap-allocated array of (gate_type, target, control) triples in
 * `*out_gates`, with count in `*out_count`.  Caller frees with
 * free().  Returns 0 on success, -1 on failure. */
static int extract_gates(const moonlab_job_t *job,
                         int **out_targets,
                         int **out_controls,
                         int **out_kinds,    /* 1 = 1q, 2 = 2q */
                         int *out_count)
{
    /* Probe size + render JSON.  The format is stable
     * ("moonlab/job/v0.7.0"); we parse the gates array. */
    const int probe = moonlab_job_to_json(job, NULL, 0);
    if (probe <= 0) return -1;
    char *json = (char *)malloc((size_t)probe + 1);
    if (!json) return -1;
    if (moonlab_job_to_json(job, json, (size_t)probe + 1) != probe) {
        free(json);
        return -1;
    }

    /* Count gates: each is `{ "type": ... }`. */
    int n = 0;
    for (const char *p = json; (p = strstr(p, "\"type\":")) != NULL; p++) n++;

    int *targets  = (int *)calloc((size_t)n, sizeof(int));
    int *controls = (int *)calloc((size_t)n, sizeof(int));
    int *kinds    = (int *)calloc((size_t)n, sizeof(int));
    if (!targets || !controls || !kinds) {
        free(targets); free(controls); free(kinds); free(json);
        return -1;
    }

    int idx = 0;
    const char *cur = json;
    while ((cur = strstr(cur, "\"type\":")) != NULL && idx < n) {
        int type = -1, target = -1, control = -1;
        sscanf(cur, "\"type\": %d", &type);
        const char *tgt = strstr(cur, "\"target\":");
        if (tgt) sscanf(tgt, "\"target\": %d", &target);
        const char *ctl = strstr(cur, "\"control\":");
        /* control field may be present only for two-qubit gates; we
         * need to make sure we didn't skip past the next gate. */
        const char *next = strstr(cur + 1, "\"type\":");
        if (ctl && (!next || ctl < next)) {
            sscanf(ctl, "\"control\": %d", &control);
        }
        targets[idx]  = target;
        controls[idx] = control;
        /* Two-qubit if control is set to a non-negative value. */
        kinds[idx]    = (control >= 0) ? 2 : 1;
        idx++;
        cur++;
    }
    free(json);
    *out_targets  = targets;
    *out_controls = controls;
    *out_kinds    = kinds;
    *out_count    = idx;
    return 0;
}

/* Apply stochastic Pauli noise to a single shot bitstring.  For each
 * gate event, with probability p draw a uniformly-random Pauli; in
 * the computational basis X and Y flip the corresponding bit, Z
 * does nothing -- the 1/3 chance of Z is what makes the channel
 * depolarising rather than purely bit-flip. */
static uint64_t apply_pauli_noise_shot(uint64_t shot,
                                       const int *targets,
                                       const int *controls,
                                       const int *kinds,
                                       int num_gates,
                                       double p1, double p2,
                                       int num_qubits,
                                       double p_readout,
                                       uint64_t *rng_state)
{
    /* Per-gate Pauli error injection. */
    for (int g = 0; g < num_gates; g++) {
        const int is_2q = (kinds[g] == 2);
        const double p = is_2q ? p2 : p1;
        if (p <= 0.0) continue;
        const uint64_t r = xs64(rng_state);
        const double u = (double)((r >> 11) & 0x1FFFFFFFFFFFFFULL) /
                          (double)(1ULL << 53);
        if (u >= p) continue;
        /* Draw a Pauli: X, Y, or Z, uniform on {1, 2, 3}. */
        const int pauli = 1 + (int)((xs64(rng_state) >> 13) % 3);
        /* In the computational basis: X flips, Z preserves, Y flips
         * (Y = iXZ -> same outcome flip as X). */
        if (pauli != 3) {
            shot ^= (1ULL << targets[g]);
            if (is_2q) shot ^= (1ULL << controls[g]);
        }
    }
    /* Per-qubit readout bit-flip. */
    if (p_readout > 0.0) {
        for (int q = 0; q < num_qubits; q++) {
            const uint64_t r = xs64(rng_state);
            const double u = (double)((r >> 11) & 0x1FFFFFFFFFFFFFULL) /
                              (double)(1ULL << 53);
            if (u < p_readout) shot ^= (1ULL << q);
        }
    }
    return shot;
}

/* Execute fn: runs the noiseless simulator first, then post-
 * processes each shot with stochastic Pauli + readout noise.
 *
 * The backend's ctx is a `const char *` -- the name of the noise
 * profile to look up in the profile registry at execute time.
 * This indirection lets live-calibration scrapers update a
 * profile in place and have future runs of the same backend pick
 * up the new calibration without re-registering the backend. */
static int vendor_noise_execute(const moonlab_job_t   *job,
                                moonlab_job_results_t *out,
                                void                  *ctx)
{
    const char *profile_name = (const char *)ctx;
    if (!profile_name) return MOONLAB_SCHED_BAD_ARG;
    const moonlab_vendor_noise_profile_t *profile =
        moonlab_lookup_vendor_noise_profile(profile_name);
    if (!profile) return MOONLAB_SCHED_BACKEND_NOT_FOUND;

    /* Extract gate-list footprint (target / control / 1q-vs-2q) up
     * front so we can post-process each shot in a single pass.  The
     * gate types and parameters themselves don't matter for the
     * noise model -- only the topology of the application
     * (which qubits got a gate, and whether it was 1q or 2q). */
    int *targets = NULL, *controls = NULL, *kinds = NULL;
    int ngates = 0;
    if (extract_gates(job, &targets, &controls, &kinds, &ngates) != 0) {
        return MOONLAB_SCHED_INTERNAL;
    }

    /* Run noiseless on the simulator.  We dispatch through the
     * simulator backend directly (rather than recursing through
     * moonlab_scheduler_run, which would route back to us if the
     * caller's job is pinned to this vendor-noise backend). */
    const moonlab_backend_t *sim = moonlab_find_backend("simulator");
    if (!sim) {
        free(targets); free(controls); free(kinds);
        return MOONLAB_SCHED_INTERNAL;
    }
    /* simulator_backend_execute reads only num_qubits / num_shots /
     * num_workers / gates / rng_seed -- it ignores backend_name --
     * so the const cast is safe.  The simulator does not mutate the
     * job. */
    const int rc = sim->execute((moonlab_job_t *)job, out, sim->ctx);
    if (rc != MOONLAB_SCHED_OK) {
        free(targets); free(controls); free(kinds);
        return rc;
    }

    /* Stochastic Pauli noise per shot. */
    const int total_shots = out->total_shots;
    const int num_qubits  = out->num_qubits;
    uint64_t rng_state = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL +
                          (uint64_t)num_qubits + 1ULL;
    for (int s = 0; s < total_shots; s++) {
        out->outcomes[s] = apply_pauli_noise_shot(
            out->outcomes[s], targets, controls, kinds, ngates,
            profile->p_gate_1q, profile->p_gate_2q, num_qubits,
            profile->p_readout, &rng_state);
    }
    free(targets); free(controls); free(kinds);
    return MOONLAB_SCHED_OK;
}

int moonlab_register_vendor_noise_backend_with_profile(
        const char *name,
        const moonlab_vendor_noise_profile_t *profile)
{
    if (!name || !profile) return MOONLAB_SCHED_BAD_ARG;
    /* Step 1: install the profile in the registry under @p name.
     * It gets copied; the caller's storage can be freed after
     * this call. */
    const int rc_p = moonlab_register_vendor_noise_profile(name, profile);
    if (rc_p != MOONLAB_SCHED_OK) return rc_p;

    /* Step 2: look the just-registered profile back up to get the
     * registry-owned name pointer (stable for the profile's
     * lifetime, which is what the backend's ctx needs). */
    ensure_profile_registry();
    pthread_mutex_lock(&g_profile_lock);
    const char *stable_name = NULL;
    for (int i = 0; i < g_num_profiles; i++) {
        if (g_profiles[i].in_use &&
            strcmp(g_profiles[i].name, name) == 0) {
            stable_name = g_profiles[i].name;
            break;
        }
    }
    /* Look up the description too, for the backend record. */
    const char *desc = NULL;
    for (int i = 0; i < g_num_profiles; i++) {
        if (g_profiles[i].in_use &&
            strcmp(g_profiles[i].name, name) == 0) {
            desc = g_profiles[i].profile.description;
            break;
        }
    }
    pthread_mutex_unlock(&g_profile_lock);
    if (!stable_name) return MOONLAB_SCHED_INTERNAL;

    const moonlab_backend_t be = {
        .name        = stable_name,
        .execute     = vendor_noise_execute,
        .ctx         = (void *)stable_name,
        .description = desc
    };
    return moonlab_register_backend(&be);
}

int moonlab_register_vendor_noise_backends(void)
{
    /* Canonical names ("-emu" suffix).  These are the new
     * recommended names; "ionq-forte" without the suffix is
     * reserved for the live-hardware backend QGTL registers. */
    int rc = moonlab_register_vendor_noise_backend_with_profile(
        "ibm-falcon-emu",    &IBM_FALCON_TYPICAL);
    if (rc != MOONLAB_SCHED_OK) return rc;
    rc = moonlab_register_vendor_noise_backend_with_profile(
        "rigetti-aspen-emu", &RIGETTI_ASPEN_TYPICAL);
    if (rc != MOONLAB_SCHED_OK) return rc;
    rc = moonlab_register_vendor_noise_backend_with_profile(
        "ionq-forte-emu",    &IONQ_FORTE_TYPICAL);
    if (rc != MOONLAB_SCHED_OK) return rc;

    /* Legacy aliases (one release of compat).  Same profile data
     * but registered under the bare names so existing code that
     * hard-codes "ibm-falcon" etc. keeps working.  Deprecated;
     * QGTL's live IonQ backend will replace "ionq-forte" in the
     * next release. */
    rc = moonlab_register_vendor_noise_backend_with_profile(
        "ibm-falcon",    &IBM_FALCON_TYPICAL);
    if (rc != MOONLAB_SCHED_OK) return rc;
    rc = moonlab_register_vendor_noise_backend_with_profile(
        "rigetti-aspen", &RIGETTI_ASPEN_TYPICAL);
    if (rc != MOONLAB_SCHED_OK) return rc;
    rc = moonlab_register_vendor_noise_backend_with_profile(
        "ionq-forte",    &IONQ_FORTE_TYPICAL);
    return rc;
}
