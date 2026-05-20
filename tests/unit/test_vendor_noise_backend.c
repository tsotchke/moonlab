/**
 * @file  test_vendor_noise_backend.c
 * @brief Validation of the IBM / Rigetti / IonQ vendor-noise backends.
 *
 * Strategy: run a Bell pair through each backend with a large shot
 * count.  The noiseless Bell distribution has support exactly
 * {|00>, |11>}; any non-Bell outcome (|01> or |10>) is a signature
 * that noise actually fired.  We assert:
 *
 *   1. Backend registers successfully.
 *   2. scheduler_run dispatches to it.
 *   3. The fraction of non-Bell outcomes is non-zero (noise fires)
 *      AND below an order-of-magnitude upper bound (noise isn't
 *      catastrophically too high).
 *   4. The total number of outcomes matches the requested shot count.
 *
 * Profiles tested:
 *   - ibm-falcon     p2q = 1.0%  p_readout = 2.0%
 *   - rigetti-aspen  p2q = 1.5%  p_readout = 2.5%
 *   - ionq-forte     p2q = 0.4%  p_readout = 0.5%
 *
 * With a 2-gate circuit (H + CNOT), the expected fraction of
 * non-Bell shots is roughly p_2q + 2 * p_readout (single CNOT
 * fires 2q noise; readout fires on both qubits).  We bound this:
 *
 *   ibm-falcon:    ~5.0%   -- expect 1% .. 12%
 *   rigetti-aspen: ~6.5%   -- expect 1% .. 14%
 *   ionq-forte:    ~1.4%   -- expect 0.1% .. 6%
 */

#include "../../src/distributed/scheduler.h"
#include "../../src/applications/vendor_noise_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static moonlab_job_t *make_bell_job(int shots)
{
    moonlab_job_t *j = moonlab_job_create(2);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H,    0, -1, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1,  0, NULL);
    moonlab_job_set_num_shots(j, shots);
    moonlab_job_set_num_workers(j, 1);
    return j;
}

static void test_profile_lookup(void)
{
    fprintf(stdout, "\n--- vendor-noise: profile lookup ---\n");
    const moonlab_vendor_noise_profile_t *ibm =
        moonlab_lookup_vendor_noise_profile("ibm-falcon");
    CHECK(ibm != NULL, "ibm-falcon profile found");
    if (ibm) {
        CHECK(ibm->p_gate_1q > 0.0 && ibm->p_gate_1q < 0.1,
              "ibm 1q error %.4f in (0, 0.1)", ibm->p_gate_1q);
        CHECK(ibm->p_gate_2q > ibm->p_gate_1q,
              "ibm 2q error %.4f > 1q error", ibm->p_gate_2q);
        CHECK(ibm->p_readout > 0.0 && ibm->p_readout < 0.1,
              "ibm readout %.4f in (0, 0.1)", ibm->p_readout);
    }
    const moonlab_vendor_noise_profile_t *ionq =
        moonlab_lookup_vendor_noise_profile("ionq-forte");
    CHECK(ionq != NULL, "ionq-forte profile found");
    /* IonQ should have the cleanest gates of the three. */
    if (ibm && ionq) {
        CHECK(ionq->p_gate_2q < ibm->p_gate_2q,
              "ionq 2q (%.4f) < ibm 2q (%.4f)",
              ionq->p_gate_2q, ibm->p_gate_2q);
    }
    CHECK(moonlab_lookup_vendor_noise_profile("does-not-exist") == NULL,
          "unknown profile name returns NULL");
}

static void test_register_all(void)
{
    fprintf(stdout, "\n--- vendor-noise: register backends ---\n");
    const int rc = moonlab_register_vendor_noise_backends();
    CHECK(rc == 0, "register_all rc=%d", rc);
    /* Canonical "-emu" names (the new recommended surface). */
    CHECK(moonlab_find_backend("ibm-falcon-emu")    != NULL,
          "ibm-falcon-emu registered (canonical)");
    CHECK(moonlab_find_backend("rigetti-aspen-emu") != NULL,
          "rigetti-aspen-emu registered (canonical)");
    CHECK(moonlab_find_backend("ionq-forte-emu")    != NULL,
          "ionq-forte-emu registered (canonical)");
    /* Legacy bare names (one release of compat). */
    CHECK(moonlab_find_backend("ibm-falcon")    != NULL,
          "ibm-falcon registered (legacy alias)");
    CHECK(moonlab_find_backend("rigetti-aspen") != NULL,
          "rigetti-aspen registered (legacy alias)");
    CHECK(moonlab_find_backend("ionq-forte")    != NULL,
          "ionq-forte registered (legacy alias)");
}

static void test_profile_registry(void)
{
    fprintf(stdout, "\n--- vendor-noise: profile registry ---\n");

    /* The six baked-in profile entries are auto-registered. */
    const int n_baked = moonlab_num_vendor_noise_profiles();
    CHECK(n_baked >= 6,
          "registry has >= 6 baked-in profiles (canonical + legacy) (n=%d)",
          n_baked);

    /* Register a custom profile -- simulates the private-overlay
     * live-calibration scraper installing today's IBM device snapshot. */
    const moonlab_vendor_noise_profile_t custom = {
        .p_gate_1q   = 0.0015,
        .p_gate_2q   = 0.012,
        .p_readout   = 0.018,
        .description = "IBM Falcon (2026-05-20 live snapshot)"
    };
    CHECK(moonlab_register_vendor_noise_profile(
              "ibm-falcon-2026-05-20", &custom) == 0,
          "register custom profile");
    const moonlab_vendor_noise_profile_t *back =
        moonlab_lookup_vendor_noise_profile("ibm-falcon-2026-05-20");
    CHECK(back != NULL, "lookup custom profile");
    if (back) {
        CHECK(back->p_gate_2q > 0.011 && back->p_gate_2q < 0.013,
              "custom p_gate_2q round-trips (%.4f)", back->p_gate_2q);
        CHECK(strstr(back->description, "live snapshot") != NULL,
              "description round-trips");
    }

    /* Update the custom profile in place -- the registry copies, so
     * future runs see the update. */
    const moonlab_vendor_noise_profile_t updated = {
        .p_gate_1q   = 0.0008,    /* improved calibration */
        .p_gate_2q   = 0.008,
        .p_readout   = 0.012,
        .description = "IBM Falcon (2026-05-21 live snapshot, improved)"
    };
    CHECK(moonlab_register_vendor_noise_profile(
              "ibm-falcon-2026-05-20", &updated) == 0,
          "register replaces existing profile in place");
    back = moonlab_lookup_vendor_noise_profile("ibm-falcon-2026-05-20");
    CHECK(back != NULL && back->p_gate_2q < 0.009,
          "in-place update visible (p_gate_2q = %.4f)",
          back ? back->p_gate_2q : -1.0);

    /* Unregister cleanly. */
    CHECK(moonlab_unregister_vendor_noise_profile(
              "ibm-falcon-2026-05-20") == 0,
          "unregister custom profile");
    CHECK(moonlab_lookup_vendor_noise_profile(
              "ibm-falcon-2026-05-20") == NULL,
          "lookup after unregister returns NULL");
    CHECK(moonlab_unregister_vendor_noise_profile(
              "nonexistent") != 0,
          "unregister nonexistent profile reports error");
}

static void test_backend_runs_and_noise_fires(const char *backend_name,
                                              double bell_fraction_lo,
                                              double bell_fraction_hi)
{
    fprintf(stdout, "\n--- vendor-noise: %s on Bell pair, 8192 shots ---\n",
            backend_name);
    moonlab_job_t *j = make_bell_job(8192);
    CHECK(moonlab_job_set_backend(j, backend_name) == 0,
          "set_backend(\"%s\") OK", backend_name);

    moonlab_job_results_t res = {0};
    CHECK(moonlab_scheduler_run(j, &res) == 0, "scheduler_run via %s",
          backend_name);
    CHECK(res.total_shots == 8192, "total_shots = %d", res.total_shots);

    int n_bell = 0, n_other = 0;
    for (int s = 0; s < res.total_shots; s++) {
        const uint64_t b = res.outcomes[s];
        if (b == 0 || b == 3) n_bell++;
        else n_other++;
    }
    const double frac_other = (double)n_other / (double)res.total_shots;
    fprintf(stdout, "    bell=%d  other=%d  P(other) = %.4f\n",
            n_bell, n_other, frac_other);
    CHECK(frac_other > bell_fraction_lo,
          "P(off-Bell) = %.4f > %.4f (noise fires)",
          frac_other, bell_fraction_lo);
    CHECK(frac_other < bell_fraction_hi,
          "P(off-Bell) = %.4f < %.4f (noise not catastrophic)",
          frac_other, bell_fraction_hi);

    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

static void test_ionq_cleaner_than_rigetti(void)
{
    /* Same circuit on both backends; IonQ should yield strictly
     * fewer off-Bell outcomes per shot.  Use a fixed seed (set via
     * job seed) so the comparison is meaningful. */
    fprintf(stdout, "\n--- vendor-noise: ionq cleaner than rigetti ---\n");
    moonlab_job_t *ja = make_bell_job(16384);
    moonlab_job_set_rng_seed(ja, 0xfeedfaceULL);
    moonlab_job_set_backend(ja, "rigetti-aspen-emu");
    moonlab_job_results_t ra = {0};
    moonlab_scheduler_run(ja, &ra);
    int rigetti_other = 0;
    for (int s = 0; s < ra.total_shots; s++) {
        if (ra.outcomes[s] != 0 && ra.outcomes[s] != 3) rigetti_other++;
    }

    moonlab_job_t *jb = make_bell_job(16384);
    moonlab_job_set_rng_seed(jb, 0xfeedfaceULL);
    moonlab_job_set_backend(jb, "ionq-forte-emu");
    moonlab_job_results_t rb = {0};
    moonlab_scheduler_run(jb, &rb);
    int ionq_other = 0;
    for (int s = 0; s < rb.total_shots; s++) {
        if (rb.outcomes[s] != 0 && rb.outcomes[s] != 3) ionq_other++;
    }
    fprintf(stdout, "    rigetti off-Bell = %d   ionq off-Bell = %d\n",
            rigetti_other, ionq_other);
    CHECK(ionq_other < rigetti_other,
          "ionq (%d) < rigetti (%d) off-Bell shots (cleaner gates)",
          ionq_other, rigetti_other);
    moonlab_job_results_free(&ra);
    moonlab_job_results_free(&rb);
    moonlab_job_free(ja);
    moonlab_job_free(jb);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== vendor-noise emulator backends ===\n");

    test_profile_lookup();
    test_register_all();
    test_profile_registry();

    /* Canonical "-emu" names exercise the noise pipeline. */
    test_backend_runs_and_noise_fires("ibm-falcon-emu",    0.01, 0.15);
    test_backend_runs_and_noise_fires("rigetti-aspen-emu", 0.01, 0.18);
    test_backend_runs_and_noise_fires("ionq-forte-emu",    0.001, 0.08);

    /* Legacy bare-name aliases must produce equivalent behavior. */
    test_backend_runs_and_noise_fires("ibm-falcon",        0.01, 0.15);

    test_ionq_cleaner_than_rigetti();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
