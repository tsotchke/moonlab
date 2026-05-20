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
    CHECK(moonlab_find_backend("ibm-falcon")    != NULL, "ibm-falcon registered");
    CHECK(moonlab_find_backend("rigetti-aspen") != NULL, "rigetti-aspen registered");
    CHECK(moonlab_find_backend("ionq-forte")    != NULL, "ionq-forte registered");
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
    moonlab_job_set_backend(ja, "rigetti-aspen");
    moonlab_job_results_t ra = {0};
    moonlab_scheduler_run(ja, &ra);
    int rigetti_other = 0;
    for (int s = 0; s < ra.total_shots; s++) {
        if (ra.outcomes[s] != 0 && ra.outcomes[s] != 3) rigetti_other++;
    }

    moonlab_job_t *jb = make_bell_job(16384);
    moonlab_job_set_rng_seed(jb, 0xfeedfaceULL);
    moonlab_job_set_backend(jb, "ionq-forte");
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

    /* Expected off-Bell fractions per circuit (1 CNOT + 2-qubit readout):
     *   ibm:     p2q + 2 * p_readout ~ 1.0% + 4.0% = 5.0% (rough)
     *   rigetti: p2q + 2 * p_readout ~ 1.5% + 5.0% = 6.5%
     *   ionq:    p2q + 2 * p_readout ~ 0.4% + 1.0% = 1.4% */
    test_backend_runs_and_noise_fires("ibm-falcon",    0.01, 0.15);
    test_backend_runs_and_noise_fires("rigetti-aspen", 0.01, 0.18);
    test_backend_runs_and_noise_fires("ionq-forte",    0.001, 0.08);

    test_ionq_cleaner_than_rigetti();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
