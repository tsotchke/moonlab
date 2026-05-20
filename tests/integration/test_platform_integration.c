/**
 * @file  test_platform_integration.c
 * @brief End-to-end smoke test of the moonlab + QGTL + libirrep + SbNN
 *        platform-integration story (since v1.1).
 *
 * What this test enforces:
 *
 *   1. QGTL circuit construction works at the C API level.
 *   2. moonlab_scheduler_run executes the circuit through the
 *      default (simulator) backend and returns Bell-correlated
 *      outcomes.
 *   3. The scheduler's vendor-noise backend (v1.1) plugs into the
 *      same dispatch path and produces shots with the expected
 *      noise signature.
 *   4. The libirrep bridge reports honest availability -- either
 *      "available" with a working factory call, or NOT_BUILT
 *      surfaced as the documented error code.
 *   5. The SbNN decoder slot reports honest availability through
 *      the moonlab_decoder_slot_available query.
 *
 * This is the "platform-of-record" integration smoke: if any of
 * the four named libraries breaks its contract with moonlab, this
 * test catches it before anything downstream notices.
 *
 * @since v1.1.0
 */

#include "../../src/applications/moonlab_qgtl_backend.h"
#include "../../src/applications/decoder_bench.h"
#include "../../src/applications/vendor_noise_backend.h"
#include "../../src/distributed/scheduler.h"
#include "../../src/integration/libirrep_bridge.h"

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

/* ============================================================ */
/* Stage 1: QGTL + scheduler                                     */
/* ============================================================ */

static void stage_qgtl_scheduler_simulator(void)
{
    fprintf(stdout, "\n=== stage 1: QGTL circuit -> scheduler -> simulator ===\n");
    moonlab_job_t *j = moonlab_job_create(2);
    CHECK(j != NULL, "moonlab_job_create(2)");
    CHECK(moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H, 0, -1, NULL) == 0,
          "QGTL_GATE_H on q0");
    CHECK(moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL) == 0,
          "QGTL_GATE_CNOT q0->q1");
    moonlab_job_set_num_shots(j, 1024);
    moonlab_job_set_num_workers(j, 2);
    moonlab_job_set_rng_seed(j, 0xdeadbeefULL);

    moonlab_job_results_t res = {0};
    const int rc = moonlab_scheduler_run(j, &res);
    CHECK(rc == 0, "scheduler_run on default simulator backend");
    CHECK(res.total_shots == 1024, "shots through");

    /* Ideal Bell: support exactly {|00>, |11>}. */
    int n_other = 0;
    for (int s = 0; s < res.total_shots; s++) {
        if (res.outcomes[s] != 0 && res.outcomes[s] != 3) n_other++;
    }
    CHECK(n_other == 0, "Bell support clean (no off-diagonal shots)");

    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

/* ============================================================ */
/* Stage 2: vendor-noise backend                                 */
/* ============================================================ */

static void stage_vendor_noise(void)
{
    fprintf(stdout, "\n=== stage 2: vendor-noise emulator backend ===\n");
    CHECK(moonlab_register_vendor_noise_backends() == 0,
          "register vendor-noise emulator backends");
    CHECK(moonlab_find_backend("ibm-falcon")    != NULL, "ibm-falcon registered");
    CHECK(moonlab_find_backend("rigetti-aspen") != NULL, "rigetti-aspen registered");
    CHECK(moonlab_find_backend("ionq-forte")    != NULL, "ionq-forte registered");

    moonlab_job_t *j = moonlab_job_create(2);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H,    0, -1, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1,  0, NULL);
    moonlab_job_set_num_shots(j, 4096);
    moonlab_job_set_num_workers(j, 1);
    moonlab_job_set_backend(j, "ibm-falcon");

    moonlab_job_results_t res = {0};
    CHECK(moonlab_scheduler_run(j, &res) == 0,
          "scheduler_run via ibm-falcon (noise emulator)");

    int n_other = 0;
    for (int s = 0; s < res.total_shots; s++) {
        if (res.outcomes[s] != 0 && res.outcomes[s] != 3) n_other++;
    }
    /* IBM Falcon typical noise: expect ~5% off-Bell shots from
     * p_2q + 2 * p_readout = 1% + 4% = 5%. */
    const double frac = (double)n_other / (double)res.total_shots;
    CHECK(frac > 0.005, "noise actually fires (off-Bell frac = %.4f)", frac);
    CHECK(frac < 0.20,  "noise not catastrophic (off-Bell frac = %.4f)", frac);

    moonlab_job_results_free(&res);
    moonlab_job_free(j);
}

/* ============================================================ */
/* Stage 3: libirrep bridge -- contract honesty                  */
/* ============================================================ */

static void stage_libirrep_contract(void)
{
    fprintf(stdout, "\n=== stage 3: libirrep bridge contract ===\n");
    const int avail = moonlab_libirrep_available();
    fprintf(stdout, "    moonlab_libirrep_available() = %d\n", avail);
    /* When QSIM_ENABLE_LIBIRREP=OFF the factory should report
     * MOONLAB_LIBIRREP_NOT_BUILT (-201) explicitly.  When ON, the
     * factory should produce a non-NULL handle.  Either is honest;
     * an availability of 1 with NULL output, or 0 with non-NULL,
     * would be a contract violation. */
    if (avail == 0) {
        double e_kagome = 0.0;
        const int rc = moonlab_libirrep_kagome12_e0(&e_kagome);
        CHECK(rc == -201 /* MOONLAB_LIBIRREP_NOT_BUILT */,
              "kagome12_e0 returns NOT_BUILT (-201) when libirrep is off (got %d)", rc);
    } else {
        double e_kagome = 0.0;
        const int rc = moonlab_libirrep_kagome12_e0(&e_kagome);
        CHECK(rc == 0,
              "kagome12_e0 returns 0 when libirrep is on (got %d, e=%.6f)",
              rc, e_kagome);
        CHECK(e_kagome < -5.4 && e_kagome > -5.5,
              "kagome12 E0 = %.6f matches libirrep reference (-5.44488)", e_kagome);
    }
}

/* ============================================================ */
/* Stage 4: SbNN decoder slot -- availability honesty            */
/* ============================================================ */

static void stage_sbnn_decoder_contract(void)
{
    fprintf(stdout, "\n=== stage 4: SbNN decoder slot contract ===\n");
    const int avail = moonlab_decoder_slot_available(MOONLAB_DECODER_SBNN);
    fprintf(stdout, "    SbNN slot available = %d\n", avail);

    /* Always-on slots regardless of QSIM_ENABLE_SBNN: */
    CHECK(moonlab_decoder_slot_available(MOONLAB_DECODER_GREEDY) == 1,
          "GREEDY slot always available");
    CHECK(moonlab_decoder_slot_available(MOONLAB_DECODER_MWPM_EXACT) == 1,
          "MWPM_EXACT slot always available");

    /* If SbNN is not linked, attempting to decode must return
     * NOT_BUILT.  If it is linked, it should produce a valid
     * correction. */
    unsigned char syndromes[9] = { 0, 1, 0, 0, 0, 0, 0, 1, 0 };
    unsigned char corrections[18] = {0};
    const moonlab_decoder_code_t code = {
        .distance = 3, .num_qubits = 18, .is_toric = 1
    };
    const moonlab_decoder_input_t in = {
        .code = &code,
        .syndromes = syndromes,
        .corrections = corrections,
        .num_stabilisers = 9,
        .rng_seed = 0xfeedfaceULL,
    };
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_SBNN, &in);
    if (avail == 0) {
        CHECK(rc == MOONLAB_DECODER_NOT_BUILT,
              "SBNN decode returns NOT_BUILT (%d) when slot is off (got %d)",
              MOONLAB_DECODER_NOT_BUILT, rc);
    } else {
        CHECK(rc == 0,
              "SBNN decode returns 0 when slot is on (got %d)", rc);
    }
}

/* ============================================================ */
/* Stage 5: backend registry inventory dump                      */
/* ============================================================ */

static void stage_registry_inventory(void)
{
    fprintf(stdout, "\n=== stage 5: scheduler backend registry inventory ===\n");
    const int n = moonlab_num_backends();
    CHECK(n >= 4,
          "registry has >= 4 backends (simulator + 3 vendor-noise) (n=%d)", n);

    const char *names[16] = {0};
    const int got = moonlab_list_backends(names, 16);
    fprintf(stdout, "    registered backends (%d):\n", got);
    int have_simulator = 0, have_ibm = 0, have_rigetti = 0, have_ionq = 0;
    for (int i = 0; i < got; i++) {
        fprintf(stdout, "      - %s\n", names[i]);
        if (strcmp(names[i], "simulator")     == 0) have_simulator = 1;
        if (strcmp(names[i], "ibm-falcon")    == 0) have_ibm = 1;
        if (strcmp(names[i], "rigetti-aspen") == 0) have_rigetti = 1;
        if (strcmp(names[i], "ionq-forte")    == 0) have_ionq = 1;
    }
    CHECK(have_simulator, "simulator backend listed");
    CHECK(have_ibm,       "ibm-falcon backend listed");
    CHECK(have_rigetti,   "rigetti-aspen backend listed");
    CHECK(have_ionq,      "ionq-forte backend listed");
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== moonlab + QGTL + libirrep + SbNN platform smoke ===\n");

    stage_qgtl_scheduler_simulator();
    stage_vendor_noise();
    stage_libirrep_contract();
    stage_sbnn_decoder_contract();
    stage_registry_inventory();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
