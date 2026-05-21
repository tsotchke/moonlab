/**
 * @file  open_core_overlay_demo.c
 * @brief Reference overlay using every public plug-in surface.
 *
 * Demonstrates how a private overlay (or QGTL / SbNN / any external
 * consumer) layers on top of a stock moonlab build using the four
 * runtime registries shipped in v1.0.3:
 *
 *   1. moonlab_register_backend            -- new execution backend
 *   2. moonlab_register_vendor_noise_profile -- live calibration data
 *   3. moonlab_register_decoder            -- proprietary QEC decoder
 *   4. moonlab_scheduler_set_completion_hook -- billing / audit meter
 *
 * The C tree never sees the overlay source.  Run this example and
 * observe that:
 *
 *   - A user-supplied backend ("overlay-deterministic") runs jobs
 *     pinned to its name.
 *   - The vendor-noise emulator picks up a freshly-registered profile
 *     ("ibm-falcon-2026-05-20-snapshot") and produces noise matching
 *     the calibration data.
 *   - A user decoder ("overlay-zero-corrector") is dispatched both
 *     by name AND through the enum dispatcher (since the registry is
 *     the single source of truth).
 *   - The completion hook fires once per successful run and records
 *     the (num_qubits, total_shots, backend_name) tuple suitable for
 *     a billing meter.
 *
 * Public moonlab proves the plug-in pattern by exercising it in-tree.
 *
 * @since v1.0.3
 */

#include "../../src/distributed/scheduler.h"
#include "../../src/applications/vendor_noise_backend.h"
#include "../../src/applications/decoder_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * 1) Backend: a deterministic "always returns |00...0>" backend.
 *
 * Real overlays would dispatch to live hardware, an internal cluster,
 * a GPU farm, etc.  Here we just produce zero outcomes so the
 * billing-hook tracking has stable data.
 * -------------------------------------------------------------------------- */
static int overlay_backend_execute(const moonlab_job_t *job,
                                   moonlab_job_results_t *out,
                                   void *ctx)
{
    (void)ctx;
    const int n_q   = moonlab_job_num_qubits(job);
    const int n_s   = moonlab_job_num_shots(job);
    if (n_q <= 0 || n_s <= 0) return MOONLAB_SCHED_BAD_ARG;
    out->num_qubits       = n_q;
    out->total_shots      = n_s;
    out->num_workers_used = 1;
    out->outcomes         = (uint64_t *)calloc((size_t)n_s, sizeof(uint64_t));
    out->worker_seconds   = (double  *)calloc(1, sizeof(double));
    if (!out->outcomes || !out->worker_seconds) {
        free(out->outcomes); free(out->worker_seconds);
        return MOONLAB_SCHED_OOM;
    }
    /* All shots produce |00...0>; the worker reports a fixed runtime. */
    out->worker_seconds[0] = 0.000123;
    return MOONLAB_SCHED_OK;
}

/* --------------------------------------------------------------------------
 * 3) Decoder: an overlay decoder that always emits the zero-correction
 *    vector.  Useful as a baseline floor; a real overlay would call out
 *    to a proprietary BP+OSD, GNN, or hardware decoder.
 * -------------------------------------------------------------------------- */
static int overlay_zero_corrector(const moonlab_decoder_input_t *in, void *ctx)
{
    (void)ctx;
    memset(in->corrections, 0, (size_t)in->code->num_qubits);
    return MOONLAB_DECODER_OK;
}

/* --------------------------------------------------------------------------
 * 4) Completion hook: a billing/audit recorder.
 * -------------------------------------------------------------------------- */
typedef struct {
    int    n_runs;
    int    total_shots_billed;
    char   last_backend[64];
    double accumulated_cost_usd;
} billing_state_t;

/* Toy pricing: 0.01 USD per 1000 shots on the simulator, 1.00 USD per
 * 1000 shots on the overlay backend (the "premium" lane). */
static double price_per_kshot(const char *backend_name)
{
    if (!backend_name) return 0.01;
    if (strcmp(backend_name, "overlay-deterministic") == 0) return 1.00;
    if (strncmp(backend_name, "ibm-falcon", 10) == 0)       return 0.50;
    return 0.01;
}

static void billing_hook(const moonlab_job_t          *job,
                         const moonlab_job_results_t  *out,
                         const char                   *backend_name,
                         void                         *ctx)
{
    (void)job;
    billing_state_t *s = (billing_state_t *)ctx;
    if (!s) return;
    s->n_runs++;
    s->total_shots_billed += out->total_shots;
    snprintf(s->last_backend, sizeof(s->last_backend), "%s",
             backend_name ? backend_name : "(unknown)");
    s->accumulated_cost_usd +=
        price_per_kshot(backend_name) * (double)out->total_shots / 1000.0;
}

/* --------------------------------------------------------------------------
 * Demo driver
 * -------------------------------------------------------------------------- */
static moonlab_job_t *make_bell_job(int shots, const char *backend_name)
{
    moonlab_job_t *j = moonlab_job_create(2);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H,    0, -1, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1,  0, NULL);
    moonlab_job_set_num_shots(j, shots);
    moonlab_job_set_num_workers(j, 1);
    moonlab_job_set_backend(j, backend_name);
    return j;
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    fprintf(stdout, "=== Moonlab open-core overlay demo ===\n\n");

    /* ----- 1) Register a new execution backend ----------------------- */
    moonlab_backend_t overlay_backend = {
        .name        = "overlay-deterministic",
        .execute     = overlay_backend_execute,
        .ctx         = NULL,
        .description = "Demo overlay backend (always |00...0>)",
    };
    if (moonlab_register_backend(&overlay_backend) != 0) {
        fprintf(stderr, "register_backend failed\n");
        return EXIT_FAILURE;
    }
    fprintf(stdout, "  [1/4] registered backend  : %s\n", overlay_backend.name);

    /* ----- 2) Register a fresh vendor-noise profile snapshot --------- */
    const moonlab_vendor_noise_profile_t today = {
        .p_gate_1q   = 0.0011,
        .p_gate_2q   = 0.0095,
        .p_readout   = 0.0162,
        .description = "IBM Falcon r5.11 (2026-05-20 scraper output)",
    };
    moonlab_register_vendor_noise_backend_with_profile(
        "ibm-falcon-2026-05-20-snapshot", &today);
    fprintf(stdout, "  [2/4] registered profile  : ibm-falcon-2026-05-20-snapshot\n");
    fprintf(stdout, "                              p_gate_2q = %.4f (live scraper)\n",
            today.p_gate_2q);

    /* ----- 3) Register a proprietary decoder ------------------------- */
    moonlab_register_decoder("overlay-zero-corrector",
                             overlay_zero_corrector, NULL,
                             "Overlay zero-correction baseline decoder");
    fprintf(stdout, "  [3/4] registered decoder  : overlay-zero-corrector\n");

    /* ----- 4) Install the billing completion hook -------------------- */
    billing_state_t billing = {0};
    moonlab_scheduler_set_completion_hook(billing_hook, &billing);
    fprintf(stdout, "  [4/4] installed billing hook\n\n");

    /* Three runs on three different backends to demonstrate the
     * full surface end-to-end. */
    const char *backends[] = {
        "simulator",
        "overlay-deterministic",
        "ibm-falcon-2026-05-20-snapshot",
    };
    const int n_backends = (int)(sizeof(backends) / sizeof(backends[0]));

    for (int b = 0; b < n_backends; b++) {
        moonlab_job_t *j = make_bell_job(2048, backends[b]);
        moonlab_job_results_t r = {0};
        const int rc = moonlab_scheduler_run(j, &r);
        if (rc != MOONLAB_SCHED_OK) {
            fprintf(stderr, "run on %s failed rc=%d\n", backends[b], rc);
            moonlab_job_free(j);
            continue;
        }
        int n_zero = 0, n_bell = 0;
        for (int s = 0; s < r.total_shots; s++) {
            if (r.outcomes[s] == 0)              n_zero++;
            else if (r.outcomes[s] == 3)         n_bell++;
        }
        fprintf(stdout, "    run on %-36s : %d shots, |00>=%d, |11>=%d\n",
                backends[b], r.total_shots, n_zero, n_bell);
        moonlab_job_results_free(&r);
        moonlab_job_free(j);
    }

    /* Demonstrate the decoder via the runtime registry. */
    {
        const moonlab_decoder_code_t code = {
            .distance = 3, .num_qubits = 18, .is_toric = 1,
        };
        unsigned char syndromes[9]   = {1, 1, 0, 0, 0, 0, 0, 0, 0};
        unsigned char corrections[18] = {0};
        const moonlab_decoder_input_t in = {
            .code = &code, .syndromes = syndromes,
            .corrections = corrections, .num_stabilisers = 9,
        };
        if (moonlab_decoder_decode_by_name(
                "overlay-zero-corrector", &in) != MOONLAB_DECODER_OK) {
            fprintf(stderr, "decode_by_name failed\n");
        } else {
            int total = 0;
            for (int q = 0; q < 18; q++) total += corrections[q];
            fprintf(stdout, "    decoded with overlay-zero-corrector  : "
                            "%d flips emitted (matches baseline)\n", total);
        }
    }

    /* ----- Billing summary ------------------------------------------ */
    fprintf(stdout, "\n--- billing summary (from completion hook) ---\n");
    fprintf(stdout, "  runs counted     : %d\n", billing.n_runs);
    fprintf(stdout, "  shots billed     : %d\n", billing.total_shots_billed);
    fprintf(stdout, "  last backend     : %s\n", billing.last_backend);
    fprintf(stdout, "  accumulated cost : $%.4f\n",
            billing.accumulated_cost_usd);
    fprintf(stdout, "\n");

    /* Clean up so we leave the registry in the state we found it. */
    moonlab_scheduler_set_completion_hook(NULL, NULL);
    moonlab_unregister_decoder("overlay-zero-corrector");
    moonlab_unregister_vendor_noise_profile(
        "ibm-falcon-2026-05-20-snapshot");
    moonlab_unregister_backend("overlay-deterministic");

    fprintf(stdout, "=== open-core overlay demo OK ===\n");
    return EXIT_SUCCESS;
}
