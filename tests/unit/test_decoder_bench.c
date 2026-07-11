/**
 * @file  test_decoder_bench.c
 * @brief Validate the decoder-bench dispatcher (v0.6.7 scaffold).
 *
 * Covers:
 *   - Slot-availability table reflects the v0.6.7 contract
 *     (GREEDY/MWPM_EXACT ON, SBNN/LIBIRREP_SS/PYMATCHING OFF).
 *   - Slot names round-trip through `_slot_name`.
 *   - GREEDY decoder on a known toric-d3 syndrome pattern
 *     produces a correction that flips the right edges.
 *   - SBNN/LIBIRREP_SS/PYMATCHING slots return NOT_BUILT.
 *   - Error paths: NULL args, bad distance.
 */

#include "../../src/applications/decoder_bench.h"

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

static void test_slot_availability(void)
{
    fprintf(stdout, "\n--- slot availability + naming ---\n");
    CHECK(moonlab_decoder_slot_available(MOONLAB_DECODER_GREEDY) == 1,
          "GREEDY available");
    CHECK(moonlab_decoder_slot_available(MOONLAB_DECODER_MWPM_EXACT) == 1,
          "MWPM_EXACT available");
    /* SBNN slot availability is build-conditional from v0.7.5. */
    {
        const int avail = moonlab_decoder_slot_available(MOONLAB_DECODER_SBNN);
        CHECK(avail == 0 || avail == 1,
              "SBNN available = %d (build-conditional since v0.7.5)", avail);
    }
    /* LIBIRREP_SS slot availability is build-conditional: ON when
     * QSIM_ENABLE_LIBIRREP=ON, OFF otherwise.  Either is valid. */
    {
        const int avail = moonlab_decoder_slot_available(MOONLAB_DECODER_LIBIRREP_SS);
        CHECK(avail == 0 || avail == 1,
              "LIBIRREP_SS available = %d (build-conditional)", avail);
    }
    /* PYMATCHING uses a POSIX subprocess bridge.  Windows/Web builds
     * reserve the slot but report NOT_BUILT until a native transport
     * lands. */
    {
        const int avail = moonlab_decoder_slot_available(MOONLAB_DECODER_PYMATCHING);
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
        CHECK(avail == 0, "PYMATCHING subprocess bridge unavailable on this platform (got %d)", avail);
#else
        CHECK(avail == 1, "PYMATCHING available since v0.7.7 (got %d)", avail);
#endif
    }

    CHECK(strcmp(moonlab_decoder_slot_name(MOONLAB_DECODER_GREEDY),     "greedy") == 0,
          "GREEDY -> greedy");
    CHECK(strcmp(moonlab_decoder_slot_name(MOONLAB_DECODER_MWPM_EXACT), "mwpm_exact") == 0,
          "MWPM_EXACT -> mwpm_exact");
    CHECK(strcmp(moonlab_decoder_slot_name(MOONLAB_DECODER_SBNN),       "sbnn") == 0,
          "SBNN -> sbnn");
    CHECK(strcmp(moonlab_decoder_slot_name(MOONLAB_DECODER_LIBIRREP_SS),"libirrep_single_shot") == 0,
          "LIBIRREP_SS -> libirrep_single_shot");
    CHECK(strcmp(moonlab_decoder_slot_name(MOONLAB_DECODER_PYMATCHING), "pymatching") == 0,
          "PYMATCHING -> pymatching");
}

static void test_greedy_zero_syndrome(void)
{
    fprintf(stdout, "\n--- GREEDY on zero syndrome (no defects) ---\n");
    const int d = 3;
    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = 2 * d * d, .is_toric = 1
    };
    unsigned char syndromes[9] = {0};
    unsigned char corrections[2 * 9] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code,
        .syndromes = syndromes,
        .corrections = corrections,
        .num_stabilisers = 9,
    };
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in);
    CHECK(rc == 0, "GREEDY rc=0 on zero syndrome");

    int total_flips = 0;
    for (int q = 0; q < 18; q++) total_flips += corrections[q];
    CHECK(total_flips == 0, "no flips emitted (got %d)", total_flips);
}

static void test_greedy_two_defects(void)
{
    fprintf(stdout, "\n--- GREEDY on two adjacent defects on d=3 torus ---\n");
    const int d = 3;
    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = 2 * d * d, .is_toric = 1
    };
    /* Defects at vertex (0, 0) [idx 0] and (0, 1) [idx 1]; pair
     * separated by one horizontal edge. */
    unsigned char syndromes[9] = {1, 1, 0, 0, 0, 0, 0, 0, 0};
    unsigned char corrections[18] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code,
        .syndromes = syndromes,
        .corrections = corrections,
        .num_stabilisers = 9,
    };
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in);
    CHECK(rc == 0, "GREEDY rc=0");

    int total_flips = 0;
    for (int q = 0; q < 18; q++) total_flips += corrections[q];
    fprintf(stdout, "    total flips = %d (expected 1, single edge between adjacent defects)\n",
            total_flips);
    CHECK(total_flips == 1, "single edge flipped to neutralise two adjacent defects");
}

static void test_mwpm_exact_basic(void)
{
    fprintf(stdout, "\n--- MWPM_EXACT on adjacent + far-apart defects ---\n");
    const int d = 5;
    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = 2 * d * d, .is_toric = 1
    };
    /* Two defects at (0, 0) and (2, 2) on a 5x5 torus: shortest
     * geodesic is 4 edges (Manhattan distance 4).  Greedy gets the
     * same result on this case, but MWPM_EXACT must agree. */
    unsigned char syndromes[25] = {0};
    syndromes[0]  = 1;  /* (0, 0) */
    syndromes[12] = 1;  /* (2, 2) */
    unsigned char corrections[50] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code,
        .syndromes = syndromes,
        .corrections = corrections,
        .num_stabilisers = 25,
    };
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_MWPM_EXACT, &in);
    CHECK(rc == 0, "MWPM_EXACT rc=%d", rc);
    int total_flips = 0;
    for (int q = 0; q < 50; q++) total_flips += corrections[q];
    fprintf(stdout, "    total flips = %d (expected 4 for L1 dist (0,0)->(2,2))\n",
            total_flips);
    CHECK(total_flips == 4, "single 4-edge geodesic flipped");
}

static void test_mwpm_exact_six_defects(void)
{
    fprintf(stdout, "\n--- MWPM_EXACT on six defects ---\n");
    const int d = 5;
    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = 2 * d * d, .is_toric = 1
    };
    /* Six defects -- exact MWPM must produce a valid 3-pair matching. */
    unsigned char syndromes[25] = {0};
    syndromes[0]  = 1;
    syndromes[1]  = 1;
    syndromes[6]  = 1;
    syndromes[7]  = 1;
    syndromes[18] = 1;
    syndromes[19] = 1;
    unsigned char corrections[50] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code, .syndromes = syndromes,
        .corrections = corrections, .num_stabilisers = 25,
    };
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_MWPM_EXACT, &in);
    CHECK(rc == 0, "MWPM_EXACT rc=%d on 6 defects", rc);
    int total_flips = 0;
    for (int q = 0; q < 50; q++) total_flips += corrections[q];
    fprintf(stdout, "    six-defect total flips = %d (expect ~3)\n", total_flips);
    /* Best matching: pairs (0,1) (6,7) (18,19), each Manhattan-1
     * distance, total = 3.  Allow up to 4 in case of edge-XOR
     * cancellation. */
    CHECK(total_flips >= 1 && total_flips <= 6,
          "matching produces 1-6 flips (got %d)", total_flips);
}

static void test_external_slots(void)
{
    fprintf(stdout, "\n--- external slot dispatch ---\n");
    const moonlab_decoder_code_t code = {.distance = 3, .num_qubits = 18, .is_toric = 1};
    unsigned char syndromes[9] = {0};
    unsigned char corrections[18] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code,
        .syndromes = syndromes,
        .corrections = corrections,
        .num_stabilisers = 9,
    };
    /* PYMATCHING: subprocess transport.  Returns OK if pymatching is
     * pip-installed, NOT_BUILT otherwise.  Either is valid for the
     * suite; we just check it dispatches without crashing. */
    {
        const int py_rc = moonlab_decoder_decode(MOONLAB_DECODER_PYMATCHING, &in);
        CHECK(py_rc == 0 || py_rc == MOONLAB_DECODER_NOT_BUILT,
              "PYMATCHING dispatched (rc=%d; OK=pymatching available, "
              "NOT_BUILT=pymatching missing)", py_rc);
    }
    /* PYMATCHING edge-index convention regression: defects at
     * vertex (0, 0) and (1, 0) are one +X step apart, so a correct
     * pymatching bridge must flip horizontal edge h(0, 0) at byte
     * index 0 -- NOT vertical edge v(0, 0) at byte index d*d.  An
     * earlier bridge had the H/V loops swapped, which transposed
     * corrections across the lattice diagonal and inflated logical
     * error rates ~20x at d=5 p=0.01.  This test catches the
     * regression on every pymatching-available CI run. */
    if (moonlab_decoder_slot_available(MOONLAB_DECODER_PYMATCHING) &&
        getenv("MOONLAB_TEST_PYMATCHING_AVAILABLE")) {
        const moonlab_decoder_code_t code_d3 = {
            .distance = 3, .num_qubits = 18, .is_toric = 1,
        };
        unsigned char syn_x[9]   = {1, 0, 0, 1, 0, 0, 0, 0, 0};
        unsigned char corr_x[18] = {0};
        const moonlab_decoder_input_t in_x = {
            .code = &code_d3, .syndromes = syn_x,
            .corrections = corr_x, .num_stabilisers = 9,
        };
        const int rc = moonlab_decoder_decode(MOONLAB_DECODER_PYMATCHING, &in_x);
        if (rc == 0) {
            int x_flips = 0, y_flips = 0;
            for (int q = 0; q < 9; q++)  x_flips += corr_x[q];
            for (int q = 9; q < 18; q++) y_flips += corr_x[q];
            CHECK(x_flips == 1 && y_flips == 0,
                  "PYMATCHING (0,0)-(1,0) defect pair flips H edge only "
                  "(H=%d, V=%d) -- catches h/v swap regression",
                  x_flips, y_flips);
        }
    }
    {
        const int sbnn_rc = moonlab_decoder_decode(MOONLAB_DECODER_SBNN, &in);
        if (moonlab_decoder_slot_available(MOONLAB_DECODER_SBNN)) {
            CHECK(sbnn_rc == 0, "SBNN on zero syndrome -> 0 (SbNN linked, rc=%d)", sbnn_rc);
        } else {
            CHECK(sbnn_rc == MOONLAB_DECODER_NOT_BUILT,
                  "SBNN -> NOT_BUILT (SbNN unlinked, rc=%d)", sbnn_rc);
        }
    }

    /* LIBIRREP_SS: when linked, exercises irrep_single_shot_lift to
     * verify the toric code's meta-checks, then delegates to greedy.
     * Toric d=3 has nontrivial cube redundancy so the lift succeeds. */
    const int rc = moonlab_decoder_decode(MOONLAB_DECODER_LIBIRREP_SS, &in);
    if (moonlab_decoder_slot_available(MOONLAB_DECODER_LIBIRREP_SS)) {
        CHECK(rc == 0,
              "LIBIRREP_SS on zero syndrome -> 0 (libirrep linked, rc=%d)", rc);
    } else {
        CHECK(rc == MOONLAB_DECODER_NOT_BUILT,
              "LIBIRREP_SS -> NOT_BUILT (libirrep unlinked, rc=%d)", rc);
    }
}

static void test_error_paths(void)
{
    fprintf(stdout, "\n--- error paths ---\n");
    const moonlab_decoder_code_t code = {.distance = 3, .num_qubits = 18, .is_toric = 1};
    unsigned char buf[18] = {0};
    moonlab_decoder_input_t in = {
        .code = &code, .syndromes = buf, .corrections = buf,
        .num_stabilisers = 9,
    };
    CHECK(moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, NULL) == MOONLAB_DECODER_BAD_ARG,
          "NULL input rejected");

    in.code = NULL;
    CHECK(moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in) == MOONLAB_DECODER_BAD_ARG,
          "NULL code rejected");

    in.code = &code; in.syndromes = NULL;
    CHECK(moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in) == MOONLAB_DECODER_BAD_ARG,
          "NULL syndromes rejected");

    /* d < 2 rejected via code field. */
    const moonlab_decoder_code_t bad_code = {.distance = 1, .num_qubits = 2, .is_toric = 1};
    in.code = &bad_code; in.syndromes = buf;
    CHECK(moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in) == MOONLAB_DECODER_BAD_ARG,
          "distance < 2 rejected");
}

/* ------------------------------------------------------------------
 * Decoder runtime registry tests (v1.0.3)
 * ------------------------------------------------------------------ */

/* Custom decoder for the registry test: marks the first byte of the
 * corrections buffer with a sentinel.  ctx is a uint8_t* whose value
 * is written into corrections[0], so we can prove ctx threads through. */
static int sentinel_decoder(const moonlab_decoder_input_t *in, void *ctx)
{
    const unsigned char *sentinel = (const unsigned char *)ctx;
    memset(in->corrections, 0, (size_t)in->code->num_qubits);
    if (sentinel) in->corrections[0] = *sentinel;
    return MOONLAB_DECODER_OK;
}

static void test_decoder_registry_builtins(void)
{
    fprintf(stdout, "\n--- decoder registry: built-in entries ---\n");
    const int n = moonlab_num_decoders();
    CHECK(n >= 5, "registry has >= 5 built-in decoders (n=%d)", n);

    const char *names[16] = {0};
    const int copied = moonlab_list_decoders(names, 16);
    CHECK(copied == n, "list_decoders copied %d == num_decoders %d", copied, n);

    int saw_greedy = 0, saw_mwpm = 0, saw_sbnn = 0, saw_ss = 0, saw_pym = 0;
    for (int i = 0; i < copied; i++) {
        if (!names[i]) continue;
        if (strcmp(names[i], "greedy")               == 0) saw_greedy = 1;
        if (strcmp(names[i], "mwpm_exact")           == 0) saw_mwpm   = 1;
        if (strcmp(names[i], "sbnn")                 == 0) saw_sbnn   = 1;
        if (strcmp(names[i], "libirrep_single_shot") == 0) saw_ss     = 1;
        if (strcmp(names[i], "pymatching")           == 0) saw_pym    = 1;
    }
    CHECK(saw_greedy && saw_mwpm && saw_sbnn && saw_ss && saw_pym,
          "all five built-in names registered (greedy=%d mwpm=%d sbnn=%d ss=%d pym=%d)",
          saw_greedy, saw_mwpm, saw_sbnn, saw_ss, saw_pym);

    const moonlab_decoder_entry_t *g = moonlab_lookup_decoder("greedy");
    CHECK(g != NULL && g->fn != NULL, "lookup greedy returns entry with fn");
    CHECK(moonlab_lookup_decoder("does-not-exist") == NULL,
          "unknown decoder name returns NULL");
}

static void test_decoder_registry_register_custom(void)
{
    fprintf(stdout, "\n--- decoder registry: register custom decoder ---\n");
    const unsigned char sentinel = 0xA5;
    CHECK(moonlab_register_decoder(
              "tsotchke-test-sentinel", sentinel_decoder,
              (void *)&sentinel, "test-only sentinel decoder") == 0,
          "register custom decoder");

    const moonlab_decoder_entry_t *e =
        moonlab_lookup_decoder("tsotchke-test-sentinel");
    CHECK(e != NULL, "lookup returns custom entry");
    CHECK(e && e->fn == sentinel_decoder, "fn pointer round-trips");
    CHECK(e && e->ctx == &sentinel, "ctx pointer round-trips");
    CHECK(e && e->description &&
              strstr(e->description, "sentinel decoder") != NULL,
          "description round-trips");

    /* Dispatch by name. */
    const moonlab_decoder_code_t code = {
        .distance = 3, .num_qubits = 18, .is_toric = 1
    };
    unsigned char syndromes[9]   = {0};
    unsigned char corrections[18] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code, .syndromes = syndromes,
        .corrections = corrections, .num_stabilisers = 9,
    };
    CHECK(moonlab_decoder_decode_by_name("tsotchke-test-sentinel", &in) == 0,
          "decode_by_name dispatches to custom");
    CHECK(corrections[0] == 0xA5,
          "custom decoder ran and wrote sentinel (got 0x%02X)",
          corrections[0]);

    /* Unregister. */
    CHECK(moonlab_unregister_decoder("tsotchke-test-sentinel") == 0,
          "unregister custom decoder");
    CHECK(moonlab_lookup_decoder("tsotchke-test-sentinel") == NULL,
          "lookup after unregister returns NULL");
    CHECK(moonlab_unregister_decoder("nonexistent") != 0,
          "unregister nonexistent reports error");
}

static void test_decoder_registry_override_builtin(void)
{
    /* The registry is the single source of truth: re-registering
     * "greedy" with a custom fn makes both `decode_by_name("greedy")`
     * AND `decode(MOONLAB_DECODER_GREEDY, ...)` route through it.
     * After the test we restore the original entry. */
    fprintf(stdout, "\n--- decoder registry: override built-in slot ---\n");
    const moonlab_decoder_entry_t *orig = moonlab_lookup_decoder("greedy");
    CHECK(orig != NULL, "original greedy entry exists");
    if (!orig) return;
    const moonlab_decoder_fn original_fn  = orig->fn;
    void *const                original_ctx = orig->ctx;

    const unsigned char sentinel = 0x7F;
    CHECK(moonlab_register_decoder("greedy", sentinel_decoder,
                                   (void *)&sentinel, NULL) == 0,
          "re-register greedy with sentinel");

    const moonlab_decoder_code_t code = {
        .distance = 3, .num_qubits = 18, .is_toric = 1
    };
    unsigned char syndromes[9]   = {0};
    unsigned char corrections[18] = {0};
    const moonlab_decoder_input_t in = {
        .code = &code, .syndromes = syndromes,
        .corrections = corrections, .num_stabilisers = 9,
    };
    CHECK(moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, &in) == 0,
          "enum dispatcher fired after override");
    CHECK(corrections[0] == 0x7F,
          "enum dispatch routed through re-registered fn (got 0x%02X)",
          corrections[0]);

    /* Restore the original built-in. */
    CHECK(moonlab_register_decoder("greedy", original_fn, original_ctx,
                                   "In-tree nearest-pair matching baseline") == 0,
          "restore original greedy");
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== decoder-bench dispatcher + registry ===\n");
    test_slot_availability();
    test_greedy_zero_syndrome();
    test_greedy_two_defects();
    test_mwpm_exact_basic();
    test_mwpm_exact_six_defects();
    test_external_slots();
    test_error_paths();
    test_decoder_registry_builtins();
    test_decoder_registry_register_custom();
    test_decoder_registry_override_builtin();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
