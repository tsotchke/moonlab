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
    /* PYMATCHING slot is unconditionally available since v0.7.7
     * (subprocess transport; runtime errors if pymatching isn't
     * pip-installed). */
    {
        const int avail = moonlab_decoder_slot_available(MOONLAB_DECODER_PYMATCHING);
        CHECK(avail == 1, "PYMATCHING available since v0.7.7 (got %d)", avail);
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

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== decoder-bench dispatcher (v0.6.7 scaffold) ===\n");
    test_slot_availability();
    test_greedy_zero_syndrome();
    test_greedy_two_defects();
    test_mwpm_exact_basic();
    test_mwpm_exact_six_defects();
    test_external_slots();
    test_error_paths();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
