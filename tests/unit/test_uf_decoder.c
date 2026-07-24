/**
 * @file test_uf_decoder.c
 * @brief Unit test for the union-find decoder.
 *
 * Pins the two structural properties that were wrong when the decoder was
 * first written, both of which produced a decoder that ran but corrected
 * badly:
 *
 *  1. Growth is in HALF-edges.  With unit edge lengths a single increment
 *     closes any edge, so every edge incident to a defect completes in the
 *     same round and a defect pair is as likely to be routed to the boundary
 *     as to its partner -- decoding a two-defect syndrome as two boundary
 *     corrections and flipping the observable.
 *
 *  2. Each boundary edge has its OWN virtual boundary node.  With a single
 *     shared node the boundary is a hub: clusters on opposite sides of the
 *     code merge through it and peeling may route a defect across the whole
 *     patch, which gets worse as the code grows.
 */
#include "../../src/qec/uf_decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } \
} while (0)

/* Repetition code: D0 -- boundary (flips observable), D0 -- D1 (flips
 * nothing), D1 -- boundary (flips nothing).  Every syndrome has one correct
 * answer, so this is checkable by hand. */
static int test_repetition_chain(void) {
    const uint32_t ea[3] = {0, 0, 1};
    const uint32_t eb[3] = {MOONLAB_UF_BOUNDARY, 1, MOONLAB_UF_BOUNDARY};
    const double   ew[3] = {1.0, 1.0, 1.0};
    const uint64_t eo[3] = {1, 0, 0};

    moonlab_uf_decoder_t* d = moonlab_uf_decoder_new(2, 1, ea, eb, ew, eo, 3);
    CHECK(d, "decoder construction");

    /* four shots, one per syndrome, detector-major */
    enum { SHOTS = 4 };
    uint8_t det[2 * SHOTS] = {
        /* D0 over shots */ 0, 1, 1, 0,
        /* D1 over shots */ 0, 0, 1, 1,
    };
    const uint8_t want[SHOTS] = {0, 1, 0, 0};
    uint8_t out[SHOTS] = {0};

    long rc = moonlab_uf_decode_batch(d, det, SHOTS, 1, out);
    CHECK(rc == (long)SHOTS, "decode_batch return");
    for (int s = 0; s < SHOTS; s++) {
        if (out[s] != want[s]) {
            fprintf(stderr, "FAIL shot %d: obs %u, expected %u "
                            "(a paired syndrome routed to the boundary means "
                            "growth is not in half-edges)\n",
                    s, out[s], want[s]);
            moonlab_uf_decoder_free(d);
            return 1;
        }
    }
    moonlab_uf_decoder_free(d);
    return 0;
}

/* Two independent defect pairs far apart, each pair joined by a zero
 * observable edge and each also adjacent to the boundary through an
 * observable-flipping edge.  A shared boundary node lets the two clusters
 * merge and peel through each other; distinct boundary nodes keep them
 * separate and the correct answer is obs = 0. */
static int test_boundary_is_not_a_hub(void) {
    /* chain A: D0 -- D1 ; chain B: D2 -- D3 ; each Di also -- boundary */
    const uint32_t ea[6] = {0, 2, 0, 1, 2, 3};
    const uint32_t eb[6] = {1, 3,
                            MOONLAB_UF_BOUNDARY, MOONLAB_UF_BOUNDARY,
                            MOONLAB_UF_BOUNDARY, MOONLAB_UF_BOUNDARY};
    const double   ew[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const uint64_t eo[6] = {0, 0, 1, 1, 1, 1};

    moonlab_uf_decoder_t* d = moonlab_uf_decoder_new(4, 1, ea, eb, ew, eo, 6);
    CHECK(d, "decoder construction");

    enum { SHOTS = 1 };
    uint8_t det[4 * SHOTS] = {1, 1, 1, 1};   /* all four detectors lit */
    uint8_t out[SHOTS] = {0};
    long rc = moonlab_uf_decode_batch(d, det, SHOTS, 1, out);
    CHECK(rc == (long)SHOTS, "decode_batch return");
    CHECK(out[0] == 0, "two local pairs must decode to obs 0; a shared "
                       "boundary node lets them peel through the boundary");
    moonlab_uf_decoder_free(d);
    return 0;
}

/* An empty syndrome must produce no correction, and the multithreaded path
 * must agree with the single-threaded one shot for shot. */
static int test_empty_and_threading(void) {
    const uint32_t ea[3] = {0, 0, 1};
    const uint32_t eb[3] = {MOONLAB_UF_BOUNDARY, 1, MOONLAB_UF_BOUNDARY};
    const double   ew[3] = {1.0, 1.0, 1.0};
    const uint64_t eo[3] = {1, 0, 0};
    moonlab_uf_decoder_t* d = moonlab_uf_decoder_new(2, 1, ea, eb, ew, eo, 3);
    CHECK(d, "decoder construction");

    enum { SHOTS = 512 };
    uint8_t* det = (uint8_t*)calloc(2 * SHOTS, 1);
    uint8_t* o1  = (uint8_t*)calloc(SHOTS, 1);
    uint8_t* oN  = (uint8_t*)calloc(SHOTS, 1);
    CHECK(det && o1 && oN, "alloc");
    for (int s = 0; s < SHOTS; s++) {          /* deterministic pattern */
        det[0 * SHOTS + s] = (uint8_t)(s & 1);
        det[1 * SHOTS + s] = (uint8_t)((s >> 1) & 1);
    }
    CHECK(moonlab_uf_decode_batch(d, det, SHOTS, 1, o1) == (long)SHOTS, "1T decode");
    CHECK(moonlab_uf_decode_batch(d, det, SHOTS, 0, oN) == (long)SHOTS, "MT decode");
    for (int s = 0; s < SHOTS; s++) {
        if (o1[s] != oN[s]) {
            fprintf(stderr, "FAIL shot %d: 1T %u vs MT %u -- decoding is "
                            "per-shot independent and must not depend on the "
                            "thread count\n", s, o1[s], oN[s]);
            free(det); free(o1); free(oN); moonlab_uf_decoder_free(d);
            return 1;
        }
    }
    /* shot 0 has an empty syndrome */
    CHECK(o1[0] == 0, "empty syndrome must yield no correction");
    free(det); free(o1); free(oN);
    moonlab_uf_decoder_free(d);
    return 0;
}

/* Correlated two-pass decoding, checkable by hand.
 *
 * One physical mechanism has components A = D0--D1 (edge 0) and
 * B = D2--D3 (edge 1), joint probability q = 5e-4.  Edge A also has an
 * independent source of the same size, so p_A = 1e-3; edge B has almost no
 * independent source (2e-5), so its marginal is ~5.2e-4 and its weight
 * ~7.56.  D2 and D3 each sit next to the boundary through weight-3.5
 * edges, of which only the D2 one flips the observable.
 *
 * Syndrome {D0,D1,D2,D3}: the mechanism fired.  The truthful correction is
 * A + B (observable 0).  Uncorrelated decoding pairs D0-D1 through A
 * (6.91 < 18) but routes D2 and D3 to the boundary (3.5 + 3.5 = 7.0 <
 * 7.56), flipping the observable once: PROVABLY WRONG -- the boundary
 * explanation has probability ~p_A_only * 0.029^2 ~ 4e-7, a thousandfold
 * less likely than the mechanism's q = 5e-4.  With the boost, pass 1's use
 * of A conditions B: P(B|A) = r(1-q_B) + (1-r)q_B ~ 0.4998 (r ~ 0.4998
 * because A's sources split evenly between joint and independent), so B's
 * pass-2 weight drops to ~1e-3 and pass 2 pairs D2-D3 through it: obs 0.
 *
 * Syndrome {D2,D3} alone must stay uncorrelated (obs 1): A was not used,
 * so nothing conditions B.  Syndrome {D0,D1} must decode to 0 either way. */
static int test_correlated_partner_boost(void) {
    const double q       = 5e-4;    /* joint mechanism probability */
    const double pa      = 1e-3;    /* edge A marginal */
    const double b_only  = 2e-5;    /* edge B independent source */
    const double pb      = q * (1 - b_only) + b_only * (1 - q);
    const double p35     = 1.0 / (1.0 + exp(3.5));
    const double p90     = 1.0 / (1.0 + exp(9.0));

    const uint32_t ea[6] = {0, 2, 0, 1, 2, 3};
    const uint32_t eb[6] = {1, 3,
                            MOONLAB_UF_BOUNDARY, MOONLAB_UF_BOUNDARY,
                            MOONLAB_UF_BOUNDARY, MOONLAB_UF_BOUNDARY};
    const double   ew[6] = {log((1 - pa) / pa), log((1 - pb) / pb),
                            9.0, 9.0, 3.5, 3.5};
    const uint64_t eo[6] = {0, 0, 1, 1, 1, 0};
    const double   ep[6] = {pa, pb, p90, p90, p35, p35};
    const uint32_t ca[1] = {0};
    const uint32_t cb[1] = {1};
    const double   cq[1] = {5e-4};

    enum { SHOTS = 3 };
    /* shot 0: all four; shot 1: D2,D3 only; shot 2: D0,D1 only */
    uint8_t det[4 * SHOTS] = {
        /* D0 */ 1, 0, 1,
        /* D1 */ 1, 0, 1,
        /* D2 */ 1, 1, 0,
        /* D3 */ 1, 1, 0,
    };
    const uint8_t want_plain[SHOTS] = {1, 1, 0};
    const uint8_t want_corr[SHOTS]  = {0, 1, 0};
    uint8_t out[SHOTS];

    /* The boost DISABLED (plain decoder) must get shot 0 wrong: this pins
     * that the toy actually discriminates, so a pass-2 regression cannot
     * hide behind an accidentally-easy syndrome. */
    moonlab_uf_decoder_t* d = moonlab_uf_decoder_new(4, 1, ea, eb, ew, eo, 6);
    CHECK(d, "plain decoder construction");
    CHECK(moonlab_uf_decode_batch(d, det, SHOTS, 1, out) == (long)SHOTS,
          "plain decode");
    for (int s = 0; s < SHOTS; s++) {
        if (out[s] != want_plain[s]) {
            fprintf(stderr, "FAIL plain shot %d: obs %u, expected %u\n",
                    s, out[s], want_plain[s]);
            moonlab_uf_decoder_free(d);
            return 1;
        }
    }
    moonlab_uf_decoder_free(d);

    /* The boost ENABLED must fix shot 0 and leave the others alone. */
    d = moonlab_uf_decoder_new_correlated(4, 1, ea, eb, ew, eo, 6,
                                          ep, ca, cb, cq, 1);
    CHECK(d, "correlated decoder construction");
    CHECK(moonlab_uf_decode_batch(d, det, SHOTS, 1, out) == (long)SHOTS,
          "correlated decode");
    for (int s = 0; s < SHOTS; s++) {
        if (out[s] != want_corr[s]) {
            fprintf(stderr, "FAIL correlated shot %d: obs %u, expected %u "
                            "(shot 0 wrong means the partner boost did not "
                            "fire; shot 1 wrong means it fired without its "
                            "trigger)\n",
                    s, out[s], want_corr[s]);
            moonlab_uf_decoder_free(d);
            return 1;
        }
    }
    moonlab_uf_decoder_free(d);
    return 0;
}

int main(void) {
    if (test_repetition_chain()     != 0) return 1;
    fprintf(stderr, "PASS test_repetition_chain\n");
    if (test_boundary_is_not_a_hub()!= 0) return 1;
    fprintf(stderr, "PASS test_boundary_is_not_a_hub\n");
    if (test_empty_and_threading()  != 0) return 1;
    fprintf(stderr, "PASS test_empty_and_threading\n");
    if (test_correlated_partner_boost() != 0) return 1;
    fprintf(stderr, "PASS test_correlated_partner_boost\n");
    return 0;
}
