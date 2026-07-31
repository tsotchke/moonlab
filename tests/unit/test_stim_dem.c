/**
 * @file test_stim_dem.c
 * @brief Unit test for the Stim detector-error-model reader and writer.
 *
 * No Stim dependency: every model here is hand written, and every expected
 * number is worked out by hand from the DEM semantics.
 *
 * The properties pinned here are the ones that silently corrupt a decoding
 * experiment when they are wrong:
 *
 *  1. `repeat` and `shift_detectors` resolve to ABSOLUTE detector indices,
 *     with the shift re-applied on every iteration.  Getting this wrong
 *     collapses distinct rounds onto the same detectors, which looks like a
 *     working decoder on a much easier problem.
 *
 *  2. Parallel mechanisms merge as p = p1(1-p2) + p2(1-p1), and a decomposed
 *     `^` mechanism yields both edges AND the correlation link between them.
 *     Dropping the link silently downgrades the correlated decoder to a
 *     matching decoder.
 *
 *  3. Export is the exact inverse of import: an edge list survives
 *     edges -> text -> parse -> edges unchanged, including the correlation
 *     links, which reference EDGE INDICES and so also pin the edge ordering.
 *
 *  4. Non-graphlike (>2 detector) and detectorless components are counted,
 *     never silently dropped, and an observable index the decoder's uint64_t
 *     mask cannot hold is a hard error rather than a wrapped bit.
 */
#include "../../src/qec/stim_dem.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } \
} while (0)

#define CHECK_NEAR(got, want, tol, msg) do { \
    double _g = (got), _w = (want); \
    if (!(fabs(_g - _w) <= (tol))) { \
        fprintf(stderr, "FAIL %s:%d: %s (got %.17g, want %.17g, delta %.3g)\n", \
                __FILE__, __LINE__, msg, _g, _w, fabs(_g - _w)); \
        return 1; \
    } \
} while (0)

/** XOR-merge of two independent flip probabilities. */
static double xor_merge(double a, double b) {
    return a * (1.0 - b) + b * (1.0 - a);
}

/* ------------------------------------------------------------------ */
/*  1. Basic parse: errors, coordinates, declarations                  */
/* ------------------------------------------------------------------ */
static int test_parse_basic(void) {
    static const char* text =
        "# a three edge repetition-code fragment\n"
        "detector(0, 0) D0\n"
        "detector(1, 0) D1\n"
        "logical_observable L0\n"
        "error(0.1) D0 L0        # left boundary edge\n"
        "error(0.2) D0 D1\n"
        "error(0.15) D1\n";

    moonlab_stim_error_t err;
    moonlab_dem_t* d = moonlab_dem_parse(text, &err);
    CHECK(d != NULL, "parse of a well formed model");
    CHECK(err.code == MOONLAB_STIM_OK, "status on success");
    CHECK(err.message[0] == '\0', "message cleared on success");

    CHECK(moonlab_dem_num_detectors(d) == 2, "num_detectors");
    CHECK(moonlab_dem_num_observables(d) == 1, "num_observables");
    CHECK(moonlab_dem_num_edges(d) == 3, "num_edges");
    CHECK(moonlab_dem_num_correlations(d) == 0, "num_correlations");
    CHECK(moonlab_dem_num_hyperedges(d) == 0, "num_hyperedges");
    CHECK(moonlab_dem_num_detectorless(d) == 0, "num_detectorless");

    uint32_t ea[3], eb[3];
    uint64_t eo[3];
    double ew[3], ep[3];
    long n = moonlab_dem_edges(d, ea, eb, ew, eo, ep, 3);
    CHECK(n == 3, "moonlab_dem_edges return");

    CHECK(ea[0] == 0 && eb[0] == MOONLAB_UF_BOUNDARY && eo[0] == 1u,
          "edge 0 is D0 -- boundary flipping L0");
    CHECK(ea[1] == 0 && eb[1] == 1 && eo[1] == 0u, "edge 1 is D0 -- D1");
    CHECK(ea[2] == 1 && eb[2] == MOONLAB_UF_BOUNDARY && eo[2] == 0u,
          "edge 2 is D1 -- boundary");
    CHECK_NEAR(ep[0], 0.1, 1e-15, "edge 0 probability");
    CHECK_NEAR(ep[1], 0.2, 1e-15, "edge 1 probability");
    CHECK_NEAR(ep[2], 0.15, 1e-15, "edge 2 probability");
    for (int i = 0; i < 3; i++)
        CHECK_NEAR(ew[i], log((1.0 - ep[i]) / ep[i]), 1e-12,
                   "edge weight is the log-likelihood ratio");

    /* A cap below the edge count must be reported, not truncated. */
    CHECK(moonlab_dem_edges(d, ea, eb, ew, eo, ep, 2) ==
              MOONLAB_STIM_ERR_OVERFLOW, "edge cap overflow");
    /* Column selection: any pointer may be NULL. */
    CHECK(moonlab_dem_edges(d, ea, NULL, NULL, NULL, NULL, 3) == 3,
          "partial edge fetch");

    double c[4];
    CHECK(moonlab_dem_detector_coords(d, 0, c, 4) == 2, "D0 coordinate count");
    CHECK_NEAR(c[0], 0.0, 0.0, "D0 x");
    CHECK_NEAR(c[1], 0.0, 0.0, "D0 y");
    CHECK(moonlab_dem_detector_coords(d, 1, c, 4) == 2, "D1 coordinate count");
    CHECK_NEAR(c[0], 1.0, 0.0, "D1 x");
    CHECK(moonlab_dem_detector_coords(d, 2, c, 4) == -1,
          "out-of-range detector index");

    moonlab_dem_free(d);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  2. Parallel mechanisms merge, case-insensitive names, tags         */
/* ------------------------------------------------------------------ */
static int test_parallel_merge(void) {
    static const char* text =
        "ERROR(0.1) D0 D1\n"
        "error[from-a-tag](0.2) D1 D0\n"   /* same edge, targets swapped   */
        "error(0.3) D0 D1 L0\n";           /* different observable mask    */

    moonlab_dem_t* d = moonlab_dem_parse(text, NULL);
    CHECK(d != NULL, "parse with mixed case and a tag");
    CHECK(moonlab_dem_num_edges(d) == 2,
          "the observable mask is part of the edge key");

    double ep[2];
    uint64_t eo[2];
    CHECK(moonlab_dem_edges(d, NULL, NULL, NULL, eo, ep, 2) == 2, "edges");
    CHECK(eo[0] == 0u && eo[1] == 1u, "observable masks");
    CHECK_NEAR(ep[0], xor_merge(0.1, 0.2), 1e-15,
               "parallel mechanisms merge as p1(1-p2) + p2(1-p1)");
    CHECK_NEAR(ep[0], 0.26, 1e-15, "0.1 xor 0.2 = 0.26");
    CHECK_NEAR(ep[1], 0.3, 1e-15, "unmerged edge");

    moonlab_dem_free(d);

    /* p outside (0, 1) is a no-op mechanism, dropped exactly as the
     * reference converter drops it -- but its detectors still count. */
    moonlab_dem_t* z = moonlab_dem_parse("error(0) D0 D1\n"
                                         "error(1) D2 D3\n", NULL);
    CHECK(z != NULL, "parse of no-op mechanisms");
    CHECK(moonlab_dem_num_edges(z) == 0, "p=0 and p=1 produce no edges");
    CHECK(moonlab_dem_num_detectors(z) == 4, "no-op mechanisms still declare "
                                             "their detectors");
    moonlab_dem_free(z);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  3. Nested repeat blocks and shift_detectors                        */
/* ------------------------------------------------------------------ */
static int test_repeat_and_shift(void) {
    /* Outer loop twice, inner loop three times.  The inner shift moves the
     * detector window by one per round, the outer by two more at the end of
     * each outer iteration, so the absolute indices are
     * (0,1) (1,2) (2,3) then (5,6) (6,7) (7,8). */
    static const char* text =
        "repeat 2 {\n"
        "    repeat 3 {\n"
        "        error(0.01) D0 D1\n"
        "        shift_detectors 1\n"
        "    }\n"
        "    shift_detectors 2\n"
        "}\n";

    moonlab_dem_t* d = moonlab_dem_parse(text, NULL);
    CHECK(d != NULL, "parse of nested repeat blocks");
    CHECK(moonlab_dem_num_edges(d) == 6, "six distinct edges");
    CHECK(moonlab_dem_num_detectors(d) == 9, "detector count after shifts");

    const uint32_t want_a[6] = {0, 1, 2, 5, 6, 7};
    const uint32_t want_b[6] = {1, 2, 3, 6, 7, 8};
    uint32_t ea[6], eb[6];
    double ep[6];
    CHECK(moonlab_dem_edges(d, ea, eb, NULL, NULL, ep, 6) == 6, "edges");
    for (int i = 0; i < 6; i++) {
        char msg[64];
        snprintf(msg, sizeof msg, "edge %d absolute detector indices", i);
        CHECK(ea[i] == want_a[i] && eb[i] == want_b[i], msg);
        CHECK_NEAR(ep[i], 0.01, 1e-15, "unmerged repeat edge probability");
    }
    moonlab_dem_free(d);

    /* The coordinate offset accumulates the same way, and is re-applied on
     * every iteration of the block that carries it. */
    static const char* ctext =
        "repeat 3 {\n"
        "    detector(0, 5) D0\n"
        "    shift_detectors(1, 0) 1\n"
        "}\n";
    moonlab_dem_t* c = moonlab_dem_parse(ctext, NULL);
    CHECK(c != NULL, "parse of shifted coordinates");
    CHECK(moonlab_dem_num_detectors(c) == 3, "three shifted detectors");
    for (unsigned i = 0; i < 3; i++) {
        double xy[2];
        CHECK(moonlab_dem_detector_coords(c, i, xy, 2) == 2, "coord count");
        CHECK_NEAR(xy[0], (double)i, 0.0, "x carries the accumulated shift");
        CHECK_NEAR(xy[1], 5.0, 0.0, "y is unshifted");
    }
    moonlab_dem_free(c);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  4. Decomposed '^' mechanisms become correlation links              */
/* ------------------------------------------------------------------ */
static int test_correlation_links(void) {
    static const char* text =
        "error(0.05) D0 D1 ^ D2 D3\n"
        "error(0.05) D0 D1 ^ D2 D3\n";

    moonlab_dem_t* d = moonlab_dem_parse(text, NULL);
    CHECK(d != NULL, "parse of a decomposed mechanism");
    CHECK(moonlab_dem_num_edges(d) == 2, "two components, two edges");
    CHECK(moonlab_dem_num_correlations(d) == 1, "one link between them");
    CHECK(moonlab_dem_num_hyperedges(d) == 0, "no hyperedges");

    double ep[2];
    CHECK(moonlab_dem_edges(d, NULL, NULL, NULL, NULL, ep, 2) == 2, "edges");
    CHECK_NEAR(ep[0], xor_merge(0.05, 0.05), 1e-15, "component 0 merged");
    CHECK_NEAR(ep[1], xor_merge(0.05, 0.05), 1e-15, "component 1 merged");

    uint32_t ca[1], cb[1];
    double cq[1];
    CHECK(moonlab_dem_correlations(d, ca, cb, cq, 1) == 1, "correlations");
    CHECK(ca[0] == 0 && cb[0] == 1, "link joins edges 0 and 1");
    CHECK_NEAR(cq[0], xor_merge(0.05, 0.05), 1e-15,
               "repeated links XOR-combine like parallel edges");
    moonlab_dem_free(d);

    /* Three components contribute all 3*2/2 pairwise links, sorted. */
    moonlab_dem_t* t =
        moonlab_dem_parse("error(0.02) D0 D1 ^ D2 D3 ^ D4 D5\n", NULL);
    CHECK(t != NULL, "parse of a three component mechanism");
    CHECK(moonlab_dem_num_edges(t) == 3, "three edges");
    CHECK(moonlab_dem_num_correlations(t) == 3, "C*(C-1)/2 links");
    uint32_t ta[3], tb[3];
    double tq[3];
    CHECK(moonlab_dem_correlations(t, ta, tb, tq, 3) == 3, "correlations");
    const uint32_t wa[3] = {0, 0, 1}, wb[3] = {1, 2, 2};
    for (int i = 0; i < 3; i++) {
        CHECK(ta[i] == wa[i] && tb[i] == wb[i],
              "links are sorted by (edge a, edge b)");
        CHECK_NEAR(tq[i], 0.02, 1e-15, "joint probability");
    }
    moonlab_dem_free(t);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  5. Hyperedges and detectorless components are counted              */
/* ------------------------------------------------------------------ */
static int test_hyperedges_counted(void) {
    static const char* text =
        "error(0.01) D0 D1 D2\n"      /* 3 detectors: not graphlike       */
        "error(0.01) L0\n"            /* no detectors: observable only    */
        "error(0.02) D0 D1 ^ L0\n"    /* one edge plus one bare component */
        "error(0) D7\n";              /* no-op, but declares D7           */

    moonlab_dem_t* d = moonlab_dem_parse(text, NULL);
    CHECK(d != NULL, "parse of a model with non-graphlike mechanisms");
    CHECK(moonlab_dem_num_hyperedges(d) == 1, "one hyperedge counted");
    CHECK(moonlab_dem_num_detectorless(d) == 2,
          "detectorless components counted separately from hyperedges");
    CHECK(moonlab_dem_num_edges(d) == 1, "only the graphlike component");
    CHECK(moonlab_dem_num_correlations(d) == 0,
          "a mechanism with one usable component contributes no link");
    CHECK(moonlab_dem_num_detectors(d) == 8, "D7 raises the detector count");
    CHECK(moonlab_dem_num_observables(d) == 1, "L0 raises the count");
    moonlab_dem_free(d);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  6. Observable index >= 64 does not fit the decoder's mask          */
/* ------------------------------------------------------------------ */
static int test_observable_limit(void) {
    moonlab_stim_error_t err;
    moonlab_dem_t* d = moonlab_dem_parse("error(0.001) D0 L64\n", &err);
    CHECK(d == NULL, "an observable index of 64 must be rejected");
    CHECK(err.code == MOONLAB_STIM_ERR_UNSUPPORTED, "status code");
    CHECK(err.line == 1, "line number");
    CHECK(strstr(err.message, "L64") != NULL, "message names the token");

    d = moonlab_dem_parse("logical_observable L200\n", &err);
    CHECK(d == NULL, "declaration beyond the mask is rejected too");
    CHECK(err.code == MOONLAB_STIM_ERR_UNSUPPORTED, "status code");
    CHECK(strstr(err.message, "L200") != NULL, "message names the token");

    /* L63 is the last index that fits. */
    d = moonlab_dem_parse("error(0.001) D0 L63\n", &err);
    CHECK(d != NULL, "L63 fits the mask");
    CHECK(moonlab_dem_num_observables(d) == 64, "64 observables");
    uint64_t eo[1];
    CHECK(moonlab_dem_edges(d, NULL, NULL, NULL, eo, NULL, 1) == 1, "edges");
    CHECK(eo[0] == ((uint64_t)1u << 63), "top observable bit");
    moonlab_dem_free(d);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  7. Every syntax error names a line and the offending token         */
/* ------------------------------------------------------------------ */
static int test_syntax_errors(void) {
    struct {
        const char* text;
        int         code;
        size_t      line;    /* 0 means "any nonzero line" */
        const char* needle;
    } cases[] = {
        {"error(0.1) X3\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "X3"},
        {"erorr(0.1) D0\n",
         MOONLAB_STIM_ERR_UNSUPPORTED, 1, "erorr"},
        {"error D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "exactly one probability"},
        {"error(0.1, 0.2) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "exactly one probability"},
        {"error(1.5) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "outside [0, 1]"},
        {"error(0.1) D\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "expected digits"},
        {"error(0.1) D0 D1x\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "D1x"},
        {"error(0.1) 4\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "must be D<n>"},
        {"error[unterminated(0.1) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "unterminated tag"},
        {"error(0.1) D0\n}\n",
         MOONLAB_STIM_ERR_SYNTAX, 2, "no 'repeat' block is open"},
        {"repeat 2 {\nerror(0.1) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 0, "unterminated 'repeat' block"},
        {"repeat 0 {\n}\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "at least 1"},
        {"repeat 2 error(0.1) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "unrecognised target"},
        {"detector(1, 2) L0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "only D<n> targets"},
        {"logical_observable D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "only L<n> targets"},
        {"logical_observable(1) L0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "no parenthesised arguments"},
        {"shift_detectors 1 2\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "exactly one integer detector offset"},
        {"error(0.1) D0 D1\n\nrepeat 2 {\n  error(0.2) D1\n  bogus D0\n}\n",
         MOONLAB_STIM_ERR_UNSUPPORTED, 5, "bogus"},
        {"error(0.5) D0 (\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "unrecognised target"},
        {"detector(1, ) D0\n",
         MOONLAB_STIM_ERR_SYNTAX, 1, "expected a number"},
    };

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        moonlab_stim_error_t err;
        memset(&err, 0, sizeof err);
        moonlab_dem_t* d = moonlab_dem_parse(cases[i].text, &err);
        if (d != NULL) {
            fprintf(stderr, "FAIL %s:%d: case %zu parsed but should not:\n%s",
                    __FILE__, __LINE__, i, cases[i].text);
            moonlab_dem_free(d);
            return 1;
        }
        if (err.code != cases[i].code) {
            fprintf(stderr, "FAIL %s:%d: case %zu code %d, expected %d (%s)\n",
                    __FILE__, __LINE__, i, err.code, cases[i].code,
                    err.message);
            return 1;
        }
        if (err.line == 0 ||
            (cases[i].line != 0 && err.line != cases[i].line)) {
            fprintf(stderr, "FAIL %s:%d: case %zu line %zu, expected %zu\n",
                    __FILE__, __LINE__, i, err.line, cases[i].line);
            return 1;
        }
        if (strstr(err.message, cases[i].needle) == NULL) {
            fprintf(stderr, "FAIL %s:%d: case %zu message '%s' does not name "
                            "'%s'\n",
                    __FILE__, __LINE__, i, err.message, cases[i].needle);
            return 1;
        }
    }

    /* A NULL model is a caller error, not a crash. */
    moonlab_stim_error_t err;
    CHECK(moonlab_dem_parse(NULL, &err) == NULL, "NULL text rejected");
    CHECK(err.code == MOONLAB_STIM_ERR_BAD_ARG, "NULL text status");
    CHECK(moonlab_dem_edges(NULL, NULL, NULL, NULL, NULL, NULL, 0) ==
              MOONLAB_STIM_ERR_BAD_ARG, "NULL model in edges()");
    CHECK(moonlab_dem_num_edges(NULL) == 0, "NULL model edge count");
    CHECK(moonlab_dem_make_uf_decoder(NULL, 0) == NULL, "NULL model decoder");
    CHECK(moonlab_dem_to_text(NULL) == NULL, "NULL model to_text");
    /* An empty model is legal and empty. */
    moonlab_dem_t* e = moonlab_dem_parse("\n  # nothing but a comment\n",
                                         &err);
    CHECK(e != NULL, "empty model parses");
    CHECK(moonlab_dem_num_edges(e) == 0 && moonlab_dem_num_detectors(e) == 0,
          "empty model is empty");
    char* et = moonlab_dem_to_text(e);
    CHECK(et != NULL && et[0] == '\0', "empty model serialises to empty text");
    moonlab_stim_text_free(et);
    moonlab_dem_free(e);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  8. Export is the exact inverse of import                           */
/* ------------------------------------------------------------------ */

/* A five detector repetition code with correlation links on three of its
 * edges, hand built so every residual after peeling stays positive. */
enum { RC_NDET = 5, RC_NOBS = 1, RC_NEDGE = 6, RC_NCORR = 3 };
static const uint32_t rc_ea[RC_NEDGE] = {0, 0, 1, 2, 3, 4};
static const uint32_t rc_eb[RC_NEDGE] = {MOONLAB_UF_BOUNDARY, 1, 2, 3, 4,
                                         MOONLAB_UF_BOUNDARY};
static const uint64_t rc_eo[RC_NEDGE] = {1, 0, 0, 0, 0, 1};
static const double   rc_ep[RC_NEDGE] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06};
static const uint32_t rc_ca[RC_NCORR] = {0, 1, 2};
static const uint32_t rc_cb[RC_NCORR] = {1, 2, 3};
static const double   rc_cq[RC_NCORR] = {0.005, 0.003, 0.004};

/** Compare a parsed model against the reference edge list.  Returns the
 *  largest absolute deviation, or -1.0 on a structural mismatch. */
static double rc_compare(const moonlab_dem_t* d) {
    if (moonlab_dem_num_detectors(d) != RC_NDET) return -1.0;
    if (moonlab_dem_num_observables(d) != RC_NOBS) return -1.0;
    if (moonlab_dem_num_edges(d) != RC_NEDGE) return -1.0;
    if (moonlab_dem_num_correlations(d) != RC_NCORR) return -1.0;

    uint32_t ea[RC_NEDGE], eb[RC_NEDGE];
    uint64_t eo[RC_NEDGE];
    double ep[RC_NEDGE], ew[RC_NEDGE];
    uint32_t ca[RC_NCORR], cb[RC_NCORR];
    double cq[RC_NCORR];
    if (moonlab_dem_edges(d, ea, eb, ew, eo, ep, RC_NEDGE) != RC_NEDGE)
        return -1.0;
    if (moonlab_dem_correlations(d, ca, cb, cq, RC_NCORR) != RC_NCORR)
        return -1.0;

    double worst = 0.0;
    for (size_t i = 0; i < RC_NEDGE; i++) {
        if (ea[i] != rc_ea[i] || eb[i] != rc_eb[i] || eo[i] != rc_eo[i])
            return -1.0;
        double dp = fabs(ep[i] - rc_ep[i]);
        if (dp > worst) worst = dp;
        double dw = fabs(ew[i] - log((1.0 - rc_ep[i]) / rc_ep[i]));
        if (dw > worst) worst = dw;
    }
    for (size_t i = 0; i < RC_NCORR; i++) {
        if (ca[i] != rc_ca[i] || cb[i] != rc_cb[i]) return -1.0;
        double dq = fabs(cq[i] - rc_cq[i]);
        if (dq > worst) worst = dq;
    }
    return worst;
}

static int test_export_roundtrip(void) {
    moonlab_stim_error_t err;
    char* text = moonlab_dem_text_from_edges(
        RC_NDET, RC_NOBS, rc_ea, rc_eb, rc_ep, rc_eo, RC_NEDGE,
        rc_ca, rc_cb, rc_cq, RC_NCORR, &err);
    CHECK(text != NULL, "export of a consistent edge list");
    CHECK(err.code == MOONLAB_STIM_OK, "export status");

    moonlab_dem_t* d = moonlab_dem_parse(text, &err);
    CHECK(d != NULL, "the exported text re-parses");
    double worst = rc_compare(d);
    CHECK(worst >= 0.0, "edges -> text -> parse preserves the edge list "
                        "structure and ordering");
    CHECK(worst <= 1e-12, "edges -> text -> parse is a fixed point to 1e-12");
    fprintf(stderr, "      from_edges round trip: max |delta| = %.3g\n", worst);

    /* Second generation: re-export what we just parsed and check it is
     * still the same model. */
    uint32_t ea[RC_NEDGE], eb[RC_NEDGE];
    uint64_t eo[RC_NEDGE];
    double ep[RC_NEDGE];
    uint32_t ca[RC_NCORR], cb[RC_NCORR];
    double cq[RC_NCORR];
    CHECK(moonlab_dem_edges(d, ea, eb, NULL, eo, ep, RC_NEDGE) == RC_NEDGE,
          "edges");
    CHECK(moonlab_dem_correlations(d, ca, cb, cq, RC_NCORR) == RC_NCORR,
          "correlations");
    char* text2 = moonlab_dem_text_from_edges(
        moonlab_dem_num_detectors(d), moonlab_dem_num_observables(d),
        ea, eb, ep, eo, RC_NEDGE, ca, cb, cq, RC_NCORR, &err);
    CHECK(text2 != NULL, "second export");
    moonlab_dem_t* d2 = moonlab_dem_parse(text2, &err);
    CHECK(d2 != NULL, "second re-parse");
    double worst2 = rc_compare(d2);
    CHECK(worst2 >= 0.0 && worst2 <= 1e-12,
          "the round trip stays a fixed point on a second pass");
    fprintf(stderr, "      second generation:     max |delta| = %.3g\n",
            worst2);

    /* moonlab_dem_to_text() replays the mechanisms instead of peeling, and
     * must land on the same model. */
    char* text3 = moonlab_dem_to_text(d);
    CHECK(text3 != NULL, "to_text");
    moonlab_dem_t* d3 = moonlab_dem_parse(text3, &err);
    CHECK(d3 != NULL, "to_text re-parses");
    double worst3 = rc_compare(d3);
    CHECK(worst3 >= 0.0 && worst3 <= 1e-12,
          "parse -> to_text -> parse is a fixed point");
    fprintf(stderr, "      to_text round trip:    max |delta| = %.3g\n",
            worst3);

    moonlab_stim_text_free(text);
    moonlab_stim_text_free(text2);
    moonlab_stim_text_free(text3);
    moonlab_dem_free(d);
    moonlab_dem_free(d2);
    moonlab_dem_free(d3);
    return 0;
}

/* to_text must also preserve what the matching-graph projection cannot
 * represent: hyperedges, detectorless mechanisms, coordinates and counts. */
static int test_to_text_preserves_everything(void) {
    static const char* text =
        "detector(1, 2, 0) D0\n"
        "detector(3, 4, 0) D2\n"
        "error(0.01) D0 D1 D2\n"
        "error(0.02) L0\n"
        "error(0.03) D0 D1 ^ D2 D3\n"
        "logical_observable L1\n"
        "shift_detectors 4\n"
        "error(0.04) D0 L0\n";

    moonlab_dem_t* a = moonlab_dem_parse(text, NULL);
    CHECK(a != NULL, "parse");
    char* out = moonlab_dem_to_text(a);
    CHECK(out != NULL, "to_text");
    moonlab_dem_t* b = moonlab_dem_parse(out, NULL);
    CHECK(b != NULL, "re-parse of to_text output");

    CHECK(moonlab_dem_num_detectors(b) == moonlab_dem_num_detectors(a),
          "detector count survives");
    CHECK(moonlab_dem_num_observables(b) == moonlab_dem_num_observables(a),
          "observable count survives, including an undisturbed L1");
    CHECK(moonlab_dem_num_edges(b) == moonlab_dem_num_edges(a),
          "edge count survives");
    CHECK(moonlab_dem_num_correlations(b) == moonlab_dem_num_correlations(a),
          "link count survives");
    CHECK(moonlab_dem_num_hyperedges(b) == moonlab_dem_num_hyperedges(a) &&
          moonlab_dem_num_hyperedges(a) == 1,
          "the non-graphlike mechanism is re-emitted, not dropped");
    CHECK(moonlab_dem_num_detectorless(b) == moonlab_dem_num_detectorless(a) &&
          moonlab_dem_num_detectorless(a) == 1,
          "the observable-only mechanism is re-emitted, not dropped");

    double ca[3], cb2[3];
    CHECK(moonlab_dem_detector_coords(a, 2, ca, 3) == 3, "source coords");
    CHECK(moonlab_dem_detector_coords(b, 2, cb2, 3) == 3, "round trip coords");
    for (int i = 0; i < 3; i++)
        CHECK_NEAR(cb2[i], ca[i], 0.0, "coordinates survive the round trip");

    /* to_text is idempotent from here on. */
    char* out2 = moonlab_dem_to_text(b);
    CHECK(out2 != NULL, "second to_text");
    CHECK(strcmp(out, out2) == 0, "to_text is idempotent");

    moonlab_stim_text_free(out);
    moonlab_stim_text_free(out2);
    moonlab_dem_free(a);
    moonlab_dem_free(b);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  9. Inconsistent inputs fail loudly rather than being clamped       */
/* ------------------------------------------------------------------ */
static int test_export_rejects_inconsistent(void) {
    const uint32_t ea[2] = {0, 1};
    const uint32_t eb[2] = {1, 2};
    const uint64_t eo[2] = {1, 0};
    const double   ep[2] = {0.001, 0.001};
    const uint32_t ca[1] = {0};
    const uint32_t cb[1] = {1};
    const double   cq[1] = {0.01};   /* larger than either edge probability */

    moonlab_stim_error_t err;
    char* t = moonlab_dem_text_from_edges(3, 1, ea, eb, ep, eo, 2,
                                          ca, cb, cq, 1, &err);
    CHECK(t == NULL, "a negative residual must not be clamped away");
    CHECK(err.code == MOONLAB_STIM_ERR_BAD_ARG, "status code");
    CHECK(strstr(err.message, "residual") != NULL,
          "message explains the residual");

    /* Other caller mistakes are caught with their own messages. */
    const double bad_p[2] = {0.0, 0.5};
    t = moonlab_dem_text_from_edges(3, 1, ea, eb, bad_p, eo, 2,
                                    NULL, NULL, NULL, 0, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "a zero edge probability is rejected");
    CHECK(strstr(err.message, "(0, 1)") != NULL, "message names the interval");

    const uint32_t rev_a[1] = {2};
    const uint32_t rev_b[1] = {1};
    const uint64_t rev_o[1] = {0};
    const double   rev_p[1] = {0.1};
    t = moonlab_dem_text_from_edges(3, 1, rev_a, rev_b, rev_p, rev_o, 1,
                                    NULL, NULL, NULL, 0, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "out-of-order endpoints are rejected");

    const uint32_t big_a[1] = {9};
    t = moonlab_dem_text_from_edges(3, 1, big_a, rev_b, rev_p, rev_o, 1,
                                    NULL, NULL, NULL, 0, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "a detector beyond num_detectors is rejected");

    t = moonlab_dem_text_from_edges(3, 65, ea, eb, ep, eo, 2,
                                    NULL, NULL, NULL, 0, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_UNSUPPORTED,
          "more than 64 observables is unsupported");

    const uint32_t dup_a[2] = {0, 1};
    const uint32_t dup_b[2] = {1, 0};
    const double   dup_q[2] = {0.0005, 0.0005};
    t = moonlab_dem_text_from_edges(3, 1, ea, eb, ep, eo, 2,
                                    dup_a, dup_b, dup_q, 2, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "a repeated link pair is rejected, not silently merged");
    CHECK(strstr(err.message, "pre-combine") != NULL,
          "message says how to fix it");

    const double half_q[1] = {0.5};
    t = moonlab_dem_text_from_edges(3, 1, ea, eb, ep, eo, 2,
                                    ca, cb, half_q, 1, &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "q = 0.5 has no inverse and is rejected");

    /* A mechanism with three or more components contributes to several links
     * at once, so peeling would subtract its probability more than once.  The
     * pairwise link list cannot express it and the export says so instead of
     * quietly emitting a different model.  moonlab_dem_to_text(), which
     * replays the mechanisms rather than peeling, handles it. */
    moonlab_dem_t* tri =
        moonlab_dem_parse("error(0.02) D0 D1 ^ D2 D3 ^ D4 D5\n", NULL);
    CHECK(tri != NULL, "parse of a three component mechanism");
    uint32_t ta[3], tb[3];
    uint64_t to[3];
    double tp[3];
    uint32_t la[3], lb[3];
    double lq[3];
    CHECK(moonlab_dem_edges(tri, ta, tb, NULL, to, tp, 3) == 3, "edges");
    CHECK(moonlab_dem_correlations(tri, la, lb, lq, 3) == 3, "links");
    t = moonlab_dem_text_from_edges(6, 1, ta, tb, tp, to, 3, la, lb, lq, 3,
                                    &err);
    CHECK(t == NULL && err.code == MOONLAB_STIM_ERR_BAD_ARG,
          "a three component mechanism cannot be rebuilt from pairwise links");
    CHECK(strstr(err.message, "residual") != NULL, "message explains why");
    char* replay = moonlab_dem_to_text(tri);
    CHECK(replay != NULL, "to_text handles what from_edges cannot");
    moonlab_dem_t* tri2 = moonlab_dem_parse(replay, NULL);
    CHECK(tri2 != NULL, "replayed text re-parses");
    CHECK(moonlab_dem_num_edges(tri2) == 3 &&
          moonlab_dem_num_correlations(tri2) == 3,
          "the three component mechanism survives to_text intact");
    moonlab_stim_text_free(replay);
    moonlab_dem_free(tri);
    moonlab_dem_free(tri2);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  9b. The round trip holds on randomly generated models              */
/* ------------------------------------------------------------------ */

/* Deterministic xorshift, so a failure here is always reproducible. */
static uint64_t rt_state = 0x243f6a8885a308d3ull;
static uint32_t rt_rand(void) {
    rt_state ^= rt_state << 13;
    rt_state ^= rt_state >> 7;
    rt_state ^= rt_state << 17;
    return (uint32_t)(rt_state >> 32);
}

/**
 * Generate small random models out of one and two component mechanisms --
 * the shapes a Stim decomposition produces -- and check that
 * edges -> text -> parse returns the identical arrays.  This is the guard on
 * the emission ORDER: an edge that lives only inside decomposed mechanisms
 * has no residual instruction to introduce it, so it has to be introduced by
 * a link, and getting that wrong permutes the edge indices the correlation
 * links refer to without changing any probability.
 */
static int test_export_roundtrip_stress(void) {
    double worst = 0.0;
    for (int trial = 0; trial < 400; trial++) {
        char text[8192];
        size_t len = 0;
        unsigned ndet = 3 + rt_rand() % 8;
        unsigned nmech = 1 + rt_rand() % 20;
        for (unsigned m = 0; m < nmech && len < sizeof text - 128; m++) {
            double p = 0.0005 + (rt_rand() % 10000) * 1e-5;
            len += (size_t)snprintf(text + len, sizeof text - len,
                                    "error(%.17g)", p);
            unsigned comps = 1 + rt_rand() % 2;
            for (unsigned c = 0; c < comps; c++) {
                if (c)
                    len += (size_t)snprintf(text + len, sizeof text - len,
                                            " ^");
                unsigned a = rt_rand() % ndet;
                len += (size_t)snprintf(text + len, sizeof text - len,
                                        " D%u", a);
                if (rt_rand() % 2) {
                    unsigned b = rt_rand() % ndet;
                    if (b != a)
                        len += (size_t)snprintf(text + len, sizeof text - len,
                                                " D%u", b);
                }
                if (rt_rand() % 3 == 0)
                    len += (size_t)snprintf(text + len, sizeof text - len,
                                            " L%u", rt_rand() % 2);
            }
            len += (size_t)snprintf(text + len, sizeof text - len, "\n");
        }
        snprintf(text + len, sizeof text - len,
                 "logical_observable L1\ndetector D%u\n", ndet - 1);

        moonlab_stim_error_t err;
        moonlab_dem_t* a = moonlab_dem_parse(text, &err);
        CHECK(a != NULL, "generated model parses");
        size_t ne = moonlab_dem_num_edges(a);
        size_t nc = moonlab_dem_num_correlations(a);
        if (ne == 0) { moonlab_dem_free(a); continue; }

        uint32_t ea[512], eb[512], ca2[2048], cb2[2048];
        uint64_t eo[512];
        double ep[512], cq[2048];
        if (ne > 512 || nc > 2048) { moonlab_dem_free(a); continue; }
        CHECK(moonlab_dem_edges(a, ea, eb, NULL, eo, ep, ne) == (long)ne,
              "edges");
        CHECK(moonlab_dem_correlations(a, ca2, cb2, cq, nc) == (long)nc,
              "correlations");

        char* out = moonlab_dem_text_from_edges(
            moonlab_dem_num_detectors(a), moonlab_dem_num_observables(a),
            ea, eb, ep, eo, ne, ca2, cb2, cq, nc, &err);
        if (!out) {
            fprintf(stderr, "FAIL trial %d export: [%d] %s\n%s",
                    trial, err.code, err.message, text);
            moonlab_dem_free(a);
            return 1;
        }
        moonlab_dem_t* b = moonlab_dem_parse(out, &err);
        CHECK(b != NULL, "exported text re-parses");
        CHECK(moonlab_dem_num_edges(b) == ne, "edge count preserved");
        CHECK(moonlab_dem_num_correlations(b) == nc, "link count preserved");
        CHECK(moonlab_dem_num_detectors(b) == moonlab_dem_num_detectors(a),
              "detector count preserved");
        CHECK(moonlab_dem_num_observables(b) == moonlab_dem_num_observables(a),
              "observable count preserved");

        uint32_t ea2[512], eb2[512], ca3[2048], cb3[2048];
        uint64_t eo2[512];
        double ep2[512], cq2[2048];
        CHECK(moonlab_dem_edges(b, ea2, eb2, NULL, eo2, ep2, ne) == (long)ne,
              "edges");
        CHECK(moonlab_dem_correlations(b, ca3, cb3, cq2, nc) == (long)nc,
              "correlations");
        for (size_t i = 0; i < ne; i++) {
            CHECK(ea2[i] == ea[i] && eb2[i] == eb[i] && eo2[i] == eo[i],
                  "edge identity and index order preserved");
            double dp = fabs(ep2[i] - ep[i]);
            if (dp > worst) worst = dp;
        }
        for (size_t i = 0; i < nc; i++) {
            CHECK(ca3[i] == ca2[i] && cb3[i] == cb2[i],
                  "correlation link endpoints preserved");
            double dq = fabs(cq2[i] - cq[i]);
            if (dq > worst) worst = dq;
        }
        moonlab_stim_text_free(out);
        moonlab_dem_free(a);
        moonlab_dem_free(b);
    }
    CHECK(worst <= 1e-12, "randomised round trips stay within 1e-12");
    fprintf(stderr, "      randomised round trips: max |delta| = %.3g\n",
            worst);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 10. The model builds a decoder that decodes                         */
/* ------------------------------------------------------------------ */
static int decode_planted(moonlab_uf_decoder_t* dec, const char* what) {
    /* Shot 0: D0 alone, which is the boundary edge that flips L0.
     * Shot 1: D1 and D2, which is the interior edge D1--D2, flipping
     * nothing.  Any other explanation is far heavier. */
    enum { SHOTS = 2 };
    uint8_t det[RC_NDET * SHOTS] = {
        /* D0 */ 1, 0,
        /* D1 */ 0, 1,
        /* D2 */ 0, 1,
        /* D3 */ 0, 0,
        /* D4 */ 0, 0,
    };
    uint8_t obs[SHOTS] = {0, 0};
    long rc = moonlab_uf_decode_batch(dec, det, SHOTS, 1, obs);
    if (rc != (long)SHOTS) {
        fprintf(stderr, "FAIL %s: decode_batch returned %ld\n", what, rc);
        return 1;
    }
    if (obs[0] != 1) {
        fprintf(stderr, "FAIL %s: a lone D0 must be corrected through the "
                        "observable-flipping boundary edge (got %u)\n",
                what, obs[0]);
        return 1;
    }
    if (obs[1] != 0) {
        fprintf(stderr, "FAIL %s: the D1--D2 pair must be corrected by their "
                        "shared edge (got %u)\n", what, obs[1]);
        return 1;
    }
    return 0;
}

static int test_make_decoder(void) {
    moonlab_stim_error_t err;
    char* text = moonlab_dem_text_from_edges(
        RC_NDET, RC_NOBS, rc_ea, rc_eb, rc_ep, rc_eo, RC_NEDGE,
        rc_ca, rc_cb, rc_cq, RC_NCORR, &err);
    CHECK(text != NULL, "export");
    moonlab_dem_t* d = moonlab_dem_parse(text, &err);
    moonlab_stim_text_free(text);
    CHECK(d != NULL, "parse");

    moonlab_uf_decoder_t* plain = moonlab_dem_make_uf_decoder(d, 0);
    CHECK(plain != NULL, "plain decoder construction");
    CHECK(moonlab_uf_decoder_num_edges(plain) == RC_NEDGE, "decoder edges");
    int rc = decode_planted(plain, "plain decoder");
    moonlab_uf_decoder_free(plain);
    if (rc) { moonlab_dem_free(d); return 1; }

    moonlab_uf_decoder_t* corr = moonlab_dem_make_uf_decoder(d, 1);
    CHECK(corr != NULL, "correlated decoder construction");
    CHECK(moonlab_uf_decoder_num_edges(corr) == RC_NEDGE, "decoder edges");
    rc = decode_planted(corr, "correlated decoder");
    moonlab_uf_decoder_free(corr);
    moonlab_dem_free(d);
    return rc;
}

/* ------------------------------------------------------------------ */
/* 11. Reading from disk                                               */
/* ------------------------------------------------------------------ */
static int test_parse_file(void) {
    static const char* text =
        "detector(0, 0) D0\n"
        "error(0.125) D0 D1 L0\n"
        "error(0.25) D1\n";
    static const char* path = "test_stim_dem_tmp.dem";

    FILE* f = fopen(path, "wb");
    CHECK(f != NULL, "creating the temporary model file");
    fputs(text, f);
    fclose(f);

    moonlab_stim_error_t err;
    moonlab_dem_t* d = moonlab_dem_parse_file(path, &err);
    remove(path);
    CHECK(d != NULL, "parse_file");
    CHECK(moonlab_dem_num_edges(d) == 2, "edges from file");
    CHECK(moonlab_dem_num_detectors(d) == 2, "detectors from file");
    double ep[2];
    CHECK(moonlab_dem_edges(d, NULL, NULL, NULL, NULL, ep, 2) == 2, "edges");
    CHECK_NEAR(ep[0], 0.125, 1e-15, "probability from file");
    moonlab_dem_free(d);

    d = moonlab_dem_parse_file("no_such_directory_xyz/no_such_file.dem", &err);
    CHECK(d == NULL, "a missing file is an error");
    CHECK(err.code == MOONLAB_STIM_ERR_IO, "IO status code");
    CHECK(strstr(err.message, "no_such_file.dem") != NULL,
          "message names the path");
    return 0;
}

int main(void) {
    if (test_parse_basic()                  != 0) return 1;
    fprintf(stderr, "PASS test_parse_basic\n");
    if (test_parallel_merge()               != 0) return 1;
    fprintf(stderr, "PASS test_parallel_merge\n");
    if (test_repeat_and_shift()             != 0) return 1;
    fprintf(stderr, "PASS test_repeat_and_shift\n");
    if (test_correlation_links()            != 0) return 1;
    fprintf(stderr, "PASS test_correlation_links\n");
    if (test_hyperedges_counted()           != 0) return 1;
    fprintf(stderr, "PASS test_hyperedges_counted\n");
    if (test_observable_limit()             != 0) return 1;
    fprintf(stderr, "PASS test_observable_limit\n");
    if (test_syntax_errors()                != 0) return 1;
    fprintf(stderr, "PASS test_syntax_errors\n");
    if (test_export_roundtrip()             != 0) return 1;
    fprintf(stderr, "PASS test_export_roundtrip\n");
    if (test_to_text_preserves_everything() != 0) return 1;
    fprintf(stderr, "PASS test_to_text_preserves_everything\n");
    if (test_export_rejects_inconsistent()  != 0) return 1;
    fprintf(stderr, "PASS test_export_rejects_inconsistent\n");
    if (test_export_roundtrip_stress()      != 0) return 1;
    fprintf(stderr, "PASS test_export_roundtrip_stress\n");
    if (test_make_decoder()                 != 0) return 1;
    fprintf(stderr, "PASS test_make_decoder\n");
    if (test_parse_file()                   != 0) return 1;
    fprintf(stderr, "PASS test_parse_file\n");
    return 0;
}
