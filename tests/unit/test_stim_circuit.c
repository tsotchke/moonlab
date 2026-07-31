/**
 * @file test_stim_circuit.c
 * @brief Unit test for the Stim `.stim` circuit reader / writer / sampler.
 *
 * No stim dependency: every expectation here is either a hand-computed
 * property of a hand-written circuit or an internal consistency check
 * (round-trip, sizing, rejection).  The exhaustive gate-table proof against
 * stim.Tableau.from_named_gate lives in the python test suite.
 *
 * Covers: every supported instruction parses and lowers; parens arguments,
 * tags and comments; nested REPEAT; SHIFT_COORDS accumulation across REPEAT
 * iterations; rec[-k] resolution per iteration; MPAD record placeholders;
 * text round-trip (semantic and byte-exact) on a surface-code-shaped
 * circuit; every documented rejection case producing the right code, a
 * nonzero line and a message naming the offending token; lowering sizes;
 * PAULI_CHANNEL_1(1,0,0) behaving exactly like X_ERROR(1.0);
 * PAULI_CHANNEL_2 sizing; noiseless detector sampling being all zero; a
 * p=1.0 X_ERROR lighting exactly the expected detector; and the
 * channel-op-without-a-table rejection in the Pauli-frame sampler.
 */

#include "../../src/qec/stim_circuit.h"
#include "../../src/backends/clifford/pauli_frame.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } \
} while (0)

/* ================================================================== */
/*  Fixtures                                                           */
/* ================================================================== */

/* Exercises every supported instruction, alias, tag, comment, parens
 * argument form, nested REPEAT, SHIFT_COORDS, MPAD and inverted target. */
static const char* KITCHEN_SINK =
    "# a leading comment\n"
    "QUBIT_COORDS(0, 0) 0\n"
    "QUBIT_COORDS(1, 0) 1\n"
    "QUBIT_COORDS(2, 0) 2\n"
    "QUBIT_COORDS(3, 0) 3\n"
    "R 0 1\n"
    "RZ 2\n"
    "RX 3\n"
    "RY 0\n"
    "TICK\n"
    "I 0\n"
    "X 0\n"
    "Y 1\n"
    "Z 2\n"
    "H[hadamard-tag] 0\n"
    "H_XZ 1\n"
    "H_XY 2\n"
    "H_YZ 3\n"
    "S 0\n"
    "SQRT_Z 1\n"
    "S_DAG 2\n"
    "SQRT_Z_DAG 3\n"
    "SQRT_X 0\n"
    "SQRT_X_DAG 1\n"
    "SQRT_Y 2\n"
    "SQRT_Y_DAG 3\n"
    "C_XYZ 0\n"
    "C_ZYX 1\n"
    "TICK\n"
    "CX 0 1\n"
    "CNOT 2 3\n"
    "ZCX 0 2\n"
    "CY 1 2\n"
    "ZCY 0 3\n"
    "CZ 0 1\n"
    "ZCZ 2 3\n"
    "XCX 0 1\n"
    "XCY 1 2\n"
    "XCZ 2 3\n"
    "YCX 0 3\n"
    "YCY 1 3\n"
    "YCZ 0 2\n"
    "SWAP 0 1\n"
    "ISWAP 1 2\n"
    "ISWAP_DAG 2 3\n"
    "CXSWAP 0 3\n"
    "SWAPCX 1 0\n"
    "CZSWAP 2 0\n"
    "SWAPCZ 3 1\n"
    "SQRT_XX 0 1\n"
    "SQRT_XX_DAG 1 2\n"
    "SQRT_YY 2 3\n"
    "SQRT_YY_DAG 3 0\n"
    "SQRT_ZZ 0 2\n"
    "SQRT_ZZ_DAG 1 3\n"
    "TICK    # trailing comment on a TICK\n"
    "X_ERROR(0.001) 0 1\n"
    "Y_ERROR(0.002) 2\n"
    "Z_ERROR(0.003) 3\n"
    "DEPOLARIZE1(0.004) 0 1 2 3\n"
    "DEPOLARIZE2(0.005) 0 1 2 3\n"
    "PAULI_CHANNEL_1(0.001, 0.002, 0.003) 0 1\n"
    "PAULI_CHANNEL_2(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, "
    "0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001) 0 1 2 3\n"
    "TICK\n"
    "M 0\n"
    "MZ 1\n"
    "MX 2\n"
    "MY 3\n"
    "MR 0\n"
    "MRZ 1\n"
    "MRX 2\n"
    "MRY 3\n"
    "M(0.01) 0\n"
    "M !1\n"
    "MPAD 0 1\n"
    "DETECTOR(0, 0) rec[-4] rec[-5]\n"
    "SHIFT_COORDS(0, 1, 0)\n"
    "DETECTOR(1, 0) rec[-1] rec[-2]\n"
    "DETECTOR\n"
    "OBSERVABLE_INCLUDE(0) rec[-3]\n"
    "OBSERVABLE_INCLUDE(2) rec[-4]\n"
    "OBSERVABLE_INCLUDE(0) rec[-5]\n"
    "REPEAT 3 {\n"
    "    TICK\n"
    "    CX 0 1\n"
    "    MR 0\n"
    "    SHIFT_COORDS(0, 0, 1)\n"
    "    DETECTOR(0, 0, 0) rec[-1]\n"
    "    REPEAT 2 {\n"
    "        MR 1\n"
    "        DETECTOR(9, 9, 9) rec[-1]\n"
    "    }\n"
    "}\n";

/* Distance-3 repetition-code memory, three rounds.  Data 0/2/4, ancilla
 * 1/3.  Records 0..5 are the ancilla rounds, 6..8 the final data readout. */
static const char* REP3_TEMPLATE =
    "R 0 1 2 3 4\n"
    "%s"                                  /* optional noise instruction */
    "TICK\n"
    "CX 0 1 2 3\n"
    "CX 2 1 4 3\n"
    "MR 1 3\n"
    "DETECTOR(1, 0) rec[-2]\n"
    "DETECTOR(3, 0) rec[-1]\n"
    "REPEAT 2 {\n"
    "    TICK\n"
    "    CX 0 1 2 3\n"
    "    CX 2 1 4 3\n"
    "    MR 1 3\n"
    "    SHIFT_COORDS(0, 1)\n"
    "    DETECTOR(1, 0) rec[-2] rec[-4]\n"
    "    DETECTOR(3, 0) rec[-1] rec[-3]\n"
    "}\n"
    "M 0 2 4\n"
    "DETECTOR(1, 1) rec[-3] rec[-2] rec[-5]\n"
    "DETECTOR(3, 1) rec[-2] rec[-1] rec[-4]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\n";

static char* rep3_with(const char* noise) {
    const size_t n = strlen(REP3_TEMPLATE) + strlen(noise) + 8;
    char* s = (char*)malloc(n);
    if (!s) return NULL;
    snprintf(s, n, REP3_TEMPLATE, noise);
    return s;
}

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

static moonlab_stim_circuit_t* must_parse(const char* text, const char* what) {
    moonlab_stim_error_t err;
    moonlab_stim_circuit_t* c = moonlab_stim_circuit_parse(text, &err);
    if (!c)
        fprintf(stderr, "parse of %s failed: code=%d line=%zu msg=%s\n",
                what, err.code, err.line, err.message);
    return c;
}

/* Compare everything the lowering and the annotations produce. */
static int same_semantics(const moonlab_stim_circuit_t* a,
                          const moonlab_stim_circuit_t* b) {
    if (moonlab_stim_circuit_num_qubits(a)       != moonlab_stim_circuit_num_qubits(b))       return 0;
    if (moonlab_stim_circuit_num_measurements(a) != moonlab_stim_circuit_num_measurements(b)) return 0;
    if (moonlab_stim_circuit_num_detectors(a)    != moonlab_stim_circuit_num_detectors(b))    return 0;
    if (moonlab_stim_circuit_num_observables(a)  != moonlab_stim_circuit_num_observables(b))  return 0;
    if (moonlab_stim_circuit_num_ticks(a)        != moonlab_stim_circuit_num_ticks(b))        return 0;

    const long na = moonlab_stim_circuit_num_ops(a);
    const long nb = moonlab_stim_circuit_num_ops(b);
    if (na != nb || na < 0) return 0;
    const long ca = moonlab_stim_circuit_num_channel_args(a);
    const long cb = moonlab_stim_circuit_num_channel_args(b);
    if (ca != cb || ca < 0) return 0;

    int ok = 1;
    pf_circuit_op_t* oa = (pf_circuit_op_t*)calloc((size_t)na + 1, sizeof(*oa));
    pf_circuit_op_t* ob = (pf_circuit_op_t*)calloc((size_t)nb + 1, sizeof(*ob));
    double* ga = (double*)calloc((size_t)ca + 1, sizeof(double));
    double* gb = (double*)calloc((size_t)cb + 1, sizeof(double));
    if (!oa || !ob || !ga || !gb) { ok = 0; goto done_ops; }
    if (moonlab_stim_circuit_lower(a, oa, (size_t)na, ga, (size_t)ca, NULL) != na) { ok = 0; goto done_ops; }
    if (moonlab_stim_circuit_lower(b, ob, (size_t)nb, gb, (size_t)cb, NULL) != nb) { ok = 0; goto done_ops; }
    for (long i = 0; i < na; i++) {
        if (oa[i].kind != ob[i].kind || oa[i].q0 != ob[i].q0 ||
            oa[i].q1 != ob[i].q1 || oa[i].p != ob[i].p) { ok = 0; goto done_ops; }
    }
    for (long i = 0; i < ca; i++) if (ga[i] != gb[i]) { ok = 0; goto done_ops; }
done_ops:
    free(oa); free(ob); free(ga); free(gb);
    if (!ok) return 0;

    /* Detector / observable CSRs. */
    for (int which = 0; which < 2; which++) {
        const size_t rows = which == 0 ? moonlab_stim_circuit_num_detectors(a)
                                       : moonlab_stim_circuit_num_observables(a);
        const long ia = which == 0 ? moonlab_stim_circuit_detector_csr(a, NULL, 0, NULL, 0)
                                   : moonlab_stim_circuit_observable_csr(a, NULL, 0, NULL, 0);
        const long ib = which == 0 ? moonlab_stim_circuit_detector_csr(b, NULL, 0, NULL, 0)
                                   : moonlab_stim_circuit_observable_csr(b, NULL, 0, NULL, 0);
        if (ia != ib || ia < 0) return 0;
        size_t* fa = (size_t*)calloc(rows + 1, sizeof(size_t));
        size_t* fb = (size_t*)calloc(rows + 1, sizeof(size_t));
        uint32_t* xa = (uint32_t*)calloc((size_t)ia + 1, sizeof(uint32_t));
        uint32_t* xb = (uint32_t*)calloc((size_t)ib + 1, sizeof(uint32_t));
        int sub = (fa && fb && xa && xb);
        if (sub) {
            if (which == 0) {
                sub = moonlab_stim_circuit_detector_csr(a, fa, rows + 1, xa, (size_t)ia) == ia &&
                      moonlab_stim_circuit_detector_csr(b, fb, rows + 1, xb, (size_t)ib) == ib;
            } else {
                sub = moonlab_stim_circuit_observable_csr(a, fa, rows + 1, xa, (size_t)ia) == ia &&
                      moonlab_stim_circuit_observable_csr(b, fb, rows + 1, xb, (size_t)ib) == ib;
            }
        }
        if (sub) {
            for (size_t i = 0; i <= rows; i++) if (fa[i] != fb[i]) { sub = 0; break; }
            for (long i = 0; sub && i < ia; i++) if (xa[i] != xb[i]) sub = 0;
        }
        free(fa); free(fb); free(xa); free(xb);
        if (!sub) return 0;
    }

    /* Measurement inversion mask. */
    {
        const size_t nm = moonlab_stim_circuit_num_measurements(a);
        uint8_t* ma = (uint8_t*)calloc(nm + 1, 1);
        uint8_t* mb = (uint8_t*)calloc(nm + 1, 1);
        int sub = (ma && mb);
        if (sub) {
            sub = moonlab_stim_circuit_measurement_inversions(a, ma, nm + 1) == (long)nm &&
                  moonlab_stim_circuit_measurement_inversions(b, mb, nm + 1) == (long)nm &&
                  memcmp(ma, mb, nm) == 0;
        }
        free(ma); free(mb);
        if (!sub) return 0;
    }

    /* Coordinates. */
    for (size_t q = 0; q < moonlab_stim_circuit_num_qubits(a); q++) {
        double va[8] = {0}, vb[8] = {0};
        const long la = moonlab_stim_circuit_qubit_coords(a, q, va, 8);
        const long lb = moonlab_stim_circuit_qubit_coords(b, q, vb, 8);
        if (la != lb) return 0;
        for (long i = 0; i < la && i < 8; i++) if (va[i] != vb[i]) return 0;
    }
    for (size_t d = 0; d < moonlab_stim_circuit_num_detectors(a); d++) {
        double va[8] = {0}, vb[8] = {0};
        const long la = moonlab_stim_circuit_detector_coords(a, d, va, 8);
        const long lb = moonlab_stim_circuit_detector_coords(b, d, vb, 8);
        if (la != lb) return 0;
        for (long i = 0; i < la && i < 8; i++) if (va[i] != vb[i]) return 0;
    }
    return 1;
}

/* ================================================================== */
/*  Tests                                                              */
/* ================================================================== */

static int test_kitchen_sink_parses(void) {
    moonlab_stim_circuit_t* c = must_parse(KITCHEN_SINK, "KITCHEN_SINK");
    ASSERT(c, "kitchen-sink circuit must parse");
    ASSERT(moonlab_stim_circuit_num_qubits(c) == 4, "num_qubits");

    /* 10 measurement records before the REPEAT (M, MZ, MX, MY, MR x4,
     * M(0.01), M !1) plus 2 MPAD entries, then 3 outer iterations each
     * contributing MR 0 plus two inner MR 1 = 9. */
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 10 + 2 + 9,
           "measurement record length (MPAD counted)");
    /* 3 DETECTORs outside the loop, then 3 * (1 + 2) inside. */
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 3 + 9, "num_detectors");
    /* OBSERVABLE_INCLUDE indices 0 and 2 -> one past the largest is 3. */
    ASSERT(moonlab_stim_circuit_num_observables(c) == 3, "num_observables");
    ASSERT(moonlab_stim_circuit_num_ticks(c) == 4 + 3, "num_ticks");

    /* Sparse observable indices keep an empty row rather than shifting. */
    long nobs_idx = moonlab_stim_circuit_observable_csr(c, NULL, 0, NULL, 0);
    ASSERT(nobs_idx == 3, "observable index entries");
    size_t off[4] = {0};
    uint32_t idx[3] = {0};
    ASSERT(moonlab_stim_circuit_observable_csr(c, off, 4, idx, 3) == 3,
           "observable csr fill");
    ASSERT(off[0] == 0 && off[1] == 2, "observable 0 accumulates both includes");
    ASSERT(off[1] == off[2], "observable 1 is present and empty");
    ASSERT(off[3] - off[2] == 1, "observable 2 has one record");

    /* A zero-target DETECTOR keeps its row. */
    size_t doff[13] = {0};
    long ndi = moonlab_stim_circuit_detector_csr(c, NULL, 0, NULL, 0);
    ASSERT(ndi >= 0, "detector csr sizing");
    uint32_t* dix = (uint32_t*)calloc((size_t)ndi + 1, sizeof(uint32_t));
    ASSERT(dix, "alloc");
    ASSERT(moonlab_stim_circuit_detector_csr(c, doff, 13, dix, (size_t)ndi) == ndi,
           "detector csr fill");
    ASSERT(doff[2] == doff[3], "the empty DETECTOR row is preserved");
    free(dix);

    /* Coordinates, including a cap smaller than the declared count. */
    double xy[2] = {-1, -1};
    ASSERT(moonlab_stim_circuit_qubit_coords(c, 2, xy, 2) == 2, "qubit coords n");
    ASSERT(xy[0] == 2.0 && xy[1] == 0.0, "qubit coords values");
    double one = -1;
    ASSERT(moonlab_stim_circuit_qubit_coords(c, 2, &one, 1) == 2,
           "qubit coords returns n even when capped");
    ASSERT(one == 2.0, "qubit coords capped write");
    ASSERT(moonlab_stim_circuit_qubit_coords(c, 99, xy, 2) == -1,
           "out-of-range qubit coords");
    ASSERT(moonlab_stim_circuit_detector_coords(c, 999, xy, 2) == -1,
           "out-of-range detector coords");

    moonlab_stim_circuit_free(c);
    return 0;
}

static int test_round_trip(void) {
    moonlab_stim_circuit_t* a = must_parse(KITCHEN_SINK, "KITCHEN_SINK");
    ASSERT(a, "parse a");

    for (int flatten = 0; flatten <= 1; flatten++) {
        char* text = moonlab_stim_circuit_to_text(a, flatten);
        ASSERT(text, "to_text");
        moonlab_stim_circuit_t* b = must_parse(text, "round-tripped text");
        ASSERT(b, "re-parse");
        ASSERT(same_semantics(a, b), "round trip changed the circuit");
        /* Canonical spelling is a fixed point, so a second pass must be
         * byte-identical. */
        char* text2 = moonlab_stim_circuit_to_text(b, flatten);
        ASSERT(text2, "to_text again");
        ASSERT(strcmp(text, text2) == 0, "to_text is not idempotent");
        moonlab_stim_text_free(text2);
        moonlab_stim_text_free(text);
        moonlab_stim_circuit_free(b);
    }

    /* The flattened form must agree with the block form as well. */
    char* flat = moonlab_stim_circuit_to_text(a, 1);
    ASSERT(flat, "flatten");
    ASSERT(strstr(flat, "REPEAT") == NULL, "flatten left a REPEAT behind");
    moonlab_stim_circuit_t* f = must_parse(flat, "flattened text");
    ASSERT(f, "parse flattened");
    ASSERT(same_semantics(a, f), "flattened form is not equivalent");
    moonlab_stim_text_free(flat);
    moonlab_stim_circuit_free(f);

    /* The block form must round-trip the REPEAT structure textually. */
    char* blocked = moonlab_stim_circuit_to_text(a, 0);
    ASSERT(blocked, "blocked");
    ASSERT(strstr(blocked, "REPEAT 3 {") != NULL, "REPEAT 3 not preserved");
    ASSERT(strstr(blocked, "REPEAT 2 {") != NULL, "nested REPEAT 2 not preserved");
    ASSERT(strstr(blocked, "H[hadamard-tag] 0") != NULL, "tag not preserved");
    moonlab_stim_text_free(blocked);

    moonlab_stim_circuit_free(a);
    return 0;
}

static int test_rec_resolution_across_repeat(void) {
    /* Each REPEAT iteration resolves rec[-1] against its own record. */
    static const char* src =
        "REPEAT 3 {\n"
        "    M 0\n"
        "    DETECTOR rec[-1]\n"
        "}\n";
    moonlab_stim_circuit_t* c = must_parse(src, "rec-in-repeat");
    ASSERT(c, "parse");
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 3, "3 records");
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 3, "3 detectors");
    size_t off[4] = {0};
    uint32_t idx[3] = {0};
    ASSERT(moonlab_stim_circuit_detector_csr(c, off, 4, idx, 3) == 3, "csr");
    ASSERT(idx[0] == 0 && idx[1] == 1 && idx[2] == 2,
           "rec[-1] must resolve to a different record each iteration");
    moonlab_stim_circuit_free(c);

    /* Nested REPEAT with a lookback that reaches into the previous
     * iteration of the outer loop. */
    static const char* nested =
        "M 0\n"
        "REPEAT 2 {\n"
        "    REPEAT 2 {\n"
        "        M 0\n"
        "        DETECTOR rec[-1] rec[-2]\n"
        "    }\n"
        "}\n";
    c = must_parse(nested, "nested rec");
    ASSERT(c, "parse nested");
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 5, "5 records");
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 4, "4 detectors");
    size_t noff[5] = {0};
    uint32_t nidx[8] = {0};
    ASSERT(moonlab_stim_circuit_detector_csr(c, noff, 5, nidx, 8) == 8, "csr");
    for (int d = 0; d < 4; d++) {
        ASSERT(nidx[2 * d] == (uint32_t)(d + 1), "rec[-1] per iteration");
        ASSERT(nidx[2 * d + 1] == (uint32_t)d, "rec[-2] per iteration");
    }
    moonlab_stim_circuit_free(c);
    return 0;
}

static int test_shift_coords_accumulates(void) {
    static const char* src =
        "SHIFT_COORDS(10, 100)\n"
        "M 0\n"
        "DETECTOR(1, 2) rec[-1]\n"
        "REPEAT 3 {\n"
        "    M 0\n"
        "    SHIFT_COORDS(0, 1)\n"
        "    DETECTOR(1, 2) rec[-1]\n"
        "}\n"
        "QUBIT_COORDS(5, 5) 7\n";
    moonlab_stim_circuit_t* c = must_parse(src, "shift-coords");
    ASSERT(c, "parse");
    double v[2];
    ASSERT(moonlab_stim_circuit_detector_coords(c, 0, v, 2) == 2, "d0 coords");
    ASSERT(v[0] == 11.0 && v[1] == 102.0, "d0 shifted once");
    /* The in-loop SHIFT_COORDS runs before the DETECTOR, and accumulates
     * across iterations. */
    for (int k = 0; k < 3; k++) {
        ASSERT(moonlab_stim_circuit_detector_coords(c, (size_t)k + 1, v, 2) == 2,
               "loop detector coords");
        ASSERT(v[0] == 11.0, "x offset unchanged in loop");
        ASSERT(v[1] == 102.0 + (double)(k + 1), "y offset accumulates per iteration");
    }
    /* SHIFT_COORDS applies to QUBIT_COORDS too. */
    ASSERT(moonlab_stim_circuit_qubit_coords(c, 7, v, 2) == 2, "qubit coords");
    ASSERT(v[0] == 15.0 && v[1] == 108.0, "qubit coords shifted");
    moonlab_stim_circuit_free(c);
    return 0;
}

static int test_mpad_is_a_record_not_a_measurement(void) {
    static const char* src =
        "R 0\n"
        "M 0\n"
        "MPAD 0 1 0\n"
        "M 0\n"
        "DETECTOR rec[-1] rec[-5]\n";
    moonlab_stim_circuit_t* c = must_parse(src, "mpad");
    ASSERT(c, "parse");
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 5,
           "MPAD advances the record");

    const long nops = moonlab_stim_circuit_num_ops(c);
    ASSERT(nops == 3, "R + M + M, MPAD emits no op");
    pf_circuit_op_t ops[3];
    ASSERT(moonlab_stim_circuit_lower(c, ops, 3, NULL, 0, NULL) == 3, "lower");
    ASSERT(ops[0].kind == PF_OP_RESET, "op0 reset");
    ASSERT(ops[1].kind == PF_OP_MEASURE, "op1 measure");
    ASSERT(ops[2].kind == PF_OP_MEASURE, "op2 measure");

    /* MPAD records are constant, so the detector still reads zero. */
    uint8_t det[8];
    ASSERT(moonlab_stim_circuit_sample_detectors(c, 8, 1234, 1, det, NULL) == 1,
           "sample detectors");
    for (int s = 0; s < 8; s++) ASSERT(det[s] == 0, "MPAD detector must be 0");

    /* The raw record shows the MPAD literals. */
    uint8_t recs[5 * 4];
    ASSERT(moonlab_stim_circuit_sample_measurements(c, 4, 99, 1, recs) == 5,
           "sample measurements");
    for (int s = 0; s < 4; s++) {
        ASSERT(recs[1 * 4 + s] == 0, "MPAD 0");
        ASSERT(recs[2 * 4 + s] == 1, "MPAD 1");
        ASSERT(recs[3 * 4 + s] == 0, "MPAD 0 again");
    }
    moonlab_stim_circuit_free(c);
    return 0;
}

static int test_inverted_measurements(void) {
    static const char* src =
        "R 0 1\n"
        "M 0\n"
        "M !1\n"
        "DETECTOR rec[-1] rec[-2]\n";
    moonlab_stim_circuit_t* c = must_parse(src, "inversion");
    ASSERT(c, "parse");
    uint8_t inv[2] = {9, 9};
    ASSERT(moonlab_stim_circuit_measurement_inversions(c, inv, 2) == 2, "mask");
    ASSERT(inv[0] == 0 && inv[1] == 1, "only the '!' record is inverted");

    uint8_t recs[2 * 4];
    ASSERT(moonlab_stim_circuit_sample_measurements(c, 4, 7, 1, recs) == 2,
           "sample");
    for (int s = 0; s < 4; s++) {
        ASSERT(recs[0 * 4 + s] == 0, "record 0 is |0>");
        ASSERT(recs[1 * 4 + s] == 1, "record 1 is inverted |0>");
    }
    /* The constant cancels in the detector's deviation from the reference. */
    uint8_t det[4];
    ASSERT(moonlab_stim_circuit_sample_detectors(c, 4, 7, 1, det, NULL) == 1,
           "detectors");
    for (int s = 0; s < 4; s++) ASSERT(det[s] == 0, "inversion must not fire a detector");
    moonlab_stim_circuit_free(c);
    return 0;
}

/* Sample a fully deterministic circuit and check every record against the
 * expected bit string.  This runs each decomposition through the reference
 * Clifford tableau, so it validates the encoded gate table and the
 * measurement / reset basis rotations end to end. */
static int expect_records(const char* src, const char* bits) {
    moonlab_stim_circuit_t* c = must_parse(src, src);
    if (!c) { fprintf(stderr, "  circuit:\n%s", src); return 1; }
    const size_t nm = moonlab_stim_circuit_num_measurements(c);
    const size_t want = strlen(bits);
    int rc = 0;
    if (nm != want) {
        fprintf(stderr, "FAIL `%s` produced %zu records, want %zu\n",
                src, nm, want);
        rc = 1;
    } else {
        const size_t shots = 32;
        uint8_t* out = (uint8_t*)malloc(nm * shots);
        if (!out) { moonlab_stim_circuit_free(c); return 1; }
        if (moonlab_stim_circuit_sample_measurements(c, shots, 424242, 1, out)
            != (long)nm) {
            fprintf(stderr, "FAIL `%s` sampling failed\n", src);
            rc = 1;
        } else {
            for (size_t m = 0; m < nm && !rc; m++)
                for (size_t s = 0; s < shots; s++)
                    if (out[m * shots + s] != (uint8_t)(bits[m] - '0')) {
                        fprintf(stderr, "FAIL `%s` record %zu is %u, want %c\n",
                                src, m, out[m * shots + s], bits[m]);
                        rc = 1;
                        break;
                    }
        }
        free(out);
    }
    moonlab_stim_circuit_free(c);
    return rc;
}

static int test_deterministic_gate_actions(void) {
    struct { const char* src; const char* bits; } cases[] = {
        /* -- Pauli and basis preparation / readout ------------------- */
        {"R 0\nM 0\n",                                   "0"},
        {"R 0\nX 0\nM 0\n",                              "1"},
        {"R 0\nY 0\nM 0\n",                              "1"},
        {"R 0\nZ 0\nM 0\n",                              "0"},
        {"R 0\nI 0\nM 0\n",                              "0"},
        {"RX 0\nMX 0\n",                                 "0"},
        {"RX 0\nZ 0\nMX 0\n",                            "1"},
        {"RY 0\nMY 0\n",                                 "0"},
        {"RY 0\nZ 0\nMY 0\n",                            "1"},
        {"R 0\nH 0\nMX 0\n",                             "0"},
        /* MR-family resets restore the promised basis state. */
        {"R 0\nX 0\nMR 0\nM 0\n",                        "10"},
        {"RX 0\nZ 0\nMRX 0\nMX 0\n",                     "10"},
        {"RY 0\nZ 0\nMRY 0\nMY 0\n",                     "10"},
        {"RY 0\nMRY 0\nMY 0\n",                          "00"},
        /* -- single-qubit Clifford flows ----------------------------- */
        {"R 0\nSQRT_X 0\nMY 0\n",                        "1"},  /* Z -> -Y */
        {"R 0\nSQRT_X_DAG 0\nMY 0\n",                    "0"},  /* Z ->  Y */
        {"R 0\nSQRT_Y 0\nMX 0\n",                        "0"},  /* Z ->  X */
        {"R 0\nSQRT_Y_DAG 0\nMX 0\n",                    "1"},  /* Z -> -X */
        {"R 0\nC_XYZ 0\nMX 0\n",                         "0"},  /* Z ->  X */
        {"R 0\nC_ZYX 0\nMY 0\n",                         "0"},  /* Z ->  Y */
        {"R 0\nH_YZ 0\nMY 0\n",                          "0"},  /* Z ->  Y */
        {"RX 0\nH_XY 0\nMY 0\n",                         "0"},  /* X ->  Y */
        {"R 0\nS 0\nM 0\n",                              "0"},
        {"R 0\nSQRT_Z 0\nSQRT_Z_DAG 0\nM 0\n",           "0"},
        /* -- two-qubit Clifford flows -------------------------------- */
        {"R 0 1\nX 0\nCX 0 1\nM 0 1\n",                  "11"},
        {"R 0 1\nX 0\nCNOT 0 1\nM 0 1\n",                "11"},
        {"R 0 1\nX 1\nXCZ 0 1\nM 0 1\n",                 "11"},
        {"R 0 1\nX 0\nCY 0 1\nM 0 1\n",                  "11"},
        {"R 0 1\nX 1\nYCZ 0 1\nM 0 1\n",                 "11"},
        {"R 0 1\nX 0\nCZ 0 1\nM 0 1\n",                  "10"},
        {"R 0 1\nX 0\nSWAP 0 1\nM 0 1\n",                "01"},
        {"R 0 1\nX 0\nISWAP 0 1\nM 0 1\n",               "01"},
        {"R 0 1\nX 0\nISWAP_DAG 0 1\nM 0 1\n",           "01"},
        {"R 0 1\nX 0\nCXSWAP 0 1\nM 0 1\n",              "11"},
        {"R 0 1\nX 0\nSWAPCX 0 1\nM 0 1\n",              "01"},
        {"R 0 1\nX 0\nCZSWAP 0 1\nM 0 1\n",              "01"},
        {"R 0 1\nX 0\nSWAPCZ 0 1\nM 0 1\n",              "01"},
        /* Squaring a square root reproduces the two-qubit Pauli. */
        {"R 0 1\nSQRT_XX 0 1\nSQRT_XX 0 1\nM 0 1\n",             "11"},
        {"R 0 1\nX 0\nSQRT_XX_DAG 0 1\nSQRT_XX_DAG 0 1\nM 0 1\n", "01"},
        {"R 0 1\nSQRT_YY 0 1\nSQRT_YY 0 1\nM 0 1\n",             "11"},
        {"R 0 1\nSQRT_YY_DAG 0 1\nSQRT_YY_DAG 0 1\nM 0 1\n",     "11"},
        {"R 0 1\nX 0\nSQRT_ZZ 0 1\nSQRT_ZZ 0 1\nM 0 1\n",        "10"},
        {"R 0 1\nX 0\nSQRT_ZZ_DAG 0 1\nSQRT_ZZ_DAG 0 1\nM 0 1\n","10"},
        /* Controls in the X and Y bases. */
        {"R 0 1\nH 0\nXCX 0 1\nM 1\n",                   "0"},
        {"R 0 1\nH 0\nZ 0\nXCX 0 1\nM 1\n",              "1"},
        {"R 0 1\nH 0\nZ 0\nXCY 0 1\nM 1\n",              "1"},
        {"R 1\nRY 0\nZ 0\nYCX 0 1\nM 1\n",               "1"},
        {"R 1\nRY 0\nZ 0\nYCY 0 1\nM 1\n",               "1"},
        {"R 1\nRY 0\nYCX 0 1\nM 1\n",                    "0"},
        /* Multi-target instructions apply per target / per pair. */
        {"R 0 1 2 3\nX 0 2\nCX 0 1 2 3\nM 0 1 2 3\n",    "1111"},
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (expect_records(cases[i].src, cases[i].bits)) rc = 1;
    return rc;
}

static int test_lowering_sizes(void) {
    struct { const char* src; long ops; long chan; } cases[] = {
        {"H 0\n",                       1,  0},
        {"I 0\n",                       0,  0},
        {"SQRT_X 0\n",                  3,  0},      /* H S H            */
        {"C_XYZ 0\n",                   2,  0},      /* S_DAG H          */
        {"YCY 0 1\n",                   7,  0},      /* longest 2q chain */
        {"CX 0 1 2 3\n",                2,  0},      /* pairs            */
        {"MX 0\n",                      3,  0},      /* H M H            */
        {"MY 0\n",                      5,  0},      /* S_DAG H M H S    */
        {"MR 0\n",                      2,  0},      /* M R              */
        {"MRX 0\n",                     4,  0},      /* H M R H          */
        {"MRY 0\n",                     6,  0},      /* S_DAG H M R H S  */
        {"RX 0\n",                      2,  0},      /* R H              */
        {"RY 0\n",                      3,  0},      /* R H S            */
        {"PAULI_CHANNEL_1(0.1,0.2,0.3) 0 1\n", 2, 3},
        {"PAULI_CHANNEL_2(0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 1 2 3\n", 2, 15},
        {"DETECTOR\nTICK\nSHIFT_COORDS(1)\n", 0, 0},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        moonlab_stim_circuit_t* c = must_parse(cases[i].src, cases[i].src);
        if (!c) { fprintf(stderr, "  case: %s", cases[i].src); return 1; }
        const long n = moonlab_stim_circuit_num_ops(c);
        const long g = moonlab_stim_circuit_num_channel_args(c);
        if (n != cases[i].ops || g != cases[i].chan) {
            fprintf(stderr, "FAIL lowering size for `%s`: ops=%ld want %ld, "
                            "chan=%ld want %ld\n",
                    cases[i].src, n, cases[i].ops, g, cases[i].chan);
            moonlab_stim_circuit_free(c);
            return 1;
        }
        moonlab_stim_circuit_free(c);
    }

    /* Two PAULI_CHANNEL_2 instructions get independent argument blocks. */
    moonlab_stim_circuit_t* c = must_parse(
        "PAULI_CHANNEL_2(0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 1\n"
        "PAULI_CHANNEL_2(0,0.02,0,0,0,0,0,0,0,0,0,0,0,0,0) 2 3\n", "two pc2");
    ASSERT(c, "parse pc2 pair");
    ASSERT(moonlab_stim_circuit_num_channel_args(c) == 30, "30 channel args");
    pf_circuit_op_t ops[2];
    double chan[30];
    ASSERT(moonlab_stim_circuit_lower(c, ops, 2, chan, 30, NULL) == 2, "lower");
    ASSERT(ops[0].kind == PF_OP_PAULI_CHANNEL_2, "op0 kind");
    ASSERT(ops[1].kind == PF_OP_PAULI_CHANNEL_2, "op1 kind");
    ASSERT(ops[0].p == 0.0 && ops[1].p == 15.0, "channel base indices");
    ASSERT(chan[0] == 0.01 && chan[16] == 0.02, "channel args laid out in order");

    /* Overflow reporting. */
    moonlab_stim_error_t err;
    ASSERT(moonlab_stim_circuit_lower(c, ops, 1, chan, 30, &err)
               == MOONLAB_STIM_ERR_OVERFLOW, "op cap overflow");
    ASSERT(err.code == MOONLAB_STIM_ERR_OVERFLOW, "overflow code");
    ASSERT(moonlab_stim_circuit_lower(c, ops, 2, chan, 3, &err)
               == MOONLAB_STIM_ERR_OVERFLOW, "chan cap overflow");
    ASSERT(moonlab_stim_circuit_detector_csr(c, NULL, 0, NULL, 0) == 0,
           "no detectors");
    moonlab_stim_circuit_free(c);
    return 0;
}

/* Every documented rejection: right code, nonzero line, message names the
 * offending token. */
static int test_rejections(void) {
    struct { const char* src; int code; const char* needle; size_t line; } cases[] = {
        {"H 0\nMPP X0*X1\n",                     MOONLAB_STIM_ERR_UNSUPPORTED, "'MPP'", 2},
        {"SPP X0\n",                             MOONLAB_STIM_ERR_UNSUPPORTED, "'SPP'", 1},
        {"SPP_DAG X0\n",                         MOONLAB_STIM_ERR_UNSUPPORTED, "'SPP_DAG'", 1},
        {"E(0.1) X0\n",                          MOONLAB_STIM_ERR_UNSUPPORTED, "'E'", 1},
        {"CORRELATED_ERROR(0.1) X0\n",           MOONLAB_STIM_ERR_UNSUPPORTED, "'CORRELATED_ERROR'", 1},
        {"ELSE_CORRELATED_ERROR(0.1) X0\n",      MOONLAB_STIM_ERR_UNSUPPORTED, "'ELSE_CORRELATED_ERROR'", 1},
        {"HERALDED_ERASE(0.1) 0\n",              MOONLAB_STIM_ERR_UNSUPPORTED, "'HERALDED_ERASE'", 1},
        {"HERALDED_PAULI_CHANNEL_1(0.1,0,0,0) 0\n", MOONLAB_STIM_ERR_UNSUPPORTED, "'HERALDED_PAULI_CHANNEL_1'", 1},
        {"II 0 1\n",                             MOONLAB_STIM_ERR_UNSUPPORTED, "'II'", 1},
        {"II_ERROR(0.1) 0 1\n",                  MOONLAB_STIM_ERR_UNSUPPORTED, "'II_ERROR'", 1},
        {"H 0\nFLUBBERCAKE 1\n",                 MOONLAB_STIM_ERR_UNSUPPORTED, "'FLUBBERCAKE'", 2},
        {"CX sweep[0] 1\n",                      MOONLAB_STIM_ERR_UNSUPPORTED, "sweep[0]", 1},
        {"H 0\nDETECTOR *\n",                    MOONLAB_STIM_ERR_UNSUPPORTED, "combiner", 2},
        {"M 0\nM 1\nDETECTOR rec[-1]*rec[-2]\n", MOONLAB_STIM_ERR_UNSUPPORTED, "combiner", 3},
        {"H X0\n",                               MOONLAB_STIM_ERR_UNSUPPORTED, "Pauli target", 1},
        {"OBSERVABLE_INCLUDE(0) X1\n",           MOONLAB_STIM_ERR_UNSUPPORTED, "Pauli target", 1},
        {"X !3\n",                               MOONLAB_STIM_ERR_UNSUPPORTED, "'!3'", 1},
        {"H 0\nR !2\n",                          MOONLAB_STIM_ERR_UNSUPPORTED, "'!2'", 2},
        {"H rec[-1]\n",                          MOONLAB_STIM_ERR_UNSUPPORTED, "measurement-record", 1},
        {"DETECTOR 3\n",                         MOONLAB_STIM_ERR_UNSUPPORTED, "rec[-k]", 1},
        {"H 0 {\nX 1\n}\n",                      MOONLAB_STIM_ERR_UNSUPPORTED, "cannot open a block", 1},
        {"CX 0 1 2\n",                           MOONLAB_STIM_ERR_SYNTAX,      "pairs", 1},
        {"DEPOLARIZE2(0.01) 0 1 2\n",            MOONLAB_STIM_ERR_SYNTAX,      "pairs", 1},
        {"PAULI_CHANNEL_2(0.01,0,0) 0 1 2\n",    MOONLAB_STIM_ERR_SYNTAX,      "exactly 15", 1},
        {"PAULI_CHANNEL_1(0.1, 0.1) 0\n",        MOONLAB_STIM_ERR_SYNTAX,      "exactly 3", 1},
        {"R(0.1) 0\n",                           MOONLAB_STIM_ERR_SYNTAX,      "exactly 0", 1},
        {"TICK 0\n",                             MOONLAB_STIM_ERR_SYNTAX,      "no targets", 1},
        {"SHIFT_COORDS(1) 0\n",                  MOONLAB_STIM_ERR_SYNTAX,      "no targets", 1},
        {"MPAD 3\n",                             MOONLAB_STIM_ERR_SYNTAX,      "0 or 1", 1},
        {"REPEAT 2 {\nH 0\n",                    MOONLAB_STIM_ERR_SYNTAX,      "left open", 3},
        {"H 0\n}\n",                             MOONLAB_STIM_ERR_SYNTAX,      "unmatched", 2},
        {"REPEAT 2\nH 0\n",                      MOONLAB_STIM_ERR_SYNTAX,      "followed by '{'", 1},
        {"X_ERROR(1.5) 0\n",                     MOONLAB_STIM_ERR_BAD_ARG,     "outside [0, 1]", 1},
        {"X_ERROR(-0.5) 0\n",                    MOONLAB_STIM_ERR_BAD_ARG,     "outside [0, 1]", 1},
        {"M(2.0) 0\n",                           MOONLAB_STIM_ERR_BAD_ARG,     "outside [0, 1]", 1},
        {"PAULI_CHANNEL_1(0.5, 0.5, 0.5) 0\n",   MOONLAB_STIM_ERR_BAD_ARG,     "exceeds 1", 1},
        {"M 0\nDETECTOR rec[-5]\n",              MOONLAB_STIM_ERR_BAD_ARG,     "rec[-5]", 2},
        {"REPEAT 0 {\nH 0\n}\n",                 MOONLAB_STIM_ERR_BAD_ARG,     "at least 1", 1},
        {"CX 0 0\n",                             MOONLAB_STIM_ERR_BAD_ARG,     "twice", 1},
        {"QUBIT_COORDS(1,2) 0\nQUBIT_COORDS(3,4) 0\n", MOONLAB_STIM_ERR_BAD_ARG, "more than once", 2},
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        moonlab_stim_error_t err;
        memset(&err, 0, sizeof(err));
        moonlab_stim_circuit_t* c =
            moonlab_stim_circuit_parse(cases[i].src, &err);
        if (c) {
            fprintf(stderr, "FAIL case %zu accepted an unsupported circuit:\n%s",
                    i, cases[i].src);
            moonlab_stim_circuit_free(c);
            rc = 1;
            continue;
        }
        if (err.code != cases[i].code) {
            fprintf(stderr, "FAIL case %zu code=%d want %d (%s)\n",
                    i, err.code, cases[i].code, err.message);
            rc = 1;
        }
        if (err.line == 0) {
            fprintf(stderr, "FAIL case %zu has no line number (%s)\n",
                    i, err.message);
            rc = 1;
        } else if (err.line != cases[i].line) {
            fprintf(stderr, "FAIL case %zu line=%zu want %zu (%s)\n",
                    i, err.line, cases[i].line, err.message);
            rc = 1;
        }
        if (!strstr(err.message, cases[i].needle)) {
            fprintf(stderr, "FAIL case %zu message '%s' does not name '%s'\n",
                    i, err.message, cases[i].needle);
            rc = 1;
        }
    }
    /* NULL text is a caller error, not a parse error. */
    moonlab_stim_error_t err;
    ASSERT(moonlab_stim_circuit_parse(NULL, &err) == NULL, "NULL text");
    ASSERT(err.code == MOONLAB_STIM_ERR_BAD_ARG, "NULL text code");
    /* A missing file reports IO, not a syntax error. */
    ASSERT(moonlab_stim_circuit_parse_file(
               "/nonexistent/moonlab/stim/circuit.stim", &err) == NULL,
           "missing file");
    ASSERT(err.code == MOONLAB_STIM_ERR_IO, "missing file code");
    /* NULL err must be tolerated. */
    ASSERT(moonlab_stim_circuit_parse("MPP X0\n", NULL) == NULL,
           "NULL err out-parameter");
    return rc;
}

static int test_noiseless_detectors_are_zero(void) {
    char* src = rep3_with("");
    ASSERT(src, "alloc");
    moonlab_stim_circuit_t* c = must_parse(src, "rep3 noiseless");
    free(src);
    ASSERT(c, "parse");
    ASSERT(moonlab_stim_circuit_num_qubits(c) == 5, "5 qubits");
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 9, "9 records");
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 8, "8 detectors");
    ASSERT(moonlab_stim_circuit_num_observables(c) == 1, "1 observable");
    ASSERT(moonlab_stim_circuit_num_ticks(c) == 3, "3 ticks");

    const size_t shots = 256;
    uint8_t* det = (uint8_t*)malloc(8 * shots);
    uint8_t* obs = (uint8_t*)malloc(1 * shots);
    ASSERT(det && obs, "alloc");
    /* num_threads = 0 selects every core; the answer must not depend on it. */
    for (int threads = 0; threads <= 1; threads++) {
        ASSERT(moonlab_stim_circuit_sample_detectors(c, shots, 20260731,
                                                     threads, det, obs) == 8,
               "sample");
        for (size_t i = 0; i < 8 * shots; i++)
            ASSERT(det[i] == 0, "noiseless detector fired");
        for (size_t i = 0; i < shots; i++)
            ASSERT(obs[i] == 0, "noiseless observable flipped");
    }
    /* The raw measurement path must agree: all records read 0. */
    uint8_t* recs = (uint8_t*)malloc(9 * shots);
    ASSERT(recs, "alloc");
    ASSERT(moonlab_stim_circuit_sample_measurements(c, shots, 20260731, 0, recs)
               == 9, "sample measurements");
    for (size_t i = 0; i < 9 * shots; i++)
        ASSERT(recs[i] == 0, "noiseless record is not 0");
    free(recs);
    free(det); free(obs);
    moonlab_stim_circuit_free(c);
    return 0;
}

/* A p=1.0 X_ERROR on data qubit 0 lights exactly detector 0: it flips the
 * first round's Z0Z2 ancilla, every later round compares consecutive
 * measurements of a persistent error and so stays quiet, and the final data
 * readout is flipped consistently with the last ancilla. */
static int test_deterministic_x_error_lights_one_detector(void) {
    char* src = rep3_with("X_ERROR(1) 0\n");
    ASSERT(src, "alloc");
    moonlab_stim_circuit_t* c = must_parse(src, "rep3 with X_ERROR(1)");
    free(src);
    ASSERT(c, "parse");

    const size_t shots = 64;
    uint8_t* det = (uint8_t*)malloc(8 * shots);
    ASSERT(det, "alloc");
    ASSERT(moonlab_stim_circuit_sample_detectors(c, shots, 5, 1, det, NULL) == 8,
           "sample");
    for (size_t s = 0; s < shots; s++) {
        ASSERT(det[0 * shots + s] == 1, "detector 0 must fire");
        for (size_t d = 1; d < 8; d++)
            ASSERT(det[d * shots + s] == 0, "only detector 0 may fire");
    }
    free(det);
    moonlab_stim_circuit_free(c);
    return 0;
}

/* PAULI_CHANNEL_1(1, 0, 0) is X_ERROR(1.0): both are deterministic, so the
 * detector output must be bit-identical. */
static int test_pauli_channel_1_matches_x_error(void) {
    char* sa = rep3_with("X_ERROR(1) 0\n");
    char* sb = rep3_with("PAULI_CHANNEL_1(1, 0, 0) 0\n");
    ASSERT(sa && sb, "alloc");
    moonlab_stim_circuit_t* a = must_parse(sa, "x_error");
    moonlab_stim_circuit_t* b = must_parse(sb, "pauli_channel_1");
    free(sa); free(sb);
    ASSERT(a && b, "parse");
    ASSERT(moonlab_stim_circuit_num_channel_args(a) == 0, "x_error has no chan args");
    ASSERT(moonlab_stim_circuit_num_channel_args(b) == 3, "pc1 has 3 chan args");

    const size_t shots = 128;
    uint8_t* da = (uint8_t*)malloc(8 * shots);
    uint8_t* db = (uint8_t*)malloc(8 * shots);
    ASSERT(da && db, "alloc");
    ASSERT(moonlab_stim_circuit_sample_detectors(a, shots, 11, 1, da, NULL) == 8, "sample a");
    ASSERT(moonlab_stim_circuit_sample_detectors(b, shots, 11, 1, db, NULL) == 8, "sample b");
    ASSERT(memcmp(da, db, 8 * shots) == 0,
           "PAULI_CHANNEL_1(1,0,0) must equal X_ERROR(1.0)");
    free(da); free(db);
    moonlab_stim_circuit_free(a);
    moonlab_stim_circuit_free(b);

    /* And the sampled rate of a p=0.5 X component lands where it should. */
    char* sc = rep3_with("PAULI_CHANNEL_1(0.5, 0, 0) 0\n");
    ASSERT(sc, "alloc");
    moonlab_stim_circuit_t* c = must_parse(sc, "pc1 half");
    free(sc);
    ASSERT(c, "parse");
    const size_t n = 20000;
    uint8_t* det = (uint8_t*)malloc(8 * n);
    ASSERT(det, "alloc");
    ASSERT(moonlab_stim_circuit_sample_detectors(c, n, 3, 1, det, NULL) == 8, "sample");
    size_t fired = 0;
    for (size_t s = 0; s < n; s++) fired += det[0 * n + s];
    const double rate = (double)fired / (double)n;
    const double sigma = 0.5 / sqrt((double)n);
    ASSERT(fabs(rate - 0.5) < 5.0 * sigma, "PAULI_CHANNEL_1 X rate off by >5 sigma");
    free(det);
    moonlab_stim_circuit_free(c);
    return 0;
}

/* PAULI_CHANNEL_2 with all weight on ZZ flips nothing measurable in the
 * Z basis; with all weight on XI it flips exactly the same detector as an
 * X_ERROR(1) on the first target. */
static int test_pauli_channel_2_semantics(void) {
    char* sx = rep3_with(
        "PAULI_CHANNEL_2(0,0,0,1,0,0,0,0,0,0,0,0,0,0,0) 0 2\n");
    char* sz = rep3_with(
        "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1) 0 2\n");
    char* sref = rep3_with("X_ERROR(1) 0\n");
    ASSERT(sx && sz && sref, "alloc");
    moonlab_stim_circuit_t* cx = must_parse(sx, "pc2 XI");
    moonlab_stim_circuit_t* cz = must_parse(sz, "pc2 ZZ");
    moonlab_stim_circuit_t* cr = must_parse(sref, "x_error ref");
    free(sx); free(sz); free(sref);
    ASSERT(cx && cz && cr, "parse");
    ASSERT(moonlab_stim_circuit_num_channel_args(cx) == 15, "15 chan args");

    const size_t shots = 64;
    uint8_t* dx = (uint8_t*)malloc(8 * shots);
    uint8_t* dz = (uint8_t*)malloc(8 * shots);
    uint8_t* dr = (uint8_t*)malloc(8 * shots);
    ASSERT(dx && dz && dr, "alloc");
    ASSERT(moonlab_stim_circuit_sample_detectors(cx, shots, 2, 1, dx, NULL) == 8, "sample x");
    ASSERT(moonlab_stim_circuit_sample_detectors(cz, shots, 2, 1, dz, NULL) == 8, "sample z");
    ASSERT(moonlab_stim_circuit_sample_detectors(cr, shots, 2, 1, dr, NULL) == 8, "sample ref");
    ASSERT(memcmp(dx, dr, 8 * shots) == 0,
           "PAULI_CHANNEL_2 index 3 (XI) must act as X on the first target");
    for (size_t i = 0; i < 8 * shots; i++)
        ASSERT(dz[i] == 0, "a pure ZZ channel must not light a Z-basis detector");
    free(dx); free(dz); free(dr);
    moonlab_stim_circuit_free(cx);
    moonlab_stim_circuit_free(cz);
    moonlab_stim_circuit_free(cr);
    return 0;
}

/* A channel op with no argument table must fail, never be skipped. */
static int test_channel_requires_table(void) {
    pf_circuit_op_t ops[3];
    memset(ops, 0, sizeof(ops));
    ops[0].kind = PF_OP_RESET;            ops[0].q0 = 0;
    ops[1].kind = PF_OP_PAULI_CHANNEL_1;  ops[1].q0 = 0; ops[1].p = 0.0;
    ops[2].kind = PF_OP_MEASURE;          ops[2].q0 = 0;

    const size_t shots = 32;
    uint8_t out[32];
    ASSERT(pauli_frame_batch_sample_circuit(1, ops, 3, shots, 1, 1, out) == -1,
           "a channel op without a table must return -1");

    const double chan[3] = {1.0, 0.0, 0.0};
    ASSERT(pauli_frame_batch_sample_circuit_ex(1, ops, 3, chan, 3, shots, 1, 1, out)
               == 1, "the _ex entry point must accept the table");
    for (size_t s = 0; s < shots; s++)
        ASSERT(out[s] == 1, "PAULI_CHANNEL_1(1,0,0) must flip every shot");

    /* A base index whose block runs off the table is an error too. */
    ops[1].p = 1.0;
    ASSERT(pauli_frame_batch_sample_circuit_ex(1, ops, 3, chan, 3, shots, 1, 1, out)
               == -1, "an out-of-range channel base must return -1");

    /* Same contract on the detector entry point. */
    ops[1].p = 0.0;
    const size_t det_off[2] = {0, 1};
    const uint32_t det_idx[1] = {0};
    uint8_t det[32];
    ASSERT(pauli_frame_batch_sample_detectors(1, ops, 3, det_off, det_idx, 1,
                                              shots, 1, 1, det) == -1,
           "detector sampler must reject an untabled channel op");
    ASSERT(pauli_frame_batch_sample_detectors_ex(1, ops, 3, chan, 3, det_off,
                                                 det_idx, 1, shots, 1, 1, det)
               == 1, "detector _ex entry point");
    return 0;
}

static int test_parse_file(void) {
    const char* path = "/tmp/moonlab_test_stim_circuit.stim";
    FILE* f = fopen(path, "wb");
    ASSERT(f, "open temp file");
    fputs("R 0\nX_ERROR(0.25) 0\nM 0\nDETECTOR rec[-1]\n", f);
    fclose(f);
    moonlab_stim_error_t err;
    moonlab_stim_circuit_t* c = moonlab_stim_circuit_parse_file(path, &err);
    ASSERT(c, "parse_file");
    ASSERT(err.code == MOONLAB_STIM_OK, "ok status");
    ASSERT(moonlab_stim_circuit_num_measurements(c) == 1, "one record");
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 1, "one detector");
    moonlab_stim_circuit_free(c);
    remove(path);
    return 0;
}

int main(void) {
    if (test_kitchen_sink_parses() != 0) return 1;
    fprintf(stderr, "PASS test_kitchen_sink_parses\n");
    if (test_round_trip() != 0) return 1;
    fprintf(stderr, "PASS test_round_trip\n");
    if (test_rec_resolution_across_repeat() != 0) return 1;
    fprintf(stderr, "PASS test_rec_resolution_across_repeat\n");
    if (test_shift_coords_accumulates() != 0) return 1;
    fprintf(stderr, "PASS test_shift_coords_accumulates\n");
    if (test_mpad_is_a_record_not_a_measurement() != 0) return 1;
    fprintf(stderr, "PASS test_mpad_is_a_record_not_a_measurement\n");
    if (test_inverted_measurements() != 0) return 1;
    fprintf(stderr, "PASS test_inverted_measurements\n");
    if (test_deterministic_gate_actions() != 0) return 1;
    fprintf(stderr, "PASS test_deterministic_gate_actions\n");
    if (test_lowering_sizes() != 0) return 1;
    fprintf(stderr, "PASS test_lowering_sizes\n");
    if (test_rejections() != 0) return 1;
    fprintf(stderr, "PASS test_rejections\n");
    if (test_noiseless_detectors_are_zero() != 0) return 1;
    fprintf(stderr, "PASS test_noiseless_detectors_are_zero\n");
    if (test_deterministic_x_error_lights_one_detector() != 0) return 1;
    fprintf(stderr, "PASS test_deterministic_x_error_lights_one_detector\n");
    if (test_pauli_channel_1_matches_x_error() != 0) return 1;
    fprintf(stderr, "PASS test_pauli_channel_1_matches_x_error\n");
    if (test_pauli_channel_2_semantics() != 0) return 1;
    fprintf(stderr, "PASS test_pauli_channel_2_semantics\n");
    if (test_channel_requires_table() != 0) return 1;
    fprintf(stderr, "PASS test_channel_requires_table\n");
    if (test_parse_file() != 0) return 1;
    fprintf(stderr, "PASS test_parse_file\n");
    return 0;
}
