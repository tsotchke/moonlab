/**
 * @file test_noise_marginals.c
 * @brief High-N marginal statistics for every noise instruction, against
 *        CLOSED-FORM expectations.
 *
 * WHY THIS EXISTS.  The gate decompositions are proved exactly against
 * `stim.Tableau` (tests/unit/test_stim_circuit.c and the python flow
 * battery), so a disagreement with Stim on a QEC workload can only come from
 * the noise and measurement semantics: the DEPOLARIZE2 15-way convention, the
 * placement of a measurement flip relative to the reset in MR, the reset's
 * own frame semantics, or the probability handling in the channel sampler.
 * Sampling Stim and comparing is the weak form of that check -- it costs a
 * dependency, it is only as tight as the shot count on BOTH sides, and a
 * common-mode misreading of Stim's convention would pass it.  Every number
 * below is instead the exact probability the instruction is defined to
 * produce, computed by hand.
 *
 * SENSITIVITY.  Each case runs NM_SHOTS = 2^27 = 134,217,728 shots and admits
 * a deviation of at most NM_SIGMA = 5 standard errors at the analytic rate.
 * At the p = 1e-3 QEC operating point one standard error is 2.7e-6, so the
 * budget is 1.4e-5 and a systematic offset of 3e-5 -- the size of the largest
 * discrepancy ever reported against Stim on a surface-code detector marginal
 * -- lands at 11 sigma and fails on the single instruction that causes it.
 * This is verified by mutation: adding 3e-5 to the rate inside
 * pf_noise_1comp() fails 15 assertions (maximum z = 16.06).  At p = 0.1 the
 * same budget resolves a relative error of 0.13%, tighter than any convention
 * error can be: getting the DEPOLARIZE2 denominator wrong by one (14 or 16
 * instead of 15) moves the marginal by 6.7%, and permuting a
 * PAULI_CHANNEL_2 slot moves it by 100%.
 *
 * Exact-0 and exact-1 expectations are asserted as exact counts, not
 * tolerances: those instructions are deterministic and any nonzero count is a
 * bug however small.
 *
 * Circuits are fed in as `.stim` text so the parser's lowering of the noise
 * instructions is under test too, not just the sampler kernels.
 */

#include "../../src/qec/stim_circuit.h"
#include "../../src/backends/clifford/pauli_frame.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } \
} while (0)

/* 2^27 shots per case, drawn in 2^23-shot blocks so the record buffer stays
 * a few MB and every case spans sixteen independent seeds. */
#define NM_SHOTS   (1u << 27)
#define NM_CHUNK   (1u << 23)
#define NM_BLOCKS  (NM_SHOTS / NM_CHUNK)
#define NM_SIGMA   5.0
#define NM_MAX_REC 3

/* A record whose expectation is exactly 0 or exactly 1. */
#define NM_EXACT_TOL 1e-12

typedef struct {
    char   name[64];
    char   src[640];
    size_t nrec;              /**< records the circuit produces          */
    double exact[NM_MAX_REC]; /**< closed-form P(record = 1)             */
    double joint11;           /**< P(rec0 = 1 AND rec1 = 1), <0 to skip  */
} nm_case_t;

/* ------------------------------------------------------------------ */

static int nm_run_case(const nm_case_t* c) {
    moonlab_stim_error_t err;
    moonlab_stim_circuit_t* circ = moonlab_stim_circuit_parse(c->src, &err);
    if (!circ) {
        fprintf(stderr, "FAIL %s: parse code=%d line=%zu %s\n",
                c->name, err.code, err.line, err.message);
        return 1;
    }
    const size_t nrec = moonlab_stim_circuit_num_measurements(circ);
    if (nrec != c->nrec) {
        fprintf(stderr, "FAIL %s: %zu records, expected %zu\n",
                c->name, nrec, c->nrec);
        moonlab_stim_circuit_free(circ);
        return 1;
    }

    uint8_t* buf = (uint8_t*)malloc((size_t)nrec * NM_CHUNK);
    if (!buf) { moonlab_stim_circuit_free(circ); return 1; }

    uint64_t ones[NM_MAX_REC];
    memset(ones, 0, sizeof(ones));
    uint64_t both = 0;

    for (unsigned blk = 0; blk < NM_BLOCKS; blk++) {
        const uint64_t seed = 0x51ED5EEDULL * (blk + 1u) + 0x9E37U;
        const long rc = moonlab_stim_circuit_sample_measurements(
            circ, NM_CHUNK, seed, 0, buf);
        if (rc != (long)nrec) {
            fprintf(stderr, "FAIL %s: sampler returned %ld\n", c->name, rc);
            free(buf); moonlab_stim_circuit_free(circ); return 1;
        }
        for (size_t r = 0; r < nrec; r++) {
            const uint8_t* row = buf + r * (size_t)NM_CHUNK;
            uint64_t k = 0;
            for (size_t s = 0; s < NM_CHUNK; s++) k += row[s];
            ones[r] += k;
        }
        if (c->joint11 >= 0.0 && nrec >= 2) {
            const uint8_t* r0 = buf;
            const uint8_t* r1 = buf + (size_t)NM_CHUNK;
            uint64_t k = 0;
            for (size_t s = 0; s < NM_CHUNK; s++) k += (uint64_t)(r0[s] & r1[s]);
            both += k;
        }
    }
    free(buf);
    moonlab_stim_circuit_free(circ);

    const double N = (double)NM_SHOTS;
    int bad = 0;
    for (size_t r = 0; r < nrec; r++) {
        const double ex  = c->exact[r];
        const double obs = (double)ones[r] / N;
        if (ex < NM_EXACT_TOL) {
            if (ones[r] != 0) {
                fprintf(stderr, "FAIL %s rec%zu: %llu of %.0f shots fired, "
                        "the instruction is deterministic at 0\n",
                        c->name, r, (unsigned long long)ones[r], N);
                bad = 1;
            }
            continue;
        }
        if (ex > 1.0 - NM_EXACT_TOL) {
            if (ones[r] != NM_SHOTS) {
                fprintf(stderr, "FAIL %s rec%zu: %llu of %.0f shots fired, "
                        "the instruction is deterministic at 1\n",
                        c->name, r, (unsigned long long)ones[r], N);
                bad = 1;
            }
            continue;
        }
        const double se = sqrt(ex * (1.0 - ex) / N);
        const double z  = (obs - ex) / se;
        if (fabs(z) > NM_SIGMA) {
            fprintf(stderr, "FAIL %s rec%zu: observed %.8f, exact %.8f, "
                    "diff %+.3e, z = %+.2f (limit %.1f sigma = %.3e)\n",
                    c->name, r, obs, ex, obs - ex, z, NM_SIGMA, NM_SIGMA * se);
            bad = 1;
        }
    }
    if (c->joint11 >= 0.0 && nrec >= 2) {
        const double ex = c->joint11, obs = (double)both / N;
        if (ex < NM_EXACT_TOL) {
            if (both != 0) {
                fprintf(stderr, "FAIL %s joint: %llu shots fired both records, "
                        "exact 0\n", c->name, (unsigned long long)both);
                bad = 1;
            }
        } else {
            const double se = sqrt(ex * (1.0 - ex) / N);
            const double z  = (obs - ex) / se;
            if (fabs(z) > NM_SIGMA) {
                fprintf(stderr, "FAIL %s joint: observed %.8f, exact %.8f, "
                        "z = %+.2f\n", c->name, obs, ex, z);
                bad = 1;
            }
        }
    }
    return bad;
}

/* ------------------------------------------------------------------ */
/*  Case table                                                         */
/* ------------------------------------------------------------------ */

static void nm_add(nm_case_t** v, size_t* n, size_t* cap, const char* name,
                   const char* src, size_t nrec, double e0, double e1,
                   double joint11) {
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 32;
        *v = (nm_case_t*)realloc(*v, *cap * sizeof(nm_case_t));
    }
    nm_case_t* c = &(*v)[(*n)++];
    memset(c, 0, sizeof(*c));
    snprintf(c->name, sizeof(c->name), "%s", name);
    snprintf(c->src, sizeof(c->src), "%s", src);
    c->nrec = nrec;
    c->exact[0] = e0;
    c->exact[1] = e1;
    c->joint11 = joint11;
}

/* stim's PAULI_CHANNEL_2 argument order. */
static const char* const NM_PC2[15] = {
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
};

/* Build the full table for one error rate.
 *
 * Every expectation below is the probability the Stim instruction reference
 * defines, worked out on the stated input state:
 *   X_ERROR(p)      -- X with probability p; a Z-basis readout reports it.
 *   Y_ERROR(p)      -- Y = iXZ, so the X part flips a Z readout: p.
 *   Z_ERROR(p)      -- commutes with a Z readout: 0.  Conjugating by H turns
 *                      it into an X error: p.
 *   DEPOLARIZE1(p)  -- X, Y, Z each with p/3; two of the three flip a Z
 *                      readout: 2p/3.  The channel is basis-invariant, so the
 *                      H-conjugated form must give the same number.
 *   DEPOLARIZE2(p)  -- the 15 non-identity two-qubit Paulis each with p/15.
 *                      8 of the 15 carry X or Y on the first factor, so each
 *                      single marginal is 8p/15; 4 of the 15 carry X or Y on
 *                      BOTH, so the joint is 4p/15.  The joint is what
 *                      distinguishes the correct 15-way channel from two
 *                      independent single-qubit channels, which would give
 *                      (8p/15)^2 instead.
 *   M(p) / MR(p)    -- the flip is applied to the REPORTED bit only, so a
 *                      following noiseless measurement of the same qubit is
 *                      unaffected, and a preceding X error composes as
 *                      2p(1-p).
 *   MR / R          -- destructive: the following M reads exactly 0.
 *   PAULI_CHANNEL_* -- driven one slot at a time, so a permuted slot table
 *                      shows up as a marginal on the wrong record.
 */
static nm_case_t* nm_build_table(double p, size_t* out_n) {
    nm_case_t* v = NULL;
    size_t n = 0, cap = 0;
    char src[640], name[64];
    const double q = 1.0 - p;

#define ADD1(nm_, src_, e0_) nm_add(&v, &n, &cap, nm_, src_, 1, e0_, 0.0, -1.0)
#define ADD2(nm_, src_, e0_, e1_, j_) nm_add(&v, &n, &cap, nm_, src_, 2, e0_, e1_, j_)

    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nM 0\n", p);
    ADD1("X_ERROR", src, p);
    snprintf(src, sizeof(src), "Y_ERROR(%.9g) 0\nM 0\n", p);
    ADD1("Y_ERROR", src, p);
    snprintf(src, sizeof(src), "Z_ERROR(%.9g) 0\nM 0\n", p);
    ADD1("Z_ERROR", src, 0.0);
    snprintf(src, sizeof(src), "H 0\nZ_ERROR(%.9g) 0\nH 0\nM 0\n", p);
    ADD1("Z_ERROR/X-basis", src, p);
    snprintf(src, sizeof(src), "H 0\nY_ERROR(%.9g) 0\nH 0\nM 0\n", p);
    ADD1("Y_ERROR/X-basis", src, p);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nX_ERROR(%.9g) 0\nM 0\n", p, p);
    ADD1("X_ERROR composed", src, 2.0 * p * q);

    snprintf(src, sizeof(src), "DEPOLARIZE1(%.9g) 0\nM 0\n", p);
    ADD1("DEPOLARIZE1", src, 2.0 * p / 3.0);
    snprintf(src, sizeof(src), "H 0\nDEPOLARIZE1(%.9g) 0\nH 0\nM 0\n", p);
    ADD1("DEPOLARIZE1/X-basis", src, 2.0 * p / 3.0);
    snprintf(src, sizeof(src), "DEPOLARIZE2(%.9g) 0 1\nM 0\nM 1\n", p);
    ADD2("DEPOLARIZE2", src, 8.0 * p / 15.0, 8.0 * p / 15.0, 4.0 * p / 15.0);
    snprintf(src, sizeof(src), "DEPOLARIZE2(%.9g) 1 0\nM 0\nM 1\n", p);
    ADD2("DEPOLARIZE2 swapped", src, 8.0 * p / 15.0, 8.0 * p / 15.0, 4.0 * p / 15.0);
    snprintf(src, sizeof(src), "H 0\nDEPOLARIZE2(%.9g) 0 1\nH 0\nM 0\nM 1\n", p);
    ADD2("DEPOLARIZE2/X-basis", src, 8.0 * p / 15.0, 8.0 * p / 15.0, 4.0 * p / 15.0);

    snprintf(src, sizeof(src), "M(%.9g) 0\n", p);
    ADD1("M(p)", src, p);
    snprintf(src, sizeof(src), "M(%.9g) 0\nM(%.9g) 0\n", p, p);
    ADD2("M(p) repeated", src, p, p, p * p);
    /* rec0 = X xor flip, rec1 = X: both read 1 exactly when the X error fired
     * and the readout flip did not, so the joint is p(1-p) -- which is what
     * pins the flip to the reported bit rather than to the frame. */
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nM(%.9g) 0\nM 0\n", p, p);
    ADD2("X_ERROR then M(p)", src, 2.0 * p * q, p, p * q);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nMR(%.9g) 0\nM 0\n", p, p);
    ADD2("MR(p)", src, 2.0 * p * q, 0.0, 0.0);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nMR 0\nM 0\n", p);
    ADD2("MR resets", src, p, 0.0, 0.0);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nMRZ 0\nM 0\n", p);
    ADD2("MRZ alias", src, p, 0.0, 0.0);
    snprintf(src, sizeof(src), "MX(%.9g) 0\n", p);
    ADD1("MX on |0>", src, 0.5);

    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nR 0\nM 0\n", p);
    ADD1("R clears the frame", src, 0.0);
    snprintf(src, sizeof(src), "R 0\nX_ERROR(%.9g) 0\nM 0\n", p);
    ADD1("R then X_ERROR", src, p);
    snprintf(src, sizeof(src), "RX 0\nZ_ERROR(%.9g) 0\nMX 0\n", p);
    ADD1("RX/MX sees Z_ERROR", src, p);
    snprintf(src, sizeof(src), "RX 0\nX_ERROR(%.9g) 0\nMX 0\n", p);
    ADD1("RX/MX blind to X_ERROR", src, 0.0);
    snprintf(src, sizeof(src), "RY 0\nX_ERROR(%.9g) 0\nMY 0\n", p);
    ADD1("RY/MY sees X_ERROR", src, p);
    snprintf(src, sizeof(src), "RX 0\nDEPOLARIZE1(%.9g) 0\nMX 0\n", p);
    ADD1("DEPOLARIZE1 in X basis", src, 2.0 * p / 3.0);

    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nCNOT 0 1\nM 1\n", p);
    ADD1("CNOT propagates X", src, p);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nH 1\nCZ 0 1\nH 1\nM 1\n", p);
    ADD1("CZ propagates X", src, p);
    snprintf(src, sizeof(src), "X_ERROR(%.9g) 0\nSWAP 0 1\nM 1\n", p);
    ADD1("SWAP propagates X", src, p);

    snprintf(src, sizeof(src), "PAULI_CHANNEL_1(%.9g, 0, 0) 0\nM 0\n", p);
    ADD1("PAULI_CHANNEL_1 px", src, p);
    snprintf(src, sizeof(src), "PAULI_CHANNEL_1(0, %.9g, 0) 0\nM 0\n", p);
    ADD1("PAULI_CHANNEL_1 py", src, p);
    snprintf(src, sizeof(src), "PAULI_CHANNEL_1(0, 0, %.9g) 0\nM 0\n", p);
    ADD1("PAULI_CHANNEL_1 pz", src, 0.0);
    snprintf(src, sizeof(src), "PAULI_CHANNEL_1(%.9g, %.9g, %.9g) 0\nM 0\n",
             p, p / 2.0, p / 4.0);
    ADD1("PAULI_CHANNEL_1 mixed", src, p + p / 2.0);
    snprintf(src, sizeof(src),
             "H 0\nPAULI_CHANNEL_1(%.9g, %.9g, %.9g) 0\nH 0\nM 0\n",
             p, p / 2.0, p / 4.0);
    ADD1("PAULI_CHANNEL_1 mixed/X", src, p / 2.0 + p / 4.0);

    for (int k = 0; k < 15; k++) {
        char args[256];
        size_t off = 0;
        for (int j = 0; j < 15; j++) {
            off += (size_t)snprintf(args + off, sizeof(args) - off, "%s%s",
                                    j ? ", " : "",
                                    j == k ? "P" : "0");
        }
        /* substitute the probability for the placeholder */
        char args2[256];
        size_t o2 = 0;
        for (size_t i = 0; i < off; i++) {
            if (args[i] == 'P')
                o2 += (size_t)snprintf(args2 + o2, sizeof(args2) - o2, "%.9g", p);
            else
                args2[o2++] = args[i];
        }
        args2[o2] = '\0';
        snprintf(src, sizeof(src),
                 "PAULI_CHANNEL_2(%s) 0 1\nM 0\nM 1\n", args2);
        snprintf(name, sizeof(name), "PAULI_CHANNEL_2 %s", NM_PC2[k]);
        const char* nm2 = NM_PC2[k];
        const double e0 = (nm2[0] == 'X' || nm2[0] == 'Y') ? p : 0.0;
        const double e1 = (nm2[1] == 'X' || nm2[1] == 'Y') ? p : 0.0;
        nm_add(&v, &n, &cap, name, src, 2, e0, e1,
               (e0 > 0.0 && e1 > 0.0) ? p : 0.0);
    }

#undef ADD1
#undef ADD2
    *out_n = n;
    return v;
}

/* ------------------------------------------------------------------ */

static int test_marginals_at(double p) {
    size_t n = 0;
    nm_case_t* v = nm_build_table(p, &n);
    if (!v) return 1;
    int bad = 0;
    for (size_t i = 0; i < n; i++) bad |= nm_run_case(&v[i]);
    fprintf(stderr, "  %zu cases at p=%g, %u shots each: %s\n",
            n, p, (unsigned)NM_SHOTS, bad ? "FAIL" : "ok");
    free(v);
    return bad;
}

/* A detector marginal, closed form: the deviation of a single readout from a
 * noiseless trajectory fires exactly when the error fired. */
static int test_detector_marginal(double p) {
    char src[256];
    snprintf(src, sizeof(src),
             "R 0\nX_ERROR(%.9g) 0\nM 0\nDETECTOR rec[-1]\n", p);
    moonlab_stim_error_t err;
    moonlab_stim_circuit_t* c = moonlab_stim_circuit_parse(src, &err);
    ASSERT(c, "detector circuit parses");
    ASSERT(moonlab_stim_circuit_num_detectors(c) == 1, "one detector");

    uint8_t* buf = (uint8_t*)malloc(NM_CHUNK);
    ASSERT(buf, "alloc");
    uint64_t fired = 0;
    for (unsigned blk = 0; blk < NM_BLOCKS; blk++) {
        const uint64_t seed = 0xD37EC7EDULL * (blk + 1u) + 0x1234U;
        const long rc = moonlab_stim_circuit_sample_detectors(
            c, NM_CHUNK, seed, 0, buf, NULL);
        if (rc != 1) { free(buf); moonlab_stim_circuit_free(c); return 1; }
        for (size_t s = 0; s < NM_CHUNK; s++) fired += buf[s];
    }
    free(buf);
    moonlab_stim_circuit_free(c);
    const double obs = (double)fired / (double)NM_SHOTS;
    const double se = sqrt(p * (1.0 - p) / (double)NM_SHOTS);
    ASSERT(fabs(obs - p) <= NM_SIGMA * se, "detector fire rate matches p");
    return 0;
}

/* Same seed, same bytes -- including on the multithreaded path, where the
 * per-block stream is keyed on the absolute shot offset. */
static int test_bit_exact_determinism(void) {
    const char* src =
        "R 0 1\n"
        "DEPOLARIZE1(0.01) 0\n"
        "H 1\n"
        "CNOT 1 0\n"
        "DEPOLARIZE2(0.01) 0 1\n"
        "M(0.01) 0\n"
        "MR 1\n";
    moonlab_stim_error_t err;
    moonlab_stim_circuit_t* c = moonlab_stim_circuit_parse(src, &err);
    ASSERT(c, "determinism circuit parses");
    const size_t nrec = moonlab_stim_circuit_num_measurements(c);
    const size_t shots = 1u << 16;
    uint8_t* a = (uint8_t*)malloc(nrec * shots);
    uint8_t* b = (uint8_t*)malloc(nrec * shots);
    ASSERT(a && b, "alloc");
    for (int threads = 1; threads <= 4; threads *= 4) {
        moonlab_stim_circuit_sample_measurements(c, shots, 0xBEEFCAFEULL, threads, a);
        moonlab_stim_circuit_sample_measurements(c, shots, 0xBEEFCAFEULL, threads, b);
        ASSERT(memcmp(a, b, nrec * shots) == 0, "same seed gives the same bytes");
    }
    free(a); free(b);
    moonlab_stim_circuit_free(c);
    return 0;
}

int main(void) {
    if (test_bit_exact_determinism() != 0) return 1;
    fprintf(stderr, "PASS test_bit_exact_determinism\n");

    /* p = 0.1 maximises sensitivity to a convention error (which scales with
     * p); p = 0.001 is the circuit-level QEC operating point and is where an
     * absolute 3e-5 offset has to be caught. */
    if (test_marginals_at(0.1) != 0) return 1;
    fprintf(stderr, "PASS test_noise_marginals p=0.1\n");
    if (test_marginals_at(0.001) != 0) return 1;
    fprintf(stderr, "PASS test_noise_marginals p=0.001\n");

    if (test_detector_marginal(0.001) != 0) return 1;
    fprintf(stderr, "PASS test_detector_marginal\n");
    return 0;
}
