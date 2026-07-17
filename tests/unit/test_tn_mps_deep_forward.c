/**
 * @file test_tn_mps_deep_forward.c
 * @brief Regression: deep forward-only MPS circuits must stay normalized.
 *
 * A layered ry/rz + FORWARD (control<target) cz circuit at n=12, depth=32 grows
 * the bond dimension past the GPU dispatch threshold (32), so the Metal 2q-gate
 * kernel takes over.  That kernel left the two-site block in a different gauge
 * and normalization convention than the CPU/WebGPU SVD paths (it absorbed S into
 * the left tensor and its float32 SVD drifted the block norm a few percent per
 * gate), so the read-out state vector came back scaled by ~exp(-log_norm_factor)
 * -- a ~1e-2 divergence in the raw probability vector that was chi-independent
 * (a gauge/bookkeeping error, not truncation) and only appeared once the bond
 * crossed the threshold at n>=10.
 *
 * The fix makes the Metal path leave the same convention as the CPU path
 * (tl left-orthonormal, block unit-norm, physical norm parked in
 * log_norm_factor), so tn_mps_amplitude = amp*state->norm reads back correctly.
 *
 * This checks the full state vector against an independent dense reference: the
 * L2 norm must be 1 (the bug drove it to ~0.06) and the normalized state must
 * match the dense state (fidelity).  The CPU path is exact to ~1e-13; the norm
 * tolerance also absorbs the float32 precision of the Metal GPU kernel, whose
 * physical state is otherwise correct.
 */

#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

static uint64_t sm(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double ang(uint64_t *s) { return ((double)(sm(s) >> 11) / (double)(1ULL << 53)) * 2.0 * M_PI; }

static int probe_bit(uint32_t n, uint32_t q) {
    tn_state_config_t c = tn_state_config_create(8, 1e-14);
    tn_mps_state_t *s = tn_mps_create_zero(n, &c);
    tn_apply_x(s, q);
    uint64_t dim = 1ULL << n;
    double complex *sv = calloc(dim, sizeof(double complex));
    tn_mps_to_statevector(s, sv);
    int pos = -1;
    for (uint64_t b = 0; b < dim; b++) {
        double p = creal(sv[b]) * creal(sv[b]) + cimag(sv[b]) * cimag(sv[b]);
        if (p > 0.5) for (int k = 0; k < (int)n; k++) if (b == (1ULL << k)) pos = k;
    }
    free(sv); tn_mps_free(s); return pos;
}

static void d_ry(double complex *sv, uint32_t n, int bp, double th) {
    uint64_t dim = 1ULL << n; double c = cos(th / 2), s = sin(th / 2);
    for (uint64_t b = 0; b < dim; b++) if (!((b >> bp) & 1)) {
        uint64_t b1 = b | (1ULL << bp); double complex a0 = sv[b], a1 = sv[b1];
        sv[b] = c * a0 - s * a1; sv[b1] = s * a0 + c * a1;
    }
}
static void d_rz(double complex *sv, uint32_t n, int bp, double th) {
    uint64_t dim = 1ULL << n; double complex em = cexp(-I * th / 2), ep = cexp(I * th / 2);
    for (uint64_t b = 0; b < dim; b++) sv[b] *= ((b >> bp) & 1) ? ep : em;
}
static void d_cz(double complex *sv, uint32_t n, int bpc, int bpt) {
    uint64_t dim = 1ULL << n;
    for (uint64_t b = 0; b < dim; b++) if (((b >> bpc) & 1) && ((b >> bpt) & 1)) sv[b] = -sv[b];
}

int main(void) {
    fprintf(stdout, "=== deep forward-only MPS circuit vs dense (n=12, depth=32) ===\n");

    const uint32_t n = 12;
    const int depth = 32;
    const uint32_t chi = 256;   /* far above the ~64 needed: exact representation */
    uint64_t dim = 1ULL << n;

    int *bp = malloc(n * sizeof(int));
    for (uint32_t q = 0; q < n; q++) bp[q] = probe_bit(n, q);

    tn_state_config_t cfg = tn_state_config_create(chi, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero(n, &cfg);
    double complex *ref = calloc(dim, sizeof(double complex));
    ref[0] = 1.0;

    uint64_t rng = 0xC0FFEEULL;
    for (int d = 0; d < depth; d++) {
        for (uint32_t q = 0; q < n; q++) {
            double ty = ang(&rng), tz = ang(&rng);
            tn_apply_ry(mps, q, ty); tn_apply_rz(mps, q, tz);
            d_ry(ref, n, bp[q], ty); d_rz(ref, n, bp[q], tz);
        }
        for (uint32_t q = 0; q + 1 < n; q++) {   /* forward cz ladder */
            tn_apply_cz(mps, q, q + 1);
            d_cz(ref, n, bp[q], bp[q + 1]);
        }
    }

    double complex *got = calloc(dim, sizeof(double complex));
    tn_state_error_t rc = tn_mps_to_statevector(mps, got);
    CHECK(rc == TN_STATE_SUCCESS, "to_statevector rc=%d", rc);

    double ng = 0.0;
    for (uint64_t b = 0; b < dim; b++) ng += creal(got[b]) * creal(got[b]) + cimag(got[b]) * cimag(got[b]);
    ng = sqrt(ng);
    fprintf(stdout, "  read-out state-vector L2 norm = %.12f (expect 1)\n", ng);
    CHECK(fabs(ng - 1.0) < 1e-6, "state-vector norm %.9f != 1 (Metal/CPU gate-path norm convention mismatch?)", ng);

    double complex ov = 0.0; double nr = 0.0;
    for (uint64_t b = 0; b < dim; b++) {
        ov += conj(got[b]) * ref[b];
        nr += creal(ref[b]) * creal(ref[b]) + cimag(ref[b]) * cimag(ref[b]);
    }
    nr = sqrt(nr);
    double infid = (ng > 0.0 && nr > 0.0) ? fabs(1.0 - cabs(ov) / (ng * nr)) : 1.0;
    fprintf(stdout, "  1 - fidelity vs dense = %.3e\n", infid);
    CHECK(infid < 1e-9, "physical state diverges from dense (1-F = %.3e)", infid);

    free(bp); free(ref); free(got);
    tn_mps_free(mps);

    if (failures == 0) { fprintf(stdout, "PASS\n"); return 0; }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
