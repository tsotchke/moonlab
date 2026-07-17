/**
 * @file test_tn_gate_order.c
 * @brief Regression for tn_apply_gate_2q operand ordering.
 *
 * When qubit1 > qubit2, tn_apply_gate_2q reorders the indices to lower<higher
 * but must also permute the 4x4 gate matrix (conjugation by SWAP).  The old
 * code swapped the indices without permuting the matrix, so
 * tn_apply_cnot(state, control, target) with control > target silently applied
 * the reversed CNOT (control and target exchanged).
 *
 * This test applies CNOT in both operand orders on adjacent and non-adjacent
 * pairs and compares the full state vector against an independent dense CNOT
 * reference.  The MPS qubit-to-bit convention is probed empirically so the
 * reference needs no assumption about endianness.
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

/* Determine the bit position that MPS qubit q occupies in the basis index
 * used by tn_mps_to_statevector, by preparing X_q|0..0> and finding the set
 * bit of the unique populated basis state. */
static int probe_bitpos(uint32_t n, uint32_t q) {
    tn_state_config_t cfg = tn_state_config_create(8, 1e-14);
    tn_mps_state_t *s = tn_mps_create_zero(n, &cfg);
    if (!s) return -1;
    tn_apply_x(s, q);
    uint64_t dim = (uint64_t)1 << n;
    double complex *sv = calloc(dim, sizeof(double complex));
    tn_mps_to_statevector(s, sv);
    int pos = -1;
    for (uint64_t b = 0; b < dim; b++) {
        double p = creal(sv[b]) * creal(sv[b]) + cimag(sv[b]) * cimag(sv[b]);
        if (p > 0.5) {
            /* b must be a single power of two. */
            for (int k = 0; k < (int)n; k++) if (b == ((uint64_t)1 << k)) pos = k;
        }
    }
    free(sv);
    tn_mps_free(s);
    return pos;
}

/* Dense CNOT(control,target) on a 2^n vector using the given bit positions. */
static void dense_cnot(double complex *sv, uint32_t n, int cbit, int tbit) {
    uint64_t dim = (uint64_t)1 << n;
    double complex *out = calloc(dim, sizeof(double complex));
    for (uint64_t b = 0; b < dim; b++) {
        uint64_t dst = b;
        if ((b >> cbit) & 1) dst = b ^ ((uint64_t)1 << tbit);
        out[dst] = sv[b];
    }
    for (uint64_t b = 0; b < dim; b++) sv[b] = out[b];
    free(out);
}

/* Physical-state agreement via fidelity |<a|b>| / (||a|| ||b||).  The MPS 2q-gate
 * path tracks the two-site block norm in log_norm_factor (which to_statevector
 * does not fold back in), so the raw amplitudes can differ from the dense
 * reference by a real positive scale; comparing the normalized overlap removes
 * that scale (and any global phase) while still catching a wrong permutation --
 * a reversed CNOT produces a genuinely different physical state, dropping the
 * fidelity well below 1. */
static double infidelity(const double complex *a, const double complex *b, uint64_t dim) {
    double complex ov = 0.0;
    double na = 0.0, nb = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        ov += conj(a[i]) * b[i];
        na += creal(a[i]) * creal(a[i]) + cimag(a[i]) * cimag(a[i]);
        nb += creal(b[i]) * creal(b[i]) + cimag(b[i]) * cimag(b[i]);
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0;
    double fid = cabs(ov) / sqrt(na * nb);
    return fabs(1.0 - fid);
}

/* Apply CNOT(control,target) on a random n-qubit state and compare to the
 * dense reference. */
static void check_cnot(uint32_t n, uint32_t control, uint32_t target,
                       const int *bitpos) {
    uint64_t dim = (uint64_t)1 << n;

    /* Random normalized initial state. */
    double complex *init = calloc(dim, sizeof(double complex));
    double nrm = 0.0;
    /* Portable deterministic PRNG (xorshift32): rand_r is POSIX-only and absent
     * on Windows/UCRT. Any deterministic random normalized state exercises the
     * gate-order comparison, so the specific stream does not matter. */
    unsigned int seed = 12345u + control * 131u + target * 17u + n * 7u;
    for (uint64_t b = 0; b < dim; b++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        double re = (double)seed / 4294967296.0 - 0.5;
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        double im = (double)seed / 4294967296.0 - 0.5;
        init[b] = re + im * I;
        nrm += re * re + im * im;
    }
    nrm = sqrt(nrm);
    for (uint64_t b = 0; b < dim; b++) init[b] /= nrm;

    tn_state_config_t cfg = tn_state_config_create(64, 1e-14);
    tn_mps_state_t *mps = tn_mps_from_statevector(init, n, &cfg);
    CHECK(mps != NULL, "from_statevector n=%u", n);
    if (!mps) { free(init); return; }

    tn_gate_error_t err = tn_apply_cnot(mps, control, target);
    CHECK(err == TN_GATE_SUCCESS, "tn_apply_cnot(%u,%u) rc=%d", control, target, err);

    /* Dense reference on the original amplitudes. */
    double complex *ref = calloc(dim, sizeof(double complex));
    for (uint64_t b = 0; b < dim; b++) ref[b] = init[b];
    dense_cnot(ref, n, bitpos[control], bitpos[target]);

    double complex *got = calloc(dim, sizeof(double complex));
    tn_mps_to_statevector(mps, got);

    double infid = infidelity(got, ref, dim);
    fprintf(stdout, "  n=%u CNOT(c=%u,t=%u): 1 - fidelity = %.3e\n",
            n, control, target, infid);
    CHECK(infid < 1e-9, "CNOT(%u,%u) physical-state mismatch (1-F=%.3e)",
          control, target, infid);

    free(init); free(ref); free(got);
    tn_mps_free(mps);
}

int main(void) {
    fprintf(stdout, "=== tn_apply_gate_2q operand-order regression ===\n");

    /* 2 qubits: both orders (adjacent). */
    int bp2[2];
    for (uint32_t q = 0; q < 2; q++) bp2[q] = probe_bitpos(2, q);
    CHECK(bp2[0] >= 0 && bp2[1] >= 0, "bit-position probe (n=2)");
    check_cnot(2, 0, 1, bp2);   /* control < target */
    check_cnot(2, 1, 0, bp2);   /* control > target  <-- the bug case */

    /* 3 qubits: adjacent and non-adjacent, both orders. */
    int bp3[3];
    for (uint32_t q = 0; q < 3; q++) bp3[q] = probe_bitpos(3, q);
    CHECK(bp3[0] >= 0 && bp3[1] >= 0 && bp3[2] >= 0, "bit-position probe (n=3)");
    check_cnot(3, 0, 2, bp3);   /* non-adjacent, control < target */
    check_cnot(3, 2, 0, bp3);   /* non-adjacent, control > target <-- bug case */
    check_cnot(3, 1, 0, bp3);   /* adjacent, control > target      <-- bug case */

    if (failures == 0) {
        fprintf(stdout, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
