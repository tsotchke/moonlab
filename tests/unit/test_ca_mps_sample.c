/**
 * @file  test_ca_mps_sample.c
 * @brief Born-rule sequential sampling validation.  Runs three regimes
 *        through moonlab_ca_mps_sample_z and cross-checks empirical
 *        marginals + bitstring frequencies against the dense
 *        state-vector ground truth.
 *
 *        Regime 1: Bell state |00> + |11>.  Marginals 0.5 / 0.5,
 *                  correlated bit pattern, only 00 and 11 outcomes.
 *        Regime 2: GHZ_4 on 4 qubits.  Only 0000 and 1111.
 *        Regime 3: H on every qubit + T on qubit 0, exercises the
 *                  non-Clifford layer and a uniform marginal.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <math.h>
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

/* Use a deterministic LCG so the test is reproducible.  Period is
 * 2^32 which is more than enough for these shot counts. */
static uint32_t lcg_state = 0xdeadbeefu;
static double next_uniform(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (lcg_state >> 8) * (1.0 / 16777216.0);  /* 24 bits */
}

static double sv_prob_z(const quantum_state_t* sv, uint32_t q) {
    double p0 = 0.0;
    for (size_t s = 0; s < sv->state_dim; s++) {
        if (((s >> q) & 1u) == 0u) {
            double re = creal(sv->amplitudes[s]);
            double im = cimag(sv->amplitudes[s]);
            p0 += re * re + im * im;
        }
    }
    return p0;
}

static double bitstring_prob(const quantum_state_t* sv, uint64_t b) {
    if (b >= sv->state_dim) return 0.0;
    double re = creal(sv->amplitudes[b]);
    double im = cimag(sv->amplitudes[b]);
    return re * re + im * im;
}

static uint64_t bitstring_from_bits(const uint8_t* bits, uint32_t n) {
    uint64_t b = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (bits[i]) b |= ((uint64_t)1 << i);
    }
    return b;
}

static int run_one(const char* name,
                   const moonlab_ca_mps_t* ca,
                   const quantum_state_t* sv,
                   uint32_t shots) {
    uint32_t n = sv->num_qubits;
    double* uniforms = (double*)malloc(shots * (size_t)n * sizeof(double));
    uint8_t* bits    = (uint8_t*)malloc(shots * (size_t)n * sizeof(uint8_t));
    if (!uniforms || !bits) { free(uniforms); free(bits); return -1; }
    for (size_t i = 0; i < (size_t)shots * n; i++) uniforms[i] = next_uniform();

    ca_mps_error_t e = moonlab_ca_mps_sample_z(ca, shots, uniforms, bits);
    CHECK(e == CA_MPS_SUCCESS, "%s sample_z rc=%d", name, e);
    if (e != CA_MPS_SUCCESS) { free(uniforms); free(bits); return -1; }

    /* Empirical marginals. */
    for (uint32_t q = 0; q < n; q++) {
        uint64_t count_zero = 0;
        for (uint32_t s = 0; s < shots; s++) {
            if (bits[s * (size_t)n + q] == 0) count_zero++;
        }
        double emp = (double)count_zero / (double)shots;
        double ref = sv_prob_z(sv, q);
        double tol = 5.0 / sqrt((double)shots);
        CHECK(fabs(emp - ref) < tol,
              "%s q%u marginal emp=%.4f ref=%.4f (tol=%.3f)",
              name, q, emp, ref, tol);
    }

    /* Empirical bitstring frequencies on the support. */
    size_t dim = (size_t)1 << n;
    uint64_t* hist = (uint64_t*)calloc(dim, sizeof(uint64_t));
    if (!hist) { free(uniforms); free(bits); return -1; }
    for (uint32_t s = 0; s < shots; s++) {
        hist[bitstring_from_bits(&bits[s * (size_t)n], n)]++;
    }
    for (uint64_t b = 0; b < dim; b++) {
        double ref = bitstring_prob(sv, b);
        double emp = (double)hist[b] / (double)shots;
        double tol = 5.0 / sqrt((double)shots) + 1e-6;
        if (ref < 1e-12 && emp == 0.0) continue;
        CHECK(fabs(emp - ref) < tol,
              "%s string %02llu emp=%.4f ref=%.4f",
              name, (unsigned long long)b, emp, ref);
    }

    free(hist);
    free(uniforms);
    free(bits);
    return 0;
}

int main(void) {
    fprintf(stdout, "=== test_ca_mps_sample ===\n\n");

    /* Regime 1: Bell pair. */
    {
        const uint32_t n = 2;
        moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, 16);
        quantum_state_t*  sv = quantum_state_create((int)n);
        if (!ca || !sv) return 2;
        moonlab_ca_mps_h(ca, 0);              gate_hadamard(sv, 0);
        moonlab_ca_mps_cnot(ca, 0, 1);        gate_cnot(sv, 0, 1);
        run_one("bell", ca, sv, 4096);
        moonlab_ca_mps_free(ca);
        quantum_state_free(sv);
    }

    /* Regime 2: GHZ_4. */
    {
        const uint32_t n = 4;
        moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, 16);
        quantum_state_t*  sv = quantum_state_create((int)n);
        if (!ca || !sv) return 2;
        moonlab_ca_mps_h(ca, 0);              gate_hadamard(sv, 0);
        for (uint32_t k = 1; k < n; k++) {
            moonlab_ca_mps_cnot(ca, k - 1, k);
            gate_cnot(sv, k - 1, k);
        }
        run_one("ghz4", ca, sv, 4096);
        moonlab_ca_mps_free(ca);
        quantum_state_free(sv);
    }

    /* Regime 3: Hadamard wall + T on qubit 0.  Non-Clifford. */
    {
        const uint32_t n = 3;
        moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, 16);
        quantum_state_t*  sv = quantum_state_create((int)n);
        if (!ca || !sv) return 2;
        for (uint32_t q = 0; q < n; q++) {
            moonlab_ca_mps_h(ca, q); gate_hadamard(sv, q);
        }
        moonlab_ca_mps_t_gate(ca, 0); gate_t(sv, 0);
        run_one("h_wall_t", ca, sv, 8192);
        moonlab_ca_mps_free(ca);
        quantum_state_free(sv);
    }

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
