/**
 * @file quantum_volume.c
 * @brief IBM Quantum Volume implementation.
 */

#include "quantum_volume.h"
#include "../quantum/state.h"
#include "../quantum/gates.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

qs_error_t quantum_state_init(quantum_state_t* state, size_t num_qubits);
void       quantum_state_free(quantum_state_t* state);

/* ---- xorshift64* RNG --------------------------------------------------- */

static uint64_t rng_next(uint64_t* s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}
static double rng_uniform(uint64_t* s) {
    return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}
static double rng_gauss(uint64_t* s) {
    double u = rng_uniform(s);
    if (u < 1e-300) u = 1e-300;
    double v = rng_uniform(s);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

/* ---- Haar-random U(4) via Mezzadri 2007 -------------------------------- */

static void haar_u4(uint64_t* rng, complex_t U[4][4]) {
    /* Fill Z with i.i.d. complex standard normals. */
    complex_t Z[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double r = rng_gauss(rng) / M_SQRT2;
            double im = rng_gauss(rng) / M_SQRT2;
            Z[i][j] = r + _Complex_I * im;
        }
    }
    /* QR via modified Gram-Schmidt. */
    complex_t Q[4][4];
    complex_t R[4][4] = { 0 };
    for (int j = 0; j < 4; j++) {
        complex_t v[4];
        for (int i = 0; i < 4; i++) v[i] = Z[i][j];
        for (int k = 0; k < j; k++) {
            complex_t dot = 0;
            for (int i = 0; i < 4; i++) dot += conj(Q[i][k]) * v[i];
            R[k][j] = dot;
            for (int i = 0; i < 4; i++) v[i] -= dot * Q[i][k];
        }
        double nrm = 0;
        for (int i = 0; i < 4; i++) {
            nrm += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
        }
        nrm = sqrt(nrm);
        if (nrm < 1e-300) nrm = 1e-300;
        R[j][j] = nrm;
        for (int i = 0; i < 4; i++) Q[i][j] = v[i] / nrm;
    }
    /* Phase-correct columns so R has positive real diagonal (makes Q
     * Haar-distributed rather than having a phase bias). */
    for (int j = 0; j < 4; j++) {
        double mag = cabs(R[j][j]);
        complex_t ph = (mag < 1e-300) ? 1.0 : R[j][j] / mag;
        for (int i = 0; i < 4; i++) Q[i][j] /= ph;
    }
    memcpy(U, Q, sizeof(Q));
}

/* ---- double-precision heap qsort helper for median --------------------- */

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* ---- QV one trial: build circuit, exact HOP ---------------------------- */

static int qv_one_trial(size_t width, uint64_t* rng, double* out_hop) {
    quantum_state_t st;
    if (quantum_state_init(&st, width) != QS_SUCCESS) return -1;

    size_t pairs = width / 2;
    if (pairs == 0) { quantum_state_free(&st); return -1; }

    size_t* perm = malloc(width * sizeof(size_t));
    if (!perm) { quantum_state_free(&st); return -1; }

    for (size_t layer = 0; layer < width; layer++) {
        for (size_t i = 0; i < width; i++) perm[i] = i;
        /* Fisher-Yates shuffle. */
        for (size_t i = width - 1; i > 0; i--) {
            size_t j = (size_t)(rng_next(rng) % (uint64_t)(i + 1));
            size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for (size_t p = 0; p < pairs; p++) {
            int a = (int)perm[2 * p];
            int b = (int)perm[2 * p + 1];
            complex_t U[4][4];
            haar_u4(rng, U);
            if (apply_two_qubit_gate(&st, a, b, U) != QS_SUCCESS) {
                free(perm);
                quantum_state_free(&st);
                return -1;
            }
        }
    }
    free(perm);

    /* Compute all probabilities and sort to find the median.  For
     * width d there are 2^d probabilities; width >= 16 pushes memory
     * and time but stays tractable (64 KiB of doubles at d=13, 8 MiB
     * at d=20). */
    uint64_t dim = (uint64_t)1 << width;
    double* probs = malloc((size_t)dim * sizeof(double));
    if (!probs) { quantum_state_free(&st); return -1; }

    for (uint64_t i = 0; i < dim; i++) {
        double re = creal(st.amplitudes[i]);
        double im = cimag(st.amplitudes[i]);
        probs[i] = re * re + im * im;
    }
    double* sorted = malloc((size_t)dim * sizeof(double));
    if (!sorted) { free(probs); quantum_state_free(&st); return -1; }
    memcpy(sorted, probs, (size_t)dim * sizeof(double));
    qsort(sorted, (size_t)dim, sizeof(double), cmp_double);
    double median;
    if (dim % 2 == 0) {
        median = 0.5 * (sorted[dim / 2 - 1] + sorted[dim / 2]);
    } else {
        median = sorted[dim / 2];
    }
    free(sorted);

    double hop = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        if (probs[i] > median) hop += probs[i];
    }
    free(probs);
    quantum_state_free(&st);
    *out_hop = hop;
    return 0;
}

/* ---- public entry point ------------------------------------------------ */

int quantum_volume_run(size_t width,
                       size_t num_trials,
                       uint64_t rng_seed,
                       qv_result_t* out) {
    if (!out) return -1;
    if (width < 2 || width > 16) return -2;
    if (num_trials < 10) return -3;
    if (rng_seed == 0) rng_seed = 0xBADDCAFE0FEEC0DEULL;

    uint64_t rng = rng_seed;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t t = 0; t < num_trials; t++) {
        double hop;
        if (qv_one_trial(width, &rng, &hop) != 0) return -5;
        sum += hop;
        sum_sq += hop * hop;
    }
    double mean = sum / (double)num_trials;
    double var_s = (sum_sq - num_trials * mean * mean) / (double)(num_trials - 1);
    if (var_s < 0.0) var_s = 0.0;
    double sd = sqrt(var_s);
    double se = sd / sqrt((double)num_trials);
    /* One-sided 97.5% normal CI lower bound: mean - 1.96 * se. */
    double lower = mean - 1.96 * se;

    out->width = width;
    out->num_trials = num_trials;
    out->mean_hop = mean;
    out->stddev_hop = sd;
    out->lower_ci_97p5 = lower;
    out->passed = (lower > 2.0 / 3.0) ? 1 : 0;

    return 0;
}
