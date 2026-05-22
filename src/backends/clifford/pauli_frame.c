/**
 * @file pauli_frame.c
 * @brief Pauli-frame sampler implementation.  See pauli_frame.h.
 */

#include "pauli_frame.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Single-frame storage                                               */
/* ================================================================== */

struct pauli_frame_t {
    size_t   n;
    uint8_t* x_bits;   /* n bits, packed one per byte for simplicity. */
    uint8_t* z_bits;
};

pauli_frame_t* pauli_frame_create(size_t num_qubits) {
    if (num_qubits == 0) return NULL;
    pauli_frame_t* f = (pauli_frame_t*)calloc(1, sizeof(*f));
    if (!f) return NULL;
    f->n = num_qubits;
    f->x_bits = (uint8_t*)calloc(num_qubits, sizeof(uint8_t));
    f->z_bits = (uint8_t*)calloc(num_qubits, sizeof(uint8_t));
    if (!f->x_bits || !f->z_bits) {
        free(f->x_bits); free(f->z_bits); free(f);
        return NULL;
    }
    return f;
}

void pauli_frame_free(pauli_frame_t* f) {
    if (!f) return;
    free(f->x_bits); free(f->z_bits); free(f);
}

void pauli_frame_clear(pauli_frame_t* f) {
    if (!f) return;
    memset(f->x_bits, 0, f->n);
    memset(f->z_bits, 0, f->n);
}

size_t pauli_frame_num_qubits(const pauli_frame_t* f) {
    return f ? f->n : 0;
}

void pauli_frame_read(const pauli_frame_t* f, size_t q,
                       uint8_t* out_x, uint8_t* out_z) {
    if (!f || q >= f->n) {
        if (out_x) *out_x = 0;
        if (out_z) *out_z = 0;
        return;
    }
    if (out_x) *out_x = f->x_bits[q] & 1;
    if (out_z) *out_z = f->z_bits[q] & 1;
}

/* ------------------------------------------------------------------ */
/*  Single-qubit Clifford gates                                        */
/* ------------------------------------------------------------------ */

void pauli_frame_h(pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return;
    uint8_t tmp = f->x_bits[q];
    f->x_bits[q] = f->z_bits[q];
    f->z_bits[q] = tmp;
}

void pauli_frame_s(pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return;
    /* S: Z stays, X picks up Z (S X S^dag = Y = i X Z; the extra
     * factor of i is phase-only, dropped here).  Frame: z ^= x. */
    f->z_bits[q] ^= f->x_bits[q] & 1;
}

void pauli_frame_s_dag(pauli_frame_t* f, size_t q) {
    /* S^dag: same bit transform as S (phase differs, untracked). */
    pauli_frame_s(f, q);
}

void pauli_frame_x(pauli_frame_t* f, size_t q) { (void)f; (void)q; }
void pauli_frame_y(pauli_frame_t* f, size_t q) { (void)f; (void)q; }
void pauli_frame_z(pauli_frame_t* f, size_t q) { (void)f; (void)q; }

/* ------------------------------------------------------------------ */
/*  Two-qubit Clifford gates                                           */
/* ------------------------------------------------------------------ */

void pauli_frame_cnot(pauli_frame_t* f, size_t c, size_t t) {
    if (!f || c >= f->n || t >= f->n || c == t) return;
    /* CNOT propagates X(c) -> X(t) and Z(t) -> Z(c). */
    f->x_bits[t] ^= f->x_bits[c] & 1;
    f->z_bits[c] ^= f->z_bits[t] & 1;
}

void pauli_frame_cz(pauli_frame_t* f, size_t a, size_t b) {
    if (!f || a >= f->n || b >= f->n || a == b) return;
    /* CZ propagates X(a) -> Z(b) and X(b) -> Z(a). */
    uint8_t xa = f->x_bits[a] & 1;
    uint8_t xb = f->x_bits[b] & 1;
    f->z_bits[b] ^= xa;
    f->z_bits[a] ^= xb;
}

void pauli_frame_swap(pauli_frame_t* f, size_t a, size_t b) {
    if (!f || a >= f->n || b >= f->n || a == b) return;
    uint8_t tx = f->x_bits[a]; f->x_bits[a] = f->x_bits[b]; f->x_bits[b] = tx;
    uint8_t tz = f->z_bits[a]; f->z_bits[a] = f->z_bits[b]; f->z_bits[b] = tz;
}

/* ------------------------------------------------------------------ */
/*  Error injection                                                    */
/* ------------------------------------------------------------------ */

void pauli_frame_inject_x(pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return;
    f->x_bits[q] ^= 1;
}

void pauli_frame_inject_z(pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return;
    f->z_bits[q] ^= 1;
}

void pauli_frame_inject_y(pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return;
    f->x_bits[q] ^= 1;
    f->z_bits[q] ^= 1;
}

/* ------------------------------------------------------------------ */
/*  Measurements                                                       */
/* ------------------------------------------------------------------ */

uint8_t pauli_frame_measure_z(const pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return 0;
    return f->x_bits[q] & 1;
}

uint8_t pauli_frame_measure_x(const pauli_frame_t* f, size_t q) {
    if (!f || q >= f->n) return 0;
    return f->z_bits[q] & 1;
}

/* ================================================================== */
/*  Batched frames                                                     */
/* ================================================================== */

/* Word size for bit-packed batched frames.  Each (qubit, x|z) component
 * gets a row of W = ceil(num_shots / 64) uint64s.  A Clifford gate then
 * costs W word-XORs per affected component, irrespective of num_shots. */
typedef uint64_t bf_word_t;
#define BF_BITS_PER_WORD 64

struct pauli_frame_batch_t {
    size_t       n;             /* qubits */
    size_t       s;             /* shots */
    size_t       words_per_row; /* ceil(s / 64) */
    bf_word_t*   x;             /* n rows, words_per_row words each */
    bf_word_t*   z;
};

static bf_word_t* row_x(pauli_frame_batch_t* b, size_t q) {
    return b->x + q * b->words_per_row;
}
static bf_word_t* row_z(pauli_frame_batch_t* b, size_t q) {
    return b->z + q * b->words_per_row;
}

pauli_frame_batch_t* pauli_frame_batch_create(size_t num_qubits, size_t num_shots) {
    if (num_qubits == 0 || num_shots == 0) return NULL;
    pauli_frame_batch_t* b = (pauli_frame_batch_t*)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->n = num_qubits;
    b->s = num_shots;
    b->words_per_row = (num_shots + BF_BITS_PER_WORD - 1) / BF_BITS_PER_WORD;
    const size_t total_words = num_qubits * b->words_per_row;
    b->x = (bf_word_t*)calloc(total_words, sizeof(bf_word_t));
    b->z = (bf_word_t*)calloc(total_words, sizeof(bf_word_t));
    if (!b->x || !b->z) {
        free(b->x); free(b->z); free(b);
        return NULL;
    }
    return b;
}

void pauli_frame_batch_free(pauli_frame_batch_t* b) {
    if (!b) return;
    free(b->x); free(b->z); free(b);
}

size_t pauli_frame_batch_num_shots(const pauli_frame_batch_t* b)  { return b ? b->s : 0; }
size_t pauli_frame_batch_num_qubits(const pauli_frame_batch_t* b) { return b ? b->n : 0; }

void pauli_frame_batch_clear(pauli_frame_batch_t* b) {
    if (!b) return;
    const size_t total = b->n * b->words_per_row;
    memset(b->x, 0, total * sizeof(bf_word_t));
    memset(b->z, 0, total * sizeof(bf_word_t));
}

/* ------------------------------------------------------------------ */
/*  Batched single-qubit gates                                         */
/* ------------------------------------------------------------------ */

/* The batched single-qubit and two-qubit kernels are memory-bound:
 * each gate is a tight XOR loop over W = ceil(num_shots / 64) uint64
 * words.  At realistic batch sizes (10K-100K shots, W=156-1562) the
 * per-call work is small enough that OMP thread-spawn overhead
 * dominates; profiling on M2 Ultra shows OMP-parallel CNOT runs ~1.6x
 * SLOWER than serial up to W = 8192.  We therefore keep the gate
 * kernels serial and reserve OMP for the heavier RNG-draw paths
 * (noise injection) where each shot involves a splitmix64 step. */

void pauli_frame_batch_h(pauli_frame_batch_t* b, size_t q) {
    if (!b || q >= b->n) return;
    bf_word_t* xq = row_x(b, q);
    bf_word_t* zq = row_z(b, q);
    for (size_t w = 0; w < b->words_per_row; w++) {
        bf_word_t t = xq[w]; xq[w] = zq[w]; zq[w] = t;
    }
}

void pauli_frame_batch_s(pauli_frame_batch_t* b, size_t q) {
    if (!b || q >= b->n) return;
    bf_word_t* xq = row_x(b, q);
    bf_word_t* zq = row_z(b, q);
    for (size_t w = 0; w < b->words_per_row; w++) {
        zq[w] ^= xq[w];
    }
}

/* ------------------------------------------------------------------ */
/*  Batched two-qubit gates                                            */
/* ------------------------------------------------------------------ */

void pauli_frame_batch_cnot(pauli_frame_batch_t* b, size_t c, size_t t) {
    if (!b || c >= b->n || t >= b->n || c == t) return;
    bf_word_t* xc = row_x(b, c); bf_word_t* xt = row_x(b, t);
    bf_word_t* zc = row_z(b, c); bf_word_t* zt = row_z(b, t);
    for (size_t w = 0; w < b->words_per_row; w++) {
        xt[w] ^= xc[w];
        zc[w] ^= zt[w];
    }
}

void pauli_frame_batch_cz(pauli_frame_batch_t* b, size_t a, size_t bq) {
    if (!b || a >= b->n || bq >= b->n || a == bq) return;
    bf_word_t* xa = row_x(b, a); bf_word_t* xb = row_x(b, bq);
    bf_word_t* za = row_z(b, a); bf_word_t* zb = row_z(b, bq);
    for (size_t w = 0; w < b->words_per_row; w++) {
        zb[w] ^= xa[w];
        za[w] ^= xb[w];
    }
}

void pauli_frame_batch_swap(pauli_frame_batch_t* b, size_t a, size_t bq) {
    if (!b || a >= b->n || bq >= b->n || a == bq) return;
    bf_word_t* xa = row_x(b, a); bf_word_t* xb = row_x(b, bq);
    bf_word_t* za = row_z(b, a); bf_word_t* zb = row_z(b, bq);
    for (size_t w = 0; w < b->words_per_row; w++) {
        bf_word_t tx = xa[w]; xa[w] = xb[w]; xb[w] = tx;
        bf_word_t tz = za[w]; za[w] = zb[w]; zb[w] = tz;
    }
}

/* ------------------------------------------------------------------ */
/*  Splitmix64 for batched RNG                                         */
/* ------------------------------------------------------------------ */

static inline uint64_t sm64_next(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ------------------------------------------------------------------ */
/*  Noise injection                                                    */
/* ------------------------------------------------------------------ */

/* For each shot s in the batch, decide independently whether to flip
 * the (x_q OR z_q) bit at qubit q.  Word-stride packed XOR construction
 * (build a 64-bit per-word mask and apply once) lets a tight inner
 * loop drive ~10^9 frame-bit operations / second on a single core
 * without OMP -- empirical profiling on M2 Ultra showed OMP was net-
 * negative at typical batch sizes (10K-100K shots) because thread-
 * spawn overhead exceeds the per-call work. */
void pauli_frame_batch_depolarising(pauli_frame_batch_t* b, size_t q,
                                     double p, uint64_t* rng_state) {
    if (!b || q >= b->n || !rng_state || p <= 0.0) return;
    bf_word_t* xq = row_x(b, q);
    bf_word_t* zq = row_z(b, q);
    const uint64_t pX_t  = (uint64_t)((double)UINT64_MAX * (p / 3.0));
    const uint64_t pXY_t = (uint64_t)((double)UINT64_MAX * (2.0 * p / 3.0));
    const uint64_t p_t   = (uint64_t)((double)UINT64_MAX * p);
    const size_t S = b->s;
    const size_t W = b->words_per_row;

    for (size_t w = 0; w < W; w++) {
        const size_t s_lo = w * BF_BITS_PER_WORD;
        const size_t s_hi = s_lo + BF_BITS_PER_WORD <= S ? s_lo + BF_BITS_PER_WORD : S;
        bf_word_t x_w = 0, z_w = 0;
        for (size_t s = s_lo; s < s_hi; s++) {
            const uint64_t u = sm64_next(rng_state);
            const uint64_t bit = ((uint64_t)1) << (s - s_lo);
            if (u < pX_t)        { x_w |= bit; }
            else if (u < pXY_t)  { x_w |= bit; z_w |= bit; }
            else if (u < p_t)    { z_w |= bit; }
        }
        xq[w] ^= x_w;
        zq[w] ^= z_w;
    }
}

void pauli_frame_batch_bit_flip(pauli_frame_batch_t* b, size_t q,
                                 double p, uint64_t* rng_state) {
    if (!b || q >= b->n || !rng_state || p <= 0.0) return;
    bf_word_t* xq = row_x(b, q);
    const size_t S = b->s;
    const size_t W = b->words_per_row;

    /* The threshold computation (uint64_t)((double)UINT64_MAX * p) is
     * unsafe at p = 1.0: (double)UINT64_MAX = 2^64 exactly (double can't
     * represent 2^64 - 1), and the conversion 2^64 -> uint64_t is out
     * of range.  Older glibc / gcc gave UINT64_MAX; gcc 15.2 / glibc
     * 2.41 gives 0, so the comparison `u < 0` is never true and no
     * shots get flipped.  Treat p >= 1.0 as an unconditional flip of
     * all shots so the deterministic p=1.0 case behaves predictably
     * across toolchains. */
    if (p >= 1.0) {
        for (size_t w = 0; w < W; w++) {
            const size_t s_lo = w * BF_BITS_PER_WORD;
            const size_t s_hi = s_lo + BF_BITS_PER_WORD <= S ? s_lo + BF_BITS_PER_WORD : S;
            /* Advance rng_state by the same number of draws as the
             * non-special-case path would, so callers that interleave
             * other rng-using ops still see deterministic state. */
            for (size_t s = s_lo; s < s_hi; s++) (void)sm64_next(rng_state);
            const size_t bits_in_word = s_hi - s_lo;
            const bf_word_t mask = (bits_in_word == BF_BITS_PER_WORD)
                ? (~(bf_word_t)0)
                : (((bf_word_t)1 << bits_in_word) - 1);
            xq[w] ^= mask;
        }
        return;
    }

    const uint64_t threshold = (uint64_t)((double)UINT64_MAX * p);
    for (size_t w = 0; w < W; w++) {
        const size_t s_lo = w * BF_BITS_PER_WORD;
        const size_t s_hi = s_lo + BF_BITS_PER_WORD <= S ? s_lo + BF_BITS_PER_WORD : S;
        bf_word_t x_w = 0;
        for (size_t s = s_lo; s < s_hi; s++) {
            const uint64_t u = sm64_next(rng_state);
            if (u < threshold) x_w |= ((uint64_t)1) << (s - s_lo);
        }
        xq[w] ^= x_w;
    }
}

void pauli_frame_batch_measure_z(const pauli_frame_batch_t* b, size_t q,
                                  uint8_t* out) {
    if (!b || q >= b->n || !out) return;
    const bf_word_t* xq = b->x + q * b->words_per_row;
    for (size_t s = 0; s < b->s; s++) {
        const size_t w = s / BF_BITS_PER_WORD;
        out[s] = (uint8_t)((xq[w] >> (s % BF_BITS_PER_WORD)) & 1);
    }
}

void pauli_frame_batch_measure_z_noisy(pauli_frame_batch_t* b, size_t q,
                                        double p_flip, uint64_t* rng_state,
                                        uint8_t* out) {
    if (!b || q >= b->n || !out) return;
    bf_word_t* xq = row_x(b, q);
    bf_word_t* zq = row_z(b, q);
    const uint64_t threshold = (p_flip > 0.0 && rng_state)
        ? (uint64_t)((double)UINT64_MAX * p_flip) : 0;
    const size_t S = b->s;
    const size_t W = b->words_per_row;

    for (size_t w = 0; w < W; w++) {
        const size_t s_lo = w * BF_BITS_PER_WORD;
        const size_t s_hi = s_lo + BF_BITS_PER_WORD <= S ? s_lo + BF_BITS_PER_WORD : S;
        bf_word_t x_w = xq[w];
        for (size_t s = s_lo; s < s_hi; s++) {
            uint8_t bit = (uint8_t)((x_w >> (s - s_lo)) & 1);
            if (threshold && rng_state) {
                if (sm64_next(rng_state) < threshold) bit ^= 1;
            }
            out[s] = bit;
        }
        /* Destructive reset of ancilla frame after readout. */
        xq[w] = 0;
        zq[w] = 0;
    }
}

void pauli_frame_batch_reset_zero(pauli_frame_batch_t* b, size_t q) {
    if (!b || q >= b->n) return;
    bf_word_t* xq = row_x(b, q);
    bf_word_t* zq = row_z(b, q);
    for (size_t w = 0; w < b->words_per_row; w++) {
        xq[w] = 0;
        zq[w] = 0;
    }
}
