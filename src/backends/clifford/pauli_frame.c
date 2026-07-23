/**
 * @file pauli_frame.c
 * @brief Pauli-frame sampler implementation.  See pauli_frame.h.
 */

#include "pauli_frame.h"
#include "clifford.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  SIMD backend selection                                             */
/*                                                                    */
/*  The batched frame gate kernels are word-parallel XOR / row-swap    */
/*  loops over W = ceil(num_shots/64) uint64 words.  We widen the      */
/*  inner loop to the host lane width: NEON (2x u64) on AArch64,       */
/*  AVX-512 (8x) / AVX2 (4x) on x86, scalar u64 otherwise.  Build with */
/*  -DQSIM_NATIVE_ARCH=ON (-> -mcpu=native / -march=native) so the     */
/*  right macro is defined.                                            */
/* ------------------------------------------------------------------ */
#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define PF_SIMD_NEON 1
#  define PF_SIMD_NAME "neon"
#  define PF_SIMD_LANES 2
#elif defined(__AVX512F__)
#  include <immintrin.h>
#  define PF_SIMD_AVX512 1
#  define PF_SIMD_NAME "avx512"
#  define PF_SIMD_LANES 8
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define PF_SIMD_AVX2 1
#  define PF_SIMD_NAME "avx2"
#  define PF_SIMD_LANES 4
#else
#  define PF_SIMD_NAME "scalar"
#  define PF_SIMD_LANES 1
#endif

const char* pauli_frame_simd_backend(void) { return PF_SIMD_NAME; }
int         pauli_frame_simd_lanes(void)   { return PF_SIMD_LANES; }

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
 * words.  Below, the inner loop is widened to the host SIMD lane width
 * (NEON 2x u64 / AVX2 4x / AVX-512 8x), which closes the per-core gap
 * to Stim's frame simulator on a single thread; across-shot threading
 * is applied at the circuit-sampler level, not per gate, because a
 * single gate's W words is too little work to amortise thread spawn. */

/* d[w] ^= s[w] for w in [0,W), SIMD-widened. */
static inline void pf_row_xor(bf_word_t* restrict d,
                              const bf_word_t* restrict s, size_t W) {
    size_t w = 0;
#if defined(PF_SIMD_NEON)
    for (; w + 2 <= W; w += 2) {
        uint64x2_t vd = vld1q_u64(&d[w]);
        uint64x2_t vs = vld1q_u64(&s[w]);
        vst1q_u64(&d[w], veorq_u64(vd, vs));
    }
#elif defined(PF_SIMD_AVX512)
    for (; w + 8 <= W; w += 8) {
        __m512i vd = _mm512_loadu_si512((const void*)&d[w]);
        __m512i vs = _mm512_loadu_si512((const void*)&s[w]);
        _mm512_storeu_si512((void*)&d[w], _mm512_xor_si512(vd, vs));
    }
#elif defined(PF_SIMD_AVX2)
    for (; w + 4 <= W; w += 4) {
        __m256i vd = _mm256_loadu_si256((const __m256i*)&d[w]);
        __m256i vs = _mm256_loadu_si256((const __m256i*)&s[w]);
        _mm256_storeu_si256((__m256i*)&d[w], _mm256_xor_si256(vd, vs));
    }
#endif
    for (; w < W; w++) d[w] ^= s[w];
}

/* swap rows a[w] <-> b[w] for w in [0,W), SIMD-widened. */
static inline void pf_row_swap(bf_word_t* restrict a,
                               bf_word_t* restrict b, size_t W) {
    size_t w = 0;
#if defined(PF_SIMD_NEON)
    for (; w + 2 <= W; w += 2) {
        uint64x2_t va = vld1q_u64(&a[w]);
        uint64x2_t vb = vld1q_u64(&b[w]);
        vst1q_u64(&a[w], vb);
        vst1q_u64(&b[w], va);
    }
#elif defined(PF_SIMD_AVX512)
    for (; w + 8 <= W; w += 8) {
        __m512i va = _mm512_loadu_si512((const void*)&a[w]);
        __m512i vb = _mm512_loadu_si512((const void*)&b[w]);
        _mm512_storeu_si512((void*)&a[w], vb);
        _mm512_storeu_si512((void*)&b[w], va);
    }
#elif defined(PF_SIMD_AVX2)
    for (; w + 4 <= W; w += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&a[w]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&b[w]);
        _mm256_storeu_si256((__m256i*)&a[w], vb);
        _mm256_storeu_si256((__m256i*)&b[w], va);
    }
#endif
    for (; w < W; w++) { bf_word_t t = a[w]; a[w] = b[w]; b[w] = t; }
}

void pauli_frame_batch_h(pauli_frame_batch_t* b, size_t q) {
    if (!b || q >= b->n) return;
    pf_row_swap(row_x(b, q), row_z(b, q), b->words_per_row);
}

void pauli_frame_batch_s(pauli_frame_batch_t* b, size_t q) {
    if (!b || q >= b->n) return;
    pf_row_xor(row_z(b, q), row_x(b, q), b->words_per_row);
}

/* ------------------------------------------------------------------ */
/*  Batched two-qubit gates                                            */
/* ------------------------------------------------------------------ */

void pauli_frame_batch_cnot(pauli_frame_batch_t* b, size_t c, size_t t) {
    if (!b || c >= b->n || t >= b->n || c == t) return;
    const size_t W = b->words_per_row;
    pf_row_xor(row_x(b, t), row_x(b, c), W);   /* x_t ^= x_c */
    pf_row_xor(row_z(b, c), row_z(b, t), W);   /* z_c ^= z_t */
}

void pauli_frame_batch_cz(pauli_frame_batch_t* b, size_t a, size_t bq) {
    if (!b || a >= b->n || bq >= b->n || a == bq) return;
    const size_t W = b->words_per_row;
    pf_row_xor(row_z(b, bq), row_x(b, a), W);  /* z_b ^= x_a */
    pf_row_xor(row_z(b, a), row_x(b, bq), W);  /* z_a ^= x_b */
}

void pauli_frame_batch_swap(pauli_frame_batch_t* b, size_t a, size_t bq) {
    if (!b || a >= b->n || bq >= b->n || a == bq) return;
    const size_t W = b->words_per_row;
    pf_row_swap(row_x(b, a), row_x(b, bq), W);
    pf_row_swap(row_z(b, a), row_z(b, bq), W);
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
/*  Sparse Bernoulli noise sampling                                    */
/*                                                                     */
/* Drawing one variate per shot costs O(shots) per channel however small
 * p is.  QEC circuits are dominated by noise instructions at p ~ 1e-3,
 * where all but a thousandth of those draws produce nothing.  Sample the
 * gap to the next affected shot from the geometric distribution instead:
 * with P(gap = k) = p(1-p)^(k-1), a uniform u in (0,1) gives
 * gap = floor(log u / log(1-p)) + 1, so the cost falls to O(p * shots)
 * draws.  This is the same sparse-error strategy stim uses, and it is what
 * makes a noisy circuit tractable at 10^6 shots.                        */

/* Gap to the next selected index, >= 1.  inv_log_q is 1/log(1-p). */
static inline size_t pf_next_gap(double inv_log_q, uint64_t* rng) {
    /* 53-bit uniform in (0,1); the +2^-53 floor keeps log() finite. */
    double u = (double)(sm64_next(rng) >> 11) * 0x1.0p-53;
    if (u <= 0.0) u = 0x1.0p-53;
    return (size_t)(log(u) * inv_log_q) + 1;
}

static inline void pf_flip_all(bf_word_t* row, size_t S) {
    const size_t W = (S + BF_BITS_PER_WORD - 1) / BF_BITS_PER_WORD;
    for (size_t w = 0; w < W; w++) {
        const size_t lo = w * BF_BITS_PER_WORD;
        const size_t bits = (lo + BF_BITS_PER_WORD <= S) ? BF_BITS_PER_WORD : S - lo;
        row[w] ^= (bits == BF_BITS_PER_WORD) ? ~(bf_word_t)0
                                             : (((bf_word_t)1 << bits) - 1);
    }
}

/* X_ERROR / Z_ERROR: flip one component on a Bernoulli(p) subset of shots. */
static void pf_noise_1comp(bf_word_t* row, size_t S, double p, uint64_t* rng) {
    if (p <= 0.0 || S == 0) return;
    if (p >= 1.0) { pf_flip_all(row, S); return; }
    const double ilq = 1.0 / log1p(-p);
    long long pos = -1;
    for (;;) {
        pos += (long long)pf_next_gap(ilq, rng);
        if (pos >= (long long)S) return;
        row[(size_t)pos >> 6] ^= (bf_word_t)1 << ((size_t)pos & 63);
    }
}

/* Y_ERROR: flips both components on the SAME shots (not two independent
 * draws), so the x and z rows must share one position stream. */
static void pf_noise_y(bf_word_t* xr, bf_word_t* zr, size_t S,
                       double p, uint64_t* rng) {
    if (p <= 0.0 || S == 0) return;
    if (p >= 1.0) { pf_flip_all(xr, S); pf_flip_all(zr, S); return; }
    const double ilq = 1.0 / log1p(-p);
    long long pos = -1;
    for (;;) {
        pos += (long long)pf_next_gap(ilq, rng);
        if (pos >= (long long)S) return;
        const bf_word_t bit = (bf_word_t)1 << ((size_t)pos & 63);
        xr[(size_t)pos >> 6] ^= bit;
        zr[(size_t)pos >> 6] ^= bit;
    }
}

/* DEPOLARIZE1: on a Bernoulli(p) subset, apply X, Y or Z uniformly.
 * X sets x; Y sets x and z; Z sets z. */
static void pf_noise_depol1(bf_word_t* xr, bf_word_t* zr, size_t S,
                            double p, uint64_t* rng) {
    if (p <= 0.0 || S == 0) return;
    if (p >= 1.0) {
        for (size_t s = 0; s < S; s++) {
            const bf_word_t bit = (bf_word_t)1 << (s & 63);
            const unsigned k = (unsigned)(sm64_next(rng) % 3u);
            if (k <= 1) xr[s >> 6] ^= bit;
            if (k >= 1) zr[s >> 6] ^= bit;
        }
        return;
    }
    const double ilq = 1.0 / log1p(-p);
    long long pos = -1;
    for (;;) {
        pos += (long long)pf_next_gap(ilq, rng);
        if (pos >= (long long)S) return;
        const bf_word_t bit = (bf_word_t)1 << ((size_t)pos & 63);
        const size_t w = (size_t)pos >> 6;
        const unsigned k = (unsigned)(sm64_next(rng) % 3u);  /* 0=X 1=Y 2=Z */
        if (k <= 1) xr[w] ^= bit;
        if (k >= 1) zr[w] ^= bit;
    }
}

/* DEPOLARIZE2: on a Bernoulli(p) subset, apply one of the 15 non-identity
 * two-qubit Paulis uniformly.  Index 1..15 encodes (x0,z0,x1,z1). */
static void pf_noise_depol2(bf_word_t* xa, bf_word_t* za,
                            bf_word_t* xb, bf_word_t* zb, size_t S,
                            double p, uint64_t* rng) {
    if (p <= 0.0 || S == 0) return;
    const double ilq = 1.0 / log1p(-p);
    long long pos = -1;
    for (;;) {
        pos += (long long)pf_next_gap(ilq, rng);
        if (pos >= (long long)S) return;
        const bf_word_t bit = (bf_word_t)1 << ((size_t)pos & 63);
        const size_t w = (size_t)pos >> 6;
        const unsigned k = 1u + (unsigned)(sm64_next(rng) % 15u);
        if (k & 1u) xa[w] ^= bit;
        if (k & 2u) za[w] ^= bit;
        if (k & 4u) xb[w] ^= bit;
        if (k & 8u) zb[w] ^= bit;
    }
}

/* Flip a Bernoulli(p) subset of already-written measurement result bytes. */
static void pf_noise_flip_bytes(uint8_t* dst, size_t S, double p, uint64_t* rng) {
    if (p <= 0.0 || S == 0) return;
    if (p >= 1.0) { for (size_t s = 0; s < S; s++) dst[s] ^= 1u; return; }
    const double ilq = 1.0 / log1p(-p);
    long long pos = -1;
    for (;;) {
        pos += (long long)pf_next_gap(ilq, rng);
        if (pos >= (long long)S) return;
        dst[(size_t)pos] ^= 1u;
    }
}

/* Derive an independent RNG stream for the shot block beginning at
 * absolute shot index `offset`.
 *
 * This must NOT be `seed + k * 0x9E3779B97F4A7C15`: that constant is exactly
 * the increment sm64_next applies to its state per draw, so an additive
 * stride hands block k the state block 0 reaches after k draws.  Every block
 * then walks a single shared stream at a k-step offset and their shots are
 * correlated -- measured as up to 7.4 sigma bias in the GHZ P(all-1) statistic
 * on the multithreaded path while the single-threaded path stayed within
 * 1.3 sigma.  Avalanching the (seed, offset) pair destroys that additive
 * structure so neighbouring blocks differ in all bits.
 *
 * Keying on the absolute shot offset rather than the thread id means a block
 * covering the same shots draws the same randomness wherever it is scheduled;
 * for a fixed thread count the sample stream is reproducible.  (Changing the
 * thread count repartitions the blocks and therefore changes the stream.) */
static inline uint64_t pf_stream_seed(uint64_t seed, uint64_t offset) {
    uint64_t z = seed ^ (offset * 0xD1B54A32D192ED03ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 29)) * 0x94D049BB133111EBULL;
    z ^= z >> 32;
    return z ? z : 0x9E3779B97F4A7C15ULL;
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

/* ================================================================== */
/*  Circuit-level batch shot sampler                                   */
/* ================================================================== */

size_t pauli_frame_circuit_num_measurements(const pf_circuit_op_t* ops,
                                            size_t num_ops) {
    if (!ops) return 0;
    size_t m = 0;
    for (size_t i = 0; i < num_ops; i++)
        if (ops[i].kind == PF_OP_MEASURE || ops[i].kind == PF_OP_MEASURE_NOISY) m++;
    return m;
}

/* Reference pass: one deterministic tableau run recording, per MEASURE op,
 * the outcome bit and whether it was random (kind 1) or deterministic
 * (kind 0).  This anchors the affine outcome offset (sign flips from
 * X/Y/Z, and one valid choice for each random measurement); the frame
 * pass then spans the free subspace around it. */
static int pf_compute_reference(size_t n, const pf_circuit_op_t* ops,
                                size_t num_ops, uint64_t seed,
                                uint8_t* m_ref, uint8_t* m_kind) {
    clifford_tableau_t* t = clifford_tableau_create(n);
    if (!t) return -1;
    uint64_t rng = seed ? seed : 0xA5A5A5A5DEADBEEFULL;
    size_t mi = 0;
    for (size_t i = 0; i < num_ops; i++) {
        const uint32_t q0 = ops[i].q0, q1 = ops[i].q1;
        int outcome = 0, kind = 0;
        switch (ops[i].kind) {
            case PF_OP_H:    clifford_h(t, q0); break;
            case PF_OP_S:    clifford_s(t, q0); break;
            case PF_OP_S_DAG:clifford_s_dag(t, q0); break;
            case PF_OP_X:    clifford_x(t, q0); break;
            case PF_OP_Y:    clifford_y(t, q0); break;
            case PF_OP_Z:    clifford_z(t, q0); break;
            case PF_OP_CNOT: clifford_cnot(t, q0, q1); break;
            case PF_OP_CZ:   clifford_cz(t, q0, q1); break;
            case PF_OP_SWAP: clifford_swap(t, q0, q1); break;
            case PF_OP_RESET:
                clifford_measure(t, q0, &rng, &outcome, &kind);
                if (outcome) clifford_x(t, q0);
                break;
            case PF_OP_MEASURE:
            case PF_OP_MEASURE_NOISY:
                clifford_measure(t, q0, &rng, &outcome, &kind);
                m_ref[mi]  = (uint8_t)(outcome & 1);
                m_kind[mi] = (uint8_t)(kind & 1);
                mi++;
                break;
            /* Noise channels are per-shot deviations from this reference
             * trajectory, so they contribute nothing here.  Applying them
             * would corrupt the very baseline the frames are measured
             * against. */
            case PF_OP_X_ERROR: case PF_OP_Y_ERROR: case PF_OP_Z_ERROR:
            case PF_OP_DEPOLARIZE1: case PF_OP_DEPOLARIZE2:
                break;
            default: break;
        }
    }
    clifford_tableau_free(t);
    return 0;
}

/* Fill a W-word frame row with fresh per-shot random bits. */
static inline void pf_row_rand(bf_word_t* r, size_t W, uint64_t* rng) {
    for (size_t w = 0; w < W; w++) r[w] = sm64_next(rng);
}

/* Run the full circuit over one shot block, writing measurement outcomes
 * measurement-major into out[ m*total_shots + (global_offset + shot) ]. */
/* Run one shot block.  Measurement mi of shot s is written to
 * mdst[mi * mstride + s], so the caller chooses the layout: the measurement
 * sampler points mdst straight into the caller's full (nmeas x total_shots)
 * buffer with mstride = total_shots, while the detector sampler points it at
 * a small block-local buffer and reduces to detectors before returning. */
static int pf_run_block(size_t n, const pf_circuit_op_t* ops, size_t num_ops,
                        const uint8_t* m_ref, const uint8_t* m_kind,
                        size_t block_shots, uint64_t seed,
                        uint8_t* mdst, size_t mstride) {
    pauli_frame_batch_t* b = pauli_frame_batch_create(n, block_shots);
    if (!b) return -1;
    const size_t W = b->words_per_row;
    uint64_t rng = seed ? seed : 0x1ULL;

    /* Initial state |0>^n: x-frame is zero (calloc); z-frame is free and
     * seeded random so that a later H moves genuine entropy into the
     * measured X-component. */
    for (size_t q = 0; q < n; q++) pf_row_rand(row_z(b, q), W, &rng);

    size_t mi = 0;
    for (size_t i = 0; i < num_ops; i++) {
        const uint32_t q0 = ops[i].q0, q1 = ops[i].q1;
        switch (ops[i].kind) {
            case PF_OP_H:    pauli_frame_batch_h(b, q0); break;
            case PF_OP_S:    pauli_frame_batch_s(b, q0); break;
            case PF_OP_S_DAG:pauli_frame_batch_s(b, q0); break;
            case PF_OP_X: case PF_OP_Y: case PF_OP_Z: break; /* frame no-op */
            case PF_OP_CNOT: pauli_frame_batch_cnot(b, q0, q1); break;
            case PF_OP_CZ:   pauli_frame_batch_cz(b, q0, q1); break;
            case PF_OP_SWAP: pauli_frame_batch_swap(b, q0, q1); break;
            case PF_OP_RESET: {
                bf_word_t* xq = row_x(b, q0);
                for (size_t w = 0; w < W; w++) xq[w] = 0;
                pf_row_rand(row_z(b, q0), W, &rng);
                break;
            }
            case PF_OP_MEASURE:
            case PF_OP_MEASURE_NOISY: {
                const bf_word_t* xq = row_x(b, q0);
                const uint8_t mr = m_ref[mi];
                uint8_t* dst = mdst + (size_t)mi * mstride;
                for (size_t s = 0; s < block_shots; s++)
                    dst[s] = (uint8_t)((xq[s >> 6] >> (s & 63)) & 1) ^ mr;
                /* Measurement error flips the REPORTED outcome only: the
                 * frame is untouched, so a repeated measurement of the same
                 * qubit still agrees with the state.  (This is stim's M(p)
                 * semantics, and is what makes a flipped syndrome bit show
                 * up as two detector events rather than one.) */
                if (ops[i].kind == PF_OP_MEASURE_NOISY)
                    pf_noise_flip_bytes(dst, block_shots, ops[i].p, &rng);
                /* A random measurement injects fresh Z-frame entropy so a
                 * later basis change yields an independent outcome; the
                 * X-frame is preserved so downstream deterministic
                 * measurements stay correlated to this result. */
                if (m_kind[mi]) pf_row_rand(row_z(b, q0), W, &rng);
                mi++;
                break;
            }
            case PF_OP_X_ERROR:
                pf_noise_1comp(row_x(b, q0), block_shots, ops[i].p, &rng);
                break;
            case PF_OP_Z_ERROR:
                pf_noise_1comp(row_z(b, q0), block_shots, ops[i].p, &rng);
                break;
            case PF_OP_Y_ERROR:
                pf_noise_y(row_x(b, q0), row_z(b, q0), block_shots, ops[i].p, &rng);
                break;
            case PF_OP_DEPOLARIZE1:
                pf_noise_depol1(row_x(b, q0), row_z(b, q0), block_shots,
                                ops[i].p, &rng);
                break;
            case PF_OP_DEPOLARIZE2:
                pf_noise_depol2(row_x(b, q0), row_z(b, q0),
                                row_x(b, q1), row_z(b, q1), block_shots,
                                ops[i].p, &rng);
                break;
            default: break;
        }
    }
    pauli_frame_batch_free(b);
    return 0;
}

long pauli_frame_batch_sample_circuit(size_t num_qubits,
                                      const pf_circuit_op_t* ops, size_t num_ops,
                                      size_t num_shots, uint64_t seed,
                                      int num_threads, uint8_t* out) {
    if (num_qubits == 0 || !ops || num_shots == 0 || !out) return -1;
    const size_t nmeas = pauli_frame_circuit_num_measurements(ops, num_ops);
    if (nmeas == 0) return 0;

    uint8_t* m_ref  = (uint8_t*)malloc(nmeas);
    uint8_t* m_kind = (uint8_t*)malloc(nmeas);
    if (!m_ref || !m_kind) { free(m_ref); free(m_kind); return -1; }
    if (pf_compute_reference(num_qubits, ops, num_ops, seed, m_ref, m_kind)) {
        free(m_ref); free(m_kind); return -1;
    }

    int nthreads = num_threads;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
    if (nthreads <= 0) nthreads = 1;
#endif
    if ((size_t)nthreads > num_shots) nthreads = (int)num_shots;
    if (nthreads < 1) nthreads = 1;

    const size_t base = num_shots / (size_t)nthreads;
    const size_t rem  = num_shots % (size_t)nthreads;
    int err = 0;

#ifdef _OPENMP
#   pragma omp parallel for num_threads(nthreads) schedule(static, 1) reduction(|:err)
#endif
    for (int tid = 0; tid < nthreads; tid++) {
        /* Contiguous, non-overlapping shot ranges; the first `rem` blocks
         * take one extra shot so all shots are covered. */
        size_t bs = base + ((size_t)tid < rem ? 1 : 0);
        size_t off = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
        if (bs == 0) continue;
        uint64_t bseed = pf_stream_seed(seed, (uint64_t)off);
        if (pf_run_block(num_qubits, ops, num_ops, m_ref, m_kind,
                         bs, bseed, out + off, num_shots) != 0)
            err |= 1;
    }

    free(m_ref); free(m_kind);
    return err ? -1 : (long)nmeas;
}

long pauli_frame_batch_sample_detectors(size_t num_qubits,
                                        const pf_circuit_op_t* ops, size_t num_ops,
                                        const size_t* det_offsets,
                                        const uint32_t* det_indices,
                                        size_t num_detectors,
                                        size_t num_shots, uint64_t seed,
                                        int num_threads, uint8_t* out) {
    if (num_qubits == 0 || !ops || num_shots == 0 || !out) return -1;
    if (num_detectors == 0 || !det_offsets || !det_indices) return -1;
    const size_t nmeas = pauli_frame_circuit_num_measurements(ops, num_ops);
    if (nmeas == 0) return -1;
    for (size_t d = 0; d < num_detectors; d++) {
        if (det_offsets[d] > det_offsets[d + 1]) return -1;
        for (size_t k = det_offsets[d]; k < det_offsets[d + 1]; k++)
            if (det_indices[k] >= nmeas) return -1;
    }

    uint8_t* m_ref  = (uint8_t*)malloc(nmeas);
    uint8_t* m_kind = (uint8_t*)malloc(nmeas);
    if (!m_ref || !m_kind) { free(m_ref); free(m_kind); return -1; }
    if (pf_compute_reference(num_qubits, ops, num_ops, seed, m_ref, m_kind)) {
        free(m_ref); free(m_kind); return -1;
    }

    /* A detector reports the DEVIATION from the noiseless trajectory, so the
     * reference parity of its measurement set is XORed out.  Without this a
     * detector whose noiseless parity is 1 would fire on every shot. */
    uint8_t* det_ref = (uint8_t*)calloc(num_detectors, 1);
    if (!det_ref) { free(m_ref); free(m_kind); return -1; }
    for (size_t d = 0; d < num_detectors; d++) {
        uint8_t acc = 0;
        for (size_t k = det_offsets[d]; k < det_offsets[d + 1]; k++)
            acc ^= m_ref[det_indices[k]];
        det_ref[d] = acc;
    }

    int nthreads = num_threads;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
    if (nthreads <= 0) nthreads = 1;
#endif
    if ((size_t)nthreads > num_shots) nthreads = (int)num_shots;
    if (nthreads < 1) nthreads = 1;

    const size_t base = num_shots / (size_t)nthreads;
    const size_t rem  = num_shots % (size_t)nthreads;
    int err = 0;

#ifdef _OPENMP
#   pragma omp parallel for num_threads(nthreads) schedule(static, 1) reduction(|:err)
#endif
    for (int tid = 0; tid < nthreads; tid++) {
        size_t bs = base + ((size_t)tid < rem ? 1 : 0);
        size_t off = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
        if (bs == 0) continue;

        /* Block-local measurement buffer: detectors are reduced here, so the
         * full nmeas x num_shots measurement record is never materialised. */
        uint8_t* mbuf = (uint8_t*)malloc(nmeas * bs);
        if (!mbuf) { err |= 1; continue; }
        uint64_t bseed = pf_stream_seed(seed, (uint64_t)off);
        if (pf_run_block(num_qubits, ops, num_ops, m_ref, m_kind,
                         bs, bseed, mbuf, bs) != 0) {
            free(mbuf); err |= 1; continue;
        }
        for (size_t d = 0; d < num_detectors; d++) {
            uint8_t* dst = out + d * num_shots + off;
            const size_t k0 = det_offsets[d], k1 = det_offsets[d + 1];
            if (k0 == k1) { memset(dst, 0, bs); continue; }
            const uint8_t* src = mbuf + (size_t)det_indices[k0] * bs;
            for (size_t s = 0; s < bs; s++) dst[s] = src[s];
            for (size_t k = k0 + 1; k < k1; k++) {
                const uint8_t* m = mbuf + (size_t)det_indices[k] * bs;
                for (size_t s = 0; s < bs; s++) dst[s] ^= m[s];
            }
            if (det_ref[d]) for (size_t s = 0; s < bs; s++) dst[s] ^= 1u;
        }
        free(mbuf);
    }

    free(m_ref); free(m_kind); free(det_ref);
    return err ? -1 : (long)num_detectors;
}
