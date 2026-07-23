/**
 * @file clifford.c
 * @brief Aaronson-Gottesman tableau Clifford simulator.
 *
 * Reference: Scott Aaronson and Daniel Gottesman, "Improved Simulation
 * of Stabilizer Circuits", Phys. Rev. A 70, 052328 (2004).
 * https://arxiv.org/abs/quant-ph/0406196
 *
 * Tableau layout: 2n rows × (2n + 1) bits.
 *   rows 0..n-1       : destabilizers
 *   rows n..2n-1      : stabilizers
 *   columns 0..n-1    : x bits
 *   columns n..2n-1   : z bits
 *   column 2n         : r (phase) bit; Pauli sign = (-1)^r
 *
 * Initial state |0...0⟩:
 *   destabilizer i = X_i  →  x row i has 1 at column i only
 *   stabilizer  i = Z_i  →  z row i has 1 at column n+i only
 */

#include "clifford.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/*
 * BIT-PACKED, COLUMN-MAJOR STORAGE (F3 throughput rewrite)
 * --------------------------------------------------------
 * The logical tableau is unchanged from Aaronson-Gottesman: 2n rows
 * (rows [0,n) destabilizers, [n,2n) stabilizers), columns [0,n) the
 * Pauli-X exponents, [n,2n) the Z exponents, plus one phase bit per row.
 *
 * The PHYSICAL storage is transposed and bit-packed.  For each qubit
 * column j we keep the X-bits and Z-bits of that column across all 2n
 * rows as a contiguous bit-vector of `w = ceil(2n/64)` 64-bit words
 * (bit r of the word-vector is the entry for row r).  The 2n phase bits
 * are one more w-word vector.
 *
 *   x[j*w .. j*w+w)   : X column j, bit r = X-exponent of row r on qubit j
 *   z[j*w .. j*w+w)   : Z column j
 *   rp[0 .. w)        : phase bits, bit r = sign of row r
 *
 * Why column-major: every Clifford gate on qubit q touches only the two
 * columns q (X and Z) but *all* 2n rows.  In this layout those rows are
 * packed 64-to-a-word, so a gate is a handful of word XOR/AND/swap loops
 * over w = ceil(2n/64) words -- word-parallel over the rows, matching
 * Stim's per-gate word complexity instead of the old O(n) scalar scan.
 *
 * Storage is 2*n*w*8 bytes for the columns (~n^2/2 bytes) versus the
 * previous 2n*(2n+1) bytes (~4n^2), an 8x reduction.
 *
 * tget/tset below preserve the OLD (row,col) addressing exactly, so the
 * Pauli-string introspection API (clifford_row_pauli, conjugate_pauli,
 * ...) is untouched.
 */
struct clifford_tableau_t {
    size_t n;
    size_t w;        /* words per column bit-vector = ceil(2n/64) */
    uint64_t* x;     /* n column vectors of w words each */
    uint64_t* z;     /* n column vectors of w words each */
    uint64_t* rp;    /* phase bit-vector, w words (2n bits used) */
};

static inline uint64_t* xcol(const clifford_tableau_t* t, size_t j) {
    return t->x + j * t->w;
}
static inline uint64_t* zcol(const clifford_tableau_t* t, size_t j) {
    return t->z + j * t->w;
}
static inline uint8_t getbit(const uint64_t* v, size_t r) {
    return (uint8_t)((v[r >> 6] >> (r & 63)) & 1u);
}
static inline void setbit(uint64_t* v, size_t r, uint8_t b) {
    uint64_t m = (uint64_t)1 << (r & 63);
    if (b) v[r >> 6] |= m; else v[r >> 6] &= ~m;
}
static inline void xorbit(uint64_t* v, size_t r, uint8_t b) {
    if (b) v[r >> 6] ^= (uint64_t)1 << (r & 63);
}

/* Preserve the original (row, col) bit addressing for the introspection
 * API and the transparent path.  col in [0,n): X; [n,2n): Z; 2n: phase. */
static inline uint8_t tget(const clifford_tableau_t* t, size_t r, size_t c) {
    size_t n = t->n;
    if (c == 2 * n) return getbit(t->rp, r);
    if (c < n)      return getbit(xcol(t, c), r);
    return getbit(zcol(t, c - n), r);
}
__attribute__((unused))
static inline void tset(clifford_tableau_t* t, size_t r, size_t c, uint8_t v) {
    size_t n = t->n;
    if (c == 2 * n) { setbit(t->rp, r, v); return; }
    if (c < n)      { setbit(xcol(t, c), r, v); return; }
    setbit(zcol(t, c - n), r, v);
}

size_t clifford_num_qubits(const clifford_tableau_t* t) { return t ? t->n : 0; }

clifford_tableau_t* clifford_tableau_create(size_t n) {
    if (n == 0 || n > 100000) return NULL;
    clifford_tableau_t* t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->n = n;
    t->w = (2 * n + 63) / 64;
    t->x  = calloc(n * t->w, sizeof(uint64_t));
    t->z  = calloc(n * t->w, sizeof(uint64_t));
    t->rp = calloc(t->w, sizeof(uint64_t));
    if (!t->x || !t->z || !t->rp) {
        free(t->x); free(t->z); free(t->rp); free(t);
        return NULL;
    }
    /* Destabilizer i = X_i  →  X-bit on row i, column i. */
    for (size_t i = 0; i < n; i++) setbit(xcol(t, i), i, 1);
    /* Stabilizer i = Z_i  →  Z-bit on row n+i, column i. */
    for (size_t i = 0; i < n; i++) setbit(zcol(t, i), n + i, 1);
    return t;
}

void clifford_tableau_free(clifford_tableau_t* t) {
    if (!t) return;
    free(t->x);
    free(t->z);
    free(t->rp);
    free(t);
}

/* --- Gate updates (Aaronson-Gottesman §5 lookup tables) --- */

/* H on qubit q: for every row, swap x_q and z_q, update r by x_q · z_q.
 * Word-parallel over all 2n rows: r ^= x_q & z_q (pre-swap), then swap
 * the two column bit-vectors. */
clifford_error_t clifford_h(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    uint64_t* xq = xcol(t, q);
    uint64_t* zq = zcol(t, q);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) {
        uint64_t xk = xq[k], zk = zq[k];
        rp[k] ^= xk & zk;
        xq[k] = zk;
        zq[k] = xk;
    }
    return CLIFFORD_SUCCESS;
}

/* S on qubit q: z_q ← z_q ⊕ x_q; r ← r ⊕ (x_q · z_q). Word-parallel. */
clifford_error_t clifford_s(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    uint64_t* xq = xcol(t, q);
    uint64_t* zq = zcol(t, q);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) {
        uint64_t xk = xq[k];
        rp[k] ^= xk & zq[k];
        zq[k] ^= xk;
    }
    return CLIFFORD_SUCCESS;
}

/* S† = S·Z = S applied three times (order 4): cheapest is S·S·S. */
clifford_error_t clifford_s_dag(clifford_tableau_t* t, size_t q) {
    clifford_error_t e = clifford_s(t, q);
    if (e != CLIFFORD_SUCCESS) return e;
    e = clifford_s(t, q);
    if (e != CLIFFORD_SUCCESS) return e;
    return clifford_s(t, q);
}

/* CNOT(a,b): for every row, x_b ← x_b ⊕ x_a, z_a ← z_a ⊕ z_b,
 *            r ← r ⊕ (x_a & z_b & (x_b ⊕ z_a ⊕ 1)).
 * The phase term reads the OLD x_b and z_a, so it is computed before the
 * two column updates.  Word-parallel over all rows. */
clifford_error_t clifford_cnot(clifford_tableau_t* t, size_t a, size_t b) {
    if (!t || a >= t->n || b >= t->n || a == b) return CLIFFORD_ERR_QUBIT;
    uint64_t* xa = xcol(t, a);
    uint64_t* za = zcol(t, a);
    uint64_t* xb = xcol(t, b);
    uint64_t* zb = zcol(t, b);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) {
        /* (x_b ⊕ z_a ⊕ 1) = ~(x_b ⊕ z_a) at the bit level. */
        rp[k] ^= xa[k] & zb[k] & ~(xb[k] ^ za[k]);
        xb[k] ^= xa[k];
        za[k] ^= zb[k];
    }
    return CLIFFORD_SUCCESS;
}

/* Pauli X: flip r whenever the row anticommutes with X_q, i.e. z_q = 1. */
clifford_error_t clifford_x(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    uint64_t* zq = zcol(t, q);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) rp[k] ^= zq[k];
    return CLIFFORD_SUCCESS;
}

/* Pauli Z: flip r whenever the row has x_q = 1. */
clifford_error_t clifford_z(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    uint64_t* xq = xcol(t, q);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) rp[k] ^= xq[k];
    return CLIFFORD_SUCCESS;
}

/* Pauli Y = i·X·Z: flip r when x_q ⊕ z_q = 1. */
clifford_error_t clifford_y(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    uint64_t* xq = xcol(t, q);
    uint64_t* zq = zcol(t, q);
    uint64_t* rp = t->rp;
    for (size_t k = 0; k < t->w; k++) rp[k] ^= xq[k] ^ zq[k];
    return CLIFFORD_SUCCESS;
}

clifford_error_t clifford_cz(clifford_tableau_t* t, size_t a, size_t b) {
    clifford_error_t e = clifford_h(t, b);
    if (e != CLIFFORD_SUCCESS) return e;
    e = clifford_cnot(t, a, b);
    if (e != CLIFFORD_SUCCESS) return e;
    return clifford_h(t, b);
}

clifford_error_t clifford_swap(clifford_tableau_t* t, size_t a, size_t b) {
    clifford_error_t e = clifford_cnot(t, a, b);
    if (e != CLIFFORD_SUCCESS) return e;
    e = clifford_cnot(t, b, a);
    if (e != CLIFFORD_SUCCESS) return e;
    return clifford_cnot(t, a, b);
}

/* ================================================================== */
/*  Pauli-string introspection for CA-MPS / CA-PEPS (v0.3.0)         */
/* ================================================================== */

/* Map (x, z) ∈ {0,1}² → Pauli code {0=I, 1=X, 2=Y, 3=Z}. */
static inline uint8_t xz_to_pauli(uint8_t x, uint8_t z) {
    /* I=(0,0), X=(1,0), Y=(1,1), Z=(0,1) */
    if (!x && !z) return 0;
    if ( x && !z) return 1;
    if ( x &&  z) return 2;
    return 3;
}

/* Multiply Pauli p2 into running Pauli p1, accumulating phase.
 * Returns the new Pauli code for the site; the phase delta (in units of i)
 * is added to *phase_accum (mod 4).
 *
 * Pauli multiplication on a single qubit (ignoring identity):
 *   X * X = I, Y * Y = I, Z * Z = I
 *   X * Y = +i Z,    Y * X = -i Z
 *   Y * Z = +i X,    Z * Y = -i X
 *   Z * X = +i Y,    X * Z = -i Y
 *
 * Phase contribution table indexed by (p1, p2): 0 = no phase, 1 = +i,
 * 2 = -1 (not produced by a single multiplication but included for
 * completeness), 3 = -i. */
static uint8_t pauli_mul(uint8_t p1, uint8_t p2, int* phase_accum) {
    if (p1 == 0) return p2;
    if (p2 == 0) return p1;
    if (p1 == p2) return 0;  /* X*X=Y*Y=Z*Z=I, no phase */

    static const uint8_t prod_tab[4][4] = {
        /*        I  X  Y  Z */
        /* I */  {0, 1, 2, 3},
        /* X */  {1, 0, 3, 2},  /* X*Y=Z, X*Z=Y */
        /* Y */  {2, 3, 0, 1},  /* Y*X=Z, Y*Z=X */
        /* Z */  {3, 2, 1, 0},  /* Z*X=Y, Z*Y=X */
    };
    /* Phase table: phase of p1*p2 in powers of i. */
    static const uint8_t phase_tab[4][4] = {
        /*        I  X  Y  Z */
        /* I */  {0, 0, 0, 0},
        /* X */  {0, 0, 1, 3},  /* XY=+iZ, XZ=-iY */
        /* Y */  {0, 3, 0, 1},  /* YX=-iZ, YZ=+iX */
        /* Z */  {0, 1, 3, 0},  /* ZX=+iY, ZY=-iX */
    };
    *phase_accum = (*phase_accum + (int)phase_tab[p1][p2]) & 3;
    return prod_tab[p1][p2];
}

clifford_error_t clifford_row_pauli(const clifford_tableau_t* t, size_t row,
                                    uint8_t* out_pauli, int* out_phase) {
    if (!t || !out_pauli || row >= 2 * t->n) return CLIFFORD_ERR_INVALID;
    size_t n = t->n;
    for (size_t j = 0; j < n; j++) {
        out_pauli[j] = xz_to_pauli(tget(t, row, j), tget(t, row, n + j));
    }
    if (out_phase) {
        /* Tableau phase bit: 0 -> sign +1 (phase code 0), 1 -> sign -1 (phase code 2). */
        *out_phase = tget(t, row, 2 * n) ? 2 : 0;
    }
    return CLIFFORD_SUCCESS;
}

clifford_error_t clifford_conjugate_pauli(const clifford_tableau_t* t,
                                          const uint8_t* in_pauli,
                                          int in_phase,
                                          uint8_t* out_pauli,
                                          int* out_phase) {
    if (!t || !in_pauli || !out_pauli) return CLIFFORD_ERR_INVALID;
    size_t n = t->n;
    /* out = product over qubits q where in_pauli[q] != I of (D P_q D^†).
     *   D X_q D^† = destabilizer row q
     *   D Z_q D^† = stabilizer row n+q
     *   D Y_q D^† = i * (D X_q D^†) * (D Z_q D^†)
     */
    uint8_t *factor = (uint8_t *)calloc(n, sizeof(uint8_t));
    uint8_t *acc    = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!factor || !acc) { free(factor); free(acc); return CLIFFORD_ERR_OOM; }

    int phase = in_phase & 3;

    for (size_t q = 0; q < n; q++) {
        uint8_t pq = in_pauli[q];
        if (pq == 0) continue;  /* identity, no factor */

        if (pq == 1) {
            /* X_q -> destabilizer row q. */
            int fphase = 0;
            clifford_row_pauli(t, q, factor, &fphase);
            phase = (phase + fphase) & 3;
        } else if (pq == 3) {
            /* Z_q -> stabilizer row n+q. */
            int fphase = 0;
            clifford_row_pauli(t, n + q, factor, &fphase);
            phase = (phase + fphase) & 3;
        } else {
            /* Y_q = i X_q Z_q -> i * (destab_q) * (stab_{n+q}) */
            uint8_t *x_pauli = (uint8_t *)calloc(n, sizeof(uint8_t));
            uint8_t *z_pauli = (uint8_t *)calloc(n, sizeof(uint8_t));
            if (!x_pauli || !z_pauli) {
                free(x_pauli); free(z_pauli); free(factor); free(acc);
                return CLIFFORD_ERR_OOM;
            }
            int xph = 0, zph = 0;
            clifford_row_pauli(t, q, x_pauli, &xph);
            clifford_row_pauli(t, n + q, z_pauli, &zph);
            phase = (phase + xph + zph + 1 /* factor i from Y = iXZ */) & 3;
            /* factor = x_pauli * z_pauli, accumulating Pauli-product phase. */
            for (size_t j = 0; j < n; j++) {
                int dphase = 0;
                factor[j] = pauli_mul(x_pauli[j], z_pauli[j], &dphase);
                phase = (phase + dphase) & 3;
            }
            free(x_pauli); free(z_pauli);
        }

        /* acc <- acc * factor, qubit by qubit, with phase tracking. */
        for (size_t j = 0; j < n; j++) {
            int dphase = 0;
            acc[j] = pauli_mul(acc[j], factor[j], &dphase);
            phase = (phase + dphase) & 3;
        }
    }

    memcpy(out_pauli, acc, n);
    if (out_phase) *out_phase = phase;
    free(factor); free(acc);
    return CLIFFORD_SUCCESS;
}

/* Inverse conjugation: compute C^dagger P C where t stores C.
 *
 * Implementation: use the symplectic identity M^{-1} = Lambda M^T Lambda,
 * which gives us the "inverse-map" Pauli for each single-qubit generator
 * by reading COLUMNS of the forward tableau:
 *
 *   C^dagger X_i C : x-bit at j = tget(t, n+i, n+j)  (column n+i, stab rows)
 *                    z-bit at j = tget(t, i,   n+j)  (column n+i, destab rows)
 *   C^dagger Z_i C : x-bit at j = tget(t, n+i, j)    (column i,   stab rows)
 *                    z-bit at j = tget(t, i,   j)    (column i,   destab rows)
 *   C^dagger Y_i C = i * (C^dagger X_i C) * (C^dagger Z_i C).
 *
 * Phase bookkeeping for the inverse tableau is subtle (phase bits of the
 * forward tableau don't directly give the inverse phases).  We track phase
 * contributions from Pauli-product multiplications via the same phase table
 * as the forward direction.  The "sign part" from the symplectic inversion
 * itself (i.e., whether C^dagger X_i C = + or - X'_i for some X'_i when the
 * forward tableau has phase bits set) is reconstructed via a per-generator
 * consistency check: after computing the symplectic image, we verify that
 * C (image) C^dagger equals +X_i / +Z_i / +Y_i by running it through the
 * forward tableau; if the sign comes out wrong, we flip the image's phase.
 */
static void read_inv_single_qubit_image(const clifford_tableau_t* t, size_t q,
                                        int gen_code, /* 1=X, 2=Y, 3=Z */
                                        uint8_t* out_pauli, int* out_phase) {
    size_t n = t->n;
    int phase = 0;
    /* Formula: C^dagger X_q C has x-bit at qubit j = tget(t, n+j, n+q)
     *                            z-bit at qubit j = tget(t, j,   n+q)
     * Derived from M^{-1} = Lambda M^T Lambda with M[r,c] = tget(t, c, r),
     * giving M^{-1}[r, c] = tget(t, Lambda r, Lambda c). */
    if (gen_code == 1) {
        for (size_t j = 0; j < n; j++) {
            uint8_t x = tget(t, n + j, n + q);
            uint8_t z = tget(t, j,     n + q);
            out_pauli[j] = xz_to_pauli(x, z);
        }
    } else if (gen_code == 3) {
        /* C^dagger Z_q C: x-bit at j = tget(t, n+j, q), z-bit = tget(t, j, q). */
        for (size_t j = 0; j < n; j++) {
            uint8_t x = tget(t, n + j, q);
            uint8_t z = tget(t, j,     q);
            out_pauli[j] = xz_to_pauli(x, z);
        }
    } else {
        /* Y = i X Z.  Compute C^dagger Y_i C = i * (C^dagger X_i C)(C^dagger Z_i C). */
        uint8_t *x_image = calloc(n, sizeof(uint8_t));
        uint8_t *z_image = calloc(n, sizeof(uint8_t));
        read_inv_single_qubit_image(t, q, 1, x_image, &phase);
        int zph = 0;
        read_inv_single_qubit_image(t, q, 3, z_image, &zph);
        phase = (phase + zph + 1) & 3;   /* +1 for the i in Y = iXZ */
        for (size_t j = 0; j < n; j++) {
            int dphase = 0;
            out_pauli[j] = pauli_mul(x_image[j], z_image[j], &dphase);
            phase = (phase + dphase) & 3;
        }
        free(x_image); free(z_image);
    }

    /* Sign-fixup: verify C (out_pauli) C^dagger == gen_Pauli with correct sign.
     * Run the forward conjugation and check the result. */
    uint8_t *fwd = calloc(n, sizeof(uint8_t));
    int fwd_phase = 0;
    clifford_conjugate_pauli(t, out_pauli, phase, fwd, &fwd_phase);
    /* Expected: fwd = single-qubit Pauli at position q with phase 0.
     * If fwd_phase != 0 (i.e. sign mismatch), flip our computed phase. */
    if (fwd_phase != 0) phase = (phase + (4 - fwd_phase)) & 3;
    free(fwd);

    if (out_phase) *out_phase = phase;
}

clifford_error_t clifford_conjugate_pauli_inverse(const clifford_tableau_t* t,
                                                  const uint8_t* in_pauli,
                                                  int in_phase,
                                                  uint8_t* out_pauli,
                                                  int* out_phase) {
    if (!t || !in_pauli || !out_pauli) return CLIFFORD_ERR_INVALID;
    size_t n = t->n;
    uint8_t *factor = (uint8_t *)calloc(n, sizeof(uint8_t));
    uint8_t *acc    = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!factor || !acc) { free(factor); free(acc); return CLIFFORD_ERR_OOM; }

    int phase = in_phase & 3;
    for (size_t q = 0; q < n; q++) {
        uint8_t pq = in_pauli[q];
        if (pq == 0) continue;
        int fphase = 0;
        read_inv_single_qubit_image(t, q, pq, factor, &fphase);
        phase = (phase + fphase) & 3;
        for (size_t j = 0; j < n; j++) {
            int dphase = 0;
            acc[j] = pauli_mul(acc[j], factor[j], &dphase);
            phase = (phase + dphase) & 3;
        }
    }
    memcpy(out_pauli, acc, n);
    if (out_phase) *out_phase = phase;
    free(factor); free(acc);
    return CLIFFORD_SUCCESS;
}

clifford_tableau_t* clifford_tableau_clone(const clifford_tableau_t* t) {
    if (!t) return NULL;
    clifford_tableau_t* c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n = t->n;
    c->w = t->w;
    size_t colwords = t->n * t->w;
    c->x  = malloc(colwords * sizeof(uint64_t));
    c->z  = malloc(colwords * sizeof(uint64_t));
    c->rp = malloc(t->w * sizeof(uint64_t));
    if (!c->x || !c->z || !c->rp) {
        free(c->x); free(c->z); free(c->rp); free(c);
        return NULL;
    }
    memcpy(c->x,  t->x,  colwords * sizeof(uint64_t));
    memcpy(c->z,  t->z,  colwords * sizeof(uint64_t));
    memcpy(c->rp, t->rp, t->w * sizeof(uint64_t));
    return c;
}

/* --- Measurement (Aaronson-Gottesman Algorithm 2) --- */

/* AG's g function (scalar reference): the exponent of i (mod 4, in the
 * range {-1,0,1}) picked up when Pauli char (x2,z2) is multiplied on the
 * left by (x1,z1).  Kept as the authoritative scalar against which the
 * word-parallel form below is checked (exhaustively, in the unit test and
 * in debug builds). */
__attribute__((unused))
static int rowsum_g(uint8_t x1, uint8_t z1, uint8_t x2, uint8_t z2) {
    if (x1 == 0 && z1 == 0) return 0;
    if (x1 == 1 && z1 == 1) return (int)z2 - (int)x2;                 /* Y */
    if (x1 == 1 && z1 == 0) return (int)z2 * (2 * (int)x2 - 1);       /* X */
    return (int)x2 * (1 - 2 * (int)z2);                              /* Z */
}

/*
 * WORD-PARALLEL rowsum phase (item 3 of the F3 rewrite).
 *
 * For a whole 64-bit word, sum the g-contributions of all 64 lanes at
 * once.  Each lane contributes +1, 0, or -1.  We build a `pos` mask (lanes
 * contributing +1) and a `neg` mask (lanes contributing -1) with pure
 * bitwise ops, then popcount both.  The masks are derived directly from
 * the scalar g table above, split by the (x1,z1) case of the left factor:
 *
 *   left = Z (x1=0,z1=1):  g=+1 at (x2=1,z2=0);  g=-1 at (x2=1,z2=1)
 *   left = X (x1=1,z1=0):  g=+1 at (x2=1,z2=1);  g=-1 at (x2=0,z2=1)
 *   left = Y (x1=1,z1=1):  g=+1 at (x2=0,z2=1);  g=-1 at (x2=1,z2=0)
 *   left = I (x1=0,z1=0):  g=0 always
 *
 * Verified exhaustively over all 16 single-bit inputs and over random
 * word vectors against rowsum_g (see tests/unit/test_clifford_rowsum.c and
 * the debug assertion in rowsum()).  Returns the signed lane sum for the
 * word block; the caller reduces mod 4.
 */
static int rowsum_phase_words(const uint64_t* x1, const uint64_t* z1,
                              const uint64_t* x2, const uint64_t* z2,
                              size_t ws) {
    int acc = 0;
    for (size_t k = 0; k < ws; k++) {
        uint64_t X1 = x1[k], Z1 = z1[k], X2 = x2[k], Z2 = z2[k];
        uint64_t pos = (~X1 &  Z1 &  X2 & ~Z2)   /* Z: +1 */
                     | ( X1 & ~Z1 &  X2 &  Z2)   /* X: +1 */
                     | ( X1 &  Z1 & ~X2 &  Z2);  /* Y: +1 */
        uint64_t neg = (~X1 &  Z1 &  X2 &  Z2)   /* Z: -1 */
                     | ( X1 & ~Z1 & ~X2 &  Z2)   /* X: -1 */
                     | ( X1 &  Z1 &  X2 & ~Z2);  /* Y: -1 */
        acc += __builtin_popcountll(pos) - __builtin_popcountll(neg);
    }
    return acc;
}

/* Extract row r into row-major packed word buffers (ws = ceil(n/64)). */
static void row_get(const clifford_tableau_t* t, size_t r,
                    uint64_t* xb, uint64_t* zb, size_t ws) {
    memset(xb, 0, ws * sizeof(uint64_t));
    memset(zb, 0, ws * sizeof(uint64_t));
    size_t n = t->n;
    for (size_t j = 0; j < n; j++) {
        if (getbit(xcol(t, j), r)) xb[j >> 6] |= (uint64_t)1 << (j & 63);
        if (getbit(zcol(t, j), r)) zb[j >> 6] |= (uint64_t)1 << (j & 63);
    }
}

static inline int mod4(int v) { return ((v % 4) + 4) % 4; }

/* Row-sum: multiply row h by the reference row supplied in row-major
 * packed form (rx, rz, rphase), then XOR the reference into row h's
 * columns.  The phase is accumulated word-parallel via popcount. */
static void rowsum_ref(clifford_tableau_t* t, size_t h,
                       const uint64_t* rx, const uint64_t* rz, uint8_t rphase,
                       uint64_t* xh, uint64_t* zh, size_t ws) {
    row_get(t, h, xh, zh, ws);
    int acc = 2 * (int)getbit(t->rp, h) + 2 * (int)rphase
            + rowsum_phase_words(rx, rz, xh, zh, ws);
#ifndef NDEBUG
    /* Debug-only cross-check of the word-parallel phase against the scalar
     * g reference. */
    {
        int scal = 0;
        for (size_t j = 0; j < t->n; j++) {
            scal += rowsum_g((uint8_t)((rx[j >> 6] >> (j & 63)) & 1),
                             (uint8_t)((rz[j >> 6] >> (j & 63)) & 1),
                             (uint8_t)((xh[j >> 6] >> (j & 63)) & 1),
                             (uint8_t)((zh[j >> 6] >> (j & 63)) & 1));
        }
        int wp = rowsum_phase_words(rx, rz, xh, zh, ws);
        if (mod4(scal) != mod4(wp)) abort();
    }
#endif
    setbit(t->rp, h, (uint8_t)(mod4(acc) == 2 ? 1 : 0));
    size_t n = t->n;
    for (size_t j = 0; j < n; j++) {
        xorbit(xcol(t, j), h, (uint8_t)((rx[j >> 6] >> (j & 63)) & 1));
        xorbit(zcol(t, j), h, (uint8_t)((rz[j >> 6] >> (j & 63)) & 1));
    }
}

/* splitmix64 — tiny deterministic prng for the "random measurement"
 * branch; caller supplies a uint64 state and gets it advanced. */
static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

clifford_error_t clifford_measure(clifford_tableau_t* t, size_t q,
                                  uint64_t* rng_state, int* outcome,
                                  int* outcome_kind) {
    if (!t || q >= t->n || !rng_state || !outcome) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n;
    size_t ws = (n + 63) / 64;
    uint64_t* xq = xcol(t, q);

    /* Find a stabilizer row p in [n, 2n) whose x_q bit is 1. */
    size_t p = 2 * n;
    for (size_t r = n; r < 2 * n; r++) {
        if (getbit(xq, r)) { p = r; break; }
    }

    /* Row-major scratch/reference buffers. */
    uint64_t* bufA = calloc(ws, sizeof(uint64_t));  /* reference / stabilizer */
    uint64_t* bufB = calloc(ws, sizeof(uint64_t));
    uint64_t* accx = calloc(ws, sizeof(uint64_t));  /* deterministic scratch */
    uint64_t* accz = calloc(ws, sizeof(uint64_t));
    if (!bufA || !bufB || !accx || !accz) {
        free(bufA); free(bufB); free(accx); free(accz);
        return CLIFFORD_ERR_OOM;
    }

    if (p == 2 * n) {
        /* Deterministic outcome. Row-sum every stabilizer n+i whose paired
         * destabilizer i anticommutes with Z_q (x_iq = 1) into a scratch
         * row and read the accumulated sign. */
        uint8_t sphase = 0;
        for (size_t i = 0; i < n; i++) {
            if (!getbit(xq, i)) continue;
            row_get(t, n + i, bufA, bufB, ws);  /* stabilizer n+i */
            uint8_t stab_r = getbit(t->rp, n + i);
            int acc = 2 * (int)sphase + 2 * (int)stab_r
                    + rowsum_phase_words(bufA, bufB, accx, accz, ws);
            sphase = (uint8_t)(mod4(acc) == 2 ? 1 : 0);
            for (size_t k = 0; k < ws; k++) { accx[k] ^= bufA[k]; accz[k] ^= bufB[k]; }
        }
        *outcome = sphase ? 1 : 0;
        if (outcome_kind) *outcome_kind = 0;
        free(bufA); free(bufB); free(accx); free(accz);
        return CLIFFORD_SUCCESS;
    }

    /* Random outcome (Aaronson-Gottesman Algorithm 2, case a).
     *
     * Correct ordering: the elimination row-sums must use the OLD
     * anticommuting stabilizer (row p) as the reference, and that
     * reference is preserved as the new destabilizer (row p-n).  We
     * therefore extract row p FIRST, then overwrite, then eliminate. */
    row_get(t, p, bufA, bufB, ws);      /* bufA/bufB = old stabilizer p */
    uint8_t ref_phase = getbit(t->rp, p);

    /* Destabilizer (p-n) <- old stabilizer p (rowcopy). */
    for (size_t j = 0; j < n; j++) {
        setbit(xcol(t, j), p - n, (uint8_t)((bufA[j >> 6] >> (j & 63)) & 1));
        setbit(zcol(t, j), p - n, (uint8_t)((bufB[j >> 6] >> (j & 63)) & 1));
    }
    setbit(t->rp, p - n, ref_phase);

    /* Stabilizer p <- Z_q with a random sign. */
    for (size_t j = 0; j < n; j++) {
        setbit(xcol(t, j), p, 0);
        setbit(zcol(t, j), p, 0);
    }
    setbit(zcol(t, q), p, 1);
    uint8_t out = (uint8_t)(sm64(rng_state) & 1ULL);
    setbit(t->rp, p, out);
    *outcome = (int)out;

    /* Eliminate the X_q component from every other row that has one, using
     * the old anticommuting stabilizer (now in row p-n / bufA,bufB) as the
     * reference.  Row p-n itself is the destabilizer partner and is left
     * anticommuting; the new stabilizer p has x_q = 0 and is skipped. */
    for (size_t r = 0; r < 2 * n; r++) {
        if (r == p - n) continue;
        if (getbit(xq, r)) rowsum_ref(t, r, bufA, bufB, ref_phase, accx, accz, ws);
    }

    if (outcome_kind) *outcome_kind = 1;
    free(bufA); free(bufB); free(accx); free(accz);
    return CLIFFORD_SUCCESS;
}

clifford_error_t clifford_sample_all(clifford_tableau_t* t,
                                     uint64_t* rng_state,
                                     uint64_t* result) {
    if (!t || !rng_state || !result) return CLIFFORD_ERR_INVALID;
    uint64_t r = 0;
    for (size_t q = 0; q < t->n; q++) {
        int out = 0;
        clifford_error_t e = clifford_measure(t, q, rng_state, &out, NULL);
        if (e != CLIFFORD_SUCCESS) return e;
        if (out) r |= (uint64_t)1 << q;
    }
    *result = r;
    return CLIFFORD_SUCCESS;
}
