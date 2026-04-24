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

struct clifford_tableau_t {
    size_t n;
    /* Flat layout: rows * (2n + 1) uint8_t. Row r starts at
     * row(r) = r * (2n + 1); columns [0..n-1] are x, [n..2n-1] are z,
     * column 2n is the phase bit. One byte per bit keeps the code
     * transparent; a bit-packed variant can follow. */
    uint8_t* tab;
};

static inline uint8_t tget(const clifford_tableau_t* t, size_t r, size_t c) {
    return t->tab[r * (2 * t->n + 1) + c];
}
static inline void tset(clifford_tableau_t* t, size_t r, size_t c, uint8_t v) {
    t->tab[r * (2 * t->n + 1) + c] = v;
}

size_t clifford_num_qubits(const clifford_tableau_t* t) { return t ? t->n : 0; }

clifford_tableau_t* clifford_tableau_create(size_t n) {
    if (n == 0 || n > 100000) return NULL;
    clifford_tableau_t* t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->n = n;
    t->tab = calloc((size_t)2 * n * (2 * n + 1), sizeof(uint8_t));
    if (!t->tab) { free(t); return NULL; }
    /* Destabilizer i = X_i  →  x_i bit on row i, column i. */
    for (size_t i = 0; i < n; i++) tset(t, i, i, 1);
    /* Stabilizer i = Z_i  →  z_i bit on row n+i, column n+i. */
    for (size_t i = 0; i < n; i++) tset(t, n + i, n + i, 1);
    return t;
}

void clifford_tableau_free(clifford_tableau_t* t) {
    if (!t) return;
    free(t->tab);
    free(t);
}

/* --- Gate updates (Aaronson-Gottesman §5 lookup tables) --- */

/* H on qubit q: for every row, swap x_q and z_q, update r by x_q * z_q. */
clifford_error_t clifford_h(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++) {
        uint8_t x = tget(t, r, q);
        uint8_t z = tget(t, r, n + q);
        tset(t, r, rbit, tget(t, r, rbit) ^ (x & z));
        tset(t, r, q, z);
        tset(t, r, n + q, x);
    }
    return CLIFFORD_SUCCESS;
}

/* S on qubit q: z_q ← z_q ⊕ x_q; r ← r ⊕ (x_q · z_q). */
clifford_error_t clifford_s(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++) {
        uint8_t x = tget(t, r, q);
        uint8_t z = tget(t, r, n + q);
        tset(t, r, rbit, tget(t, r, rbit) ^ (x & z));
        tset(t, r, n + q, z ^ x);
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
 *            r ← r ⊕ (x_a & z_b & (x_b ⊕ z_a ⊕ 1)). */
clifford_error_t clifford_cnot(clifford_tableau_t* t, size_t a, size_t b) {
    if (!t || a >= t->n || b >= t->n || a == b) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++) {
        uint8_t xa = tget(t, r, a);
        uint8_t xb = tget(t, r, b);
        uint8_t za = tget(t, r, n + a);
        uint8_t zb = tget(t, r, n + b);
        uint8_t r_old = tget(t, r, rbit);
        tset(t, r, rbit, r_old ^ (xa & zb & ((xb ^ za ^ 1) & 1)));
        tset(t, r, b, xb ^ xa);
        tset(t, r, n + a, za ^ zb);
    }
    return CLIFFORD_SUCCESS;
}

/* Pauli X = H·S²·H. Cheaper: just update r whenever the stabilizer
 * row anticommutes with X_q, i.e., has z_q = 1. */
clifford_error_t clifford_x(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++)
        tset(t, r, rbit, tget(t, r, rbit) ^ tget(t, r, n + q));
    return CLIFFORD_SUCCESS;
}

/* Pauli Z: flip r whenever the row has x_q = 1. */
clifford_error_t clifford_z(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++)
        tset(t, r, rbit, tget(t, r, rbit) ^ tget(t, r, q));
    return CLIFFORD_SUCCESS;
}

/* Pauli Y = i·X·Z: anticommutes with any Pauli containing X or Z at q,
 * but not with the identity on q. Flip r when x_q ⊕ z_q = 1. */
clifford_error_t clifford_y(clifford_tableau_t* t, size_t q) {
    if (!t || q >= t->n) return CLIFFORD_ERR_QUBIT;
    size_t n = t->n, rbit = 2 * n;
    for (size_t r = 0; r < 2 * n; r++)
        tset(t, r, rbit, tget(t, r, rbit) ^ tget(t, r, q) ^ tget(t, r, n + q));
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

clifford_tableau_t* clifford_tableau_clone(const clifford_tableau_t* t) {
    if (!t) return NULL;
    clifford_tableau_t* c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n = t->n;
    size_t nbytes = 2 * t->n * (2 * t->n + 1);
    c->tab = malloc(nbytes);
    if (!c->tab) { free(c); return NULL; }
    memcpy(c->tab, t->tab, nbytes);
    return c;
}

/* --- Measurement (Aaronson-Gottesman Algorithm 2) --- */

/* Row-sum helper for the measurement case: multiply row h by row i,
 * picking up the right phase from the commutation cocycle. */
static uint8_t rowsum_g(uint8_t x1, uint8_t z1, uint8_t x2, uint8_t z2) {
    /* AG's g function: returns the exponent of i in the product of
     * Pauli chars (x1,z1) and (x2,z2). 0,1,-1 for I/X/Y/Z multiplication.
     * Encoded in two bits: phase_change = g mod 4. */
    if (x1 == 0 && z1 == 0) return 0;
    if (x1 == 1 && z1 == 1) return (uint8_t)((z2 - x2) & 3);
    if (x1 == 1 && z1 == 0) return (uint8_t)((z2 * (2 * x2 - 1)) & 3);
    /* x1 == 0, z1 == 1 */
    return (uint8_t)((x2 * (1 - 2 * z2)) & 3);
}

static void rowsum(clifford_tableau_t* t, size_t h, size_t i) {
    size_t n = t->n, rbit = 2 * n;
    int acc = 2 * (int)tget(t, h, rbit) + 2 * (int)tget(t, i, rbit);
    for (size_t j = 0; j < n; j++) {
        acc += (int)rowsum_g(tget(t, i, j), tget(t, i, n + j),
                             tget(t, h, j), tget(t, h, n + j));
    }
    acc &= 3;
    tset(t, h, rbit, (uint8_t)(acc == 2 ? 1 : 0));
    for (size_t j = 0; j < n; j++) {
        tset(t, h, j,     tget(t, h, j)     ^ tget(t, i, j));
        tset(t, h, n + j, tget(t, h, n + j) ^ tget(t, i, n + j));
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
    size_t n = t->n, rbit = 2 * n;

    /* Find a stabilizer row p in [n, 2n) whose x_q bit is 1. */
    size_t p = 2 * n;
    for (size_t r = n; r < 2 * n; r++) {
        if (tget(t, r, q)) { p = r; break; }
    }

    if (p == 2 * n) {
        /* Deterministic outcome. Append an auxiliary row at index 2n
         * by row-summing every destabilizer whose x_q is 1. Read sign. */
        /* Scratch row: we reuse storage by using a temp buffer. */
        uint8_t* scratch = calloc(2 * n + 1, sizeof(uint8_t));
        if (!scratch) return CLIFFORD_ERR_OOM;
        for (size_t i = 0; i < n; i++) {
            if (tget(t, i, q)) {
                /* Add stabilizer row n+i into scratch. */
                /* Compute phase using rowsum_g like rowsum() does. */
                int acc = 2 * (int)scratch[rbit] + 2 * (int)tget(t, n + i, rbit);
                for (size_t j = 0; j < n; j++) {
                    acc += (int)rowsum_g(tget(t, n + i, j),
                                         tget(t, n + i, n + j),
                                         scratch[j], scratch[n + j]);
                }
                acc &= 3;
                scratch[rbit] = (uint8_t)(acc == 2 ? 1 : 0);
                for (size_t j = 0; j < n; j++) {
                    scratch[j] ^= tget(t, n + i, j);
                    scratch[n + j] ^= tget(t, n + i, n + j);
                }
            }
        }
        *outcome = scratch[rbit] ? 1 : 0;
        free(scratch);
        if (outcome_kind) *outcome_kind = 0;
        return CLIFFORD_SUCCESS;
    }

    /* Random outcome. Replace destabilizer p-n with stabilizer p, then
     * set stabilizer p to Z_q with a random sign. */
    for (size_t j = 0; j <= 2 * n; j++) {
        tset(t, p - n, j, tget(t, p, j));
        tset(t, p, j, 0);
    }
    tset(t, p, n + q, 1);
    uint8_t out = (uint8_t)(sm64(rng_state) & 1ULL);
    tset(t, p, rbit, out);
    *outcome = (int)out;

    /* Update all other rows whose x_q is 1 by row-summing the new
     * stabilizer p into them. */
    for (size_t r = 0; r < 2 * n; r++) {
        if (r == p) continue;
        if (tget(t, r, q)) rowsum(t, r, p);
    }
    if (outcome_kind) *outcome_kind = 1;
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
