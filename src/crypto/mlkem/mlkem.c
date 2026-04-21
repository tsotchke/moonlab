/**
 * @file mlkem.c
 * @brief FIPS 203 ML-KEM-512 KeyGen / Encaps / Decaps.
 *
 * Built on src/crypto/mlkem/poly.c (ring + NTT + CBD + gen_matrix)
 * and src/crypto/sha3/sha3.c (SHA3, SHAKE).  The structure follows
 * FIPS 203 Section 7.  Variable names mirror the FIPS pseudocode
 * (d, z, rho, sigma, m, K, r, ek, dk) so the implementation is
 * straightforwardly auditable against the standard.
 *
 * Only ML-KEM-512 is wired up right now: k = 2, eta1 = 3, eta2 = 2,
 * du = 10, dv = 4.  The 768 and 1024 parameter sets are a straight
 * retarget of this code -- the arithmetic layer is already parameter-
 * generic.
 */

#include "mlkem.h"
#include "poly.h"
#include "../sha3/sha3.h"
#include "../../applications/moonlab_export.h"

#include <stdint.h>
#include <string.h>

#define K     MLKEM512_K
#define ETA1  MLKEM512_ETA1
#define ETA2  MLKEM512_ETA2
#define DU    MLKEM512_DU
#define DV    MLKEM512_DV

/* -------------------------------------------------------------- */
/* Poly-vector helpers                                             */
/* -------------------------------------------------------------- */

typedef struct { mlkem_poly_t vec[K]; } mlkem_polyvec_t;

static void polyvec_ntt(mlkem_polyvec_t *v) {
    for (int i = 0; i < K; i++) mlkem_poly_ntt(&v->vec[i]);
}
static void polyvec_invntt(mlkem_polyvec_t *v) {
    for (int i = 0; i < K; i++) mlkem_poly_invntt(&v->vec[i]);
}
static void polyvec_add(mlkem_polyvec_t *r, const mlkem_polyvec_t *a,
                         const mlkem_polyvec_t *b) {
    for (int i = 0; i < K; i++) mlkem_poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
}
static void polyvec_reduce(mlkem_polyvec_t *v) {
    for (int i = 0; i < K; i++) mlkem_poly_reduce(&v->vec[i]);
}

/* Inner product of two polyvecs in NTT domain.  The output is in
 * pq-crystals "sub-normal" form (one factor of 1/R relative to the
 * NTT-input form); downstream code either applies @ref mlkem_poly_tomont
 * (when staying in NTT domain) or @ref mlkem_poly_invntt (which
 * multiplies by R internally and restores normal form in the regular
 * time domain). */
static void polyvec_basemul_accum(mlkem_poly_t *r,
                                   const mlkem_polyvec_t *a,
                                   const mlkem_polyvec_t *b) {
    mlkem_poly_t t;
    mlkem_poly_basemul(r, &a->vec[0], &b->vec[0]);
    for (int i = 1; i < K; i++) {
        mlkem_poly_basemul(&t, &a->vec[i], &b->vec[i]);
        mlkem_poly_add(r, r, &t);
    }
    mlkem_poly_reduce(r);
}

/* Byte encode / decode for an entire polyvec. */
static void polyvec_tobytes(uint8_t *out, const mlkem_polyvec_t *v) {
    for (int i = 0; i < K; i++)
        mlkem_poly_tobytes(out + i * MLKEM_POLYBYTES, &v->vec[i]);
}
static void polyvec_frombytes(mlkem_polyvec_t *v, const uint8_t *in) {
    for (int i = 0; i < K; i++)
        mlkem_poly_frombytes(&v->vec[i], in + i * MLKEM_POLYBYTES);
}

/* Compress / decompress a polyvec at coefficient width du. */
static void polyvec_compress(uint8_t *out, const mlkem_polyvec_t *v, int du) {
    for (int i = 0; i < K; i++)
        mlkem_poly_compress(out + i * (MLKEM_N * du / 8), &v->vec[i], du);
}
static void polyvec_decompress(mlkem_polyvec_t *v, const uint8_t *in, int du) {
    for (int i = 0; i < K; i++)
        mlkem_poly_decompress(&v->vec[i], in + i * (MLKEM_N * du / 8), du);
}

/* -------------------------------------------------------------- */
/* CBD noise / message helpers                                     */
/* -------------------------------------------------------------- */

/* PRF(seed, nonce) -> CBD_eta polynomial. */
static void sample_cbd(mlkem_poly_t *p, const uint8_t seed[32],
                        uint8_t nonce, int eta) {
    uint8_t input[33];
    memcpy(input, seed, 32);
    input[32] = nonce;
    uint8_t buf[3 * 256 / 4];   /* up to eta * n / 4 = 192 bytes for eta=3 */
    size_t blen = (size_t)(eta * MLKEM_N / 4);
    shake256(input, 33, buf, blen);
    mlkem_poly_cbd(p, buf, eta);
}

/* Encode / decode a 32-byte message as a polynomial with 0 -> 0 and
 * 1 -> floor(q/2 + 1/2) = 1665.  Each message bit lives on one
 * coefficient. */
static void poly_from_msg(mlkem_poly_t *p, const uint8_t msg[32]) {
    for (int i = 0; i < MLKEM_N / 8; i++) {
        for (int j = 0; j < 8; j++) {
            int bit = (msg[i] >> j) & 1;
            p->coeffs[8 * i + j] = (int16_t)(bit * ((MLKEM_Q + 1) / 2));
        }
    }
}
static void poly_to_msg(uint8_t msg[32], const mlkem_poly_t *p) {
    /* Round each coefficient to the nearest multiple of q/2. */
    memset(msg, 0, 32);
    for (int i = 0; i < MLKEM_N / 8; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t t = p->coeffs[8 * i + j];
            if (t < 0) t += MLKEM_Q;
            /* Bit = 1 if closer to q/2 than to 0, i.e. abs(t - q/2) < q/4. */
            int32_t d = (int32_t)t - (MLKEM_Q / 2);
            if (d < 0) d = -d;
            /* We want the closer of {0, q/2}; bit = 1 iff |t - q/2| < q/4. */
            int bit = (d <= MLKEM_Q / 4) ? 1 : 0;
            msg[i] |= (uint8_t)(bit << j);
        }
    }
}

/* -------------------------------------------------------------- */
/* K-PKE primitives (FIPS 203 Section 5)                           */
/* -------------------------------------------------------------- */

static void kpke_keygen(uint8_t pk[MLKEM512_PUBLICKEYBYTES],
                         uint8_t sk[MLKEM512_POLYVECBYTES],
                         const uint8_t d[32]) {
    /* SHA3-512(d || k_byte) -> rho (32) || sigma (32). */
    uint8_t buf[33];
    memcpy(buf, d, 32);
    buf[32] = (uint8_t)K;
    uint8_t rho_sigma[64];
    sha3_512(buf, 33, rho_sigma);
    const uint8_t *rho   = rho_sigma;
    const uint8_t *sigma = rho_sigma + 32;

    /* A = GenMatrix(rho). */
    mlkem_poly_t A[K * K];
    mlkem_gen_matrix(A, K, rho, 0);

    /* s, e <- CBD_eta1(sigma). */
    mlkem_polyvec_t s, e;
    for (int i = 0; i < K; i++) sample_cbd(&s.vec[i], sigma, (uint8_t)i, ETA1);
    for (int i = 0; i < K; i++) sample_cbd(&e.vec[i], sigma, (uint8_t)(K + i), ETA1);

    polyvec_ntt(&s);
    polyvec_ntt(&e);

    /* t = A @ s + e.  basemul leaves sub-normal form; tomont lifts
     * it back to normal form so e_hat (normal) can be added. */
    mlkem_polyvec_t t;
    for (int i = 0; i < K; i++) {
        mlkem_polyvec_t row;
        for (int j = 0; j < K; j++) row.vec[j] = A[i * K + j];
        polyvec_basemul_accum(&t.vec[i], &row, &s);
        mlkem_poly_tomont(&t.vec[i]);
    }
    polyvec_add(&t, &t, &e);
    polyvec_reduce(&t);

    /* Public key: encode(t) || rho.  Secret key: encode(s_hat). */
    polyvec_tobytes(pk, &t);
    memcpy(pk + MLKEM512_POLYVECBYTES, rho, 32);
    polyvec_tobytes(sk, &s);
}

static void kpke_encrypt(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                          const uint8_t pk[MLKEM512_PUBLICKEYBYTES],
                          const uint8_t msg[32],
                          const uint8_t coins[32]) {
    /* Decode pk. */
    mlkem_polyvec_t t;
    polyvec_frombytes(&t, pk);
    const uint8_t *rho = pk + MLKEM512_POLYVECBYTES;

    /* A^T = GenMatrix(rho, transposed = 1). */
    mlkem_poly_t A[K * K];
    mlkem_gen_matrix(A, K, rho, 1);

    /* y, e1 <- CBD_eta1(coins); e2 <- CBD_eta2(coins). */
    mlkem_polyvec_t y, e1;
    mlkem_poly_t e2;
    for (int i = 0; i < K; i++) sample_cbd(&y.vec[i],  coins, (uint8_t)i,           ETA1);
    for (int i = 0; i < K; i++) sample_cbd(&e1.vec[i], coins, (uint8_t)(K + i),     ETA2);
    sample_cbd(&e2, coins, (uint8_t)(2 * K), ETA2);

    polyvec_ntt(&y);

    /* u = invNTT(A^T @ y) + e1. */
    mlkem_polyvec_t u;
    for (int i = 0; i < K; i++) {
        mlkem_polyvec_t row;
        for (int j = 0; j < K; j++) row.vec[j] = A[i * K + j];
        polyvec_basemul_accum(&u.vec[i], &row, &y);
    }
    polyvec_invntt(&u);
    polyvec_add(&u, &u, &e1);
    polyvec_reduce(&u);

    /* v = invNTT(t^T @ y) + e2 + Decompress_1(msg). */
    mlkem_poly_t v;
    polyvec_basemul_accum(&v, &t, &y);
    mlkem_poly_invntt(&v);
    mlkem_poly_add(&v, &v, &e2);
    mlkem_poly_t mp;
    poly_from_msg(&mp, msg);
    mlkem_poly_add(&v, &v, &mp);
    mlkem_poly_reduce(&v);

    polyvec_compress(c, &u, DU);
    mlkem_poly_compress(c + MLKEM512_POLYVEC_COMPRESSED, &v, DV);
}

static void kpke_decrypt(uint8_t msg[32],
                          const uint8_t sk[MLKEM512_POLYVECBYTES],
                          const uint8_t c[MLKEM512_CIPHERTEXTBYTES]) {
    mlkem_polyvec_t u;
    mlkem_poly_t    v;
    polyvec_decompress(&u, c, DU);
    mlkem_poly_decompress(&v, c + MLKEM512_POLYVEC_COMPRESSED, DV);

    mlkem_polyvec_t s;
    polyvec_frombytes(&s, sk);

    /* w = v - invNTT(s^T @ NTT(u)). */
    polyvec_ntt(&u);
    mlkem_poly_t t;
    polyvec_basemul_accum(&t, &s, &u);
    mlkem_poly_invntt(&t);
    mlkem_poly_sub(&v, &v, &t);
    mlkem_poly_reduce(&v);

    poly_to_msg(msg, &v);
}

/* -------------------------------------------------------------- */
/* ML-KEM KeyGen / Encaps / Decaps (Section 7)                     */
/* -------------------------------------------------------------- */

void moonlab_mlkem512_keygen(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              uint8_t dk[MLKEM512_SECRETKEYBYTES],
                              const uint8_t d[32],
                              const uint8_t z[32]) {
    uint8_t pk[MLKEM512_PUBLICKEYBYTES];
    uint8_t sk_pke[MLKEM512_POLYVECBYTES];
    kpke_keygen(pk, sk_pke, d);
    memcpy(ek, pk, MLKEM512_PUBLICKEYBYTES);
    /* dk = sk_pke || ek || H(ek) || z. */
    memcpy(dk, sk_pke, MLKEM512_POLYVECBYTES);
    memcpy(dk + MLKEM512_POLYVECBYTES, pk, MLKEM512_PUBLICKEYBYTES);
    uint8_t hpk[32];
    sha3_256(pk, MLKEM512_PUBLICKEYBYTES, hpk);
    memcpy(dk + MLKEM512_POLYVECBYTES + MLKEM512_PUBLICKEYBYTES, hpk, 32);
    memcpy(dk + MLKEM512_POLYVECBYTES + MLKEM512_PUBLICKEYBYTES + 32, z, 32);
}

void moonlab_mlkem512_encaps(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              uint8_t K_out[32],
                              const uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              const uint8_t m[32]) {
    /* (K, r) = SHA3-512(m || H(ek)). */
    uint8_t hpk[32];
    sha3_256(ek, MLKEM512_PUBLICKEYBYTES, hpk);

    uint8_t input[64];
    memcpy(input, m, 32);
    memcpy(input + 32, hpk, 32);
    uint8_t Kr[64];
    sha3_512(input, 64, Kr);
    memcpy(K_out, Kr, 32);
    const uint8_t *r = Kr + 32;

    kpke_encrypt(c, ek, m, r);
}

void moonlab_mlkem512_decaps(uint8_t K_out[32],
                              const uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM512_SECRETKEYBYTES]) {
    const uint8_t *sk_pke = dk;
    const uint8_t *ek     = dk + MLKEM512_POLYVECBYTES;
    const uint8_t *h      = dk + MLKEM512_POLYVECBYTES + MLKEM512_PUBLICKEYBYTES;
    const uint8_t *z      = h + 32;

    uint8_t m_prime[32];
    kpke_decrypt(m_prime, sk_pke, c);

    /* (K', r') = SHA3-512(m' || h). */
    uint8_t input[64];
    memcpy(input, m_prime, 32);
    memcpy(input + 32, h, 32);
    uint8_t Kr[64];
    sha3_512(input, 64, Kr);
    const uint8_t *K_candidate = Kr;
    const uint8_t *r_prime     = Kr + 32;

    /* Re-encrypt, compare, and select via constant-time mask. */
    uint8_t c_prime[MLKEM512_CIPHERTEXTBYTES];
    kpke_encrypt(c_prime, ek, m_prime, r_prime);

    /* Implicit rejection: K_bad = SHAKE256(z || c, 32). */
    sha3_ctx_t sh; shake256_init(&sh);
    sha3_update(&sh, z, 32);
    sha3_update(&sh, c, MLKEM512_CIPHERTEXTBYTES);
    uint8_t K_bad[32];
    shake_squeeze(&sh, K_bad, 32);

    /* Constant-time equality: diff = OR of XOR-bytes. */
    uint8_t diff = 0;
    for (size_t i = 0; i < MLKEM512_CIPHERTEXTBYTES; i++) {
        diff |= (uint8_t)(c[i] ^ c_prime[i]);
    }
    /* mask = 0 if match, 0xFF if mismatch. */
    uint8_t mask = (uint8_t)(-(int)((diff | (uint8_t)-diff) >> 7));
    for (int i = 0; i < 32; i++) {
        K_out[i] = (uint8_t)((K_candidate[i] & (uint8_t)~mask) |
                              (K_bad[i]     & mask));
    }
}

/* -------------------------------------------------------------- */
/* QRNG-sourced convenience wrappers                               */
/* -------------------------------------------------------------- */

int moonlab_mlkem512_keygen_qrng(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                                   uint8_t dk[MLKEM512_SECRETKEYBYTES]) {
    uint8_t buf[64];
    if (moonlab_qrng_bytes(buf, sizeof buf) != 0) return -1;
    moonlab_mlkem512_keygen(ek, dk, buf, buf + 32);
    return 0;
}

int moonlab_mlkem512_encaps_qrng(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                                   uint8_t K_out[32],
                                   const uint8_t ek[MLKEM512_PUBLICKEYBYTES]) {
    uint8_t m[32];
    if (moonlab_qrng_bytes(m, sizeof m) != 0) return -1;
    moonlab_mlkem512_encaps(c, K_out, ek, m);
    return 0;
}
