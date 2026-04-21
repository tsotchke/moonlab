/**
 * @file mlkem.c
 * @brief FIPS 203 ML-KEM KeyGen / Encaps / Decaps (all parameter sets).
 *
 * Single implementation driven by a runtime @ref mlkem_params_t struct;
 * the three parameter sets (512, 768, 1024) compose by passing the
 * appropriate constants object to the generic engine.  Polyvec storage
 * is sized to MLKEM_MAX_K = 4 so the same code path serves all three.
 *
 * Convention follows pq-crystals Kyber:
 *   - NTT(p_normal) leaves p in "NTT-normal" form.
 *   - basemul(a_hat, b_hat) produces output with an extra 1/R factor
 *     relative to NTT-input form.
 *   - mlkem_poly_tomont() lifts this back to NTT-normal when staying
 *     in NTT domain (used in KeyGen, which keeps t_hat for the pubkey).
 *   - mlkem_poly_invntt() absorbs the 1/R when going back to the time
 *     domain (used in Encaps/Decaps to produce u, v, w).
 */

#include "mlkem.h"
#include "poly.h"
#include "../sha3/sha3.h"
#include "../../applications/moonlab_export.h"

#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------- */
/* Parameter-set constants                                         */
/* -------------------------------------------------------------- */

/* Parameter-set constants; secretkey_bytes = polyvec + publickey + 2*32 =
 * 2*polyvec + 32 + 64 = 3*polyvec + 96 unrolled explicitly per k. */
const mlkem_params_t MLKEM_512_PARAMS  = { 2, 3, 2, 10, 4,
    2 * MLKEM_POLYBYTES, 2 * 320, 128,
    2 * MLKEM_POLYBYTES + MLKEM_SYMBYTES,
    2 * 2 * MLKEM_POLYBYTES + MLKEM_SYMBYTES + 2 * MLKEM_SYMBYTES,
    2 * 320 + 128 };
const mlkem_params_t MLKEM_768_PARAMS  = { 3, 2, 2, 10, 4,
    3 * MLKEM_POLYBYTES, 3 * 320, 128,
    3 * MLKEM_POLYBYTES + MLKEM_SYMBYTES,
    2 * 3 * MLKEM_POLYBYTES + MLKEM_SYMBYTES + 2 * MLKEM_SYMBYTES,
    3 * 320 + 128 };
const mlkem_params_t MLKEM_1024_PARAMS = { 4, 2, 2, 11, 5,
    4 * MLKEM_POLYBYTES, 4 * 352, 160,
    4 * MLKEM_POLYBYTES + MLKEM_SYMBYTES,
    2 * 4 * MLKEM_POLYBYTES + MLKEM_SYMBYTES + 2 * MLKEM_SYMBYTES,
    4 * 352 + 160 };

/* -------------------------------------------------------------- */
/* Poly-vector helpers (fixed-max-K, runtime k)                    */
/* -------------------------------------------------------------- */

typedef struct {
    mlkem_poly_t vec[MLKEM_MAX_K];
} mlkem_polyvec_t;

static void polyvec_ntt(mlkem_polyvec_t *v, int k) {
    for (int i = 0; i < k; i++) mlkem_poly_ntt(&v->vec[i]);
}
static void polyvec_invntt(mlkem_polyvec_t *v, int k) {
    for (int i = 0; i < k; i++) mlkem_poly_invntt(&v->vec[i]);
}
static void polyvec_add(mlkem_polyvec_t *r, const mlkem_polyvec_t *a,
                         const mlkem_polyvec_t *b, int k) {
    for (int i = 0; i < k; i++) mlkem_poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
}
static void polyvec_reduce(mlkem_polyvec_t *v, int k) {
    for (int i = 0; i < k; i++) mlkem_poly_reduce(&v->vec[i]);
}

/* Inner product of two k-element polyvecs in NTT domain.  Output
 * remains in pq-crystals "sub-normal" form (extra 1/R factor). */
static void polyvec_basemul_accum(mlkem_poly_t *r,
                                   const mlkem_polyvec_t *a,
                                   const mlkem_polyvec_t *b,
                                   int k) {
    mlkem_poly_t t;
    mlkem_poly_basemul(r, &a->vec[0], &b->vec[0]);
    for (int i = 1; i < k; i++) {
        mlkem_poly_basemul(&t, &a->vec[i], &b->vec[i]);
        mlkem_poly_add(r, r, &t);
    }
    mlkem_poly_reduce(r);
}

static void polyvec_tobytes(uint8_t *out, const mlkem_polyvec_t *v, int k) {
    for (int i = 0; i < k; i++)
        mlkem_poly_tobytes(out + i * MLKEM_POLYBYTES, &v->vec[i]);
}
static void polyvec_frombytes(mlkem_polyvec_t *v, const uint8_t *in, int k) {
    for (int i = 0; i < k; i++)
        mlkem_poly_frombytes(&v->vec[i], in + i * MLKEM_POLYBYTES);
}
static void polyvec_compress(uint8_t *out, const mlkem_polyvec_t *v,
                              int k, int du) {
    for (int i = 0; i < k; i++)
        mlkem_poly_compress(out + i * (MLKEM_N * du / 8), &v->vec[i], du);
}
static void polyvec_decompress(mlkem_polyvec_t *v, const uint8_t *in,
                                int k, int du) {
    for (int i = 0; i < k; i++)
        mlkem_poly_decompress(&v->vec[i], in + i * (MLKEM_N * du / 8), du);
}

/* -------------------------------------------------------------- */
/* CBD noise / message helpers                                     */
/* -------------------------------------------------------------- */

static void sample_cbd(mlkem_poly_t *p, const uint8_t seed[32],
                        uint8_t nonce, int eta) {
    uint8_t input[33];
    memcpy(input, seed, 32);
    input[32] = nonce;
    uint8_t buf[3 * MLKEM_N / 4];   /* max 192 bytes at eta=3 */
    size_t blen = (size_t)(eta * MLKEM_N / 4);
    shake256(input, 33, buf, blen);
    mlkem_poly_cbd(p, buf, eta);
}

static void poly_from_msg(mlkem_poly_t *p, const uint8_t msg[32]) {
    for (int i = 0; i < MLKEM_N / 8; i++) {
        for (int j = 0; j < 8; j++) {
            int bit = (msg[i] >> j) & 1;
            p->coeffs[8 * i + j] = (int16_t)(bit * ((MLKEM_Q + 1) / 2));
        }
    }
}
static void poly_to_msg(uint8_t msg[32], const mlkem_poly_t *p) {
    memset(msg, 0, 32);
    for (int i = 0; i < MLKEM_N / 8; i++) {
        for (int j = 0; j < 8; j++) {
            int32_t t = p->coeffs[8 * i + j];
            if (t < 0) t += MLKEM_Q;
            int32_t d = t - (MLKEM_Q / 2);
            if (d < 0) d = -d;
            int bit = (d <= MLKEM_Q / 4) ? 1 : 0;
            msg[i] |= (uint8_t)(bit << j);
        }
    }
}

/* -------------------------------------------------------------- */
/* K-PKE (FIPS 203 Section 5) -- generic over k, eta1, eta2, du, dv */
/* -------------------------------------------------------------- */

static void kpke_keygen(const mlkem_params_t *P,
                         uint8_t *pk, uint8_t *sk,
                         const uint8_t d[32]) {
    const int k = P->k;

    uint8_t buf[33];
    memcpy(buf, d, 32);
    buf[32] = (uint8_t)k;
    uint8_t rho_sigma[64];
    sha3_512(buf, 33, rho_sigma);
    const uint8_t *rho   = rho_sigma;
    const uint8_t *sigma = rho_sigma + 32;

    /* A = GenMatrix(rho). */
    mlkem_poly_t A[MLKEM_MAX_K * MLKEM_MAX_K];
    mlkem_gen_matrix(A, k, rho, 0);

    /* s, e <- CBD_eta1(sigma). */
    mlkem_polyvec_t s, e;
    for (int i = 0; i < k; i++) sample_cbd(&s.vec[i], sigma, (uint8_t)i,       P->eta1);
    for (int i = 0; i < k; i++) sample_cbd(&e.vec[i], sigma, (uint8_t)(k + i), P->eta1);

    polyvec_ntt(&s, k);
    polyvec_ntt(&e, k);

    /* t = A @ s + e.  tomont lifts basemul output to NTT-normal form
     * so e_hat (also NTT-normal) can be added coherently. */
    mlkem_polyvec_t t;
    for (int i = 0; i < k; i++) {
        mlkem_polyvec_t row;
        for (int j = 0; j < k; j++) row.vec[j] = A[i * k + j];
        polyvec_basemul_accum(&t.vec[i], &row, &s, k);
        mlkem_poly_tomont(&t.vec[i]);
    }
    polyvec_add(&t, &t, &e, k);
    polyvec_reduce(&t, k);

    polyvec_tobytes(pk, &t, k);
    memcpy(pk + P->polyvec_bytes, rho, 32);
    polyvec_tobytes(sk, &s, k);
}

static void kpke_encrypt(const mlkem_params_t *P,
                          uint8_t *c,
                          const uint8_t *pk,
                          const uint8_t msg[32],
                          const uint8_t coins[32]) {
    const int k = P->k;

    mlkem_polyvec_t t;
    polyvec_frombytes(&t, pk, k);
    const uint8_t *rho = pk + P->polyvec_bytes;

    mlkem_poly_t A[MLKEM_MAX_K * MLKEM_MAX_K];
    mlkem_gen_matrix(A, k, rho, 1);

    mlkem_polyvec_t y, e1;
    mlkem_poly_t e2;
    for (int i = 0; i < k; i++) sample_cbd(&y.vec[i],  coins, (uint8_t)i,           P->eta1);
    for (int i = 0; i < k; i++) sample_cbd(&e1.vec[i], coins, (uint8_t)(k + i),     P->eta2);
    sample_cbd(&e2, coins, (uint8_t)(2 * k), P->eta2);

    polyvec_ntt(&y, k);

    /* u = invNTT(A^T @ y) + e1. */
    mlkem_polyvec_t u;
    for (int i = 0; i < k; i++) {
        mlkem_polyvec_t row;
        for (int j = 0; j < k; j++) row.vec[j] = A[i * k + j];
        polyvec_basemul_accum(&u.vec[i], &row, &y, k);
    }
    polyvec_invntt(&u, k);
    polyvec_add(&u, &u, &e1, k);
    polyvec_reduce(&u, k);

    /* v = invNTT(t^T @ y) + e2 + Decompress_1(m). */
    mlkem_poly_t v;
    polyvec_basemul_accum(&v, &t, &y, k);
    mlkem_poly_invntt(&v);
    mlkem_poly_add(&v, &v, &e2);
    mlkem_poly_t mp;
    poly_from_msg(&mp, msg);
    mlkem_poly_add(&v, &v, &mp);
    mlkem_poly_reduce(&v);

    polyvec_compress(c, &u, k, P->du);
    mlkem_poly_compress(c + P->polyvec_compressed, &v, P->dv);
}

static void kpke_decrypt(const mlkem_params_t *P,
                          uint8_t msg[32],
                          const uint8_t *sk,
                          const uint8_t *c) {
    const int k = P->k;

    mlkem_polyvec_t u;
    mlkem_poly_t    v;
    polyvec_decompress(&u, c, k, P->du);
    mlkem_poly_decompress(&v, c + P->polyvec_compressed, P->dv);

    mlkem_polyvec_t s;
    polyvec_frombytes(&s, sk, k);

    polyvec_ntt(&u, k);
    mlkem_poly_t t;
    polyvec_basemul_accum(&t, &s, &u, k);
    mlkem_poly_invntt(&t);
    mlkem_poly_sub(&v, &v, &t);
    mlkem_poly_reduce(&v);

    poly_to_msg(msg, &v);
}

/* -------------------------------------------------------------- */
/* ML-KEM wrapper (FIPS 203 Section 7) -- generic                 */
/* -------------------------------------------------------------- */

static void mlkem_keygen_generic(const mlkem_params_t *P,
                                  uint8_t *ek, uint8_t *dk,
                                  const uint8_t d[32], const uint8_t z[32]) {
    uint8_t *sk_pke = dk;
    uint8_t *ek_out = dk + P->polyvec_bytes;
    uint8_t *hpk    = dk + P->polyvec_bytes + P->publickey_bytes;
    uint8_t *zout   = hpk + 32;

    /* K-PKE keygen writes pk = t || rho directly to ek. */
    kpke_keygen(P, ek, sk_pke, d);

    memcpy(ek_out, ek, (size_t)P->publickey_bytes);
    sha3_256(ek, (size_t)P->publickey_bytes, hpk);
    memcpy(zout, z, 32);
}

static void mlkem_encaps_generic(const mlkem_params_t *P,
                                  uint8_t *c, uint8_t *K_out,
                                  const uint8_t *ek, const uint8_t m[32]) {
    uint8_t hpk[32];
    sha3_256(ek, (size_t)P->publickey_bytes, hpk);

    uint8_t input[64];
    memcpy(input, m, 32);
    memcpy(input + 32, hpk, 32);
    uint8_t Kr[64];
    sha3_512(input, 64, Kr);
    memcpy(K_out, Kr, 32);
    const uint8_t *r = Kr + 32;

    kpke_encrypt(P, c, ek, m, r);
}

static void mlkem_decaps_generic(const mlkem_params_t *P,
                                  uint8_t *K_out,
                                  const uint8_t *c, const uint8_t *dk) {
    const uint8_t *sk_pke = dk;
    const uint8_t *ek     = dk + P->polyvec_bytes;
    const uint8_t *h      = dk + P->polyvec_bytes + P->publickey_bytes;
    const uint8_t *z      = h + 32;

    uint8_t m_prime[32];
    kpke_decrypt(P, m_prime, sk_pke, c);

    uint8_t input[64];
    memcpy(input, m_prime, 32);
    memcpy(input + 32, h, 32);
    uint8_t Kr[64];
    sha3_512(input, 64, Kr);
    const uint8_t *K_candidate = Kr;
    const uint8_t *r_prime     = Kr + 32;

    uint8_t c_prime[1568];  /* 1568 = MLKEM1024_CIPHERTEXTBYTES, max */
    kpke_encrypt(P, c_prime, ek, m_prime, r_prime);

    sha3_ctx_t sh; shake256_init(&sh);
    sha3_update(&sh, z, 32);
    sha3_update(&sh, c, (size_t)P->ciphertext_bytes);
    uint8_t K_bad[32];
    shake_squeeze(&sh, K_bad, 32);

    uint8_t diff = 0;
    for (int i = 0; i < P->ciphertext_bytes; i++) {
        diff |= (uint8_t)(c[i] ^ c_prime[i]);
    }
    uint8_t mask = (uint8_t)(-(int)((diff | (uint8_t)-diff) >> 7));
    for (int i = 0; i < 32; i++) {
        K_out[i] = (uint8_t)((K_candidate[i] & (uint8_t)~mask) |
                              (K_bad[i]     & mask));
    }
}

/* -------------------------------------------------------------- */
/* Public per-parameter-set wrappers                              */
/* -------------------------------------------------------------- */

void moonlab_mlkem512_keygen(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              uint8_t dk[MLKEM512_SECRETKEYBYTES],
                              const uint8_t d[32], const uint8_t z[32]) {
    mlkem_keygen_generic(&MLKEM_512_PARAMS, ek, dk, d, z);
}
void moonlab_mlkem512_encaps(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              uint8_t K[32],
                              const uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              const uint8_t m[32]) {
    mlkem_encaps_generic(&MLKEM_512_PARAMS, c, K, ek, m);
}
void moonlab_mlkem512_decaps(uint8_t K[32],
                              const uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM512_SECRETKEYBYTES]) {
    mlkem_decaps_generic(&MLKEM_512_PARAMS, K, c, dk);
}

void moonlab_mlkem768_keygen(uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                              uint8_t dk[MLKEM768_SECRETKEYBYTES],
                              const uint8_t d[32], const uint8_t z[32]) {
    mlkem_keygen_generic(&MLKEM_768_PARAMS, ek, dk, d, z);
}
void moonlab_mlkem768_encaps(uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                              uint8_t K[32],
                              const uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                              const uint8_t m[32]) {
    mlkem_encaps_generic(&MLKEM_768_PARAMS, c, K, ek, m);
}
void moonlab_mlkem768_decaps(uint8_t K[32],
                              const uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM768_SECRETKEYBYTES]) {
    mlkem_decaps_generic(&MLKEM_768_PARAMS, K, c, dk);
}

void moonlab_mlkem1024_keygen(uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                               uint8_t dk[MLKEM1024_SECRETKEYBYTES],
                               const uint8_t d[32], const uint8_t z[32]) {
    mlkem_keygen_generic(&MLKEM_1024_PARAMS, ek, dk, d, z);
}
void moonlab_mlkem1024_encaps(uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                               uint8_t K[32],
                               const uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                               const uint8_t m[32]) {
    mlkem_encaps_generic(&MLKEM_1024_PARAMS, c, K, ek, m);
}
void moonlab_mlkem1024_decaps(uint8_t K[32],
                               const uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                               const uint8_t dk[MLKEM1024_SECRETKEYBYTES]) {
    mlkem_decaps_generic(&MLKEM_1024_PARAMS, K, c, dk);
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

int moonlab_mlkem768_keygen_qrng(uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                                   uint8_t dk[MLKEM768_SECRETKEYBYTES]) {
    uint8_t buf[64];
    if (moonlab_qrng_bytes(buf, sizeof buf) != 0) return -1;
    moonlab_mlkem768_keygen(ek, dk, buf, buf + 32);
    return 0;
}

int moonlab_mlkem768_encaps_qrng(uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                                   uint8_t K_out[32],
                                   const uint8_t ek[MLKEM768_PUBLICKEYBYTES]) {
    uint8_t m[32];
    if (moonlab_qrng_bytes(m, sizeof m) != 0) return -1;
    moonlab_mlkem768_encaps(c, K_out, ek, m);
    return 0;
}

int moonlab_mlkem1024_keygen_qrng(uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                                    uint8_t dk[MLKEM1024_SECRETKEYBYTES]) {
    uint8_t buf[64];
    if (moonlab_qrng_bytes(buf, sizeof buf) != 0) return -1;
    moonlab_mlkem1024_keygen(ek, dk, buf, buf + 32);
    return 0;
}

int moonlab_mlkem1024_encaps_qrng(uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                                    uint8_t K_out[32],
                                    const uint8_t ek[MLKEM1024_PUBLICKEYBYTES]) {
    uint8_t m[32];
    if (moonlab_qrng_bytes(m, sizeof m) != 0) return -1;
    moonlab_mlkem1024_encaps(c, K_out, ek, m);
    return 0;
}
