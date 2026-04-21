/**
 * @file params.h
 * @brief ML-KEM (FIPS 203) shared parameters for all three parameter sets.
 *
 * Parameter sets (FIPS 203 Table 2):
 *   ML-KEM-512  : k = 2, eta1 = 3, eta2 = 2, du = 10, dv = 4
 *   ML-KEM-768  : k = 3, eta1 = 2, eta2 = 2, du = 10, dv = 4
 *   ML-KEM-1024 : k = 4, eta1 = 2, eta2 = 2, du = 11, dv = 5
 */

#ifndef MOONLAB_MLKEM_PARAMS_H
#define MOONLAB_MLKEM_PARAMS_H

/* Ring Z_q[X] / (X^n + 1). */
#define MLKEM_N 256
#define MLKEM_Q 3329

/* Common encoded sizes. */
#define MLKEM_SYMBYTES      32   /* shared secret length */
#define MLKEM_POLYBYTES     384  /* 12-bit packed polynomial */

/* Upper bound on k used to size static polyvec buffers. */
#define MLKEM_MAX_K  4

/* ---- ML-KEM-512 ------------------------------------------------- */
#define MLKEM512_K                   2
#define MLKEM512_ETA1                3
#define MLKEM512_ETA2                2
#define MLKEM512_DU                  10
#define MLKEM512_DV                  4
#define MLKEM512_POLYVECBYTES        (MLKEM512_K * MLKEM_POLYBYTES)
#define MLKEM512_POLYVEC_COMPRESSED  (MLKEM512_K * 320)   /* k * n * du / 8 */
#define MLKEM512_POLY_COMPRESSED     128                    /* n * dv / 8   */
#define MLKEM512_PUBLICKEYBYTES      (MLKEM512_POLYVECBYTES + MLKEM_SYMBYTES)
#define MLKEM512_SECRETKEYBYTES      (MLKEM512_POLYVECBYTES + \
                                       MLKEM512_PUBLICKEYBYTES + 2 * MLKEM_SYMBYTES)
#define MLKEM512_CIPHERTEXTBYTES     (MLKEM512_POLYVEC_COMPRESSED + \
                                       MLKEM512_POLY_COMPRESSED)

/* ---- ML-KEM-768 ------------------------------------------------- */
#define MLKEM768_K                   3
#define MLKEM768_ETA1                2
#define MLKEM768_ETA2                2
#define MLKEM768_DU                  10
#define MLKEM768_DV                  4
#define MLKEM768_POLYVECBYTES        (MLKEM768_K * MLKEM_POLYBYTES)
#define MLKEM768_POLYVEC_COMPRESSED  (MLKEM768_K * 320)
#define MLKEM768_POLY_COMPRESSED     128
#define MLKEM768_PUBLICKEYBYTES      (MLKEM768_POLYVECBYTES + MLKEM_SYMBYTES)
#define MLKEM768_SECRETKEYBYTES      (MLKEM768_POLYVECBYTES + \
                                       MLKEM768_PUBLICKEYBYTES + 2 * MLKEM_SYMBYTES)
#define MLKEM768_CIPHERTEXTBYTES     (MLKEM768_POLYVEC_COMPRESSED + \
                                       MLKEM768_POLY_COMPRESSED)

/* ---- ML-KEM-1024 ------------------------------------------------ */
#define MLKEM1024_K                   4
#define MLKEM1024_ETA1                2
#define MLKEM1024_ETA2                2
#define MLKEM1024_DU                  11
#define MLKEM1024_DV                  5
#define MLKEM1024_POLYVECBYTES        (MLKEM1024_K * MLKEM_POLYBYTES)
#define MLKEM1024_POLYVEC_COMPRESSED  (MLKEM1024_K * 352)
#define MLKEM1024_POLY_COMPRESSED     160
#define MLKEM1024_PUBLICKEYBYTES      (MLKEM1024_POLYVECBYTES + MLKEM_SYMBYTES)
#define MLKEM1024_SECRETKEYBYTES      (MLKEM1024_POLYVECBYTES + \
                                        MLKEM1024_PUBLICKEYBYTES + 2 * MLKEM_SYMBYTES)
#define MLKEM1024_CIPHERTEXTBYTES     (MLKEM1024_POLYVEC_COMPRESSED + \
                                        MLKEM1024_POLY_COMPRESSED)

/**
 * @brief Runtime parameter bundle.  Used internally by the generic
 *        KEM engine so the three parameter sets can share code.
 */
typedef struct {
    int k;
    int eta1;
    int eta2;
    int du;
    int dv;
    /* Derived. */
    int polyvec_bytes;
    int polyvec_compressed;
    int poly_compressed;
    int publickey_bytes;
    int secretkey_bytes;
    int ciphertext_bytes;
} mlkem_params_t;

#ifdef __cplusplus
extern "C" {
#endif

extern const mlkem_params_t MLKEM_512_PARAMS;
extern const mlkem_params_t MLKEM_768_PARAMS;
extern const mlkem_params_t MLKEM_1024_PARAMS;

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MLKEM_PARAMS_H */
