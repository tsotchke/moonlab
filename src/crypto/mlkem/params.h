/**
 * @file params.h
 * @brief ML-KEM (FIPS 203) shared parameters.
 *
 * Parameter sets (Table 2, FIPS 203):
 *   ML-KEM-512  : k = 2, eta1 = 3, eta2 = 2, du = 10, dv = 4
 *   ML-KEM-768  : k = 3, eta1 = 2, eta2 = 2, du = 10, dv = 4
 *   ML-KEM-1024 : k = 4, eta1 = 2, eta2 = 2, du = 11, dv = 5
 *
 * Only ML-KEM-512 is wired up in v0.2 (enough for end-to-end KEM
 * demo + KAT validation); the remaining parameter sets are
 * one-struct changes once the primitives are solid.
 */

#ifndef MOONLAB_MLKEM_PARAMS_H
#define MOONLAB_MLKEM_PARAMS_H

/* Ring Z_q[X] / (X^n + 1). */
#define MLKEM_N 256
#define MLKEM_Q 3329

/* Encoded sizes in bytes -- common to all parameter sets. */
#define MLKEM_SYMBYTES      32   /* shared secret length */
#define MLKEM_POLYBYTES     384  /* 12-bit packed polynomial */

/* ML-KEM-512 values. */
#define MLKEM512_K    2
#define MLKEM512_ETA1 3
#define MLKEM512_ETA2 2
#define MLKEM512_DU   10
#define MLKEM512_DV   4

/* ML-KEM-512 derived sizes. */
#define MLKEM512_POLYVECBYTES          (MLKEM512_K * MLKEM_POLYBYTES)
#define MLKEM512_POLYVEC_COMPRESSED    (MLKEM512_K * 320)  /* k * n * du / 8 */
#define MLKEM512_POLY_COMPRESSED       128                  /* n * dv / 8   */
#define MLKEM512_PUBLICKEYBYTES        (MLKEM512_POLYVECBYTES + MLKEM_SYMBYTES)
#define MLKEM512_SECRETKEYBYTES        (MLKEM512_POLYVECBYTES + \
                                         MLKEM512_PUBLICKEYBYTES + \
                                         2 * MLKEM_SYMBYTES)
#define MLKEM512_CIPHERTEXTBYTES       (MLKEM512_POLYVEC_COMPRESSED + \
                                         MLKEM512_POLY_COMPRESSED)

#endif
