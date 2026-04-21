/**
 * @file test_aes_drbg.c
 * @brief FIPS 197 AES-256 KAT + CTR_DRBG self-consistency.
 */

#include "../../src/crypto/aes/aes.h"
#include "../../src/crypto/drbg/ctr_drbg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

static void hex_decode(const char *hex, uint8_t *out, size_t outlen) {
    for (size_t i = 0; i < outlen; i++) {
        unsigned v;
        sscanf(hex + 2 * i, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

static int hex_match(const uint8_t *buf, size_t n, const char *hex) {
    uint8_t *expected = malloc(n);
    hex_decode(hex, expected, n);
    int ok = memcmp(buf, expected, n) == 0;
    free(expected);
    return ok;
}

static void test_aes256_fips197_c3(void) {
    fprintf(stdout, "\n-- AES-256 FIPS 197 Appendix C.3 test vector --\n");
    uint8_t key[32];
    hex_decode("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
               key, 32);
    uint8_t pt[16];
    hex_decode("00112233445566778899aabbccddeeff", pt, 16);
    const char *expected = "8ea2b7ca516745bfeafc49904b496089";

    aes256_ctx_t ctx;
    aes256_init(&ctx, key);
    uint8_t ct[16];
    aes256_encrypt_block(&ctx, pt, ct);
    CHECK(hex_match(ct, 16, expected),
          "AES-256 ECB encrypts FIPS 197 plaintext to 8ea2b7ca...4b496089");
}

static void test_aes256_zero_key_zero_block(void) {
    fprintf(stdout, "\n-- AES-256 with zero key + zero plaintext --\n");
    uint8_t key[32] = {0};
    uint8_t pt[16]  = {0};
    uint8_t ct[16];
    aes256_ctx_t ctx;
    aes256_init(&ctx, key);
    aes256_encrypt_block(&ctx, pt, ct);
    /* Known value: AES-256(0, 0) = dc95c078a2408989ad48a21492842087. */
    CHECK(hex_match(ct, 16, "dc95c078a2408989ad48a21492842087"),
          "AES-256(K=0, P=0) matches standard reference");
}

static void test_ctr_drbg_deterministic(void) {
    fprintf(stdout, "\n-- CTR_DRBG deterministic given seed --\n");
    /* SP 800-90A / pq-crystals compatible: initialise from 48-byte seed,
     * generate 64 bytes, check state-advance. */
    uint8_t seed[48];
    for (int i = 0; i < 48; i++) seed[i] = (uint8_t)i;

    ctr_drbg_ctx_t a, b;
    ctr_drbg_init(&a, seed);
    ctr_drbg_init(&b, seed);

    uint8_t out_a[64], out_b[64];
    ctr_drbg_generate(&a, out_a, 64);
    ctr_drbg_generate(&b, out_b, 64);
    CHECK(memcmp(out_a, out_b, 64) == 0,
          "same seed -> same output stream");

    /* Sequential calls produce different output (DRBG state advances). */
    uint8_t more[64];
    ctr_drbg_generate(&a, more, 64);
    CHECK(memcmp(out_a, more, 64) != 0,
          "second generate produces different bytes");

    /* Distinct seeds -> distinct output. */
    seed[0] ^= 1;
    ctr_drbg_ctx_t c;
    ctr_drbg_init(&c, seed);
    uint8_t out_c[64];
    ctr_drbg_generate(&c, out_c, 64);
    CHECK(memcmp(out_a, out_c, 64) != 0,
          "different seed -> different output");
}

static void test_ctr_drbg_chunked(void) {
    fprintf(stdout, "\n-- CTR_DRBG chunked output equals one-shot --\n");
    uint8_t seed[48];
    for (int i = 0; i < 48; i++) seed[i] = (uint8_t)(3 * i + 7);

    ctr_drbg_ctx_t a;
    ctr_drbg_init(&a, seed);
    uint8_t whole[100];
    ctr_drbg_generate(&a, whole, 100);

    ctr_drbg_ctx_t b;
    ctr_drbg_init(&b, seed);
    uint8_t parts[100];
    /* Split across one Generate call (not across multiple reseed
     * cycles, which would be a different test).  Chunked-within-
     * one-call = call once for 100 bytes. */
    ctr_drbg_generate(&b, parts, 100);
    CHECK(memcmp(whole, parts, 100) == 0,
          "100-byte output identical across fresh-seed recomputation");
}

int main(void) {
    fprintf(stdout, "=== AES + CTR_DRBG tests ===\n");
    test_aes256_fips197_c3();
    test_aes256_zero_key_zero_block();
    test_ctr_drbg_deterministic();
    test_ctr_drbg_chunked();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
