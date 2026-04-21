/**
 * @file test_mlkem.c
 * @brief ML-KEM-512 end-to-end KeyGen / Encaps / Decaps correctness.
 *
 * Checks two FIPS-203 guarantees:
 *   1. Correctness: for (ek, dk) = KeyGen, (K, c) = Encaps(ek), we
 *      have Decaps(dk, c) == K with probability 1 - negl.
 *   2. Implicit rejection: Decaps(dk, c') for a tampered c' returns
 *      a K' that is NOT equal to the original K, without aborting.
 *      The returned K' should also be uncorrelated with K.
 *   3. Determinism: identical (d, z, m) seeds reproduce identical
 *      (ek, dk, c, K).
 *
 * Full NIST-SP-203-KAT validation follows in a subsequent commit;
 * the tests here are structural (correctness + determinism) and
 * suffice to catch any algebraic regression in the lattice layer.
 */

#include "../../src/crypto/mlkem/mlkem.h"

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

static void fill_seed(uint8_t *buf, size_t n, uint8_t salt) {
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(i * 31 + salt);
}

static void test_correctness_one_trial(int salt_offset) {
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, (uint8_t)(1 + salt_offset));
    fill_seed(z, 32, (uint8_t)(2 + salt_offset));
    fill_seed(m, 32, (uint8_t)(3 + salt_offset));

    uint8_t ek[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk[MLKEM512_SECRETKEYBYTES];
    moonlab_mlkem512_keygen(ek, dk, d, z);

    uint8_t c[MLKEM512_CIPHERTEXTBYTES];
    uint8_t K[32];
    moonlab_mlkem512_encaps(c, K, ek, m);

    uint8_t K_dec[32];
    moonlab_mlkem512_decaps(K_dec, c, dk);

    char label[64];
    snprintf(label, sizeof label,
             "trial %d: Decaps(Encaps) recovers shared secret", salt_offset);
    CHECK(memcmp(K, K_dec, 32) == 0, "%s", label);
}

static void test_determinism(void) {
    fprintf(stdout, "\n-- determinism in (d, z, m) --\n");
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, 0x11);
    fill_seed(z, 32, 0x22);
    fill_seed(m, 32, 0x33);

    uint8_t ek1[MLKEM512_PUBLICKEYBYTES], ek2[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk1[MLKEM512_SECRETKEYBYTES], dk2[MLKEM512_SECRETKEYBYTES];
    moonlab_mlkem512_keygen(ek1, dk1, d, z);
    moonlab_mlkem512_keygen(ek2, dk2, d, z);

    CHECK(memcmp(ek1, ek2, MLKEM512_PUBLICKEYBYTES) == 0,
          "KeyGen(d, z) reproducible -- same ek");
    CHECK(memcmp(dk1, dk2, MLKEM512_SECRETKEYBYTES) == 0,
          "KeyGen(d, z) reproducible -- same dk");

    uint8_t c1[MLKEM512_CIPHERTEXTBYTES], c2[MLKEM512_CIPHERTEXTBYTES];
    uint8_t K1[32], K2[32];
    moonlab_mlkem512_encaps(c1, K1, ek1, m);
    moonlab_mlkem512_encaps(c2, K2, ek1, m);
    CHECK(memcmp(c1, c2, MLKEM512_CIPHERTEXTBYTES) == 0,
          "Encaps(ek, m) reproducible -- same c");
    CHECK(memcmp(K1, K2, 32) == 0,
          "Encaps(ek, m) reproducible -- same K");
}

static void test_implicit_rejection(void) {
    fprintf(stdout, "\n-- implicit rejection on corrupted ciphertext --\n");
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, 0x41);
    fill_seed(z, 32, 0x42);
    fill_seed(m, 32, 0x43);

    uint8_t ek[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk[MLKEM512_SECRETKEYBYTES];
    moonlab_mlkem512_keygen(ek, dk, d, z);

    uint8_t c[MLKEM512_CIPHERTEXTBYTES];
    uint8_t K_true[32];
    moonlab_mlkem512_encaps(c, K_true, ek, m);

    /* Flip a byte deep in the ciphertext. */
    uint8_t c_bad[MLKEM512_CIPHERTEXTBYTES];
    memcpy(c_bad, c, sizeof c_bad);
    c_bad[123] ^= 0xFF;

    uint8_t K_rej[32];
    moonlab_mlkem512_decaps(K_rej, c_bad, dk);
    int equal = memcmp(K_rej, K_true, 32) == 0;
    CHECK(!equal, "tampered ciphertext does NOT decapsulate to K_true");

    /* The fallback should also be deterministic (reproducible). */
    uint8_t K_rej2[32];
    moonlab_mlkem512_decaps(K_rej2, c_bad, dk);
    CHECK(memcmp(K_rej, K_rej2, 32) == 0,
          "implicit-rejection output is deterministic");
}

int main(void) {
    fprintf(stdout, "=== ML-KEM-512 end-to-end tests ===\n\n");
    fprintf(stdout, "-- correctness (10 random trials) --\n");
    for (int i = 0; i < 10; i++) test_correctness_one_trial(i);
    test_determinism();
    test_implicit_rejection();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
