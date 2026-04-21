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
#include "../../src/crypto/sha3/sha3.h"

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

static void test_qrng_wrapper(void) {
    fprintf(stdout, "\n-- keygen_qrng + encaps_qrng round-trip --\n");
    uint8_t ek[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk[MLKEM512_SECRETKEYBYTES];
    int rc = moonlab_mlkem512_keygen_qrng(ek, dk);
    CHECK(rc == 0, "keygen_qrng succeeds");

    uint8_t c[MLKEM512_CIPHERTEXTBYTES];
    uint8_t K[32], K_dec[32];
    rc = moonlab_mlkem512_encaps_qrng(c, K, ek);
    CHECK(rc == 0, "encaps_qrng succeeds");

    moonlab_mlkem512_decaps(K_dec, c, dk);
    CHECK(memcmp(K, K_dec, 32) == 0,
          "QRNG-sourced KeyGen + Encaps + Decaps round-trip");

    /* QRNG path is non-deterministic: two successive keygens should
     * produce different ek (with overwhelming probability). */
    uint8_t ek2[MLKEM512_PUBLICKEYBYTES];
    uint8_t dk2[MLKEM512_SECRETKEYBYTES];
    moonlab_mlkem512_keygen_qrng(ek2, dk2);
    CHECK(memcmp(ek, ek2, MLKEM512_PUBLICKEYBYTES) != 0,
          "QRNG-sourced keygens produce distinct key pairs");
}

static void test_mlkem768_roundtrip(void) {
    fprintf(stdout, "\n-- ML-KEM-768 KeyGen + Encaps + Decaps --\n");
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, 0x51);
    fill_seed(z, 32, 0x52);
    fill_seed(m, 32, 0x53);
    uint8_t ek[MLKEM768_PUBLICKEYBYTES];
    uint8_t dk[MLKEM768_SECRETKEYBYTES];
    uint8_t c [MLKEM768_CIPHERTEXTBYTES];
    uint8_t K[32], K_dec[32];
    moonlab_mlkem768_keygen(ek, dk, d, z);
    moonlab_mlkem768_encaps(c, K, ek, m);
    moonlab_mlkem768_decaps(K_dec, c, dk);
    CHECK(memcmp(K, K_dec, 32) == 0, "ML-KEM-768 round-trip");

    /* Tamper detection. */
    uint8_t c_bad[MLKEM768_CIPHERTEXTBYTES];
    memcpy(c_bad, c, sizeof c_bad);
    c_bad[200] ^= 0x55;
    uint8_t K_rej[32];
    moonlab_mlkem768_decaps(K_rej, c_bad, dk);
    CHECK(memcmp(K_rej, K, 32) != 0,
          "ML-KEM-768 tampered ciphertext -> different K");
}

static void test_mlkem1024_roundtrip(void) {
    fprintf(stdout, "\n-- ML-KEM-1024 KeyGen + Encaps + Decaps --\n");
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, 0x61);
    fill_seed(z, 32, 0x62);
    fill_seed(m, 32, 0x63);
    uint8_t ek[MLKEM1024_PUBLICKEYBYTES];
    uint8_t dk[MLKEM1024_SECRETKEYBYTES];
    uint8_t c [MLKEM1024_CIPHERTEXTBYTES];
    uint8_t K[32], K_dec[32];
    moonlab_mlkem1024_keygen(ek, dk, d, z);
    moonlab_mlkem1024_encaps(c, K, ek, m);
    moonlab_mlkem1024_decaps(K_dec, c, dk);
    CHECK(memcmp(K, K_dec, 32) == 0, "ML-KEM-1024 round-trip");
}

/* Regression KAT: every ML-KEM primitive with fixed (d, z, m) produces
 * exactly the same output bytes forever.  Fingerprint with SHA3-256
 * so the test is compact but byte-level strict.
 *
 * These hashes were captured on 2026-04-21 from the reference
 * implementation that ships in this tree.  They validate *regression*
 * (the algebra doesn't silently drift) rather than *NIST conformance*
 * (which requires AES-256-CTR DRBG seed expansion, deferred to 0.2.1).
 */
static int expect_sha256(const char *label,
                          const uint8_t *buf, size_t n,
                          const char *hex_expected) {
    uint8_t h[32];
    sha3_256(buf, n, h);
    char hex_got[65];
    for (int i = 0; i < 32; i++) snprintf(hex_got + 2 * i, 3, "%02x", h[i]);
    if (memcmp(hex_got, hex_expected, 64) != 0) {
        fprintf(stderr, "  FAIL  %s\n", label);
        fprintf(stderr, "    got      %s\n", hex_got);
        fprintf(stderr, "    expected %s\n", hex_expected);
        return 0;
    }
    fprintf(stdout, "  OK    %s\n", label);
    return 1;
}

static void test_regression_kat(void) {
    fprintf(stdout, "\n-- ML-KEM regression KAT (fixed seeds -> hashed outputs) --\n");
    uint8_t d[32], z[32], m[32];
    fill_seed(d, 32, 0xA1); fill_seed(z, 32, 0xB2); fill_seed(m, 32, 0xC3);

    {
        uint8_t ek[MLKEM512_PUBLICKEYBYTES], dk[MLKEM512_SECRETKEYBYTES];
        uint8_t c [MLKEM512_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem512_keygen(ek, dk, d, z);
        moonlab_mlkem512_encaps(c, K, ek, m);
        failures += !expect_sha256("ML-KEM-512 ek", ek, sizeof ek,
            "572c0f80280465f39e6cc8abc1e612b3bc10e978ca40de6645f92dea9f1c7551");
        failures += !expect_sha256("ML-KEM-512 dk", dk, sizeof dk,
            "077bab78f0e0f77aa438ed5fc844009f0387f3ed3e51d69248fdb4bedba0b757");
        failures += !expect_sha256("ML-KEM-512 ct", c, sizeof c,
            "9c20736f46be9f29ce98e11ce7b7b251b718b22f711d86f8119361aa22537320");
        failures += !expect_sha256("ML-KEM-512 K",  K, sizeof K,
            "e0362ae478e9dc79c571d56c3f586ee567b2516fb803b17552fc5e9f557b0d5c");
    }
    {
        uint8_t ek[MLKEM768_PUBLICKEYBYTES], dk[MLKEM768_SECRETKEYBYTES];
        uint8_t c [MLKEM768_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem768_keygen(ek, dk, d, z);
        moonlab_mlkem768_encaps(c, K, ek, m);
        failures += !expect_sha256("ML-KEM-768 ek", ek, sizeof ek,
            "bf1a854e976c8b5e9dc159bd2e0ec21f662ff2c319929f56cdaca56e1c09f513");
        failures += !expect_sha256("ML-KEM-768 dk", dk, sizeof dk,
            "782f9bccc6179cb1907e5b3aa12b6a0038cba282d8c6227d75ff83940fd6fafc");
        failures += !expect_sha256("ML-KEM-768 ct", c, sizeof c,
            "21ae64af0f0c76749463ec0c890cd69f48ccc279747d69a19bf6f8248bd16f0e");
        failures += !expect_sha256("ML-KEM-768 K",  K, sizeof K,
            "49a9b53e69d7bd5f0eb5c71e25edb0b3600dc9947d6da93b7cf9623cd1e02b2f");
    }
    {
        uint8_t ek[MLKEM1024_PUBLICKEYBYTES], dk[MLKEM1024_SECRETKEYBYTES];
        uint8_t c [MLKEM1024_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem1024_keygen(ek, dk, d, z);
        moonlab_mlkem1024_encaps(c, K, ek, m);
        failures += !expect_sha256("ML-KEM-1024 ek", ek, sizeof ek,
            "d6b13ad6298f5598b650426afc2f2de2e8a090484fc9e214395ac9e6979c27d2");
        failures += !expect_sha256("ML-KEM-1024 dk", dk, sizeof dk,
            "398ed401acde75aa3119c32f79b77a27fb502deb1d6c516f70feb110d6d15c69");
        failures += !expect_sha256("ML-KEM-1024 ct", c, sizeof c,
            "cfb11fac6a4e2080713987d51b353135614d0d76a00437b7e1ccaf29b9d89326");
        failures += !expect_sha256("ML-KEM-1024 K",  K, sizeof K,
            "c8cd19682387db61683742c36a9993f05d284105f251da064f0acfaff04a24a9");
    }
}

static void test_mlkem_qrng_all_three(void) {
    fprintf(stdout, "\n-- QRNG-sourced keygen/encaps round-trips for 512/768/1024 --\n");
    {
        uint8_t ek[MLKEM512_PUBLICKEYBYTES], dk[MLKEM512_SECRETKEYBYTES];
        uint8_t c [MLKEM512_CIPHERTEXTBYTES], K[32], K_dec[32];
        CHECK(moonlab_mlkem512_keygen_qrng(ek, dk) == 0, "512 keygen_qrng");
        CHECK(moonlab_mlkem512_encaps_qrng(c, K, ek) == 0, "512 encaps_qrng");
        moonlab_mlkem512_decaps(K_dec, c, dk);
        CHECK(memcmp(K, K_dec, 32) == 0, "512 QRNG round-trip");
    }
    {
        uint8_t ek[MLKEM768_PUBLICKEYBYTES], dk[MLKEM768_SECRETKEYBYTES];
        uint8_t c [MLKEM768_CIPHERTEXTBYTES], K[32], K_dec[32];
        CHECK(moonlab_mlkem768_keygen_qrng(ek, dk) == 0, "768 keygen_qrng");
        CHECK(moonlab_mlkem768_encaps_qrng(c, K, ek) == 0, "768 encaps_qrng");
        moonlab_mlkem768_decaps(K_dec, c, dk);
        CHECK(memcmp(K, K_dec, 32) == 0, "768 QRNG round-trip");
    }
    {
        uint8_t ek[MLKEM1024_PUBLICKEYBYTES], dk[MLKEM1024_SECRETKEYBYTES];
        uint8_t c [MLKEM1024_CIPHERTEXTBYTES], K[32], K_dec[32];
        CHECK(moonlab_mlkem1024_keygen_qrng(ek, dk) == 0, "1024 keygen_qrng");
        CHECK(moonlab_mlkem1024_encaps_qrng(c, K, ek) == 0, "1024 encaps_qrng");
        moonlab_mlkem1024_decaps(K_dec, c, dk);
        CHECK(memcmp(K, K_dec, 32) == 0, "1024 QRNG round-trip");
    }
}

int main(void) {
    fprintf(stdout, "=== ML-KEM end-to-end tests ===\n\n");
    fprintf(stdout, "-- ML-KEM-512 correctness (10 random trials) --\n");
    for (int i = 0; i < 10; i++) test_correctness_one_trial(i);
    test_determinism();
    test_implicit_rejection();
    test_qrng_wrapper();
    test_mlkem768_roundtrip();
    test_mlkem1024_roundtrip();
    test_regression_kat();
    test_mlkem_qrng_all_three();

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
