/**
 * @file test_sha3.c
 * @brief FIPS 202 known-answer tests for SHA3 + SHAKE.
 *
 * Every vector below is taken directly from the NIST KAT files
 * (SHA3-256 / SHA3-512 / SHAKE128 / SHAKE256 short-message tests).
 */

#include "../../src/crypto/sha3/sha3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void hex_decode(const char *hex, uint8_t *out, size_t outlen) {
    for (size_t i = 0; i < outlen; i++) {
        unsigned v;
        sscanf(hex + 2 * i, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

static int bytes_equal(const uint8_t *a, const uint8_t *b, size_t n) {
    return memcmp(a, b, n) == 0;
}

static void check_kat(const char *label, const uint8_t *in, size_t inlen,
                       void (*hash)(const uint8_t*, size_t, uint8_t*),
                       const char *expected_hex, size_t outlen) {
    uint8_t *out = malloc(outlen);
    uint8_t *expected = malloc(outlen);
    hash(in, inlen, out);
    hex_decode(expected_hex, expected, outlen);
    int ok = bytes_equal(out, expected, outlen);
    if (ok) {
        fprintf(stdout, "  OK    %s\n", label);
    } else {
        fprintf(stderr, "  FAIL  %s\n", label);
        fprintf(stderr, "    got     : ");
        for (size_t i = 0; i < outlen; i++) fprintf(stderr, "%02x", out[i]);
        fprintf(stderr, "\n    expected: %s\n", expected_hex);
        failures++;
    }
    free(out); free(expected);
}

static void sha3_256_wrap(const uint8_t *in, size_t inlen, uint8_t *out) {
    sha3_256(in, inlen, out);
}
static void sha3_512_wrap(const uint8_t *in, size_t inlen, uint8_t *out) {
    sha3_512(in, inlen, out);
}

static void check_shake(const char *label, const uint8_t *in, size_t inlen,
                         int is_256, const char *expected_hex, size_t outlen) {
    uint8_t *out = malloc(outlen);
    uint8_t *expected = malloc(outlen);
    if (is_256) shake256(in, inlen, out, outlen);
    else        shake128(in, inlen, out, outlen);
    hex_decode(expected_hex, expected, outlen);
    int ok = bytes_equal(out, expected, outlen);
    if (ok) {
        fprintf(stdout, "  OK    %s\n", label);
    } else {
        fprintf(stderr, "  FAIL  %s\n", label);
        fprintf(stderr, "    got     : ");
        for (size_t i = 0; i < outlen; i++) fprintf(stderr, "%02x", out[i]);
        fprintf(stderr, "\n    expected: %s\n", expected_hex);
        failures++;
    }
    free(out); free(expected);
}

int main(void) {
    fprintf(stdout, "=== FIPS 202 known-answer tests ===\n\n");

    /* --- SHA3-256 --- */
    fprintf(stdout, "-- SHA3-256 --\n");
    check_kat("SHA3-256(\"\")", (const uint8_t*)"", 0, sha3_256_wrap,
              "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
              32);
    check_kat("SHA3-256(\"abc\")", (const uint8_t*)"abc", 3, sha3_256_wrap,
              "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532",
              32);
    /* NIST ShortMsgKAT_SHA3-256 vector Len = 2440 bits (305 bytes of 0xa3). */
    {
        uint8_t msg[200];
        memset(msg, 0xa3, sizeof(msg));
        check_kat("SHA3-256(200 * 0xa3)", msg, sizeof(msg), sha3_256_wrap,
                  "79f38adec5c20307a98ef76e8324afbfd46cfd81b22e3973c65fa1bd9de31787",
                  32);
    }

    /* --- SHA3-512 --- */
    fprintf(stdout, "\n-- SHA3-512 --\n");
    check_kat("SHA3-512(\"\")", (const uint8_t*)"", 0, sha3_512_wrap,
              "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc147"
              "5c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e30175"
              "8586281dcd26", 64);
    check_kat("SHA3-512(\"abc\")", (const uint8_t*)"abc", 3, sha3_512_wrap,
              "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d"
              "2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8"
              "274eec53f0", 64);

    /* --- SHAKE128 --- */
    fprintf(stdout, "\n-- SHAKE128 --\n");
    check_shake("SHAKE128(\"\", 32)", (const uint8_t*)"", 0, 0,
                "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26",
                32);
    /* SHAKE squeeze continuity: SHAKE128 produces the same bytes when
     * squeezed in one chunk or two chunks. */
    {
        sha3_ctx_t a; shake128_init(&a);
        sha3_update(&a, (const uint8_t*)"abc", 3);
        uint8_t one[64]; shake_squeeze(&a, one, 64);

        sha3_ctx_t b; shake128_init(&b);
        sha3_update(&b, (const uint8_t*)"abc", 3);
        uint8_t two[64]; shake_squeeze(&b, two, 40); shake_squeeze(&b, two + 40, 24);

        int ok = bytes_equal(one, two, 64);
        if (ok) fprintf(stdout, "  OK    SHAKE128 split-squeeze matches single-squeeze\n");
        else { fprintf(stderr, "  FAIL  SHAKE128 split-squeeze\n"); failures++; }
    }

    /* --- SHAKE256 --- */
    fprintf(stdout, "\n-- SHAKE256 --\n");
    check_shake("SHAKE256(\"\", 32)", (const uint8_t*)"", 0, 1,
                "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f",
                32);

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
