/**
 * @file pqc_qrng_demo.c
 * @brief End-to-end demonstration of the Moonlab quantum-entropy +
 *        post-quantum cryptography pipeline.
 *
 * This example shows, in about 100 lines of plain C, the full
 * "quantum-source to quantum-safe" story:
 *
 *   1. Alice and Bob each want a quantum-safe shared secret.
 *   2. Alice generates an ML-KEM-512 key pair using the Bell-verified
 *      quantum RNG as entropy.
 *   3. Bob receives Alice's public key and encapsulates a shared
 *      secret against it, again drawing entropy from the quantum RNG.
 *   4. Alice decapsulates the ciphertext to recover the same secret.
 *
 * The output prints the public/secret/ciphertext/shared-secret sizes
 * and verifies that both parties agreed on the same 32-byte key.
 *
 * Build (on top of libquantumsim):
 *     cc examples/applications/pqc_qrng_demo.c -lquantumsim -o pqc_qrng_demo
 * Run:
 *     ./pqc_qrng_demo
 */

#include "../../src/applications/moonlab_export.h"
#include "../../src/crypto/sha3/sha3.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

static void hex_print(const char *label, const uint8_t *buf, size_t n) {
    printf("  %s (%zu bytes): ", label, n);
    size_t show = n > 16 ? 16 : n;
    for (size_t i = 0; i < show; i++) printf("%02x", buf[i]);
    if (show < n) printf("...");
    printf("\n");
}

int main(void) {
    printf("=== Moonlab quantum-RNG + ML-KEM-512 end-to-end demo ===\n\n");

    /* --- Verify the stable ABI is live. --- */
    int abi_major = 0, abi_minor = 0, abi_patch = 0;
    moonlab_abi_version(&abi_major, &abi_minor, &abi_patch);
    printf("libquantumsim ABI version: %d.%d.%d\n\n",
           abi_major, abi_minor, abi_patch);

    /* --- 1. Alice generates a key pair with Bell-verified QRNG. --- */
    printf("[1] Alice: KeyGen with moonlab_qrng_bytes-sourced entropy\n");
    uint8_t ek[MOONLAB_MLKEM512_PUBLICKEYBYTES];
    uint8_t dk[MOONLAB_MLKEM512_SECRETKEYBYTES];
    if (moonlab_mlkem512_keygen_qrng(ek, dk) != 0) {
        fprintf(stderr, "KeyGen failed -- QRNG unavailable?\n");
        return 1;
    }
    hex_print("public key (ek)",  ek, sizeof ek);
    hex_print("secret key (dk)",  dk, sizeof dk);
    printf("\n");

    /* --- 2. Bob encapsulates a shared secret against ek. --- */
    printf("[2] Bob: Encaps(ek) to produce a ciphertext + shared secret\n");
    uint8_t ct[MOONLAB_MLKEM512_CIPHERTEXTBYTES];
    uint8_t K_bob[MOONLAB_MLKEM512_SHAREDSECRETBYTES];
    if (moonlab_mlkem512_encaps_qrng(ct, K_bob, ek) != 0) {
        fprintf(stderr, "Encaps failed\n");
        return 1;
    }
    hex_print("ciphertext",    ct,    sizeof ct);
    hex_print("K_Bob (shared)", K_bob, sizeof K_bob);
    printf("\n");

    /* --- 3. Alice decapsulates to recover the same key. --- */
    printf("[3] Alice: Decaps(ct, dk) to recover the shared secret\n");
    uint8_t K_alice[MOONLAB_MLKEM512_SHAREDSECRETBYTES];
    moonlab_mlkem512_decaps(K_alice, ct, dk);
    hex_print("K_Alice (shared)", K_alice, sizeof K_alice);
    printf("\n");

    int match = memcmp(K_alice, K_bob, sizeof K_alice) == 0;
    printf("Shared-secret agreement: %s\n", match ? "YES" : "NO");

    /* --- 4. Demonstrate that the shared key can immediately feed
     *       SHAKE to produce a long keystream (e.g. for a session
     *       key). --- */
    printf("\n[4] Derive an 8 KiB keystream from K via SHAKE256\n");
    uint8_t stream[64];
    shake256(K_alice, sizeof K_alice, stream, sizeof stream);
    hex_print("first 64 B of stream", stream, sizeof stream);

    /* --- 5. Tamper detection: flip a ciphertext byte and re-decaps. --- */
    printf("\n[5] Flip one ciphertext byte; Decaps should NOT return K_Bob\n");
    uint8_t ct_bad[MOONLAB_MLKEM512_CIPHERTEXTBYTES];
    memcpy(ct_bad, ct, sizeof ct_bad);
    ct_bad[77] ^= 0xA5;
    uint8_t K_rej[MOONLAB_MLKEM512_SHAREDSECRETBYTES];
    moonlab_mlkem512_decaps(K_rej, ct_bad, dk);
    int rejected = memcmp(K_rej, K_bob, sizeof K_bob) != 0;
    printf("  implicit rejection engaged (K differs): %s\n",
           rejected ? "YES" : "NO");

    printf("\n=== demo complete ===\n");
    return (match && rejected) ? 0 : 1;
}
