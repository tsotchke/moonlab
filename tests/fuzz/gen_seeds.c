/**
 * @file    gen_seeds.c
 * @brief   Generate the binary seed corpus entries that must be *real*.
 *
 * Most seed files in `tests/fuzz/corpora/` are hand-written text (circuit
 * frames, config JSON, control-plane verbs) and are checked in directly.
 * The ML-KEM decode target, however, benefits from a genuine
 * (ciphertext, secret-key) pair so the fuzzer starts from a corpus that
 * actually reaches the FO re-encryption / equality branch of decaps.
 *
 * This helper is NOT part of the CMake build.  It is a standalone
 * reproducer, compiled on demand against the crypto sources:
 *
 *   clang -I src \
 *     tests/fuzz/gen_seeds.c \
 *     src/crypto/mlkem/mlkem.c src/crypto/mlkem/poly.c src/crypto/sha3/sha3.c \
 *     -o /tmp/gen_seeds && /tmp/gen_seeds tests/fuzz/corpora
 *
 * The resulting *.bin files are committed so the corpus is reproducible
 * without a build step.  The layout matches fuzz_target_mlkem_decode.c:
 * byte 0 = parameter set (0/1/2), byte 1 = op (0=decaps), then the
 * correctly-sized ciphertext followed by the secret key.
 */

#include "crypto/mlkem/mlkem.h"
#include "crypto/mlkem/params.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The deterministic ML-KEM entry points do not touch the QRNG, but the
 * _qrng convenience wrappers in mlkem.c reference this symbol, so provide
 * a stub to satisfy the linker for this standalone tool. */
int moonlab_qrng_bytes(unsigned char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) b[i] = (unsigned char)(i * 7u + 1u);
    return 0;
}

static void fill_pattern(uint8_t *b, size_t n, uint8_t base)
{
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(base + i * 3u);
}

static void write_file(const char *dir, const char *name,
                       const uint8_t *data, size_t n)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/mlkem_decode_fuzz/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(data, 1, n, f);
    fclose(f);
    printf("wrote %s (%zu bytes)\n", path, n);
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "tests/fuzz/corpora";
    uint8_t d[32], z[32], m[32];
    fill_pattern(d, sizeof(d), 0x11);
    fill_pattern(z, sizeof(z), 0x22);
    fill_pattern(m, sizeof(m), 0x33);

    uint8_t out[2 + MLKEM1024_CIPHERTEXTBYTES + MLKEM1024_SECRETKEYBYTES];

    /* ML-KEM-512 valid decaps seed. */
    {
        uint8_t ek[MLKEM512_PUBLICKEYBYTES], dk[MLKEM512_SECRETKEYBYTES];
        uint8_t c[MLKEM512_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem512_keygen(ek, dk, d, z);
        moonlab_mlkem512_encaps(c, K, ek, m);
        size_t off = 0;
        out[off++] = 0; out[off++] = 0;
        memcpy(out + off, c, sizeof(c)); off += sizeof(c);
        memcpy(out + off, dk, sizeof(dk)); off += sizeof(dk);
        write_file(dir, "valid_decaps_512.bin", out, off);
    }

    /* ML-KEM-768 valid decaps seed. */
    {
        uint8_t ek[MLKEM768_PUBLICKEYBYTES], dk[MLKEM768_SECRETKEYBYTES];
        uint8_t c[MLKEM768_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem768_keygen(ek, dk, d, z);
        moonlab_mlkem768_encaps(c, K, ek, m);
        size_t off = 0;
        out[off++] = 1; out[off++] = 0;
        memcpy(out + off, c, sizeof(c)); off += sizeof(c);
        memcpy(out + off, dk, sizeof(dk)); off += sizeof(dk);
        write_file(dir, "valid_decaps_768.bin", out, off);
    }

    /* ML-KEM-1024 valid decaps seed. */
    {
        uint8_t ek[MLKEM1024_PUBLICKEYBYTES], dk[MLKEM1024_SECRETKEYBYTES];
        uint8_t c[MLKEM1024_CIPHERTEXTBYTES], K[32];
        moonlab_mlkem1024_keygen(ek, dk, d, z);
        moonlab_mlkem1024_encaps(c, K, ek, m);
        size_t off = 0;
        out[off++] = 2; out[off++] = 0;
        memcpy(out + off, c, sizeof(c)); off += sizeof(c);
        memcpy(out + off, dk, sizeof(dk)); off += sizeof(dk);
        write_file(dir, "valid_decaps_1024.bin", out, off);
    }

    return 0;
}
