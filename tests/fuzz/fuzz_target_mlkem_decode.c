/**
 * @file    fuzz_target_mlkem_decode.c
 * @brief   Surface: ML-KEM (FIPS 203) decode paths + CTR_DRBG seed input.
 *
 * The security-critical contract for ML-KEM decapsulation is that a
 * malformed ciphertext must never abort, read out of bounds, or leak the
 * decode failure through anything other than the pseudorandom implicit-
 * rejection secret (Fujisaki-Okamoto).  This target feeds fully attacker-
 * controlled bytes into:
 *
 *   - `moonlab_mlkem{512,768,1024}_decaps` -- garbage ciphertext + garbage
 *     secret key, all three parameter sets.  This is the primary decode
 *     surface: an attacker chooses the ciphertext.
 *   - `moonlab_mlkem{512,768,1024}_encaps` -- garbage public key (the
 *     encapsulator ingests a peer-supplied ek).
 *   - `moonlab_mlkem{512,768,1024}_keygen` -- adversarial (d, z) seeds.
 *   - `ctr_drbg_init` + `ctr_drbg_generate` -- the 48-byte seed input path
 *     of the SP 800-90A CTR_DRBG that backs the KAT harness.
 *
 * The three parameter sets have different encoded sizes, so the target
 * always presents correctly-sized buffers (zero-padded from the fuzz
 * input) -- the fuzzed quantity is the *content*, matching the real
 * threat model where the buffer lengths are fixed by the wire format.
 */

#include "fuzz_common.h"

#include "crypto/mlkem/mlkem.h"
#include "crypto/mlkem/params.h"
#include "crypto/drbg/ctr_drbg.h"

/* Largest encoded object across the three parameter sets, so one stack
 * buffer covers every case. */
#define MLKEM_MAX_SK  MLKEM1024_SECRETKEYBYTES  /* 3168 */
#define MLKEM_MAX_PK  MLKEM1024_PUBLICKEYBYTES  /* 1568 */
#define MLKEM_MAX_CT  MLKEM1024_CIPHERTEXTBYTES /* 1568 */

static void run_512(const uint8_t **p, const uint8_t *end, int op)
{
    uint8_t K[32];
    if (op == 0) {
        uint8_t c[MLKEM512_CIPHERTEXTBYTES];
        uint8_t dk[MLKEM512_SECRETKEYBYTES];
        fuzz_fill(p, end, c, sizeof(c));
        fuzz_fill(p, end, dk, sizeof(dk));
        moonlab_mlkem512_decaps(K, c, dk);
    } else if (op == 1) {
        uint8_t ek[MLKEM512_PUBLICKEYBYTES];
        uint8_t m[32], c[MLKEM512_CIPHERTEXTBYTES];
        fuzz_fill(p, end, ek, sizeof(ek));
        fuzz_fill(p, end, m, sizeof(m));
        moonlab_mlkem512_encaps(c, K, ek, m);
    } else {
        uint8_t d[32], z[32];
        uint8_t ek[MLKEM512_PUBLICKEYBYTES], dk[MLKEM512_SECRETKEYBYTES];
        fuzz_fill(p, end, d, sizeof(d));
        fuzz_fill(p, end, z, sizeof(z));
        moonlab_mlkem512_keygen(ek, dk, d, z);
    }
}

static void run_768(const uint8_t **p, const uint8_t *end, int op)
{
    uint8_t K[32];
    if (op == 0) {
        uint8_t c[MLKEM768_CIPHERTEXTBYTES];
        uint8_t dk[MLKEM768_SECRETKEYBYTES];
        fuzz_fill(p, end, c, sizeof(c));
        fuzz_fill(p, end, dk, sizeof(dk));
        moonlab_mlkem768_decaps(K, c, dk);
    } else if (op == 1) {
        uint8_t ek[MLKEM768_PUBLICKEYBYTES];
        uint8_t m[32], c[MLKEM768_CIPHERTEXTBYTES];
        fuzz_fill(p, end, ek, sizeof(ek));
        fuzz_fill(p, end, m, sizeof(m));
        moonlab_mlkem768_encaps(c, K, ek, m);
    } else {
        uint8_t d[32], z[32];
        uint8_t ek[MLKEM768_PUBLICKEYBYTES], dk[MLKEM768_SECRETKEYBYTES];
        fuzz_fill(p, end, d, sizeof(d));
        fuzz_fill(p, end, z, sizeof(z));
        moonlab_mlkem768_keygen(ek, dk, d, z);
    }
}

static void run_1024(const uint8_t **p, const uint8_t *end, int op)
{
    uint8_t K[32];
    if (op == 0) {
        uint8_t c[MLKEM1024_CIPHERTEXTBYTES];
        uint8_t dk[MLKEM1024_SECRETKEYBYTES];
        fuzz_fill(p, end, c, sizeof(c));
        fuzz_fill(p, end, dk, sizeof(dk));
        moonlab_mlkem1024_decaps(K, c, dk);
    } else if (op == 1) {
        uint8_t ek[MLKEM1024_PUBLICKEYBYTES];
        uint8_t m[32], c[MLKEM1024_CIPHERTEXTBYTES];
        fuzz_fill(p, end, ek, sizeof(ek));
        fuzz_fill(p, end, m, sizeof(m));
        moonlab_mlkem1024_encaps(c, K, ek, m);
    } else {
        uint8_t d[32], z[32];
        uint8_t ek[MLKEM1024_PUBLICKEYBYTES], dk[MLKEM1024_SECRETKEYBYTES];
        fuzz_fill(p, end, d, sizeof(d));
        fuzz_fill(p, end, z, sizeof(z));
        moonlab_mlkem1024_keygen(ek, dk, d, z);
    }
}

static void run_drbg(const uint8_t **p, const uint8_t *end)
{
    ctr_drbg_ctx_t ctx;
    uint8_t seed[48];
    fuzz_fill(p, end, seed, sizeof(seed));
    ctr_drbg_init(&ctx, seed);

    /* Draw a bounded, fuzz-chosen amount so the output-length loop and
     * its reseed-counter bump are exercised, including the len==0 and
     * non-multiple-of-16 tail cases. */
    uint32_t want = fuzz_u32(p, end) % 1025u; /* 0..1024 bytes */
    uint8_t out[1024];
    ctr_drbg_generate(&ctx, out, want);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    const uint8_t *p   = data;
    const uint8_t *end = data + size;

    const uint8_t set = fuzz_u8(&p, end) % 4u; /* 0=512 1=768 2=1024 3=drbg */
    const int     op  = (int)(fuzz_u8(&p, end) % 3u); /* decaps/encaps/keygen */

    switch (set) {
    case 0: run_512(&p, end, op);  break;
    case 1: run_768(&p, end, op);  break;
    case 2: run_1024(&p, end, op); break;
    default: run_drbg(&p, end);    break;
    }
    return 0;
}
