/**
 * @file test_mlkem_adversarial.c
 * @brief Negative-space / tamper harness for ML-KEM-512/768/1024 decaps.
 *
 * Complements the crypto lane's positive KATs by attacking the negative space:
 *
 *   - Systematic ciphertext corruption: single-bit flips across positions,
 *     trailing zero "truncation", all-zero, and all-0xFF ciphertexts.
 *   - Malformed public and secret keys (bit flips, all-zero, all-0xFF).
 *   - FIPS 203 implicit-rejection contract: a tampered ciphertext must yield a
 *     *deterministic* pseudo-random shared secret (SHAKE256(z || c)) that is
 *     distinct from the valid secret -- never a crash, never the valid key.
 *   - Shared-secret avalanche: flipping one bit of the encaps message seed
 *     flips ~half the shared-secret bits.
 *
 * The whole harness is deterministic (fixed keygen/encaps seeds, an internal
 * splitmix64 for test messages), so a failure is a real defect, not variance.
 *
 * MEMORY SAFETY: every ciphertext / public key / secret key handed to the
 * implementation is a *tight* heap allocation of exactly its declared size, so
 * when this target is built with -fsanitize=address any read or write past the
 * declared bound is caught by ASan red zones.  The fixed-size C ABI makes
 * "short/oversized" a content-malformation question, which we cover directly;
 * the length bound itself is enforced by the tight allocations + ASan.  Run
 * this target under ASan in CI (statistical.yml PR lane does).
 */

#include "stat_common.h"
#include "../../src/crypto/mlkem/mlkem.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*kem_keygen_fn)(uint8_t *, uint8_t *, const uint8_t *, const uint8_t *);
typedef void (*kem_encaps_fn)(uint8_t *, uint8_t *, const uint8_t *, const uint8_t *);
typedef void (*kem_decaps_fn)(uint8_t *, const uint8_t *, const uint8_t *);

typedef struct {
    const char *name;
    size_t pk, sk, ct, ss;
    kem_keygen_fn keygen;
    kem_encaps_fn encaps;
    kem_decaps_fn decaps;
} kem_set_t;

/* splitmix64: deterministic, independent of the RNG under test. */
static uint64_t sm_state = 0x9E3779B97F4A7C15ULL;
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void sm_fill(uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(sm_next() & 0xFF);
}

static size_t hamming(const uint8_t *a, const uint8_t *b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned x = (unsigned)(a[i] ^ b[i]);
        x = x - ((x >> 1) & 0x55u);
        x = (x & 0x33u) + ((x >> 2) & 0x33u);
        x = (x + (x >> 4)) & 0x0Fu;
        d += x;
    }
    return d;
}

/* A tight allocation of exactly n bytes (no slack for the impl to hide in). */
static uint8_t *tight(size_t n) {
    uint8_t *p = (uint8_t *)malloc(n ? n : 1);
    return p;
}

/* ------------------------------------------------------------------ */

typedef struct {
    long decaps_calls;        /* total tampered decaps performed         */
    long distinct_from_valid; /* tampered secret != valid secret         */
    long deterministic;       /* second decaps equals first (must be all)*/
    int  det_violation;       /* any non-deterministic tampered decaps   */
} fuzz_stats_t;

static void decaps_and_check(const kem_set_t *k, const uint8_t *ct,
                             const uint8_t *dk, const uint8_t *k_valid,
                             fuzz_stats_t *fs) {
    uint8_t *k1 = tight(k->ss);
    uint8_t *k2 = tight(k->ss);
    memset(k1, 0, k->ss);
    memset(k2, 0, k->ss);

    k->decaps(k1, ct, dk);
    k->decaps(k2, ct, dk);           /* determinism: same c,dk -> same K */

    fs->decaps_calls++;
    if (memcmp(k1, k2, k->ss) == 0) fs->deterministic++;
    else fs->det_violation = 1;
    if (memcmp(k1, k_valid, k->ss) != 0) fs->distinct_from_valid++;

    free(k1);
    free(k2);
}

typedef struct {
    int    fuzz_ok;
    int    avalanche_ok;
    double avalanche_frac;
    double distinct_frac;
} set_result_t;

static set_result_t run_set(const kem_set_t *k) {
    set_result_t res = { 1, 1, 0.0, 0.0 };
    printf("--- %s (pk=%zu sk=%zu ct=%zu) ---\n", k->name, k->pk, k->sk, k->ct);

    uint8_t d[32], z[32], m[32];
    memset(d, 0x11, sizeof(d));
    memset(z, 0x22, sizeof(z));
    memset(m, 0x33, sizeof(m));

    uint8_t *ek = tight(k->pk);
    uint8_t *dk = tight(k->sk);
    uint8_t *ct = tight(k->ct);
    uint8_t *k_valid = tight(k->ss);
    uint8_t *k_dec = tight(k->ss);

    k->keygen(ek, dk, d, z);
    k->encaps(ct, k_valid, ek, m);
    k->decaps(k_dec, ct, dk);

    if (memcmp(k_dec, k_valid, k->ss) != 0) {
        fprintf(stderr, "%s: valid decaps mismatch (harness/API sanity)\n", k->name);
        res.fuzz_ok = 0;
    }

    fuzz_stats_t fs;
    memset(&fs, 0, sizeof(fs));

    /* --- ciphertext single-bit flips (deterministic stride sample) --- */
    size_t ct_bits = k->ct * 8;
    size_t stride = ct_bits / 256; if (stride == 0) stride = 1;
    uint8_t *tc = tight(k->ct);
    for (size_t bit = 0; bit < ct_bits; bit += stride) {
        memcpy(tc, ct, k->ct);
        tc[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
        decaps_and_check(k, tc, dk, k_valid, &fs);
    }
    /* explicit edge bits */
    for (int e = 0; e < 3; e++) {
        size_t bit = (e == 0) ? 0 : (e == 1) ? (ct_bits / 2) : (ct_bits - 1);
        memcpy(tc, ct, k->ct);
        tc[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
        decaps_and_check(k, tc, dk, k_valid, &fs);
    }

    /* --- trailing-zero "truncation", all-zero, all-0xFF ciphertexts --- */
    memcpy(tc, ct, k->ct);
    memset(tc + k->ct / 2, 0x00, k->ct - k->ct / 2);  /* zero the tail */
    decaps_and_check(k, tc, dk, k_valid, &fs);

    memset(tc, 0x00, k->ct);
    decaps_and_check(k, tc, dk, k_valid, &fs);

    memset(tc, 0xFF, k->ct);
    decaps_and_check(k, tc, dk, k_valid, &fs);
    free(tc);

    /* --- malformed secret key: decaps must not crash / read OOB --- */
    uint8_t *bad_dk = tight(k->sk);
    for (int variant = 0; variant < 3; variant++) {
        memcpy(bad_dk, dk, k->sk);
        if (variant == 0) { for (size_t i = 0; i < k->sk; i += 7) bad_dk[i] ^= 0xA5; }
        else if (variant == 1) memset(bad_dk, 0x00, k->sk);
        else memset(bad_dk, 0xFF, k->sk);
        uint8_t *ktmp = tight(k->ss);
        k->decaps(ktmp, ct, bad_dk);   /* just must return without OOB */
        free(ktmp);
    }
    free(bad_dk);

    /* --- malformed public key: encaps must not crash / read OOB --- */
    uint8_t *bad_ek = tight(k->pk);
    for (int variant = 0; variant < 3; variant++) {
        memcpy(bad_ek, ek, k->pk);
        if (variant == 0) { for (size_t i = 0; i < k->pk; i += 5) bad_ek[i] ^= 0x5A; }
        else if (variant == 1) memset(bad_ek, 0x00, k->pk);
        else memset(bad_ek, 0xFF, k->pk);
        uint8_t *ctmp = tight(k->ct);
        uint8_t *ktmp = tight(k->ss);
        k->encaps(ctmp, ktmp, bad_ek, m);
        free(ctmp);
        free(ktmp);
    }
    free(bad_ek);

    /* --- shared-secret avalanche over encaps message-seed bit flips --- */
    const int trials = 8;
    const int flips_per_trial = 16;
    double frac_sum = 0.0;
    long frac_n = 0;
    uint8_t base_m[32], flip_m[32];
    uint8_t *k0 = tight(k->ss), *k1 = tight(k->ss);
    uint8_t *ctmp = tight(k->ct);
    for (int t = 0; t < trials; t++) {
        sm_fill(base_m, sizeof(base_m));
        k->encaps(ctmp, k0, ek, base_m);
        for (int f = 0; f < flips_per_trial; f++) {
            size_t bit = (size_t)(sm_next() % (sizeof(base_m) * 8));
            memcpy(flip_m, base_m, sizeof(base_m));
            flip_m[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
            k->encaps(ctmp, k1, ek, flip_m);
            size_t diff = hamming(k0, k1, k->ss);
            frac_sum += (double)diff / (double)(k->ss * 8);
            frac_n++;
        }
    }
    free(k0); free(k1); free(ctmp);
    double avalanche = frac_n ? frac_sum / (double)frac_n : 0.0;
    res.avalanche_frac = avalanche;

    double distinct_frac = fs.decaps_calls
        ? (double)fs.distinct_from_valid / (double)fs.decaps_calls : 0.0;
    res.distinct_frac = distinct_frac;

    printf("  tampered decaps=%ld distinct_from_valid=%ld (frac=%.5f) "
           "deterministic=%ld det_violation=%d\n",
           fs.decaps_calls, fs.distinct_from_valid, distinct_frac,
           fs.deterministic, fs.det_violation);
    printf("  shared-secret avalanche frac=%.5f (want ~0.5)\n", avalanche);

    if (fs.det_violation) { fprintf(stderr, "%s: implicit rejection not deterministic\n", k->name); res.fuzz_ok = 0; }
    if (distinct_frac < 0.999) { fprintf(stderr, "%s: tampered secret matched valid too often\n", k->name); res.fuzz_ok = 0; }
    if (!(avalanche > 0.40 && avalanche < 0.60)) { fprintf(stderr, "%s: avalanche out of band\n", k->name); res.avalanche_ok = 0; }

    free(ek); free(dk); free(ct); free(k_valid); free(k_dec);
    return res;
}

int main(void) {
    const kem_set_t sets[3] = {
        { "ML-KEM-512", MLKEM512_PUBLICKEYBYTES, MLKEM512_SECRETKEYBYTES,
          MLKEM512_CIPHERTEXTBYTES, 32,
          (kem_keygen_fn)moonlab_mlkem512_keygen,
          (kem_encaps_fn)moonlab_mlkem512_encaps,
          (kem_decaps_fn)moonlab_mlkem512_decaps },
        { "ML-KEM-768", MLKEM768_PUBLICKEYBYTES, MLKEM768_SECRETKEYBYTES,
          MLKEM768_CIPHERTEXTBYTES, 32,
          (kem_keygen_fn)moonlab_mlkem768_keygen,
          (kem_encaps_fn)moonlab_mlkem768_encaps,
          (kem_decaps_fn)moonlab_mlkem768_decaps },
        { "ML-KEM-1024", MLKEM1024_PUBLICKEYBYTES, MLKEM1024_SECRETKEYBYTES,
          MLKEM1024_CIPHERTEXTBYTES, 32,
          (kem_keygen_fn)moonlab_mlkem1024_keygen,
          (kem_encaps_fn)moonlab_mlkem1024_encaps,
          (kem_decaps_fn)moonlab_mlkem1024_decaps },
    };

    printf("ML-KEM adversarial / tamper harness\n");

    int fuzz_ok = 1;
    int avalanche_ok = 1;
    double min_avalanche = 1.0;
    double min_distinct = 1.0;
    double aval[3] = {0, 0, 0};

    for (int i = 0; i < 3; i++) {
        set_result_t r = run_set(&sets[i]);
        aval[i] = r.avalanche_frac;
        if (r.avalanche_frac < min_avalanche) min_avalanche = r.avalanche_frac;
        if (r.distinct_frac < min_distinct) min_distinct = r.distinct_frac;
        if (!r.fuzz_ok) fuzz_ok = 0;
        if (!r.avalanche_ok) avalanche_ok = 0;
    }

    char stats[512];
    snprintf(stats, sizeof(stats),
        "{\"param_sets\":3,\"distinct_frac_min\":%.5f,"
        "\"determinism\":\"enforced\",\"asan\":\"tight-alloc\"}",
        min_distinct);
    stat_emit_result("mlkem_negative_fuzz", fuzz_ok, 1, stats);

    snprintf(stats, sizeof(stats),
        "{\"avalanche_512\":%.5f,\"avalanche_768\":%.5f,"
        "\"avalanche_1024\":%.5f,\"min\":%.5f,\"band\":\"0.40-0.60\"}",
        aval[0], aval[1], aval[2], min_avalanche);
    stat_emit_result("mlkem_avalanche", avalanche_ok, 1, stats);

    if (!fuzz_ok) fprintf(stderr, "FAIL: ML-KEM negative fuzz\n");
    if (!avalanche_ok) fprintf(stderr, "FAIL: ML-KEM avalanche\n");
    return (fuzz_ok && avalanche_ok) ? 0 : 1;
}
