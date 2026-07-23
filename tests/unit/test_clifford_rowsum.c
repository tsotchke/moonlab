/**
 * @file test_clifford_rowsum.c
 * @brief Exhaustive verification of the word-parallel rowsum phase formula.
 *
 * The bit-packed Clifford backend replaces the scalar Aaronson-Gottesman
 * g-function accumulation with a word-parallel popcount formula.  This test
 * proves the formula matches the scalar g exactly:
 *   1. over all 16 single-bit (x1,z1,x2,z2) combinations, and
 *   2. over random 64-bit word vectors (lane-sum mod 4).
 *
 * The formula under test is duplicated here from clifford.c so the test is a
 * standalone oracle (no reliance on the code it validates).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

/* Scalar reference: AG g function, signed in {-1,0,1}. */
static int g_scalar(int x1, int z1, int x2, int z2) {
    if (x1 == 0 && z1 == 0) return 0;
    if (x1 == 1 && z1 == 1) return z2 - x2;                 /* Y */
    if (x1 == 1 && z1 == 0) return z2 * (2 * x2 - 1);       /* X */
    return x2 * (1 - 2 * z2);                               /* Z */
}

/* Word-parallel formula (identical to clifford.c). */
static int g_words(const uint64_t* x1, const uint64_t* z1,
                   const uint64_t* x2, const uint64_t* z2, size_t ws) {
    int acc = 0;
    for (size_t k = 0; k < ws; k++) {
        uint64_t X1 = x1[k], Z1 = z1[k], X2 = x2[k], Z2 = z2[k];
        uint64_t pos = (~X1 &  Z1 &  X2 & ~Z2)
                     | ( X1 & ~Z1 &  X2 &  Z2)
                     | ( X1 &  Z1 & ~X2 &  Z2);
        uint64_t neg = (~X1 &  Z1 &  X2 &  Z2)
                     | ( X1 & ~Z1 & ~X2 &  Z2)
                     | ( X1 &  Z1 &  X2 & ~Z2);
        acc += __builtin_popcountll(pos) - __builtin_popcountll(neg);
    }
    return acc;
}

static int mod4(int v) { return ((v % 4) + 4) % 4; }

static uint64_t xs = 0x243F6A8885A308D3ULL;
static uint64_t rnd(void) {
    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs;
}

int main(void) {
    printf("=== word-parallel rowsum phase formula ===\n");

    /* 1. Exhaustive single-bit check. */
    int all16_ok = 1;
    for (int m = 0; m < 16; m++) {
        int x1 = (m >> 3) & 1, z1 = (m >> 2) & 1;
        int x2 = (m >> 1) & 1, z2 = m & 1;
        uint64_t X1 = x1, Z1 = z1, X2 = x2, Z2 = z2;
        int wp = g_words(&X1, &Z1, &X2, &Z2, 1);
        int sc = g_scalar(x1, z1, x2, z2);
        if (wp != sc) {
            printf("  FAIL  (x1z1x2z2)=%d%d%d%d  word=%d scalar=%d\n",
                   x1, z1, x2, z2, wp, sc);
            all16_ok = 0; failures++;
        }
    }
    if (all16_ok) printf("  OK    all 16 single-bit lanes match rowsum_g\n");

    /* 2. Random word vectors: lane-sum mod 4 must match the scalar sum. */
    int rand_ok = 1;
    for (int trial = 0; trial < 200000; trial++) {
        size_t ws = 1 + (rnd() % 8);
        uint64_t x1[8], z1[8], x2[8], z2[8];
        for (size_t k = 0; k < ws; k++) {
            x1[k] = rnd(); z1[k] = rnd(); x2[k] = rnd(); z2[k] = rnd();
        }
        int wp = g_words(x1, z1, x2, z2, ws);
        int sc = 0;
        for (size_t k = 0; k < ws; k++)
            for (int b = 0; b < 64; b++)
                sc += g_scalar((x1[k] >> b) & 1, (z1[k] >> b) & 1,
                               (x2[k] >> b) & 1, (z2[k] >> b) & 1);
        if (mod4(wp) != mod4(sc)) {
            printf("  FAIL  trial %d ws=%zu word=%d(%d) scalar=%d(%d)\n",
                   trial, ws, wp, mod4(wp), sc, mod4(sc));
            rand_ok = 0; failures++;
            if (failures > 5) break;
        }
    }
    if (rand_ok) printf("  OK    200000 random word vectors match rowsum_g mod 4\n");

    printf("\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
