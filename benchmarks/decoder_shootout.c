/**
 * @file  decoder_shootout.c
 * @brief Decoder-zoo logical-error-rate shoot-out (v0.7.8).
 *
 * Drives every available decoder slot on the same syndrome stream
 * at multiple physical-error rates.  Emits JSON suitable for
 * plotting.  Slots that aren't built / linked are skipped silently.
 *
 * Random X-error model: each data qubit gets an X with probability
 * p, independently.  After error injection we compute the Z-vertex
 * syndrome, hand it to each decoder, apply the proposed correction,
 * and check whether the leftover X-pattern has odd parity around
 * the logical-Z_1 loop (X on column 0).  Each (p, decoder, trial)
 * is one Monte-Carlo sample.
 */

#include "../src/applications/decoder_bench.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* xorshift64 for reproducibility independent of system RNG. */
static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

static double uniform01(uint64_t *s) {
    return (double)(xorshift64(s) >> 11) / (double)(1ULL << 53);
}

/* Toric d x d, num_qubits = 2*d*d.  Horizontal edge (x, y) at
 * index x*d+y; vertical edge (x, y) at d*d+x*d+y. */
static inline int h_idx(int d, int x, int y) { return x * d + y; }
static inline int v_idx(int d, int x, int y) { return d * d + x * d + y; }

/* Compute the Z-vertex syndrome from the X-error pattern.  Vertex
 * (vx, vy) is flagged if the four incident edges have odd parity:
 * horizontal h(vx-1, vy) + h(vx, vy) + vertical v(vx, vy-1) + v(vx, vy). */
static void compute_syndrome(int d, const unsigned char *x_err,
                              unsigned char *syndromes) {
    for (int vx = 0; vx < d; vx++) {
        for (int vy = 0; vy < d; vy++) {
            int p = 0;
            p ^= x_err[h_idx(d, (vx + d - 1) % d, vy)];
            p ^= x_err[h_idx(d, vx, vy)];
            p ^= x_err[v_idx(d, vx, (vy + d - 1) % d)];
            p ^= x_err[v_idx(d, vx, vy)];
            syndromes[vx * d + vy] = (unsigned char)(p & 1);
        }
    }
}

/* Logical-Z_1 = Z on column 0 horizontal edges {h(x, 0) : x in [0, d)}.
 * A residual X-pattern causes a logical fault iff its support has
 * odd intersection with that column. */
static int logical_failure(int d, const unsigned char *x_err) {
    int p = 0;
    for (int x = 0; x < d; x++) p ^= x_err[h_idx(d, x, 0)];
    return p & 1;
}

static double estimate_rate(moonlab_decoder_kind_t slot,
                             int d, double p, int trials, uint64_t seed)
{
    const int n_q = 2 * d * d;
    const int n_s = d * d;
    unsigned char *x_err = (unsigned char *)malloc((size_t)n_q);
    unsigned char *syn   = (unsigned char *)malloc((size_t)n_s);
    unsigned char *corr  = (unsigned char *)malloc((size_t)n_q);
    if (!x_err || !syn || !corr) { free(x_err); free(syn); free(corr); return -1.0; }

    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = n_q, .is_toric = 1
    };

    int failures = 0;
    int attempted = 0;
    for (int t = 0; t < trials; t++) {
        for (int q = 0; q < n_q; q++) {
            x_err[q] = uniform01(&seed) < p ? 1 : 0;
        }
        compute_syndrome(d, x_err, syn);
        memset(corr, 0, (size_t)n_q);
        const moonlab_decoder_input_t in = {
            .code = &code,
            .syndromes = syn,
            .corrections = corr,
            .num_stabilisers = n_s,
            .rng_seed = seed,
        };
        const int rc = moonlab_decoder_decode(slot, &in);
        if (rc != MOONLAB_DECODER_OK) {
            return -1.0; /* slot unavailable -- caller skips this column. */
        }
        attempted++;
        /* Residual = x_err XOR correction. */
        for (int q = 0; q < n_q; q++) x_err[q] ^= corr[q];
        if (logical_failure(d, x_err)) failures++;
    }
    free(x_err); free(syn); free(corr);
    if (attempted == 0) return -1.0;
    return (double)failures / (double)attempted;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    const int d = (argc > 1) ? atoi(argv[1]) : 5;
    const int trials = (argc > 2) ? atoi(argv[2]) : 200;
    const double p_values[] = {0.01, 0.02, 0.04, 0.06, 0.08, 0.10};
    const int n_p = sizeof(p_values) / sizeof(p_values[0]);

    const moonlab_decoder_kind_t slots[] = {
        MOONLAB_DECODER_GREEDY,
        MOONLAB_DECODER_MWPM_EXACT,
        MOONLAB_DECODER_LIBIRREP_SS,
        MOONLAB_DECODER_SBNN,
        MOONLAB_DECODER_PYMATCHING,
    };
    const int n_slots = sizeof(slots) / sizeof(slots[0]);

    fprintf(stdout, "{\n");
    fprintf(stdout, "  \"schema\": \"moonlab/decoder_shootout/v0.7.8\",\n");
    fprintf(stdout, "  \"distance\": %d,\n", d);
    fprintf(stdout, "  \"trials_per_p\": %d,\n", trials);
    fprintf(stdout, "  \"code\": \"toric_%dx%d\",\n", d, d);
    fprintf(stdout, "  \"p_values\": [");
    for (int i = 0; i < n_p; i++) {
        fprintf(stdout, "%s%.4f", i ? ", " : "", p_values[i]);
    }
    fprintf(stdout, "],\n");
    fprintf(stdout, "  \"decoders\": {\n");

    int first_d = 1;
    for (int sl = 0; sl < n_slots; sl++) {
        if (!moonlab_decoder_slot_available(slots[sl])) continue;
        const char *name = moonlab_decoder_slot_name(slots[sl]);
        fprintf(stderr, "  bench %s (slot %d) ...\n", name, (int)slots[sl]);
        if (!first_d) fprintf(stdout, ",\n");
        fprintf(stdout, "    \"%s\": [", name);
        for (int pi = 0; pi < n_p; pi++) {
            const double rate = estimate_rate(slots[sl], d, p_values[pi],
                                              trials, 0xdeadbeefULL + pi);
            if (pi > 0) fprintf(stdout, ", ");
            if (rate < 0) {
                fprintf(stdout, "null");
            } else {
                fprintf(stdout, "%.6f", rate);
            }
        }
        fprintf(stdout, "]");
        first_d = 0;
    }
    fprintf(stdout, "\n  }\n");
    fprintf(stdout, "}\n");
    return 0;
}
