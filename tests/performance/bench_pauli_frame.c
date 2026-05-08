/**
 * @file bench_pauli_frame.c
 * @brief Pauli-frame batched-shot throughput on surface-code circuits.
 *
 * Reports shots/sec for a depth-d surface-code Z-stabilizer cycle with
 * i.i.d. depolarising noise on every data qubit per round.  This is
 * the core kernel that lets surface-code threshold sweeps run at
 * distances >= 30, where a tableau-per-shot simulator would be too
 * slow.  Stim's published baseline at d=51 with 10^4 shots is on the
 * order of 10^9 frame-bit operations / second.  Closes the moonlab
 * release-arc §2B claim ("Stim-style Pauli-frame Clifford backend").
 *
 * The simulated circuit per round:
 *   - Reset Z-ancillas to |0>.
 *   - For each Z-stabilizer: CNOT each of its 2-4 data qubits onto
 *     the ancilla.
 *   - Inject i.i.d. depolarising error on every data qubit.
 *   - Measure the ancilla.
 *
 * Output JSON: schema "moonlab/pauli_frame_v1".
 */

#include "../../src/backends/clifford/pauli_frame.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* Rotated surface-code Z-stabilizer pattern: a d-by-d data lattice
 * with (d-1)*(d-1) plaquettes, half (a+b odd) Z-checks.  Each Z-check
 * at (a, b) acts on data (a,b), (a+1,b), (a,b+1), (a+1,b+1).
 *
 * For the bench we only simulate the Z-stabilizer rounds (the X
 * rounds are symmetric; the throughput claim is the same).  Each Z-
 * round requires one ancilla per Z-check, giving d^2 + (d^2-1)/2
 * total qubits.  We use 4*d^2 as a conservative upper bound to allow
 * room for ancilla bookkeeping. */

typedef struct {
    uint32_t d;
    uint32_t n_data;       /* d*d */
    uint32_t n_z_checks;   /* (d-1)*(d-1)/2, but we use the full
                            * (d-1)^2 grid and gate it by parity.    */
    uint32_t n_total;      /* data + ancillas in our flat indexing.  */
} bench_lattice_t;

static void lattice_init(bench_lattice_t* L, uint32_t d) {
    L->d = d;
    L->n_data = d * d;
    L->n_z_checks = (d - 1) * (d - 1) / 2;
    /* Layout: data qubits first (indices 0..d^2-1), then ancillas
     * (one per dual site at (a, b) with a+b odd, indices d^2 + ...). */
    L->n_total = L->n_data + (d - 1) * (d - 1);  /* ample */
}

static uint32_t data_idx(const bench_lattice_t* L, uint32_t i, uint32_t j) {
    return i * L->d + j;
}

static uint32_t ancilla_idx(const bench_lattice_t* L,
                             uint32_t a, uint32_t b) {
    return L->n_data + a * (L->d - 1) + b;
}

/* One round of Z-stabilizer measurement on a batched-frame state.
 * Steps per round (frame propagation):
 *   - Inject depolarising noise on every data qubit.
 *   - For each Z-check at dual (a, b) with (a+b) odd:
 *       . CNOT data->ancilla for each of its (up to 4) data qubits.
 *       . Noisy measurement of the ancilla (with measurement-error
 *         probability @p p_meas).  The recorded syndrome bit is
 *         discarded by this throughput bench; in production a real
 *         sim would XOR each syndrome bit into the matched-graph
 *         input for the decoder.
 *       . Ancilla resets to |0> as part of the measurement.
 */
static void run_one_round(pauli_frame_batch_t* b,
                           const bench_lattice_t* L,
                           double p,
                           double p_meas,
                           uint64_t* rng,
                           uint8_t* syndrome_buf) {
    /* 1) Depolarising error on every data qubit. */
    for (uint32_t q = 0; q < L->n_data; q++) {
        pauli_frame_batch_depolarising(b, q, p, rng);
    }
    /* 2) Z-stabilizer measurements via CNOT to ancilla + noisy readout. */
    const uint32_t d = L->d;
    for (uint32_t a = 0; a + 1 < d; a++) {
        for (uint32_t bb = 0; bb + 1 < d; bb++) {
            if (((a + bb) & 1) == 0) continue;  /* X-checks; skip */
            const uint32_t anc = ancilla_idx(L, a, bb);
            const uint32_t q00 = data_idx(L, a,     bb);
            const uint32_t q10 = data_idx(L, a + 1, bb);
            const uint32_t q01 = data_idx(L, a,     bb + 1);
            const uint32_t q11 = data_idx(L, a + 1, bb + 1);
            pauli_frame_batch_cnot(b, q00, anc);
            pauli_frame_batch_cnot(b, q10, anc);
            pauli_frame_batch_cnot(b, q01, anc);
            pauli_frame_batch_cnot(b, q11, anc);
            pauli_frame_batch_measure_z_noisy(b, anc, p_meas, rng,
                                               syndrome_buf);
        }
    }
}

typedef struct {
    uint32_t d;
    uint32_t rounds;
    size_t   n_shots;
    double   p;
    double   total_wall_s;
    double   shots_per_s;
    double   frame_ops_per_s;
} pf_row_t;

static void bench_one(pf_row_t* row, uint32_t d, uint32_t rounds,
                       size_t n_shots, double p) {
    bench_lattice_t L; lattice_init(&L, d);
    pauli_frame_batch_t* b =
        pauli_frame_batch_create(L.n_total, n_shots);
    if (!b) {
        fprintf(stderr, "alloc failed for d=%u shots=%zu\n", d, n_shots);
        row->total_wall_s = 0.0; row->shots_per_s = 0.0;
        return;
    }
    uint64_t rng = 0xBEEFULL ^ ((uint64_t)d << 32) ^ (uint64_t)rounds;
    /* Per-shot syndrome buffer used by the noisy-measurement primitive. */
    uint8_t* syndrome_buf = (uint8_t*)malloc(n_shots);
    /* Phenomenological noise: data depolarising at p, ancilla
     * measurement flip at the same rate (standard Wang convention). */
    const double p_meas = p;

    /* Warm cache. */
    run_one_round(b, &L, p, p_meas, &rng, syndrome_buf);

    pauli_frame_batch_clear(b);
    rng = 0xBEEFULL ^ ((uint64_t)d << 32) ^ (uint64_t)rounds;

    const double t0 = now_s();
    for (uint32_t r = 0; r < rounds; r++) {
        run_one_round(b, &L, p, p_meas, &rng, syndrome_buf);
    }
    const double dt = now_s() - t0;
    free(syndrome_buf);

    /* Frame-ops counted as: per round, n_data depolarising draws
     * (each ~64 frame-bit XORs) + n_z_checks * 4 CNOTs (each ~128
     * frame-bit XORs).  Order-of-magnitude estimate. */
    const double ops_per_round =
        (double)L.n_data * (double)n_shots +
        (double)L.n_z_checks * 4.0 * 2.0 * (double)n_shots;
    const double total_ops = ops_per_round * (double)rounds;

    row->d = d;
    row->rounds = rounds;
    row->n_shots = n_shots;
    row->p = p;
    row->total_wall_s = dt;
    row->shots_per_s = (double)n_shots * (double)rounds / dt;
    row->frame_ops_per_s = total_ops / dt;

    pauli_frame_batch_free(b);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "pauli_frame.json";
    const size_t n_shots = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10)
                                         : 10000;
    const uint32_t rounds = (argc >= 4) ? (uint32_t)strtoul(argv[3], NULL, 10)
                                         : 50;
    const double p = (argc >= 5) ? strtod(argv[4], NULL) : 0.005;

    printf("=== Pauli-frame surface-code throughput ===\n");
    printf("  schema: moonlab/pauli_frame_v1\n");
    printf("  shots: %zu  rounds: %u  p_depol: %g  out: %s\n\n",
           n_shots, rounds, p, out_path);
    printf("  %-4s %-7s %-12s %-14s %-14s\n",
           "d", "n_data", "wall_s", "shots/sec", "frame_ops/sec");

    const uint32_t distances[] = { 5, 9, 15, 23, 31 };
    const size_t n_d = sizeof(distances) / sizeof(distances[0]);
    pf_row_t rows[8];

    for (size_t i = 0; i < n_d; i++) {
        bench_one(&rows[i], distances[i], rounds, n_shots, p);
        printf("  %-4u %-7u %-12.3f %-14.3e %-14.3e\n",
               rows[i].d, rows[i].d * rows[i].d,
               rows[i].total_wall_s,
               rows[i].shots_per_s, rows[i].frame_ops_per_s);
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/pauli_frame_v1\",\n");
    fprintf(f, "  \"description\": \"Pauli-frame batched-shot throughput "
               "on a rotated surface-code Z-stabilizer cycle.  Each "
               "round injects i.i.d. depolarising noise on every data "
               "qubit and runs the standard CNOT-to-ancilla syndrome "
               "extraction.  The bit-packed batch lets a single AVX-1 "
               "uint64 XOR advance 64 shots simultaneously.\",\n");
    fprintf(f, "  \"params\": {\"n_shots\": %zu, \"rounds\": %u, "
                "\"p_depolarising\": %g},\n",
            n_shots, rounds, p);
    fprintf(f, "  \"rows\": [");
    for (size_t i = 0; i < n_d; i++) {
        fprintf(f, "%s\n    {\"d\": %u, \"n_data\": %u, "
                   "\"total_wall_s\": %.6f, \"shots_per_s\": %.3f, "
                   "\"frame_ops_per_s\": %.3f}",
                i == 0 ? "" : ",",
                rows[i].d, rows[i].d * rows[i].d,
                rows[i].total_wall_s,
                rows[i].shots_per_s, rows[i].frame_ops_per_s);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("\nwrote %s\n", out_path);
    return 0;
}
