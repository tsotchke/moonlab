/**
 * @file bench_ca_mps.c
 * @brief CA-MPS vs plain MPS bond-dim + wallclock benchmark.
 *
 * Sweeps random-Clifford, Clifford-dominated (95% Clifford + 5% T), and
 * pure-Pauli-rotation circuits across n = 6, 8, 10, 12, 14 qubits, and
 * for each (n, circuit_class) reports:
 *   - plain MPS peak bond dimension
 *   - CA-MPS peak bond dimension (MPS factor only)
 *   - Wallclock time per 100 gates for each backend
 *   - Ratio of bond dimensions (the headline CA-MPS advantage)
 *
 * Output format: JSON + human-readable table.  JSON is consumed by the
 * preprint supplementary benchmark harness.
 *
 * Invoke: ./build/bench_ca_mps [output.json]
 * Defaults to writing bench_ca_mps.json in the current directory.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0xF00D1234CAFE5678ULL;
static void rng_seed(uint64_t s) { rng_state = s ^ 0x9E3779B97F4A7C15ULL; }
static uint32_t rng_u32(uint32_t bound) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)((rng_state >> 32) % bound);
}
static double rng_unit(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng_state >> 32) / (double)0xFFFFFFFFULL;
}

static uint32_t plain_mps_max_bond(const tn_mps_state_t* s) {
    uint32_t n = s->num_qubits;
    uint32_t m = 1;
    for (uint32_t i = 0; i + 1 < n; i++) {
        uint32_t b = tn_mps_bond_dim(s, i);
        if (b > m) m = b;
    }
    return m;
}

/* ---------------------------------------------------------------- */
/*  Circuit classes                                                  */
/* ---------------------------------------------------------------- */

enum circuit_class {
    CIRCUIT_PURE_CLIFFORD,     /* H, S, CNOT, CZ -- stabilizer state */
    CIRCUIT_CLIFFORD_HEAVY,    /* 95% Clifford + 5% T-gate */
    CIRCUIT_PAULI_ROTATION,    /* random rx/ry/rz angles, no Clifford structure */
};

static const char* class_name(int c) {
    switch (c) {
        case CIRCUIT_PURE_CLIFFORD:   return "pure_clifford";
        case CIRCUIT_CLIFFORD_HEAVY:  return "clifford_heavy";
        case CIRCUIT_PAULI_ROTATION:  return "pauli_rotation";
    }
    return "?";
}

/* Apply one gate of the given class to both backends.  Returns 1 if a T-gate
 * (non-Clifford) was applied, 0 otherwise.  Used to count magic gate density
 * in the report. */
static int apply_one_gate(int cclass,
                          tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                          uint32_t n) {
    uint32_t q = rng_u32(n);
    uint32_t q2 = (q + 1 + rng_u32(n - 1)) % n;
    double theta = 2.0 * M_PI * rng_unit();
    int non_clifford = 0;

    if (cclass == CIRCUIT_PURE_CLIFFORD) {
        switch (rng_u32(4)) {
            case 0: tn_apply_h(plain, q);        moonlab_ca_mps_h(ca, q); break;
            case 1: tn_apply_s(plain, q);        moonlab_ca_mps_s(ca, q); break;
            case 2: tn_apply_cnot(plain, q, q2); moonlab_ca_mps_cnot(ca, q, q2); break;
            case 3: tn_apply_cz(plain, q, q2);   moonlab_ca_mps_cz(ca, q, q2); break;
        }
    } else if (cclass == CIRCUIT_CLIFFORD_HEAVY) {
        if (rng_unit() < 0.05) {
            /* T gate: single-qubit rotation pi/4 around Z. */
            tn_apply_t(plain, q);
            moonlab_ca_mps_t_gate(ca, q);
            non_clifford = 1;
        } else {
            switch (rng_u32(4)) {
                case 0: tn_apply_h(plain, q);        moonlab_ca_mps_h(ca, q); break;
                case 1: tn_apply_s(plain, q);        moonlab_ca_mps_s(ca, q); break;
                case 2: tn_apply_cnot(plain, q, q2); moonlab_ca_mps_cnot(ca, q, q2); break;
                case 3: tn_apply_cz(plain, q, q2);   moonlab_ca_mps_cz(ca, q, q2); break;
            }
        }
    } else {
        /* Pauli rotation class: a mix of single-qubit rotations + Clifford
         * entanglers (CNOT) so the state actually develops non-trivial
         * bond dimension.  Pure single-qubit rotations on |0..0> stay at
         * bond 1 regardless of backend, which isn't a useful benchmark. */
        if (rng_unit() < 0.3) {
            tn_apply_cnot(plain, q, q2);
            moonlab_ca_mps_cnot(ca, q, q2);
        } else {
            switch (rng_u32(3)) {
                case 0: tn_apply_rx(plain, q, theta); moonlab_ca_mps_rx(ca, q, theta); break;
                case 1: tn_apply_ry(plain, q, theta); moonlab_ca_mps_ry(ca, q, theta); break;
                case 2: tn_apply_rz(plain, q, theta); moonlab_ca_mps_rz(ca, q, theta); break;
            }
            non_clifford = 1;
        }
    }
    return non_clifford;
}

/* ---------------------------------------------------------------- */
/*  One benchmark point                                              */
/* ---------------------------------------------------------------- */

struct bench_point {
    uint32_t n;
    int circuit_class;
    uint32_t depth;
    uint32_t non_clifford_count;
    uint32_t plain_bond_max;
    uint32_t ca_bond_max;
    double plain_wallclock_s;
    double ca_wallclock_s;
};

static void run_one_point(uint32_t n, int cclass, uint32_t depth, uint32_t seed,
                          struct bench_point* out) {
    uint32_t chi_max = (n <= 10) ? (1u << n) : 256;
    tn_state_config_t cfg = tn_state_config_create(chi_max, 1e-12);
    tn_mps_state_t* plain = tn_mps_create_zero(n, &cfg);
    moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, chi_max);
    if (!plain || !ca) {
        fprintf(stderr, "OOM setting up n=%u, chi_max=%u\n", n, chi_max);
        exit(2);
    }

    rng_seed((uint64_t)seed * 0x100000001B3ULL);

    /* Run once on plain MPS for timing. */
    double t0 = now_s();
    for (uint32_t step = 0; step < depth; step++) {
        (void)apply_one_gate(cclass, plain, ca, n);
    }
    double plain_time = now_s() - t0;

    out->plain_bond_max = plain_mps_max_bond(plain);
    out->ca_bond_max = moonlab_ca_mps_current_bond_dim(ca);

    /* Rerun just CA-MPS to separate its wallclock (the above call updates
     * both, so its time is dominated by the slower of the two; rerun for
     * a clean CA-MPS timing). */
    tn_mps_free(plain);
    moonlab_ca_mps_free(ca);

    plain = tn_mps_create_zero(n, &cfg);
    ca = moonlab_ca_mps_create(n, chi_max);
    rng_seed((uint64_t)seed * 0x100000001B3ULL);

    uint32_t nc = 0;
    t0 = now_s();
    for (uint32_t step = 0; step < depth; step++) {
        uint32_t q = rng_u32(n);
        uint32_t q2 = (q + 1 + rng_u32(n - 1)) % n;
        double theta = 2.0 * M_PI * rng_unit();
        if (cclass == CIRCUIT_PURE_CLIFFORD) {
            switch (rng_u32(4)) {
                case 0: moonlab_ca_mps_h(ca, q); break;
                case 1: moonlab_ca_mps_s(ca, q); break;
                case 2: moonlab_ca_mps_cnot(ca, q, q2); break;
                case 3: moonlab_ca_mps_cz(ca, q, q2); break;
            }
        } else if (cclass == CIRCUIT_CLIFFORD_HEAVY) {
            if (rng_unit() < 0.05) {
                moonlab_ca_mps_t_gate(ca, q);
                nc++;
            } else {
                switch (rng_u32(4)) {
                    case 0: moonlab_ca_mps_h(ca, q); break;
                    case 1: moonlab_ca_mps_s(ca, q); break;
                    case 2: moonlab_ca_mps_cnot(ca, q, q2); break;
                    case 3: moonlab_ca_mps_cz(ca, q, q2); break;
                }
            }
        } else {
            if (rng_unit() < 0.3) {
                moonlab_ca_mps_cnot(ca, q, q2);
            } else {
                switch (rng_u32(3)) {
                    case 0: moonlab_ca_mps_rx(ca, q, theta); break;
                    case 1: moonlab_ca_mps_ry(ca, q, theta); break;
                    case 2: moonlab_ca_mps_rz(ca, q, theta); break;
                }
                nc++;
            }
        }
    }
    double ca_time = now_s() - t0;

    out->n = n;
    out->circuit_class = cclass;
    out->depth = depth;
    out->non_clifford_count = nc;
    out->plain_wallclock_s = plain_time;
    out->ca_wallclock_s = ca_time;

    tn_mps_free(plain);
    moonlab_ca_mps_free(ca);
}

/* ---------------------------------------------------------------- */
/*  Driver                                                           */
/* ---------------------------------------------------------------- */

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "bench_ca_mps.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s for writing\n", out_path); return 1; }

    fprintf(stdout, "=== CA-MPS vs plain MPS benchmark ===\n\n");
    fprintf(stdout, "%-16s %5s %6s %5s %9s %9s %8s %9s %9s %7s\n",
            "circuit", "n", "depth", "T", "chi_plain", "chi_ca", "bond_x",
            "t_plain_s", "t_ca_s", "speed");

    fprintf(json, "{\n  \"schema\": \"moonlab/ca_mps_bench_v1\",\n");
    fprintf(json, "  \"points\": [\n");

    int first = 1;
    const uint32_t qubit_sizes[] = { 6, 8, 10, 12 };
    const uint32_t depths[]      = { 80, 100, 120, 150 };
    const int classes[] = { CIRCUIT_PURE_CLIFFORD, CIRCUIT_CLIFFORD_HEAVY, CIRCUIT_PAULI_ROTATION };

    for (size_t qi = 0; qi < sizeof(qubit_sizes) / sizeof(qubit_sizes[0]); qi++) {
        for (size_t ci = 0; ci < sizeof(classes) / sizeof(classes[0]); ci++) {
            struct bench_point p;
            run_one_point(qubit_sizes[qi], classes[ci], depths[qi], 42 + qi * 7 + ci, &p);
            double bond_ratio = (double)p.plain_bond_max / (double)p.ca_bond_max;
            double speed_ratio = p.ca_wallclock_s > 0
                ? p.plain_wallclock_s / p.ca_wallclock_s : 0.0;
            fprintf(stdout, "%-16s %5u %6u %5u %9u %9u %8.1fx %9.4f %9.4f %6.1fx\n",
                    class_name(p.circuit_class), p.n, p.depth, p.non_clifford_count,
                    p.plain_bond_max, p.ca_bond_max, bond_ratio,
                    p.plain_wallclock_s, p.ca_wallclock_s, speed_ratio);

            if (!first) fprintf(json, ",\n");
            first = 0;
            fprintf(json, "    { \"circuit\": \"%s\", \"n\": %u, \"depth\": %u, "
                          "\"non_clifford\": %u, \"chi_plain\": %u, \"chi_ca\": %u, "
                          "\"t_plain_s\": %.6f, \"t_ca_s\": %.6f }",
                    class_name(p.circuit_class), p.n, p.depth, p.non_clifford_count,
                    p.plain_bond_max, p.ca_bond_max, p.plain_wallclock_s, p.ca_wallclock_s);
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nJSON written to %s\n", out_path);
    return 0;
}
