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
    CIRCUIT_PAULI_ROTATION,    /* random rx/ry/rz angles, mixed CNOTs */
    /* Structured (non-random) workloads that match the section 6
     * design-doc predictions.  These are the regimes CA-MPS is
     * actually targeted at -- a Clifford-rich entangling structure
     * with a sparse, predictable distribution of non-Clifford gates. */
    CIRCUIT_VQE_HEA,           /* hardware-efficient ansatz: Ry+Rz on each qubit, CNOT chain */
    CIRCUIT_QAOA_RING,         /* QAOA on an n-cycle ring: ZZ cost + X mixer */
    CIRCUIT_SURFACE_CYCLE,     /* repeated stabilizer-extraction (pure Clifford) */
};

static const char* class_name(int c) {
    switch (c) {
        case CIRCUIT_PURE_CLIFFORD:   return "pure_clifford";
        case CIRCUIT_CLIFFORD_HEAVY:  return "clifford_heavy";
        case CIRCUIT_PAULI_ROTATION:  return "pauli_rotation";
        case CIRCUIT_VQE_HEA:         return "vqe_hea";
        case CIRCUIT_QAOA_RING:       return "qaoa_ring";
        case CIRCUIT_SURFACE_CYCLE:   return "surface_cycle";
    }
    return "?";
}

/* Each structured class executes one full layer/round per step, so
 * the depth parameter can stay loop-controlled.  Returns the count
 * of non-Clifford gates emitted by the layer. */
static uint32_t apply_layer_vqe_hea(tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                                     uint32_t n) {
    /* VQE-HEA layer: per-qubit Ry(theta) Rz(phi) + linear CNOT chain.
     * The angles are deterministic functions of qubit index + a per-call
     * counter so successive layers are not identical (which would make
     * Rz collapse into a global phase).  Rotation angles are kept
     * generic (well off any magic-angle multiple of pi/4) so the
     * non-Clifford content is real. */
    uint32_t nc = 0;
    for (uint32_t q = 0; q < n; q++) {
        double th = 0.137 + 0.29 * (double)q + 0.413 * rng_unit();
        double ph = 0.241 + 0.31 * (double)q + 0.519 * rng_unit();
        if (plain) tn_apply_ry(plain, q, th);
        if (ca)    moonlab_ca_mps_ry(ca, q, th);
        if (plain) tn_apply_rz(plain, q, ph);
        if (ca)    moonlab_ca_mps_rz(ca, q, ph);
        nc += 2;
    }
    for (uint32_t q = 0; q + 1 < n; q++) {
        if (plain) tn_apply_cnot(plain, q, q + 1);
        if (ca)    moonlab_ca_mps_cnot(ca, q, q + 1);
    }
    return nc;
}

/* Apply exp(-i * gamma * Z_i Z_j) via CNOT-Rz-CNOT decomposition.
 * The Rz is the only non-Clifford gate. */
static void apply_zz_rotation(tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                               uint32_t i, uint32_t j, double gamma) {
    if (plain) tn_apply_cnot(plain, i, j);
    if (ca)    moonlab_ca_mps_cnot(ca, i, j);
    if (plain) tn_apply_rz(plain, j, 2.0 * gamma);
    if (ca)    moonlab_ca_mps_rz(ca, j, 2.0 * gamma);
    if (plain) tn_apply_cnot(plain, i, j);
    if (ca)    moonlab_ca_mps_cnot(ca, i, j);
}

static uint32_t apply_round_qaoa_ring(tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                                       uint32_t n, double gamma, double beta) {
    /* One QAOA round on an n-cycle: cost layer over edges (i, i+1 mod n),
     * then mixer layer Rx(2*beta) on each qubit.  Each ZZ rotation
     * contributes one non-Clifford Rz; mixer contributes n. */
    uint32_t nc = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = (i + 1) % n;
        apply_zz_rotation(plain, ca, i, j, gamma);
        nc += 1;
    }
    for (uint32_t q = 0; q < n; q++) {
        if (plain) tn_apply_rx(plain, q, 2.0 * beta);
        if (ca)    moonlab_ca_mps_rx(ca, q, 2.0 * beta);
        nc += 1;
    }
    return nc;
}

static uint32_t apply_layer_surface_cycle(tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                                           uint32_t n) {
    /* Surface-code-like stabilizer-extraction cycle (no measurements, the
     * Clifford content is what we benchmark): X-stabilizer pattern
     * H ; CNOT(a, d_left) ; CNOT(a, d_right) ; H, plus Z-stabilizer
     * CNOT(d, a) chains.  Even-indexed qubits act as data, odd-indexed
     * as ancillas; this is a 1D "stabilizer chain" that stresses both
     * short-range entanglement *and* CA-MPS's pure-Clifford limit
     * (chi_ca should stay at 1). */
    for (uint32_t a = 1; a + 1 < n; a += 2) {
        if (plain) tn_apply_h(plain, a);
        if (ca)    moonlab_ca_mps_h(ca, a);
        if (plain) tn_apply_cnot(plain, a, a - 1);
        if (ca)    moonlab_ca_mps_cnot(ca, a, a - 1);
        if (plain) tn_apply_cnot(plain, a, a + 1);
        if (ca)    moonlab_ca_mps_cnot(ca, a, a + 1);
        if (plain) tn_apply_h(plain, a);
        if (ca)    moonlab_ca_mps_h(ca, a);
    }
    /* Z-stabilizer pattern shifted by one: even-indexed ancillas. */
    for (uint32_t a = 2; a + 1 < n; a += 2) {
        if (plain) tn_apply_cnot(plain, a - 1, a);
        if (ca)    moonlab_ca_mps_cnot(ca, a - 1, a);
        if (plain) tn_apply_cnot(plain, a + 1, a);
        if (ca)    moonlab_ca_mps_cnot(ca, a + 1, a);
    }
    return 0;
}

/* Apply one gate of the given class to whichever backend is non-NULL.
 * Returns 1 if a T-gate or Pauli rotation (non-Clifford) was applied,
 * 0 otherwise.  Used to count magic gate density in the report. */
static int apply_one_gate(int cclass,
                          tn_mps_state_t* plain, moonlab_ca_mps_t* ca,
                          uint32_t n) {
    uint32_t q = rng_u32(n);
    uint32_t q2 = (q + 1 + rng_u32(n - 1)) % n;
    double theta = 2.0 * M_PI * rng_unit();
    int non_clifford = 0;

    if (cclass == CIRCUIT_PURE_CLIFFORD) {
        switch (rng_u32(4)) {
            case 0:
                if (plain) tn_apply_h(plain, q);
                if (ca)    moonlab_ca_mps_h(ca, q);
                break;
            case 1:
                if (plain) tn_apply_s(plain, q);
                if (ca)    moonlab_ca_mps_s(ca, q);
                break;
            case 2:
                if (plain) tn_apply_cnot(plain, q, q2);
                if (ca)    moonlab_ca_mps_cnot(ca, q, q2);
                break;
            case 3:
                if (plain) tn_apply_cz(plain, q, q2);
                if (ca)    moonlab_ca_mps_cz(ca, q, q2);
                break;
        }
    } else if (cclass == CIRCUIT_CLIFFORD_HEAVY) {
        if (rng_unit() < 0.05) {
            /* T gate: single-qubit rotation pi/4 around Z. */
            if (plain) tn_apply_t(plain, q);
            if (ca)    moonlab_ca_mps_t_gate(ca, q);
            non_clifford = 1;
        } else {
            switch (rng_u32(4)) {
                case 0:
                    if (plain) tn_apply_h(plain, q);
                    if (ca)    moonlab_ca_mps_h(ca, q);
                    break;
                case 1:
                    if (plain) tn_apply_s(plain, q);
                    if (ca)    moonlab_ca_mps_s(ca, q);
                    break;
                case 2:
                    if (plain) tn_apply_cnot(plain, q, q2);
                    if (ca)    moonlab_ca_mps_cnot(ca, q, q2);
                    break;
                case 3:
                    if (plain) tn_apply_cz(plain, q, q2);
                    if (ca)    moonlab_ca_mps_cz(ca, q, q2);
                    break;
            }
        }
    } else {
        /* Pauli rotation class: a mix of single-qubit rotations + Clifford
         * entanglers (CNOT) so the state actually develops non-trivial
         * bond dimension.  Pure single-qubit rotations on |0..0> stay at
         * bond 1 regardless of backend, which isn't a useful benchmark. */
        if (rng_unit() < 0.3) {
            if (plain) tn_apply_cnot(plain, q, q2);
            if (ca)    moonlab_ca_mps_cnot(ca, q, q2);
        } else {
            switch (rng_u32(3)) {
                case 0:
                    if (plain) tn_apply_rx(plain, q, theta);
                    if (ca)    moonlab_ca_mps_rx(ca, q, theta);
                    break;
                case 1:
                    if (plain) tn_apply_ry(plain, q, theta);
                    if (ca)    moonlab_ca_mps_ry(ca, q, theta);
                    break;
                case 2:
                    if (plain) tn_apply_rz(plain, q, theta);
                    if (ca)    moonlab_ca_mps_rz(ca, q, theta);
                    break;
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

/* True iff the class advances depth in units of "one full layer" rather
 * than "one random gate".  Affects how max_bond is reported (always
 * the post-circuit peak) but the timing convention is identical. */
static int is_structured(int cclass) {
    return cclass == CIRCUIT_VQE_HEA
        || cclass == CIRCUIT_QAOA_RING
        || cclass == CIRCUIT_SURFACE_CYCLE;
}

/* One step of the named circuit class on both backends.
 * Returns the number of non-Clifford gates applied. */
static uint32_t apply_one_step(int cclass, tn_mps_state_t* plain,
                                moonlab_ca_mps_t* ca, uint32_t n,
                                uint32_t step_idx) {
    (void)step_idx;
    if (is_structured(cclass)) {
        switch (cclass) {
            case CIRCUIT_VQE_HEA:
                return apply_layer_vqe_hea(plain, ca, n);
            case CIRCUIT_QAOA_RING: {
                /* Use the standard QAOA "fixed-angle" schedule for p=1 on
                 * unweighted ring: gamma ~ pi/4, beta ~ pi/8.  Subsequent
                 * rounds drift the angles slightly so the workload isn't
                 * a single repeated layer. */
                double gamma = 0.785 + 0.05 * (double)step_idx;
                double beta  = 0.392 + 0.03 * (double)step_idx;
                return apply_round_qaoa_ring(plain, ca, n, gamma, beta);
            }
            case CIRCUIT_SURFACE_CYCLE:
                return apply_layer_surface_cycle(plain, ca, n);
        }
        return 0;
    }
    /* Random (non-structured) classes: one gate per step. */
    return (uint32_t)apply_one_gate(cclass, plain, ca, n);
}

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

    /* Pass 1: run on plain MPS only (CA-MPS gets re-run for clean timing). */
    double t0 = now_s();
    for (uint32_t step = 0; step < depth; step++) {
        (void)apply_one_step(cclass, plain, NULL, n, step);
    }
    double plain_time = now_s() - t0;
    out->plain_bond_max = plain_mps_max_bond(plain);
    tn_mps_free(plain);

    /* Pass 2: run on CA-MPS only with the same RNG sequence. */
    moonlab_ca_mps_free(ca);
    ca = moonlab_ca_mps_create(n, chi_max);
    rng_seed((uint64_t)seed * 0x100000001B3ULL);

    uint32_t nc = 0;
    t0 = now_s();
    for (uint32_t step = 0; step < depth; step++) {
        nc += apply_one_step(cclass, NULL, ca, n, step);
    }
    double ca_time = now_s() - t0;

    out->n = n;
    out->circuit_class = cclass;
    out->depth = depth;
    out->non_clifford_count = nc;
    out->ca_bond_max = moonlab_ca_mps_current_bond_dim(ca);
    out->plain_wallclock_s = plain_time;
    out->ca_wallclock_s = ca_time;

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

    /* The schedule pairs each (n, class) point with a class-appropriate
     * depth.  Random-circuit classes use gate-count depth; structured
     * classes use layer count.  A VQE/HEA layer is ~3n gates, a QAOA
     * round is ~5n gates, and a surface-cycle layer is ~3n gates. */
    struct bench_schedule {
        int cclass;
        const uint32_t* depths;     /* indexed by qubit_sizes index */
    };
    const uint32_t qubit_sizes[] = { 6, 8, 10, 12 };
    const uint32_t depths_random[]   = { 80, 100, 120, 150 };
    const uint32_t depths_vqe[]      = {  4,   4,   4,   4 };  /* layers */
    const uint32_t depths_qaoa[]     = {  4,   4,   4,   4 };  /* rounds */
    const uint32_t depths_surface[]  = {  6,   8,  10,  12 };  /* cycles */
    const struct bench_schedule schedule[] = {
        { CIRCUIT_PURE_CLIFFORD,  depths_random  },
        { CIRCUIT_CLIFFORD_HEAVY, depths_random  },
        { CIRCUIT_PAULI_ROTATION, depths_random  },
        { CIRCUIT_VQE_HEA,        depths_vqe     },
        { CIRCUIT_QAOA_RING,      depths_qaoa    },
        { CIRCUIT_SURFACE_CYCLE,  depths_surface },
    };
    const size_t num_classes = sizeof(schedule) / sizeof(schedule[0]);

    for (size_t qi = 0; qi < sizeof(qubit_sizes) / sizeof(qubit_sizes[0]); qi++) {
        for (size_t ci = 0; ci < num_classes; ci++) {
            struct bench_point p;
            run_one_point(qubit_sizes[qi], schedule[ci].cclass,
                          schedule[ci].depths[qi], 42 + qi * 7 + ci, &p);
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
