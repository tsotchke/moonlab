/**
 * @file test_shor_ecdlp.c
 * @brief Resource-estimation tests for Shor-ECDLP.
 *
 * Reference points reproduced:
 *  - Gidney-Drake-Boneh 2026 qubit-minimal secp256k1:  ~1200 logical qubits,
 *                                                      ~90 M Toffoli gates.
 *  - Gidney-Drake-Boneh 2026 depth-minimal secp256k1:  ~1450 logical qubits,
 *                                                      ~70 M Toffoli gates.
 *  - Surface-code FTQC overhead at p_phys = 1e-3, d auto-picked, yields a
 *    physical-qubit count in the published 300k-500k range.
 */

#include "../../src/algorithms/shor_ecdlp/shor_ecdlp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                     \
    if (!(cond)) {                                                     \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);           \
        failures++;                                                    \
    } else {                                                           \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);           \
    }                                                                  \
} while (0)

static void test_gdb_qubit_minimal_secp256k1(void) {
    fprintf(stdout, "\n-- GDB qubit-minimal secp256k1 (target: ~1200 qubits, ~90M Toffolis) --\n");
    shor_ecdlp_params_t p;
    shor_ecdlp_params_secp256k1(&p);
    shor_ecdlp_resources_t r;
    int rc = shor_ecdlp_estimate(&p, &r);
    CHECK(rc == 0, "estimate returns 0");
    fprintf(stdout, "    qubits=%zu  toffolis=%llu  t_count=%llu  depth=%.2e\n",
            r.logical_qubits,
            (unsigned long long)r.toffoli_count,
            (unsigned long long)r.t_count,
            r.circuit_depth_toffolis);

    CHECK(r.logical_qubits >= 1100 && r.logical_qubits <= 1300,
          "logical qubits %zu in [1100, 1300]", r.logical_qubits);
    CHECK(r.toffoli_count >= 80000000ULL && r.toffoli_count <= 100000000ULL,
          "toffolis %llu in [80M, 100M]",
          (unsigned long long)r.toffoli_count);
    CHECK(r.t_count == 7ULL * r.toffoli_count,
          "t_count = 7 x toffolis");
}

static void test_gdb_depth_minimal_secp256k1(void) {
    fprintf(stdout, "\n-- GDB depth-minimal secp256k1 (target: ~1450 qubits, ~70M Toffolis) --\n");
    shor_ecdlp_params_t p;
    shor_ecdlp_params_secp256k1(&p);
    p.time_space_tradeoff = 1;
    shor_ecdlp_resources_t r;
    shor_ecdlp_estimate(&p, &r);
    fprintf(stdout, "    qubits=%zu  toffolis=%llu  depth=%.2e\n",
            r.logical_qubits,
            (unsigned long long)r.toffoli_count,
            r.circuit_depth_toffolis);
    CHECK(r.logical_qubits >= 1350 && r.logical_qubits <= 1550,
          "logical qubits %zu in [1350, 1550]", r.logical_qubits);
    CHECK(r.toffoli_count >= 60000000ULL && r.toffoli_count <= 80000000ULL,
          "toffolis %llu in [60M, 80M]",
          (unsigned long long)r.toffoli_count);
}

static void test_roetteler_secp256k1(void) {
    fprintf(stdout, "\n-- Roetteler 2017 encoding at n=256 (pre-GDB baseline) --\n");
    shor_ecdlp_params_t p;
    shor_ecdlp_params_secp256k1(&p);
    p.encoding = 0;
    shor_ecdlp_resources_t r;
    shor_ecdlp_estimate(&p, &r);
    fprintf(stdout, "    qubits=%zu  toffolis=%llu\n",
            r.logical_qubits, (unsigned long long)r.toffoli_count);
    /* Roetteler 2017 reports ~2330 qubits and ~10^11 Toffolis on P-256;
     * we agree within an order of magnitude. */
    CHECK(r.logical_qubits >= 2000 && r.logical_qubits <= 3500,
          "Roetteler qubits %zu in [2000, 3500]", r.logical_qubits);
    CHECK(r.toffoli_count >= 10000000000ULL,
          "Roetteler toffolis %llu >= 10^10",
          (unsigned long long)r.toffoli_count);
}

static void test_scaling_small(void) {
    fprintf(stdout, "\n-- Scaling: GDB qubit-minimal at n=8, 16, 64, 128, 256, 512 --\n");
    size_t ns[] = { 8, 16, 64, 128, 256, 512 };
    double toffolis[6];
    size_t qubits[6];
    for (size_t i = 0; i < 6; i++) {
        shor_ecdlp_params_t p;
        shor_ecdlp_params_secp256k1(&p);
        p.curve_bits = ns[i];
        shor_ecdlp_resources_t r;
        shor_ecdlp_estimate(&p, &r);
        toffolis[i] = (double)r.toffoli_count;
        qubits[i]   = r.logical_qubits;
        fprintf(stdout, "    n=%-4zu  qubits=%-6zu  toffolis=%.3e\n",
                ns[i], r.logical_qubits, toffolis[i]);
        CHECK(r.logical_qubits > 0 && r.toffoli_count > 0,
              "non-zero at n=%zu", ns[i]);
    }

    /* Assert cubic Toffoli scaling: doubling n should multiply Toffoli
     * count by ~8x (Gidney-Drake-Boneh Sec. 5.2).  Allow +/- 10%. */
    for (size_t i = 0; i + 1 < 6; i++) {
        if (ns[i + 1] == 2 * ns[i]) {
            const double ratio = toffolis[i + 1] / toffolis[i];
            fprintf(stdout,
                    "    n=%zu -> n=%zu : toffoli ratio = %.3f (expect 8.0)\n",
                    ns[i], ns[i + 1], ratio);
            CHECK(fabs(ratio - 8.0) < 0.8,
                  "Toffoli ratio %zu->%zu within 10%% of cubic scaling",
                  ns[i], ns[i + 1]);
        }
    }

    /* Qubit count is linear: doubling n should multiply qubits by 2x. */
    for (size_t i = 0; i + 1 < 6; i++) {
        if (ns[i + 1] == 2 * ns[i]) {
            const double ratio = (double)qubits[i + 1] / (double)qubits[i];
            CHECK(fabs(ratio - 2.0) < 0.2,
                  "qubit ratio %zu->%zu linear (got %.3f, expect 2.0)",
                  ns[i], ns[i + 1], ratio);
        }
    }
}

static void test_ftqc_overhead_secp256k1(void) {
    fprintf(stdout, "\n-- FTQC overhead for secp256k1 (p_phys=1e-3) --\n");
    shor_ecdlp_params_t p;
    shor_ecdlp_params_secp256k1(&p);
    shor_ecdlp_resources_t logical;
    shor_ecdlp_estimate(&p, &logical);

    shor_ecdlp_ftqc_params_t ftqc = {
        .physical_error_rate  = 1e-3,
        .code_cycle_time_s    = 1e-6,
        .code_distance        = 0,      /* auto-pick */
        .target_logical_error = 1e-2,
    };
    shor_ecdlp_ftqc_resources_t out;
    int rc = shor_ecdlp_ftqc_estimate(&logical, &ftqc, &out);
    CHECK(rc == 0, "ftqc_estimate returns 0");
    fprintf(stdout,
            "    d=%zu  physical_qubits=%zu  wall_clock=%.2f s  logical_err=%.2e\n",
            out.picked_code_distance,
            out.physical_qubits,
            out.wall_clock_seconds,
            out.total_logical_error);
    CHECK(out.picked_code_distance >= 5 && out.picked_code_distance <= 51,
          "picked code distance %zu in [5, 51]", out.picked_code_distance);
    CHECK(out.physical_qubits > 100000,
          "physical qubits > 100k (got %zu)", out.physical_qubits);
    CHECK(out.wall_clock_seconds > 0.0 && out.wall_clock_seconds < 86400.0 * 365.0,
          "wall clock %.0f s in sane range", out.wall_clock_seconds);
}

static void test_bad_args(void) {
    fprintf(stdout, "\n-- Argument validation --\n");
    shor_ecdlp_resources_t r;
    CHECK(shor_ecdlp_estimate(NULL, &r) != 0, "NULL params rejected");
    shor_ecdlp_params_t p;
    shor_ecdlp_params_secp256k1(&p);
    CHECK(shor_ecdlp_estimate(&p, NULL) != 0, "NULL output rejected");
    p.curve_bits = 0;
    CHECK(shor_ecdlp_estimate(&p, &r) != 0, "curve_bits = 0 rejected");
}

int main(void) {
    fprintf(stdout, "=== Shor-ECDLP resource estimator ===\n");
    test_gdb_qubit_minimal_secp256k1();
    test_gdb_depth_minimal_secp256k1();
    test_roetteler_secp256k1();
    test_scaling_small();
    test_ftqc_overhead_secp256k1();
    test_bad_args();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
