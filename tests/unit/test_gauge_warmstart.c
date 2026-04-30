/**
 * @file test_gauge_warmstart.c
 * @brief Unit test for the stabilizer-subgroup warmstart Clifford builder.
 *
 * Pins, for each test case:
 *   - The warmstart returns CA_MPS_SUCCESS on commuting/independent
 *     generators.
 *   - The prepared CA-MPS state |psi> = D|phi_0> with phi_0 = |0^n>
 *     satisfies <psi|g|psi> = +1 (within numerical tolerance) for
 *     every input generator g.
 *   - Bad input (non-commuting generators) returns CA_MPS_ERR_INVALID.
 *
 * Cases:
 *   1. Single 1-qubit generator g_0 = X_0.
 *   2. Bell-pair stabilizers {XX, ZZ} on 2 qubits.
 *   3. GHZ-3 stabilizers {XXX, ZZI, IZZ} on 3 qubits (mixed weight,
 *      mixed Pauli types).
 *   4. 1+1D Z2 LGT Gauss-law operators on N = 4 matter sites
 *      (7 qubits, 2 interior Gauss-law operators of weight 3).
 *   5. Anti-commuting generators {X_0, Z_0} -> error path.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h"
#include "../../src/applications/hep/lattice_z2_1d.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); failures++; } \
} while (0)

/* Evaluate <psi|g|psi> for a single Pauli string g. */
static double expect_pauli_string(const moonlab_ca_mps_t* s,
                                    const uint8_t* g,
                                    uint32_t n) {
    double _Complex coeff = 1.0 + 0.0 * _Complex_I;
    double _Complex out   = 0.0 + 0.0 * _Complex_I;
    ca_mps_error_t e = moonlab_ca_mps_expect_pauli_sum(s, g, &coeff, 1, &out);
    if (e != CA_MPS_SUCCESS) {
        fprintf(stderr, "expect_pauli_sum returned %d\n", (int)e);
        return NAN;
    }
    /* For a Hermitian Pauli string, the imaginary part should be ~0. */
    if (fabs(cimag(out)) > 1e-9) {
        fprintf(stderr, "WARN imag(<g>) = %.3e\n", cimag(out));
    }
    return creal(out);
}

static int run_case(const char* label,
                    uint32_t num_qubits,
                    uint32_t num_gens,
                    const uint8_t* paulis) {
    fprintf(stdout, "\n--- %s ---\n", label);
    moonlab_ca_mps_t* s = moonlab_ca_mps_create(num_qubits, 32);
    CHECK(s != NULL, "create CA-MPS");
    if (!s) return -1;

    ca_mps_error_t err = moonlab_ca_mps_apply_stab_subgroup_warmstart(
        s, paulis, num_gens);
    CHECK(err == CA_MPS_SUCCESS, "warmstart apply (got %d)", (int)err);

    if (err == CA_MPS_SUCCESS) {
        for (uint32_t i = 0; i < num_gens; i++) {
            double v = expect_pauli_string(s, &paulis[(size_t)i * num_qubits],
                                              num_qubits);
            CHECK(fabs(v - 1.0) < 1e-6,
                  "<g_%u> = %.9f (expected +1)", (unsigned)i, v);
        }
    }

    moonlab_ca_mps_free(s);
    return 0;
}

int main(void) {
    fprintf(stdout, "=== gauge-aware stabilizer-subgroup warmstart ===\n");

    /* --- Case 1: g_0 = X_0 on 1 qubit. ----------------------------- */
    {
        uint8_t g[1] = { 1 };  /* X */
        run_case("case 1: X_0 (single qubit)", 1, 1, g);
    }

    /* --- Case 2: Bell-pair stabilizers {XX, ZZ}. ------------------- */
    {
        uint8_t g[4] = {
            1, 1,   /* X X */
            3, 3    /* Z Z */
        };
        run_case("case 2: Bell pair {XX, ZZ}", 2, 2, g);
    }

    /* --- Case 3: GHZ-3 stabilizers {XXX, ZZI, IZZ}. ---------------- */
    {
        uint8_t g[9] = {
            1, 1, 1,   /* X X X */
            3, 3, 0,   /* Z Z I */
            0, 3, 3    /* I Z Z */
        };
        run_case("case 3: GHZ-3 {XXX, ZZI, IZZ}", 3, 3, g);
    }

    /* --- Case 4: Z2 LGT Gauss-law operators, N=4 matter sites. ----- */
    {
        z2_lgt_config_t cfg = {0};
        cfg.num_matter_sites = 4;
        cfg.t_hop = 1.0; cfg.h_link = 1.0; cfg.mass = 0.0;
        cfg.gauss_penalty = 0.0;
        uint32_t n = z2_lgt_1d_num_qubits(&cfg);   /* 7 */
        uint32_t k = cfg.num_matter_sites - 2;     /* 2 interior Gauss-law */
        uint8_t* paulis = (uint8_t*)calloc((size_t)k * n, 1);
        for (uint32_t i = 0; i < k; i++) {
            int rc = z2_lgt_1d_gauss_law_pauli(&cfg, i + 1,
                                                  &paulis[(size_t)i * n]);
            CHECK(rc == 0, "build Gauss-law %u (rc=%d)", (unsigned)(i + 1), rc);
        }
        run_case("case 4: Z2 LGT N=4 Gauss-law generators", n, k, paulis);
        free(paulis);
    }

    /* --- Case 5: anti-commuting generators -> error. --------------- */
    {
        uint8_t g[2] = { 1, 3 };  /* g_0 = X_0, g_1 = Z_0; anti-commute */
        moonlab_ca_mps_t* s = moonlab_ca_mps_create(1, 8);
        CHECK(s != NULL, "create CA-MPS for case 5");
        if (s) {
            ca_mps_error_t err =
                moonlab_ca_mps_apply_stab_subgroup_warmstart(s, g, 2);
            fprintf(stdout, "\n--- case 5: {X_0, Z_0} anti-commute ---\n");
            CHECK(err == CA_MPS_ERR_INVALID,
                  "expected CA_MPS_ERR_INVALID, got %d", (int)err);
            moonlab_ca_mps_free(s);
        }
    }

    /* --- Case 6: through the var-D entry point with the new enum. -- */
    {
        z2_lgt_config_t cfg = {0};
        cfg.num_matter_sites = 4;
        cfg.t_hop = 1.0; cfg.h_link = 1.0; cfg.mass = 0.5;
        cfg.gauss_penalty = 5.0;
        uint8_t* paulis_h = NULL;
        double*  coeffs   = NULL;
        uint32_t T = 0, nq = 0;
        int rc = z2_lgt_1d_build_pauli_sum(&cfg, &paulis_h, &coeffs, &T, &nq);
        CHECK(rc == 0, "build Z2 LGT Pauli-sum (rc=%d)", rc);

        uint32_t k = cfg.num_matter_sites - 2;
        uint8_t* gens = (uint8_t*)calloc((size_t)k * nq, 1);
        for (uint32_t i = 0; i < k; i++) {
            z2_lgt_1d_gauss_law_pauli(&cfg, i + 1, &gens[(size_t)i * nq]);
        }

        moonlab_ca_mps_t* s = moonlab_ca_mps_create(nq, 32);
        ca_mps_var_d_alt_config_t acfg = ca_mps_var_d_alt_config_default();
        acfg.max_outer_iters             = 1;       /* warmstart-only smoke */
        acfg.imag_time_steps_per_outer   = 0;
        acfg.clifford_passes_per_outer   = 0;
        acfg.warmstart                   = CA_MPS_WARMSTART_STABILIZER_SUBGROUP;
        acfg.warmstart_stab_paulis       = gens;
        acfg.warmstart_stab_num_gens     = k;
        ca_mps_var_d_alt_result_t res = {0};
        ca_mps_error_t err = moonlab_ca_mps_optimize_var_d_alternating(
            s, paulis_h, coeffs, T, &acfg, &res);
        fprintf(stdout, "\n--- case 6: var-D entry, STABILIZER_SUBGROUP ---\n");
        CHECK(err == CA_MPS_SUCCESS, "var-D dispatch (got %d)", (int)err);
        for (uint32_t i = 0; i < k; i++) {
            double v = expect_pauli_string(s, &gens[(size_t)i * nq], nq);
            CHECK(fabs(v - 1.0) < 1e-5,
                  "case6 <G_%u> = %.9f", (unsigned)(i + 1), v);
        }

        moonlab_ca_mps_free(s);
        free(gens);
        free(paulis_h);
        free(coeffs);
    }

    if (failures == 0) {
        fprintf(stdout, "\nALL TESTS PASS\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d FAILURES\n", failures);
        return 1;
    }
}
