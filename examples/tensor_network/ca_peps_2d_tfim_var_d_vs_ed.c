/**
 * @file ca_peps_2d_tfim_var_d_vs_ed.c
 * @brief CA-PEPS variational-D ground-state search on 2D TFIM,
 *        cross-checked against dense ED.
 *
 * Companion to ca_peps_2d_tfim_vs_ed.c (which uses pure imag-time
 * Strang Trotter).  Here we run the alternating Clifford-disentangle
 * + imag-time loop @c moonlab_ca_peps_var_d_run, which absorbs the
 * Clifford-resolvable part of the entanglement into the prefactor D
 * and only pushes the genuinely non-Clifford residual into |phi>.
 *
 * Picks the warmstart that matches the regime:
 *   - g <= 1.0 (ferromagnetic side): FERRO_TFIM (H_0 + CNOT chain)
 *     prepares |GHZ>-like states the |0>^n -> ferro target.
 *   - g >  1.0 (paramagnetic side):  H_ALL prepares |+>^n.
 *
 * Output JSON: schema "moonlab/ca_peps_2d_tfim_var_d_vs_ed_v1",
 * tagging which warmstart was used per (Lx, Ly, g) point so the
 * paper can render a clean comparison.
 */

#include "../../src/algorithms/tensor_network/ca_peps.h"
#include "../../src/algorithms/vqe.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Warmstart codes (mirror moonlab_export.h public ABI). */
#define WARMSTART_IDENTITY  0
#define WARMSTART_H_ALL     1
#define WARMSTART_DUAL_TFIM 2
#define WARMSTART_FERRO_TFIM 3

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void build_tfim_bytes(uint32_t Lx, uint32_t Ly,
                              double J, double g,
                              uint8_t** out_paulis, double** out_coeffs,
                              uint32_t* out_T) {
    const uint32_t n = Lx * Ly;
    const uint32_t n_horiz = (Lx > 0 ? (Lx - 1) : 0) * Ly;
    const uint32_t n_vert  = Lx * (Ly > 0 ? (Ly - 1) : 0);
    const uint32_t T = n_horiz + n_vert + n;

    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, sizeof(uint8_t));
    double*  coeffs = (double*) calloc(T, sizeof(double));

    uint32_t k = 0;
    for (uint32_t y = 0; y < Ly; y++) {
        for (uint32_t x = 0; x + 1 < Lx; x++) {
            uint32_t i = x + Lx * y, j = (x + 1) + Lx * y;
            paulis[(size_t)k * n + i] = 3;
            paulis[(size_t)k * n + j] = 3;
            coeffs[k] = -J; k++;
        }
    }
    for (uint32_t y = 0; y + 1 < Ly; y++) {
        for (uint32_t x = 0; x < Lx; x++) {
            uint32_t i = x + Lx * y, j = x + Lx * (y + 1);
            paulis[(size_t)k * n + i] = 3;
            paulis[(size_t)k * n + j] = 3;
            coeffs[k] = -J; k++;
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        paulis[(size_t)k * n + i] = 1;
        coeffs[k] = -g; k++;
    }
    *out_paulis = paulis; *out_coeffs = coeffs; *out_T = T;
}

static pauli_hamiltonian_t* build_tfim_strings(uint32_t Lx, uint32_t Ly,
                                                double J, double g) {
    const uint32_t n = Lx * Ly;
    const uint32_t n_horiz = (Lx > 0 ? (Lx - 1) : 0) * Ly;
    const uint32_t n_vert  = Lx * (Ly > 0 ? (Ly - 1) : 0);
    const uint32_t T = n_horiz + n_vert + n;

    pauli_hamiltonian_t* H = pauli_hamiltonian_create(n, T);
    if (!H) return NULL;

    char* buf = (char*)malloc(n + 1);
    buf[n] = '\0';

    uint32_t k = 0;
    for (uint32_t y = 0; y < Ly; y++) {
        for (uint32_t x = 0; x + 1 < Lx; x++) {
            memset(buf, 'I', n);
            buf[x + Lx * y] = 'Z';
            buf[(x + 1) + Lx * y] = 'Z';
            pauli_hamiltonian_add_term(H, -J, buf, k++);
        }
    }
    for (uint32_t y = 0; y + 1 < Ly; y++) {
        for (uint32_t x = 0; x < Lx; x++) {
            memset(buf, 'I', n);
            buf[x + Lx * y] = 'Z';
            buf[x + Lx * (y + 1)] = 'Z';
            pauli_hamiltonian_add_term(H, -J, buf, k++);
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        memset(buf, 'I', n);
        buf[i] = 'X';
        pauli_hamiltonian_add_term(H, -g, buf, k++);
    }
    free(buf);
    return H;
}

/* Run the var-D alternating loop on the given Hamiltonian and pick the
 * warmstart based on the field strength. */
static int run_var_d(uint32_t Lx, uint32_t Ly, double J, double g,
                     uint32_t chi,
                     uint32_t outer_iters, double dtau,
                     uint32_t inner_imag_steps, uint32_t cliff_passes,
                     int composite_2gate,
                     double* out_E,
                     double* out_S_phi,
                     uint32_t* out_chi_now,
                     int* out_warmstart_used) {
    uint8_t* paulis;
    double*  coeffs;
    uint32_t T;
    build_tfim_bytes(Lx, Ly, J, g, &paulis, &coeffs, &T);

    moonlab_ca_peps_t* p = moonlab_ca_peps_create(Lx, Ly, chi);

    /* g <= 1: ferromagnetic side -> FERRO warmstart (|+,+,...> after
     * H+CNOT-chain absorbs the Z2 cat structure into D, leaving the
     * MPS factor as a near-product state that imag-time can then
     * polish).  g > 1: paramagnetic side -> H_ALL absorbs the |+>^n
     * structure into D so |phi> stays close to |0>^n. */
    int warm = (g <= 1.0) ? WARMSTART_FERRO_TFIM : WARMSTART_H_ALL;

    double E = 0.0;
    int rc = moonlab_ca_peps_var_d_run(
        p, paulis, coeffs, T,
        outer_iters, dtau, inner_imag_steps, cliff_passes,
        composite_2gate, warm,
        /*stab_paulis=*/NULL, /*stab_num_gens=*/0,
        &E);

    if (rc == 0) {
        *out_E = E;
        *out_S_phi = moonlab_ca_peps_max_half_cut_entropy(p);
        *out_chi_now = moonlab_ca_peps_current_bond_dim(p);
        *out_warmstart_used = warm;
    }

    moonlab_ca_peps_free(p);
    free(paulis); free(coeffs);
    return rc;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "ca_peps_2d_tfim_var_d_vs_ed.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout,
            "=== CA-PEPS variational-D 2D TFIM vs dense ED ===\n"
            "  schema: moonlab/ca_peps_2d_tfim_var_d_vs_ed_v1\n\n");

    fprintf(json,
            "{\n"
            "  \"schema\": \"moonlab/ca_peps_2d_tfim_var_d_vs_ed_v1\",\n"
            "  \"description\": \"CA-PEPS variational-D vs dense-ED ground-state "
            "energies for 2D TFIM H = -J sum<ij> Z_i Z_j - g sum_i X_i\",\n"
            "  \"warmstart_codes\": {\"0\": \"IDENTITY\", \"1\": \"H_ALL\", "
            "\"2\": \"DUAL_TFIM\", \"3\": \"FERRO_TFIM\"},\n"
            "  \"points\": [");

    typedef struct {
        uint32_t Lx, Ly, chi;
        uint32_t outer; double dtau;
        uint32_t inner; uint32_t passes;
        int composite;
    } sweep_t;
    /* var-D parameters tuned for paper-friendly wall-clock.  Per-pass
     * cost is O(num_candidates * num_terms * n * chi^2) where
     * num_candidates ~ 6n (singles + 2q) or ~9 n^2 (composite).
     * Candidate evaluations now run in parallel via OpenMP, but the
     * inner imag-time evolution is single-threaded; combined cost on
     * n=9, 12 lattices runs into many minutes per g, beyond the paper-
     * grade single-host budget.  We keep this bench at 2x2 / 3x2 where
     * var-D + Strang gives O(1e-7) parity and 14-30x entropy reduction
     * vs |psi> in single-digit seconds.  The Strang-only imag-time
     * bench in ca_peps_2d_tfim_vs_ed.c covers the 3x3 / 4x3 parity. */
    const sweep_t sweeps[] = {
        { 2, 2,  8, 20, 0.1, 3, 3, 1 },
        { 3, 2, 16, 12, 0.1, 3, 2, 1 },
    };
    const size_t n_sweeps = sizeof(sweeps) / sizeof(sweeps[0]);
    const double g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 };
    const size_t n_g = sizeof(g_values) / sizeof(g_values[0]);

    int first = 1;
    for (size_t s = 0; s < n_sweeps; s++) {
        const uint32_t Lx = sweeps[s].Lx, Ly = sweeps[s].Ly;
        const uint32_t n = Lx * Ly;
        const uint32_t chi = sweeps[s].chi;

        fprintf(stdout,
                "  Lx=%u Ly=%u (n=%u) chi=%u outer=%u dtau=%g inner=%u passes=%u\n",
                Lx, Ly, n, chi, sweeps[s].outer, sweeps[s].dtau,
                sweeps[s].inner, sweeps[s].passes);
        fprintf(stdout, "  %-6s %-7s %12s %12s %12s %10s %8s %8s %8s\n",
                "g", "warm", "E_varD", "E_ED", "abs_err", "rel_err",
                "S(phi)", "chi", "wall_s");

        for (size_t i = 0; i < n_g; i++) {
            const double g = g_values[i];

            double E_ca = 0, S_phi = 0;
            uint32_t chi_now = 0;
            int warm = -1;

            const double t0 = now_s();
            int rc = run_var_d(Lx, Ly, 1.0, g, chi,
                               sweeps[s].outer, sweeps[s].dtau,
                               sweeps[s].inner, sweeps[s].passes,
                               sweeps[s].composite,
                               &E_ca, &S_phi, &chi_now, &warm);
            const double dt = now_s() - t0;

            pauli_hamiltonian_t* H = build_tfim_strings(Lx, Ly, 1.0, g);
            const double E_ed = vqe_exact_ground_state_energy(H);
            pauli_hamiltonian_free(H);

            const double abs_err = fabs(E_ca - E_ed);
            const double rel_err = abs_err / (fabs(E_ed) > 1e-12 ? fabs(E_ed) : 1.0);

            const char* warm_name = (warm == WARMSTART_FERRO_TFIM) ? "ferro"
                                  : (warm == WARMSTART_H_ALL)     ? "H_all"
                                  : (warm == WARMSTART_DUAL_TFIM) ? "dual"
                                  : "I";

            fprintf(stdout,
                    "  %-6.2f %-7s %12.6f %12.6f %12.3e %10.3e %8.4f %8u %8.2f%s\n",
                    g, warm_name, E_ca, E_ed, abs_err, rel_err, S_phi, chi_now, dt,
                    rc == 0 ? "" : "  [FAIL]");

            fprintf(json, "%s\n    {\"Lx\":%u,\"Ly\":%u,\"n\":%u,"
                          "\"chi\":%u,\"outer\":%u,\"dtau\":%g,"
                          "\"inner_imag_steps\":%u,\"cliff_passes\":%u,"
                          "\"composite_2gate\":%d,"
                          "\"J\":1.0,\"g\":%g,\"warmstart\":%d,"
                          "\"E_varD\":%.10g,\"E_ED\":%.10g,"
                          "\"abs_err\":%.6e,\"rel_err\":%.6e,"
                          "\"S_phi\":%.6g,\"chi_now\":%u,"
                          "\"wall_s\":%.4f,\"rc\":%d}",
                    first ? "" : ",",
                    Lx, Ly, n, chi, sweeps[s].outer, sweeps[s].dtau,
                    sweeps[s].inner, sweeps[s].passes, sweeps[s].composite,
                    g, warm, E_ca, E_ed, abs_err, rel_err,
                    S_phi, chi_now, dt, rc);
            first = 0;
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "wrote %s\n", out_path);
    return 0;
}
