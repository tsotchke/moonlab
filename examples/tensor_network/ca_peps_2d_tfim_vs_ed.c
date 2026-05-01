/**
 * @file ca_peps_2d_tfim_vs_ed.c
 * @brief CA-PEPS imag-time evolution on 2D TFIM cross-checked against
 *        dense exact diagonalisation.
 *
 * Builds the 2D transverse-field Ising Hamiltonian
 *
 *     H = -J sum_{<ij>} Z_i Z_j  -  g sum_i X_i
 *
 * on small Lx-by-Ly square lattices (open BC), runs first-order
 * Trotter imag-time evolution through CA-PEPS, and compares the
 * converged variational energy against
 * @c vqe_exact_ground_state_energy (shifted power iteration on the
 * full 2^N x 2^N Hamiltonian matrix).  The lattice cap is 12 qubits
 * (4x3) since the dense ED is O(4^N).
 *
 * Output: JSON with schema "moonlab/ca_peps_2d_tfim_vs_ed_v1",
 * recording per-(Lx, Ly, g) tuples the imag-time energy, ED energy,
 * absolute error |E_ca - E_ed|, and relative error.
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

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Build a TFIM Pauli sum as a length-T uint8 array (CA-PEPS form). */
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

/* Same Hamiltonian as a string-encoded pauli_hamiltonian_t for ED.
 * pauli_string[q] = Pauli on linear-index qubit q. */
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

static double measure_energy_peps(const moonlab_ca_peps_t* p,
                                   const uint8_t* paulis,
                                   const double* coeffs,
                                   uint32_t T) {
    double _Complex* cc = (double _Complex*)calloc(T, sizeof(double _Complex));
    for (uint32_t k = 0; k < T; k++) cc[k] = coeffs[k];
    double _Complex e = 0.0;
    moonlab_ca_peps_expect_pauli_sum(p, paulis, cc, T, &e);
    free(cc);
    return creal(e);
}

static double run_imag_time(uint32_t Lx, uint32_t Ly, double J, double g,
                             double dtau, uint32_t steps, uint32_t chi) {
    uint8_t* paulis;
    double* coeffs;
    uint32_t T;
    build_tfim_bytes(Lx, Ly, J, g, &paulis, &coeffs, &T);
    const uint32_t n = Lx * Ly;

    moonlab_ca_peps_t* p = moonlab_ca_peps_create(Lx, Ly, chi);
    /* |+>^n initial state: best overlap with deep-paramagnet, modest with ferro. */
    for (uint32_t i = 0; i < n; i++) moonlab_ca_peps_h(p, i);

    for (uint32_t step = 0; step < steps; step++) {
        for (uint32_t k = 0; k < T; k++) {
            moonlab_ca_peps_imag_pauli_rotation(
                p, paulis + (size_t)k * n, dtau * coeffs[k]);
        }
        if (((step + 1) & 3) == 0) moonlab_ca_peps_normalize(p);
    }
    moonlab_ca_peps_normalize(p);

    double E = measure_energy_peps(p, paulis, coeffs, T);

    moonlab_ca_peps_free(p);
    free(paulis); free(coeffs);
    return E;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "ca_peps_2d_tfim_vs_ed.json";
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout,
            "=== CA-PEPS 2D TFIM vs dense ED parity ===\n"
            "  schema: moonlab/ca_peps_2d_tfim_vs_ed_v1\n\n");

    fprintf(json,
            "{\n"
            "  \"schema\": \"moonlab/ca_peps_2d_tfim_vs_ed_v1\",\n"
            "  \"description\": \"CA-PEPS imag-time vs dense-ED ground-state "
            "energies for 2D TFIM H = -J sum<ij> Z_i Z_j - g sum_i X_i\",\n"
            "  \"points\": [");

    /* Lattice sizes within the 12-qubit dense-ED cap.  4x3 (12q) is at
     * the cap; 3x3 (9q) is the paper-friendly point; 2x2 / 2x3 are
     * sanity checks. */
    typedef struct { uint32_t Lx, Ly; uint32_t chi; double dtau; uint32_t steps; } sweep_t;
    const sweep_t sweeps[] = {
        { 2, 2,  8, 0.05,  80 },
        { 3, 2, 16, 0.05, 100 },
        { 3, 3, 32, 0.04, 150 },
        { 4, 3, 48, 0.03, 200 },
    };
    const size_t n_sweeps = sizeof(sweeps) / sizeof(sweeps[0]);
    const double g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 };
    const size_t n_g = sizeof(g_values) / sizeof(g_values[0]);

    int first = 1;
    for (size_t s = 0; s < n_sweeps; s++) {
        const uint32_t Lx = sweeps[s].Lx;
        const uint32_t Ly = sweeps[s].Ly;
        const uint32_t n  = Lx * Ly;
        const uint32_t chi = sweeps[s].chi;
        const double dtau = sweeps[s].dtau;
        const uint32_t steps = sweeps[s].steps;

        fprintf(stdout, "  Lx=%u Ly=%u (n=%u qubits) chi=%u dtau=%g steps=%u\n",
                Lx, Ly, n, chi, dtau, steps);
        fprintf(stdout, "  %-6s %12s %12s %12s %10s %10s\n",
                "g", "E_imag", "E_ED", "abs_err", "rel_err", "wall_s");

        for (size_t i = 0; i < n_g; i++) {
            const double g = g_values[i];

            const double t0 = now_s();
            const double E_ca = run_imag_time(Lx, Ly, /*J=*/1.0, g,
                                               dtau, steps, chi);
            const double dt_ca = now_s() - t0;

            pauli_hamiltonian_t* H = build_tfim_strings(Lx, Ly, 1.0, g);
            const double E_ed = vqe_exact_ground_state_energy(H);
            pauli_hamiltonian_free(H);

            const double abs_err = fabs(E_ca - E_ed);
            const double rel_err = abs_err / (fabs(E_ed) > 1e-12 ? fabs(E_ed) : 1.0);

            fprintf(stdout, "  %-6.2f %12.6f %12.6f %12.3e %10.3e %10.2f\n",
                    g, E_ca, E_ed, abs_err, rel_err, dt_ca);

            fprintf(json, "%s\n    {\"Lx\":%u,\"Ly\":%u,\"n\":%u,"
                          "\"chi\":%u,\"dtau\":%g,\"steps\":%u,"
                          "\"J\":1.0,\"g\":%g,"
                          "\"E_imag\":%.10g,\"E_ED\":%.10g,"
                          "\"abs_err\":%.6e,\"rel_err\":%.6e,"
                          "\"wall_s_imag\":%.4f}",
                    first ? "" : ",",
                    Lx, Ly, n, chi, dtau, steps,
                    g, E_ca, E_ed, abs_err, rel_err, dt_ca);
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
