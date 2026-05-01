/**
 * @file ca_peps_2d_tfim.c
 * @brief CA-PEPS imag-time evolution on the 2D transverse-field Ising model.
 *
 * Hamiltonian on an Lx-by-Ly square lattice (open BC):
 *
 *     H = -J sum_{<ij>} Z_i Z_j  -  g sum_i X_i
 *
 * with linear index q = x + Lx*y.  Builds the Pauli sum, runs first-
 * order Trotter imag-time evolution
 *
 *     exp(-dtau H) ~ prod_k exp(-dtau h_k P_k)
 *
 * via CA-PEPS Pauli-rotation primitives, and reports converged
 * variational energy + half-cut entanglement entropy of the
 * underlying |phi> across a sweep of g.  Output: JSON with schema
 * "moonlab/ca_peps_2d_tfim_v1".
 *
 * Reproducibility: deterministic; no RNG.  Lattice and parameters
 * are CLI-overrideable but default to a paper-friendly 3x3 grid.
 */

#include "../../src/algorithms/tensor_network/ca_peps.h"

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Build the 2D TFIM Pauli sum.  Term order: ZZ bonds first (horizontal
 * row-major then vertical), then X transverse-field terms. */
static void build_tfim_paulis(uint32_t Lx, uint32_t Ly,
                              double J, double g,
                              uint8_t** out_paulis, double** out_coeffs,
                              uint32_t* out_nterms) {
    const uint32_t n = Lx * Ly;
    const uint32_t n_horiz = (Lx > 0 ? (Lx - 1) : 0) * Ly;
    const uint32_t n_vert  = Lx * (Ly > 0 ? (Ly - 1) : 0);
    const uint32_t n_field = n;
    const uint32_t T = n_horiz + n_vert + n_field;

    uint8_t* paulis = (uint8_t*)calloc((size_t)T * n, sizeof(uint8_t));
    double*  coeffs = (double*) calloc(T, sizeof(double));

    uint32_t k = 0;
    for (uint32_t y = 0; y < Ly; y++) {
        for (uint32_t x = 0; x + 1 < Lx; x++) {
            uint32_t i = x + Lx * y;
            uint32_t j = (x + 1) + Lx * y;
            paulis[(size_t)k * n + i] = 3;  /* Z */
            paulis[(size_t)k * n + j] = 3;
            coeffs[k] = -J;
            k++;
        }
    }
    for (uint32_t y = 0; y + 1 < Ly; y++) {
        for (uint32_t x = 0; x < Lx; x++) {
            uint32_t i = x + Lx * y;
            uint32_t j = x + Lx * (y + 1);
            paulis[(size_t)k * n + i] = 3;
            paulis[(size_t)k * n + j] = 3;
            coeffs[k] = -J;
            k++;
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        paulis[(size_t)k * n + i] = 1;  /* X */
        coeffs[k] = -g;
        k++;
    }

    *out_paulis = paulis;
    *out_coeffs = coeffs;
    *out_nterms = T;
}

/* Compute <H> via the underlying CA-MPS expectation-sum primitive. */
static double measure_energy(const moonlab_ca_peps_t* p,
                             const uint8_t* paulis,
                             const double* coeffs,
                             uint32_t T,
                             uint32_t n) {
    /* Pack real coeffs as complex doubles for the API. */
    double _Complex* cc = (double _Complex*)calloc(T, sizeof(double _Complex));
    for (uint32_t k = 0; k < T; k++) cc[k] = coeffs[k];
    double _Complex e = 0.0;
    moonlab_ca_peps_expect_pauli_sum(p, paulis, cc, T, &e);
    free(cc);
    (void)n;
    return creal(e);
}

static void run_one_g(FILE* json, int* first,
                      uint32_t Lx, uint32_t Ly,
                      double J, double g,
                      double dtau, uint32_t steps,
                      uint32_t chi) {
    uint8_t*  paulis;
    double*   coeffs;
    uint32_t  T;
    build_tfim_paulis(Lx, Ly, J, g, &paulis, &coeffs, &T);
    const uint32_t n = Lx * Ly;

    moonlab_ca_peps_t* p = moonlab_ca_peps_create(Lx, Ly, chi);
    /* Cold start in |0...0>; apply H on every site to get |+>^n,
     * which has <X>=+1 on every qubit (good initial overlap with
     * the deep-paramagnet ground state at large g). */
    for (uint32_t i = 0; i < n; i++) moonlab_ca_peps_h(p, i);

    double t0 = now_s();
    /* First-order Trotter: apply exp(-dtau h_k P_k) for each Pauli term
     * sequentially per step.  For X-only and ZZ terms the rotation is
     * exact; the only error is the Trotter splitting between non-
     * commuting H-pieces, which is O(dtau^2) per step. */
    for (uint32_t step = 0; step < steps; step++) {
        for (uint32_t k = 0; k < T; k++) {
            const uint8_t* row = paulis + (size_t)k * n;
            moonlab_ca_peps_imag_pauli_rotation(p, row, dtau * coeffs[k]);
            /* Wait -- imag_pauli_rotation applies exp(-tau P), so to
             * apply exp(-dtau h_k P_k) we want tau = dtau * h_k.  Done. */
        }
        /* Renormalise periodically; cheap and prevents underflow. */
        if (((step + 1) & 3) == 0) moonlab_ca_peps_normalize(p);
    }
    moonlab_ca_peps_normalize(p);
    double dt = now_s() - t0;

    double E = measure_energy(p, paulis, coeffs, T, n);
    double S_phi = moonlab_ca_peps_max_half_cut_entropy(p);
    uint32_t chi_now = moonlab_ca_peps_current_bond_dim(p);

    fprintf(stdout,
            "  g=%-5.2f  E=%12.6f  E/N=%9.5f  S(phi)=%6.4f  "
            "chi=%-3u  wall=%6.2fs\n",
            g, E, E / (double)n, S_phi, chi_now, dt);

    fprintf(json, "%s\n    {\"Lx\":%u,\"Ly\":%u,\"J\":%g,\"g\":%g,"
                  "\"dtau\":%g,\"steps\":%u,\"chi\":%u,"
                  "\"E\":%.10g,\"E_per_site\":%.10g,"
                  "\"S_phi\":%.6g,\"chi_now\":%u,\"wall_s\":%.4f}",
            *first ? "" : ",",
            Lx, Ly, J, g, dtau, steps, chi,
            E, E / (double)n, S_phi, chi_now, dt);
    *first = 0;

    moonlab_ca_peps_free(p);
    free(paulis);
    free(coeffs);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "ca_peps_2d_tfim.json";
    const uint32_t Lx = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 3;
    const uint32_t Ly = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 3;
    const uint32_t chi   = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 32;
    const double   dtau  = (argc >= 6) ? atof(argv[5]) : 0.05;
    const uint32_t steps = (argc >= 7) ? (uint32_t)atoi(argv[6]) : 80;

    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(stdout,
            "=== CA-PEPS 2D TFIM imag-time evolution ===\n"
            "  Lx=%u  Ly=%u  chi=%u  dtau=%g  steps=%u\n\n",
            Lx, Ly, chi, dtau, steps);

    fprintf(json,
            "{\n"
            "  \"schema\": \"moonlab/ca_peps_2d_tfim_v1\",\n"
            "  \"params\": {\"Lx\": %u, \"Ly\": %u, \"chi\": %u, "
            "\"dtau\": %g, \"steps\": %u},\n"
            "  \"points\": [",
            Lx, Ly, chi, dtau, steps);

    int first = 1;
    /* Sweep across the phase diagram.  J=1 throughout; g varies. */
    const double g_values[] = { 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 };
    const size_t n_g = sizeof(g_values) / sizeof(g_values[0]);
    for (size_t i = 0; i < n_g; i++) {
        run_one_g(json, &first, Lx, Ly, /*J=*/1.0, g_values[i],
                  dtau, steps, chi);
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nwrote %s\n", out_path);
    return 0;
}
