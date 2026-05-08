/**
 * @file bench_cross_backend_tfim.c
 * @brief Cross-backend validation: same observable via state-vector ED
 *        vs MPS DMRG vs CA-MPS var-D, on identical 1D TFIM Hamiltonian.
 *
 * Closes the moonlab paper §4 cross-validation audit point: the paper
 * claims to function as a "cross-validation reference" but never runs
 * the same physical observable through more than one backend on the
 * same model.  This harness fixes that.
 *
 * Model: 1D transverse-field Ising chain with open boundary conditions,
 *   H = -J sum_<ij> Z_i Z_j  -  g sum_i X_i
 * with J = 1.  Sweep g across {0.25, 0.5, 1.0, 1.5, 2.5}.
 *
 * Three engines:
 *   - exact_ed:   vqe_exact_ground_state_energy on the dense Pauli sum
 *                 (4^n diagonalisation).
 *   - mps_dmrg:   moonlab_dmrg_tfim_energy with chi=64, 8 sweeps.
 *   - ca_mps_vd:  CA-MPS var-D alternating loop (Pauli-rotation form).
 *
 * Reports E_ED, E_DMRG, E_varD, |E_DMRG - E_ED| / |E_ED|, and
 * |E_varD - E_ED| / |E_ED| in a single JSON.  The paper pulls this
 * directly into the cross-validation table.
 *
 * Output schema: "moonlab/cross_backend_tfim_v1".
 */

#include "../../src/applications/moonlab_export.h"
#include "../../src/algorithms/vqe.h"
#include "../../src/algorithms/tensor_network/ca_mps.h"

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

/* Build the TFIM Pauli sum for vqe_exact_ground_state_energy. */
static pauli_hamiltonian_t* build_tfim_paulis(int n, double J, double g) {
    size_t n_terms = (size_t)((n - 1) + n);
    pauli_hamiltonian_t* H =
        pauli_hamiltonian_create((size_t)n, n_terms);
    size_t idx = 0;
    /* -J Z_i Z_{i+1} for i in [0, n-1). */
    for (int i = 0; i + 1 < n; i++) {
        char* op = (char*)calloc((size_t)n + 1, 1);
        for (int q = 0; q < n; q++) op[q] = 'I';
        op[i]     = 'Z';
        op[i + 1] = 'Z';
        pauli_hamiltonian_add_term(H, -J, op, idx++);
        free(op);
    }
    /* -g X_i for i in [0, n). */
    for (int i = 0; i < n; i++) {
        char* op = (char*)calloc((size_t)n + 1, 1);
        for (int q = 0; q < n; q++) op[q] = 'I';
        op[i] = 'X';
        pauli_hamiltonian_add_term(H, -g, op, idx++);
        free(op);
    }
    return H;
}

/* Build the TFIM Pauli sum in the (paulis, coeffs) byte/double form
 * that moonlab_ca_mps_var_d_run consumes. */
static void build_tfim_byte_paulis(int n, double J, double g,
                                     uint8_t** out_paulis, double** out_coeffs,
                                     uint32_t* out_num_terms) {
    int n_terms = (n - 1) + n;
    uint8_t* p = (uint8_t*)calloc((size_t)n_terms * (size_t)n, 1);
    double*  c = (double*)calloc((size_t)n_terms, sizeof(double));
    int idx = 0;
    for (int i = 0; i + 1 < n; i++) {
        for (int q = 0; q < n; q++) p[(size_t)idx * n + q] = 0;  /* I */
        p[(size_t)idx * n + i]     = 3;  /* Z */
        p[(size_t)idx * n + i + 1] = 3;
        c[idx] = -J;
        idx++;
    }
    for (int i = 0; i < n; i++) {
        for (int q = 0; q < n; q++) p[(size_t)idx * n + q] = 0;
        p[(size_t)idx * n + i] = 1;  /* X */
        c[idx] = -g;
        idx++;
    }
    *out_paulis = p; *out_coeffs = c; *out_num_terms = (uint32_t)n_terms;
}

typedef struct {
    int n;
    double J, g;
    double E_ED;
    double E_DMRG;
    double E_varD;
    double rel_DMRG;
    double rel_varD;
    double t_ED, t_DMRG, t_varD;
} row_t;

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "cross_backend_tfim.json";
    const int n = (argc >= 3) ? atoi(argv[2]) : 8;

    const double J = 1.0;
    const double gs[] = { 0.25, 0.5, 1.0, 1.5, 2.5 };
    const size_t n_g = sizeof(gs) / sizeof(gs[0]);

    fprintf(stdout, "=== Cross-backend TFIM ground-state validation ===\n");
    fprintf(stdout, "  n=%d, J=%g, OBC, sweep over g\n\n", n, J);
    fprintf(stdout,
            "  %-5s %-12s %-12s %-12s %-12s %-12s\n",
            "g", "E_ED", "E_DMRG", "E_varD", "|dE_DMRG|/|E|", "|dE_varD|/|E|");

    row_t rows[8];

    for (size_t gi = 0; gi < n_g; gi++) {
        const double g = gs[gi];

        /* (1) ED. */
        double t0 = now_s();
        pauli_hamiltonian_t* H = build_tfim_paulis(n, J, g);
        double E_ED = vqe_exact_ground_state_energy(H);
        pauli_hamiltonian_free(H);
        double t_ED = now_s() - t0;

        /* (2) DMRG. */
        t0 = now_s();
        double E_DMRG = moonlab_dmrg_tfim_energy((uint32_t)n, g, 64, 8);
        double t_DMRG = now_s() - t0;

        /* (3) CA-MPS var-D. */
        t0 = now_s();
        moonlab_ca_mps_t* state = moonlab_ca_mps_create((uint32_t)n, 32);
        uint8_t* p; double* c; uint32_t nt;
        build_tfim_byte_paulis(n, J, g, &p, &c, &nt);
        double E_varD = 0.0;
        int rc = moonlab_ca_mps_var_d_run(
            state, p, c, nt,
            /*outer_iters*/ 12,
            /*dtau*/ 0.10,
            /*imag_steps*/ 4,
            /*cliff_passes*/ 2,
            /*composite_2gate*/ 1,
            /*warmstart*/ (g <= 1.0) ? 3 : 1,
            NULL, 0,
            &E_varD);
        moonlab_ca_mps_free(state);
        free(p); free(c);
        double t_varD = now_s() - t0;
        if (rc != 0) E_varD = nan("");

        rows[gi].n = n;
        rows[gi].J = J;
        rows[gi].g = g;
        rows[gi].E_ED   = E_ED;
        rows[gi].E_DMRG = E_DMRG;
        rows[gi].E_varD = E_varD;
        rows[gi].rel_DMRG = fabs(E_DMRG - E_ED) / fabs(E_ED);
        rows[gi].rel_varD = fabs(E_varD - E_ED) / fabs(E_ED);
        rows[gi].t_ED = t_ED;
        rows[gi].t_DMRG = t_DMRG;
        rows[gi].t_varD = t_varD;
        fprintf(stdout, "  %-5.2f %-12.6f %-12.6f %-12.6f %-12.3e %-12.3e\n",
                g, E_ED, E_DMRG, E_varD,
                rows[gi].rel_DMRG, rows[gi].rel_varD);
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/cross_backend_tfim_v1\",\n");
    fprintf(f, "  \"description\": \"Cross-backend TFIM ground-state "
               "validation: same Hamiltonian H = -J sum<ij> Z_i Z_{i+1} "
               "- g sum_i X_i computed by three independent Moonlab "
               "engines (state-vector dense ED via vqe_exact_ground_state_"
               "energy; MPS DMRG via moonlab_dmrg_tfim_energy; CA-MPS "
               "var-D via moonlab_ca_mps_var_d_run).  Reports |dE|/|E| "
               "for each engine relative to ED.\",\n");
    fprintf(f, "  \"params\": {\"n\": %d, \"J\": %.6f, "
               "\"obc\": true, \"dmrg_chi\": 64, \"dmrg_sweeps\": 8, "
               "\"ca_mps_chi\": 32, \"var_d_outer_iters\": 12, "
               "\"var_d_dtau\": 0.10, \"var_d_imag_steps\": 4, "
               "\"var_d_cliff_passes\": 2},\n", n, J);
    fprintf(f, "  \"rows\": [");
    for (size_t i = 0; i < n_g; i++) {
        fprintf(f, "%s\n    {\"n\": %d, \"J\": %.6f, \"g\": %.6f, "
                   "\"E_ED\": %.10f, \"E_DMRG\": %.10f, \"E_varD\": %.10f, "
                   "\"rel_err_DMRG\": %.6e, \"rel_err_varD\": %.6e, "
                   "\"t_ED\": %.4f, \"t_DMRG\": %.4f, \"t_varD\": %.4f}",
                i == 0 ? "" : ",",
                rows[i].n, rows[i].J, rows[i].g,
                rows[i].E_ED, rows[i].E_DMRG, rows[i].E_varD,
                rows[i].rel_DMRG, rows[i].rel_varD,
                rows[i].t_ED, rows[i].t_DMRG, rows[i].t_varD);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stdout, "\nwrote %s\n", out_path);
    return 0;
}
