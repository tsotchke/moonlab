/**
 * @file bench_cross_backend_kagome12.c
 * @brief Cross-backend ground-state validation on the 12-site kagome
 *        AFM Heisenberg cluster (the frustrated-magnet ED reference).
 *
 * Companion to bench_cross_backend_tfim.c (gentle test, var-D works)
 * and bench_cross_backend_xxz.c (SU(2)-symmetric, var-D underperforms).
 * The kagome 2x2 cluster (n=12, 24 NN bonds, p6mm symmetry) is the
 * literature-anchored frustrated-magnet test that pins the result
 * against an external reference.
 *
 * Three engines:
 *   - ED via vqe_exact_ground_state_energy (Hermitian eigendecomp on
 *     the dense 4096x4096 Pauli sum).
 *   - DMRG via dmrg_ground_state on the kagome MPO from
 *     mpo_2d_heisenberg_dmi_create.
 *   - var-D via moonlab_ca_mps_var_d_run on the byte-encoded Pauli sum.
 *
 * Reference: Läuchli-Sudan-Sørensen PRB 83, 212401 (2011) Table I
 * cluster "12" (rectangular 2x2 torus): E_0 = -5.444875216 J in spin
 * convention.  Moonlab's Pauli-operator Heisenberg builder uses
 * H_Pauli = J' sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j) which equals
 * 4 J' sum S_i.S_j; passing J' = 0.25 reproduces the J = 1 spin
 * value E_0 = -5.44487522.
 *
 * Output schema "moonlab/cross_backend_kagome12_v1".
 */

#include "../../src/applications/moonlab_export.h"
#include "../../src/algorithms/vqe.h"
#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/lattice_2d.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"

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

#define LIBIRREP_KAGOME12_E0  (-5.44487522)
#define J_PAULI                0.25  /* Pauli convention matching J_spin = 1 */

/* Collect i<j unique bonds from a lattice_2d, dedup'd against PBC
 * doubling on small Ly. */
static uint32_t collect_unique_bonds(const lattice_2d_t* lat,
                                       uint32_t (*bonds)[2]) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < lat->num_sites; i++) {
        for (uint32_t k = 0; k < lat->num_neighbors[i]; k++) {
            uint32_t j = lat->neighbors[i][k].site;
            if (j <= i) continue;
            int dup = 0;
            for (uint32_t m = 0; m < n; m++) {
                if (bonds[m][0] == i && bonds[m][1] == j) { dup = 1; break; }
            }
            if (dup) continue;
            bonds[n][0] = i; bonds[n][1] = j;
            n++;
        }
    }
    return n;
}

/* Build Pauli-string sum for vqe_exact_ground_state_energy. */
static pauli_hamiltonian_t* build_kagome_paulis(uint32_t n,
                                                  const uint32_t (*bonds)[2],
                                                  uint32_t num_bonds) {
    size_t n_terms = (size_t)num_bonds * 3;
    pauli_hamiltonian_t* H =
        pauli_hamiltonian_create((size_t)n, n_terms);
    size_t idx = 0;
    char* op = (char*)calloc((size_t)n + 1, 1);
    const char letters[3] = {'X', 'Y', 'Z'};
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (int k = 0; k < 3; k++) {
            for (uint32_t q = 0; q < n; q++) op[q] = 'I';
            op[i] = letters[k];
            op[j] = letters[k];
            pauli_hamiltonian_add_term(H, J_PAULI, op, idx++);
        }
    }
    free(op);
    return H;
}

/* Build byte-encoded Pauli sum for moonlab_ca_mps_var_d_run.
 * 1=X, 2=Y, 3=Z (0=I). */
static void build_kagome_byte_paulis(uint32_t n,
                                       const uint32_t (*bonds)[2],
                                       uint32_t num_bonds,
                                       uint8_t** out_paulis,
                                       double**  out_coeffs,
                                       uint32_t* out_num_terms) {
    uint32_t nt = num_bonds * 3;
    uint8_t* p = (uint8_t*)calloc((size_t)nt * (size_t)n, 1);
    double*  c = (double*)calloc((size_t)nt, sizeof(double));
    uint32_t idx = 0;
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (uint8_t k = 1; k <= 3; k++) {
            p[(size_t)idx * n + i] = k;
            p[(size_t)idx * n + j] = k;
            c[idx] = J_PAULI;
            idx++;
        }
    }
    *out_paulis = p; *out_coeffs = c; *out_num_terms = nt;
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1]
                                        : "cross_backend_kagome12.json";

    fprintf(stdout, "=== Cross-backend kagome 12-site Heisenberg ===\n");
    fprintf(stdout, "  reference (LSS / libirrep): E_0 = %.8f J\n\n",
            LIBIRREP_KAGOME12_E0);

    /* Lattice + bonds. */
    const uint32_t Lx = 6, Ly = 2;  /* 6x2 = 12 sites with c.x%3 sublattice. */
    lattice_2d_t* lat = lattice_2d_create(Lx, Ly, LATTICE_KAGOME, BC_PERIODIC_XY);
    if (!lat || lat->num_sites != 12) {
        fprintf(stderr, "lattice_2d_create failed\n"); return 1;
    }
    const uint32_t n = lat->num_sites;
    const size_t max_bonds = (size_t)lat->max_neighbors * (size_t)n / 2;
    uint32_t (*bonds)[2] = (uint32_t (*)[2])calloc(max_bonds, sizeof(uint32_t[2]));
    uint32_t num_bonds = collect_unique_bonds(lat, bonds);
    fprintf(stdout, "  lattice: kagome 6x2 PBC, n=%u, %u unique NN bonds\n",
            n, num_bonds);

    /* (1) ED via vqe_exact_ground_state_energy. */
    fprintf(stdout, "\n[1] dense ED via Hermitian eigendecomp ...\n");
    double t0 = now_s();
    pauli_hamiltonian_t* H_pauli = build_kagome_paulis(n, bonds, num_bonds);
    double E_ED = vqe_exact_ground_state_energy(H_pauli);
    pauli_hamiltonian_free(H_pauli);
    double t_ED = now_s() - t0;
    fprintf(stdout, "  E_ED = %.8f  (wall %.2f s)\n", E_ED, t_ED);

    /* (2) DMRG via dmrg_ground_state on the MPO. */
    fprintf(stdout, "\n[2] DMRG via dmrg_ground_state ...\n");
    t0 = now_s();
    double bond_vectors_unused[64][3] = {{0}};
    mpo_t* H_mpo = mpo_2d_heisenberg_dmi_create(n, num_bonds, bonds,
                                                  bond_vectors_unused,
                                                  /*J*/ J_PAULI, /*D*/ 0.0,
                                                  /*B*/ 0.0,     /*K*/ 0.0);
    if (!H_mpo) {
        fprintf(stderr, "mpo_2d_heisenberg_dmi_create failed\n");
        return 1;
    }
    /* The BLAS-backed environment contractor (v0.2.4) makes chi=128
     * tractable on the wide kagome 2D MPO -- before the rewrite the
     * 7-deep nested loop made even chi=64 hours-long.  Now the GS
     * converges to ED-equivalent precision in seconds. */
    tn_mps_state_t* mps = dmrg_init_random_mps(n, /*chi_init*/ 32, NULL);
    if (!mps) { fprintf(stderr, "dmrg_init_random_mps failed\n"); return 1; }
    dmrg_config_t dmrg_cfg = dmrg_config_default();
    dmrg_cfg.max_bond_dim = 128;
    dmrg_cfg.max_sweeps   = 20;
    dmrg_cfg.energy_tol   = 1e-9;
    dmrg_result_t* dmrg_res = dmrg_ground_state(mps, H_mpo, &dmrg_cfg);
    double E_DMRG = (dmrg_res ? dmrg_res->ground_energy : NAN);
    if (dmrg_res) dmrg_result_free(dmrg_res);
    tn_mps_free(mps);
    mpo_free(H_mpo);
    double t_DMRG = now_s() - t0;
    fprintf(stdout, "  E_DMRG = %.8f  (wall %.2f s)\n", E_DMRG, t_DMRG);

    /* (3) var-D via moonlab_ca_mps_var_d_run.  Tightening
     * convergence_eps via var_d_run_v2 was tried (60 outer / 10 imag
     * / chi=64 / eps=1e-12) and reproduces the same 21%% residual at
     * 4-5x wall clock -- the kagome 12 frustrated AFM ground state
     * lives in the same symmetry-protected fixed point as XXZ
     * Delta=1, where the alternating loop's local-Clifford search
     * cannot escape regardless of eps.  Same v0.3 escape mechanisms
     * apply (2-site Gibbs TEBD or symmetry-aware warmstart). */
    fprintf(stdout, "\n[3] CA-MPS var-D ...\n");
    t0 = now_s();
    moonlab_ca_mps_t* state = moonlab_ca_mps_create(n, 32);
    uint8_t* p; double* c; uint32_t nt;
    build_kagome_byte_paulis(n, bonds, num_bonds, &p, &c, &nt);
    double E_varD = 0.0;
    int rc = moonlab_ca_mps_var_d_run(
        state, p, c, nt,
        /*outer_iters*/ 30,
        /*dtau*/ 0.10,
        /*imag_steps*/ 5,
        /*cliff_passes*/ 1,
        /*composite_2gate*/ 0,
        /*warmstart*/ 1,    /* H_ALL */
        NULL, 0,
        &E_varD);
    moonlab_ca_mps_free(state);
    free(p); free(c);
    double t_varD = now_s() - t0;
    if (rc != 0) E_varD = NAN;
    fprintf(stdout, "  E_varD = %.8f  (wall %.2f s)  rc=%d\n",
            E_varD, t_varD, rc);

    free(bonds);
    lattice_2d_free(lat);

    /* Compute relative errors against ED + against the libirrep reference. */
    const double rel_DMRG = fabs(E_DMRG - E_ED) / fabs(E_ED);
    const double rel_varD = fabs(E_varD - E_ED) / fabs(E_ED);
    const double rel_libirrep_ED   = fabs(E_ED   - LIBIRREP_KAGOME12_E0)
                                       / fabs(LIBIRREP_KAGOME12_E0);
    const double rel_libirrep_DMRG = fabs(E_DMRG - LIBIRREP_KAGOME12_E0)
                                       / fabs(LIBIRREP_KAGOME12_E0);

    fprintf(stdout, "\n=== Summary ===\n");
    fprintf(stdout, "  E_ED       = %+.10f   (vs libirrep -5.44487522, rel %.3e)\n",
            E_ED, rel_libirrep_ED);
    fprintf(stdout, "  E_DMRG     = %+.10f   (vs ED, rel %.3e; vs libirrep, rel %.3e)\n",
            E_DMRG, rel_DMRG, rel_libirrep_DMRG);
    fprintf(stdout, "  E_varD     = %+.10f   (vs ED, rel %.3e)\n",
            E_varD, rel_varD);

    /* JSON archive. */
    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/cross_backend_kagome12_v1\",\n");
    fprintf(f, "  \"description\": \"Cross-backend kagome-12 AFM Heisenberg "
               "ground-state validation: ED + DMRG + CA-MPS var-D on the "
               "same Pauli-operator Hamiltonian.  Lattice is the standard "
               "p6mm 2x2 kagome torus (12 sites, 24 NN bonds).  J_Pauli = "
               "0.25 reproduces the libirrep / Lauchli-Sudan-Sorensen PRB "
               "83, 212401 cluster-12 reference E_0 = -5.44487522.\",\n");
    fprintf(f, "  \"params\": {\"n\": %u, \"num_bonds\": %u, "
               "\"J_pauli\": %.6f, \"libirrep_E0\": %.8f, "
               "\"dmrg_chi\": %u, \"dmrg_sweeps\": %u, "
               "\"ca_mps_chi\": 32, \"var_d_outer_iters\": 30, "
               "\"var_d_dtau\": 0.10, \"var_d_imag_steps\": 5, "
               "\"var_d_warmstart\": \"H_ALL\"},\n",
            n, num_bonds, J_PAULI, LIBIRREP_KAGOME12_E0,
            dmrg_cfg.max_bond_dim, dmrg_cfg.max_sweeps);
    fprintf(f, "  \"row\": {\"E_ED\": %.10f, \"E_DMRG\": %.10f, "
                "\"E_varD\": %.10f, \"E_libirrep\": %.8f, "
                "\"rel_err_DMRG_vs_ED\": %.6e, "
                "\"rel_err_varD_vs_ED\": %.6e, "
                "\"rel_err_ED_vs_libirrep\": %.6e, "
                "\"rel_err_DMRG_vs_libirrep\": %.6e, "
                "\"t_ED\": %.4f, \"t_DMRG\": %.4f, \"t_varD\": %.4f}\n",
            E_ED, E_DMRG, E_varD, LIBIRREP_KAGOME12_E0,
            rel_DMRG, rel_varD, rel_libirrep_ED, rel_libirrep_DMRG,
            t_ED, t_DMRG, t_varD);
    fprintf(f, "}\n");
    fclose(f);
    fprintf(stdout, "\nwrote %s\n", out_path);
    return 0;
}
