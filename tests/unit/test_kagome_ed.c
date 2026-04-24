/**
 * @file test_kagome_ed.c
 * @brief Moonlab <-> libirrep cross-check: 12-site kagome Heisenberg ED.
 *
 * libirrep `docs/PHYSICS_RESULTS.md` (table in section 1.1) reports
 * E_0 = -5.44487522 J for the 2x2 kagome torus with p6mm symmetry,
 * N = 12 sites, 24 NN bonds, H = J sum_<ij> S_i.S_j, J = 1, S = 1/2.
 * Ground state sits in the B1 irrep of the little group at Gamma
 * with total spin J = 0.
 *
 * Moonlab's mpo_2d_heisenberg_dmi_create builds H in Pauli operators:
 *   H_Pauli = J' sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j)
 * which equals 4 * J' * sum S_i.S_j under sigma = 2 S.  So passing
 * J' = 0.25 makes Moonlab's Hamiltonian identical to libirrep's
 * J = 1 spin-1/2 form.
 *
 * On agreement we learn:
 *   - Moonlab's kagome neighbor-list topology matches the canonical
 *     kagome graph.
 *   - Moonlab's MPO builder applies the Heisenberg interaction on
 *     every bond with the right sign and scale.
 *   - The mpo_to_matrix + zheev_ pipeline is internally consistent.
 *
 * On disagreement we learn which of the three is wrong.  Without this
 * smoke test every downstream kagome result (DMRG convergence study,
 * Kitaev-Preskill gamma driver, flux-threading gap extraction) rests
 * on an unchecked assumption about the bond graph.
 *
 * See also:
 *   - libirrep PHYSICS_RESULTS.md section 1.1 (the target number)
 *   - libirrep MOONLAB_INTEGRATION.md section 6 (handshake plan)
 *   - libirrep examples/kagome12_ed.c (the reference ED run)
 */

#include "../../src/algorithms/tensor_network/lattice_2d.h"
#include "../../src/algorithms/tensor_network/mpo_2d.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tensor.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void zheev_(char *jobz, char *uplo, int *n, double complex *a, int *lda,
                   double *w, double complex *work, int *lwork,
                   double *rwork, int *info);
#endif

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

/* libirrep PHYSICS_RESULTS.md, section 1.1. */
#define LIBIRREP_KAGOME12_E0  (-5.44487522)

/* The test uses Moonlab's Pauli-operator Heisenberg convention:
 * H = J' * sum (X X + Y Y + Z Z). To match libirrep's spin-1/2
 * J = 1 convention we pass J' = 0.25 (because sigma = 2 S gives
 * sigma_i.sigma_j = 4 S_i.S_j). */
#define J_PAULI  0.25

/* Extract unique i<j bond pairs from a lattice_2d's neighbor list.
 * num_bonds_out gets written; bonds array must be preallocated large
 * enough (coordination * num_sites / 2 is an upper bound). */
static void collect_bonds(const lattice_2d_t *lat,
                          uint32_t (*bonds)[2], double (*bond_vectors)[3],
                          uint32_t *num_bonds_out) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < lat->num_sites; i++) {
        for (uint32_t k = 0; k < lat->num_neighbors[i]; k++) {
            uint32_t j = lat->neighbors[i][k].site;
            if (j <= i) continue;  /* keep each bond once */

            /* Dedup: the same (i, j) can appear multiple times in
             * neighbor arrays when small-Ly PBC wraps send two
             * distinct neighbor offsets to the same site (e.g. y+1
             * and y-1 are the same neighbor when Ly = 2). */
            int dup = 0;
            for (uint32_t m = 0; m < n; m++) {
                if (bonds[m][0] == i && bonds[m][1] == j) { dup = 1; break; }
            }
            if (dup) continue;

            bonds[n][0] = i;
            bonds[n][1] = j;
            bond_vectors[n][0] = lat->neighbors[i][k].bond.dx;
            bond_vectors[n][1] = lat->neighbors[i][k].bond.dy;
            bond_vectors[n][2] = lat->neighbors[i][k].bond.dz;
            n++;
        }
    }
    *num_bonds_out = n;
}

/* Diagonalise a column-major Hermitian matrix, returning the lowest
 * eigenvalue.  The full spectrum is written into w[] (ascending). */
static int dense_lowest_eigenvalue(double complex *H, int n, double *w) {
    char jobz = 'N', uplo = 'U';  /* eigenvalues only */
    int lda = n, lwork = -1, info = 0;
    double *rwork = (double *)malloc((size_t)(3 * n) * sizeof(double));
    double complex work_query;
#ifdef __APPLE__
    __CLPK_integer n_c = n, lda_c = lda, lwork_c = lwork, info_c = 0;
    __CLPK_doublecomplex wq;
    zheev_(&jobz, &uplo, &n_c, (__CLPK_doublecomplex *)H, &lda_c,
           w, &wq, &lwork_c, rwork, &info_c);
    memcpy(&work_query, &wq, sizeof(work_query));
    lwork = (int)creal(work_query);
    info = (int)info_c;
#else
    zheev_(&jobz, &uplo, &n, H, &lda, w, &work_query, &lwork, rwork, &info);
    lwork = (int)creal(work_query);
#endif
    if (info != 0) { free(rwork); return info; }
    double complex *work = (double complex *)malloc((size_t)lwork * sizeof(double complex));
    if (!work) { free(rwork); return -1; }
#ifdef __APPLE__
    n_c = n; lda_c = lda; lwork_c = lwork;
    zheev_(&jobz, &uplo, &n_c, (__CLPK_doublecomplex *)H, &lda_c,
           w, (__CLPK_doublecomplex *)work, &lwork_c, rwork, &info_c);
    info = (int)info_c;
#else
    zheev_(&jobz, &uplo, &n, H, &lda, w, work, &lwork, rwork, &info);
#endif
    free(work);
    free(rwork);
    return info;
}

int main(void) {
    fprintf(stdout, "=== kagome 12-site Heisenberg ED: Moonlab vs libirrep ===\n");

    /* 1. Build the lattice. */
    const uint32_t Lx = 6, Ly = 2;  /* 6 x 2 = 12 sites with c.x%3 sublattice encoding */
    lattice_2d_t *lat = lattice_2d_create(Lx, Ly, LATTICE_KAGOME, BC_PERIODIC_XY);
    CHECK(lat != NULL, "lattice_2d_create(6, 2, KAGOME, PBC_XY)");
    if (!lat) return 1;
    fprintf(stdout, "    num_sites = %u  max_neighbors = %u\n",
            lat->num_sites, lat->max_neighbors);
    CHECK(lat->num_sites == 12, "12 sites as expected");

    /* 2. Collect unique bonds. */
    const size_t max_bonds = (size_t)lat->max_neighbors * lat->num_sites / 2;
    uint32_t (*bonds)[2] = calloc(max_bonds, sizeof(*bonds));
    double (*bond_vectors)[3] = calloc(max_bonds, sizeof(*bond_vectors));
    uint32_t num_bonds = 0;
    collect_bonds(lat, bonds, bond_vectors, &num_bonds);
    fprintf(stdout, "    num_bonds (unique i<j) = %u  (expect 24)\n", num_bonds);
    CHECK(num_bonds == 24, "kagome 2x2 torus has 24 NN bonds");

    /* 3. Build the Heisenberg MPO with J' = 0.25 (matches libirrep J = 1). */
    mpo_t *H_mpo = mpo_2d_heisenberg_dmi_create(
        (uint32_t)lat->num_sites, num_bonds, bonds, bond_vectors,
        /*J*/ J_PAULI, /*D*/ 0.0, /*B*/ 0.0, /*K*/ 0.0);
    CHECK(H_mpo != NULL, "mpo_2d_heisenberg_dmi_create(J=0.25, D=0)");
    if (!H_mpo) { free(bonds); free(bond_vectors); lattice_2d_free(lat); return 1; }

    /* 4. Dense H via mpo_to_matrix (4096 x 4096 complex). */
    tensor_t *H_dense_t = mpo_to_matrix(H_mpo);
    CHECK(H_dense_t != NULL, "mpo_to_matrix");
    if (!H_dense_t) {
        mpo_free(H_mpo); free(bonds); free(bond_vectors);
        lattice_2d_free(lat);
        return 1;
    }
    const int N = (int)H_dense_t->dims[0];
    fprintf(stdout, "    dense H is %d x %d (complex128)\n", N, N);
    CHECK(N == 4096, "Hilbert space dimension = 2^12 = 4096");

    double complex *H = malloc((size_t)N * (size_t)N * sizeof(double complex));
    memcpy(H, H_dense_t->data, (size_t)N * (size_t)N * sizeof(double complex));
    tensor_free(H_dense_t);

    /* 5. Diagonalise. */
    double *w = (double *)malloc((size_t)N * sizeof(double));
    int info = dense_lowest_eigenvalue(H, N, w);
    CHECK(info == 0, "zheev_ returns info = 0");

    const double E0 = w[0];
    const double E0_libirrep = LIBIRREP_KAGOME12_E0;
    const double diff = fabs(E0 - E0_libirrep);

    fprintf(stdout, "\n    Moonlab E_0 (J=0.25 Pauli) = %.8f\n", E0);
    fprintf(stdout, "    libirrep E_0 (J=1 spin)    = %.8f\n", E0_libirrep);
    fprintf(stdout, "    |diff|                     = %.3e\n\n", diff);

    /* Agreement to ~1e-8 is what we expect from two independent
     * Lanczos-precision computations on a 4096-dim problem. */
    CHECK(diff < 1e-7, "Moonlab agrees with libirrep to <1e-7");

    /* Also print the full low spectrum for context. */
    fprintf(stdout, "\n    Lowest 6 eigenvalues:\n");
    for (int i = 0; i < 6; i++) {
        fprintf(stdout, "      E_%d = %+.8f\n", i, w[i]);
    }

    free(H); free(w); free(bonds); free(bond_vectors);
    mpo_free(H_mpo);
    lattice_2d_free(lat);

    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
