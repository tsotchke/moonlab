/**
 * @file    libirrep_bridge.c
 * @brief   Implementation of the libirrep bridge (header for rationale).
 *
 * Two-mode TU: when MOONLAB_HAS_LIBIRREP is defined the real
 * libirrep-calling implementation compiles; otherwise the file compiles
 * to a stub that returns MOONLAB_LIBIRREP_NOT_BUILT from every entry
 * point.  Keeping both modes in one TU keeps the build-system surface
 * one source file rather than two and lets the linker drop unused
 * stubs automatically.
 */

#include "libirrep_bridge.h"

#ifdef MOONLAB_HAS_LIBIRREP

#include <irrep/lattice.h>
#include <irrep/hamiltonian.h>
#include <irrep/rdm.h>
#include <irrep/types.h>

#include <stdlib.h>

int moonlab_libirrep_available(void) { return 1; }

int moonlab_libirrep_kagome12_e0(double *out_energy)
{
    if (!out_energy) return MOONLAB_LIBIRREP_BAD_ARG;

    /* 2x2 kagome unit cells -> 12 sites, 24 NN bonds, |H| = 2^12 = 4096. */
    irrep_lattice_t *L = irrep_lattice_build(IRREP_LATTICE_KAGOME, 2, 2);
    if (!L) return MOONLAB_LIBIRREP_INTERNAL;

    const int N = irrep_lattice_num_sites(L);
    const int M = irrep_lattice_num_bonds_nn(L);
    if (N != 12 || M <= 0) {
        irrep_lattice_free(L);
        return MOONLAB_LIBIRREP_INTERNAL;
    }

    int *bi = (int *)malloc((size_t)M * sizeof(int));
    int *bj = (int *)malloc((size_t)M * sizeof(int));
    if (!bi || !bj) {
        free(bi); free(bj);
        irrep_lattice_free(L);
        return MOONLAB_LIBIRREP_OOM;
    }

    /* `_fill_bonds_nn` is void; it writes the bond list directly. */
    irrep_lattice_fill_bonds_nn(L, bi, bj);

    /* J = 1 in spin units (libirrep uses S = sigma/2, so
     * H_spin = sum S_i.S_j; the 12-site E_0 = -5.44487522 in that
     * convention is the value moonlab cross-validates against). */
    irrep_heisenberg_t *H = irrep_heisenberg_new(N, M, bi, bj, 1.0);
    free(bi); free(bj);
    if (!H) {
        irrep_lattice_free(L);
        return MOONLAB_LIBIRREP_INTERNAL;
    }

    const long long dim = irrep_heisenberg_dim(H);
    double e0 = 0.0;
    /* Full-reorth Lanczos handles the near-degeneracy of kagome's
     * low-singlet tower at modest max_iters; 200 iterations is
     * conservative for the 12-site cluster (matches libirrep's own
     * `examples/kagome12_ed.c` defaults). */
    const irrep_status_t st = irrep_lanczos_eigvals_reorth(
        irrep_heisenberg_apply, H, dim,
        /*k_wanted=*/1, /*max_iters=*/200, /*seed=*/NULL, &e0);

    irrep_heisenberg_free(H);
    irrep_lattice_free(L);

    if (st != IRREP_OK) return MOONLAB_LIBIRREP_INTERNAL;
    *out_energy = e0;
    return MOONLAB_LIBIRREP_OK;
}

#else /* !MOONLAB_HAS_LIBIRREP */

int moonlab_libirrep_available(void) { return 0; }

int moonlab_libirrep_kagome12_e0(double *out_energy)
{
    (void)out_energy;
    return MOONLAB_LIBIRREP_NOT_BUILT;
}

#endif /* MOONLAB_HAS_LIBIRREP */
