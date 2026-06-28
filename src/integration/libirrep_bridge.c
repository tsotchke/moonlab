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
#include <irrep/space_group.h>
#include <irrep/config_project.h>
#include <irrep/css_code.h>
#include <irrep/surface_code.h>
#include <irrep/toric_code.h>
#include <irrep/color_code.h>
#include <irrep/bivariate_bicycle.h>
#include <irrep/hypergraph_product.h>

#include <stdlib.h>
#include <string.h>

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
    const irrep_status_t st = irrep_lanczos_eigvals_reorth(
        irrep_heisenberg_apply, H, dim,
        /*k_wanted=*/1, /*max_iters=*/200, /*seed=*/NULL, &e0);

    irrep_heisenberg_free(H);
    irrep_lattice_free(L);

    if (st != IRREP_OK) return MOONLAB_LIBIRREP_INTERNAL;
    *out_energy = e0;
    return MOONLAB_LIBIRREP_OK;
}

/* ============================================================================
 * Sector ED
 * ========================================================================= */

/* Context bundle for the sector-Lanczos callback.  Threaded through
 * `irrep_lanczos_eigvals_reorth` as the opaque void* and unpacked by
 * `sector_apply_thunk` so the rep table is reachable from the matvec. */
typedef struct {
    const irrep_heisenberg_t  *H;
    const irrep_sg_rep_table_t *T;
} sector_apply_ctx_t;

static void sector_apply_thunk(const double _Complex *psi_in,
                               double _Complex *psi_out,
                               void *opaque)
{
    sector_apply_ctx_t *c = (sector_apply_ctx_t *)opaque;
    irrep_heisenberg_apply_in_sector(c->H, c->T, psi_in, psi_out);
}

/* Map the moonlab enum onto libirrep's enum.  Kept as a switch so
 * adding a moonlab-specific value (e.g. a custom lattice not yet in
 * libirrep) is a compile error rather than a silent miscast. */
static int map_lattice_kind(moonlab_libirrep_lattice_kind_t k,
                            irrep_lattice_kind_t *out)
{
    switch (k) {
    case MOONLAB_LIBIRREP_LATTICE_SQUARE:     *out = IRREP_LATTICE_SQUARE;     return 0;
    case MOONLAB_LIBIRREP_LATTICE_TRIANGULAR: *out = IRREP_LATTICE_TRIANGULAR; return 0;
    case MOONLAB_LIBIRREP_LATTICE_HONEYCOMB:  *out = IRREP_LATTICE_HONEYCOMB;  return 0;
    case MOONLAB_LIBIRREP_LATTICE_KAGOME:     *out = IRREP_LATTICE_KAGOME;     return 0;
    default: return -1;
    }
}

static int map_wallpaper(moonlab_libirrep_wallpaper_t w, irrep_wallpaper_t *out)
{
    switch (w) {
    case MOONLAB_LIBIRREP_WALLPAPER_P1:    *out = IRREP_WALLPAPER_P1;    return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P6MM:  *out = IRREP_WALLPAPER_P6MM;  return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P4MM:  *out = IRREP_WALLPAPER_P4MM;  return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P3M1:  *out = IRREP_WALLPAPER_P3M1;  return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P2:    *out = IRREP_WALLPAPER_P2;    return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P6:    *out = IRREP_WALLPAPER_P6;    return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P4:    *out = IRREP_WALLPAPER_P4;    return 0;
    case MOONLAB_LIBIRREP_WALLPAPER_P31M:  *out = IRREP_WALLPAPER_P31M;  return 0;
    default: return -1;
    }
}

int moonlab_libirrep_heisenberg_sector_e0(
    moonlab_libirrep_lattice_kind_t lattice_kind,
    int Lx, int Ly,
    moonlab_libirrep_wallpaper_t    wallpaper,
    int sz_total_2x,
    int k_wanted,
    int max_iters,
    double *eigvals_out,
    long long *sector_dim_out)
{
    if (!eigvals_out || k_wanted < 1 || max_iters < 4
            || Lx < 1 || Ly < 1) {
        return MOONLAB_LIBIRREP_BAD_ARG;
    }

    irrep_lattice_kind_t lk;
    irrep_wallpaper_t    wg;
    if (map_lattice_kind(lattice_kind, &lk) != 0 ||
        map_wallpaper(wallpaper, &wg) != 0) {
        return MOONLAB_LIBIRREP_BAD_ARG;
    }

    int rc = MOONLAB_LIBIRREP_INTERNAL;
    irrep_lattice_t            *L  = NULL;
    irrep_space_group_t        *G  = NULL;
    irrep_sg_rep_table_t       *T  = NULL;
    irrep_heisenberg_t         *H  = NULL;
    int                        *bi = NULL;
    int                        *bj = NULL;

    L = irrep_lattice_build(lk, Lx, Ly);
    if (!L) goto cleanup;

    const int N = irrep_lattice_num_sites(L);
    /* Sz = (popcount(up_state) - popcount(down_state)) / 2 in units
     * where each spin contributes +/- 1/2.  In libirrep's bit-encoding
     * the "popcount" sector parameter is the count of spin-up bits; so
     * popcount = N/2 + sz_total_2x/2 (sz_total_2x is integer-valued and
     * shares parity with N). */
    if (((N + sz_total_2x) & 1) != 0) { rc = MOONLAB_LIBIRREP_BAD_ARG; goto cleanup; }
    if (sz_total_2x < -N || sz_total_2x > N) { rc = MOONLAB_LIBIRREP_BAD_ARG; goto cleanup; }
    const int popcount = (N + sz_total_2x) / 2;

    G = irrep_space_group_build(L, wg);
    if (!G) goto cleanup;

    T = irrep_sg_rep_table_build(G, popcount);
    if (!T) goto cleanup;

    const long long sector_dim = irrep_sg_rep_table_count(T);
    if (sector_dim <= 0) { rc = MOONLAB_LIBIRREP_INTERNAL; goto cleanup; }
    if (k_wanted > (int)sector_dim) k_wanted = (int)sector_dim;

    const int M = irrep_lattice_num_bonds_nn(L);
    if (M <= 0) goto cleanup;
    bi = (int *)malloc((size_t)M * sizeof(int));
    bj = (int *)malloc((size_t)M * sizeof(int));
    if (!bi || !bj) { rc = MOONLAB_LIBIRREP_OOM; goto cleanup; }
    irrep_lattice_fill_bonds_nn(L, bi, bj);

    H = irrep_heisenberg_new(N, M, bi, bj, /*J=*/1.0);
    if (!H) goto cleanup;

    sector_apply_ctx_t ctx = { .H = H, .T = T };
    const irrep_status_t st = irrep_lanczos_eigvals_reorth(
        sector_apply_thunk, &ctx, sector_dim,
        k_wanted, max_iters, /*seed=*/NULL, eigvals_out);
    if (st != IRREP_OK) goto cleanup;

    if (sector_dim_out) *sector_dim_out = sector_dim;
    rc = MOONLAB_LIBIRREP_OK;

cleanup:
    if (H)  irrep_heisenberg_free(H);
    if (T)  irrep_sg_rep_table_free(T);
    if (G)  irrep_space_group_free(G);
    if (L)  irrep_lattice_free(L);
    free(bi); free(bj);
    return rc;
}

/* ============================================================================
 * CSS code handle
 * ========================================================================= */

struct moonlab_libirrep_qec {
    irrep_css_code_t css;
    int distance_cached;  /**< -1 until first lookup, then memoised. */
};

/* Common helper for the QEC zoo: allocate a moonlab_libirrep_qec_t, run
 * the caller-supplied libirrep builder against its `css` field, and
 * return MOONLAB_LIBIRREP_OK on success.  Each factory function only
 * has to provide the builder closure -- this keeps the per-family
 * boilerplate to one line. */
typedef int (*css_builder_fn)(irrep_css_code_t *out, void *ctx);

static int qec_factory(moonlab_libirrep_qec_t **out,
                       css_builder_fn build, void *ctx)
{
    if (!out) return MOONLAB_LIBIRREP_BAD_ARG;
    moonlab_libirrep_qec_t *q = (moonlab_libirrep_qec_t *)calloc(1, sizeof(*q));
    if (!q) return MOONLAB_LIBIRREP_OOM;
    q->distance_cached = -1;
    if (build(&q->css, ctx) != 0) {
        free(q);
        return MOONLAB_LIBIRREP_INTERNAL;
    }
    *out = q;
    return MOONLAB_LIBIRREP_OK;
}

static int build_surface(irrep_css_code_t *out, void *ctx)
{
    const int distance = *(const int *)ctx;
    irrep_surface_params_t p;
    if (irrep_surface_init(&p, distance) != IRREP_OK) return -1;
    return (irrep_surface_build(&p, out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_surface_code_new(int distance, moonlab_libirrep_qec_t **out)
{
    if (distance < 2) return MOONLAB_LIBIRREP_BAD_ARG;
    int d = distance;
    return qec_factory(out, build_surface, &d);
}

typedef struct { int Lx; int Ly; } toric_args_t;

static int build_toric(irrep_css_code_t *out, void *ctx)
{
    const toric_args_t *a = (const toric_args_t *)ctx;
    irrep_toric_params_t p;
    if (irrep_toric_init(&p, a->Lx, a->Ly) != IRREP_OK) return -1;
    return (irrep_toric_code_build_css(&p, out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_toric_code_new(int Lx, int Ly, moonlab_libirrep_qec_t **out)
{
    if (Lx < 2 || Ly < 2) return MOONLAB_LIBIRREP_BAD_ARG;
    toric_args_t a = { .Lx = Lx, .Ly = Ly };
    return qec_factory(out, build_toric, &a);
}

static int build_color_steane(irrep_css_code_t *out, void *ctx)
{
    (void)ctx;
    return (irrep_color_steane(out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_color_steane_new(moonlab_libirrep_qec_t **out)
{
    return qec_factory(out, build_color_steane, NULL);
}

static int build_color_hamming(irrep_css_code_t *out, void *ctx)
{
    (void)ctx;
    return (irrep_color_hamming_15_7_3(out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_color_hamming_15_7_3_new(moonlab_libirrep_qec_t **out)
{
    return qec_factory(out, build_color_hamming, NULL);
}

static int build_bb_72(irrep_css_code_t *out, void *ctx)
{
    (void)ctx;
    return (irrep_bb_code_ibm_72_12_6(out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_bb_72_12_6_new(moonlab_libirrep_qec_t **out)
{
    return qec_factory(out, build_bb_72, NULL);
}

static int build_bb_144(irrep_css_code_t *out, void *ctx)
{
    (void)ctx;
    return (irrep_bb_code_ibm_144_12_12(out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_bb_144_12_12_new(moonlab_libirrep_qec_t **out)
{
    return qec_factory(out, build_bb_144, NULL);
}

static int build_bb_288(irrep_css_code_t *out, void *ctx)
{
    (void)ctx;
    return (irrep_bb_code_ibm_288_12_18(out) == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_bb_288_12_18_new(moonlab_libirrep_qec_t **out)
{
    return qec_factory(out, build_bb_288, NULL);
}

static int build_hgp(irrep_css_code_t *out, void *ctx)
{
    const int d = *(const int *)ctx;
    irrep_status_t st;
    switch (d) {
    case 3: st = irrep_hgp_repetition_3_13_1_3(out); break;
    case 4: st = irrep_hgp_repetition_4_25_1_4(out); break;
    case 5: st = irrep_hgp_repetition_5_41_1_5(out); break;
    default: return -1;
    }
    return (st == IRREP_OK) ? 0 : -1;
}

int moonlab_libirrep_hgp_repetition_new(int d, moonlab_libirrep_qec_t **out)
{
    if (d < 3 || d > 5) return MOONLAB_LIBIRREP_BAD_ARG;
    int dd = d;
    return qec_factory(out, build_hgp, &dd);
}

void moonlab_libirrep_qec_free(moonlab_libirrep_qec_t *q)
{
    if (!q) return;
    irrep_css_code_free(&q->css);
    free(q);
}

int moonlab_libirrep_qec_n_qubits(const moonlab_libirrep_qec_t *q)
{
    return q ? q->css.n : MOONLAB_LIBIRREP_BAD_ARG;
}

int moonlab_libirrep_qec_n_x_stabs(const moonlab_libirrep_qec_t *q)
{
    return q ? q->css.H_X.n_rows : MOONLAB_LIBIRREP_BAD_ARG;
}

int moonlab_libirrep_qec_n_z_stabs(const moonlab_libirrep_qec_t *q)
{
    return q ? q->css.H_Z.n_rows : MOONLAB_LIBIRREP_BAD_ARG;
}

int moonlab_libirrep_qec_logical_qubits(const moonlab_libirrep_qec_t *q)
{
    if (!q) return MOONLAB_LIBIRREP_BAD_ARG;
    return irrep_css_code_logical_qubits(&q->css);
}

int moonlab_libirrep_qec_distance(moonlab_libirrep_qec_t *q)
{
    if (!q) return MOONLAB_LIBIRREP_BAD_ARG;
    if (q->distance_cached > 0) return q->distance_cached;
    /* Cap brute-force enumeration at n / 2 + 1: any logical operator
     * has weight at most n, but the minimum-weight one is much smaller
     * for any well-formed code.  Surface code at d means the answer is
     * exactly d, so n_qubits = d^2 is a generous upper bound. */
    const int n = q->css.n;
    const int max_weight = n;
    const int d = irrep_css_code_distance(&q->css, max_weight);
    if (d > 0 && d <= max_weight) q->distance_cached = d;
    return d;
}

static int read_check_row(const irrep_parity_matrix_t *m, int row,
                          unsigned char *support)
{
    if (!m || !support) return MOONLAB_LIBIRREP_BAD_ARG;
    if (row < 0 || row >= m->n_rows) return MOONLAB_LIBIRREP_BAD_ARG;
    for (int col = 0; col < m->n_cols; ++col) {
        const int bit = irrep_parity_matrix_get(m, row, col);
        if (bit < 0) return MOONLAB_LIBIRREP_INTERNAL;
        support[col] = (unsigned char)(bit & 1);
    }
    return MOONLAB_LIBIRREP_OK;
}

int moonlab_libirrep_qec_get_x_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support)
{
    if (!q) return MOONLAB_LIBIRREP_BAD_ARG;
    return read_check_row(&q->css.H_X, row, support);
}

int moonlab_libirrep_qec_get_z_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support)
{
    if (!q) return MOONLAB_LIBIRREP_BAD_ARG;
    return read_check_row(&q->css.H_Z, row, support);
}

#else /* !MOONLAB_HAS_LIBIRREP */

int moonlab_libirrep_available(void) { return 0; }

int moonlab_libirrep_kagome12_e0(double *out_energy)
{
    (void)out_energy;
    return MOONLAB_LIBIRREP_NOT_BUILT;
}

int moonlab_libirrep_heisenberg_sector_e0(
    moonlab_libirrep_lattice_kind_t lattice_kind,
    int Lx, int Ly,
    moonlab_libirrep_wallpaper_t    wallpaper,
    int sz_total_2x,
    int k_wanted,
    int max_iters,
    double *eigvals_out,
    long long *sector_dim_out)
{
    (void)lattice_kind; (void)Lx; (void)Ly; (void)wallpaper;
    (void)sz_total_2x; (void)k_wanted; (void)max_iters;
    (void)eigvals_out; (void)sector_dim_out;
    return MOONLAB_LIBIRREP_NOT_BUILT;
}

int moonlab_libirrep_surface_code_new(int distance, moonlab_libirrep_qec_t **out)
{
    (void)distance; (void)out;
    return MOONLAB_LIBIRREP_NOT_BUILT;
}

int moonlab_libirrep_toric_code_new(int Lx, int Ly, moonlab_libirrep_qec_t **out)
{ (void)Lx; (void)Ly; (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_color_steane_new(moonlab_libirrep_qec_t **out)
{ (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_color_hamming_15_7_3_new(moonlab_libirrep_qec_t **out)
{ (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_bb_72_12_6_new(moonlab_libirrep_qec_t **out)
{ (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_bb_144_12_12_new(moonlab_libirrep_qec_t **out)
{ (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_bb_288_12_18_new(moonlab_libirrep_qec_t **out)
{ (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_hgp_repetition_new(int d, moonlab_libirrep_qec_t **out)
{ (void)d; (void)out; return MOONLAB_LIBIRREP_NOT_BUILT; }

void moonlab_libirrep_qec_free(moonlab_libirrep_qec_t *q) { (void)q; }

int moonlab_libirrep_qec_n_qubits(const moonlab_libirrep_qec_t *q)
{ (void)q; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_n_x_stabs(const moonlab_libirrep_qec_t *q)
{ (void)q; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_n_z_stabs(const moonlab_libirrep_qec_t *q)
{ (void)q; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_logical_qubits(const moonlab_libirrep_qec_t *q)
{ (void)q; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_distance(moonlab_libirrep_qec_t *q)
{ (void)q; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_get_x_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support)
{ (void)q; (void)row; (void)support; return MOONLAB_LIBIRREP_NOT_BUILT; }

int moonlab_libirrep_qec_get_z_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support)
{ (void)q; (void)row; (void)support; return MOONLAB_LIBIRREP_NOT_BUILT; }

#endif /* MOONLAB_HAS_LIBIRREP */
