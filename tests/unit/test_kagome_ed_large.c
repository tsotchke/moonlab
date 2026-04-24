/**
 * @file test_kagome_ed_large.c
 * @brief Moonlab <-> canonical-ED cross-check at N=18 kagome (2x3 torus).
 *
 * Reference: Läuchli, Sudan, Sørensen, "Ground-State Energy and Spin Gap of
 * Spin-1/2 Kagomé Heisenberg Antiferromagnetic Clusters: Large Scale Exact
 * Diagonalization Results," Phys. Rev. B 83, 212401 (2011), arXiv:1103.1159,
 * Table I:
 *
 *   N=12  (a=(2,0), b=(0,2))        E = -5.444875216 J   [dM=4, |G|=48]
 *   N=18a (a=(2,-1), b=(0,3))       E = -8.064482605 J   [dM=4, |G|=24, sheared]
 *   N=18b (a=(2,-2), b=(-2,-1))     E = -8.048270773 J   [dM=4, |G|=48, rect 2x3]
 *
 * Moonlab's lattice_2d encodes the 3 kagome sublattices along the x axis
 * (c.x % 3 == sublattice).  With Lx=6, Ly=3 and BC_PERIODIC_XY we generate the
 * rectangular 2x3 unit-cell torus, matching the N=18b cluster in the PRB table.
 *
 * Method: matrix-free Lanczos with full reorthogonalization on a bond-list
 * Heisenberg Hamiltonian.  dim = 2^18 = 262144, peak memory ~840 MB for 200
 * Krylov vectors.  Runtime ~8s on M2 Ultra.
 *
 * On agreement with PRB at the 1e-7 level we verify that Moonlab's
 * lattice_2d_create(kagome, 6, 3, PBC_XY) produces the canonical 2x3 torus,
 * that its bond graph reproduces the PRB 18b cluster (up to relabeling), and
 * that downstream DMRG / TDVP / KPM work on kagome will see the correct
 * Hamiltonian.
 *
 * Independent cross-check: on 2026-04-24 libirrep's PHYSICS_RESULTS.md
 * reported E_0(N=18) = -8.04719493 J, which disagrees with PRB Table I by
 * 1e-3.  That entry comes from a 2000-iter shifted power iteration which
 * under-converges on kagome's dense singlet tower.  Moonlab's Lanczos here
 * matches PRB to the published precision (9 significant digits).
 */

#include "../../src/algorithms/tensor_network/lattice_2d.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __APPLE__
extern void dstev_(char *jobz, int *n, double *d, double *e,
                   double *z, int *ldz, double *work, int *info);
#endif

/* ------------------------------------------------------------------ */
/*  Heisenberg matrix-free apply:                                     */
/*    H = J * sum_bonds (X_i X_j + Y_i Y_j + Z_i Z_j)                 */
/*                                                                    */
/*  Z_iZ_j|sigma> = (+1 if b_i==b_j else -1) |sigma>                  */
/*  (X_iX_j + Y_iY_j)|sigma>: when bits differ, both flip with +2;    */
/*    when bits match, X contributes +1 and Y contributes -1 so sum 0 */
/* ------------------------------------------------------------------ */
static void apply_heisenberg(const uint32_t (*bonds)[2], uint32_t num_bonds,
                             double J, uint32_t N,
                             const double complex *in, double complex *out) {
    uint64_t dim = 1ULL << N;
    for (uint64_t s = 0; s < dim; s++) {
        int diag = 0;
        for (uint32_t b = 0; b < num_bonds; b++) {
            int bi = (int)((s >> bonds[b][0]) & 1);
            int bj = (int)((s >> bonds[b][1]) & 1);
            diag += (bi == bj) ? 1 : -1;
        }
        out[s] = J * (double)diag * in[s];
    }
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        uint64_t mask = (1ULL << i) | (1ULL << j);
        for (uint64_t s = 0; s < dim; s++) {
            if (((s >> i) & 1u) != ((s >> j) & 1u)) {
                out[s ^ mask] += 2.0 * J * in[s];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Complex vector primitives                                         */
/* ------------------------------------------------------------------ */
static double complex cdot(const double complex *a, const double complex *b, uint64_t n) {
    double re = 0.0, im = 0.0;
    for (uint64_t s = 0; s < n; s++) {
        double complex z = conj(a[s]) * b[s];
        re += creal(z); im += cimag(z);
    }
    return re + I * im;
}
static void cvec_axpy(double complex *v, double complex alpha, const double complex *u, uint64_t n) {
    for (uint64_t s = 0; s < n; s++) v[s] -= alpha * u[s];
}
static double cvec_norm(const double complex *v, uint64_t n) {
    double acc = 0.0;
    for (uint64_t s = 0; s < n; s++) {
        double re = creal(v[s]), im = cimag(v[s]);
        acc += re * re + im * im;
    }
    return sqrt(acc);
}
static void cvec_scale(double complex *v, double s, uint64_t n) {
    for (uint64_t k = 0; k < n; k++) v[k] *= s;
}

/* ------------------------------------------------------------------ */
/*  Tridiagonal diagonalization via LAPACK dstev                      */
/* ------------------------------------------------------------------ */
static double tridiag_min_eigenvalue(const double *alpha, const double *beta, int m) {
    double *d = (double *)malloc((size_t)m * sizeof(double));
    double *e = (double *)malloc((size_t)m * sizeof(double));
    memcpy(d, alpha, (size_t)m * sizeof(double));
    for (int i = 0; i < m - 1; i++) e[i] = beta[i];
    char jobz = 'N';
    double work_dummy;
#ifdef __APPLE__
    __CLPK_integer nn = m, ldz = 1, info = 0;
    dstev_(&jobz, &nn, d, e, NULL, &ldz, &work_dummy, &info);
#else
    int nn = m, ldz = 1, info = 0;
    dstev_(&jobz, &nn, d, e, NULL, &ldz, &work_dummy, &info);
#endif
    double lo = d[0];
    for (int i = 1; i < m; i++) if (d[i] < lo) lo = d[i];
    free(d); free(e);
    return lo;
}

/* ------------------------------------------------------------------ */
/*  Full-reorthogonalization Lanczos.                                 */
/*                                                                    */
/*  The full Krylov basis V[0..k] is stored explicitly so we can      */
/*  Gram-Schmidt against every prior vector at every step (with a     */
/*  second DGKS pass for robustness).  This prevents the              */
/*  orthogonality loss that gives plain three-term Lanczos spurious   */
/*  copies of the ground-state eigenvalue.  Essential on kagome       */
/*  where the singlet tower is dense in the low spectrum.             */
/* ------------------------------------------------------------------ */
static double lanczos_full_reorth_gs(const uint32_t (*bonds)[2], uint32_t num_bonds,
                                     double J, uint32_t N, int max_iters, int seed,
                                     int *iters_used, double *converged_delta) {
    uint64_t dim = 1ULL << N;
    size_t bytes_per_vec = dim * sizeof(double complex);
    double complex *V = (double complex *)calloc((size_t)max_iters * dim, sizeof(double complex));
    double complex *w = (double complex *)calloc(dim, sizeof(double complex));
    if (!V || !w) {
        fprintf(stderr, "OOM: need %zu MB for Krylov basis\n",
                (size_t)max_iters * bytes_per_vec / (1024 * 1024));
        free(V); free(w);
        return NAN;
    }

    /* Deterministic seed, restricted to Sz=0 sector (popcount == N/2). */
    uint64_t rng = 0x123456789abcdefULL ^ (uint64_t)seed;
    double n0 = 0.0;
    uint32_t half = N / 2;
    for (uint64_t s = 0; s < dim; s++) {
        if (__builtin_popcountll(s) != (int)half) { V[s] = 0.0; continue; }
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double x = (double)(rng >> 32) / (double)0xFFFFFFFFULL - 0.5;
        V[s] = x; n0 += x * x;
    }
    double inv_n0 = 1.0 / sqrt(n0);
    for (uint64_t s = 0; s < dim; s++) V[s] *= inv_n0;

    double *alpha = (double *)calloc((size_t)max_iters, sizeof(double));
    double *beta  = (double *)calloc((size_t)max_iters, sizeof(double));

    double prev_gs = 0.0, curr_gs = 0.0;
    double delta = 1e30;
    int k;
    for (k = 0; k < max_iters; k++) {
        double complex *vk = V + (size_t)k * dim;

        memset(w, 0, bytes_per_vec);
        apply_heisenberg(bonds, num_bonds, J, N, vk, w);

        double complex a = cdot(vk, w, dim);
        alpha[k] = creal(a);

        for (uint64_t s = 0; s < dim; s++) w[s] -= alpha[k] * vk[s];

        /* Two-pass Gram-Schmidt (DGKS) against full basis. */
        for (int pass = 0; pass < 2; pass++) {
            for (int j = 0; j <= k; j++) {
                double complex *vj = V + (size_t)j * dim;
                double complex c = cdot(vj, w, dim);
                cvec_axpy(w, c, vj, dim);
            }
        }

        double bn = cvec_norm(w, dim);
        beta[k] = bn;
        if (bn < 1e-14) { k++; break; }

        if (k + 1 < max_iters) {
            cvec_scale(w, 1.0 / bn, dim);
            memcpy(V + (size_t)(k + 1) * dim, w, bytes_per_vec);
        }

        if (k + 1 >= 5) {
            prev_gs = curr_gs;
            curr_gs = tridiag_min_eigenvalue(alpha, beta, k + 1);
            delta = fabs(curr_gs - prev_gs);
            if (k >= 10 && delta < 1e-12) { k++; break; }
        }
    }

    double gs = tridiag_min_eigenvalue(alpha, beta, k);
    if (iters_used) *iters_used = k;
    if (converged_delta) *converged_delta = delta;

    free(V); free(w); free(alpha); free(beta);
    return gs;
}

/* ------------------------------------------------------------------ */
/*  Bond list construction from lattice_2d, with PBC dedup            */
/* ------------------------------------------------------------------ */
static uint32_t collect_unique_bonds(const lattice_2d_t *lat, uint32_t (*bonds)[2]) {
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
            bonds[n][0] = i;
            bonds[n][1] = j;
            n++;
        }
    }
    return n;
}

int main(void) {
    int failures = 0;
    fprintf(stdout, "=== kagome 18-site Heisenberg ED: Moonlab vs PRB 83, 212401 (2011) ===\n\n");

    lattice_2d_t *lat = lattice_2d_create(6, 3, LATTICE_KAGOME, BC_PERIODIC_XY);
    if (!lat || lat->num_sites != 18) {
        fprintf(stderr, "  FAIL  lattice_2d_create(6, 3, KAGOME, PBC_XY) -> %u sites\n",
                lat ? lat->num_sites : 0);
        if (lat) lattice_2d_free(lat);
        return 1;
    }
    fprintf(stdout, "  num_sites     = %u\n", lat->num_sites);
    fprintf(stdout, "  max_neighbors = %u (expect 4)\n", lat->max_neighbors);

    const size_t max_bonds = (size_t)lat->max_neighbors * lat->num_sites / 2;
    uint32_t (*bonds)[2] = calloc(max_bonds, sizeof(*bonds));
    uint32_t num_bonds = collect_unique_bonds(lat, bonds);
    fprintf(stdout, "  num_bonds     = %u (expect 36)\n", num_bonds);
    if (num_bonds != 36) {
        fprintf(stderr, "  FAIL  bond count mismatch\n");
        free(bonds); lattice_2d_free(lat); return 1;
    }

    fprintf(stdout, "\n  Running matrix-free Lanczos (full reorth, Sz=0 seed, max 200 iters)...\n");
    int iters = 0; double delta = 0.0;
    /* J' = 0.25 Pauli-form equals J = 1 spin-1/2 convention (sigma = 2 S). */
    double e0 = lanczos_full_reorth_gs(bonds, num_bonds, 0.25,
                                       lat->num_sites, 200, 42,
                                       &iters, &delta);

    /* PRB Table I, cluster "18 b": a=(2,-2), b=(-2,-1), rectangular 2x3 torus. */
    const double E0_prb_18b = -8.048270773;
    const double diff = fabs(e0 - E0_prb_18b);

    fprintf(stdout, "\n  Moonlab E_0             = %+.10f\n", e0);
    fprintf(stdout, "  PRB 83, 212401 (18b)    = %+.10f\n", E0_prb_18b);
    fprintf(stdout, "  |diff|                  = %.3e\n", diff);
    fprintf(stdout, "  Lanczos iters / delta   = %d / %.2e\n\n", iters, delta);

    /* Läuchli et al. quote 9 digits; demand 1e-7 agreement. */
    if (diff < 1e-7) {
        fprintf(stdout, "  OK    Moonlab kagome 2x3 torus matches PRB 18b within 1e-7\n");
    } else {
        fprintf(stderr, "  FAIL  diff %.3e exceeds 1e-7\n", diff);
        failures++;
    }

    free(bonds); lattice_2d_free(lat);
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
