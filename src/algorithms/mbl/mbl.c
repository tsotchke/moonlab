/**
 * @file mbl.c
 * @brief Many-Body Localization simulation implementation
 *
 * Full implementation of MBL diagnostics and simulation:
 * - Disordered XXZ Heisenberg model
 * - Exact diagonalization via LAPACK
 * - Level spacing statistics (Poisson vs GOE)
 * - Time evolution (exact and Krylov methods)
 * - Entanglement and imbalance dynamics
 * - LIOM construction
 *
 * @stability stable
 * @since v1.0.0
 */

#include "mbl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
// LAPACK prototypes for non-Apple systems
extern void zheev_(char *jobz, char *uplo, int *n, double complex *a, int *lda,
                   double *w, double complex *work, int *lwork, double *rwork, int *info);
extern void zgemv_(char *trans, int *m, int *n, double complex *alpha,
                   double complex *a, int *lda, double complex *x, int *incx,
                   double complex *beta, double complex *y, int *incy);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
                   double complex *alpha, double complex *a, int *lda,
                   double complex *b, int *ldb, double complex *beta,
                   double complex *c, int *ldc);
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Level statistics reference values
#define POISSON_R 0.3863
#define GOE_R 0.5307

// ============================================================================
// RANDOM NUMBER GENERATION (Xoshiro256**)
// ============================================================================

static uint64_t xoshiro_state[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(void) {
    const uint64_t result = rotl(xoshiro_state[1] * 5, 7) * 9;
    const uint64_t t = xoshiro_state[1] << 17;
    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[3] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[0] ^= xoshiro_state[3];
    xoshiro_state[2] ^= t;
    xoshiro_state[3] = rotl(xoshiro_state[3], 45);
    return result;
}

static void xoshiro_seed(uint64_t seed) {
    // SplitMix64 to initialize state
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        xoshiro_state[i] = z ^ (z >> 31);
    }
}

static double random_uniform(void) {
    return (xoshiro_next() >> 11) * 0x1.0p-53;
}

static double random_uniform_range(double a, double b) {
    return a + (b - a) * random_uniform();
}

// ============================================================================
// XXZ HAMILTONIAN
// ============================================================================

xxz_hamiltonian_t *xxz_hamiltonian_create(uint32_t num_sites,
                                           double J, double delta,
                                           double disorder_strength,
                                           bool periodic_bc,
                                           uint64_t seed) {
    xxz_hamiltonian_t *h = malloc(sizeof(xxz_hamiltonian_t));
    if (!h) return NULL;

    h->num_sites = num_sites;
    h->J = J;
    h->delta = delta;
    h->disorder_strength = disorder_strength;
    h->periodic_bc = periodic_bc;
    h->disorder_seed = seed;

    h->random_fields = malloc(num_sites * sizeof(double));
    if (!h->random_fields) {
        free(h);
        return NULL;
    }

    // Generate random fields from uniform distribution [-W, W]
    xoshiro_seed(seed);
    for (uint32_t i = 0; i < num_sites; i++) {
        h->random_fields[i] = random_uniform_range(-disorder_strength, disorder_strength);
    }

    return h;
}

void xxz_hamiltonian_free(xxz_hamiltonian_t *h) {
    if (!h) return;
    free(h->random_fields);
    free(h);
}

/**
 * @brief Count bits set in integer (population count)
 */
static inline int popcount64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}

/**
 * @brief Get spin value at site (±0.5)
 */
static inline double spin_z(uint64_t state, uint32_t site) {
    return ((state >> site) & 1) ? 0.5 : -0.5;
}

sparse_hamiltonian_t *xxz_build_sparse(const xxz_hamiltonian_t *xxz) {
    if (!xxz) return NULL;

    uint32_t L = xxz->num_sites;
    uint32_t dim = 1U << L;

    sparse_hamiltonian_t *h = malloc(sizeof(sparse_hamiltonian_t));
    if (!h) return NULL;

    h->dim = dim;
    h->eigenvalues = NULL;
    h->eigenvectors = NULL;
    h->eigensystem_computed = false;

    // First pass: count non-zeros
    // Diagonal: dim entries (random fields + Sz-Sz terms)
    // Off-diagonal: each bond contributes 2 entries for S+S- + S-S+ terms
    uint32_t num_bonds = xxz->periodic_bc ? L : (L - 1);
    uint32_t max_nnz = dim + 2 * num_bonds * dim;  // Upper bound

    h->values = malloc(max_nnz * sizeof(double complex));
    h->col_indices = malloc(max_nnz * sizeof(uint32_t));
    h->row_ptr = malloc((dim + 1) * sizeof(uint32_t));

    if (!h->values || !h->col_indices || !h->row_ptr) {
        free(h->values);
        free(h->col_indices);
        free(h->row_ptr);
        free(h);
        return NULL;
    }

    // Build CSR matrix
    uint32_t nnz = 0;
    double J = xxz->J;
    double delta = xxz->delta;

    for (uint32_t row = 0; row < dim; row++) {
        h->row_ptr[row] = nnz;

        // Store entries for this row in sorted order by column
        // We'll use a simple approach: diagonal first, then off-diagonals

        // Diagonal term: Σ_i h_i S^z_i + J*Δ Σ_<ij> S^z_i S^z_j
        double diag = 0.0;

        // Random field term
        for (uint32_t i = 0; i < L; i++) {
            diag += xxz->random_fields[i] * spin_z(row, i);
        }

        // Ising (S^z S^z) term
        for (uint32_t i = 0; i < L; i++) {
            uint32_t j = (i + 1) % L;
            if (!xxz->periodic_bc && j == 0) continue;  // Skip wrap-around for OBC

            diag += J * delta * spin_z(row, i) * spin_z(row, j);
        }

        // Off-diagonal terms: (J/2)(S^+_i S^-_j + S^-_i S^+_j)
        // S^+ |↓⟩ = |↑⟩, S^- |↑⟩ = |↓⟩
        // S^+_i S^-_j flips spin i up and spin j down

        // Collect all off-diagonal entries for this row
        uint32_t off_diag_cols[64];  // Max 64 sites = 63 bonds
        double complex off_diag_vals[64];
        uint32_t num_off = 0;

        for (uint32_t i = 0; i < L; i++) {
            uint32_t j = (i + 1) % L;
            if (!xxz->periodic_bc && j == 0) continue;

            int si = (row >> i) & 1;  // 1 = up, 0 = down
            int sj = (row >> j) & 1;

            // S^+_i S^-_j: requires site i down, site j up
            if (si == 0 && sj == 1) {
                uint32_t new_state = row ^ (1U << i) ^ (1U << j);
                off_diag_cols[num_off] = new_state;
                off_diag_vals[num_off] = J * 0.5;  // J/2 coefficient
                num_off++;
            }

            // S^-_i S^+_j: requires site i up, site j down
            if (si == 1 && sj == 0) {
                uint32_t new_state = row ^ (1U << i) ^ (1U << j);
                off_diag_cols[num_off] = new_state;
                off_diag_vals[num_off] = J * 0.5;
                num_off++;
            }
        }

        // Sort off-diagonal entries by column index and merge with diagonal
        // Simple insertion sort (small number of off-diagonals per row)
        for (uint32_t i = 1; i < num_off; i++) {
            uint32_t key_col = off_diag_cols[i];
            double complex key_val = off_diag_vals[i];
            int j = i - 1;
            while (j >= 0 && off_diag_cols[j] > key_col) {
                off_diag_cols[j + 1] = off_diag_cols[j];
                off_diag_vals[j + 1] = off_diag_vals[j];
                j--;
            }
            off_diag_cols[j + 1] = key_col;
            off_diag_vals[j + 1] = key_val;
        }

        // Insert entries in column order
        uint32_t off_idx = 0;
        while (off_idx < num_off && off_diag_cols[off_idx] < row) {
            h->col_indices[nnz] = off_diag_cols[off_idx];
            h->values[nnz] = off_diag_vals[off_idx];
            nnz++;
            off_idx++;
        }

        // Diagonal
        h->col_indices[nnz] = row;
        h->values[nnz] = diag;
        nnz++;

        // Remaining off-diagonals
        while (off_idx < num_off) {
            h->col_indices[nnz] = off_diag_cols[off_idx];
            h->values[nnz] = off_diag_vals[off_idx];
            nnz++;
            off_idx++;
        }
    }

    h->row_ptr[dim] = nnz;
    h->nnz = nnz;

    return h;
}

void sparse_hamiltonian_free(sparse_hamiltonian_t *h) {
    if (!h) return;
    free(h->values);
    free(h->col_indices);
    free(h->row_ptr);
    free(h->eigenvalues);
    free(h->eigenvectors);
    free(h);
}

qs_error_t sparse_hamiltonian_diagonalize(sparse_hamiltonian_t *h) {
    if (!h) return QS_ERROR_INVALID_STATE;
    if (h->eigensystem_computed) return QS_SUCCESS;

    uint32_t dim = h->dim;

    // Convert sparse to dense for LAPACK
    double complex *dense = calloc(dim * dim, sizeof(double complex));
    if (!dense) return QS_ERROR_OUT_OF_MEMORY;

    for (uint32_t row = 0; row < dim; row++) {
        for (uint32_t idx = h->row_ptr[row]; idx < h->row_ptr[row + 1]; idx++) {
            uint32_t col = h->col_indices[idx];
            dense[row + col * dim] = h->values[idx];  // Column-major for LAPACK
        }
    }

    // Allocate eigenvalue/eigenvector storage
    h->eigenvalues = malloc(dim * sizeof(double));
    h->eigenvectors = malloc(dim * dim * sizeof(double complex));
    if (!h->eigenvalues || !h->eigenvectors) {
        free(dense);
        free(h->eigenvalues);
        free(h->eigenvectors);
        h->eigenvalues = NULL;
        h->eigenvectors = NULL;
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // LAPACK zheev: compute all eigenvalues and eigenvectors
    char jobz = 'V';  // Compute eigenvectors
    char uplo = 'L';  // Lower triangle stored
    int n = (int)dim;
    int lda = n;
    int lwork = -1;
    int info;
    double complex work_query;
    double *rwork = malloc(3 * dim * sizeof(double));

    if (!rwork) {
        free(dense);
        free(h->eigenvalues);
        free(h->eigenvectors);
        h->eigenvalues = NULL;
        h->eigenvectors = NULL;
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // Query optimal workspace
#ifdef __APPLE__
    __CLPK_integer n_clpk = n;
    __CLPK_integer lda_clpk = lda;
    __CLPK_integer lwork_clpk = lwork;
    __CLPK_integer info_clpk;
    __CLPK_doublecomplex work_q;

    zheev_(&jobz, &uplo, &n_clpk, (__CLPK_doublecomplex *)dense, &lda_clpk,
           h->eigenvalues, &work_q, &lwork_clpk, rwork, &info_clpk);
    info = (int)info_clpk;
    lwork = (int)creal(*(double complex *)&work_q);
#else
    zheev_(&jobz, &uplo, &n, dense, &lda, h->eigenvalues, &work_query, &lwork, rwork, &info);
    lwork = (int)creal(work_query);
#endif

    double complex *work = malloc(lwork * sizeof(double complex));
    if (!work) {
        free(dense);
        free(rwork);
        free(h->eigenvalues);
        free(h->eigenvectors);
        h->eigenvalues = NULL;
        h->eigenvectors = NULL;
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // Actual diagonalization
#ifdef __APPLE__
    lwork_clpk = lwork;
    zheev_(&jobz, &uplo, &n_clpk, (__CLPK_doublecomplex *)dense, &lda_clpk,
           h->eigenvalues, (__CLPK_doublecomplex *)work, &lwork_clpk, rwork, &info_clpk);
    info = (int)info_clpk;
#else
    zheev_(&jobz, &uplo, &n, dense, &lda, h->eigenvalues, work, &lwork, rwork, &info);
#endif

    free(work);
    free(rwork);

    if (info != 0) {
        free(dense);
        free(h->eigenvalues);
        free(h->eigenvectors);
        h->eigenvalues = NULL;
        h->eigenvectors = NULL;
        return QS_ERROR_INVALID_STATE;
    }

    // Copy eigenvectors
    memcpy(h->eigenvectors, dense, dim * dim * sizeof(double complex));
    free(dense);

    h->eigensystem_computed = true;
    return QS_SUCCESS;
}

// ============================================================================
// LEVEL STATISTICS
// ============================================================================

level_statistics_t *compute_level_statistics(const double *eigenvalues,
                                              uint32_t num_eigenvalues,
                                              double filter_edges) {
    if (!eigenvalues || num_eigenvalues < 3) return NULL;

    level_statistics_t *stats = malloc(sizeof(level_statistics_t));
    if (!stats) return NULL;

    // Filter out edge of spectrum (typically 10% on each side)
    uint32_t skip = (uint32_t)(filter_edges * num_eigenvalues);
    if (skip >= num_eigenvalues / 2) skip = 0;

    uint32_t start = skip;
    uint32_t end = num_eigenvalues - skip;
    uint32_t n = end - start;

    if (n < 3) {
        free(stats);
        return NULL;
    }

    // Compute level spacings
    uint32_t num_ratios = n - 2;
    stats->ratios = malloc(num_ratios * sizeof(double));
    if (!stats->ratios) {
        free(stats);
        return NULL;
    }

    stats->num_ratios = num_ratios;

    // r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    double sum = 0.0;
    double sum_sq = 0.0;

    for (uint32_t i = 0; i < num_ratios; i++) {
        double s1 = eigenvalues[start + i + 1] - eigenvalues[start + i];
        double s2 = eigenvalues[start + i + 2] - eigenvalues[start + i + 1];

        double r;
        if (s1 < 1e-15 && s2 < 1e-15) {
            r = 1.0;  // Degenerate case
        } else if (s1 < s2) {
            r = s1 / s2;
        } else {
            r = s2 / s1;
        }

        stats->ratios[i] = r;
        sum += r;
        sum_sq += r * r;
    }

    stats->mean_ratio = sum / num_ratios;
    stats->std_ratio = sqrt(sum_sq / num_ratios - stats->mean_ratio * stats->mean_ratio);
    stats->poisson_distance = fabs(stats->mean_ratio - POISSON_R);
    stats->goe_distance = fabs(stats->mean_ratio - GOE_R);

    return stats;
}

void level_statistics_free(level_statistics_t *stats) {
    if (!stats) return;
    free(stats->ratios);
    free(stats);
}

int classify_phase_from_levels(const level_statistics_t *stats) {
    if (!stats) return -1;

    // Threshold at midpoint
    double midpoint = (POISSON_R + GOE_R) / 2.0;  // ~0.4585

    if (stats->mean_ratio < midpoint - 0.02) {
        return 1;  // MBL (Poisson-like)
    } else if (stats->mean_ratio > midpoint + 0.02) {
        return 0;  // Thermal (GOE-like)
    } else {
        return -1;  // Inconclusive (near critical point)
    }
}

// ============================================================================
// TIME EVOLUTION
// ============================================================================

/**
 * @brief Sparse matrix-vector product y = H * x
 */
static void sparse_matvec(const sparse_hamiltonian_t *h,
                          const double complex *x, double complex *y) {
    uint32_t dim = h->dim;

    for (uint32_t row = 0; row < dim; row++) {
        double complex sum = 0.0;
        for (uint32_t idx = h->row_ptr[row]; idx < h->row_ptr[row + 1]; idx++) {
            sum += h->values[idx] * x[h->col_indices[idx]];
        }
        y[row] = sum;
    }
}

qs_error_t mbl_evolve_exact(quantum_state_t *state,
                             const sparse_hamiltonian_t *h,
                             double time) {
    if (!state || !h) return QS_ERROR_INVALID_STATE;
    if (!h->eigensystem_computed) return QS_ERROR_INVALID_STATE;

    uint32_t dim = h->dim;
    if (state->state_dim != dim) return QS_ERROR_INVALID_DIMENSION;

    // Project state onto eigenbasis: c_n = ⟨n|ψ⟩
    double complex *coeffs = malloc(dim * sizeof(double complex));
    double complex *result = malloc(dim * sizeof(double complex));
    if (!coeffs || !result) {
        free(coeffs);
        free(result);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // c_n = Σ_i V^†_{ni} ψ_i = Σ_i V*_{in} ψ_i
    for (uint32_t n = 0; n < dim; n++) {
        double complex c = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            c += conj(h->eigenvectors[i + n * dim]) * state->amplitudes[i];
        }
        coeffs[n] = c;
    }

    // Apply phase evolution: c_n → c_n * exp(-i E_n t)
    for (uint32_t n = 0; n < dim; n++) {
        coeffs[n] *= cexp(-I * h->eigenvalues[n] * time);
    }

    // Transform back: ψ_i = Σ_n V_{in} c_n
    for (uint32_t i = 0; i < dim; i++) {
        double complex sum = 0.0;
        for (uint32_t n = 0; n < dim; n++) {
            sum += h->eigenvectors[i + n * dim] * coeffs[n];
        }
        result[i] = sum;
    }

    memcpy(state->amplitudes, result, dim * sizeof(double complex));

    free(coeffs);
    free(result);
    return QS_SUCCESS;
}

qs_error_t mbl_evolve_krylov(quantum_state_t *state,
                              const sparse_hamiltonian_t *h,
                              double time, uint32_t krylov_dim) {
    if (!state || !h) return QS_ERROR_INVALID_STATE;

    uint32_t dim = h->dim;
    if (state->state_dim != dim) return QS_ERROR_INVALID_DIMENSION;
    if (krylov_dim > dim) krylov_dim = dim;
    if (krylov_dim < 2) krylov_dim = 2;

    // Allocate Krylov basis vectors and tridiagonal matrix
    double complex **V = malloc(krylov_dim * sizeof(double complex *));
    double *alpha = calloc(krylov_dim, sizeof(double));  // Diagonal
    double *beta = calloc(krylov_dim, sizeof(double));   // Off-diagonal
    double complex *w = malloc(dim * sizeof(double complex));

    if (!V || !alpha || !beta || !w) {
        free(V);
        free(alpha);
        free(beta);
        free(w);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t k = 0; k < krylov_dim; k++) {
        V[k] = malloc(dim * sizeof(double complex));
        if (!V[k]) {
            for (uint32_t j = 0; j < k; j++) free(V[j]);
            free(V);
            free(alpha);
            free(beta);
            free(w);
            return QS_ERROR_OUT_OF_MEMORY;
        }
    }

    // Lanczos iteration
    // v_0 = |ψ⟩ / ||ψ||
    double norm = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
        norm += cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
    }
    norm = sqrt(norm);
    for (uint32_t i = 0; i < dim; i++) {
        V[0][i] = state->amplitudes[i] / norm;
    }

    for (uint32_t k = 0; k < krylov_dim; k++) {
        // w = H * v_k
        sparse_matvec(h, V[k], w);

        // α_k = ⟨v_k|w⟩
        double complex a = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            a += conj(V[k][i]) * w[i];
        }
        alpha[k] = creal(a);  // Should be real for Hermitian H

        // w = w - α_k v_k - β_{k-1} v_{k-1}
        for (uint32_t i = 0; i < dim; i++) {
            w[i] -= alpha[k] * V[k][i];
            if (k > 0) {
                w[i] -= beta[k - 1] * V[k - 1][i];
            }
        }

        // Reorthogonalization (full) for numerical stability
        for (uint32_t j = 0; j <= k; j++) {
            double complex overlap = 0.0;
            for (uint32_t i = 0; i < dim; i++) {
                overlap += conj(V[j][i]) * w[i];
            }
            for (uint32_t i = 0; i < dim; i++) {
                w[i] -= overlap * V[j][i];
            }
        }

        // β_k = ||w||
        double b = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            b += cabs(w[i]) * cabs(w[i]);
        }
        b = sqrt(b);
        beta[k] = b;

        // v_{k+1} = w / β_k
        if (k + 1 < krylov_dim && b > 1e-14) {
            for (uint32_t i = 0; i < dim; i++) {
                V[k + 1][i] = w[i] / b;
            }
        } else if (b <= 1e-14) {
            // Invariant subspace reached
            krylov_dim = k + 1;
            break;
        }
    }

    // Diagonalize tridiagonal matrix in Krylov space
    double *krylov_evals = malloc(krylov_dim * sizeof(double));
    double *krylov_evecs = malloc(krylov_dim * krylov_dim * sizeof(double));

    if (!krylov_evals || !krylov_evecs) {
        for (uint32_t k = 0; k < krylov_dim; k++) free(V[k]);
        free(V);
        free(alpha);
        free(beta);
        free(w);
        free(krylov_evals);
        free(krylov_evecs);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // Build dense tridiagonal matrix (symmetric)
    memset(krylov_evecs, 0, krylov_dim * krylov_dim * sizeof(double));
    for (uint32_t k = 0; k < krylov_dim; k++) {
        krylov_evecs[k + k * krylov_dim] = alpha[k];
        if (k + 1 < krylov_dim) {
            krylov_evecs[k + (k + 1) * krylov_dim] = beta[k];
            krylov_evecs[(k + 1) + k * krylov_dim] = beta[k];
        }
    }

    // LAPACK dsyev for real symmetric matrix
#ifdef __APPLE__
    char jobz = 'V';
    char uplo = 'U';
    __CLPK_integer n = krylov_dim;
    __CLPK_integer lwork = -1;
    __CLPK_integer info;
    double work_query;

    dsyev_(&jobz, &uplo, &n, krylov_evecs, &n, krylov_evals, &work_query, &lwork, &info);
    lwork = (int)work_query;

    double *work = malloc(lwork * sizeof(double));
    if (!work) {
        for (uint32_t k = 0; k < krylov_dim; k++) free(V[k]);
        free(V);
        free(alpha);
        free(beta);
        free(w);
        free(krylov_evals);
        free(krylov_evecs);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    dsyev_(&jobz, &uplo, &n, krylov_evecs, &n, krylov_evals, work, &lwork, &info);
    free(work);
#else
    // QL algorithm with implicit shifts for symmetric tridiagonal matrix
    // This is a standard numerical algorithm (same as LAPACK dsteqr)

    // Copy diagonal and subdiagonal for QL algorithm
    double *d = malloc(krylov_dim * sizeof(double));
    double *e = malloc(krylov_dim * sizeof(double));
    if (!d || !e) {
        free(d);
        free(e);
        for (uint32_t k = 0; k < krylov_dim; k++) free(V[k]);
        free(V);
        free(alpha);
        free(beta);
        free(w);
        free(krylov_evals);
        free(krylov_evecs);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < krylov_dim; i++) {
        d[i] = alpha[i];
        e[i] = (i < krylov_dim - 1) ? beta[i] : 0.0;
    }

    // Initialize eigenvector matrix to identity
    for (uint32_t i = 0; i < krylov_dim; i++) {
        for (uint32_t j = 0; j < krylov_dim; j++) {
            krylov_evecs[i + j * krylov_dim] = (i == j) ? 1.0 : 0.0;
        }
    }

    // QL iterations with implicit shift
    const int max_iter = 30;
    const double eps = 1e-15;

    for (uint32_t l = 0; l < krylov_dim; l++) {
        int iter = 0;
        uint32_t m;

        do {
            // Find small subdiagonal element
            for (m = l; m < krylov_dim - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= eps * dd) break;
            }

            if (m != l) {
                if (iter++ >= max_iter) {
                    // Convergence failure - return best estimate
                    break;
                }

                // Form shift from bottom 2x2 minor
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0 ? fabs(r) : -fabs(r)));

                double s = 1.0;
                double c = 1.0;
                double p = 0.0;

                // QL transformation from m-1 to l
                for (int i = (int)m - 1; i >= (int)l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];

                    // Givens rotation
                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        e[i + 1] = f * r;
                        s = 1.0 / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        e[i + 1] = g * r;
                        c = 1.0 / r;
                        s *= c;
                    }

                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;

                    // Update eigenvectors
                    for (uint32_t k = 0; k < krylov_dim; k++) {
                        double temp = krylov_evecs[k + (i + 1) * krylov_dim];
                        krylov_evecs[k + (i + 1) * krylov_dim] =
                            s * krylov_evecs[k + i * krylov_dim] + c * temp;
                        krylov_evecs[k + i * krylov_dim] =
                            c * krylov_evecs[k + i * krylov_dim] - s * temp;
                    }
                }

                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    // Sort eigenvalues and eigenvectors in ascending order
    for (uint32_t i = 0; i < krylov_dim - 1; i++) {
        uint32_t k = i;
        double p = d[i];
        for (uint32_t j = i + 1; j < krylov_dim; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            // Swap eigenvector columns
            for (uint32_t j = 0; j < krylov_dim; j++) {
                double temp = krylov_evecs[j + i * krylov_dim];
                krylov_evecs[j + i * krylov_dim] = krylov_evecs[j + k * krylov_dim];
                krylov_evecs[j + k * krylov_dim] = temp;
            }
        }
    }

    // Copy eigenvalues
    memcpy(krylov_evals, d, krylov_dim * sizeof(double));

    free(d);
    free(e);
#endif

    // Time evolve in Krylov space
    // exp(-iHt)|ψ⟩ ≈ V * exp(-iTt) * e_0 where T is tridiagonal
    // e_0 = (1, 0, 0, ...)

    double complex *krylov_coeff = calloc(krylov_dim, sizeof(double complex));
    if (!krylov_coeff) {
        for (uint32_t k = 0; k < krylov_dim; k++) free(V[k]);
        free(V);
        free(alpha);
        free(beta);
        free(w);
        free(krylov_evals);
        free(krylov_evecs);
        return QS_ERROR_OUT_OF_MEMORY;
    }

    // Transform e_0 to eigenbasis, apply phases, transform back
    for (uint32_t n = 0; n < krylov_dim; n++) {
        double c = krylov_evecs[0 + n * krylov_dim];  // First component of eigenvector n
        double complex phase = cexp(-I * krylov_evals[n] * time);

        // Add contribution to all Krylov coefficients
        for (uint32_t k = 0; k < krylov_dim; k++) {
            krylov_coeff[k] += c * phase * krylov_evecs[k + n * krylov_dim];
        }
    }

    // Transform back to full Hilbert space
    // |ψ(t)⟩ = Σ_k c_k |v_k⟩
    memset(state->amplitudes, 0, dim * sizeof(double complex));
    for (uint32_t k = 0; k < krylov_dim; k++) {
        for (uint32_t i = 0; i < dim; i++) {
            state->amplitudes[i] += krylov_coeff[k] * V[k][i];
        }
    }

    // Normalize (Krylov approximation may not be exactly unitary)
    norm = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
        norm += cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
    }
    norm = sqrt(norm);
    for (uint32_t i = 0; i < dim; i++) {
        state->amplitudes[i] /= norm;
    }

    // Cleanup
    for (uint32_t k = 0; k < krylov_dim; k++) free(V[k]);
    free(V);
    free(alpha);
    free(beta);
    free(w);
    free(krylov_evals);
    free(krylov_evecs);
    free(krylov_coeff);

    return QS_SUCCESS;
}

/**
 * @brief Apply two-site XX+YY gate: exp(-i dt J/2 (S+S- + S-S+))
 *
 * For sites i and i+1, the non-zero elements are:
 * |01⟩ and |10⟩ states swap with phase exp(-i J dt / 2)
 */
static void apply_xx_yy_gate(double complex *amplitudes, size_t dim,
                              uint32_t site, double J, double dt) {
    double theta = J * dt / 2.0;
    double c = cos(theta);
    double complex s = -I * sin(theta);

    uint32_t mask_i = 1U << site;
    uint32_t mask_j = 1U << (site + 1);
    uint32_t mask_both = mask_i | mask_j;

    // Process all computational basis states
    for (size_t idx = 0; idx < dim; idx++) {
        // Only process states where exactly one of {site, site+1} is set
        uint32_t bits = (idx & mask_both);
        if (bits == mask_i || bits == mask_j) {
            // Find partner state (flip both bits)
            size_t partner = idx ^ mask_both;
            if (idx < partner) {
                // Apply 2x2 rotation to (idx, partner) pair
                double complex a0 = amplitudes[idx];
                double complex a1 = amplitudes[partner];
                amplitudes[idx]     = c * a0 + s * a1;
                amplitudes[partner] = s * a0 + c * a1;
            }
        }
    }
}

/**
 * @brief Apply diagonal ZZ and h_i Sz terms: exp(-i dt H_diag)
 *
 * H_diag = Δ Σ S^z_i S^z_{i+1} + Σ h_i S^z_i
 */
static void apply_diagonal_phase(double complex *amplitudes, size_t dim,
                                  uint32_t num_sites, double delta,
                                  const double *h_fields, double dt,
                                  bool periodic_bc) {
    for (size_t idx = 0; idx < dim; idx++) {
        double diag = 0.0;

        // Sz-Sz terms
        uint32_t num_bonds = periodic_bc ? num_sites : (num_sites - 1);
        for (uint32_t bond = 0; bond < num_bonds; bond++) {
            uint32_t i = bond;
            uint32_t j = (bond + 1) % num_sites;
            double sz_i = ((idx >> i) & 1) ? 0.5 : -0.5;
            double sz_j = ((idx >> j) & 1) ? 0.5 : -0.5;
            diag += delta * sz_i * sz_j;
        }

        // On-site disorder terms
        for (uint32_t i = 0; i < num_sites; i++) {
            double sz = ((idx >> i) & 1) ? 0.5 : -0.5;
            diag += h_fields[i] * sz;
        }

        amplitudes[idx] *= cexp(-I * diag * dt);
    }
}

/**
 * @brief Trotter evolution for XXZ Hamiltonian
 *
 * Uses second-order Suzuki-Trotter decomposition:
 * exp(-iHt) ≈ [exp(-iH_diag dt/2) exp(-iH_even dt/2) exp(-iH_odd dt)
 *              exp(-iH_even dt/2) exp(-iH_diag dt/2)]^n
 *
 * where n = total_time / dt and dt is the Trotter step size.
 */
static qs_error_t mbl_evolve_trotter(quantum_state_t *state,
                                      const xxz_hamiltonian_t *xxz,
                                      double total_time,
                                      uint32_t num_steps) {
    if (!state || !xxz) return QS_ERROR_INVALID_STATE;
    if (state->num_qubits != xxz->num_sites) return QS_ERROR_INVALID_DIMENSION;

    double dt = total_time / num_steps;
    uint32_t L = xxz->num_sites;
    size_t dim = state->state_dim;

    for (uint32_t step = 0; step < num_steps; step++) {
        // Half step of diagonal terms
        apply_diagonal_phase(state->amplitudes, dim, L, xxz->delta,
                            xxz->random_fields, dt / 2.0, xxz->periodic_bc);

        // Half step of even bonds (0-1, 2-3, 4-5, ...)
        for (uint32_t site = 0; site < L - 1; site += 2) {
            apply_xx_yy_gate(state->amplitudes, dim, site, xxz->J, dt / 2.0);
        }

        // Full step of odd bonds (1-2, 3-4, 5-6, ...)
        for (uint32_t site = 1; site < L - 1; site += 2) {
            apply_xx_yy_gate(state->amplitudes, dim, site, xxz->J, dt);
        }

        // Handle periodic boundary condition
        if (xxz->periodic_bc && L > 2) {
            // Bond L-1 to 0 (wraps around)
            // This requires special handling since bits aren't adjacent
            double theta = xxz->J * dt / 2.0;  // Even bond → dt/2
            double c = cos(theta);
            double complex s_phase = -I * sin(theta);

            uint32_t mask_last = 1U << (L - 1);
            uint32_t mask_first = 1U;

            for (size_t idx = 0; idx < dim; idx++) {
                uint32_t bits = ((idx & mask_last) ? 2 : 0) | (idx & mask_first);
                if (bits == 1 || bits == 2) {  // Exactly one set
                    size_t partner = idx ^ (mask_last | mask_first);
                    if (idx < partner) {
                        double complex a0 = state->amplitudes[idx];
                        double complex a1 = state->amplitudes[partner];
                        state->amplitudes[idx]     = c * a0 + s_phase * a1;
                        state->amplitudes[partner] = s_phase * a0 + c * a1;
                    }
                }
            }
        }

        // Half step of even bonds again
        for (uint32_t site = 0; site < L - 1; site += 2) {
            apply_xx_yy_gate(state->amplitudes, dim, site, xxz->J, dt / 2.0);
        }

        // Half step of diagonal terms again
        apply_diagonal_phase(state->amplitudes, dim, L, xxz->delta,
                            xxz->random_fields, dt / 2.0, xxz->periodic_bc);
    }

    // Normalize (Trotter is exactly unitary in exact arithmetic,
    // but floating point errors may accumulate)
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_sq += cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
    }
    double norm = sqrt(norm_sq);
    if (norm > 1e-15) {
        for (size_t i = 0; i < dim; i++) {
            state->amplitudes[i] /= norm;
        }
    }

    return QS_SUCCESS;
}

// Global variable to store current XXZ parameters for Trotter (set by user)
static const xxz_hamiltonian_t *g_current_xxz = NULL;

void mbl_set_xxz_for_trotter(const xxz_hamiltonian_t *xxz) {
    g_current_xxz = xxz;
}

qs_error_t mbl_time_evolve(quantum_state_t *state,
                            const sparse_hamiltonian_t *h,
                            double time,
                            evolution_method_t method) {
    switch (method) {
        case EVOLUTION_EXACT:
            return mbl_evolve_exact(state, h, time);
        case EVOLUTION_KRYLOV:
            return mbl_evolve_krylov(state, h, time, 30);  // Default Krylov dim
        case EVOLUTION_TROTTER:
            // Trotter requires the XXZ Hamiltonian structure
            if (g_current_xxz != NULL) {
                // Use 100 Trotter steps by default (dt = time/100)
                // This gives O(dt^2) = O(time^2/10000) error per step
                uint32_t num_steps = (uint32_t)(100.0 * fabs(time) + 1);
                return mbl_evolve_trotter(state, g_current_xxz, time, num_steps);
            }
            // Fall back to Krylov if XXZ structure not available
            return mbl_evolve_krylov(state, h, time, 30);
        default:
            return QS_ERROR_INVALID_STATE;
    }
}

// ============================================================================
// INITIAL STATES
// ============================================================================

qs_error_t prepare_neel_state(quantum_state_t *state) {
    if (!state) return QS_ERROR_INVALID_STATE;

    // |↑↓↑↓...⟩ = |101010...⟩ in computational basis
    // where |1⟩ = |↑⟩, |0⟩ = |↓⟩

    quantum_state_reset(state);

    // Find the Néel state index: alternating 1s and 0s
    uint64_t neel_index = 0;
    for (size_t i = 0; i < state->num_qubits; i += 2) {
        neel_index |= (1ULL << i);
    }

    state->amplitudes[neel_index] = 1.0;

    return QS_SUCCESS;
}

qs_error_t prepare_domain_wall_state(quantum_state_t *state) {
    if (!state) return QS_ERROR_INVALID_STATE;

    quantum_state_reset(state);

    // |↑↑...↓↓⟩ = first half up, second half down
    size_t half = state->num_qubits / 2;
    uint64_t domain_wall_index = 0;
    for (size_t i = 0; i < half; i++) {
        domain_wall_index |= (1ULL << i);
    }

    state->amplitudes[domain_wall_index] = 1.0;

    return QS_SUCCESS;
}

qs_error_t prepare_random_product_state(quantum_state_t *state, uint64_t seed) {
    if (!state) return QS_ERROR_INVALID_STATE;

    quantum_state_reset(state);

    xoshiro_seed(seed);
    uint64_t index = 0;
    for (size_t i = 0; i < state->num_qubits; i++) {
        if (random_uniform() > 0.5) {
            index |= (1ULL << i);
        }
    }

    state->amplitudes[index] = 1.0;

    return QS_SUCCESS;
}

// ============================================================================
// OBSERVABLES
// ============================================================================

double expectation_sz(const quantum_state_t *state, uint32_t site) {
    if (!state || site >= state->num_qubits) return 0.0;

    double result = 0.0;
    for (size_t i = 0; i < state->state_dim; i++) {
        double prob = cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
        double sz = ((i >> site) & 1) ? 0.5 : -0.5;
        result += prob * sz;
    }

    return result;
}

double expectation_sz_total(const quantum_state_t *state) {
    if (!state) return 0.0;

    double result = 0.0;
    for (uint32_t site = 0; site < state->num_qubits; site++) {
        result += expectation_sz(state, site);
    }

    return result;
}

double correlation_sz_sz(const quantum_state_t *state,
                          uint32_t site_i, uint32_t site_j) {
    if (!state || site_i >= state->num_qubits || site_j >= state->num_qubits) {
        return 0.0;
    }

    double result = 0.0;
    for (size_t k = 0; k < state->state_dim; k++) {
        double prob = cabs(state->amplitudes[k]) * cabs(state->amplitudes[k]);
        double sz_i = ((k >> site_i) & 1) ? 0.5 : -0.5;
        double sz_j = ((k >> site_j) & 1) ? 0.5 : -0.5;
        result += prob * sz_i * sz_j;
    }

    return result;
}

double correlation_connected(const quantum_state_t *state,
                              uint32_t site_i, uint32_t site_j) {
    double sz_i = expectation_sz(state, site_i);
    double sz_j = expectation_sz(state, site_j);
    double sz_ij = correlation_sz_sz(state, site_i, site_j);

    return sz_ij - sz_i * sz_j;
}

double expectation_energy(const quantum_state_t *state,
                           const sparse_hamiltonian_t *h) {
    if (!state || !h) return 0.0;
    if (state->state_dim != h->dim) return 0.0;

    // ⟨H⟩ = ⟨ψ|H|ψ⟩ = Σ_{ij} ψ*_i H_{ij} ψ_j
    double complex result = 0.0;

    for (uint32_t i = 0; i < h->dim; i++) {
        for (uint32_t idx = h->row_ptr[i]; idx < h->row_ptr[i + 1]; idx++) {
            uint32_t j = h->col_indices[idx];
            result += conj(state->amplitudes[i]) * h->values[idx] * state->amplitudes[j];
        }
    }

    return creal(result);
}

double energy_variance(const quantum_state_t *state,
                        const sparse_hamiltonian_t *h) {
    if (!state || !h) return 0.0;
    if (state->state_dim != h->dim) return 0.0;

    // ⟨H²⟩ - ⟨H⟩²
    double E = expectation_energy(state, h);

    // Compute H|ψ⟩
    double complex *Hpsi = malloc(h->dim * sizeof(double complex));
    if (!Hpsi) return 0.0;

    sparse_matvec(h, state->amplitudes, Hpsi);

    // ⟨H²⟩ = ⟨ψ|H²|ψ⟩ = ⟨Hψ|Hψ⟩
    double H2 = 0.0;
    for (uint32_t i = 0; i < h->dim; i++) {
        H2 += cabs(Hpsi[i]) * cabs(Hpsi[i]);
    }

    free(Hpsi);

    return H2 - E * E;
}

// ============================================================================
// ENTANGLEMENT DYNAMICS
// ============================================================================

/**
 * @brief Compute bipartite entanglement entropy
 *
 * Traces out complement of subsystem and computes von Neumann entropy
 * of reduced density matrix.
 */
static double compute_bipartite_entropy(const quantum_state_t *state,
                                         const uint32_t *subsystem,
                                         uint32_t num_subsystem) {
    if (!state || !subsystem || num_subsystem == 0) return 0.0;
    if (num_subsystem >= state->num_qubits) return 0.0;

    uint32_t L = state->num_qubits;
    uint32_t dim_A = 1U << num_subsystem;
    uint32_t dim_B = 1U << (L - num_subsystem);

    // Build reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|)
    double complex *rho_A = calloc(dim_A * dim_A, sizeof(double complex));
    if (!rho_A) return 0.0;

    // Create masks for subsystem A and B
    uint32_t mask_A = 0;
    for (uint32_t i = 0; i < num_subsystem; i++) {
        mask_A |= (1U << subsystem[i]);
    }

    // For each pair of basis states that agree on B, contribute to ρ_A
    for (size_t idx1 = 0; idx1 < state->state_dim; idx1++) {
        for (size_t idx2 = 0; idx2 < state->state_dim; idx2++) {
            // Extract A and B parts
            uint32_t a1 = 0, a2 = 0;
            uint32_t b1 = 0, b2 = 0;
            uint32_t a_bit = 0, b_bit = 0;

            for (uint32_t q = 0; q < L; q++) {
                int in_A = 0;
                for (uint32_t k = 0; k < num_subsystem; k++) {
                    if (subsystem[k] == q) {
                        in_A = 1;
                        break;
                    }
                }

                if (in_A) {
                    a1 |= ((idx1 >> q) & 1) << a_bit;
                    a2 |= ((idx2 >> q) & 1) << a_bit;
                    a_bit++;
                } else {
                    b1 |= ((idx1 >> q) & 1) << b_bit;
                    b2 |= ((idx2 >> q) & 1) << b_bit;
                    b_bit++;
                }
            }

            // Only contribute if B indices match (partial trace condition)
            if (b1 == b2) {
                rho_A[a1 + a2 * dim_A] += state->amplitudes[idx1] *
                                          conj(state->amplitudes[idx2]);
            }
        }
    }

    // Diagonalize ρ_A and compute entropy
    double *evals = malloc(dim_A * sizeof(double));
    if (!evals) {
        free(rho_A);
        return 0.0;
    }

#ifdef __APPLE__
    char jobz = 'N';  // Eigenvalues only
    char uplo = 'L';
    __CLPK_integer n = dim_A;
    __CLPK_integer lwork = -1;
    __CLPK_integer info;
    __CLPK_doublecomplex work_query;
    double *rwork = malloc(3 * dim_A * sizeof(double));

    if (!rwork) {
        free(rho_A);
        free(evals);
        return 0.0;
    }

    zheev_(&jobz, &uplo, &n, (__CLPK_doublecomplex *)rho_A, &n, evals,
           &work_query, &lwork, rwork, &info);
    lwork = (int)creal(*(double complex *)&work_query);

    __CLPK_doublecomplex *work = malloc(lwork * sizeof(__CLPK_doublecomplex));
    if (!work) {
        free(rho_A);
        free(evals);
        free(rwork);
        return 0.0;
    }

    zheev_(&jobz, &uplo, &n, (__CLPK_doublecomplex *)rho_A, &n, evals,
           work, &lwork, rwork, &info);
    free(work);
    free(rwork);
#else
    // Jacobi eigenvalue algorithm for Hermitian matrix
    // More robust than power iteration for computing all eigenvalues

    // Work arrays for Jacobi iteration
    double complex *A = malloc(dim_A * dim_A * sizeof(double complex));
    if (!A) {
        free(rho_A);
        free(evals);
        return 0.0;
    }
    memcpy(A, rho_A, dim_A * dim_A * sizeof(double complex));

    const int max_sweeps = 50;
    const double tol = 1e-14;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Find largest off-diagonal element
        double max_off = 0.0;
        uint32_t p_max = 0, q_max = 1;

        for (uint32_t p = 0; p < dim_A; p++) {
            for (uint32_t q = p + 1; q < dim_A; q++) {
                double off = cabs(A[p + q * dim_A]);
                if (off > max_off) {
                    max_off = off;
                    p_max = p;
                    q_max = q;
                }
            }
        }

        // Check convergence
        if (max_off < tol) break;

        // Jacobi rotation to zero out A[p][q]
        uint32_t p = p_max, q = q_max;
        double complex Apq = A[p + q * dim_A];
        double App = creal(A[p + p * dim_A]);
        double Aqq = creal(A[q + q * dim_A]);

        // For Hermitian matrix, compute rotation angle
        double diff = Aqq - App;
        double complex t;

        if (cabs(Apq) < tol * fabs(diff)) {
            t = Apq / diff;
        } else {
            double phi = diff / (2.0 * cabs(Apq));
            double sign_phi = (phi >= 0) ? 1.0 : -1.0;
            t = 1.0 / (fabs(phi) + sqrt(phi * phi + 1.0));
            // Adjust for complex phase
            t *= conj(Apq) / cabs(Apq) * sign_phi;
        }

        double c = 1.0 / sqrt(1.0 + cabs(t) * cabs(t));
        double complex s = c * t;

        // Apply rotation to rows and columns p, q
        for (uint32_t k = 0; k < dim_A; k++) {
            if (k != p && k != q) {
                double complex Akp = A[k + p * dim_A];
                double complex Akq = A[k + q * dim_A];
                A[k + p * dim_A] = c * Akp - conj(s) * Akq;
                A[k + q * dim_A] = s * Akp + c * Akq;
                A[p + k * dim_A] = conj(A[k + p * dim_A]);
                A[q + k * dim_A] = conj(A[k + q * dim_A]);
            }
        }

        // Update diagonal and off-diagonal elements at p, q
        double new_App = App - 2.0 * creal(conj(t) * Apq);
        double new_Aqq = Aqq + 2.0 * creal(conj(t) * Apq);
        A[p + p * dim_A] = new_App;
        A[q + q * dim_A] = new_Aqq;
        A[p + q * dim_A] = 0.0;
        A[q + p * dim_A] = 0.0;
    }

    // Extract eigenvalues from diagonal
    for (uint32_t i = 0; i < dim_A; i++) {
        evals[i] = creal(A[i + i * dim_A]);
        // Clamp to non-negative (density matrix eigenvalues must be >= 0)
        if (evals[i] < 0.0) evals[i] = 0.0;
    }

    free(A);
#endif

    // Compute von Neumann entropy S = -Σ λ log₂(λ)
    double entropy = 0.0;
    for (uint32_t i = 0; i < dim_A; i++) {
        if (evals[i] > 1e-15) {
            entropy -= evals[i] * log2(evals[i]);
        }
    }

    free(rho_A);
    free(evals);

    return entropy;
}

entropy_dynamics_t *simulate_entropy_dynamics(const sparse_hamiltonian_t *h,
                                               const quantum_state_t *initial_state,
                                               const uint32_t *subsystem_qubits,
                                               uint32_t num_subsystem,
                                               double t_max, uint32_t num_steps) {
    if (!h || !initial_state || !subsystem_qubits) return NULL;
    if (num_steps == 0) num_steps = 100;

    entropy_dynamics_t *dyn = malloc(sizeof(entropy_dynamics_t));
    if (!dyn) return NULL;

    dyn->times = malloc(num_steps * sizeof(double));
    dyn->entropies = malloc(num_steps * sizeof(double));
    dyn->num_points = num_steps;

    if (!dyn->times || !dyn->entropies) {
        free(dyn->times);
        free(dyn->entropies);
        free(dyn);
        return NULL;
    }

    // Clone initial state for evolution
    quantum_state_t state;
    if (quantum_state_clone(&state, initial_state) != QS_SUCCESS) {
        free(dyn->times);
        free(dyn->entropies);
        free(dyn);
        return NULL;
    }

    double dt = t_max / (num_steps - 1);

    for (uint32_t step = 0; step < num_steps; step++) {
        double t = step * dt;
        dyn->times[step] = t;

        // Compute entropy at this time
        dyn->entropies[step] = compute_bipartite_entropy(&state, subsystem_qubits,
                                                          num_subsystem);

        // Evolve to next time step (if not last)
        if (step + 1 < num_steps) {
            // Use Krylov for efficiency if eigensystem not available
            if (h->eigensystem_computed) {
                mbl_evolve_exact(&state, h, dt);
            } else {
                mbl_evolve_krylov(&state, (sparse_hamiltonian_t *)h, dt, 20);
            }
        }
    }

    quantum_state_free(&state);

    // Fit saturation value (average of last 10%)
    uint32_t avg_start = num_steps - num_steps / 10;
    if (avg_start < 1) avg_start = 1;

    double sum = 0.0;
    for (uint32_t i = avg_start; i < num_steps; i++) {
        sum += dyn->entropies[i];
    }
    dyn->saturation_value = sum / (num_steps - avg_start);

    // Fit growth exponent (simple log-log regression for S ~ t^α)
    // Exclude early times where S ≈ 0
    double sum_lnt = 0.0, sum_lnS = 0.0, sum_lnt2 = 0.0, sum_lntlnS = 0.0;
    uint32_t n_fit = 0;

    for (uint32_t i = 1; i < num_steps; i++) {
        if (dyn->times[i] > 0.1 && dyn->entropies[i] > 0.01) {
            double lnt = log(dyn->times[i]);
            double lnS = log(dyn->entropies[i]);
            sum_lnt += lnt;
            sum_lnS += lnS;
            sum_lnt2 += lnt * lnt;
            sum_lntlnS += lnt * lnS;
            n_fit++;
        }
    }

    if (n_fit > 2) {
        double denom = n_fit * sum_lnt2 - sum_lnt * sum_lnt;
        if (fabs(denom) > 1e-10) {
            dyn->growth_exponent = (n_fit * sum_lntlnS - sum_lnt * sum_lnS) / denom;
        } else {
            dyn->growth_exponent = 0.0;
        }
    } else {
        dyn->growth_exponent = 0.0;
    }

    // Fit log coefficient (for S ~ c·log(t) fit)
    dyn->log_coefficient = 0.0;
    if (dyn->growth_exponent < 0.5) {
        // Likely logarithmic growth
        double sum_t = 0.0, sum_S = 0.0;
        for (uint32_t i = 1; i < num_steps; i++) {
            if (dyn->times[i] > 0.1) {
                sum_t += log(dyn->times[i]);
                sum_S += dyn->entropies[i];
            }
        }
        if (n_fit > 0) {
            dyn->log_coefficient = sum_S / sum_t;
        }
    }

    return dyn;
}

void entropy_dynamics_free(entropy_dynamics_t *dyn) {
    if (!dyn) return;
    free(dyn->times);
    free(dyn->entropies);
    free(dyn);
}

int fit_entropy_growth(const entropy_dynamics_t *dyn,
                       double *log_fit_quality, double *linear_fit_quality) {
    if (!dyn || dyn->num_points < 5) {
        if (log_fit_quality) *log_fit_quality = 0.0;
        if (linear_fit_quality) *linear_fit_quality = 0.0;
        return -1;
    }

    // Compute R² for S = a + b·log(t) vs S = a + b·t
    double mean_S = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 1; i < dyn->num_points; i++) {
        if (dyn->times[i] > 0.1) {
            mean_S += dyn->entropies[i];
            n++;
        }
    }
    mean_S /= n;

    double SS_tot = 0.0;
    double SS_res_log = 0.0;
    double SS_res_lin = 0.0;

    // Fit parameters for log: S = a + b·log(t)
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (uint32_t i = 1; i < dyn->num_points; i++) {
        if (dyn->times[i] > 0.1) {
            double x = log(dyn->times[i]);
            double y = dyn->entropies[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
    }
    double b_log = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double a_log = (sum_y - b_log * sum_x) / n;

    // Fit parameters for linear: S = a + b·t
    sum_x = sum_y = sum_xy = sum_x2 = 0;
    for (uint32_t i = 1; i < dyn->num_points; i++) {
        if (dyn->times[i] > 0.1) {
            double x = dyn->times[i];
            double y = dyn->entropies[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
    }
    double b_lin = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double a_lin = (sum_y - b_lin * sum_x) / n;

    // Compute residuals
    for (uint32_t i = 1; i < dyn->num_points; i++) {
        if (dyn->times[i] > 0.1) {
            double y = dyn->entropies[i];
            SS_tot += (y - mean_S) * (y - mean_S);

            double pred_log = a_log + b_log * log(dyn->times[i]);
            SS_res_log += (y - pred_log) * (y - pred_log);

            double pred_lin = a_lin + b_lin * dyn->times[i];
            SS_res_lin += (y - pred_lin) * (y - pred_lin);
        }
    }

    double R2_log = 1.0 - SS_res_log / SS_tot;
    double R2_lin = 1.0 - SS_res_lin / SS_tot;

    if (log_fit_quality) *log_fit_quality = R2_log;
    if (linear_fit_quality) *linear_fit_quality = R2_lin;

    return (R2_log > R2_lin) ? 1 : 0;
}

// ============================================================================
// IMBALANCE DYNAMICS
// ============================================================================

imbalance_dynamics_t *simulate_imbalance_dynamics(const sparse_hamiltonian_t *h,
                                                   double t_max,
                                                   uint32_t num_steps) {
    if (!h) return NULL;
    if (num_steps == 0) num_steps = 100;

    // Determine number of sites from dimension
    uint32_t dim = h->dim;
    uint32_t L = 0;
    while ((1U << L) < dim) L++;

    imbalance_dynamics_t *dyn = malloc(sizeof(imbalance_dynamics_t));
    if (!dyn) return NULL;

    dyn->times = malloc(num_steps * sizeof(double));
    dyn->imbalance = malloc(num_steps * sizeof(double));
    dyn->num_points = num_steps;

    if (!dyn->times || !dyn->imbalance) {
        free(dyn->times);
        free(dyn->imbalance);
        free(dyn);
        return NULL;
    }

    // Initialize Néel state
    quantum_state_t state;
    if (quantum_state_init(&state, L) != QS_SUCCESS) {
        free(dyn->times);
        free(dyn->imbalance);
        free(dyn);
        return NULL;
    }
    prepare_neel_state(&state);

    dyn->initial_imbalance = 0.0;
    for (uint32_t i = 0; i < L; i++) {
        int sign = (i % 2 == 0) ? 1 : -1;
        dyn->initial_imbalance += sign * expectation_sz(&state, i);
    }
    dyn->initial_imbalance /= L;

    double dt = t_max / (num_steps - 1);

    for (uint32_t step = 0; step < num_steps; step++) {
        double t = step * dt;
        dyn->times[step] = t;

        // Compute imbalance I(t) = (1/L) Σ_i (-1)^i ⟨S^z_i⟩
        double imb = 0.0;
        for (uint32_t i = 0; i < L; i++) {
            int sign = (i % 2 == 0) ? 1 : -1;
            imb += sign * expectation_sz(&state, i);
        }
        dyn->imbalance[step] = imb / L;

        // Evolve to next time step
        if (step + 1 < num_steps) {
            if (h->eigensystem_computed) {
                mbl_evolve_exact(&state, h, dt);
            } else {
                mbl_evolve_krylov(&state, (sparse_hamiltonian_t *)h, dt, 20);
            }
        }
    }

    quantum_state_free(&state);

    // Fit asymptotic imbalance (average of last 10%)
    uint32_t avg_start = num_steps - num_steps / 10;
    if (avg_start < 1) avg_start = 1;

    double sum = 0.0;
    for (uint32_t i = avg_start; i < num_steps; i++) {
        sum += dyn->imbalance[i];
    }
    dyn->asymptotic_imbalance = sum / (num_steps - avg_start);

    // Fit exponential decay rate (if thermal)
    // I(t) ≈ I_0 exp(-γt) → log(I) = log(I_0) - γt
    double sum_t = 0, sum_logI = 0, sum_t2 = 0, sum_tlogI = 0;
    uint32_t n_fit = 0;

    for (uint32_t i = 0; i < num_steps; i++) {
        if (dyn->imbalance[i] > 0.01) {  // Only positive values
            double t = dyn->times[i];
            double logI = log(dyn->imbalance[i]);
            sum_t += t;
            sum_logI += logI;
            sum_t2 += t * t;
            sum_tlogI += t * logI;
            n_fit++;
        }
    }

    if (n_fit > 2) {
        double denom = n_fit * sum_t2 - sum_t * sum_t;
        if (fabs(denom) > 1e-10) {
            dyn->decay_rate = -(n_fit * sum_tlogI - sum_t * sum_logI) / denom;
        } else {
            dyn->decay_rate = 0.0;
        }
    } else {
        dyn->decay_rate = 0.0;
    }

    return dyn;
}

void imbalance_dynamics_free(imbalance_dynamics_t *dyn) {
    if (!dyn) return;
    free(dyn->times);
    free(dyn->imbalance);
    free(dyn);
}

int classify_phase_from_imbalance(const imbalance_dynamics_t *dyn,
                                   double threshold) {
    if (!dyn) return -1;

    // MBL: persistent imbalance > threshold
    // Thermal: imbalance decays to ~0
    return (fabs(dyn->asymptotic_imbalance) > threshold) ? 1 : 0;
}

// ============================================================================
// LOCAL INTEGRALS OF MOTION
// ============================================================================

liom_system_t *construct_lioms(const sparse_hamiltonian_t *h) {
    if (!h || !h->eigensystem_computed) return NULL;

    uint32_t dim = h->dim;
    uint32_t L = 0;
    while ((1U << L) < dim) L++;

    liom_system_t *sys = malloc(sizeof(liom_system_t));
    if (!sys) return NULL;

    sys->lioms = malloc(L * sizeof(liom_t *));
    sys->num_lioms = L;

    if (!sys->lioms) {
        free(sys);
        return NULL;
    }

    // Construct LIOM for each site
    // τ^z_i = Σ_n |n⟩⟨n| ⟨n|S^z_i|n⟩
    // This is diagonal in energy eigenbasis with eigenvalues ⟨n|S^z_i|n⟩

    for (uint32_t site = 0; site < L; site++) {
        liom_t *liom = malloc(sizeof(liom_t));
        if (!liom) {
            for (uint32_t j = 0; j < site; j++) {
                free(sys->lioms[j]->operator);
                free(sys->lioms[j]->locality_profile);
                free(sys->lioms[j]);
            }
            free(sys->lioms);
            free(sys);
            return NULL;
        }

        liom->site = site;
        liom->num_sites = L;
        liom->operator = calloc(dim * dim, sizeof(double complex));
        liom->locality_profile = calloc(L, sizeof(double));

        if (!liom->operator || !liom->locality_profile) {
            free(liom->operator);
            free(liom->locality_profile);
            free(liom);
            for (uint32_t j = 0; j < site; j++) {
                free(sys->lioms[j]->operator);
                free(sys->lioms[j]->locality_profile);
                free(sys->lioms[j]);
            }
            free(sys->lioms);
            free(sys);
            return NULL;
        }

        // Build τ^z_i in computational basis
        // τ^z_i = V * diag(⟨n|S^z_i|n⟩) * V†
        // where V is the eigenvector matrix

        // First compute diagonal elements in eigenbasis
        double *diag = malloc(dim * sizeof(double));
        if (!diag) {
            free(liom->operator);
            free(liom->locality_profile);
            free(liom);
            for (uint32_t j = 0; j < site; j++) {
                free(sys->lioms[j]->operator);
                free(sys->lioms[j]->locality_profile);
                free(sys->lioms[j]);
            }
            free(sys->lioms);
            free(sys);
            return NULL;
        }

        for (uint32_t n = 0; n < dim; n++) {
            // ⟨n|S^z_i|n⟩ = Σ_k |⟨n|k⟩|² S^z_i(k)
            double sz = 0.0;
            for (uint32_t k = 0; k < dim; k++) {
                double prob = cabs(h->eigenvectors[k + n * dim]) *
                              cabs(h->eigenvectors[k + n * dim]);
                sz += prob * spin_z(k, site);
            }
            diag[n] = sz;
        }

        // Transform back: τ = V * diag * V†
        for (uint32_t i = 0; i < dim; i++) {
            for (uint32_t j = 0; j < dim; j++) {
                double complex sum = 0.0;
                for (uint32_t n = 0; n < dim; n++) {
                    sum += h->eigenvectors[i + n * dim] * diag[n] *
                           conj(h->eigenvectors[j + n * dim]);
                }
                liom->operator[i + j * dim] = sum;
            }
        }

        free(diag);

        // Compute locality profile
        // |⟨τ^z_i|S^z_j⟩| where inner product is Tr(A†B)/dim
        for (uint32_t j = 0; j < L; j++) {
            double overlap = 0.0;
            for (uint32_t k = 0; k < dim; k++) {
                // Diagonal of S^z_j
                double sz_j = spin_z(k, j);
                // Diagonal of τ^z_i
                overlap += creal(liom->operator[k + k * dim]) * sz_j;
            }
            liom->locality_profile[j] = fabs(overlap);
        }

        // Fit localization length from exponential decay
        liom->localization_length = liom_localization_length(liom);

        sys->lioms[site] = liom;
    }

    // Compute mean localization length
    double sum_xi = 0.0;
    for (uint32_t i = 0; i < L; i++) {
        sum_xi += sys->lioms[i]->localization_length;
    }
    sys->mean_loc_length = sum_xi / L;

    // Compute max inter-LIOM overlap (should be small in MBL phase)
    sys->max_overlap = 0.0;
    for (uint32_t i = 0; i < L; i++) {
        for (uint32_t j = i + 1; j < L; j++) {
            // [τ^z_i, τ^z_j] should be ~0
            // Compute ||[A,B]||/||A||||B||
            double norm_comm = 0.0;
            double norm_i = 0.0;
            double norm_j = 0.0;

            for (uint32_t k = 0; k < dim * dim; k++) {
                norm_i += cabs(sys->lioms[i]->operator[k]) *
                          cabs(sys->lioms[i]->operator[k]);
                norm_j += cabs(sys->lioms[j]->operator[k]) *
                          cabs(sys->lioms[j]->operator[k]);
            }

            // Commutator [A,B] = AB - BA
            for (uint32_t row = 0; row < dim; row++) {
                for (uint32_t col = 0; col < dim; col++) {
                    double complex ab = 0.0, ba = 0.0;
                    for (uint32_t k = 0; k < dim; k++) {
                        ab += sys->lioms[i]->operator[row + k * dim] *
                              sys->lioms[j]->operator[k + col * dim];
                        ba += sys->lioms[j]->operator[row + k * dim] *
                              sys->lioms[i]->operator[k + col * dim];
                    }
                    norm_comm += cabs(ab - ba) * cabs(ab - ba);
                }
            }

            double overlap = sqrt(norm_comm) / sqrt(norm_i * norm_j);
            if (overlap > sys->max_overlap) {
                sys->max_overlap = overlap;
            }
        }
    }

    return sys;
}

void liom_system_free(liom_system_t *sys) {
    if (!sys) return;
    for (uint32_t i = 0; i < sys->num_lioms; i++) {
        if (sys->lioms[i]) {
            free(sys->lioms[i]->operator);
            free(sys->lioms[i]->locality_profile);
            free(sys->lioms[i]);
        }
    }
    free(sys->lioms);
    free(sys);
}

double liom_localization_length(const liom_t *liom) {
    if (!liom || !liom->locality_profile) return 0.0;

    uint32_t L = liom->num_sites;
    uint32_t center = liom->site;

    // Fit |⟨τ|S^z_j⟩| ~ exp(-|j - center|/ξ)
    // log(profile) = const - |j - center|/ξ

    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    uint32_t n = 0;

    for (uint32_t j = 0; j < L; j++) {
        if (liom->locality_profile[j] > 1e-10) {
            double x = (j > center) ? (j - center) : (center - j);
            double y = log(liom->locality_profile[j]);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            n++;
        }
    }

    if (n < 3) return L;  // Not enough data, assume delocalized

    double denom = n * sum_x2 - sum_x * sum_x;
    if (fabs(denom) < 1e-10) return L;

    double slope = (n * sum_xy - sum_x * sum_y) / denom;

    // ξ = -1/slope (slope should be negative)
    if (slope >= 0) return L;  // Delocalized

    return -1.0 / slope;
}

// ============================================================================
// PHASE DIAGRAM
// ============================================================================

phase_diagram_t *scan_phase_diagram(uint32_t num_sites,
                                     double J, double delta,
                                     double W_min, double W_max,
                                     uint32_t num_W_points,
                                     uint32_t num_realizations,
                                     bool periodic_bc) {
    if (num_W_points == 0 || num_realizations == 0) return NULL;

    phase_diagram_t *pd = malloc(sizeof(phase_diagram_t));
    if (!pd) return NULL;

    pd->points = malloc(num_W_points * sizeof(phase_point_t));
    pd->num_points = num_W_points;

    if (!pd->points) {
        free(pd);
        return NULL;
    }

    double dW = (num_W_points > 1) ? (W_max - W_min) / (num_W_points - 1) : 0;

    for (uint32_t w_idx = 0; w_idx < num_W_points; w_idx++) {
        double W = W_min + w_idx * dW;
        pd->points[w_idx].disorder_strength = W;

        // Average over disorder realizations
        double sum_r = 0.0;
        double sum_imb = 0.0;
        double sum_ent = 0.0;

        for (uint32_t real = 0; real < num_realizations; real++) {
            uint64_t seed = w_idx * 1000000 + real;

            // Create Hamiltonian
            xxz_hamiltonian_t *xxz = xxz_hamiltonian_create(num_sites, J, delta,
                                                            W, periodic_bc, seed);
            if (!xxz) continue;

            sparse_hamiltonian_t *h = xxz_build_sparse(xxz);
            xxz_hamiltonian_free(xxz);
            if (!h) continue;

            // Diagonalize for level statistics
            if (sparse_hamiltonian_diagonalize(h) == QS_SUCCESS) {
                level_statistics_t *stats = compute_level_statistics(
                    h->eigenvalues, h->dim, 0.1);
                if (stats) {
                    sum_r += stats->mean_ratio;
                    level_statistics_free(stats);
                }
            }

            // Imbalance dynamics (shorter simulation for speed)
            imbalance_dynamics_t *imb = simulate_imbalance_dynamics(h, 50.0, 50);
            if (imb) {
                sum_imb += imb->asymptotic_imbalance;
                imbalance_dynamics_free(imb);
            }

            // Entropy (half-chain cut)
            quantum_state_t state;
            if (quantum_state_init(&state, num_sites) == QS_SUCCESS) {
                prepare_neel_state(&state);
                uint32_t half = num_sites / 2;
                uint32_t *subsys = malloc(half * sizeof(uint32_t));
                if (subsys) {
                    for (uint32_t i = 0; i < half; i++) subsys[i] = i;

                    entropy_dynamics_t *ent = simulate_entropy_dynamics(
                        h, &state, subsys, half, 50.0, 50);
                    if (ent) {
                        sum_ent += ent->saturation_value;
                        entropy_dynamics_free(ent);
                    }
                    free(subsys);
                }
                quantum_state_free(&state);
            }

            sparse_hamiltonian_free(h);
        }

        pd->points[w_idx].mean_r = sum_r / num_realizations;
        pd->points[w_idx].mean_imbalance = sum_imb / num_realizations;
        pd->points[w_idx].mean_entropy = sum_ent / num_realizations;

        // Classify phase
        double midpoint = (POISSON_R + GOE_R) / 2.0;
        if (pd->points[w_idx].mean_r < midpoint - 0.02) {
            pd->points[w_idx].phase = 1;  // MBL
        } else if (pd->points[w_idx].mean_r > midpoint + 0.02) {
            pd->points[w_idx].phase = 0;  // Thermal
        } else {
            pd->points[w_idx].phase = -1;  // Critical
        }
    }

    // Estimate critical disorder
    pd->critical_disorder = estimate_critical_disorder(pd);
    pd->critical_exponent = 0.0;  // Would need finite-size scaling

    return pd;
}

void phase_diagram_free(phase_diagram_t *pd) {
    if (!pd) return;
    free(pd->points);
    free(pd);
}

double estimate_critical_disorder(const phase_diagram_t *pd) {
    if (!pd || pd->num_points < 2) return 0.0;

    // Find W where ⟨r⟩ crosses midpoint
    double midpoint = (POISSON_R + GOE_R) / 2.0;

    for (uint32_t i = 1; i < pd->num_points; i++) {
        double r0 = pd->points[i - 1].mean_r;
        double r1 = pd->points[i].mean_r;
        double W0 = pd->points[i - 1].disorder_strength;
        double W1 = pd->points[i].disorder_strength;

        // Check if midpoint is crossed
        if ((r0 >= midpoint && r1 <= midpoint) ||
            (r0 <= midpoint && r1 >= midpoint)) {
            // Linear interpolation
            double t = (midpoint - r0) / (r1 - r0);
            return W0 + t * (W1 - W0);
        }
    }

    // No crossing found
    return (pd->points[0].disorder_strength +
            pd->points[pd->num_points - 1].disorder_strength) / 2.0;
}
