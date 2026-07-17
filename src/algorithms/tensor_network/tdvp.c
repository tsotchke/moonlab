/**
 * @file tdvp.c
 * @brief TDVP implementation for MPS time evolution
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "tdvp.h"
#include "svd_compress.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#  define MOONLAB_TDVP_HAVE_BLAS 1
#elif defined(MOONLAB_HAVE_CBLAS)
#  include <cblas.h>
#  define MOONLAB_TDVP_HAVE_BLAS 1
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Per-bond PID state for the adaptive-bond controller (v0.4).
 *
 * The forward declaration lives in `tdvp.h` so `tdvp_engine_t` can
 * hold a pointer; the full definition is private to this translation
 * unit.  The struct is defined up front (rather than alongside the
 * bond-truncation helpers below) so that `tdvp_engine_create` can
 * `sizeof` it when allocating the per-bond array.
 */
struct tdvp_bond_pid_state {
    double prev_error;   /**< Previous error signal (S - S_chi). */
    double integral;     /**< Running PID integral across sweeps. */
    uint32_t chi;        /**< Current target bond dimension. */
    bool primed;         /**< True after the first update (no derivative on step 1). */
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static double get_time_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static double complex_dot(const double complex *a, const double complex *b, uint64_t n) {
    double complex sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += conj(a[i]) * b[i];
    }
    return sum;
}

static double vector_norm(const double complex *v, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    }
    return sqrt(sum);
}

static void vector_scale(double complex *v, double complex alpha, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

static void vector_axpy(double complex *y, double complex alpha,
                        const double complex *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

// ============================================================================
// HISTORY MANAGEMENT
// ============================================================================

tdvp_history_t *tdvp_history_create(uint32_t initial_capacity) {
    tdvp_history_t *hist = (tdvp_history_t *)calloc(1, sizeof(tdvp_history_t));
    if (!hist) return NULL;

    hist->capacity = initial_capacity > 0 ? initial_capacity : 100;
    hist->times = (double *)calloc(hist->capacity, sizeof(double));
    hist->energies = (double *)calloc(hist->capacity, sizeof(double));
    hist->norms = (double *)calloc(hist->capacity, sizeof(double));

    if (!hist->times || !hist->energies || !hist->norms) {
        tdvp_history_free(hist);
        return NULL;
    }

    return hist;
}

void tdvp_history_free(tdvp_history_t *hist) {
    if (!hist) return;
    free(hist->times);
    free(hist->energies);
    free(hist->norms);
    free(hist->observables);
    free(hist->bond_chi_history);
    free(hist);
}

/* Append a row to the history's scalar columns and the lazy
 * bond-chi snapshot.  Shared by tdvp_history_add and
 * tdvp_history_add_with_observable; the observable column is
 * handled by the caller because it has its own lazy-allocation
 * trigger. */
static void tdvp_history_grow_and_record(tdvp_history_t *hist,
                                          const tdvp_result_t *result) {
    if (hist->num_steps >= hist->capacity) {
        uint32_t new_cap = hist->capacity * 2;
        hist->times = (double *)realloc(hist->times, new_cap * sizeof(double));
        hist->energies = (double *)realloc(hist->energies, new_cap * sizeof(double));
        hist->norms = (double *)realloc(hist->norms, new_cap * sizeof(double));
        if (hist->observables) {
            hist->observables = (double *)realloc(
                hist->observables, new_cap * sizeof(double));
            /* Zero-fill the freshly grown tail so unrecorded slots
             * read as 0 rather than undefined. */
            for (uint32_t i = hist->capacity; i < new_cap; i++) {
                hist->observables[i] = 0.0;
            }
        }
        if (hist->bond_chi_history && hist->n_bonds > 0) {
            hist->bond_chi_history = (uint32_t *)realloc(
                hist->bond_chi_history,
                (size_t)new_cap * hist->n_bonds * sizeof(uint32_t));
        }
        hist->capacity = new_cap;
    }

    hist->times[hist->num_steps] = result->time;
    hist->energies[hist->num_steps] = result->energy;
    hist->norms[hist->num_steps] = result->norm;

    /* Bond-chi snapshot: lazy-allocate on the first result that
     * actually carries a distribution.  Subsequent calls write into
     * the same flat row-major buffer.  Results without a
     * distribution (legacy path) leave the row zeroed. */
    if (result->bond_chi_distribution && result->n_bonds > 0) {
        if (!hist->bond_chi_history) {
            hist->n_bonds = result->n_bonds;
            hist->bond_chi_history = (uint32_t *)calloc(
                (size_t)hist->capacity * hist->n_bonds, sizeof(uint32_t));
        }
        if (hist->bond_chi_history && result->n_bonds == hist->n_bonds) {
            uint32_t *row = hist->bond_chi_history
                          + (size_t)hist->num_steps * hist->n_bonds;
            for (uint32_t b = 0; b < hist->n_bonds; b++) {
                row[b] = result->bond_chi_distribution[b];
            }
        }
    }
}

void tdvp_history_add(tdvp_history_t *hist, const tdvp_result_t *result) {
    if (!hist || !result) return;
    tdvp_history_grow_and_record(hist, result);
    /* Legacy entry point: do not touch the observables column.  If
     * it has been lazily allocated by a prior
     * tdvp_history_add_with_observable call, leave this step's slot
     * at zero -- callers can interleave the two add functions and
     * tell which steps recorded an observable by checking that the
     * observables pointer is non-NULL. */
    hist->num_steps++;
}

void tdvp_history_add_with_observable(tdvp_history_t *hist,
                                       const tdvp_result_t *result,
                                       double observable) {
    if (!hist || !result) return;
    tdvp_history_grow_and_record(hist, result);

    /* Lazy-allocate the observables column on first use.  Zero-fill
     * any earlier steps that were recorded without an observable so
     * the column stays aligned with `num_steps`. */
    if (!hist->observables) {
        hist->observables = (double *)calloc(hist->capacity, sizeof(double));
    }
    if (hist->observables) {
        hist->observables[hist->num_steps] = observable;
    }
    hist->num_steps++;
}

void tdvp_result_clear(tdvp_result_t *result) {
    if (!result) return;
    free(result->bond_chi_distribution);
    result->bond_chi_distribution = NULL;
    result->n_bonds = 0;
    result->time = 0.0;
    result->energy = 0.0;
    result->norm = 0.0;
    result->truncation_error = 0.0;
    result->max_bond_dim = 0;
    result->step_time = 0.0;
}

// ============================================================================
// LANCZOS MATRIX EXPONENTIAL
// ============================================================================

/**
 * @brief Compute exp(alpha * T) @ e1 for tridiagonal matrix T
 *
 * Uses QL algorithm to diagonalize T, then computes matrix exponential.
 *
 * @param alpha Diagonal elements
 * @param beta Off-diagonal elements
 * @param n Matrix size
 * @param coeff Exponent coefficient
 * @param result Output: result vector (length n)
 */
static void tridiag_expm_e1(const double *alpha, const double *beta,
                             uint32_t n, double complex coeff,
                             double complex *result) {
    if (n == 0) return;

    if (n == 1) {
        result[0] = cexp(coeff * alpha[0]);
        return;
    }

    // Copy to working arrays
    double *d = (double *)malloc(n * sizeof(double));
    double *e = (double *)malloc((n + 1) * sizeof(double));
    double *z = (double *)calloc(n * n, sizeof(double));

    if (!d || !e || !z) {
        free(d);
        free(e);
        free(z);
        result[0] = 1.0;
        return;
    }
    e[n] = 0.0;

    for (uint32_t i = 0; i < n; i++) {
        d[i] = alpha[i];
        e[i] = (i > 0) ? beta[i] : 0.0;
        z[i * n + i] = 1.0;
    }

    // QL algorithm
    const int MAX_ITER = 30;
    const double EPS = 1e-15;

    for (uint32_t l = 0; l < n; l++) {
        int iter = 0;
        uint32_t m;

        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m + 1]) <= EPS * dd) break;
            }

            if (m != l) {
                if (iter++ >= MAX_ITER) break;

                double g = (d[l + 1] - d[l]) / (2.0 * e[l + 1]);
                double r = sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l + 1] / (g + (g >= 0 ? r : -r));

                double s = 1.0, c = 1.0, p = 0.0;

                for (int i = m - 1; i >= (int)l; i--) {
                    double f = s * e[i + 1];
                    double b = c * e[i + 1];

                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        e[i + 2] = f * r;
                        s = 1.0 / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        e[i + 2] = g * r;
                        c = 1.0 / r;
                        s *= c;
                    }

                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;

                    for (uint32_t k = 0; k < n; k++) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }

                d[l] -= p;
                e[l + 1] = g;
                e[m + 1] = 0.0;
            }
        } while (m != l);
    }

    // Compute exp(coeff * D) @ Z^T @ e1
    // result[i] = sum_j Z[i,j] * exp(coeff * d[j]) * Z[0,j]
    for (uint32_t i = 0; i < n; i++) {
        result[i] = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            double complex exp_val = cexp(coeff * d[j]);
            result[i] += z[i * n + j] * exp_val * z[0 * n + j];
        }
    }

    free(d);
    free(e);
    free(z);
}

/* Generic Krylov (Lanczos) matrix-exponential-times-vector.
 *
 * Computes y = exp(alpha * H) x on length-`vec_size` complex vectors, where
 * `apply(ctx, in, out)` evaluates out = H @ in.  Shared by the two-site TDVP
 * propagator and the single-site backward propagator so both inherit the same
 * OOM/failure handling: `num_iter` only advances once a tridiagonal row is
 * fully populated, so an allocation or apply failure drops the partial row
 * rather than reading a NULL vector or a stale alpha entry. */
typedef int (*krylov_apply_fn)(void *ctx, const double complex *in,
                               double complex *out);

static int krylov_expm(krylov_apply_fn apply, void *ctx,
                       const double complex *x_data, uint64_t vec_size,
                       double complex alpha, uint32_t max_iter, double tol,
                       double complex *y_data) {
    if (!apply || !x_data || !y_data || vec_size == 0 || max_iter == 0) return -1;

    double complex *v_prev = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_curr = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_next = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *w = (double complex *)calloc(vec_size, sizeof(double complex));
    double *lanczos_alpha = (double *)calloc(max_iter, sizeof(double));
    double *lanczos_beta = (double *)calloc(max_iter + 1, sizeof(double));
    double complex **V = (double complex **)calloc(max_iter, sizeof(double complex *));

    if (!v_prev || !v_curr || !v_next || !w || !lanczos_alpha || !lanczos_beta || !V) {
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta); free(V);
        return -1;
    }

    memcpy(v_curr, x_data, vec_size * sizeof(double complex));
    double norm_x = vector_norm(v_curr, vec_size);
    if (norm_x < 1e-15) {
        memcpy(y_data, x_data, vec_size * sizeof(double complex));
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta); free(V);
        return 0;
    }
    vector_scale(v_curr, 1.0 / norm_x, vec_size);

    lanczos_beta[0] = 0.0;
    uint32_t num_iter = 0;   // count of fully-completed Lanczos iterations

    for (uint32_t iter = 0; iter < max_iter; iter++) {
        V[iter] = (double complex *)malloc(vec_size * sizeof(double complex));
        if (!V[iter]) break;
        memcpy(V[iter], v_curr, vec_size * sizeof(double complex));

        if (apply(ctx, v_curr, w) != 0) {
            free(V[iter]);
            V[iter] = NULL;
            break;
        }

        double complex alpha_c = complex_dot(v_curr, w, vec_size);
        lanczos_alpha[iter] = creal(alpha_c);

        vector_axpy(w, -lanczos_alpha[iter], v_curr, vec_size);
        if (iter > 0) {
            vector_axpy(w, -lanczos_beta[iter], v_prev, vec_size);
        }

        for (uint32_t j = 0; j <= iter; j++) {
            double complex overlap = complex_dot(V[j], w, vec_size);
            vector_axpy(w, -overlap, V[j], vec_size);
        }

        lanczos_beta[iter + 1] = vector_norm(w, vec_size);
        num_iter = iter + 1;

        if (lanczos_beta[iter + 1] < tol) break;

        memcpy(v_next, w, vec_size * sizeof(double complex));
        vector_scale(v_next, 1.0 / lanczos_beta[iter + 1], vec_size);
        memcpy(v_prev, v_curr, vec_size * sizeof(double complex));
        memcpy(v_curr, v_next, vec_size * sizeof(double complex));
    }

    if (num_iter == 0) {
        free(V);
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta);
        return -1;
    }

    double complex *expm_coeffs = (double complex *)calloc(num_iter, sizeof(double complex));
    if (!expm_coeffs) {
        for (uint32_t i = 0; i < num_iter; i++) free(V[i]);
        free(V);
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta);
        return -1;
    }
    tridiag_expm_e1(lanczos_alpha, lanczos_beta, num_iter, alpha, expm_coeffs);

    memset(y_data, 0, vec_size * sizeof(double complex));
    for (uint32_t i = 0; i < num_iter; i++) {
        for (uint64_t j = 0; j < vec_size; j++) {
            y_data[j] += norm_x * expm_coeffs[i] * V[i][j];
        }
    }

    for (uint32_t i = 0; i < num_iter; i++) free(V[i]);
    free(V);
    free(expm_coeffs);
    free(v_prev); free(v_curr); free(v_next); free(w);
    free(lanczos_alpha); free(lanczos_beta);
    return 0;
}

/* Two-site apply: wraps effective_hamiltonian_apply, reusing scratch tensors. */
struct krylov_h2_ctx {
    const effective_hamiltonian_t *H;
    tensor_t *xt;
    tensor_t *yt;
};

static int krylov_h2_apply(void *vctx, const double complex *in,
                           double complex *out) {
    struct krylov_h2_ctx *c = (struct krylov_h2_ctx *)vctx;
    memcpy(c->xt->data, in, (size_t)c->xt->total_size * sizeof(double complex));
    if (effective_hamiltonian_apply(c->H, c->xt, c->yt) != 0) return -1;
    memcpy(out, c->yt->data, (size_t)c->yt->total_size * sizeof(double complex));
    return 0;
}

int lanczos_expm(const effective_hamiltonian_t *H_eff,
                  const tensor_t *x,
                  double complex alpha,
                  uint32_t max_iter,
                  double tol,
                  tensor_t *y) {
    if (!H_eff || !x || !y) return -1;

    uint32_t chi_l = H_eff->chi_l;
    uint32_t chi_r = H_eff->chi_r;
    uint32_t d = H_eff->phys_dim;
    uint64_t vec_size = (uint64_t)chi_l * d * d * chi_r;

    uint32_t dims[4] = { chi_l, d, d, chi_r };
    tensor_t *xt = tensor_create(4, dims);
    tensor_t *yt = tensor_create(4, dims);
    if (!xt || !yt) {
        if (xt) tensor_free(xt);
        if (yt) tensor_free(yt);
        return -1;
    }

    struct krylov_h2_ctx ctx = { H_eff, xt, yt };
    int rc = krylov_expm(krylov_h2_apply, &ctx, x->data, vec_size,
                         alpha, max_iter, tol, y->data);
    tensor_free(xt);
    tensor_free(yt);
    return rc;
}

/* Single-site effective Hamiltonian H^{1s} acting on a rank-3 center tensor
 * C[chi_l, d, chi_r], used for the TDVP backward one-site sub-step:
 *
 *   y[l,s,r] = sum_{lp,sp,rp,bl,br}
 *              L[l,bl,lp] * W[bl,s,sp,br] * R[r,br,rp] * C[lp,sp,rp]
 *
 * with the same L/R/W index conventions as effective_hamiltonian_apply
 * (L: [chi_l,b_l,chi_l], R: [chi_r,b_r,chi_r], W->W: [b_l,d,d,b_r]). */
struct krylov_h1_ctx {
    const tensor_t *L;
    const tensor_t *R;
    const mpo_tensor_t *W;
    uint32_t chi_l, chi_r, d;
};

static int krylov_h1_apply(void *vctx, const double complex *in,
                           double complex *out) {
    struct krylov_h1_ctx *c = (struct krylov_h1_ctx *)vctx;
    uint32_t chi_l = c->chi_l, chi_r = c->chi_r, d = c->d;
    uint32_t b_l = c->W->bond_dim_left;
    uint32_t b_r = c->W->bond_dim_right;
    const double complex *L = c->L->data;
    const double complex *R = c->R->data;
    const double complex *W = c->W->W->data;

    size_t t1_size = (size_t)chi_l * d * b_r * chi_r;   // t1[lp,sp,br,r]
    size_t t2_size = (size_t)chi_l * b_l * d * chi_r;   // t2[lp,bl,s,r]
    double complex *t1 = (double complex *)calloc(t1_size, sizeof(double complex));
    double complex *t2 = (double complex *)calloc(t2_size, sizeof(double complex));
    if (!t1 || !t2) { free(t1); free(t2); return -1; }

    // Step 1: t1[lp,sp,br,r] = sum_rp in[lp,sp,rp] * R[r,br,rp]
    for (uint32_t lp = 0; lp < chi_l; lp++) {
        for (uint32_t sp = 0; sp < d; sp++) {
            for (uint32_t br = 0; br < b_r; br++) {
                for (uint32_t r = 0; r < chi_r; r++) {
                    double complex sum = 0.0;
                    for (uint32_t rp = 0; rp < chi_r; rp++) {
                        uint64_t in_idx = (uint64_t)lp * d * chi_r + sp * chi_r + rp;
                        uint64_t R_idx = (uint64_t)r * b_r * chi_r + br * chi_r + rp;
                        sum += in[in_idx] * R[R_idx];
                    }
                    uint64_t t1_idx = (uint64_t)lp * d * b_r * chi_r + sp * b_r * chi_r + br * chi_r + r;
                    t1[t1_idx] = sum;
                }
            }
        }
    }

    // Step 2: t2[lp,bl,s,r] = sum_{sp,br} t1[lp,sp,br,r] * W[bl,s,sp,br]
    for (uint32_t lp = 0; lp < chi_l; lp++) {
        for (uint32_t bl = 0; bl < b_l; bl++) {
            for (uint32_t s = 0; s < d; s++) {
                for (uint32_t r = 0; r < chi_r; r++) {
                    double complex sum = 0.0;
                    for (uint32_t sp = 0; sp < d; sp++) {
                        for (uint32_t br = 0; br < b_r; br++) {
                            uint64_t t1_idx = (uint64_t)lp * d * b_r * chi_r + sp * b_r * chi_r + br * chi_r + r;
                            uint64_t W_idx = (uint64_t)bl * d * d * b_r + s * d * b_r + sp * b_r + br;
                            sum += t1[t1_idx] * W[W_idx];
                        }
                    }
                    uint64_t t2_idx = (uint64_t)lp * b_l * d * chi_r + bl * d * chi_r + s * chi_r + r;
                    t2[t2_idx] = sum;
                }
            }
        }
    }

    // Step 3: out[l,s,r] = sum_{lp,bl} t2[lp,bl,s,r] * L[l,bl,lp]
    for (uint32_t l = 0; l < chi_l; l++) {
        for (uint32_t s = 0; s < d; s++) {
            for (uint32_t r = 0; r < chi_r; r++) {
                double complex sum = 0.0;
                for (uint32_t lp = 0; lp < chi_l; lp++) {
                    for (uint32_t bl = 0; bl < b_l; bl++) {
                        uint64_t t2_idx = (uint64_t)lp * b_l * d * chi_r + bl * d * chi_r + s * chi_r + r;
                        uint64_t L_idx = (uint64_t)l * b_l * chi_l + bl * chi_l + lp;
                        sum += t2[t2_idx] * L[L_idx];
                    }
                }
                uint64_t y_idx = (uint64_t)l * d * chi_r + s * chi_r + r;
                out[y_idx] = sum;
            }
        }
    }

    free(t1);
    free(t2);
    return 0;
}

/* Evolve a single-site center tensor in place: C <- exp(alpha * H^{1s}) C. */
static int lanczos_expm_single_site(const tensor_t *L, const tensor_t *R,
                                    const mpo_tensor_t *W,
                                    uint32_t chi_l, uint32_t d, uint32_t chi_r,
                                    double complex *inout,
                                    double complex alpha,
                                    uint32_t max_iter, double tol) {
    if (!L || !R || !W || !inout) return -1;
    struct krylov_h1_ctx ctx = { L, R, W, chi_l, chi_r, d };
    uint64_t vec_size = (uint64_t)chi_l * d * chi_r;
    double complex *y = (double complex *)calloc(vec_size, sizeof(double complex));
    if (!y) return -1;
    int rc = krylov_expm(krylov_h1_apply, &ctx, inout, vec_size,
                         alpha, max_iter, tol, y);
    if (rc == 0) memcpy(inout, y, vec_size * sizeof(double complex));
    free(y);
    return rc;
}

// ============================================================================
// TDVP ENGINE
// ============================================================================

tdvp_engine_t *tdvp_engine_create(tn_mps_state_t *mps,
                                   mpo_t *mpo,
                                   const tdvp_config_t *config) {
    if (!mps || !mpo || !config) return NULL;

    tdvp_engine_t *engine = (tdvp_engine_t *)calloc(1, sizeof(tdvp_engine_t));
    if (!engine) return NULL;

    engine->mps = mps;
    engine->mpo = mpo;
    engine->config = *config;
    engine->current_time = 0.0;
    engine->bond_states = NULL;
    engine->num_bond_states = 0;

    // Create environments
    engine->env = dmrg_environments_create(mps->num_qubits);
    if (!engine->env) {
        free(engine);
        return NULL;
    }

    // Initialize environments - must initialize BOTH left and right
    // Left environments are needed for L->R sweep, right for R->L sweep
    if (dmrg_init_left_environments(engine->env, mps, mpo) != 0) {
        dmrg_environments_free(engine->env);
        free(engine);
        return NULL;
    }
    if (dmrg_init_right_environments(engine->env, mps, mpo) != 0) {
        dmrg_environments_free(engine->env);
        free(engine);
        return NULL;
    }

    /* Allocate the per-bond PID state array when the adaptive-bond
     * controller is enabled.  An n-qubit chain has (n - 1) inter-site
     * bonds; each entry is zero-initialised (chi=0 primes on first
     * visit, integral starts at 0, primed=false). */
    if (config->adaptive_bond.enabled && mps->num_qubits >= 2) {
        engine->num_bond_states = mps->num_qubits - 1;
        engine->bond_states =
            (struct tdvp_bond_pid_state *)calloc(
                engine->num_bond_states,
                sizeof(struct tdvp_bond_pid_state));
        if (!engine->bond_states) {
            dmrg_environments_free(engine->env);
            free(engine);
            return NULL;
        }
    }

    return engine;
}

void tdvp_engine_free(tdvp_engine_t *engine) {
    if (!engine) return;
    free(engine->bond_states);
    dmrg_environments_free(engine->env);
    free(engine);
}

uint32_t tdvp_bond_chi(const tdvp_engine_t *engine, uint32_t bond) {
    if (!engine || !engine->bond_states) return 0;
    if (bond >= engine->num_bond_states) return 0;
    return engine->bond_states[bond].chi;
}

void tdvp_set_dt(tdvp_engine_t *engine, double dt) {
    if (engine) engine->config.dt = dt;
}

double tdvp_get_time(const tdvp_engine_t *engine) {
    return engine ? engine->current_time : 0.0;
}

// ============================================================================
// BOND TRUNCATION (entropy-feedback PID for v0.4 adaptive-bond TDVP)
// ============================================================================

/**
 * @brief Compute the von Neumann entropy of a singular-value
 *        spectrum and the entropy of its first `chi` components.
 *
 * Both quantities are normalised so that the underlying probability
 * distributions sum to 1.  `sv` is assumed sorted in descending
 * order, as produced by `svd_compress`.
 */
static void tdvp_spectrum_entropy(const double *sv, uint32_t n,
                                   uint32_t chi,
                                   double *S_full, double *S_chi) {
    double total = 0.0;
    for (uint32_t i = 0; i < n; i++) total += sv[i] * sv[i];
    double Sf = 0.0;
    if (total > 0.0) {
        for (uint32_t i = 0; i < n; i++) {
            double p = (sv[i] * sv[i]) / total;
            if (p > 1e-30) Sf -= p * log(p);
        }
    }
    *S_full = Sf;

    uint32_t k = chi < n ? chi : n;
    double partial = 0.0;
    for (uint32_t i = 0; i < k; i++) partial += sv[i] * sv[i];
    double Sc = 0.0;
    if (partial > 0.0) {
        for (uint32_t i = 0; i < k; i++) {
            double p = (sv[i] * sv[i]) / partial;
            if (p > 1e-30) Sc -= p * log(p);
        }
    }
    *S_chi = Sc;
}

/**
 * @brief PID update for one bond: returns the new target bond
 *        dimension given the current PID state and the entropy
 *        error signal `e = S_full - S_chi`.
 */
static uint32_t tdvp_pid_select_chi(
    const tdvp_adaptive_bond_config_t *cfg,
    struct tdvp_bond_pid_state *state,
    double e, uint32_t chi_current)
{
    double derivative = 0.0;
    if (state->primed) derivative = e - state->prev_error;
    state->integral  += e;
    state->prev_error = e;
    state->primed     = true;

    double signal = cfg->kp * e
                  + cfg->ki * state->integral
                  + cfg->kd * derivative;

    /* Map the entropy-domain signal into bond-dim increments via
     * `alpha / eps_S`; clamp to [chi_floor, chi_ceiling]. */
    double scale = (cfg->target_entropy_error > 0.0)
                 ? cfg->alpha / cfg->target_entropy_error
                 : cfg->alpha;
    int64_t delta = (int64_t)llround(scale * signal);

    int64_t target = (int64_t)chi_current + delta;
    if (target < (int64_t)cfg->chi_floor)  target = cfg->chi_floor;
    if (target > (int64_t)cfg->chi_ceiling) target = cfg->chi_ceiling;
    return (uint32_t)target;
}

/**
 * @brief Truncate an evolved two-site tensor matrix.
 *
 * Two paths:
 *
 *   * Legacy (`cfg->adaptive_bond.enabled == false` or
 *     `bond_state == NULL`): single SVD with `max_bond_dim` /
 *     `svd_cutoff` from the TDVP config; behaviour is bit-identical
 *     to v0.3.1.
 *
 *   * Adaptive: first SVD at `chi_ceiling` to expose the spectrum,
 *     entropy-feedback PID picks `target_chi`, and (if necessary) a
 *     second SVD re-truncates to that bond dimension.
 *
 * Returns the `svd_compress_result_t` (caller frees with
 * `svd_compress_result_free`), or `NULL` on allocation failure.
 */
static svd_compress_result_t *tdvp_truncate_bond(
    const tensor_t *mat,
    const tdvp_config_t *cfg,
    struct tdvp_bond_pid_state *bond_state)
{
    /* ---- Legacy path ------------------------------------------------ */
    if (!cfg->adaptive_bond.enabled || bond_state == NULL) {
        svd_compress_config_t svd_cfg = svd_compress_config_default();
        svd_cfg.max_bond_dim = cfg->max_bond_dim;
        svd_cfg.cutoff       = cfg->svd_cutoff;
        return svd_compress(mat, &svd_cfg);
    }

    /* ---- Adaptive path: PID controller ----------------------------- */
    const tdvp_adaptive_bond_config_t *ab = &cfg->adaptive_bond;

    /* Pass 1: SVD at chi_ceiling to obtain the spectrum the PID will
     * reason over.  We honour `svd_cutoff` as a floor so the spectrum
     * still discards numerical noise. */
    svd_compress_config_t svd_cfg = svd_compress_config_default();
    svd_cfg.max_bond_dim = ab->chi_ceiling;
    svd_cfg.cutoff       = cfg->svd_cutoff;
    svd_compress_result_t *first = svd_compress(mat, &svd_cfg);
    if (!first) return NULL;

    /* Prime the bond's chi on the first visit. */
    if (bond_state->chi == 0) bond_state->chi = first->bond_dim;

    /* Compute the entropy error e = S_full - S_chi at the current
     * working chi. */
    double S_full = 0.0, S_chi = 0.0;
    tdvp_spectrum_entropy(first->singular_values, first->bond_dim,
                          bond_state->chi, &S_full, &S_chi);
    double e = S_full - S_chi;

    uint32_t target_chi = tdvp_pid_select_chi(ab, bond_state, e,
                                               bond_state->chi);
    bond_state->chi = target_chi;

    /* If the PID-selected chi covers (or exceeds) what we got from
     * pass 1, return that result directly. */
    if (target_chi >= first->bond_dim) {
        return first;
    }

    /* Otherwise, redo the SVD at the controller-selected chi. */
    svd_compress_result_free(first);
    svd_cfg.max_bond_dim = target_chi;
    return svd_compress(mat, &svd_cfg);
}

// ============================================================================
// TWO-SITE TDVP STEP
// ============================================================================

/* Sweep directions for the projector-splitting integrator. */
#define TDVP_DIR_LR (+1)   /* left-to-right half-sweep */
#define TDVP_DIR_RL (-1)   /* right-to-left half-sweep */

/**
 * @brief One bond of the two-site projector-splitting TDVP integrator.
 *
 * Performs the genuine 2TDVP sub-steps at bond (site, site+1):
 *   1. forward-evolve the two-site tensor theta by exp(mp * H^{2s} * dt_half);
 *   2. SVD-split theta, truncating (adaptive-PID aware), absorbing the singular
 *      values TOWARD the sweep direction so the orthogonality center follows
 *      the sweep (S into the right tensor for L->R, into the left for R->L);
 *   3. update the environment on the trailing side;
 *   4. unless this is the sweep's turning bond, backward-evolve the resulting
 *      single-site center by exp(-mp * H^{1s} * dt_half) with the freshly
 *      updated environment -- the projector-splitting correction that the old
 *      "two forward half-sweeps, never back-evolve" code omitted entirely.
 *
 * @param dt_half     Half the step (the caller passes dt/2).
 * @param direction   TDVP_DIR_LR or TDVP_DIR_RL.
 * @param back_evolve false at the sweep's turning bond (no one-site back-step).
 * @param bond_state  Per-bond adaptive-PID state, or NULL for the fixed cap.
 */
static int tdvp_two_site_ps(tn_mps_state_t *mps,
                            const mpo_t *mpo,
                            dmrg_environments_t *env,
                            uint32_t site,
                            double complex dt_half,
                            const tdvp_config_t *config,
                            struct tdvp_bond_pid_state *bond_state,
                            int direction,
                            bool back_evolve,
                            double *truncation_error) {
    if (!mps || !mpo || !env || site >= mps->num_qubits - 1) return -1;

    tensor_t *A = mps->tensors[site];
    tensor_t *B = mps->tensors[site + 1];

    uint32_t chi_l = A->dims[0];
    uint32_t d = A->dims[1];
    uint32_t chi_m = A->dims[2];
    uint32_t chi_r = B->dims[2];

    if (!env->L[site] || !env->R[site + 1]) return -1;

    effective_hamiltonian_t H_eff = {
        .L = env->L[site],
        .R = env->R[site + 1],
        .W_left = &mpo->tensors[site],
        .W_right = &mpo->tensors[site + 1],
        .chi_l = chi_l,
        .chi_r = chi_r,
        .phys_dim = d,
    };

    // theta = A @ B via a single zgemm into the row-major [chi_l,d,d,chi_r]
    // layout (scalar fallback when BLAS is unavailable).
    uint32_t theta_dims[4] = {chi_l, d, d, chi_r};
    tensor_t *theta = tensor_create(4, theta_dims);
    if (!theta) return -1;

#if defined(MOONLAB_TDVP_HAVE_BLAS)
    {
        const double complex one = 1.0, zero = 0.0;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)(chi_l * d), (int)(d * chi_r), (int)chi_m,
                    &one,
                    A->data, (int)chi_m,
                    B->data, (int)(d * chi_r),
                    &zero,
                    theta->data, (int)(d * chi_r));
    }
#else
    for (uint32_t l = 0; l < chi_l; l++) {
        for (uint32_t s1 = 0; s1 < d; s1++) {
            for (uint32_t s2 = 0; s2 < d; s2++) {
                for (uint32_t r = 0; r < chi_r; r++) {
                    double complex sum = 0.0;
                    for (uint32_t m = 0; m < chi_m; m++) {
                        uint64_t A_idx = l * d * chi_m + s1 * chi_m + m;
                        uint64_t B_idx = m * d * chi_r + s2 * chi_r + r;
                        sum += A->data[A_idx] * B->data[B_idx];
                    }
                    uint64_t theta_idx = l * d * d * chi_r + s1 * d * chi_r + s2 * chi_r + r;
                    theta->data[theta_idx] = sum;
                }
            }
        }
    }
#endif

    // Forward two-site sub-step: theta <- exp(fwd * H^{2s} * dt_half) theta.
    // Real time: fwd = -i; imaginary time: fwd = -1.
    tensor_t *theta_evolved = tensor_create(4, theta_dims);
    if (!theta_evolved) {
        tensor_free(theta);
        return -1;
    }

    double complex fwd = (config->evolution_type == TDVP_IMAGINARY_TIME)
                             ? (double complex)(-1.0)
                             : (double complex)(-I);
    double complex fwd_exponent = fwd * dt_half;

    if (lanczos_expm(&H_eff, theta, fwd_exponent,
                      config->lanczos_max_iter, config->lanczos_tol,
                      theta_evolved) != 0) {
        tensor_free(theta);
        tensor_free(theta_evolved);
        return -1;
    }
    tensor_free(theta);

    /* Imag-time conditioning: exp(-H dt/2) shrinks the two-site tensor toward
     * zero; renormalise to unit Frobenius norm so the downstream SVD stays
     * well-conditioned (a global rescaling, irrelevant to the ground state the
     * imag-time flow converges to, and absorbed by the end-of-step
     * renormalisation).  Real-time evolution is norm preserving, so skip it. */
    if (config->evolution_type == TDVP_IMAGINARY_TIME) {
        double tnorm = tensor_norm_frobenius(theta_evolved);
        if (tnorm > 0.0 && isfinite(tnorm) && tnorm != 1.0) {
            double inv = 1.0 / tnorm;
            for (uint64_t i = 0; i < theta_evolved->total_size; i++) {
                theta_evolved->data[i] *= inv;
            }
        }
    }

    // SVD-split with truncation (adaptive-PID aware).
    uint32_t mat_dims[2] = {chi_l * d, d * chi_r};
    tensor_t *mat = tensor_reshape(theta_evolved, 2, mat_dims);
    tensor_free(theta_evolved);
    if (!mat) return -1;

    svd_compress_result_t *svd = tdvp_truncate_bond(mat, config, bond_state);
    tensor_free(mat);
    if (!svd) return -1;

    *truncation_error = svd->truncation_error;
    uint32_t new_bond = svd->bond_dim;

    tensor_t *A_new = NULL, *B_new = NULL;
    if (direction == TDVP_DIR_LR) {
        // Center moves right: A = U (left-isometric), absorb S into V^dag.
        for (uint32_t i = 0; i < new_bond; i++) {
            for (uint32_t j = 0; j < svd->right->dims[1]; j++) {
                svd->right->data[i * svd->right->dims[1] + j] *= svd->singular_values[i];
            }
        }
    } else {
        // Center moves left: B = V^dag (right-isometric), absorb S into U.
        for (uint32_t i = 0; i < svd->left->dims[0]; i++) {
            for (uint32_t j = 0; j < new_bond; j++) {
                svd->left->data[i * new_bond + j] *= svd->singular_values[j];
            }
        }
    }

    uint32_t A_dims[3] = {chi_l, d, new_bond};
    A_new = tensor_reshape(svd->left, 3, A_dims);
    uint32_t B_dims[3] = {new_bond, d, chi_r};
    B_new = tensor_reshape(svd->right, 3, B_dims);

    if (!A_new || !B_new) {
        if (A_new) tensor_free(A_new);
        if (B_new) tensor_free(B_new);
        svd_compress_result_free(svd);
        return -1;
    }

    tensor_free(mps->tensors[site]);
    tensor_free(mps->tensors[site + 1]);
    mps->tensors[site] = A_new;
    mps->tensors[site + 1] = B_new;
    mps->bond_dims[site] = new_bond;
    svd_compress_result_free(svd);

    // Update the trailing-side environment from the freshly isometric tensor,
    // then (unless this is the turning bond) apply the backward one-site
    // sub-step to the center with the updated environment.
    double complex bwd_exponent = -fwd_exponent;   // exp(+... ) inverse sub-step

    if (direction == TDVP_DIR_LR) {
        // A[site] is now left-isometric -> refresh L[site+1].
        if (dmrg_update_left_environment(env, mps, mpo, site) != 0) return -1;
        if (back_evolve) {
            // Center C = B[site+1], shape [new_bond, d, chi_r].
            if (!env->L[site + 1] || !env->R[site + 1]) return -1;
            if (lanczos_expm_single_site(env->L[site + 1], env->R[site + 1],
                                         &mpo->tensors[site + 1],
                                         new_bond, d, chi_r,
                                         mps->tensors[site + 1]->data,
                                         bwd_exponent,
                                         config->lanczos_max_iter,
                                         config->lanczos_tol) != 0) {
                return -1;
            }
        }
    } else {
        // B[site+1] is now right-isometric -> refresh R[site].
        if (dmrg_update_right_environment(env, mps, mpo, site + 1) != 0) return -1;
        if (back_evolve) {
            // Center C = A[site], shape [chi_l, d, new_bond].
            if (!env->L[site] || !env->R[site]) return -1;
            if (lanczos_expm_single_site(env->L[site], env->R[site],
                                         &mpo->tensors[site],
                                         chi_l, d, new_bond,
                                         mps->tensors[site]->data,
                                         bwd_exponent,
                                         config->lanczos_max_iter,
                                         config->lanczos_tol) != 0) {
                return -1;
            }
        }
    }

    return 0;
}

// ============================================================================
// TDVP STEP
// ============================================================================

int tdvp_step(tdvp_engine_t *engine, tdvp_result_t *result) {
    if (!engine) return -1;

    double start_time = get_time_sec();
    double total_trunc_error = 0.0;
    uint32_t max_bond = 0;
    double dt = engine->config.dt;

    tn_mps_state_t *mps = engine->mps;
    mpo_t *mpo = engine->mpo;
    dmrg_environments_t *env = engine->env;
    uint32_t n = mps->num_qubits;

    // Genuine two-site projector-splitting TDVP for a symmetric step of size
    // dt: a left-to-right half-sweep followed by a right-to-left half-sweep,
    // each forward-evolving every two-site block by dt/2 and backward-evolving
    // the single-site center by dt/2 (except at the turning bond).  Establish
    // the required gauge first: right-canonicalise so the orthogonality centre
    // sits at site 0 and every site to its right is right-isometric, which is
    // exactly the environment the L->R sweep consumes.
    if (tn_mps_right_canonicalize(mps) != TN_STATE_SUCCESS) {
        fprintf(stderr, "TDVP: Failed to right-canonicalize MPS\n");
        return -1;
    }
    if (dmrg_init_left_environments(env, mps, mpo) != 0) {
        fprintf(stderr, "TDVP: Failed to initialize left environments\n");
        return -1;
    }
    if (dmrg_init_right_environments(env, mps, mpo) != 0) {
        fprintf(stderr, "TDVP: Failed to initialize right environments\n");
        return -1;
    }

    // Left-to-right half-sweep.  The environment on the trailing (left) side is
    // refreshed inside tdvp_two_site_ps as the center advances.
    for (uint32_t site = 0; site < n - 1; site++) {
        double trunc_err = 0.0;
        struct tdvp_bond_pid_state *bs =
            (engine->bond_states && site < engine->num_bond_states)
                ? &engine->bond_states[site]
                : NULL;
        bool turning = (site == n - 2);   // last bond: no backward sub-step
        if (tdvp_two_site_ps(mps, mpo, env, site, dt / 2,
                             &engine->config, bs, TDVP_DIR_LR,
                             /*back_evolve=*/!turning, &trunc_err) != 0) {
            fprintf(stderr, "TDVP: Failed at site %u (L->R)\n", site);
            return -1;
        }
        total_trunc_error += trunc_err;
        if (mps->bond_dims[site] > max_bond) max_bond = mps->bond_dims[site];
    }

    // The state is now left-canonical with the center at site n-1.  Rebuild the
    // right environments for the return sweep (the left environments carried
    // forward from the L->R sweep remain valid for the sites not yet revisited).
    if (dmrg_init_right_environments(env, mps, mpo) != 0) {
        fprintf(stderr, "TDVP: Failed to rebuild right environments\n");
        return -1;
    }

    // Right-to-left half-sweep.
    for (int site = n - 2; site >= 0; site--) {
        double trunc_err = 0.0;
        struct tdvp_bond_pid_state *bs =
            (engine->bond_states &&
             (uint32_t)site < engine->num_bond_states)
                ? &engine->bond_states[site]
                : NULL;
        bool turning = (site == 0);   // last bond of the return sweep
        if (tdvp_two_site_ps(mps, mpo, env, (uint32_t)site, dt / 2,
                             &engine->config, bs, TDVP_DIR_RL,
                             /*back_evolve=*/!turning, &trunc_err) != 0) {
            fprintf(stderr, "TDVP: Failed at site %d (R->L)\n", site);
            return -1;
        }
        total_trunc_error += trunc_err;
        if (mps->bond_dims[site] > max_bond) max_bond = mps->bond_dims[site];
    }

    // Normalize if requested
    double norm = 1.0;
    if (engine->config.normalize) {
        norm = tn_mps_norm(mps);
        if (norm > 0) {
            // Normalize by scaling first tensor
            double scale = 1.0 / norm;
            for (uint64_t i = 0; i < mps->tensors[0]->total_size; i++) {
                mps->tensors[0]->data[i] *= scale;
            }
        }
    }

    engine->current_time += dt;

    // Fill result
    if (result) {
        result->time = engine->current_time;
        result->energy = dmrg_compute_energy(mps, mpo);
        result->norm = norm;
        result->truncation_error = total_trunc_error;
        result->max_bond_dim = max_bond;
        result->step_time = get_time_sec() - start_time;

        /* Per-bond chi snapshot from the adaptive-bond controller.
         * Lazy-allocate the buffer on first call; reuse it if the
         * size matches (engine bond count is fixed for the engine's
         * lifetime, so realloc is only needed if the caller swaps
         * the result between differently sized engines). */
        if (engine->bond_states && engine->num_bond_states > 0) {
            if (result->bond_chi_distribution == NULL ||
                result->n_bonds != engine->num_bond_states) {
                free(result->bond_chi_distribution);
                result->bond_chi_distribution = (uint32_t *)calloc(
                    engine->num_bond_states, sizeof(uint32_t));
                result->n_bonds = engine->num_bond_states;
            }
            if (result->bond_chi_distribution) {
                for (uint32_t b = 0; b < engine->num_bond_states; b++) {
                    result->bond_chi_distribution[b] =
                        engine->bond_states[b].chi;
                }
            }
        } else {
            /* Legacy path: keep the result clean.  If a buffer is
             * left over from an earlier adaptive run, free it. */
            if (result->bond_chi_distribution) {
                free(result->bond_chi_distribution);
                result->bond_chi_distribution = NULL;
                result->n_bonds = 0;
            }
        }
    }

    return 0;
}

int tdvp_evolve_to(tdvp_engine_t *engine,
                    double target_time,
                    tdvp_history_t *history) {
    if (!engine) return -1;

    /* Zero-init: tdvp_step's adaptive-bond branch dereferences and
     * frees result->bond_chi_distribution if it isn't NULL, so we
     * must not pass it stack garbage on the first iteration.  The
     * struct is reused across iterations -- tdvp_step realloc's in
     * place if the bond count is stable, which it is for an engine
     * with a fixed MPS. */
    tdvp_result_t result = {0};
    int rc = 0;

    // Preserve the configured step: the final partial step shrinks dt only for
    // that iteration, and the original value is restored on exit.  A
    // non-positive configured dt can never reach the target.
    const double dt_full = engine->config.dt;
    if (!(dt_full > 0.0)) return -1;
    const double eps = 1e-12 * (fabs(target_time) + 1.0);

    while (engine->current_time < target_time - eps) {
        double remaining = target_time - engine->current_time;
        engine->config.dt = (remaining < dt_full) ? remaining : dt_full;
        if (engine->config.dt <= 0.0) break;   // underflow guard: guarantee progress

        double t_before = engine->current_time;
        if (tdvp_step(engine, &result) != 0) {
            rc = -1;
            break;
        }

        if (history) {
            tdvp_history_add(history, &result);
        }

        if (engine->config.verbose) {
            printf("  t=%.4f: E=%.8f, norm=%.6f, chi_max=%u, trunc=%.2e\n",
                   result.time, result.energy, result.norm,
                   result.max_bond_dim, result.truncation_error);
        }

        // If a step failed to advance time (degenerate dt), stop instead of
        // spinning forever.
        if (engine->current_time <= t_before) { rc = -1; break; }
    }

    engine->config.dt = dt_full;   // restore the configured step
    tdvp_result_clear(&result);
    return rc;
}

// ============================================================================
// SINGLE-STEP STATELESS VERSION
// ============================================================================

int tdvp_single_step(tn_mps_state_t *mps,
                      const mpo_t *mpo,
                      double dt,
                      const tdvp_config_t *config,
                      double *energy) {
    if (!mps || !mpo || !config) return -1;

    // Create temporary engine
    tdvp_config_t cfg = *config;
    cfg.dt = dt;

    tdvp_engine_t *engine = tdvp_engine_create(mps, (mpo_t *)mpo, &cfg);
    if (!engine) return -1;

    /* Zero-init mirrors the contract in tdvp_evolve_to: the adaptive
     * branch in tdvp_step may free this result's bond_chi_distribution
     * if non-NULL. */
    tdvp_result_t result = {0};
    int ret = tdvp_step(engine, &result);

    if (energy) *energy = result.energy;

    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    return ret;
}

// ============================================================================
// EVOLUTION WITH OBSERVABLES
// ============================================================================

int tdvp_evolve_with_observables(tdvp_engine_t *engine,
                                  double target_time,
                                  observable_callback_t callback,
                                  void *user_data,
                                  uint32_t measure_interval) {
    if (!engine) return -1;

    uint32_t step_count = 0;
    tdvp_result_t result = {0};
    int rc = 0;

    while (engine->current_time < target_time) {
        double remaining = target_time - engine->current_time;
        if (remaining < engine->config.dt) {
            engine->config.dt = remaining;
        }

        if (tdvp_step(engine, &result) != 0) {
            rc = -1;
            break;
        }

        step_count++;

        if (callback && measure_interval > 0
            && (step_count % measure_interval == 0)) {
            callback(engine->mps, engine->current_time, user_data);
        }
    }

    // Final measurement
    if (rc == 0 && callback) {
        callback(engine->mps, engine->current_time, user_data);
    }

    tdvp_result_clear(&result);
    return rc;
}

int tdvp_evolve_to_with_observable(tdvp_engine_t *engine,
                                    double target_time,
                                    tdvp_history_t *history,
                                    observable_value_callback_t observable_fn,
                                    void *user_data) {
    if (!engine || !history) return -1;

    tdvp_result_t result = {0};
    int rc = 0;

    const double dt_full = engine->config.dt;
    if (!(dt_full > 0.0)) return -1;
    const double eps = 1e-12 * (fabs(target_time) + 1.0);

    while (engine->current_time < target_time - eps) {
        double remaining = target_time - engine->current_time;
        engine->config.dt = (remaining < dt_full) ? remaining : dt_full;
        if (engine->config.dt <= 0.0) break;   // underflow guard: guarantee progress

        double t_before = engine->current_time;
        if (tdvp_step(engine, &result) != 0) {
            rc = -1;
            break;
        }

        if (observable_fn) {
            double obs = observable_fn(engine->mps,
                                        engine->current_time,
                                        user_data);
            tdvp_history_add_with_observable(history, &result, obs);
        } else {
            tdvp_history_add(history, &result);
        }

        if (engine->current_time <= t_before) { rc = -1; break; }
    }

    engine->config.dt = dt_full;   // restore the configured step
    tdvp_result_clear(&result);
    return rc;
}
