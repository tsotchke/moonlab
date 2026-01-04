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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    free(hist);
}

void tdvp_history_add(tdvp_history_t *hist, const tdvp_result_t *result) {
    if (!hist || !result) return;

    // Expand if needed
    if (hist->num_steps >= hist->capacity) {
        hist->capacity *= 2;
        hist->times = (double *)realloc(hist->times, hist->capacity * sizeof(double));
        hist->energies = (double *)realloc(hist->energies, hist->capacity * sizeof(double));
        hist->norms = (double *)realloc(hist->norms, hist->capacity * sizeof(double));
    }

    hist->times[hist->num_steps] = result->time;
    hist->energies[hist->num_steps] = result->energy;
    hist->norms[hist->num_steps] = result->norm;
    hist->num_steps++;
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
    uint64_t vec_size = H_eff->two_site ? chi_l * d * d * chi_r : chi_l * d * chi_r;

    // Lanczos vectors
    double complex *v_prev = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_curr = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_next = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *w = (double complex *)calloc(vec_size, sizeof(double complex));

    // Tridiagonal matrix
    double *lanczos_alpha = (double *)calloc(max_iter, sizeof(double));
    double *lanczos_beta = (double *)calloc(max_iter + 1, sizeof(double));

    // Store Lanczos vectors
    double complex **V = (double complex **)calloc(max_iter, sizeof(double complex *));

    if (!v_prev || !v_curr || !v_next || !w || !lanczos_alpha || !lanczos_beta || !V) {
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta); free(V);
        return -1;
    }

    // Initialize with input vector
    memcpy(v_curr, x->data, vec_size * sizeof(double complex));
    double norm_x = vector_norm(v_curr, vec_size);
    if (norm_x < 1e-15) {
        memcpy(y->data, x->data, vec_size * sizeof(double complex));
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta); free(V);
        return 0;
    }
    vector_scale(v_curr, 1.0 / norm_x, vec_size);

    lanczos_beta[0] = 0.0;
    uint32_t num_iter = 0;

    // Create tensors for H_eff application
    uint32_t dims[4];
    if (H_eff->two_site) {
        dims[0] = chi_l; dims[1] = d; dims[2] = d; dims[3] = chi_r;
    } else {
        dims[0] = chi_l; dims[1] = d; dims[2] = chi_r; dims[3] = 0;
    }

    tensor_t *x_temp = tensor_create(H_eff->two_site ? 4 : 3, dims);
    tensor_t *y_temp = tensor_create(H_eff->two_site ? 4 : 3, dims);

    if (!x_temp || !y_temp) {
        if (x_temp) tensor_free(x_temp);
        if (y_temp) tensor_free(y_temp);
        free(v_prev); free(v_curr); free(v_next); free(w);
        free(lanczos_alpha); free(lanczos_beta); free(V);
        return -1;
    }

    // Lanczos iteration
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        num_iter = iter + 1;

        // Store current vector
        V[iter] = (double complex *)malloc(vec_size * sizeof(double complex));
        if (!V[iter]) break;
        memcpy(V[iter], v_curr, vec_size * sizeof(double complex));

        // w = H @ v_curr
        memcpy(x_temp->data, v_curr, vec_size * sizeof(double complex));
        if (effective_hamiltonian_apply(H_eff, x_temp, y_temp) != 0) {
            break;
        }
        memcpy(w, y_temp->data, vec_size * sizeof(double complex));

        // alpha[iter] = v_curr^H @ w
        double complex alpha_c = complex_dot(v_curr, w, vec_size);
        lanczos_alpha[iter] = creal(alpha_c);

        // w = w - alpha * v_curr - beta * v_prev
        vector_axpy(w, -lanczos_alpha[iter], v_curr, vec_size);
        if (iter > 0) {
            vector_axpy(w, -lanczos_beta[iter], v_prev, vec_size);
        }

        // Reorthogonalization
        for (uint32_t j = 0; j <= iter; j++) {
            double complex overlap = complex_dot(V[j], w, vec_size);
            vector_axpy(w, -overlap, V[j], vec_size);
        }

        // beta[iter+1] = ||w||
        lanczos_beta[iter + 1] = vector_norm(w, vec_size);

        // Check convergence
        if (lanczos_beta[iter + 1] < tol) {
            num_iter = iter + 1;
            break;
        }

        // v_next = w / beta
        memcpy(v_next, w, vec_size * sizeof(double complex));
        vector_scale(v_next, 1.0 / lanczos_beta[iter + 1], vec_size);

        // Shift
        memcpy(v_prev, v_curr, vec_size * sizeof(double complex));
        memcpy(v_curr, v_next, vec_size * sizeof(double complex));
    }

    tensor_free(x_temp);
    tensor_free(y_temp);

    // Compute exp(alpha * T) @ e1
    double complex *expm_coeffs = (double complex *)calloc(num_iter, sizeof(double complex));
    tridiag_expm_e1(lanczos_alpha, lanczos_beta, num_iter, alpha, expm_coeffs);

    // Reconstruct: y = norm_x * sum_i expm_coeffs[i] * V[i]
    memset(y->data, 0, vec_size * sizeof(double complex));
    for (uint32_t i = 0; i < num_iter; i++) {
        for (uint64_t j = 0; j < vec_size; j++) {
            y->data[j] += norm_x * expm_coeffs[i] * V[i][j];
        }
    }

    // Cleanup
    for (uint32_t i = 0; i < num_iter; i++) {
        free(V[i]);
    }
    free(V);
    free(expm_coeffs);
    free(v_prev); free(v_curr); free(v_next); free(w);
    free(lanczos_alpha); free(lanczos_beta);

    return 0;
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

    return engine;
}

void tdvp_engine_free(tdvp_engine_t *engine) {
    if (!engine) return;
    dmrg_environments_free(engine->env);
    free(engine);
}

void tdvp_set_dt(tdvp_engine_t *engine, double dt) {
    if (engine) engine->config.dt = dt;
}

double tdvp_get_time(const tdvp_engine_t *engine) {
    return engine ? engine->current_time : 0.0;
}

// ============================================================================
// TWO-SITE TDVP STEP
// ============================================================================

/**
 * @brief Evolve two-site tensor
 */
static int tdvp_evolve_two_site(tn_mps_state_t *mps,
                                 const mpo_t *mpo,
                                 dmrg_environments_t *env,
                                 uint32_t site,
                                 double complex dt,
                                 const tdvp_config_t *config,
                                 double *truncation_error) {
    if (!mps || !mpo || !env || site >= mps->num_qubits - 1) return -1;

    tensor_t *A = mps->tensors[site];
    tensor_t *B = mps->tensors[site + 1];

    uint32_t chi_l = A->dims[0];
    uint32_t d = A->dims[1];
    uint32_t chi_m = A->dims[2];
    uint32_t chi_r = B->dims[2];

    // Check environments
    if (!env->L[site] || !env->R[site + 1]) return -1;

    // Build effective Hamiltonian
    effective_hamiltonian_t H_eff = {
        .L = env->L[site],
        .R = env->R[site + 1],
        .W_left = &mpo->tensors[site],
        .W_right = &mpo->tensors[site + 1],
        .chi_l = chi_l,
        .chi_r = chi_r,
        .phys_dim = d,
        .two_site = true
    };

    // Form theta = A @ B
    uint32_t theta_dims[4] = {chi_l, d, d, chi_r};
    tensor_t *theta = tensor_create(4, theta_dims);
    if (!theta) return -1;

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

    // Evolve: theta_new = exp(-i*H*dt) @ theta
    tensor_t *theta_evolved = tensor_create(4, theta_dims);
    if (!theta_evolved) {
        tensor_free(theta);
        return -1;
    }

    double complex exponent = -I * dt;  // Real time
    if (config->evolution_type == TDVP_IMAGINARY_TIME) {
        exponent = -dt;  // Imaginary time
    }

    if (lanczos_expm(&H_eff, theta, exponent,
                      config->lanczos_max_iter, config->lanczos_tol,
                      theta_evolved) != 0) {
        tensor_free(theta);
        tensor_free(theta_evolved);
        return -1;
    }

    tensor_free(theta);

    // SVD split evolved tensor
    uint32_t mat_dims[2] = {chi_l * d, d * chi_r};
    tensor_t *mat = tensor_reshape(theta_evolved, 2, mat_dims);
    tensor_free(theta_evolved);

    if (!mat) return -1;

    svd_compress_config_t svd_cfg = svd_compress_config_default();
    svd_cfg.max_bond_dim = config->max_bond_dim;
    svd_cfg.cutoff = config->svd_cutoff;

    svd_compress_result_t *svd = svd_compress(mat, &svd_cfg);
    tensor_free(mat);

    if (!svd) return -1;

    *truncation_error = svd->truncation_error;
    uint32_t new_bond = svd->bond_dim;

    // Absorb S into B (for left-canonical form)
    for (uint32_t i = 0; i < new_bond; i++) {
        for (uint32_t j = 0; j < svd->right->dims[1]; j++) {
            svd->right->data[i * svd->right->dims[1] + j] *= svd->singular_values[i];
        }
    }

    // Reshape to new tensors
    uint32_t A_dims[3] = {chi_l, d, new_bond};
    tensor_t *A_new = tensor_reshape(svd->left, 3, A_dims);

    uint32_t B_dims[3] = {new_bond, d, chi_r};
    tensor_t *B_new = tensor_reshape(svd->right, 3, B_dims);

    if (!A_new || !B_new) {
        if (A_new) tensor_free(A_new);
        if (B_new) tensor_free(B_new);
        svd_compress_result_free(svd);
        return -1;
    }

    // Update MPS
    tensor_free(mps->tensors[site]);
    tensor_free(mps->tensors[site + 1]);
    mps->tensors[site] = A_new;
    mps->tensors[site + 1] = B_new;
    mps->bond_dims[site] = new_bond;

    svd_compress_result_free(svd);
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

    // Rebuild environments - need both L and R for the sweeps
    if (dmrg_init_left_environments(env, mps, mpo) != 0) {
        fprintf(stderr, "TDVP: Failed to initialize left environments\n");
        return -1;
    }
    if (dmrg_init_right_environments(env, mps, mpo) != 0) {
        fprintf(stderr, "TDVP: Failed to initialize right environments\n");
        return -1;
    }

    // Left-to-right sweep with dt/2
    for (uint32_t site = 0; site < n - 1; site++) {
        double trunc_err;
        if (tdvp_evolve_two_site(mps, mpo, env, site, dt / 2,
                                  &engine->config, &trunc_err) != 0) {
            fprintf(stderr, "TDVP: Failed at site %u (L->R)\n", site);
            return -1;
        }
        total_trunc_error += trunc_err;

        if (mps->bond_dims[site] > max_bond) {
            max_bond = mps->bond_dims[site];
        }

        // Update left environment
        if (site < n - 2) {
            dmrg_update_left_environment(env, mps, mpo, site);
        }
    }

    // Rebuild right environments
    dmrg_init_right_environments(env, mps, mpo);

    // Right-to-left sweep with dt/2
    for (int site = n - 2; site >= 0; site--) {
        double trunc_err;
        if (tdvp_evolve_two_site(mps, mpo, env, site, dt / 2,
                                  &engine->config, &trunc_err) != 0) {
            fprintf(stderr, "TDVP: Failed at site %d (R->L)\n", site);
            return -1;
        }
        total_trunc_error += trunc_err;

        if (mps->bond_dims[site] > max_bond) {
            max_bond = mps->bond_dims[site];
        }

        // Update right environment
        if (site > 0) {
            dmrg_update_right_environment(env, mps, mpo, site + 1);
        }
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
    }

    return 0;
}

int tdvp_evolve_to(tdvp_engine_t *engine,
                    double target_time,
                    tdvp_history_t *history) {
    if (!engine) return -1;

    while (engine->current_time < target_time) {
        // Adjust dt for last step if needed
        double remaining = target_time - engine->current_time;
        if (remaining < engine->config.dt) {
            engine->config.dt = remaining;
        }

        tdvp_result_t result;
        if (tdvp_step(engine, &result) != 0) {
            return -1;
        }

        if (history) {
            tdvp_history_add(history, &result);
        }

        if (engine->config.verbose) {
            printf("  t=%.4f: E=%.8f, norm=%.6f, chi_max=%u, trunc=%.2e\n",
                   result.time, result.energy, result.norm,
                   result.max_bond_dim, result.truncation_error);
        }
    }

    return 0;
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

    tdvp_result_t result;
    int ret = tdvp_step(engine, &result);

    if (energy) *energy = result.energy;

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

    while (engine->current_time < target_time) {
        double remaining = target_time - engine->current_time;
        if (remaining < engine->config.dt) {
            engine->config.dt = remaining;
        }

        tdvp_result_t result;
        if (tdvp_step(engine, &result) != 0) {
            return -1;
        }

        step_count++;

        if (callback && (step_count % measure_interval == 0)) {
            callback(engine->mps, engine->current_time, user_data);
        }
    }

    // Final measurement
    if (callback) {
        callback(engine->mps, engine->current_time, user_data);
    }

    return 0;
}
