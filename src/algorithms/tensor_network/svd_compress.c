/**
 * @file svd_compress.c
 * @brief SVD-based tensor compression implementation
 *
 * Full production implementation of truncated SVD operations for
 * tensor network bond dimension management.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "svd_compress.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// CONFIGURATION MANAGEMENT
// ============================================================================

svd_compress_config_t svd_compress_config_default(void) {
    svd_compress_config_t config = {
        .mode = SVD_TRUNCATE_ADAPTIVE,
        .max_bond_dim = SVD_DEFAULT_MAX_BOND,
        .cutoff = SVD_DEFAULT_CUTOFF,
        .max_error = SVD_DEFAULT_MAX_ERROR,
        .normalize = false,
        .track_error = true
    };
    return config;
}

svd_compress_config_t svd_compress_config_fixed(uint32_t max_bond) {
    svd_compress_config_t config = {
        .mode = SVD_TRUNCATE_FIXED,
        .max_bond_dim = max_bond,
        .cutoff = 0.0,
        .max_error = 0.0,
        .normalize = false,
        .track_error = true
    };
    return config;
}

svd_compress_config_t svd_compress_config_error_bounded(double max_error,
                                                         uint32_t max_bond) {
    svd_compress_config_t config = {
        .mode = SVD_TRUNCATE_ERROR,
        .max_bond_dim = max_bond,
        .cutoff = 0.0,
        .max_error = max_error,
        .normalize = false,
        .track_error = true
    };
    return config;
}

svd_compress_error_t svd_compress_config_validate(const svd_compress_config_t *config) {
    if (!config) return SVD_COMPRESS_ERROR_NULL_PTR;

    if (config->max_bond_dim == 0) return SVD_COMPRESS_ERROR_INVALID_CONFIG;
    if (config->cutoff < 0.0) return SVD_COMPRESS_ERROR_INVALID_CONFIG;
    if (config->max_error < 0.0) return SVD_COMPRESS_ERROR_INVALID_CONFIG;

    return SVD_COMPRESS_SUCCESS;
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/**
 * @brief Determine truncation rank based on configuration
 */
static uint32_t determine_truncation_rank(const double *singular_values,
                                           uint32_t count,
                                           const svd_compress_config_t *config,
                                           double *truncation_error) {
    if (count == 0) {
        if (truncation_error) *truncation_error = 0.0;
        return 0;
    }

    uint32_t k = count;
    double error_sq = 0.0;

    switch (config->mode) {
        case SVD_TRUNCATE_FIXED:
            // Keep at most max_bond_dim values
            k = (count < config->max_bond_dim) ? count : config->max_bond_dim;
            // Compute truncation error
            for (uint32_t i = k; i < count; i++) {
                error_sq += singular_values[i] * singular_values[i];
            }
            break;

        case SVD_TRUNCATE_CUTOFF:
            // Discard values below cutoff
            k = count;
            error_sq = 0.0;
            while (k > SVD_MIN_BOND && singular_values[k-1] < config->cutoff) {
                error_sq += singular_values[k-1] * singular_values[k-1];
                k--;
            }
            // Also apply max bond
            while (k > config->max_bond_dim) {
                k--;
                error_sq += singular_values[k] * singular_values[k];
            }
            break;

        case SVD_TRUNCATE_ERROR:
            // Truncate to achieve error bound
            {
                double target_error_sq = config->max_error * config->max_error;
                k = count;
                error_sq = 0.0;

                // Remove values from the end until error exceeds bound
                while (k > SVD_MIN_BOND) {
                    double new_error_sq = error_sq + singular_values[k-1] * singular_values[k-1];
                    if (new_error_sq > target_error_sq) break;
                    error_sq = new_error_sq;
                    k--;
                }

                // Also apply max bond
                while (k > config->max_bond_dim) {
                    error_sq += singular_values[k-1] * singular_values[k-1];
                    k--;
                }
            }
            break;

        case SVD_TRUNCATE_ADAPTIVE:
        default:
            // Combine cutoff and max bond
            k = count;
            error_sq = 0.0;

            // First apply cutoff
            while (k > SVD_MIN_BOND && singular_values[k-1] < config->cutoff) {
                error_sq += singular_values[k-1] * singular_values[k-1];
                k--;
            }

            // Then apply max bond if needed
            while (k > config->max_bond_dim) {
                error_sq += singular_values[k-1] * singular_values[k-1];
                k--;
            }

            // Finally check if we can truncate more within error bound
            if (config->max_error > 0.0) {
                double target_error_sq = config->max_error * config->max_error;
                while (k > SVD_MIN_BOND && error_sq < target_error_sq) {
                    double new_error_sq = error_sq + singular_values[k-1] * singular_values[k-1];
                    if (new_error_sq > target_error_sq) break;
                    error_sq = new_error_sq;
                    k--;
                }
            }
            break;
    }

    if (truncation_error) *truncation_error = sqrt(error_sq);
    return k;
}

// ============================================================================
// CORE SVD COMPRESSION
// ============================================================================

svd_compress_result_t *svd_compress(const tensor_t *tensor,
                                     const svd_compress_config_t *config) {
    if (!tensor || !config) return NULL;
    if (tensor->rank != 2) return NULL;

    // Perform SVD
    tensor_svd_result_t *svd = tensor_svd(tensor, 0, 0.0);  // Full SVD first
    if (!svd) return NULL;

    // Determine truncation
    double truncation_error;
    uint32_t k = determine_truncation_rank(svd->S, svd->k, config, &truncation_error);

    if (k == 0) k = 1;  // Keep at least one

    // Allocate result
    svd_compress_result_t *result = (svd_compress_result_t *)calloc(1, sizeof(svd_compress_result_t));
    if (!result) {
        tensor_svd_free(svd);
        return NULL;
    }

    uint32_t m = tensor->dims[0];
    uint32_t n = tensor->dims[1];

    // Create truncated tensors
    result->left = tensor_create_matrix(m, k);
    result->right = tensor_create_matrix(k, n);
    result->singular_values = (double *)malloc(k * sizeof(double));

    if (!result->left || !result->right || !result->singular_values) {
        tensor_svd_free(svd);
        svd_compress_result_free(result);
        return NULL;
    }

    // Copy truncated data
    // Left: U[:, :k]
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            result->left->data[i * k + j] = svd->U->data[i * svd->k + j];
        }
    }

    // Right: Vh[:k, :]
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < n; j++) {
            result->right->data[i * n + j] = svd->Vh->data[i * n + j];
        }
    }

    // Singular values
    memcpy(result->singular_values, svd->S, k * sizeof(double));

    result->bond_dim = k;
    result->truncation_error = truncation_error;
    result->original_bond_dim = svd->k;
    result->num_discarded = svd->k - k;

    // Normalize if requested
    if (config->normalize) {
        double norm = svd_normalize_singular_values(result->singular_values, k);
        (void)norm;  // Unused but available
    }

    tensor_svd_free(svd);
    return result;
}

svd_compress_result_t *svd_compress_split(const tensor_t *tensor,
                                           uint32_t split_axis,
                                           const svd_compress_config_t *config) {
    if (!tensor || !config) return NULL;
    if (split_axis >= tensor->rank || split_axis == 0) return NULL;

    // Reshape to matrix: [d0*d1*...*d_{split-1}, d_{split}*...*d_{rank-1}]
    uint32_t left_size = 1;
    for (uint32_t i = 0; i < split_axis; i++) {
        left_size *= tensor->dims[i];
    }

    uint32_t right_size = 1;
    for (uint32_t i = split_axis; i < tensor->rank; i++) {
        right_size *= tensor->dims[i];
    }

    uint32_t mat_dims[2] = {left_size, right_size};
    tensor_t *mat = tensor_reshape(tensor, 2, mat_dims);
    if (!mat) return NULL;

    // Perform SVD compression
    svd_compress_result_t *result = svd_compress(mat, config);
    tensor_free(mat);

    if (!result) return NULL;

    // Reshape left tensor back to original left shape plus bond
    uint32_t left_rank = split_axis + 1;
    uint32_t left_dims[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < split_axis; i++) {
        left_dims[i] = tensor->dims[i];
    }
    left_dims[split_axis] = result->bond_dim;

    tensor_t *left_reshaped = tensor_reshape(result->left, left_rank, left_dims);
    if (!left_reshaped) {
        svd_compress_result_free(result);
        return NULL;
    }

    // Reshape right tensor back
    uint32_t right_rank = tensor->rank - split_axis + 1;
    uint32_t right_dims[TENSOR_MAX_RANK];
    right_dims[0] = result->bond_dim;
    for (uint32_t i = split_axis; i < tensor->rank; i++) {
        right_dims[i - split_axis + 1] = tensor->dims[i];
    }

    tensor_t *right_reshaped = tensor_reshape(result->right, right_rank, right_dims);
    if (!right_reshaped) {
        tensor_free(left_reshaped);
        svd_compress_result_free(result);
        return NULL;
    }

    // Replace tensors in result
    tensor_free(result->left);
    tensor_free(result->right);
    result->left = left_reshaped;
    result->right = right_reshaped;

    return result;
}

svd_compress_result_t *svd_compress_bond(const tensor_t *left,
                                          const tensor_t *right,
                                          const svd_compress_config_t *config,
                                          bool left_canonical) {
    if (!left || !right || !config) return NULL;
    if (left->rank < 1 || right->rank < 1) return NULL;

    // Check bond dimension compatibility
    uint32_t bond_left = left->dims[left->rank - 1];
    uint32_t bond_right = right->dims[0];
    if (bond_left != bond_right) return NULL;

    // Contract tensors
    // Reshape left to [prod(other_dims), bond]
    uint32_t left_flat = 1;
    for (uint32_t i = 0; i < left->rank - 1; i++) {
        left_flat *= left->dims[i];
    }

    // Reshape right to [bond, prod(other_dims)]
    uint32_t right_flat = 1;
    for (uint32_t i = 1; i < right->rank; i++) {
        right_flat *= right->dims[i];
    }

    uint32_t left_mat_dims[2] = {left_flat, bond_left};
    uint32_t right_mat_dims[2] = {bond_right, right_flat};

    tensor_t *left_mat = tensor_reshape(left, 2, left_mat_dims);
    tensor_t *right_mat = tensor_reshape(right, 2, right_mat_dims);

    if (!left_mat || !right_mat) {
        tensor_free(left_mat);
        tensor_free(right_mat);
        return NULL;
    }

    // Contract: result = left_mat @ right_mat
    tensor_t *contracted = tensor_matmul(left_mat, right_mat);
    tensor_free(left_mat);
    tensor_free(right_mat);

    if (!contracted) return NULL;

    // SVD compress the contracted result
    svd_compress_result_t *result = svd_compress(contracted, config);
    tensor_free(contracted);

    if (!result) return NULL;

    // Absorb singular values
    if (left_canonical) {
        // Absorb S into right: right = S @ right
        for (uint32_t i = 0; i < result->bond_dim; i++) {
            for (uint32_t j = 0; j < right_flat; j++) {
                result->right->data[i * right_flat + j] *= result->singular_values[i];
            }
        }
    } else {
        // Absorb S into left: left = left @ S
        for (uint32_t i = 0; i < left_flat; i++) {
            for (uint32_t j = 0; j < result->bond_dim; j++) {
                result->left->data[i * result->bond_dim + j] *= result->singular_values[j];
            }
        }
    }

    // Reshape back to tensor form
    uint32_t new_left_dims[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < left->rank - 1; i++) {
        new_left_dims[i] = left->dims[i];
    }
    new_left_dims[left->rank - 1] = result->bond_dim;

    tensor_t *left_reshaped = tensor_reshape(result->left, left->rank, new_left_dims);
    if (!left_reshaped) {
        svd_compress_result_free(result);
        return NULL;
    }

    uint32_t new_right_dims[TENSOR_MAX_RANK];
    new_right_dims[0] = result->bond_dim;
    for (uint32_t i = 1; i < right->rank; i++) {
        new_right_dims[i] = right->dims[i];
    }

    tensor_t *right_reshaped = tensor_reshape(result->right, right->rank, new_right_dims);
    if (!right_reshaped) {
        tensor_free(left_reshaped);
        svd_compress_result_free(result);
        return NULL;
    }

    tensor_free(result->left);
    tensor_free(result->right);
    result->left = left_reshaped;
    result->right = right_reshaped;

    return result;
}

void svd_compress_result_free(svd_compress_result_t *result) {
    if (!result) return;

    tensor_free(result->left);
    tensor_free(result->right);
    free(result->singular_values);
    free(result);
}

// ============================================================================
// CANONICAL FORMS
// ============================================================================

svd_compress_result_t *svd_left_canonicalize(const tensor_t *tensor,
                                              uint32_t bond_axis,
                                              const svd_compress_config_t *config) {
    if (!tensor) return NULL;
    if (bond_axis >= tensor->rank) return NULL;

    // Default config if not provided
    svd_compress_config_t default_cfg = svd_compress_config_default();
    if (!config) config = &default_cfg;

    // Move bond axis to last position if needed
    tensor_t *work = NULL;
    if (bond_axis != tensor->rank - 1) {
        uint32_t perm[TENSOR_MAX_RANK];
        uint32_t j = 0;
        for (uint32_t i = 0; i < tensor->rank; i++) {
            if (i != bond_axis) perm[j++] = i;
        }
        perm[tensor->rank - 1] = bond_axis;

        work = tensor_transpose(tensor, perm);
        if (!work) return NULL;
    } else {
        work = tensor_copy(tensor);
        if (!work) return NULL;
    }

    // Reshape to matrix [all_other, bond]
    uint32_t other_size = 1;
    for (uint32_t i = 0; i < work->rank - 1; i++) {
        other_size *= work->dims[i];
    }
    uint32_t bond_size = work->dims[work->rank - 1];

    uint32_t mat_dims[2] = {other_size, bond_size};
    tensor_t *mat = tensor_reshape(work, 2, mat_dims);
    tensor_free(work);

    if (!mat) return NULL;

    // QR decomposition for left-canonical form
    tensor_qr_result_t *qr = tensor_qr(mat);
    tensor_free(mat);

    if (!qr) return NULL;

    // Apply truncation if config specifies
    svd_compress_result_t *result = (svd_compress_result_t *)calloc(1, sizeof(svd_compress_result_t));
    if (!result) {
        tensor_qr_free(qr);
        return NULL;
    }

    uint32_t k = qr->Q->dims[1];  // Number of columns in Q

    // Truncate based on R if config requires
    if (config->max_bond_dim > 0 && k > config->max_bond_dim) {
        // Need SVD for truncation
        tensor_qr_free(qr);

        // Fall back to SVD-based truncation
        mat_dims[0] = other_size;
        mat_dims[1] = bond_size;
        mat = tensor_reshape(tensor, 2, mat_dims);
        if (!mat) {
            free(result);
            return NULL;
        }

        svd_compress_result_t *svd_result = svd_compress(mat, config);
        tensor_free(mat);

        if (!svd_result) {
            free(result);
            return NULL;
        }

        // Absorb singular values into right tensor
        for (uint32_t i = 0; i < svd_result->bond_dim; i++) {
            for (uint32_t j = 0; j < svd_result->right->dims[1]; j++) {
                svd_result->right->data[i * svd_result->right->dims[1] + j] *= svd_result->singular_values[i];
            }
        }

        // Reshape left back
        uint32_t left_dims[TENSOR_MAX_RANK];
        for (uint32_t i = 0; i < tensor->rank - 1; i++) {
            left_dims[i] = tensor->dims[i];
        }
        left_dims[tensor->rank - 1] = svd_result->bond_dim;

        tensor_t *left_reshaped = tensor_reshape(svd_result->left, tensor->rank, left_dims);
        if (!left_reshaped) {
            svd_compress_result_free(svd_result);
            free(result);
            return NULL;
        }

        result->left = left_reshaped;
        result->right = svd_result->right;
        result->singular_values = svd_result->singular_values;
        result->bond_dim = svd_result->bond_dim;
        result->truncation_error = svd_result->truncation_error;
        result->original_bond_dim = svd_result->original_bond_dim;
        result->num_discarded = svd_result->num_discarded;

        svd_result->right = NULL;
        svd_result->singular_values = NULL;
        tensor_free(svd_result->left);
        free(svd_result);

        return result;
    }

    // No truncation needed, use QR result
    // Reshape Q back to tensor form
    uint32_t left_dims[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < tensor->rank - 1; i++) {
        left_dims[i] = tensor->dims[i];
    }
    left_dims[tensor->rank - 1] = k;

    result->left = tensor_reshape(qr->Q, tensor->rank, left_dims);
    result->right = qr->R;
    qr->R = NULL;  // Transfer ownership

    result->singular_values = NULL;  // Not available from QR
    result->bond_dim = k;
    result->truncation_error = 0.0;
    result->original_bond_dim = bond_size;
    result->num_discarded = 0;

    tensor_qr_free(qr);

    if (!result->left) {
        svd_compress_result_free(result);
        return NULL;
    }

    return result;
}

svd_compress_result_t *svd_right_canonicalize(const tensor_t *tensor,
                                               uint32_t bond_axis,
                                               const svd_compress_config_t *config) {
    if (!tensor) return NULL;
    if (bond_axis >= tensor->rank) return NULL;

    svd_compress_config_t default_cfg = svd_compress_config_default();
    if (!config) config = &default_cfg;

    // Move bond axis to first position if needed
    tensor_t *work = NULL;
    if (bond_axis != 0) {
        uint32_t perm[TENSOR_MAX_RANK];
        perm[0] = bond_axis;
        uint32_t j = 1;
        for (uint32_t i = 0; i < tensor->rank; i++) {
            if (i != bond_axis) perm[j++] = i;
        }

        work = tensor_transpose(tensor, perm);
        if (!work) return NULL;
    } else {
        work = tensor_copy(tensor);
        if (!work) return NULL;
    }

    // Reshape to matrix [bond, all_other]
    uint32_t bond_size = work->dims[0];
    uint32_t other_size = 1;
    for (uint32_t i = 1; i < work->rank; i++) {
        other_size *= work->dims[i];
    }

    uint32_t mat_dims[2] = {bond_size, other_size};
    tensor_t *mat = tensor_reshape(work, 2, mat_dims);
    tensor_free(work);

    if (!mat) return NULL;

    // LQ decomposition for right-canonical form
    tensor_qr_result_t *lq = tensor_lq(mat);
    tensor_free(mat);

    if (!lq) return NULL;

    svd_compress_result_t *result = (svd_compress_result_t *)calloc(1, sizeof(svd_compress_result_t));
    if (!result) {
        tensor_qr_free(lq);
        return NULL;
    }

    uint32_t k = lq->Q->dims[0];  // Q is k x other_size

    // Reshape Q back to tensor form
    uint32_t right_dims[TENSOR_MAX_RANK];
    right_dims[0] = k;
    for (uint32_t i = 1; i < tensor->rank; i++) {
        right_dims[i] = tensor->dims[i];
    }

    result->right = tensor_reshape(lq->Q, tensor->rank, right_dims);
    result->left = lq->R;  // L is actually stored in R field for LQ
    lq->R = NULL;

    result->singular_values = NULL;
    result->bond_dim = k;
    result->truncation_error = 0.0;
    result->original_bond_dim = bond_size;
    result->num_discarded = 0;

    tensor_qr_free(lq);

    if (!result->right) {
        svd_compress_result_free(result);
        return NULL;
    }

    return result;
}

svd_compress_result_t *svd_mixed_canonicalize(const tensor_t *tensor,
                                               uint32_t left_axis,
                                               uint32_t right_axis,
                                               const svd_compress_config_t *config) {
    if (!tensor) return NULL;
    if (left_axis >= tensor->rank || right_axis >= tensor->rank) return NULL;
    if (left_axis >= right_axis) return NULL;

    svd_compress_config_t default_cfg = svd_compress_config_default();
    if (!config) config = &default_cfg;

    // This is essentially SVD split at the specified axes
    // First reshape to 3D: [left_dims, middle_dims, right_dims]

    uint32_t left_size = 1;
    for (uint32_t i = 0; i <= left_axis; i++) {
        left_size *= tensor->dims[i];
    }

    uint32_t middle_size = 1;
    for (uint32_t i = left_axis + 1; i < right_axis; i++) {
        middle_size *= tensor->dims[i];
    }

    uint32_t right_size = 1;
    for (uint32_t i = right_axis; i < tensor->rank; i++) {
        right_size *= tensor->dims[i];
    }

    // Reshape to matrix for SVD
    uint32_t mat_dims[2] = {left_size, middle_size * right_size};
    tensor_t *mat = tensor_reshape(tensor, 2, mat_dims);
    if (!mat) return NULL;

    svd_compress_result_t *result = svd_compress(mat, config);
    tensor_free(mat);

    return result;
}

// ============================================================================
// TRUNCATION ANALYSIS
// ============================================================================

uint32_t svd_optimal_rank(const double *singular_values, uint32_t count,
                          double max_error, bool relative,
                          double *truncation_error) {
    if (!singular_values || count == 0) {
        if (truncation_error) *truncation_error = 0.0;
        return 0;
    }

    // Compute total norm squared if relative
    double total_sq = 0.0;
    if (relative) {
        for (uint32_t i = 0; i < count; i++) {
            total_sq += singular_values[i] * singular_values[i];
        }
    }

    double target_error_sq;
    if (relative) {
        target_error_sq = max_error * max_error * total_sq;
    } else {
        target_error_sq = max_error * max_error;
    }

    // Find minimum k such that error <= target
    double error_sq = 0.0;
    uint32_t k = count;

    for (int i = (int)count - 1; i >= 0; i--) {
        double new_error_sq = error_sq + singular_values[i] * singular_values[i];
        if (new_error_sq > target_error_sq) break;
        error_sq = new_error_sq;
        k = (uint32_t)i;
    }

    if (k == 0) k = 1;  // Keep at least one

    if (truncation_error) *truncation_error = sqrt(error_sq);
    return k;
}

double svd_entanglement_entropy(const double *singular_values, uint32_t count) {
    if (!singular_values || count == 0) return 0.0;

    // Normalize to get probabilities
    double total_sq = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        total_sq += singular_values[i] * singular_values[i];
    }

    // Use 1e-200 threshold to catch only true numerical zero
    // Values between 1e-200 and 1e-30 are legitimate small numbers
    if (total_sq < 1e-200) return 0.0;

    // Compute von Neumann entropy
    double entropy = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double p = (singular_values[i] * singular_values[i]) / total_sq;
        // Use 1e-15 threshold for probability (machine epsilon level)
        // Very small probabilities contribute negligibly to entropy
        if (p > 1e-15) {
            entropy -= p * log(p);
        }
    }

    return entropy;
}

void svd_schmidt_coefficients(const double *singular_values, uint32_t count,
                               double *schmidt_coeffs) {
    if (!singular_values || !schmidt_coeffs || count == 0) return;

    double total_sq = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        total_sq += singular_values[i] * singular_values[i];
    }

    if (total_sq < 1e-200) {
        memset(schmidt_coeffs, 0, count * sizeof(double));
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        schmidt_coeffs[i] = (singular_values[i] * singular_values[i]) / total_sq;
    }
}

uint32_t svd_bond_for_fidelity(const double *singular_values, uint32_t count,
                                double target_fidelity) {
    if (!singular_values || count == 0) return 0;
    if (target_fidelity <= 0.0) return count;
    if (target_fidelity >= 1.0) return 1;

    // Fidelity = sum(s_i^2 for kept) / sum(s_j^2 for all)
    double total_sq = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        total_sq += singular_values[i] * singular_values[i];
    }

    if (total_sq < 1e-200) return 1;

    double target_sum_sq = target_fidelity * total_sq;
    double current_sum_sq = 0.0;

    for (uint32_t k = 0; k < count; k++) {
        current_sum_sq += singular_values[k] * singular_values[k];
        if (current_sum_sq >= target_sum_sq) {
            return k + 1;
        }
    }

    return count;
}

// ============================================================================
// STATISTICS TRACKING
// ============================================================================

svd_compress_stats_t svd_compress_stats_init(void) {
    svd_compress_stats_t stats = {
        .cumulative_error = 0.0,
        .max_single_error = 0.0,
        .num_truncations = 0,
        .total_discarded = 0,
        .min_singular_kept = HUGE_VAL,
        .max_singular_kept = 0.0
    };
    return stats;
}

void svd_compress_stats_update(svd_compress_stats_t *stats,
                                const svd_compress_result_t *result) {
    if (!stats || !result) return;

    stats->cumulative_error += result->truncation_error;
    if (result->truncation_error > stats->max_single_error) {
        stats->max_single_error = result->truncation_error;
    }

    stats->num_truncations++;
    stats->total_discarded += result->num_discarded;

    if (result->singular_values && result->bond_dim > 0) {
        if (result->singular_values[0] > stats->max_singular_kept) {
            stats->max_singular_kept = result->singular_values[0];
        }
        if (result->singular_values[result->bond_dim - 1] < stats->min_singular_kept) {
            stats->min_singular_kept = result->singular_values[result->bond_dim - 1];
        }
    }
}

void svd_compress_stats_reset(svd_compress_stats_t *stats) {
    if (!stats) return;
    *stats = svd_compress_stats_init();
}

void svd_compress_stats_print(const svd_compress_stats_t *stats) {
    if (!stats) return;

    printf("SVD Compression Statistics:\n");
    printf("  Truncations:       %lu\n", (unsigned long)stats->num_truncations);
    printf("  Total discarded:   %lu\n", (unsigned long)stats->total_discarded);
    printf("  Cumulative error:  %.6e\n", stats->cumulative_error);
    printf("  Max single error:  %.6e\n", stats->max_single_error);
    printf("  Max singular kept: %.6e\n", stats->max_singular_kept);
    printf("  Min singular kept: %.6e\n", stats->min_singular_kept);
}

// ============================================================================
// UTILITIES
// ============================================================================

const char *svd_compress_error_string(svd_compress_error_t error) {
    switch (error) {
        case SVD_COMPRESS_SUCCESS: return "Success";
        case SVD_COMPRESS_ERROR_NULL_PTR: return "Null pointer";
        case SVD_COMPRESS_ERROR_INVALID_DIMS: return "Invalid dimensions";
        case SVD_COMPRESS_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case SVD_COMPRESS_ERROR_SVD_FAILED: return "SVD computation failed";
        case SVD_COMPRESS_ERROR_INVALID_CONFIG: return "Invalid configuration";
        case SVD_COMPRESS_ERROR_NUMERICAL: return "Numerical error";
        default: return "Unknown error";
    }
}

svd_compress_error_t svd_absorb_singular_values(tensor_t *tensor,
                                                 const double *singular_values,
                                                 uint32_t count,
                                                 uint32_t axis) {
    if (!tensor || !singular_values) return SVD_COMPRESS_ERROR_NULL_PTR;
    if (axis >= tensor->rank) return SVD_COMPRESS_ERROR_INVALID_DIMS;
    if (tensor->dims[axis] != count) return SVD_COMPRESS_ERROR_INVALID_DIMS;

    uint64_t stride = tensor->strides[axis];
    uint32_t dim = tensor->dims[axis];

    // For each position along the axis, multiply by corresponding singular value
    uint32_t indices[TENSOR_MAX_RANK] = {0};

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        tensor_get_multi_index(tensor, i, indices);
        tensor->data[i] *= singular_values[indices[axis]];
    }

    return SVD_COMPRESS_SUCCESS;
}

double svd_normalize_singular_values(double *singular_values, uint32_t count) {
    if (!singular_values || count == 0) return 0.0;

    double norm_sq = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        norm_sq += singular_values[i] * singular_values[i];
    }

    double norm = sqrt(norm_sq);
    if (norm > 1e-30) {
        for (uint32_t i = 0; i < count; i++) {
            singular_values[i] /= norm;
        }
    }

    return norm;
}
