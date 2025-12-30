/**
 * @file svd_compress.h
 * @brief SVD-based tensor compression for bond dimension management
 *
 * Provides truncated SVD operations critical for maintaining computational
 * tractability in tensor network simulations. Controls the bond dimension
 * (entanglement capacity) through adaptive truncation strategies.
 *
 * Key concepts:
 * - Bond dimension (chi): Maximum size of tensor indices connecting tensors
 * - Truncation error: Frobenius norm of discarded singular values
 * - Canonical form: Orthonormal gauge for MPS tensors
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef SVD_COMPRESS_H
#define SVD_COMPRESS_H

#include "tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/** Default maximum bond dimension */
#define SVD_DEFAULT_MAX_BOND 256

/** Default SVD cutoff (singular values below this are discarded) */
#define SVD_DEFAULT_CUTOFF 1e-12

/** Default maximum truncation error */
#define SVD_DEFAULT_MAX_ERROR 1e-10

/** Minimum bond dimension to preserve */
#define SVD_MIN_BOND 1

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    SVD_COMPRESS_SUCCESS = 0,
    SVD_COMPRESS_ERROR_NULL_PTR = -1,
    SVD_COMPRESS_ERROR_INVALID_DIMS = -2,
    SVD_COMPRESS_ERROR_ALLOC_FAILED = -3,
    SVD_COMPRESS_ERROR_SVD_FAILED = -4,
    SVD_COMPRESS_ERROR_INVALID_CONFIG = -5,
    SVD_COMPRESS_ERROR_NUMERICAL = -6
} svd_compress_error_t;

// ============================================================================
// TRUNCATION CONFIGURATION
// ============================================================================

/**
 * @brief Truncation strategy for SVD compression
 */
typedef enum {
    SVD_TRUNCATE_FIXED,     /**< Keep fixed number of singular values */
    SVD_TRUNCATE_CUTOFF,    /**< Discard values below absolute cutoff */
    SVD_TRUNCATE_ERROR,     /**< Truncate to achieve target error bound */
    SVD_TRUNCATE_ADAPTIVE   /**< Combine cutoff and max bond adaptively */
} svd_truncation_mode_t;

/**
 * @brief SVD compression configuration
 */
typedef struct {
    svd_truncation_mode_t mode;     /**< Truncation strategy */
    uint32_t max_bond_dim;          /**< Maximum bond dimension */
    double cutoff;                  /**< Absolute cutoff for singular values */
    double max_error;               /**< Maximum allowed truncation error */
    bool normalize;                 /**< Normalize after truncation */
    bool track_error;               /**< Track cumulative truncation error */
} svd_compress_config_t;

/**
 * @brief Result of SVD compression operation
 */
typedef struct {
    tensor_t *left;                 /**< Left tensor (A in A*S*B) or left-canonical */
    tensor_t *right;                /**< Right tensor (B in A*S*B) or right-canonical */
    double *singular_values;        /**< Singular values (may be absorbed) */
    uint32_t bond_dim;              /**< Resulting bond dimension */
    double truncation_error;        /**< Truncation error from this operation */
    uint32_t original_bond_dim;     /**< Bond dimension before truncation */
    uint32_t num_discarded;         /**< Number of discarded singular values */
} svd_compress_result_t;

/**
 * @brief Statistics for tracking compression across operations
 */
typedef struct {
    double cumulative_error;        /**< Total accumulated truncation error */
    double max_single_error;        /**< Largest single truncation error */
    uint64_t num_truncations;       /**< Total number of truncation operations */
    uint64_t total_discarded;       /**< Total singular values discarded */
    double min_singular_kept;       /**< Smallest singular value kept */
    double max_singular_kept;       /**< Largest singular value kept */
} svd_compress_stats_t;

// ============================================================================
// CONFIGURATION MANAGEMENT
// ============================================================================

/**
 * @brief Create default compression configuration
 *
 * @return Default configuration (adaptive mode, chi=256, cutoff=1e-12)
 */
svd_compress_config_t svd_compress_config_default(void);

/**
 * @brief Create configuration for fixed bond dimension
 *
 * @param max_bond Maximum bond dimension to keep
 * @return Configuration for fixed truncation
 */
svd_compress_config_t svd_compress_config_fixed(uint32_t max_bond);

/**
 * @brief Create configuration for error-bounded truncation
 *
 * @param max_error Maximum allowed truncation error
 * @param max_bond Upper limit on bond dimension
 * @return Configuration for error-bounded truncation
 */
svd_compress_config_t svd_compress_config_error_bounded(double max_error,
                                                         uint32_t max_bond);

/**
 * @brief Validate compression configuration
 *
 * @param config Configuration to validate
 * @return SVD_COMPRESS_SUCCESS if valid
 */
svd_compress_error_t svd_compress_config_validate(const svd_compress_config_t *config);

// ============================================================================
// CORE SVD COMPRESSION
// ============================================================================

/**
 * @brief Compress a tensor by truncating singular values
 *
 * For a matrix A, computes truncated SVD: A ≈ U_k * S_k * V_k^H
 * where k is determined by the configuration.
 *
 * @param tensor Input tensor (rank 2)
 * @param config Compression configuration
 * @return Compression result or NULL on failure
 */
svd_compress_result_t *svd_compress(const tensor_t *tensor,
                                     const svd_compress_config_t *config);

/**
 * @brief Compress and split tensor along specified axis
 *
 * Reshapes tensor to matrix by grouping axes, performs SVD compression,
 * then reshapes back to tensor form.
 *
 * @param tensor Input tensor (any rank)
 * @param split_axis Axis to split at (axes 0..split_axis-1 go left)
 * @param config Compression configuration
 * @return Compression result with reshaped left/right tensors
 */
svd_compress_result_t *svd_compress_split(const tensor_t *tensor,
                                           uint32_t split_axis,
                                           const svd_compress_config_t *config);

/**
 * @brief Compress bond between two adjacent MPS tensors
 *
 * Given tensors A[i,a,b] and B[b,c,j], compresses the shared bond b.
 * Result can be in left-canonical, right-canonical, or mixed form.
 *
 * @param left Left tensor with bond on last axis
 * @param right Right tensor with bond on first axis
 * @param config Compression configuration
 * @param left_canonical If true, absorb S into right tensor
 * @return Compression result
 */
svd_compress_result_t *svd_compress_bond(const tensor_t *left,
                                          const tensor_t *right,
                                          const svd_compress_config_t *config,
                                          bool left_canonical);

/**
 * @brief Free compression result
 *
 * @param result Result to free
 */
void svd_compress_result_free(svd_compress_result_t *result);

// ============================================================================
// CANONICAL FORMS
// ============================================================================

/**
 * @brief Convert tensor to left-canonical form
 *
 * For MPS tensor A[i,a,b], produces L[i,a,c] and R[c,b] where
 * L satisfies: sum_ia L[i,a,c]* L[i,a,c'] = delta[c,c']
 *
 * @param tensor Input tensor (rank 3 for MPS)
 * @param bond_axis Axis of the bond to canonicalize (typically last)
 * @param config Compression config (NULL for no truncation)
 * @return Compression result with left-canonical tensor and remainder
 */
svd_compress_result_t *svd_left_canonicalize(const tensor_t *tensor,
                                              uint32_t bond_axis,
                                              const svd_compress_config_t *config);

/**
 * @brief Convert tensor to right-canonical form
 *
 * For MPS tensor A[i,a,b], produces L[a,c] and R[c,i,b] where
 * R satisfies: sum_ib R[c,i,b]* R[c',i,b] = delta[c,c']
 *
 * @param tensor Input tensor (rank 3 for MPS)
 * @param bond_axis Axis of the bond to canonicalize (typically first)
 * @param config Compression config (NULL for no truncation)
 * @return Compression result with right-canonical tensor and remainder
 */
svd_compress_result_t *svd_right_canonicalize(const tensor_t *tensor,
                                               uint32_t bond_axis,
                                               const svd_compress_config_t *config);

/**
 * @brief Convert tensor to mixed-canonical form centered at site
 *
 * Useful for DMRG-style algorithms where center site holds singular values.
 *
 * @param tensor Input tensor
 * @param left_axis Left bond axis
 * @param right_axis Right bond axis
 * @param config Compression config
 * @return Result with singular values on center tensor
 */
svd_compress_result_t *svd_mixed_canonicalize(const tensor_t *tensor,
                                               uint32_t left_axis,
                                               uint32_t right_axis,
                                               const svd_compress_config_t *config);

// ============================================================================
// TRUNCATION ANALYSIS
// ============================================================================

/**
 * @brief Determine optimal truncation rank for given error bound
 *
 * Analyzes singular values to find minimum k such that
 * ||A - A_k||_F <= max_error * ||A||_F (relative error) or
 * ||A - A_k||_F <= max_error (absolute error)
 *
 * @param singular_values Array of singular values (descending)
 * @param count Number of singular values
 * @param max_error Maximum allowed error
 * @param relative If true, use relative error
 * @param truncation_error Output: actual truncation error
 * @return Optimal truncation rank
 */
uint32_t svd_optimal_rank(const double *singular_values, uint32_t count,
                          double max_error, bool relative,
                          double *truncation_error);

/**
 * @brief Estimate entanglement entropy from singular values
 *
 * Computes von Neumann entropy: S = -sum(p_i * log(p_i))
 * where p_i = s_i^2 / sum(s_j^2)
 *
 * @param singular_values Array of singular values
 * @param count Number of singular values
 * @return Entanglement entropy
 */
double svd_entanglement_entropy(const double *singular_values, uint32_t count);

/**
 * @brief Compute Schmidt coefficients from singular values
 *
 * Normalizes singular values: p_i = s_i^2 / sum(s_j^2)
 *
 * @param singular_values Input singular values
 * @param count Number of values
 * @param schmidt_coeffs Output Schmidt coefficients (caller allocates)
 */
void svd_schmidt_coefficients(const double *singular_values, uint32_t count,
                               double *schmidt_coeffs);

/**
 * @brief Estimate required bond dimension for target fidelity
 *
 * Given current singular values, estimates bond dimension needed
 * to achieve target fidelity F = |<psi|psi_truncated>|^2
 *
 * @param singular_values Singular values
 * @param count Number of values
 * @param target_fidelity Target fidelity (0 < F <= 1)
 * @return Required bond dimension
 */
uint32_t svd_bond_for_fidelity(const double *singular_values, uint32_t count,
                                double target_fidelity);

// ============================================================================
// STATISTICS TRACKING
// ============================================================================

/**
 * @brief Initialize compression statistics
 *
 * @return Initialized statistics structure
 */
svd_compress_stats_t svd_compress_stats_init(void);

/**
 * @brief Update statistics with compression result
 *
 * @param stats Statistics to update
 * @param result Compression result
 */
void svd_compress_stats_update(svd_compress_stats_t *stats,
                                const svd_compress_result_t *result);

/**
 * @brief Reset statistics
 *
 * @param stats Statistics to reset
 */
void svd_compress_stats_reset(svd_compress_stats_t *stats);

/**
 * @brief Print statistics summary
 *
 * @param stats Statistics to print
 */
void svd_compress_stats_print(const svd_compress_stats_t *stats);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Get error string for SVD compression error
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *svd_compress_error_string(svd_compress_error_t error);

/**
 * @brief Absorb singular values into tensor
 *
 * Multiplies tensor along specified axis by singular values.
 * result[..., i, ...] = tensor[..., i, ...] * singular_values[i]
 *
 * @param tensor Tensor to modify (in-place)
 * @param singular_values Singular values
 * @param count Number of singular values
 * @param axis Axis to absorb along
 * @return SVD_COMPRESS_SUCCESS or error code
 */
svd_compress_error_t svd_absorb_singular_values(tensor_t *tensor,
                                                 const double *singular_values,
                                                 uint32_t count,
                                                 uint32_t axis);

/**
 * @brief Normalize singular values
 *
 * Scales singular values so sum of squares equals 1.
 *
 * @param singular_values Array to normalize (in-place)
 * @param count Number of values
 * @return Original norm
 */
double svd_normalize_singular_values(double *singular_values, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* SVD_COMPRESS_H */
