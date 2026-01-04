/**
 * @file tensor.h
 * @brief Core tensor data structures and operations for tensor network simulation
 *
 * Provides arbitrary-rank complex tensor operations optimized for quantum simulation.
 * All tensors use row-major storage with SIMD-aligned memory allocation.
 *
 * Memory layout for a rank-3 tensor T[i][j][k] with dimensions [d0, d1, d2]:
 *   linear_index = i * (d1 * d2) + j * d2 + k
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <complex.h>
#include <stdbool.h>

// ============================================================================
// GPU BACKEND FORWARD DECLARATIONS
// ============================================================================

/**
 * @brief Opaque GPU buffer handle
 *
 * Allows tensor data to reside on GPU for accelerated operations.
 * The actual implementation depends on the active backend (Metal, CUDA, etc.)
 */
typedef struct gpu_buffer gpu_buffer_t;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/** Maximum tensor rank supported */
#define TENSOR_MAX_RANK 16

/** Memory alignment for SIMD operations (64 bytes for AVX-512) */
#define TENSOR_ALIGNMENT 64

/** Default bond dimension for MPS/tensor networks */
#define TENSOR_DEFAULT_BOND_DIM 64

/** Maximum bond dimension before truncation warning */
#define TENSOR_MAX_BOND_DIM 4096

// ============================================================================
// ERROR CODES
// ============================================================================

/**
 * @brief Tensor operation error codes
 */
typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_ERROR_NULL_PTR = -1,
    TENSOR_ERROR_INVALID_RANK = -2,
    TENSOR_ERROR_INVALID_DIM = -3,
    TENSOR_ERROR_ALLOC_FAILED = -4,
    TENSOR_ERROR_DIM_MISMATCH = -5,
    TENSOR_ERROR_INDEX_OUT_OF_BOUNDS = -6,
    TENSOR_ERROR_CONTRACTION_FAILED = -7,
    TENSOR_ERROR_SVD_FAILED = -8,
    TENSOR_ERROR_NOT_IMPLEMENTED = -9,
    TENSOR_ERROR_INVALID_AXIS = -10,
    TENSOR_ERROR_NUMERICAL = -11,
    TENSOR_ERROR_GPU_UNAVAILABLE = -12,
    TENSOR_ERROR_GPU_SYNC_FAILED = -13
} tensor_error_t;

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/**
 * @brief Complex tensor with arbitrary rank
 *
 * Stores complex double data in row-major order with SIMD-aligned memory.
 * The tensor owns its data and is responsible for freeing it.
 *
 * GPU Synchronization Model:
 * - cpu_valid=true, gpu_valid=false: Data on CPU only (initial state)
 * - cpu_valid=false, gpu_valid=true: Data on GPU only (after GPU op)
 * - cpu_valid=true, gpu_valid=true: Both copies valid (after sync)
 * - cpu_valid=false, gpu_valid=false: Invalid state (error)
 *
 * Use tensor_sync_to_gpu() and tensor_sync_to_cpu() to synchronize.
 */
typedef struct {
    double complex *data;           /**< Complex amplitude data (SIMD-aligned) */
    uint32_t rank;                  /**< Number of dimensions (0 = scalar) */
    uint32_t dims[TENSOR_MAX_RANK]; /**< Size of each dimension */
    uint64_t strides[TENSOR_MAX_RANK]; /**< Stride for each dimension */
    uint64_t total_size;            /**< Total number of elements */
    bool owns_data;                 /**< Whether tensor owns its data buffer */

    /* GPU acceleration support */
    gpu_buffer_t *gpu_buffer;       /**< GPU buffer (NULL if not allocated) */
    bool cpu_valid;                 /**< CPU data is current */
    bool gpu_valid;                 /**< GPU data is current */
} tensor_t;

/**
 * @brief Index specification for tensor operations
 *
 * Used to specify index combinations for contraction, slicing, etc.
 */
typedef struct {
    uint32_t axis1;                 /**< First axis index */
    uint32_t axis2;                 /**< Second axis index */
} tensor_axis_pair_t;

/**
 * @brief Tensor slice specification
 *
 * Specifies a slice along one dimension.
 */
typedef struct {
    uint32_t axis;                  /**< Dimension to slice */
    uint32_t start;                 /**< Start index (inclusive) */
    uint32_t stop;                  /**< Stop index (exclusive) */
    uint32_t step;                  /**< Step size */
} tensor_slice_t;

/**
 * @brief SVD decomposition result
 *
 * For matrix A = U * S * V^H where:
 * - U is m x k orthonormal
 * - S is k diagonal (singular values)
 * - V^H is k x n orthonormal
 */
typedef struct {
    tensor_t *U;                    /**< Left singular vectors (m x k) */
    double *S;                      /**< Singular values (k) - real, non-negative */
    tensor_t *Vh;                   /**< Right singular vectors conjugate transpose (k x n) */
    uint32_t k;                     /**< Number of singular values kept */
    double truncation_error;        /**< Frobenius norm of discarded singular values */
} tensor_svd_result_t;

/**
 * @brief QR decomposition result
 *
 * For matrix A = Q * R where:
 * - Q is m x k orthonormal
 * - R is k x n upper triangular
 */
typedef struct {
    tensor_t *Q;                    /**< Orthonormal matrix */
    tensor_t *R;                    /**< Upper triangular matrix */
} tensor_qr_result_t;

/**
 * @brief Tensor contraction specification
 */
typedef struct {
    uint32_t num_contractions;      /**< Number of index pairs to contract */
    tensor_axis_pair_t pairs[TENSOR_MAX_RANK]; /**< Pairs of axes to contract */
} tensor_contraction_spec_t;

// ============================================================================
// CREATION AND DESTRUCTION
// ============================================================================

/**
 * @brief Create a new tensor with specified dimensions
 *
 * Allocates SIMD-aligned memory and initializes to zero.
 *
 * @param rank Number of dimensions
 * @param dims Array of dimension sizes
 * @return New tensor or NULL on failure
 */
tensor_t *tensor_create(uint32_t rank, const uint32_t *dims);

/**
 * @brief Create a tensor initialized with given data
 *
 * @param rank Number of dimensions
 * @param dims Array of dimension sizes
 * @param data Initial data (copied into tensor)
 * @return New tensor or NULL on failure
 */
tensor_t *tensor_create_with_data(uint32_t rank, const uint32_t *dims,
                                   const double complex *data);

/**
 * @brief Create a tensor as a view of existing data
 *
 * Does not copy data. Caller must ensure data outlives tensor.
 * The returned tensor will have owns_data = false.
 *
 * @param rank Number of dimensions
 * @param dims Array of dimension sizes
 * @param data Existing data buffer
 * @return New tensor view or NULL on failure
 */
tensor_t *tensor_create_view(uint32_t rank, const uint32_t *dims,
                              double complex *data);

/**
 * @brief Create a scalar tensor (rank 0)
 *
 * @param value Scalar value
 * @return New scalar tensor or NULL on failure
 */
tensor_t *tensor_create_scalar(double complex value);

/**
 * @brief Create a rank-1 tensor (vector)
 *
 * @param size Vector size
 * @return New vector tensor or NULL on failure
 */
tensor_t *tensor_create_vector(uint32_t size);

/**
 * @brief Create a rank-2 tensor (matrix)
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New matrix tensor or NULL on failure
 */
tensor_t *tensor_create_matrix(uint32_t rows, uint32_t cols);

/**
 * @brief Create an identity matrix tensor
 *
 * @param size Matrix dimension (creates size x size identity)
 * @return New identity matrix or NULL on failure
 */
tensor_t *tensor_create_identity(uint32_t size);

/**
 * @brief Create a copy of a tensor
 *
 * @param src Source tensor
 * @return New tensor copy or NULL on failure
 */
tensor_t *tensor_copy(const tensor_t *src);

/**
 * @brief Free a tensor and its data
 *
 * Safe to call with NULL.
 *
 * @param tensor Tensor to free
 */
void tensor_free(tensor_t *tensor);

/**
 * @brief Free SVD result structure
 *
 * @param svd SVD result to free
 */
void tensor_svd_free(tensor_svd_result_t *svd);

/**
 * @brief Free QR result structure
 *
 * @param qr QR result to free
 */
void tensor_qr_free(tensor_qr_result_t *qr);

// ============================================================================
// ELEMENT ACCESS
// ============================================================================

/**
 * @brief Get linear index from multi-dimensional indices
 *
 * @param tensor Tensor
 * @param indices Array of indices (length = rank)
 * @return Linear index into data array
 */
uint64_t tensor_get_linear_index(const tensor_t *tensor, const uint32_t *indices);

/**
 * @brief Get multi-dimensional indices from linear index
 *
 * @param tensor Tensor
 * @param linear_idx Linear index
 * @param indices Output array for indices (length = rank)
 */
void tensor_get_multi_index(const tensor_t *tensor, uint64_t linear_idx,
                            uint32_t *indices);

/**
 * @brief Get element at indices
 *
 * @param tensor Tensor
 * @param indices Array of indices
 * @return Element value
 */
double complex tensor_get(const tensor_t *tensor, const uint32_t *indices);

/**
 * @brief Set element at indices
 *
 * @param tensor Tensor
 * @param indices Array of indices
 * @param value Value to set
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_set(tensor_t *tensor, const uint32_t *indices,
                          double complex value);

/**
 * @brief Get element by linear index
 *
 * @param tensor Tensor
 * @param idx Linear index
 * @return Element value
 */
double complex tensor_get_linear(const tensor_t *tensor, uint64_t idx);

/**
 * @brief Set element by linear index
 *
 * @param tensor Tensor
 * @param idx Linear index
 * @param value Value to set
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_set_linear(tensor_t *tensor, uint64_t idx,
                                  double complex value);

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

/**
 * @brief Reshape tensor to new dimensions
 *
 * Total size must remain the same. Returns a new tensor (original unchanged).
 *
 * @param tensor Source tensor
 * @param new_rank New rank
 * @param new_dims New dimensions
 * @return Reshaped tensor or NULL on failure
 */
tensor_t *tensor_reshape(const tensor_t *tensor, uint32_t new_rank,
                         const uint32_t *new_dims);

/**
 * @brief Transpose tensor axes
 *
 * @param tensor Source tensor
 * @param perm Permutation of axes (length = rank)
 * @return Transposed tensor or NULL on failure
 */
tensor_t *tensor_transpose(const tensor_t *tensor, const uint32_t *perm);

/**
 * @brief Swap two axes of a tensor
 *
 * @param tensor Source tensor
 * @param axis1 First axis
 * @param axis2 Second axis
 * @return Tensor with swapped axes or NULL on failure
 */
tensor_t *tensor_swapaxes(const tensor_t *tensor, uint32_t axis1, uint32_t axis2);

/**
 * @brief Flatten tensor to 1D
 *
 * @param tensor Source tensor
 * @return Flattened tensor or NULL on failure
 */
tensor_t *tensor_flatten(const tensor_t *tensor);

/**
 * @brief Add a new dimension of size 1
 *
 * @param tensor Source tensor
 * @param axis Position for new axis
 * @return Expanded tensor or NULL on failure
 */
tensor_t *tensor_expand_dims(const tensor_t *tensor, uint32_t axis);

/**
 * @brief Remove dimensions of size 1
 *
 * @param tensor Source tensor
 * @return Squeezed tensor or NULL on failure
 */
tensor_t *tensor_squeeze(const tensor_t *tensor);

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

/**
 * @brief Add two tensors element-wise
 *
 * Tensors must have same shape.
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor or NULL on failure
 */
tensor_t *tensor_add(const tensor_t *a, const tensor_t *b);

/**
 * @brief Subtract two tensors element-wise
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor (a - b) or NULL on failure
 */
tensor_t *tensor_sub(const tensor_t *a, const tensor_t *b);

/**
 * @brief Multiply tensor by scalar
 *
 * @param tensor Source tensor
 * @param scalar Scalar multiplier
 * @return Scaled tensor or NULL on failure
 */
tensor_t *tensor_scale(const tensor_t *tensor, double complex scalar);

/**
 * @brief Multiply two tensors element-wise (Hadamard product)
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor or NULL on failure
 */
tensor_t *tensor_hadamard(const tensor_t *a, const tensor_t *b);

/**
 * @brief Complex conjugate of tensor
 *
 * @param tensor Source tensor
 * @return Conjugated tensor or NULL on failure
 */
tensor_t *tensor_conj(const tensor_t *tensor);

/**
 * @brief In-place scalar multiplication
 *
 * @param tensor Tensor to scale
 * @param scalar Scalar multiplier
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_scale_inplace(tensor_t *tensor, double complex scalar);

/**
 * @brief In-place addition (tensor += other)
 *
 * @param tensor Target tensor
 * @param other Tensor to add
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_add_inplace(tensor_t *tensor, const tensor_t *other);

// ============================================================================
// NORMS AND PROPERTIES
// ============================================================================

/**
 * @brief Frobenius norm (L2 norm of all elements)
 *
 * ||T||_F = sqrt(sum |t_i|^2)
 *
 * @param tensor Tensor
 * @return Frobenius norm
 */
double tensor_norm_frobenius(const tensor_t *tensor);

/**
 * @brief Maximum absolute value norm (infinity norm)
 *
 * @param tensor Tensor
 * @return Max absolute value
 */
double tensor_norm_max(const tensor_t *tensor);

/**
 * @brief Sum of all elements
 *
 * @param tensor Tensor
 * @return Sum
 */
double complex tensor_sum(const tensor_t *tensor);

/**
 * @brief Sum along an axis
 *
 * @param tensor Source tensor
 * @param axis Axis to sum along
 * @return Reduced tensor or NULL on failure
 */
tensor_t *tensor_sum_axis(const tensor_t *tensor, uint32_t axis);

/**
 * @brief Inner product (Frobenius inner product)
 *
 * <A, B> = sum(conj(a_i) * b_i)
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Inner product value
 */
double complex tensor_inner(const tensor_t *a, const tensor_t *b);

/**
 * @brief Check if tensors have same shape
 *
 * @param a First tensor
 * @param b Second tensor
 * @return true if shapes match
 */
bool tensor_shape_equal(const tensor_t *a, const tensor_t *b);

/**
 * @brief Check if tensor is approximately zero
 *
 * @param tensor Tensor
 * @param tol Tolerance
 * @return true if all elements below tolerance
 */
bool tensor_is_zero(const tensor_t *tensor, double tol);

/**
 * @brief Check if two tensors are approximately equal
 *
 * @param a First tensor
 * @param b Second tensor
 * @param tol Tolerance for element-wise comparison
 * @return true if all elements match within tolerance
 */
bool tensor_allclose(const tensor_t *a, const tensor_t *b, double tol);

// ============================================================================
// MATRIX OPERATIONS (for rank-2 tensors)
// ============================================================================

/**
 * @brief Matrix multiplication
 *
 * Computes C = A @ B for rank-2 tensors.
 *
 * @param a Left matrix (m x k)
 * @param b Right matrix (k x n)
 * @return Result matrix (m x n) or NULL on failure
 */
tensor_t *tensor_matmul(const tensor_t *a, const tensor_t *b);

/**
 * @brief Matrix-vector multiplication
 *
 * Computes y = A @ x for matrix A and vector x.
 *
 * @param mat Matrix (m x n)
 * @param vec Vector (n)
 * @return Result vector (m) or NULL on failure
 */
tensor_t *tensor_matvec(const tensor_t *mat, const tensor_t *vec);

/**
 * @brief Outer product of two vectors
 *
 * Computes M[i,j] = a[i] * b[j]
 *
 * @param a First vector
 * @param b Second vector
 * @return Outer product matrix or NULL on failure
 */
tensor_t *tensor_outer(const tensor_t *a, const tensor_t *b);

/**
 * @brief Tensor product (Kronecker product for matrices)
 *
 * For rank-2 tensors, computes standard Kronecker product.
 * For higher ranks, computes outer product and reshapes.
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Tensor product or NULL on failure
 */
tensor_t *tensor_kron(const tensor_t *a, const tensor_t *b);

/**
 * @brief Matrix trace
 *
 * @param mat Square matrix
 * @return Trace value
 */
double complex tensor_trace(const tensor_t *mat);

/**
 * @brief Hermitian conjugate (conjugate transpose)
 *
 * @param mat Matrix
 * @return Hermitian conjugate or NULL on failure
 */
tensor_t *tensor_dagger(const tensor_t *mat);

// ============================================================================
// DECOMPOSITIONS
// ============================================================================

/**
 * @brief Singular Value Decomposition
 *
 * Computes A = U * S * V^H with optional truncation.
 *
 * @param mat Matrix to decompose
 * @param max_rank Maximum number of singular values to keep (0 = all)
 * @param cutoff Discard singular values below this threshold
 * @return SVD result or NULL on failure
 */
tensor_svd_result_t *tensor_svd(const tensor_t *mat, uint32_t max_rank,
                                 double cutoff);

/**
 * @brief SVD with automatic rank selection based on error threshold
 *
 * Keeps singular values until truncation error exceeds threshold.
 *
 * @param mat Matrix to decompose
 * @param max_error Maximum allowed truncation error (Frobenius norm)
 * @return SVD result or NULL on failure
 */
tensor_svd_result_t *tensor_svd_truncate(const tensor_t *mat, double max_error);

/**
 * @brief QR decomposition
 *
 * Computes A = Q * R where Q is orthonormal.
 *
 * @param mat Matrix to decompose
 * @return QR result or NULL on failure
 */
tensor_qr_result_t *tensor_qr(const tensor_t *mat);

/**
 * @brief LQ decomposition
 *
 * Computes A = L * Q where Q is orthonormal.
 *
 * @param mat Matrix to decompose
 * @return QR result (Q in Q field, L in R field) or NULL on failure
 */
tensor_qr_result_t *tensor_lq(const tensor_t *mat);

// ============================================================================
// TENSOR CONTRACTION
// ============================================================================

/**
 * @brief Contract two tensors over specified axes
 *
 * Sums over specified index pairs between tensors A and B.
 * Example: contracting A[i,j,k] with B[k,l,m] over axis (2,0) gives C[i,j,l,m]
 *
 * @param a First tensor
 * @param b Second tensor
 * @param axes_a Axes of A to contract
 * @param axes_b Axes of B to contract
 * @param num_contract Number of axes to contract
 * @return Contracted tensor or NULL on failure
 */
tensor_t *tensor_contract(const tensor_t *a, const tensor_t *b,
                          const uint32_t *axes_a, const uint32_t *axes_b,
                          uint32_t num_contract);

/**
 * @brief Tensor dot product
 *
 * Contracts last axis of A with first axis of B.
 * For matrices, this is standard matrix multiplication.
 *
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor or NULL on failure
 */
tensor_t *tensor_tensordot(const tensor_t *a, const tensor_t *b);

/**
 * @brief Einstein summation (limited subset)
 *
 * Supports basic contractions specified by subscript string.
 * Example: "ij,jk->ik" for matrix multiplication
 *
 * @param a First tensor
 * @param b Second tensor (can be NULL for trace-like operations)
 * @param subscripts Einsum subscript string
 * @return Result tensor or NULL on failure
 */
tensor_t *tensor_einsum(const tensor_t *a, const tensor_t *b,
                        const char *subscripts);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Set all elements to zero
 *
 * @param tensor Tensor to zero
 */
void tensor_zero(tensor_t *tensor);

/**
 * @brief Set all elements to a constant
 *
 * @param tensor Tensor
 * @param value Value to fill
 */
void tensor_fill(tensor_t *tensor, double complex value);

/**
 * @brief Copy data from one tensor to another
 *
 * Tensors must have same total size.
 *
 * @param dst Destination tensor
 * @param src Source tensor
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_copy_data(tensor_t *dst, const tensor_t *src);

/**
 * @brief Print tensor shape information
 *
 * @param tensor Tensor
 * @param name Optional name to print (can be NULL)
 */
void tensor_print_shape(const tensor_t *tensor, const char *name);

/**
 * @brief Print tensor data (for debugging, small tensors only)
 *
 * @param tensor Tensor
 * @param name Optional name
 * @param max_elements Maximum elements to print (0 = all)
 */
void tensor_print_data(const tensor_t *tensor, const char *name,
                       uint32_t max_elements);

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *tensor_error_string(tensor_error_t error);

/**
 * @brief Calculate memory usage of tensor in bytes
 *
 * @param tensor Tensor
 * @return Memory usage in bytes
 */
size_t tensor_memory_usage(const tensor_t *tensor);

// ============================================================================
// RANDOM INITIALIZATION
// ============================================================================

/**
 * @brief Fill tensor with random complex values
 *
 * Fills with uniformly distributed values in [-1, 1] + i*[-1, 1].
 *
 * @param tensor Tensor to fill
 * @param seed Random seed (0 for time-based)
 */
void tensor_random_fill(tensor_t *tensor, uint64_t seed);

/**
 * @brief Create random unitary matrix
 *
 * Uses QR decomposition of random matrix.
 *
 * @param size Matrix dimension
 * @param seed Random seed (0 for time-based)
 * @return Random unitary matrix or NULL on failure
 */
tensor_t *tensor_random_unitary(uint32_t size, uint64_t seed);

// ============================================================================
// GPU ACCELERATION
// ============================================================================

/**
 * @brief Opaque GPU context handle
 *
 * Backend-specific context for GPU operations.
 * Use tensor_gpu_context_create() to initialize.
 */
typedef struct tensor_gpu_context tensor_gpu_context_t;

/**
 * @brief Create GPU context for tensor operations
 *
 * Initializes the GPU backend (Metal on macOS, CUDA on Linux/Windows).
 * Returns NULL if GPU is not available.
 *
 * @return GPU context or NULL if unavailable
 */
tensor_gpu_context_t *tensor_gpu_context_create(void);

/**
 * @brief Destroy GPU context
 *
 * Releases all GPU resources. Safe to call with NULL.
 *
 * @param ctx GPU context to destroy
 */
void tensor_gpu_context_destroy(tensor_gpu_context_t *ctx);

/**
 * @brief Check if GPU acceleration is available
 *
 * @return true if GPU backend is initialized and ready
 */
bool tensor_gpu_available(void);

/**
 * @brief Get the global GPU context (singleton)
 *
 * Returns the global GPU context, creating it if necessary.
 * This is the preferred way to get a GPU context for tensor operations.
 *
 * @return GPU context or NULL if unavailable
 */
tensor_gpu_context_t *tensor_gpu_get_context(void);

#ifdef __APPLE__
/**
 * @brief Get Metal context from GPU context
 *
 * Returns the underlying Metal context for direct Metal API calls.
 * Only available on Apple platforms.
 *
 * @param ctx GPU context
 * @return Metal context or NULL
 */
struct metal_compute_ctx *tensor_gpu_get_metal(tensor_gpu_context_t *ctx);
#endif

/**
 * @brief Allocate GPU buffer for tensor
 *
 * Creates a GPU buffer sized to hold the tensor data.
 * Does not copy data - use tensor_sync_to_gpu() for that.
 *
 * @param ctx GPU context
 * @param tensor Tensor to allocate GPU memory for
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_gpu_alloc(tensor_gpu_context_t *ctx, tensor_t *tensor);

/**
 * @brief Free GPU buffer for tensor
 *
 * Releases GPU memory. CPU data is unaffected.
 *
 * @param tensor Tensor to free GPU memory for
 */
void tensor_gpu_free(tensor_t *tensor);

/**
 * @brief Synchronize tensor data to GPU
 *
 * Copies CPU data to GPU buffer. Allocates GPU buffer if needed.
 * After this call: cpu_valid=true, gpu_valid=true
 *
 * @param ctx GPU context
 * @param tensor Tensor to synchronize
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_sync_to_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);

/**
 * @brief Synchronize tensor data from GPU to CPU
 *
 * Copies GPU data back to CPU buffer.
 * After this call: cpu_valid=true, gpu_valid=true
 *
 * @param tensor Tensor to synchronize
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_sync_to_cpu(tensor_t *tensor);

/**
 * @brief Ensure tensor data is valid on CPU
 *
 * Synchronizes from GPU if needed. No-op if CPU data is already valid.
 *
 * @param tensor Tensor to ensure CPU validity
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_ensure_cpu(tensor_t *tensor);

/**
 * @brief Ensure tensor data is valid on GPU
 *
 * Synchronizes to GPU if needed. No-op if GPU data is already valid.
 *
 * @param ctx GPU context
 * @param tensor Tensor to ensure GPU validity
 * @return TENSOR_SUCCESS or error code
 */
tensor_error_t tensor_ensure_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);

/**
 * @brief Mark CPU data as invalid
 *
 * Call after GPU operation modifies data. Next CPU access will sync.
 *
 * @param tensor Tensor to invalidate
 */
void tensor_invalidate_cpu(tensor_t *tensor);

/**
 * @brief Mark GPU data as invalid
 *
 * Call after CPU operation modifies data. Next GPU access will sync.
 *
 * @param tensor Tensor to invalidate
 */
void tensor_invalidate_gpu(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
