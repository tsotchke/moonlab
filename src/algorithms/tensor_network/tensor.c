/**
 * @file tensor.c
 * @brief Core tensor operations implementation
 *
 * Full production implementation of arbitrary-rank complex tensor operations.
 * Uses SIMD-aligned memory and optimized loops where possible.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Cross-platform BLAS/LAPACK support
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAS_BLAS 1
#define HAS_LAPACK 1
#include "../../optimization/gpu_metal.h"
#define HAS_METAL 1
#elif defined(QSIM_HAS_OPENBLAS) || defined(__linux__)
// OpenBLAS provides both CBLAS and LAPACKE interfaces
#include <cblas.h>
#include <lapacke.h>
#define HAS_BLAS 1
#define HAS_LAPACK 1
#define HAS_METAL 0
// Type compatibility for LAPACKE
typedef lapack_complex_double lapack_complex_t;
#else
#define HAS_BLAS 0
#define HAS_LAPACK 0
#define HAS_METAL 0
#endif

#ifdef HAS_CUDA
#include "../../optimization/gpu/backends/gpu_cuda.h"
#endif

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/**
 * @brief Allocate aligned memory
 */
static void *aligned_alloc_internal(size_t size, size_t alignment) {
    if (size == 0) return NULL;

    void *ptr = NULL;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
#endif
    return ptr;
}

/**
 * @brief Free aligned memory
 */
static void aligned_free_internal(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Compute strides from dimensions (row-major)
 */
static void compute_strides(uint32_t rank, const uint32_t *dims, uint64_t *strides) {
    if (rank == 0) return;

    strides[rank - 1] = 1;
    for (int i = (int)rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

/**
 * @brief Compute total size from dimensions
 */
static uint64_t compute_total_size(uint32_t rank, const uint32_t *dims) {
    if (rank == 0) return 1;  // Scalar

    uint64_t size = 1;
    for (uint32_t i = 0; i < rank; i++) {
        size *= dims[i];
    }
    return size;
}

// ============================================================================
// CREATION AND DESTRUCTION
// ============================================================================

tensor_t *tensor_create(uint32_t rank, const uint32_t *dims) {
    if (rank > TENSOR_MAX_RANK) return NULL;
    if (rank > 0 && dims == NULL) return NULL;

    tensor_t *tensor = (tensor_t *)calloc(1, sizeof(tensor_t));
    if (!tensor) return NULL;

    tensor->rank = rank;
    tensor->total_size = compute_total_size(rank, dims);

    for (uint32_t i = 0; i < rank; i++) {
        if (dims[i] == 0) {
            free(tensor);
            return NULL;
        }
        tensor->dims[i] = dims[i];
    }

    compute_strides(rank, tensor->dims, tensor->strides);

    // Allocate aligned data
    size_t data_size = tensor->total_size * sizeof(double complex);
    if (data_size == 0) data_size = sizeof(double complex);  // At least one element

    tensor->data = (double complex *)aligned_alloc_internal(data_size, TENSOR_ALIGNMENT);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }

    // Initialize to zero
    memset(tensor->data, 0, data_size);
    tensor->owns_data = true;

    // Initialize GPU fields
    tensor->gpu_buffer = NULL;
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;

    return tensor;
}

tensor_t *tensor_create_with_data(uint32_t rank, const uint32_t *dims,
                                   const double complex *data) {
    tensor_t *tensor = tensor_create(rank, dims);
    if (!tensor || !data) return tensor;

    memcpy(tensor->data, data, tensor->total_size * sizeof(double complex));
    return tensor;
}

tensor_t *tensor_create_view(uint32_t rank, const uint32_t *dims,
                              double complex *data) {
    if (rank > TENSOR_MAX_RANK) return NULL;
    if (rank > 0 && dims == NULL) return NULL;
    if (!data) return NULL;

    tensor_t *tensor = (tensor_t *)calloc(1, sizeof(tensor_t));
    if (!tensor) return NULL;

    tensor->rank = rank;
    tensor->total_size = compute_total_size(rank, dims);

    for (uint32_t i = 0; i < rank; i++) {
        tensor->dims[i] = dims[i];
    }

    compute_strides(rank, tensor->dims, tensor->strides);
    tensor->data = data;
    tensor->owns_data = false;

    // Initialize GPU fields (views don't own GPU buffers either)
    tensor->gpu_buffer = NULL;
    tensor->cpu_valid = true;
    tensor->gpu_valid = false;

    return tensor;
}

tensor_t *tensor_create_scalar(double complex value) {
    tensor_t *tensor = tensor_create(0, NULL);
    if (tensor) {
        tensor->data[0] = value;
    }
    return tensor;
}

tensor_t *tensor_create_vector(uint32_t size) {
    uint32_t dims[1] = {size};
    return tensor_create(1, dims);
}

tensor_t *tensor_create_matrix(uint32_t rows, uint32_t cols) {
    uint32_t dims[2] = {rows, cols};
    return tensor_create(2, dims);
}

tensor_t *tensor_create_identity(uint32_t size) {
    tensor_t *mat = tensor_create_matrix(size, size);
    if (!mat) return NULL;

    for (uint32_t i = 0; i < size; i++) {
        mat->data[i * size + i] = 1.0;
    }
    return mat;
}

tensor_t *tensor_copy(const tensor_t *src) {
    if (!src) return NULL;

    tensor_t *dst = tensor_create(src->rank, src->dims);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->total_size * sizeof(double complex));
    return dst;
}

void tensor_free(tensor_t *tensor) {
    if (!tensor) return;

    // Free GPU buffer if allocated
    tensor_gpu_free(tensor);

    if (tensor->owns_data && tensor->data) {
        aligned_free_internal(tensor->data);
    }
    free(tensor);
}

void tensor_svd_free(tensor_svd_result_t *svd) {
    if (!svd) return;

    tensor_free(svd->U);
    tensor_free(svd->Vh);
    if (svd->S) free(svd->S);
    free(svd);
}

void tensor_qr_free(tensor_qr_result_t *qr) {
    if (!qr) return;

    tensor_free(qr->Q);
    tensor_free(qr->R);
    free(qr);
}

// ============================================================================
// ELEMENT ACCESS
// ============================================================================

uint64_t tensor_get_linear_index(const tensor_t *tensor, const uint32_t *indices) {
    if (!tensor || !indices) return 0;

    uint64_t idx = 0;
    for (uint32_t i = 0; i < tensor->rank; i++) {
        idx += indices[i] * tensor->strides[i];
    }
    return idx;
}

void tensor_get_multi_index(const tensor_t *tensor, uint64_t linear_idx,
                            uint32_t *indices) {
    if (!tensor || !indices) return;

    uint64_t remaining = linear_idx;
    for (uint32_t i = 0; i < tensor->rank; i++) {
        indices[i] = remaining / tensor->strides[i];
        remaining = remaining % tensor->strides[i];
    }
}

double complex tensor_get(const tensor_t *tensor, const uint32_t *indices) {
    if (!tensor || !indices) return 0.0;
    return tensor->data[tensor_get_linear_index(tensor, indices)];
}

tensor_error_t tensor_set(tensor_t *tensor, const uint32_t *indices,
                          double complex value) {
    if (!tensor || !indices) return TENSOR_ERROR_NULL_PTR;

    // Bounds check
    for (uint32_t i = 0; i < tensor->rank; i++) {
        if (indices[i] >= tensor->dims[i]) {
            return TENSOR_ERROR_INDEX_OUT_OF_BOUNDS;
        }
    }

    tensor->data[tensor_get_linear_index(tensor, indices)] = value;
    return TENSOR_SUCCESS;
}

double complex tensor_get_linear(const tensor_t *tensor, uint64_t idx) {
    if (!tensor || idx >= tensor->total_size) return 0.0;
    return tensor->data[idx];
}

tensor_error_t tensor_set_linear(tensor_t *tensor, uint64_t idx,
                                  double complex value) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;
    if (idx >= tensor->total_size) return TENSOR_ERROR_INDEX_OUT_OF_BOUNDS;

    tensor->data[idx] = value;
    return TENSOR_SUCCESS;
}

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

tensor_t *tensor_reshape(const tensor_t *tensor, uint32_t new_rank,
                         const uint32_t *new_dims) {
    if (!tensor || !new_dims) return NULL;

    // Verify total size unchanged
    uint64_t new_size = compute_total_size(new_rank, new_dims);
    if (new_size != tensor->total_size) return NULL;

    return tensor_create_with_data(new_rank, new_dims, tensor->data);
}

tensor_t *tensor_transpose(const tensor_t *tensor, const uint32_t *perm) {
    if (!tensor || !perm) return NULL;
    if (tensor->rank == 0) return tensor_copy(tensor);

    // Validate permutation
    bool used[TENSOR_MAX_RANK] = {false};
    for (uint32_t i = 0; i < tensor->rank; i++) {
        if (perm[i] >= tensor->rank || used[perm[i]]) return NULL;
        used[perm[i]] = true;
    }

    // Compute new dimensions
    uint32_t new_dims[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < tensor->rank; i++) {
        new_dims[i] = tensor->dims[perm[i]];
    }

    tensor_t *result = tensor_create(tensor->rank, new_dims);
    if (!result) return NULL;

    // Compute old strides for each permuted axis
    uint64_t old_strides[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < tensor->rank; i++) {
        old_strides[i] = tensor->strides[perm[i]];
    }

    // Copy data with transposition
    uint32_t indices[TENSOR_MAX_RANK];
    for (uint64_t i = 0; i < result->total_size; i++) {
        tensor_get_multi_index(result, i, indices);

        // Compute source index
        uint64_t src_idx = 0;
        for (uint32_t j = 0; j < tensor->rank; j++) {
            src_idx += indices[j] * old_strides[j];
        }

        result->data[i] = tensor->data[src_idx];
    }

    return result;
}

tensor_t *tensor_swapaxes(const tensor_t *tensor, uint32_t axis1, uint32_t axis2) {
    if (!tensor) return NULL;
    if (axis1 >= tensor->rank || axis2 >= tensor->rank) return NULL;
    if (axis1 == axis2) return tensor_copy(tensor);

    uint32_t perm[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < tensor->rank; i++) {
        perm[i] = i;
    }
    perm[axis1] = axis2;
    perm[axis2] = axis1;

    return tensor_transpose(tensor, perm);
}

tensor_t *tensor_flatten(const tensor_t *tensor) {
    if (!tensor) return NULL;

    uint32_t dims[1] = {(uint32_t)tensor->total_size};
    return tensor_create_with_data(1, dims, tensor->data);
}

tensor_t *tensor_expand_dims(const tensor_t *tensor, uint32_t axis) {
    if (!tensor) return NULL;
    if (axis > tensor->rank) return NULL;
    if (tensor->rank >= TENSOR_MAX_RANK) return NULL;

    uint32_t new_dims[TENSOR_MAX_RANK];
    uint32_t j = 0;
    for (uint32_t i = 0; i <= tensor->rank; i++) {
        if (i == axis) {
            new_dims[i] = 1;
        } else {
            new_dims[i] = tensor->dims[j++];
        }
    }

    return tensor_create_with_data(tensor->rank + 1, new_dims, tensor->data);
}

tensor_t *tensor_squeeze(const tensor_t *tensor) {
    if (!tensor) return NULL;

    uint32_t new_dims[TENSOR_MAX_RANK];
    uint32_t new_rank = 0;

    for (uint32_t i = 0; i < tensor->rank; i++) {
        if (tensor->dims[i] != 1) {
            new_dims[new_rank++] = tensor->dims[i];
        }
    }

    if (new_rank == 0) {
        // Result is scalar
        return tensor_create_with_data(0, NULL, tensor->data);
    }

    return tensor_create_with_data(new_rank, new_dims, tensor->data);
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

tensor_t *tensor_add(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (!tensor_shape_equal(a, b)) return NULL;

    tensor_t *result = tensor_copy(a);
    if (!result) return NULL;

#if HAS_ACCELERATE
    // Use vDSP for optimized complex addition
    vDSP_zvaddD((DSPDoubleSplitComplex *)a->data, 1,
                (DSPDoubleSplitComplex *)b->data, 1,
                (DSPDoubleSplitComplex *)result->data, 1,
                (vDSP_Length)a->total_size);
#else
    for (uint64_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
#endif

    return result;
}

tensor_t *tensor_sub(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (!tensor_shape_equal(a, b)) return NULL;

    tensor_t *result = tensor_copy(a);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

tensor_t *tensor_scale(const tensor_t *tensor, double complex scalar) {
    if (!tensor) return NULL;

    tensor_t *result = tensor_copy(tensor);
    if (!result) return NULL;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        result->data[i] *= scalar;
    }

    return result;
}

tensor_t *tensor_hadamard(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (!tensor_shape_equal(a, b)) return NULL;

    tensor_t *result = tensor_create(a->rank, a->dims);
    if (!result) return NULL;

    for (uint64_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

tensor_t *tensor_conj(const tensor_t *tensor) {
    if (!tensor) return NULL;

    tensor_t *result = tensor_create(tensor->rank, tensor->dims);
    if (!result) return NULL;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        result->data[i] = conj(tensor->data[i]);
    }

    return result;
}

tensor_error_t tensor_scale_inplace(tensor_t *tensor, double complex scalar) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] *= scalar;
    }

    return TENSOR_SUCCESS;
}

tensor_error_t tensor_add_inplace(tensor_t *tensor, const tensor_t *other) {
    if (!tensor || !other) return TENSOR_ERROR_NULL_PTR;
    if (!tensor_shape_equal(tensor, other)) return TENSOR_ERROR_DIM_MISMATCH;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] += other->data[i];
    }

    return TENSOR_SUCCESS;
}

// ============================================================================
// NORMS AND PROPERTIES
// ============================================================================

double tensor_norm_frobenius(const tensor_t *tensor) {
    if (!tensor) return 0.0;

    // OPTIMIZED: Avoid cabs() which computes sqrt unnecessarily
    // |z|² = real² + imag², no sqrt needed until final result
    // Use Kahan summation for numerical stability on large tensors
    double sum = 0.0;
    double c = 0.0;  // Compensation for lost low-order bits

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        double r = creal(tensor->data[i]);
        double im = cimag(tensor->data[i]);
        double term = r * r + im * im;
        // Kahan summation
        double y = term - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sqrt(sum);
}

double tensor_norm_max(const tensor_t *tensor) {
    if (!tensor) return 0.0;

    // OPTIMIZED: Compare squared magnitudes, only sqrt at end
    double max_sq = 0.0;
    for (uint64_t i = 0; i < tensor->total_size; i++) {
        double r = creal(tensor->data[i]);
        double im = cimag(tensor->data[i]);
        double mag_sq = r * r + im * im;
        if (mag_sq > max_sq) max_sq = mag_sq;
    }

    return sqrt(max_sq);
}

double complex tensor_sum(const tensor_t *tensor) {
    if (!tensor) return 0.0;

    double complex s = 0.0;
    for (uint64_t i = 0; i < tensor->total_size; i++) {
        s += tensor->data[i];
    }

    return s;
}

tensor_t *tensor_sum_axis(const tensor_t *tensor, uint32_t axis) {
    if (!tensor) return NULL;
    if (axis >= tensor->rank) return NULL;

    // New shape without the summed axis
    uint32_t new_rank = tensor->rank - 1;
    uint32_t new_dims[TENSOR_MAX_RANK];

    if (new_rank == 0) {
        // Result is scalar
        double complex s = tensor_sum(tensor);
        return tensor_create_scalar(s);
    }

    uint32_t j = 0;
    for (uint32_t i = 0; i < tensor->rank; i++) {
        if (i != axis) {
            new_dims[j++] = tensor->dims[i];
        }
    }

    tensor_t *result = tensor_create(new_rank, new_dims);
    if (!result) return NULL;

    // Sum over axis
    uint32_t axis_size = tensor->dims[axis];
    uint64_t axis_stride = tensor->strides[axis];

    uint32_t out_indices[TENSOR_MAX_RANK];
    uint32_t in_indices[TENSOR_MAX_RANK];

    for (uint64_t out_idx = 0; out_idx < result->total_size; out_idx++) {
        tensor_get_multi_index(result, out_idx, out_indices);

        // Map output indices to input indices
        j = 0;
        for (uint32_t i = 0; i < tensor->rank; i++) {
            if (i == axis) {
                in_indices[i] = 0;  // Will iterate
            } else {
                in_indices[i] = out_indices[j++];
            }
        }

        // Sum along axis
        double complex s = 0.0;
        uint64_t base_idx = tensor_get_linear_index(tensor, in_indices);
        for (uint32_t k = 0; k < axis_size; k++) {
            s += tensor->data[base_idx + k * axis_stride];
        }

        result->data[out_idx] = s;
    }

    return result;
}

double complex tensor_inner(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return 0.0;
    if (a->total_size != b->total_size) return 0.0;

    double complex result = 0.0;
    for (uint64_t i = 0; i < a->total_size; i++) {
        result += conj(a->data[i]) * b->data[i];
    }

    return result;
}

bool tensor_shape_equal(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return false;
    if (a->rank != b->rank) return false;

    for (uint32_t i = 0; i < a->rank; i++) {
        if (a->dims[i] != b->dims[i]) return false;
    }

    return true;
}

bool tensor_is_zero(const tensor_t *tensor, double tol) {
    if (!tensor) return true;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        if (cabs(tensor->data[i]) > tol) return false;
    }

    return true;
}

bool tensor_allclose(const tensor_t *a, const tensor_t *b, double tol) {
    if (!a || !b) return false;
    if (!tensor_shape_equal(a, b)) return false;

    for (uint64_t i = 0; i < a->total_size; i++) {
        if (cabs(a->data[i] - b->data[i]) > tol) return false;
    }

    return true;
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

tensor_t *tensor_matmul(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (a->rank != 2 || b->rank != 2) return NULL;
    if (a->dims[1] != b->dims[0]) return NULL;

    uint32_t m = a->dims[0];
    uint32_t k = a->dims[1];
    uint32_t n = b->dims[1];

    tensor_t *result = tensor_create_matrix(m, n);
    if (!result) return NULL;

#if HAS_ACCELERATE
    // Use BLAS zgemm for optimized matrix multiplication
    double complex alpha = 1.0;
    double complex beta = 0.0;

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                &alpha,
                a->data, k,
                b->data, n,
                &beta,
                result->data, n);
#else
    // Standard matrix multiplication
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double complex sum = 0.0;
            for (uint32_t l = 0; l < k; l++) {
                sum += a->data[i * k + l] * b->data[l * n + j];
            }
            result->data[i * n + j] = sum;
        }
    }
#endif

    return result;
}

tensor_t *tensor_matvec(const tensor_t *mat, const tensor_t *vec) {
    if (!mat || !vec) return NULL;
    if (mat->rank != 2 || vec->rank != 1) return NULL;
    if (mat->dims[1] != vec->dims[0]) return NULL;

    uint32_t m = mat->dims[0];
    uint32_t n = mat->dims[1];

    tensor_t *result = tensor_create_vector(m);
    if (!result) return NULL;

#if HAS_ACCELERATE
    double complex alpha = 1.0;
    double complex beta = 0.0;

    cblas_zgemv(CblasRowMajor, CblasNoTrans,
                m, n,
                &alpha,
                mat->data, n,
                vec->data, 1,
                &beta,
                result->data, 1);
#else
    for (uint32_t i = 0; i < m; i++) {
        double complex sum = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            sum += mat->data[i * n + j] * vec->data[j];
        }
        result->data[i] = sum;
    }
#endif

    return result;
}

tensor_t *tensor_outer(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (a->rank != 1 || b->rank != 1) return NULL;

    uint32_t m = a->dims[0];
    uint32_t n = b->dims[0];

    tensor_t *result = tensor_create_matrix(m, n);
    if (!result) return NULL;

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            result->data[i * n + j] = a->data[i] * b->data[j];
        }
    }

    return result;
}

tensor_t *tensor_kron(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;

    // For matrices, compute standard Kronecker product
    if (a->rank == 2 && b->rank == 2) {
        uint32_t m1 = a->dims[0], n1 = a->dims[1];
        uint32_t m2 = b->dims[0], n2 = b->dims[1];

        uint32_t dims[2] = {m1 * m2, n1 * n2};
        tensor_t *result = tensor_create(2, dims);
        if (!result) return NULL;

        for (uint32_t i1 = 0; i1 < m1; i1++) {
            for (uint32_t j1 = 0; j1 < n1; j1++) {
                double complex a_val = a->data[i1 * n1 + j1];
                for (uint32_t i2 = 0; i2 < m2; i2++) {
                    for (uint32_t j2 = 0; j2 < n2; j2++) {
                        uint32_t row = i1 * m2 + i2;
                        uint32_t col = j1 * n2 + j2;
                        result->data[row * dims[1] + col] = a_val * b->data[i2 * n2 + j2];
                    }
                }
            }
        }

        return result;
    }

    // General case: outer product with reshape
    uint32_t new_rank = a->rank + b->rank;
    if (new_rank > TENSOR_MAX_RANK) return NULL;

    uint32_t new_dims[TENSOR_MAX_RANK];
    for (uint32_t i = 0; i < a->rank; i++) {
        new_dims[i] = a->dims[i];
    }
    for (uint32_t i = 0; i < b->rank; i++) {
        new_dims[a->rank + i] = b->dims[i];
    }

    tensor_t *result = tensor_create(new_rank, new_dims);
    if (!result) return NULL;

    // Compute outer product
    for (uint64_t i = 0; i < a->total_size; i++) {
        for (uint64_t j = 0; j < b->total_size; j++) {
            result->data[i * b->total_size + j] = a->data[i] * b->data[j];
        }
    }

    return result;
}

double complex tensor_trace(const tensor_t *mat) {
    if (!mat || mat->rank != 2) return 0.0;
    if (mat->dims[0] != mat->dims[1]) return 0.0;

    double complex tr = 0.0;
    uint32_t n = mat->dims[0];

    for (uint32_t i = 0; i < n; i++) {
        tr += mat->data[i * n + i];
    }

    return tr;
}

tensor_t *tensor_dagger(const tensor_t *mat) {
    if (!mat || mat->rank != 2) return NULL;

    uint32_t dims[2] = {mat->dims[1], mat->dims[0]};
    tensor_t *result = tensor_create(2, dims);
    if (!result) return NULL;

    for (uint32_t i = 0; i < mat->dims[0]; i++) {
        for (uint32_t j = 0; j < mat->dims[1]; j++) {
            result->data[j * dims[1] + i] = conj(mat->data[i * mat->dims[1] + j]);
        }
    }

    return result;
}

// ============================================================================
// DECOMPOSITIONS
// ============================================================================

tensor_svd_result_t *tensor_svd(const tensor_t *mat, uint32_t max_rank,
                                 double cutoff) {
    if (!mat || mat->rank != 2) return NULL;

    uint32_t m = mat->dims[0];
    uint32_t n = mat->dims[1];
    uint32_t min_mn = (m < n) ? m : n;

    tensor_svd_result_t *result = (tensor_svd_result_t *)calloc(1, sizeof(tensor_svd_result_t));
    if (!result) return NULL;

#if HAS_LAPACK
    // Use LAPACK zgesvd - works with both Accelerate (Apple) and OpenBLAS (Linux)
    int info;

    // Allocate working arrays
    double *s = (double *)malloc(min_mn * sizeof(double));
    double complex *u_data = (double complex *)aligned_alloc_internal(
        m * m * sizeof(double complex), TENSOR_ALIGNMENT);
    double complex *vt_data = (double complex *)aligned_alloc_internal(
        n * n * sizeof(double complex), TENSOR_ALIGNMENT);
    double *superb = (double *)malloc((min_mn > 1 ? min_mn - 1 : 1) * sizeof(double));

    // Copy input (LAPACK modifies it)
    double complex *a_copy = (double complex *)aligned_alloc_internal(
        m * n * sizeof(double complex), TENSOR_ALIGNMENT);

    if (!s || !u_data || !vt_data || !superb || !a_copy) {
        free(s);
        aligned_free_internal(u_data);
        aligned_free_internal(vt_data);
        free(superb);
        aligned_free_internal(a_copy);
        free(result);
        return NULL;
    }

    // Copy data (convert to LAPACK format - column major)
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            a_copy[j * m + i] = mat->data[i * n + j];
        }
    }

#ifdef __APPLE__
    // Apple Accelerate interface
    __CLPK_integer M = m, N = n, lda = m, ldu = m, ldvt = n, lwork = -1;
    __CLPK_doublecomplex work_query;
    double *rwork = (double *)malloc(5 * min_mn * sizeof(double));
    if (!rwork) {
        free(s);
        aligned_free_internal(u_data);
        aligned_free_internal(vt_data);
        free(superb);
        aligned_free_internal(a_copy);
        free(result);
        return NULL;
    }

    // Query workspace
    zgesvd_("A", "A", &M, &N, (__CLPK_doublecomplex *)a_copy, &lda, s,
            (__CLPK_doublecomplex *)u_data, &ldu,
            (__CLPK_doublecomplex *)vt_data, &ldvt,
            &work_query, &lwork, rwork, &info);

    lwork = (int)work_query.r + 1;
    __CLPK_doublecomplex *work = malloc(lwork * sizeof(__CLPK_doublecomplex));
    if (!work) {
        free(s);
        aligned_free_internal(u_data);
        aligned_free_internal(vt_data);
        free(superb);
        free(rwork);
        aligned_free_internal(a_copy);
        free(result);
        return NULL;
    }

    // Perform SVD
    zgesvd_("A", "A", &M, &N, (__CLPK_doublecomplex *)a_copy, &lda, s,
            (__CLPK_doublecomplex *)u_data, &ldu,
            (__CLPK_doublecomplex *)vt_data, &ldvt,
            work, &lwork, rwork, &info);

    free(work);
    free(rwork);
#else
    // LAPACKE interface (OpenBLAS, Intel MKL, etc.)
    info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n,
                          (lapack_complex_double *)a_copy, m, s,
                          (lapack_complex_double *)u_data, m,
                          (lapack_complex_double *)vt_data, n,
                          superb);
#endif

    aligned_free_internal(a_copy);
    free(superb);

    if (info != 0) {
        free(s);
        aligned_free_internal(u_data);
        aligned_free_internal(vt_data);
        free(result);
        return NULL;
    }

    // Validate SVD results - check for NaN/Inf/negative singular values
    // LAPACK can return corrupted values for ill-conditioned matrices
    for (uint32_t i = 0; i < min_mn; i++) {
        if (isnan(s[i]) || isinf(s[i]) || s[i] < 0.0) {
            free(s);
            aligned_free_internal(u_data);
            aligned_free_internal(vt_data);
            free(result);
            return NULL;
        }
    }

    // Determine truncation
    uint32_t k = min_mn;
    double truncation_error_sq = 0.0;

    // Apply cutoff
    if (cutoff > 0.0) {
        while (k > 0 && s[k-1] < cutoff) {
            truncation_error_sq += s[k-1] * s[k-1];
            k--;
        }
    }

    // Apply max rank
    if (max_rank > 0 && k > max_rank) {
        for (uint32_t i = max_rank; i < k; i++) {
            truncation_error_sq += s[i] * s[i];
        }
        k = max_rank;
    }

    if (k == 0) k = 1;  // Keep at least one singular value

    // Allocate output tensors
    result->U = tensor_create_matrix(m, k);
    result->Vh = tensor_create_matrix(k, n);
    result->S = (double *)malloc(k * sizeof(double));
    result->k = k;
    result->truncation_error = sqrt(truncation_error_sq);

    if (!result->U || !result->Vh || !result->S) {
        free(s);
        aligned_free_internal(u_data);
        aligned_free_internal(vt_data);
        tensor_svd_free(result);
        return NULL;
    }

    // Copy singular values
    memcpy(result->S, s, k * sizeof(double));

    // Copy U (column-major to row-major, first k columns)
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            result->U->data[i * k + j] = u_data[j * m + i];
        }
    }

    // Copy Vh (column-major to row-major, first k rows)
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < n; j++) {
            result->Vh->data[i * n + j] = vt_data[j * n + i];
        }
    }

    free(s);
    aligned_free_internal(u_data);
    aligned_free_internal(vt_data);

#else
    // =========================================================================
    // Golub-Kahan bidiagonalization SVD (no LAPACK fallback)
    // This is a proper SVD algorithm.
    // Uses Householder reflections to reduce to bidiagonal form,
    // then QR iteration to find singular values.
    // =========================================================================

    // One-sided Jacobi SVD algorithm
    // Computes A = U * S * V^H using Jacobi rotations on A^H * A
    // Numerically stable and accurate for all matrix sizes

    // Allocate full-size working arrays
    double complex *U_work = (double complex *)calloc(m * min_mn, sizeof(double complex));
    double complex *V_work = (double complex *)calloc(n * min_mn, sizeof(double complex));
    double *S_work = (double *)calloc(min_mn, sizeof(double));

    if (!U_work || !V_work || !S_work) {
        free(U_work);
        free(V_work);
        free(S_work);
        free(result);
        return NULL;
    }

    // Copy A into U_work (m x min_mn) for processing
    // We'll compute A = U * S * V^H by working on the columns
    double complex *A_work = (double complex *)calloc(m * n, sizeof(double complex));
    if (!A_work) {
        free(U_work);
        free(V_work);
        free(S_work);
        free(result);
        return NULL;
    }
    memcpy(A_work, mat->data, m * n * sizeof(double complex));

    // Initialize V to identity
    for (uint32_t i = 0; i < min_mn; i++) {
        for (uint32_t j = 0; j < n; j++) {
            V_work[j * min_mn + i] = (i == j) ? 1.0 : 0.0;
        }
    }

    // One-sided Jacobi: repeatedly apply rotations to orthogonalize columns
    const int max_sweeps = 30;
    const double tol = 1e-14;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double max_off = 0.0;

        // Sweep through all column pairs
        for (uint32_t i = 0; i < min_mn; i++) {
            for (uint32_t j = i + 1; j < min_mn; j++) {
                // Compute inner products for columns i and j
                // a = ||A_i||^2, b = ||A_j||^2, c = A_i^H * A_j
                double a = 0.0, b = 0.0;
                double complex c = 0.0;

                for (uint32_t row = 0; row < m; row++) {
                    double complex ai = A_work[row * n + i];
                    double complex aj = A_work[row * n + j];
                    a += creal(ai) * creal(ai) + cimag(ai) * cimag(ai);
                    b += creal(aj) * creal(aj) + cimag(aj) * cimag(aj);
                    c += conj(ai) * aj;
                }

                double off = cabs(c);
                if (off > max_off) max_off = off;

                // Skip if columns are already orthogonal
                if (off < tol * sqrt(a * b + 1e-300)) continue;

                // Compute Jacobi rotation to zero out c
                // We need to find cos(theta), sin(theta) such that
                // (c*cos + ...) = 0, with proper complex phase handling

                double zeta = (b - a) / (2.0 * cabs(c) + 1e-300);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double cs = 1.0 / sqrt(1.0 + t * t);
                double sn = cs * t;

                // Complex rotation factor
                double complex phase = (cabs(c) > 1e-300) ? conj(c) / cabs(c) : 1.0;
                double complex sn_c = sn * phase;

                // Apply rotation to columns of A_work
                for (uint32_t row = 0; row < m; row++) {
                    double complex ai = A_work[row * n + i];
                    double complex aj = A_work[row * n + j];
                    A_work[row * n + i] = cs * ai + sn_c * aj;
                    A_work[row * n + j] = -conj(sn_c) * ai + cs * aj;
                }

                // Apply rotation to V_work
                for (uint32_t row = 0; row < n; row++) {
                    double complex vi = V_work[row * min_mn + i];
                    double complex vj = V_work[row * min_mn + j];
                    V_work[row * min_mn + i] = cs * vi + sn_c * vj;
                    V_work[row * min_mn + j] = -conj(sn_c) * vi + cs * vj;
                }
            }
        }

        // Check convergence
        if (max_off < tol) break;
    }

    // Extract singular values and U from orthogonalized A_work
    for (uint32_t i = 0; i < min_mn; i++) {
        // Compute column norm = singular value
        double sigma = 0.0;
        for (uint32_t row = 0; row < m; row++) {
            double complex val = A_work[row * n + i];
            sigma += creal(val) * creal(val) + cimag(val) * cimag(val);
        }
        S_work[i] = sqrt(sigma);

        // Normalize to get U column
        if (S_work[i] > 1e-15) {
            for (uint32_t row = 0; row < m; row++) {
                U_work[row * min_mn + i] = A_work[row * n + i] / S_work[i];
            }
        } else {
            for (uint32_t row = 0; row < m; row++) {
                U_work[row * min_mn + i] = (row == i) ? 1.0 : 0.0;
            }
        }
    }

    free(A_work);

    // Sort singular values in descending order
    for (uint32_t i = 0; i < min_mn - 1; i++) {
        uint32_t max_idx = i;
        for (uint32_t j = i + 1; j < min_mn; j++) {
            if (S_work[j] > S_work[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            // Swap singular values
            double temp_s = S_work[i];
            S_work[i] = S_work[max_idx];
            S_work[max_idx] = temp_s;

            // Swap U columns
            for (uint32_t row = 0; row < m; row++) {
                double complex temp_u = U_work[row * min_mn + i];
                U_work[row * min_mn + i] = U_work[row * min_mn + max_idx];
                U_work[row * min_mn + max_idx] = temp_u;
            }

            // Swap V columns
            for (uint32_t row = 0; row < n; row++) {
                double complex temp_v = V_work[row * min_mn + i];
                V_work[row * min_mn + i] = V_work[row * min_mn + max_idx];
                V_work[row * min_mn + max_idx] = temp_v;
            }
        }
    }

    // Determine truncation
    uint32_t k = min_mn;
    double truncation_error_sq = 0.0;

    if (cutoff > 0.0) {
        while (k > 0 && S_work[k-1] < cutoff) {
            truncation_error_sq += S_work[k-1] * S_work[k-1];
            k--;
        }
    }

    if (max_rank > 0 && k > max_rank) {
        for (uint32_t i = max_rank; i < k; i++) {
            truncation_error_sq += S_work[i] * S_work[i];
        }
        k = max_rank;
    }

    if (k == 0) k = 1;

    // Allocate output
    result->U = tensor_create_matrix(m, k);
    result->Vh = tensor_create_matrix(k, n);
    result->S = (double *)malloc(k * sizeof(double));
    result->k = k;
    result->truncation_error = sqrt(truncation_error_sq);

    if (!result->U || !result->Vh || !result->S) {
        free(U_work);
        free(V_work);
        free(S_work);
        tensor_svd_free(result);
        return NULL;
    }

    // Copy results
    memcpy(result->S, S_work, k * sizeof(double));

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            result->U->data[i * k + j] = U_work[i * min_mn + j];
        }
    }

    // V^H = conjugate transpose of V
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < n; j++) {
            result->Vh->data[i * n + j] = conj(V_work[j * min_mn + i]);
        }
    }

    free(U_work);
    free(V_work);
    free(S_work);
#endif

    return result;
}

tensor_svd_result_t *tensor_svd_truncate(const tensor_t *mat, double max_error) {
    // First do full SVD
    tensor_svd_result_t *full = tensor_svd(mat, 0, 0.0);
    if (!full) return NULL;

    // Find cutoff rank
    double error_sq = 0.0;
    double max_error_sq = max_error * max_error;
    uint32_t k = full->k;

    for (int i = (int)full->k - 1; i >= 0; i--) {
        double new_error_sq = error_sq + full->S[i] * full->S[i];
        if (new_error_sq > max_error_sq) break;
        error_sq = new_error_sq;
        k = i;
    }

    if (k == 0) k = 1;
    if (k == full->k) return full;

    // Create truncated result
    tensor_svd_result_t *result = (tensor_svd_result_t *)calloc(1, sizeof(tensor_svd_result_t));
    if (!result) {
        tensor_svd_free(full);
        return NULL;
    }

    uint32_t m = full->U->dims[0];
    uint32_t n = full->Vh->dims[1];

    result->U = tensor_create_matrix(m, k);
    result->Vh = tensor_create_matrix(k, n);
    result->S = (double *)malloc(k * sizeof(double));
    result->k = k;
    result->truncation_error = sqrt(error_sq);

    if (!result->U || !result->Vh || !result->S) {
        tensor_svd_free(full);
        tensor_svd_free(result);
        return NULL;
    }

    // Copy truncated data
    memcpy(result->S, full->S, k * sizeof(double));

    for (uint32_t i = 0; i < m; i++) {
        memcpy(&result->U->data[i * k], &full->U->data[i * full->k],
               k * sizeof(double complex));
    }

    for (uint32_t i = 0; i < k; i++) {
        memcpy(&result->Vh->data[i * n], &full->Vh->data[i * n],
               n * sizeof(double complex));
    }

    tensor_svd_free(full);
    return result;
}

tensor_qr_result_t *tensor_qr(const tensor_t *mat) {
    if (!mat || mat->rank != 2) return NULL;

    uint32_t m = mat->dims[0];
    uint32_t n = mat->dims[1];
    uint32_t k = (m < n) ? m : n;

    tensor_qr_result_t *result = (tensor_qr_result_t *)calloc(1, sizeof(tensor_qr_result_t));
    if (!result) return NULL;

#if HAS_ACCELERATE
    // Use LAPACK zgeqrf + zungqr
    __CLPK_integer M = m;
    __CLPK_integer N = n;
    __CLPK_integer K = k;
    __CLPK_integer lda = n;  // Row major, but we'll transpose
    __CLPK_integer info;
    __CLPK_integer lwork = -1;

    // Convert to column-major for LAPACK
    __CLPK_doublecomplex *a_data = (__CLPK_doublecomplex *)aligned_alloc_internal(
        m * n * sizeof(__CLPK_doublecomplex), TENSOR_ALIGNMENT);
    __CLPK_doublecomplex *tau = (__CLPK_doublecomplex *)malloc(k * sizeof(__CLPK_doublecomplex));

    if (!a_data || !tau) {
        aligned_free_internal(a_data);
        free(tau);
        free(result);
        return NULL;
    }

    // Copy to column-major
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double complex val = mat->data[i * n + j];
            a_data[j * m + i].r = creal(val);
            a_data[j * m + i].i = cimag(val);
        }
    }

    // Query workspace
    __CLPK_doublecomplex work_query;
    zgeqrf_(&M, &N, a_data, &M, tau, &work_query, &lwork, &info);

    lwork = (int)work_query.r + 1;
    __CLPK_doublecomplex *work = (__CLPK_doublecomplex *)malloc(lwork * sizeof(__CLPK_doublecomplex));
    if (!work) {
        aligned_free_internal(a_data);
        free(tau);
        free(result);
        return NULL;
    }

    // Compute QR
    zgeqrf_(&M, &N, a_data, &M, tau, work, &lwork, &info);

    if (info != 0) {
        free(work);
        aligned_free_internal(a_data);
        free(tau);
        free(result);
        return NULL;
    }

    // Extract R (upper triangular part)
    result->R = tensor_create_matrix(k, n);
    if (!result->R) {
        free(work);
        aligned_free_internal(a_data);
        free(tau);
        free(result);
        return NULL;
    }

    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < n; j++) {
            if (j >= i) {
                result->R->data[i * n + j] = a_data[j * m + i].r + I * a_data[j * m + i].i;
            } else {
                result->R->data[i * n + j] = 0.0;
            }
        }
    }

    // Generate Q
    lwork = -1;
    zungqr_(&M, &K, &K, a_data, &M, tau, &work_query, &lwork, &info);
    lwork = (int)work_query.r + 1;

    __CLPK_doublecomplex *work2 = (__CLPK_doublecomplex *)realloc(work, lwork * sizeof(__CLPK_doublecomplex));
    if (!work2) {
        free(work);
        aligned_free_internal(a_data);
        free(tau);
        tensor_qr_free(result);
        return NULL;
    }
    work = work2;

    zungqr_(&M, &K, &K, a_data, &M, tau, work, &lwork, &info);

    result->Q = tensor_create_matrix(m, k);
    if (!result->Q) {
        free(work);
        aligned_free_internal(a_data);
        free(tau);
        tensor_qr_free(result);
        return NULL;
    }

    // Copy Q (column-major to row-major)
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            result->Q->data[i * k + j] = a_data[j * m + i].r + I * a_data[j * m + i].i;
        }
    }

    free(work);
    aligned_free_internal(a_data);
    free(tau);

#else
    // Modified Gram-Schmidt QR
    result->Q = tensor_create_matrix(m, k);
    result->R = tensor_create_matrix(k, n);

    if (!result->Q || !result->R) {
        tensor_qr_free(result);
        return NULL;
    }

    // Copy matrix columns for processing
    tensor_t *A = tensor_copy(mat);
    if (!A) {
        tensor_qr_free(result);
        return NULL;
    }

    for (uint32_t j = 0; j < k; j++) {
        // Copy column j to Q
        for (uint32_t i = 0; i < m; i++) {
            result->Q->data[i * k + j] = A->data[i * n + j];
        }

        // Orthogonalize against previous columns
        for (uint32_t i = 0; i < j; i++) {
            // R[i,j] = Q[:,i]^H * A[:,j]
            double complex dot = 0.0;
            for (uint32_t l = 0; l < m; l++) {
                dot += conj(result->Q->data[l * k + i]) * result->Q->data[l * k + j];
            }
            result->R->data[i * n + j] = dot;

            // Q[:,j] -= R[i,j] * Q[:,i]
            for (uint32_t l = 0; l < m; l++) {
                result->Q->data[l * k + j] -= dot * result->Q->data[l * k + i];
            }
        }

        // Normalize
        double norm = 0.0;
        for (uint32_t i = 0; i < m; i++) {
            norm += cabs(result->Q->data[i * k + j]) * cabs(result->Q->data[i * k + j]);
        }
        norm = sqrt(norm);

        result->R->data[j * n + j] = norm;

        // Use 1e-10 threshold to maintain orthogonality
        // At 1e-15, accumulated numerical errors can make Q severely non-orthogonal
        if (norm > 1e-10) {
            for (uint32_t i = 0; i < m; i++) {
                result->Q->data[i * k + j] /= norm;
            }
        } else {
            // Column is nearly linearly dependent - zero it out
            // This prevents non-orthogonal columns from corrupting later operations
            for (uint32_t i = 0; i < m; i++) {
                result->Q->data[i * k + j] = 0.0;
            }
        }

        // Fill remaining R entries
        for (uint32_t jj = j + 1; jj < n; jj++) {
            double complex dot = 0.0;
            for (uint32_t i = 0; i < m; i++) {
                dot += conj(result->Q->data[i * k + j]) * A->data[i * n + jj];
            }
            result->R->data[j * n + jj] = dot;
        }
    }

    tensor_free(A);
#endif

    return result;
}

tensor_qr_result_t *tensor_lq(const tensor_t *mat) {
    if (!mat || mat->rank != 2) return NULL;

    // LQ of A is related to QR of A^H: A = LQ means A^H = Q^H L^H
    // So we compute QR of A^H and then transpose

    tensor_t *Ah = tensor_dagger(mat);
    if (!Ah) return NULL;

    tensor_qr_result_t *qr = tensor_qr(Ah);
    tensor_free(Ah);

    if (!qr) return NULL;

    tensor_qr_result_t *result = (tensor_qr_result_t *)calloc(1, sizeof(tensor_qr_result_t));
    if (!result) {
        tensor_qr_free(qr);
        return NULL;
    }

    // Q from LQ is (Q from QR of A^H)^H
    result->Q = tensor_dagger(qr->Q);
    // L from LQ is (R from QR of A^H)^H
    result->R = tensor_dagger(qr->R);

    tensor_qr_free(qr);

    if (!result->Q || !result->R) {
        tensor_qr_free(result);
        return NULL;
    }

    return result;
}

// ============================================================================
// TENSOR CONTRACTION
// ============================================================================

tensor_t *tensor_contract(const tensor_t *a, const tensor_t *b,
                          const uint32_t *axes_a, const uint32_t *axes_b,
                          uint32_t num_contract) {
    if (!a || !b) return NULL;
    if (num_contract > 0 && (!axes_a || !axes_b)) return NULL;

    // Validate contraction axes
    for (uint32_t i = 0; i < num_contract; i++) {
        if (axes_a[i] >= a->rank || axes_b[i] >= b->rank) return NULL;
        if (a->dims[axes_a[i]] != b->dims[axes_b[i]]) return NULL;
    }

    // Compute result dimensions
    uint32_t result_rank = a->rank + b->rank - 2 * num_contract;
    if (result_rank > TENSOR_MAX_RANK) return NULL;

    uint32_t result_dims[TENSOR_MAX_RANK];
    uint32_t r_idx = 0;

    // Mark contracted axes
    bool a_contracted[TENSOR_MAX_RANK] = {false};
    bool b_contracted[TENSOR_MAX_RANK] = {false};

    for (uint32_t i = 0; i < num_contract; i++) {
        a_contracted[axes_a[i]] = true;
        b_contracted[axes_b[i]] = true;
    }

    // Result dims from A (non-contracted)
    for (uint32_t i = 0; i < a->rank; i++) {
        if (!a_contracted[i]) {
            result_dims[r_idx++] = a->dims[i];
        }
    }

    // Result dims from B (non-contracted)
    for (uint32_t i = 0; i < b->rank; i++) {
        if (!b_contracted[i]) {
            result_dims[r_idx++] = b->dims[i];
        }
    }

    // Handle scalar result (full contraction)
    // OPTIMIZED: O(contract_size) instead of O(a_size × b_size)
    if (result_rank == 0) {
        // For scalar result, all dimensions must be contracted
        // Compute contraction size and precompute strides
        uint64_t contract_size = 1;
        uint64_t a_strides[TENSOR_MAX_RANK], b_strides[TENSOR_MAX_RANK];

        for (uint32_t i = 0; i < num_contract; i++) {
            contract_size *= a->dims[axes_a[i]];
            a_strides[i] = a->strides[axes_a[i]];
            b_strides[i] = b->strides[axes_b[i]];
        }

        // Use Kahan summation for large contractions to reduce rounding error
        double complex sum = 0.0;
        double complex c = 0.0;  // Compensation for lost low-order bits
        const bool use_kahan = (contract_size > 10000);

        // Iterate over contracted indices only
        uint32_t contract_indices[TENSOR_MAX_RANK] = {0};
        for (uint64_t idx = 0; idx < contract_size; idx++) {
            // Compute linear indices using precomputed strides
            uint64_t a_lin = 0, b_lin = 0;
            for (uint32_t i = 0; i < num_contract; i++) {
                a_lin += contract_indices[i] * a_strides[i];
                b_lin += contract_indices[i] * b_strides[i];
            }

            double complex term = a->data[a_lin] * b->data[b_lin];

            if (use_kahan) {
                // Kahan summation
                double complex y = term - c;
                double complex t = sum + y;
                c = (t - sum) - y;
                sum = t;
            } else {
                sum += term;
            }

            // Increment contracted indices (odometer style)
            for (int i = num_contract - 1; i >= 0; i--) {
                contract_indices[i]++;
                if (contract_indices[i] < a->dims[axes_a[i]]) break;
                contract_indices[i] = 0;
            }
        }

        return tensor_create_scalar(sum);
    }

    tensor_t *result = tensor_create(result_rank, result_dims);
    if (!result) return NULL;

    // OPTIMIZED general contraction algorithm
    // Precompute strides to avoid function calls in inner loop

    // Compute contraction dimension size and precompute contracted strides
    uint64_t contract_size = 1;
    uint64_t a_contract_strides[TENSOR_MAX_RANK];
    uint64_t b_contract_strides[TENSOR_MAX_RANK];

    for (uint32_t i = 0; i < num_contract; i++) {
        contract_size *= a->dims[axes_a[i]];
        a_contract_strides[i] = a->strides[axes_a[i]];
        b_contract_strides[i] = b->strides[axes_b[i]];
    }

    // Precompute strides for free (non-contracted) dimensions
    uint64_t a_free_strides[TENSOR_MAX_RANK], b_free_strides[TENSOR_MAX_RANK];
    uint32_t a_free_count = 0, b_free_count = 0;

    for (uint32_t i = 0; i < a->rank; i++) {
        if (!a_contracted[i]) {
            a_free_strides[a_free_count++] = a->strides[i];
        }
    }
    for (uint32_t i = 0; i < b->rank; i++) {
        if (!b_contracted[i]) {
            b_free_strides[b_free_count++] = b->strides[i];
        }
    }

    // Use Kahan summation for large contractions
    const bool use_kahan = (contract_size > 10000);

    uint32_t out_indices[TENSOR_MAX_RANK];

    for (uint64_t out_idx = 0; out_idx < result->total_size; out_idx++) {
        tensor_get_multi_index(result, out_idx, out_indices);

        // Compute base linear indices for free dimensions (using precomputed strides)
        uint64_t a_base = 0, b_base = 0;
        for (uint32_t i = 0; i < a_free_count; i++) {
            a_base += out_indices[i] * a_free_strides[i];
        }
        for (uint32_t i = 0; i < b_free_count; i++) {
            b_base += out_indices[a_free_count + i] * b_free_strides[i];
        }

        // Sum over contracted indices
        double complex sum = 0.0;
        double complex c = 0.0;  // Kahan compensation

        uint32_t contract_indices[TENSOR_MAX_RANK] = {0};
        for (uint64_t ci = 0; ci < contract_size; ci++) {
            // Compute offset from contracted indices using precomputed strides
            uint64_t a_offset = 0, b_offset = 0;
            for (uint32_t i = 0; i < num_contract; i++) {
                a_offset += contract_indices[i] * a_contract_strides[i];
                b_offset += contract_indices[i] * b_contract_strides[i];
            }

            double complex term = a->data[a_base + a_offset] * b->data[b_base + b_offset];

            if (use_kahan) {
                double complex y = term - c;
                double complex t = sum + y;
                c = (t - sum) - y;
                sum = t;
            } else {
                sum += term;
            }

            // Increment contracted indices (odometer style)
            for (int i = num_contract - 1; i >= 0; i--) {
                contract_indices[i]++;
                if (contract_indices[i] < a->dims[axes_a[i]]) break;
                contract_indices[i] = 0;
            }
        }

        result->data[out_idx] = sum;
    }

    return result;
}

tensor_t *tensor_tensordot(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) return NULL;
    if (a->rank == 0 || b->rank == 0) {
        // Scalar multiplication
        return tensor_scale(a->rank == 0 ? b : a,
                           a->rank == 0 ? a->data[0] : b->data[0]);
    }

    // Contract last axis of a with first axis of b
    uint32_t axes_a[1] = {a->rank - 1};
    uint32_t axes_b[1] = {0};

    return tensor_contract(a, b, axes_a, axes_b, 1);
}

tensor_t *tensor_einsum(const tensor_t *a, const tensor_t *b,
                        const char *subscripts) {
    if (!a || !subscripts) return NULL;

    // Parse einsum string
    // Format: "ij,jk->ik" or "ij->" (for trace)

    char left[64] = {0};
    char right[64] = {0};
    char output[64] = {0};

    const char *arrow = strstr(subscripts, "->");
    if (!arrow) return NULL;

    // Find comma for two-operand case
    const char *comma = strchr(subscripts, ',');

    if (comma && comma < arrow) {
        // Two operands
        strncpy(left, subscripts, comma - subscripts);
        strncpy(right, comma + 1, arrow - comma - 1);
    } else {
        // Single operand
        strncpy(left, subscripts, arrow - subscripts);
    }
    strcpy(output, arrow + 2);

    // For simple cases, dispatch to existing functions
    if (strcmp(subscripts, "ij,jk->ik") == 0 && b) {
        return tensor_matmul(a, b);
    }
    if (strcmp(subscripts, "ii->") == 0 && !b) {
        return tensor_create_scalar(tensor_trace(a));
    }

    // General einsum implementation with full index parsing

    // Index analysis: determine dimension of each label (a-z supported)
    uint32_t label_dims[26] = {0};      // Dimension for each label 'a'-'z'
    bool label_in_left[26] = {false};   // Label appears in left operand
    bool label_in_right[26] = {false};  // Label appears in right operand
    bool label_in_output[26] = {false}; // Label appears in output

    // Map positions to labels
    int left_labels[16] = {-1};   // Label index for each position in left
    int right_labels[16] = {-1};  // Label index for each position in right
    int output_labels[16] = {-1}; // Label index for each position in output

    uint32_t left_rank = 0, right_rank = 0, output_rank = 0;

    // Parse left operand indices
    for (const char *p = left; *p && left_rank < 16; p++) {
        if (*p >= 'a' && *p <= 'z') {
            int label = *p - 'a';
            left_labels[left_rank] = label;
            label_in_left[label] = true;
            if (label_dims[label] == 0) {
                if (left_rank < a->rank) {
                    label_dims[label] = a->dims[left_rank];
                }
            }
            left_rank++;
        }
    }

    // Parse right operand indices (if present)
    if (b && right[0]) {
        for (const char *p = right; *p && right_rank < 16; p++) {
            if (*p >= 'a' && *p <= 'z') {
                int label = *p - 'a';
                right_labels[right_rank] = label;
                label_in_right[label] = true;
                if (label_dims[label] == 0) {
                    if (right_rank < b->rank) {
                        label_dims[label] = b->dims[right_rank];
                    }
                }
                right_rank++;
            }
        }
    }

    // Parse output indices
    for (const char *p = output; *p && output_rank < 16; p++) {
        if (*p >= 'a' && *p <= 'z') {
            int label = *p - 'a';
            output_labels[output_rank] = label;
            label_in_output[label] = true;
            output_rank++;
        }
    }

    // Build output dimensions
    uint32_t out_dims[16];
    for (uint32_t i = 0; i < output_rank; i++) {
        out_dims[i] = label_dims[output_labels[i]];
    }

    // Create output tensor (scalar if output_rank == 0)
    tensor_t *result;
    if (output_rank == 0) {
        uint32_t scalar_dim = 1;
        result = tensor_create(1, &scalar_dim);
    } else {
        result = tensor_create(output_rank, out_dims);
    }
    if (!result) return NULL;
    tensor_zero(result);

    // Identify contracted indices (in operands but not in output)
    int contracted_labels[26];
    uint32_t num_contracted = 0;
    for (int i = 0; i < 26; i++) {
        if ((label_in_left[i] || label_in_right[i]) && !label_in_output[i]) {
            contracted_labels[num_contracted++] = i;
        }
    }

    // Compute total sizes
    uint64_t output_size = (output_rank == 0) ? 1 : result->total_size;
    uint64_t contract_size = 1;
    for (uint32_t i = 0; i < num_contracted; i++) {
        contract_size *= label_dims[contracted_labels[i]];
    }

    // Main einsum loop: iterate over output indices, then contracted indices
    for (uint64_t out_flat = 0; out_flat < output_size; out_flat++) {
        // Decode output index
        uint32_t label_values[26] = {0};
        uint64_t temp = out_flat;
        for (int i = (int)output_rank - 1; i >= 0; i--) {
            uint32_t dim = label_dims[output_labels[i]];
            label_values[output_labels[i]] = temp % dim;
            temp /= dim;
        }

        // Sum over contracted indices
        double complex sum = 0.0;

        for (uint64_t c_flat = 0; c_flat < contract_size; c_flat++) {
            // Decode contracted indices
            temp = c_flat;
            for (int i = (int)num_contracted - 1; i >= 0; i--) {
                uint32_t dim = label_dims[contracted_labels[i]];
                label_values[contracted_labels[i]] = temp % dim;
                temp /= dim;
            }

            // Compute index into tensor a
            uint64_t a_idx = 0;
            uint64_t a_stride = 1;
            for (int i = (int)left_rank - 1; i >= 0; i--) {
                a_idx += label_values[left_labels[i]] * a_stride;
                a_stride *= a->dims[i];
            }

            double complex a_val = a->data[a_idx];

            double complex b_val = 1.0;
            if (b && right[0]) {
                // Compute index into tensor b
                uint64_t b_idx = 0;
                uint64_t b_stride = 1;
                for (int i = (int)right_rank - 1; i >= 0; i--) {
                    b_idx += label_values[right_labels[i]] * b_stride;
                    b_stride *= b->dims[i];
                }
                b_val = b->data[b_idx];
            }

            sum += a_val * b_val;
        }

        result->data[out_flat] = sum;
    }

    return result;
}

// ============================================================================
// UTILITIES
// ============================================================================

void tensor_zero(tensor_t *tensor) {
    if (!tensor || !tensor->data) return;
    memset(tensor->data, 0, tensor->total_size * sizeof(double complex));
}

void tensor_fill(tensor_t *tensor, double complex value) {
    if (!tensor || !tensor->data) return;

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] = value;
    }
}

tensor_error_t tensor_copy_data(tensor_t *dst, const tensor_t *src) {
    if (!dst || !src) return TENSOR_ERROR_NULL_PTR;
    if (dst->total_size != src->total_size) return TENSOR_ERROR_DIM_MISMATCH;

    memcpy(dst->data, src->data, src->total_size * sizeof(double complex));
    return TENSOR_SUCCESS;
}

void tensor_print_shape(const tensor_t *tensor, const char *name) {
    if (!tensor) {
        printf("%s: NULL\n", name ? name : "tensor");
        return;
    }

    printf("%s: shape=(", name ? name : "tensor");
    for (uint32_t i = 0; i < tensor->rank; i++) {
        printf("%u", tensor->dims[i]);
        if (i < tensor->rank - 1) printf(", ");
    }
    printf("), total_size=%lu\n", (unsigned long)tensor->total_size);
}

void tensor_print_data(const tensor_t *tensor, const char *name,
                       uint32_t max_elements) {
    if (!tensor) {
        printf("%s: NULL\n", name ? name : "tensor");
        return;
    }

    tensor_print_shape(tensor, name);

    uint64_t count = tensor->total_size;
    if (max_elements > 0 && count > max_elements) {
        count = max_elements;
    }

    printf("data: [");
    for (uint64_t i = 0; i < count; i++) {
        double r = creal(tensor->data[i]);
        double im = cimag(tensor->data[i]);

        if (fabs(im) < 1e-15) {
            printf("%.6f", r);
        } else if (fabs(r) < 1e-15) {
            printf("%.6fi", im);
        } else {
            printf("%.6f%+.6fi", r, im);
        }

        if (i < count - 1) printf(", ");
    }
    if (count < tensor->total_size) {
        printf(", ... (%lu more)", (unsigned long)(tensor->total_size - count));
    }
    printf("]\n");
}

const char *tensor_error_string(tensor_error_t error) {
    switch (error) {
        case TENSOR_SUCCESS: return "Success";
        case TENSOR_ERROR_NULL_PTR: return "Null pointer";
        case TENSOR_ERROR_INVALID_RANK: return "Invalid rank";
        case TENSOR_ERROR_INVALID_DIM: return "Invalid dimension";
        case TENSOR_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case TENSOR_ERROR_DIM_MISMATCH: return "Dimension mismatch";
        case TENSOR_ERROR_INDEX_OUT_OF_BOUNDS: return "Index out of bounds";
        case TENSOR_ERROR_CONTRACTION_FAILED: return "Contraction failed";
        case TENSOR_ERROR_SVD_FAILED: return "SVD failed";
        case TENSOR_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case TENSOR_ERROR_INVALID_AXIS: return "Invalid axis";
        case TENSOR_ERROR_NUMERICAL: return "Numerical error";
        default: return "Unknown error";
    }
}

size_t tensor_memory_usage(const tensor_t *tensor) {
    if (!tensor) return 0;

    size_t usage = sizeof(tensor_t);
    if (tensor->owns_data) {
        usage += tensor->total_size * sizeof(double complex);
    }

    return usage;
}

// ============================================================================
// RANDOM INITIALIZATION
// ============================================================================

void tensor_random_fill(tensor_t *tensor, uint64_t seed) {
    if (!tensor) return;

    if (seed == 0) {
        seed = (uint64_t)time(NULL);
    }
    srand((unsigned int)seed);

    for (uint64_t i = 0; i < tensor->total_size; i++) {
        double r = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        double im = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        tensor->data[i] = r + I * im;
    }
}

tensor_t *tensor_random_unitary(uint32_t size, uint64_t seed) {
    // Generate random matrix
    tensor_t *mat = tensor_create_matrix(size, size);
    if (!mat) return NULL;

    tensor_random_fill(mat, seed);

    // QR decomposition gives orthonormal Q
    tensor_qr_result_t *qr = tensor_qr(mat);
    tensor_free(mat);

    if (!qr) return NULL;

    tensor_t *Q = qr->Q;
    qr->Q = NULL;  // Prevent freeing
    tensor_qr_free(qr);

    // Make it exactly unitary (handle phase ambiguity)
    // Multiply each column by phase to make diagonal of R positive
    // This is already handled by most QR implementations

    return Q;
}

// ============================================================================
// GPU ACCELERATION
// ============================================================================

/**
 * @brief GPU context wrapper for backend abstraction
 */
struct tensor_gpu_context {
#if HAS_METAL
    metal_compute_ctx_t *metal_ctx;
#endif
#ifdef HAS_CUDA
    cuda_compute_ctx_t *cuda_ctx;
#endif
    int backend;  // 0=none, 1=metal, 2=cuda
};

// Global GPU context (singleton for simplicity)
static tensor_gpu_context_t *g_gpu_ctx = NULL;

tensor_gpu_context_t *tensor_gpu_context_create(void) {
    if (g_gpu_ctx) {
        return g_gpu_ctx;  // Return existing context
    }

    tensor_gpu_context_t *ctx = (tensor_gpu_context_t *)calloc(1, sizeof(tensor_gpu_context_t));
    if (!ctx) return NULL;

    ctx->backend = 0;

#if HAS_METAL
    ctx->metal_ctx = metal_compute_init();
    if (ctx->metal_ctx) {
        ctx->backend = 1;
        g_gpu_ctx = ctx;
        return ctx;
    }
#endif

#ifdef HAS_CUDA
    ctx->cuda_ctx = cuda_compute_init();
    if (ctx->cuda_ctx) {
        ctx->backend = 2;
        g_gpu_ctx = ctx;
        return ctx;
    }
#endif

    // No GPU backend available
    free(ctx);
    return NULL;
}

void tensor_gpu_context_destroy(tensor_gpu_context_t *ctx) {
    if (!ctx) return;

#if HAS_METAL
    if (ctx->metal_ctx) {
        metal_compute_free(ctx->metal_ctx);
    }
#endif

#ifdef HAS_CUDA
    if (ctx->cuda_ctx) {
        cuda_compute_cleanup(ctx->cuda_ctx);
    }
#endif

    if (g_gpu_ctx == ctx) {
        g_gpu_ctx = NULL;
    }
    free(ctx);
}

bool tensor_gpu_available(void) {
    if (g_gpu_ctx && g_gpu_ctx->backend > 0) {
        return true;
    }

    // Try to create context if not exists
    tensor_gpu_context_t *ctx = tensor_gpu_context_create();
    return (ctx != NULL && ctx->backend > 0);
}

tensor_gpu_context_t *tensor_gpu_get_context(void) {
    if (!g_gpu_ctx) {
        tensor_gpu_context_create();
    }
    return g_gpu_ctx;
}

#if HAS_METAL
metal_compute_ctx_t *tensor_gpu_get_metal(tensor_gpu_context_t *ctx) {
    if (!ctx) return NULL;
    if (ctx->backend != 1) return NULL;
    return ctx->metal_ctx;
}
#endif

tensor_error_t tensor_gpu_alloc(tensor_gpu_context_t *ctx, tensor_t *tensor) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;
    if (!ctx || ctx->backend == 0) return TENSOR_ERROR_GPU_UNAVAILABLE;

    // Already allocated
    if (tensor->gpu_buffer) return TENSOR_SUCCESS;

#if HAS_METAL
    if (ctx->backend == 1) {
        // Metal uses float2 (8 bytes per complex) - not double complex (16 bytes)
        size_t size = tensor->total_size * sizeof(float) * 2;
        tensor->gpu_buffer = (gpu_buffer_t *)metal_buffer_create(ctx->metal_ctx, size);
        if (!tensor->gpu_buffer) return TENSOR_ERROR_ALLOC_FAILED;
        return TENSOR_SUCCESS;
    }
#endif

#ifdef HAS_CUDA
    if (ctx->backend == 2) {
        // CUDA uses double complex (16 bytes per complex)
        size_t size = tensor->total_size * sizeof(double complex);
        tensor->gpu_buffer = (gpu_buffer_t *)cuda_buffer_create(ctx->cuda_ctx, size);
        if (!tensor->gpu_buffer) return TENSOR_ERROR_ALLOC_FAILED;
        return TENSOR_SUCCESS;
    }
#endif

    return TENSOR_ERROR_GPU_UNAVAILABLE;
}

void tensor_gpu_free(tensor_t *tensor) {
    if (!tensor || !tensor->gpu_buffer) return;

#if HAS_METAL
    if (g_gpu_ctx && g_gpu_ctx->backend == 1) {
        metal_buffer_free((metal_buffer_t *)tensor->gpu_buffer);
        tensor->gpu_buffer = NULL;
        tensor->gpu_valid = false;
        return;
    }
#endif

#ifdef HAS_CUDA
    if (g_gpu_ctx && g_gpu_ctx->backend == 2) {
        cuda_buffer_free((cuda_buffer_t *)tensor->gpu_buffer);
        tensor->gpu_buffer = NULL;
        tensor->gpu_valid = false;
        return;
    }
#endif

    // If no context, just NULL the pointer (leak, but safe)
    tensor->gpu_buffer = NULL;
    tensor->gpu_valid = false;
}

tensor_error_t tensor_sync_to_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;
    if (!tensor->cpu_valid) return TENSOR_ERROR_GPU_SYNC_FAILED;
    if (!ctx || ctx->backend == 0) return TENSOR_ERROR_GPU_UNAVAILABLE;

    // Allocate if needed
    if (!tensor->gpu_buffer) {
        tensor_error_t err = tensor_gpu_alloc(ctx, tensor);
        if (err != TENSOR_SUCCESS) return err;
    }

#if HAS_METAL
    if (ctx->backend == 1) {
        // Metal uses float2 - convert double complex → float2 during upload
        metal_buffer_t *buf = (metal_buffer_t *)tensor->gpu_buffer;
        float *gpu_ptr = (float *)metal_buffer_contents(buf);
        if (gpu_ptr) {
            for (uint64_t i = 0; i < tensor->total_size; i++) {
                gpu_ptr[i * 2]     = (float)creal(tensor->data[i]);
                gpu_ptr[i * 2 + 1] = (float)cimag(tensor->data[i]);
            }
            tensor->gpu_valid = true;
            return TENSOR_SUCCESS;
        }
        return TENSOR_ERROR_GPU_SYNC_FAILED;
    }
#endif

#ifdef HAS_CUDA
    if (ctx->backend == 2) {
        size_t size = tensor->total_size * sizeof(double complex);
        cuda_buffer_t *buf = (cuda_buffer_t *)tensor->gpu_buffer;
        if (cuda_buffer_upload(buf, tensor->data, size) == 0) {
            tensor->gpu_valid = true;
            return TENSOR_SUCCESS;
        }
        return TENSOR_ERROR_GPU_SYNC_FAILED;
    }
#endif

    return TENSOR_ERROR_GPU_UNAVAILABLE;
}

tensor_error_t tensor_sync_to_cpu(tensor_t *tensor) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;
    if (!tensor->gpu_valid || !tensor->gpu_buffer) {
        return TENSOR_ERROR_GPU_SYNC_FAILED;
    }

#if HAS_METAL
    if (g_gpu_ctx && g_gpu_ctx->backend == 1) {
        // Metal uses float2 - convert float2 → double complex during download
        metal_buffer_t *buf = (metal_buffer_t *)tensor->gpu_buffer;
        float *gpu_ptr = (float *)metal_buffer_contents(buf);
        if (gpu_ptr) {
            for (uint64_t i = 0; i < tensor->total_size; i++) {
                tensor->data[i] = (double)gpu_ptr[i * 2] +
                                  I * (double)gpu_ptr[i * 2 + 1];
            }
            tensor->cpu_valid = true;
            return TENSOR_SUCCESS;
        }
        return TENSOR_ERROR_GPU_SYNC_FAILED;
    }
#endif

#ifdef HAS_CUDA
    if (g_gpu_ctx && g_gpu_ctx->backend == 2) {
        size_t size = tensor->total_size * sizeof(double complex);
        cuda_buffer_t *buf = (cuda_buffer_t *)tensor->gpu_buffer;
        if (cuda_buffer_download(buf, tensor->data, size) == 0) {
            tensor->cpu_valid = true;
            return TENSOR_SUCCESS;
        }
        return TENSOR_ERROR_GPU_SYNC_FAILED;
    }
#endif

    return TENSOR_ERROR_GPU_UNAVAILABLE;
}

tensor_error_t tensor_ensure_cpu(tensor_t *tensor) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;

    if (tensor->cpu_valid) {
        return TENSOR_SUCCESS;  // Already valid
    }

    return tensor_sync_to_cpu(tensor);
}

tensor_error_t tensor_ensure_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor) {
    if (!tensor) return TENSOR_ERROR_NULL_PTR;
    if (!ctx || ctx->backend == 0) return TENSOR_ERROR_GPU_UNAVAILABLE;

    if (tensor->gpu_valid && tensor->gpu_buffer) {
        return TENSOR_SUCCESS;  // Already valid
    }

    return tensor_sync_to_gpu(ctx, tensor);
}

void tensor_invalidate_cpu(tensor_t *tensor) {
    if (tensor) {
        tensor->cpu_valid = false;
    }
}

void tensor_invalidate_gpu(tensor_t *tensor) {
    if (tensor) {
        tensor->gpu_valid = false;
    }
}
