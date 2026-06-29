# Archived Moonlab Documentation: Tensor Network API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Tensor Network API

Complete reference for tensor operations and tensor network simulation in the C library.

**Header**: `src/algorithms/tensor_network/tensor.h`

## Overview

The tensor module provides arbitrary-rank complex tensor operations optimized for quantum simulation. Key features:

- SIMD-aligned memory allocation (64-byte for AVX-512)
- GPU acceleration support (Metal on Apple Silicon)
- SVD and QR decomposition for MPS/DMRG
- Einstein summation and tensor contraction

## Configuration Constants

[archived fence delimiter: ```c]
#define TENSOR_MAX_RANK     16    // Maximum tensor rank
#define TENSOR_ALIGNMENT    64    // Memory alignment (bytes)
#define TENSOR_DEFAULT_BOND_DIM  64   // Default MPS bond dimension
#define TENSOR_MAX_BOND_DIM 4096  // Maximum before truncation warning
[archived fence delimiter: ```]

## Error Codes

### tensor_error_t

[archived fence delimiter: ```c]
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
[archived fence delimiter: ```]

## Core Data Structures

### tensor_t

Complex tensor with arbitrary rank.

[archived fence delimiter: ```c]
typedef struct {
    double complex *data;           // Complex amplitude data (SIMD-aligned)
    uint32_t rank;                  // Number of dimensions (0 = scalar)
    uint32_t dims[TENSOR_MAX_RANK]; // Size of each dimension
    uint64_t strides[TENSOR_MAX_RANK]; // Stride for each dimension
    uint64_t total_size;            // Total number of elements
    bool owns_data;                 // Whether tensor owns its data buffer

    // GPU acceleration support
    gpu_buffer_t *gpu_buffer;       // GPU buffer (NULL if not allocated)
    bool cpu_valid;                 // CPU data is current
    bool gpu_valid;                 // GPU data is current
} tensor_t;
[archived fence delimiter: ```]

**Memory Layout**: Row-major storage. For rank-3 tensor T[i][j][k] with dimensions [d0, d1, d2]:
[archived fence delimiter: ```]
linear_index = i * (d1 * d2) + j * d2 + k
[archived fence delimiter: ```]

**GPU Synchronization Model**:
- `cpu_valid=true, gpu_valid=false`: Data on CPU only (initial state)
- `cpu_valid=false, gpu_valid=true`: Data on GPU only (after GPU op)
- `cpu_valid=true, gpu_valid=true`: Both copies valid (after sync)

### tensor_svd_result_t

SVD decomposition result: $A = U \cdot S \cdot V^\dagger$

[archived fence delimiter: ```c]
typedef struct {
    tensor_t *U;                    // Left singular vectors (m x k)
    double *S;                      // Singular values (k) - real, non-negative
    tensor_t *Vh;                   // Right singular vectors conjugate transpose (k x n)
    uint32_t k;                     // Number of singular values kept
    double truncation_error;        // Frobenius norm of discarded values
} tensor_svd_result_t;
[archived fence delimiter: ```]

### tensor_qr_result_t

QR decomposition result: $A = Q \cdot R$

[archived fence delimiter: ```c]
typedef struct {
    tensor_t *Q;                    // Orthonormal matrix
    tensor_t *R;                    // Upper triangular matrix
} tensor_qr_result_t;
[archived fence delimiter: ```]

### tensor_contraction_spec_t

Tensor contraction specification.

[archived fence delimiter: ```c]
typedef struct {
    uint32_t num_contractions;      // Number of index pairs to contract
    tensor_axis_pair_t pairs[TENSOR_MAX_RANK]; // Pairs of axes to contract
} tensor_contraction_spec_t;
[archived fence delimiter: ```]

## Creation and Destruction

### tensor_create

Create a new tensor with specified dimensions.

[archived fence delimiter: ```c]
tensor_t *tensor_create(uint32_t rank, const uint32_t *dims);
[archived fence delimiter: ```]

**Parameters**:
- `rank`: Number of dimensions
- `dims`: Array of dimension sizes

**Returns**: New tensor initialized to zero, or NULL on failure

**Example**:
[archived fence delimiter: ```c]
// Create rank-3 tensor with dimensions 4 x 2 x 4
uint32_t dims[] = {4, 2, 4};
tensor_t *T = tensor_create(3, dims);
[archived fence delimiter: ```]

### tensor_create_with_data

Create tensor initialized with given data.

[archived fence delimiter: ```c]
tensor_t *tensor_create_with_data(uint32_t rank, const uint32_t *dims,
                                   const double complex *data);
[archived fence delimiter: ```]

### tensor_create_view

Create tensor as view of existing data (no copy).

[archived fence delimiter: ```c]
tensor_t *tensor_create_view(uint32_t rank, const uint32_t *dims,
                              double complex *data);
[archived fence delimiter: ```]

**Note**: Caller must ensure data outlives tensor. Returns tensor with `owns_data = false`.

### tensor_create_scalar

Create scalar tensor (rank 0).

[archived fence delimiter: ```c]
tensor_t *tensor_create_scalar(double complex value);
[archived fence delimiter: ```]

### tensor_create_vector

Create rank-1 tensor (vector).

[archived fence delimiter: ```c]
tensor_t *tensor_create_vector(uint32_t size);
[archived fence delimiter: ```]

### tensor_create_matrix

Create rank-2 tensor (matrix).

[archived fence delimiter: ```c]
tensor_t *tensor_create_matrix(uint32_t rows, uint32_t cols);
[archived fence delimiter: ```]

### tensor_create_identity

Create identity matrix tensor.

[archived fence delimiter: ```c]
tensor_t *tensor_create_identity(uint32_t size);
[archived fence delimiter: ```]

### tensor_copy

Create deep copy of tensor.

[archived fence delimiter: ```c]
tensor_t *tensor_copy(const tensor_t *src);
[archived fence delimiter: ```]

### tensor_free

Free tensor and its data.

[archived fence delimiter: ```c]
void tensor_free(tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_svd_free

Free SVD result structure.

[archived fence delimiter: ```c]
void tensor_svd_free(tensor_svd_result_t *svd);
[archived fence delimiter: ```]

### tensor_qr_free

Free QR result structure.

[archived fence delimiter: ```c]
void tensor_qr_free(tensor_qr_result_t *qr);
[archived fence delimiter: ```]

## Element Access

### tensor_get

Get element at indices.

[archived fence delimiter: ```c]
double complex tensor_get(const tensor_t *tensor, const uint32_t *indices);
[archived fence delimiter: ```]

### tensor_set

Set element at indices.

[archived fence delimiter: ```c]
tensor_error_t tensor_set(tensor_t *tensor, const uint32_t *indices,
                          double complex value);
[archived fence delimiter: ```]

### tensor_get_linear / tensor_set_linear

Access by linear index.

[archived fence delimiter: ```c]
double complex tensor_get_linear(const tensor_t *tensor, uint64_t idx);
tensor_error_t tensor_set_linear(tensor_t *tensor, uint64_t idx,
                                  double complex value);
[archived fence delimiter: ```]

### tensor_get_linear_index

Convert multi-dimensional indices to linear index.

[archived fence delimiter: ```c]
uint64_t tensor_get_linear_index(const tensor_t *tensor, const uint32_t *indices);
[archived fence delimiter: ```]

### tensor_get_multi_index

Convert linear index to multi-dimensional indices.

[archived fence delimiter: ```c]
void tensor_get_multi_index(const tensor_t *tensor, uint64_t linear_idx,
                            uint32_t *indices);
[archived fence delimiter: ```]

## Shape Operations

### tensor_reshape

Reshape tensor to new dimensions.

[archived fence delimiter: ```c]
tensor_t *tensor_reshape(const tensor_t *tensor, uint32_t new_rank,
                         const uint32_t *new_dims);
[archived fence delimiter: ```]

**Note**: Total size must remain the same. Returns new tensor.

### tensor_transpose

Transpose tensor axes.

[archived fence delimiter: ```c]
tensor_t *tensor_transpose(const tensor_t *tensor, const uint32_t *perm);
[archived fence delimiter: ```]

**Parameters**:
- `perm`: Permutation of axes (length = rank)

### tensor_swapaxes

Swap two axes of tensor.

[archived fence delimiter: ```c]
tensor_t *tensor_swapaxes(const tensor_t *tensor, uint32_t axis1, uint32_t axis2);
[archived fence delimiter: ```]

### tensor_flatten

Flatten tensor to 1D.

[archived fence delimiter: ```c]
tensor_t *tensor_flatten(const tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_expand_dims

Add new dimension of size 1.

[archived fence delimiter: ```c]
tensor_t *tensor_expand_dims(const tensor_t *tensor, uint32_t axis);
[archived fence delimiter: ```]

### tensor_squeeze

Remove dimensions of size 1.

[archived fence delimiter: ```c]
tensor_t *tensor_squeeze(const tensor_t *tensor);
[archived fence delimiter: ```]

## Arithmetic Operations

### tensor_add / tensor_sub

Element-wise addition/subtraction.

[archived fence delimiter: ```c]
tensor_t *tensor_add(const tensor_t *a, const tensor_t *b);
tensor_t *tensor_sub(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

### tensor_scale

Multiply tensor by scalar.

[archived fence delimiter: ```c]
tensor_t *tensor_scale(const tensor_t *tensor, double complex scalar);
[archived fence delimiter: ```]

### tensor_hadamard

Element-wise multiplication (Hadamard product).

[archived fence delimiter: ```c]
tensor_t *tensor_hadamard(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

### tensor_conj

Complex conjugate.

[archived fence delimiter: ```c]
tensor_t *tensor_conj(const tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_scale_inplace / tensor_add_inplace

In-place operations.

[archived fence delimiter: ```c]
tensor_error_t tensor_scale_inplace(tensor_t *tensor, double complex scalar);
tensor_error_t tensor_add_inplace(tensor_t *tensor, const tensor_t *other);
[archived fence delimiter: ```]

## Norms and Properties

### tensor_norm_frobenius

Frobenius norm (L2 norm of all elements).

[archived fence delimiter: ```c]
double tensor_norm_frobenius(const tensor_t *tensor);
[archived fence delimiter: ```]

**Formula**: $\|T\|_F = \sqrt{\sum |t_i|^2}$

### tensor_norm_max

Maximum absolute value norm.

[archived fence delimiter: ```c]
double tensor_norm_max(const tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_sum

Sum of all elements.

[archived fence delimiter: ```c]
double complex tensor_sum(const tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_sum_axis

Sum along an axis.

[archived fence delimiter: ```c]
tensor_t *tensor_sum_axis(const tensor_t *tensor, uint32_t axis);
[archived fence delimiter: ```]

### tensor_inner

Frobenius inner product.

[archived fence delimiter: ```c]
double complex tensor_inner(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

**Formula**: $\langle A, B \rangle = \sum \text{conj}(a_i) \cdot b_i$

### tensor_shape_equal

Check if tensors have same shape.

[archived fence delimiter: ```c]
bool tensor_shape_equal(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

### tensor_is_zero

Check if tensor is approximately zero.

[archived fence delimiter: ```c]
bool tensor_is_zero(const tensor_t *tensor, double tol);
[archived fence delimiter: ```]

### tensor_allclose

Check if tensors are approximately equal.

[archived fence delimiter: ```c]
bool tensor_allclose(const tensor_t *a, const tensor_t *b, double tol);
[archived fence delimiter: ```]

## Matrix Operations

### tensor_matmul

Matrix multiplication for rank-2 tensors.

[archived fence delimiter: ```c]
tensor_t *tensor_matmul(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

**Computes**: $C = A \times B$ for matrices

### tensor_matvec

Matrix-vector multiplication.

[archived fence delimiter: ```c]
tensor_t *tensor_matvec(const tensor_t *mat, const tensor_t *vec);
[archived fence delimiter: ```]

### tensor_outer

Outer product of two vectors.

[archived fence delimiter: ```c]
tensor_t *tensor_outer(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

**Formula**: $M[i,j] = a[i] \cdot b[j]$

### tensor_kron

Tensor product (Kronecker product for matrices).

[archived fence delimiter: ```c]
tensor_t *tensor_kron(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

### tensor_trace

Matrix trace.

[archived fence delimiter: ```c]
double complex tensor_trace(const tensor_t *mat);
[archived fence delimiter: ```]

### tensor_dagger

Hermitian conjugate (conjugate transpose).

[archived fence delimiter: ```c]
tensor_t *tensor_dagger(const tensor_t *mat);
[archived fence delimiter: ```]

## Decompositions

### tensor_svd

Singular Value Decomposition with optional truncation.

[archived fence delimiter: ```c]
tensor_svd_result_t *tensor_svd(const tensor_t *mat, uint32_t max_rank,
                                 double cutoff);
[archived fence delimiter: ```]

**Parameters**:
- `mat`: Matrix to decompose
- `max_rank`: Maximum singular values to keep (0 = all)
- `cutoff`: Discard singular values below threshold

**Returns**: SVD result with $A = U \cdot S \cdot V^\dagger$

### tensor_svd_truncate

SVD with automatic rank selection based on error threshold.

[archived fence delimiter: ```c]
tensor_svd_result_t *tensor_svd_truncate(const tensor_t *mat, double max_error);
[archived fence delimiter: ```]

**Parameters**:
- `max_error`: Maximum allowed truncation error (Frobenius norm)

### tensor_qr

QR decomposition.

[archived fence delimiter: ```c]
tensor_qr_result_t *tensor_qr(const tensor_t *mat);
[archived fence delimiter: ```]

**Returns**: QR result with $A = Q \cdot R$

### tensor_lq

LQ decomposition.

[archived fence delimiter: ```c]
tensor_qr_result_t *tensor_lq(const tensor_t *mat);
[archived fence delimiter: ```]

## Tensor Contraction

### tensor_contract

Contract two tensors over specified axes.

[archived fence delimiter: ```c]
tensor_t *tensor_contract(const tensor_t *a, const tensor_t *b,
                          const uint32_t *axes_a, const uint32_t *axes_b,
                          uint32_t num_contract);
[archived fence delimiter: ```]

**Parameters**:
- `axes_a`: Axes of A to contract
- `axes_b`: Axes of B to contract
- `num_contract`: Number of axes to contract

**Example**: Contracting A[i,j,k] with B[k,l,m] over axis (2,0) gives C[i,j,l,m]

### tensor_tensordot

Tensor dot product (contracts last axis of A with first axis of B).

[archived fence delimiter: ```c]
tensor_t *tensor_tensordot(const tensor_t *a, const tensor_t *b);
[archived fence delimiter: ```]

### tensor_einsum

Einstein summation (limited subset).

[archived fence delimiter: ```c]
tensor_t *tensor_einsum(const tensor_t *a, const tensor_t *b,
                        const char *subscripts);
[archived fence delimiter: ```]

**Example**: `"ij,jk->ik"` for matrix multiplication

## GPU Acceleration

### tensor_gpu_context_t

Opaque GPU context handle for tensor operations.

### tensor_gpu_context_create

Create GPU context.

[archived fence delimiter: ```c]
tensor_gpu_context_t *tensor_gpu_context_create(void);
[archived fence delimiter: ```]

**Returns**: GPU context or NULL if unavailable

### tensor_gpu_context_destroy

Destroy GPU context.

[archived fence delimiter: ```c]
void tensor_gpu_context_destroy(tensor_gpu_context_t *ctx);
[archived fence delimiter: ```]

### tensor_gpu_available

Check if GPU acceleration is available.

[archived fence delimiter: ```c]
bool tensor_gpu_available(void);
[archived fence delimiter: ```]

### tensor_gpu_get_context

Get global GPU context (singleton).

[archived fence delimiter: ```c]
tensor_gpu_context_t *tensor_gpu_get_context(void);
[archived fence delimiter: ```]

### tensor_gpu_alloc

Allocate GPU buffer for tensor.

[archived fence delimiter: ```c]
tensor_error_t tensor_gpu_alloc(tensor_gpu_context_t *ctx, tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_gpu_free

Free GPU buffer.

[archived fence delimiter: ```c]
void tensor_gpu_free(tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_sync_to_gpu

Copy CPU data to GPU.

[archived fence delimiter: ```c]
tensor_error_t tensor_sync_to_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_sync_to_cpu

Copy GPU data back to CPU.

[archived fence delimiter: ```c]
tensor_error_t tensor_sync_to_cpu(tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_ensure_cpu / tensor_ensure_gpu

Ensure data is valid on CPU/GPU, syncing if needed.

[archived fence delimiter: ```c]
tensor_error_t tensor_ensure_cpu(tensor_t *tensor);
tensor_error_t tensor_ensure_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_invalidate_cpu / tensor_invalidate_gpu

Mark data as invalid after modification.

[archived fence delimiter: ```c]
void tensor_invalidate_cpu(tensor_t *tensor);
void tensor_invalidate_gpu(tensor_t *tensor);
[archived fence delimiter: ```]

## Utilities

### tensor_zero

Set all elements to zero.

[archived fence delimiter: ```c]
void tensor_zero(tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_fill

Set all elements to a constant.

[archived fence delimiter: ```c]
void tensor_fill(tensor_t *tensor, double complex value);
[archived fence delimiter: ```]

### tensor_copy_data

Copy data between tensors.

[archived fence delimiter: ```c]
tensor_error_t tensor_copy_data(tensor_t *dst, const tensor_t *src);
[archived fence delimiter: ```]

### tensor_print_shape

Print tensor shape information.

[archived fence delimiter: ```c]
void tensor_print_shape(const tensor_t *tensor, const char *name);
[archived fence delimiter: ```]

### tensor_print_data

Print tensor data (for debugging).

[archived fence delimiter: ```c]
void tensor_print_data(const tensor_t *tensor, const char *name,
                       uint32_t max_elements);
[archived fence delimiter: ```]

### tensor_error_string

Get error string.

[archived fence delimiter: ```c]
const char *tensor_error_string(tensor_error_t error);
[archived fence delimiter: ```]

### tensor_memory_usage

Calculate memory usage in bytes.

[archived fence delimiter: ```c]
size_t tensor_memory_usage(const tensor_t *tensor);
[archived fence delimiter: ```]

### tensor_random_fill

Fill tensor with random complex values.

[archived fence delimiter: ```c]
void tensor_random_fill(tensor_t *tensor, uint64_t seed);
[archived fence delimiter: ```]

### tensor_random_unitary

Create random unitary matrix.

[archived fence delimiter: ```c]
tensor_t *tensor_random_unitary(uint32_t size, uint64_t seed);
[archived fence delimiter: ```]

## MPS Example

[archived fence delimiter: ```c]
#include "src/algorithms/tensor_network/tensor.h"

// Create MPS tensors for 4-site chain with bond dimension 8
uint32_t dims_left[] = {1, 2, 8};      // Left boundary
uint32_t dims_bulk[] = {8, 2, 8};      // Bulk sites
uint32_t dims_right[] = {8, 2, 1};     // Right boundary

tensor_t *A0 = tensor_create(3, dims_left);
tensor_t *A1 = tensor_create(3, dims_bulk);
tensor_t *A2 = tensor_create(3, dims_bulk);
tensor_t *A3 = tensor_create(3, dims_right);

// Contract A0 and A1
uint32_t axes_a[] = {2};  // Right bond of A0
uint32_t axes_b[] = {0};  // Left bond of A1
tensor_t *theta = tensor_contract(A0, A1, axes_a, axes_b, 1);

// SVD truncation
tensor_svd_result_t *svd = tensor_svd_truncate(theta, 1e-10);
printf("New bond dimension: %u\n", svd->k);
printf("Truncation error: %.2e\n", svd->truncation_error);

// Cleanup
tensor_svd_free(svd);
tensor_free(theta);
tensor_free(A0); tensor_free(A1);
tensor_free(A2); tensor_free(A3);
[archived fence delimiter: ```]

## See Also

- [DMRG API](../../algorithms/dmrg-algorithm.md) - Density Matrix Renormalization Group
- [GPU Metal API](gpu-metal.md) - GPU acceleration details
- [Concepts: Tensor Networks](../../concepts/tensor-networks.md) - Theory
```
