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

```c
#define TENSOR_MAX_RANK     16    // Maximum tensor rank
#define TENSOR_ALIGNMENT    64    // Memory alignment (bytes)
#define TENSOR_DEFAULT_BOND_DIM  64   // Default MPS bond dimension
#define TENSOR_MAX_BOND_DIM 4096  // Maximum before truncation warning
```

## Error Codes

### tensor_error_t

```c
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
```

## Core Data Structures

### tensor_t

Complex tensor with arbitrary rank.

```c
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
```

**Memory Layout**: Row-major storage. For rank-3 tensor T[i][j][k] with dimensions [d0, d1, d2]:
```
linear_index = i * (d1 * d2) + j * d2 + k
```

**GPU Synchronization Model**:
- `cpu_valid=true, gpu_valid=false`: Data on CPU only (initial state)
- `cpu_valid=false, gpu_valid=true`: Data on GPU only (after GPU op)
- `cpu_valid=true, gpu_valid=true`: Both copies valid (after sync)

### tensor_svd_result_t

SVD decomposition result: $A = U \cdot S \cdot V^\dagger$

```c
typedef struct {
    tensor_t *U;                    // Left singular vectors (m x k)
    double *S;                      // Singular values (k) - real, non-negative
    tensor_t *Vh;                   // Right singular vectors conjugate transpose (k x n)
    uint32_t k;                     // Number of singular values kept
    double truncation_error;        // Frobenius norm of discarded values
} tensor_svd_result_t;
```

### tensor_qr_result_t

QR decomposition result: $A = Q \cdot R$

```c
typedef struct {
    tensor_t *Q;                    // Orthonormal matrix
    tensor_t *R;                    // Upper triangular matrix
} tensor_qr_result_t;
```

### tensor_contraction_spec_t

Tensor contraction specification.

```c
typedef struct {
    uint32_t num_contractions;      // Number of index pairs to contract
    tensor_axis_pair_t pairs[TENSOR_MAX_RANK]; // Pairs of axes to contract
} tensor_contraction_spec_t;
```

## Creation and Destruction

### tensor_create

Create a new tensor with specified dimensions.

```c
tensor_t *tensor_create(uint32_t rank, const uint32_t *dims);
```

**Parameters**:
- `rank`: Number of dimensions
- `dims`: Array of dimension sizes

**Returns**: New tensor initialized to zero, or NULL on failure

**Example**:
```c
// Create rank-3 tensor with dimensions 4 x 2 x 4
uint32_t dims[] = {4, 2, 4};
tensor_t *T = tensor_create(3, dims);
```

### tensor_create_with_data

Create tensor initialized with given data.

```c
tensor_t *tensor_create_with_data(uint32_t rank, const uint32_t *dims,
                                   const double complex *data);
```

### tensor_create_view

Create tensor as view of existing data (no copy).

```c
tensor_t *tensor_create_view(uint32_t rank, const uint32_t *dims,
                              double complex *data);
```

**Note**: Caller must ensure data outlives tensor. Returns tensor with `owns_data = false`.

### tensor_create_scalar

Create scalar tensor (rank 0).

```c
tensor_t *tensor_create_scalar(double complex value);
```

### tensor_create_vector

Create rank-1 tensor (vector).

```c
tensor_t *tensor_create_vector(uint32_t size);
```

### tensor_create_matrix

Create rank-2 tensor (matrix).

```c
tensor_t *tensor_create_matrix(uint32_t rows, uint32_t cols);
```

### tensor_create_identity

Create identity matrix tensor.

```c
tensor_t *tensor_create_identity(uint32_t size);
```

### tensor_copy

Create deep copy of tensor.

```c
tensor_t *tensor_copy(const tensor_t *src);
```

### tensor_free

Free tensor and its data.

```c
void tensor_free(tensor_t *tensor);
```

### tensor_svd_free

Free SVD result structure.

```c
void tensor_svd_free(tensor_svd_result_t *svd);
```

### tensor_qr_free

Free QR result structure.

```c
void tensor_qr_free(tensor_qr_result_t *qr);
```

## Element Access

### tensor_get

Get element at indices.

```c
double complex tensor_get(const tensor_t *tensor, const uint32_t *indices);
```

### tensor_set

Set element at indices.

```c
tensor_error_t tensor_set(tensor_t *tensor, const uint32_t *indices,
                          double complex value);
```

### tensor_get_linear / tensor_set_linear

Access by linear index.

```c
double complex tensor_get_linear(const tensor_t *tensor, uint64_t idx);
tensor_error_t tensor_set_linear(tensor_t *tensor, uint64_t idx,
                                  double complex value);
```

### tensor_get_linear_index

Convert multi-dimensional indices to linear index.

```c
uint64_t tensor_get_linear_index(const tensor_t *tensor, const uint32_t *indices);
```

### tensor_get_multi_index

Convert linear index to multi-dimensional indices.

```c
void tensor_get_multi_index(const tensor_t *tensor, uint64_t linear_idx,
                            uint32_t *indices);
```

## Shape Operations

### tensor_reshape

Reshape tensor to new dimensions.

```c
tensor_t *tensor_reshape(const tensor_t *tensor, uint32_t new_rank,
                         const uint32_t *new_dims);
```

**Note**: Total size must remain the same. Returns new tensor.

### tensor_transpose

Transpose tensor axes.

```c
tensor_t *tensor_transpose(const tensor_t *tensor, const uint32_t *perm);
```

**Parameters**:
- `perm`: Permutation of axes (length = rank)

### tensor_swapaxes

Swap two axes of tensor.

```c
tensor_t *tensor_swapaxes(const tensor_t *tensor, uint32_t axis1, uint32_t axis2);
```

### tensor_flatten

Flatten tensor to 1D.

```c
tensor_t *tensor_flatten(const tensor_t *tensor);
```

### tensor_expand_dims

Add new dimension of size 1.

```c
tensor_t *tensor_expand_dims(const tensor_t *tensor, uint32_t axis);
```

### tensor_squeeze

Remove dimensions of size 1.

```c
tensor_t *tensor_squeeze(const tensor_t *tensor);
```

## Arithmetic Operations

### tensor_add / tensor_sub

Element-wise addition/subtraction.

```c
tensor_t *tensor_add(const tensor_t *a, const tensor_t *b);
tensor_t *tensor_sub(const tensor_t *a, const tensor_t *b);
```

### tensor_scale

Multiply tensor by scalar.

```c
tensor_t *tensor_scale(const tensor_t *tensor, double complex scalar);
```

### tensor_hadamard

Element-wise multiplication (Hadamard product).

```c
tensor_t *tensor_hadamard(const tensor_t *a, const tensor_t *b);
```

### tensor_conj

Complex conjugate.

```c
tensor_t *tensor_conj(const tensor_t *tensor);
```

### tensor_scale_inplace / tensor_add_inplace

In-place operations.

```c
tensor_error_t tensor_scale_inplace(tensor_t *tensor, double complex scalar);
tensor_error_t tensor_add_inplace(tensor_t *tensor, const tensor_t *other);
```

## Norms and Properties

### tensor_norm_frobenius

Frobenius norm (L2 norm of all elements).

```c
double tensor_norm_frobenius(const tensor_t *tensor);
```

**Formula**: $\|T\|_F = \sqrt{\sum |t_i|^2}$

### tensor_norm_max

Maximum absolute value norm.

```c
double tensor_norm_max(const tensor_t *tensor);
```

### tensor_sum

Sum of all elements.

```c
double complex tensor_sum(const tensor_t *tensor);
```

### tensor_sum_axis

Sum along an axis.

```c
tensor_t *tensor_sum_axis(const tensor_t *tensor, uint32_t axis);
```

### tensor_inner

Frobenius inner product.

```c
double complex tensor_inner(const tensor_t *a, const tensor_t *b);
```

**Formula**: $\langle A, B \rangle = \sum \text{conj}(a_i) \cdot b_i$

### tensor_shape_equal

Check if tensors have same shape.

```c
bool tensor_shape_equal(const tensor_t *a, const tensor_t *b);
```

### tensor_is_zero

Check if tensor is approximately zero.

```c
bool tensor_is_zero(const tensor_t *tensor, double tol);
```

### tensor_allclose

Check if tensors are approximately equal.

```c
bool tensor_allclose(const tensor_t *a, const tensor_t *b, double tol);
```

## Matrix Operations

### tensor_matmul

Matrix multiplication for rank-2 tensors.

```c
tensor_t *tensor_matmul(const tensor_t *a, const tensor_t *b);
```

**Computes**: $C = A \times B$ for matrices

### tensor_matvec

Matrix-vector multiplication.

```c
tensor_t *tensor_matvec(const tensor_t *mat, const tensor_t *vec);
```

### tensor_outer

Outer product of two vectors.

```c
tensor_t *tensor_outer(const tensor_t *a, const tensor_t *b);
```

**Formula**: $M[i,j] = a[i] \cdot b[j]$

### tensor_kron

Tensor product (Kronecker product for matrices).

```c
tensor_t *tensor_kron(const tensor_t *a, const tensor_t *b);
```

### tensor_trace

Matrix trace.

```c
double complex tensor_trace(const tensor_t *mat);
```

### tensor_dagger

Hermitian conjugate (conjugate transpose).

```c
tensor_t *tensor_dagger(const tensor_t *mat);
```

## Decompositions

### tensor_svd

Singular Value Decomposition with optional truncation.

```c
tensor_svd_result_t *tensor_svd(const tensor_t *mat, uint32_t max_rank,
                                 double cutoff);
```

**Parameters**:
- `mat`: Matrix to decompose
- `max_rank`: Maximum singular values to keep (0 = all)
- `cutoff`: Discard singular values below threshold

**Returns**: SVD result with $A = U \cdot S \cdot V^\dagger$

### tensor_svd_truncate

SVD with automatic rank selection based on error threshold.

```c
tensor_svd_result_t *tensor_svd_truncate(const tensor_t *mat, double max_error);
```

**Parameters**:
- `max_error`: Maximum allowed truncation error (Frobenius norm)

### tensor_qr

QR decomposition.

```c
tensor_qr_result_t *tensor_qr(const tensor_t *mat);
```

**Returns**: QR result with $A = Q \cdot R$

### tensor_lq

LQ decomposition.

```c
tensor_qr_result_t *tensor_lq(const tensor_t *mat);
```

## Tensor Contraction

### tensor_contract

Contract two tensors over specified axes.

```c
tensor_t *tensor_contract(const tensor_t *a, const tensor_t *b,
                          const uint32_t *axes_a, const uint32_t *axes_b,
                          uint32_t num_contract);
```

**Parameters**:
- `axes_a`: Axes of A to contract
- `axes_b`: Axes of B to contract
- `num_contract`: Number of axes to contract

**Example**: Contracting A[i,j,k] with B[k,l,m] over axis (2,0) gives C[i,j,l,m]

### tensor_tensordot

Tensor dot product (contracts last axis of A with first axis of B).

```c
tensor_t *tensor_tensordot(const tensor_t *a, const tensor_t *b);
```

### tensor_einsum

Einstein summation (limited subset).

```c
tensor_t *tensor_einsum(const tensor_t *a, const tensor_t *b,
                        const char *subscripts);
```

**Example**: `"ij,jk->ik"` for matrix multiplication

## GPU Acceleration

### tensor_gpu_context_t

Opaque GPU context handle for tensor operations.

### tensor_gpu_context_create

Create GPU context.

```c
tensor_gpu_context_t *tensor_gpu_context_create(void);
```

**Returns**: GPU context or NULL if unavailable

### tensor_gpu_context_destroy

Destroy GPU context.

```c
void tensor_gpu_context_destroy(tensor_gpu_context_t *ctx);
```

### tensor_gpu_available

Check if GPU acceleration is available.

```c
bool tensor_gpu_available(void);
```

### tensor_gpu_get_context

Get global GPU context (singleton).

```c
tensor_gpu_context_t *tensor_gpu_get_context(void);
```

### tensor_gpu_alloc

Allocate GPU buffer for tensor.

```c
tensor_error_t tensor_gpu_alloc(tensor_gpu_context_t *ctx, tensor_t *tensor);
```

### tensor_gpu_free

Free GPU buffer.

```c
void tensor_gpu_free(tensor_t *tensor);
```

### tensor_sync_to_gpu

Copy CPU data to GPU.

```c
tensor_error_t tensor_sync_to_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);
```

### tensor_sync_to_cpu

Copy GPU data back to CPU.

```c
tensor_error_t tensor_sync_to_cpu(tensor_t *tensor);
```

### tensor_ensure_cpu / tensor_ensure_gpu

Ensure data is valid on CPU/GPU, syncing if needed.

```c
tensor_error_t tensor_ensure_cpu(tensor_t *tensor);
tensor_error_t tensor_ensure_gpu(tensor_gpu_context_t *ctx, tensor_t *tensor);
```

### tensor_invalidate_cpu / tensor_invalidate_gpu

Mark data as invalid after modification.

```c
void tensor_invalidate_cpu(tensor_t *tensor);
void tensor_invalidate_gpu(tensor_t *tensor);
```

## Utilities

### tensor_zero

Set all elements to zero.

```c
void tensor_zero(tensor_t *tensor);
```

### tensor_fill

Set all elements to a constant.

```c
void tensor_fill(tensor_t *tensor, double complex value);
```

### tensor_copy_data

Copy data between tensors.

```c
tensor_error_t tensor_copy_data(tensor_t *dst, const tensor_t *src);
```

### tensor_print_shape

Print tensor shape information.

```c
void tensor_print_shape(const tensor_t *tensor, const char *name);
```

### tensor_print_data

Print tensor data (for debugging).

```c
void tensor_print_data(const tensor_t *tensor, const char *name,
                       uint32_t max_elements);
```

### tensor_error_string

Get error string.

```c
const char *tensor_error_string(tensor_error_t error);
```

### tensor_memory_usage

Calculate memory usage in bytes.

```c
size_t tensor_memory_usage(const tensor_t *tensor);
```

### tensor_random_fill

Fill tensor with random complex values.

```c
void tensor_random_fill(tensor_t *tensor, uint64_t seed);
```

### tensor_random_unitary

Create random unitary matrix.

```c
tensor_t *tensor_random_unitary(uint32_t size, uint64_t seed);
```

## MPS Example

```c
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
```

## See Also

- [DMRG API](../../algorithms/dmrg-algorithm.md) - Density Matrix Renormalization Group
- [GPU Metal API](gpu-metal.md) - GPU acceleration details
- [Concepts: Tensor Networks](../../concepts/tensor-networks.md) - Theory
