# Tensor Network Engine

MPS and DMRG implementation internals.

## Overview

The tensor network engine enables simulation of quantum systems far beyond state vector limits by representing states as networks of small tensors. The primary representation is Matrix Product States (MPS), with algorithms including DMRG and TEBD.

## Architecture

### Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                     Tensor Network Engine                         │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   High-Level Interface                     │   │
│  │  MPS, MPO, DMRG, TEBD classes                              │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                             │                                     │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │                   Tensor Operations                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │   │
│  │  │ Contraction│  │    SVD     │  │    Eigensolvers     │   │   │
│  │  └────────────┘  └────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────┬─────────────────────────────────┘   │
│                             │                                     │
│  ┌──────────────────────────▼─────────────────────────────────┐   │
│  │                   Storage Layer                            │   │
│  │  Dense tensors, Sparse tensors, Memory pools               │   │
│  └────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## Tensor Representation

### Dense Tensor

```c
typedef struct {
    Complex* data;           // Contiguous data array
    uint32_t rank;           // Number of indices
    uint32_t* dims;          // Dimension of each index
    uint64_t total_size;     // Product of dimensions
    uint32_t* strides;       // Strides for indexing
} tensor_t;
```

### Index Convention

For a tensor $T_{i_0, i_1, \ldots, i_{r-1}}$:

```c
// Linear index from multi-index
uint64_t linear_index(tensor_t* t, uint32_t* indices) {
    uint64_t idx = 0;
    for (uint32_t i = 0; i < t->rank; i++) {
        idx += indices[i] * t->strides[i];
    }
    return idx;
}

// Element access
Complex tensor_get(tensor_t* t, uint32_t* indices) {
    return t->data[linear_index(t, indices)];
}

void tensor_set(tensor_t* t, uint32_t* indices, Complex value) {
    t->data[linear_index(t, indices)] = value;
}
```

### Memory Layout

Column-major (Fortran-style) for BLAS compatibility:

```c
tensor_t* tensor_create(uint32_t rank, const uint32_t* dims) {
    tensor_t* t = malloc(sizeof(tensor_t));

    t->rank = rank;
    t->dims = malloc(rank * sizeof(uint32_t));
    t->strides = malloc(rank * sizeof(uint32_t));

    // Column-major strides
    t->total_size = 1;
    for (uint32_t i = 0; i < rank; i++) {
        t->dims[i] = dims[i];
        t->strides[i] = t->total_size;
        t->total_size *= dims[i];
    }

    t->data = aligned_alloc(32, t->total_size * sizeof(Complex));

    return t;
}
```

## Matrix Product State

### MPS Structure

```c
typedef struct {
    tensor_t** tensors;      // Site tensors A[0], A[1], ..., A[n-1]
    uint32_t num_sites;      // n
    uint32_t local_dim;      // d (physical dimension, usually 2)
    uint32_t* bond_dims;     // χ[i] = bond dimension between sites i and i+1
    uint32_t max_bond_dim;   // Maximum allowed χ

    // Canonical form tracking
    enum {
        CANONICAL_NONE,
        CANONICAL_LEFT,
        CANONICAL_RIGHT,
        CANONICAL_MIXED
    } canonical_form;
    uint32_t canonical_center;  // For mixed canonical form
} mps_t;
```

### Tensor Shapes

```
Site 0:     A[0]  shape: (d, χ_0)           - left boundary
Site i:     A[i]  shape: (χ_{i-1}, d, χ_i)  - bulk
Site n-1:   A[n-1] shape: (χ_{n-2}, d)      - right boundary
```

### Creation

```c
mps_t* mps_create(uint32_t num_sites, uint32_t local_dim, uint32_t bond_dim) {
    mps_t* mps = malloc(sizeof(mps_t));

    mps->num_sites = num_sites;
    mps->local_dim = local_dim;
    mps->max_bond_dim = bond_dim;
    mps->bond_dims = calloc(num_sites - 1, sizeof(uint32_t));
    mps->tensors = malloc(num_sites * sizeof(tensor_t*));

    // Initialize to |0...0⟩ product state
    for (uint32_t i = 0; i < num_sites; i++) {
        uint32_t left_dim = (i == 0) ? 1 : mps->bond_dims[i-1];
        uint32_t right_dim = (i == num_sites - 1) ? 1 : 1;

        uint32_t dims[3] = {left_dim, local_dim, right_dim};
        mps->tensors[i] = tensor_create(3, dims);

        // Set to |0⟩
        tensor_zero(mps->tensors[i]);
        uint32_t idx[3] = {0, 0, 0};
        tensor_set(mps->tensors[i], idx, (Complex){1.0, 0.0});

        if (i < num_sites - 1) {
            mps->bond_dims[i] = 1;
        }
    }

    mps->canonical_form = CANONICAL_LEFT;
    mps->canonical_center = 0;

    return mps;
}
```

## Tensor Contraction

### Two-Tensor Contraction

```c
tensor_t* tensor_contract(tensor_t* A, tensor_t* B,
                          uint32_t* contract_A, uint32_t* contract_B,
                          uint32_t num_contract) {
    // Determine output shape
    uint32_t out_rank = A->rank + B->rank - 2 * num_contract;
    uint32_t* out_dims = malloc(out_rank * sizeof(uint32_t));

    // ... compute output dimensions ...

    tensor_t* C = tensor_create(out_rank, out_dims);

    // Use optimized BLAS routines when possible
    if (can_use_gemm(A, B, contract_A, contract_B, num_contract)) {
        contract_gemm(A, B, C, contract_A, contract_B, num_contract);
    } else {
        contract_general(A, B, C, contract_A, contract_B, num_contract);
    }

    free(out_dims);
    return C;
}
```

### GEMM-based Contraction

```c
void contract_gemm(tensor_t* A, tensor_t* B, tensor_t* C,
                   uint32_t* contract_A, uint32_t* contract_B,
                   uint32_t num_contract) {
    // Reshape A to (m, k) matrix
    // Reshape B to (k, n) matrix
    // Result is (m, n) matrix

    uint64_t m = compute_free_dim(A, contract_A, num_contract);
    uint64_t k = compute_contract_dim(A, contract_A, num_contract);
    uint64_t n = compute_free_dim(B, contract_B, num_contract);

    // Permute tensors if needed for contiguous memory
    Complex* A_data = prepare_for_gemm(A, contract_A, num_contract, true);
    Complex* B_data = prepare_for_gemm(B, contract_B, num_contract, false);

    // ZGEMM: C = alpha * A * B + beta * C
    Complex alpha = {1.0, 0.0};
    Complex beta = {0.0, 0.0};

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, &alpha, A_data, m, B_data, k, &beta, C->data, m);
}
```

## Singular Value Decomposition

### SVD Interface

```c
typedef struct {
    tensor_t* U;         // Left singular vectors
    double* S;           // Singular values (real)
    tensor_t* Vh;        // Right singular vectors (conjugate transpose)
    uint32_t rank;       // Number of singular values
} svd_result_t;

svd_result_t* tensor_svd(tensor_t* T,
                         uint32_t* left_indices, uint32_t num_left,
                         uint32_t* right_indices, uint32_t num_right) {
    // Reshape to matrix
    uint64_t m = 1, n = 1;
    for (uint32_t i = 0; i < num_left; i++) m *= T->dims[left_indices[i]];
    for (uint32_t i = 0; i < num_right; i++) n *= T->dims[right_indices[i]];

    Complex* matrix = reshape_to_matrix(T, left_indices, num_left,
                                        right_indices, num_right);

    // Call LAPACK zgesvd
    uint64_t k = MIN(m, n);
    double* S = malloc(k * sizeof(double));
    Complex* U = malloc(m * k * sizeof(Complex));
    Complex* Vh = malloc(k * n * sizeof(Complex));

    lapack_zgesvd('S', 'S', m, n, matrix, m, S, U, m, Vh, k);

    // Package result
    svd_result_t* result = malloc(sizeof(svd_result_t));
    result->S = S;
    result->rank = k;

    // Reshape U and Vh back to tensors
    // ...

    free(matrix);
    return result;
}
```

### Truncated SVD

```c
svd_result_t* tensor_svd_truncated(tensor_t* T,
                                    uint32_t* left_indices, uint32_t num_left,
                                    uint32_t* right_indices, uint32_t num_right,
                                    uint32_t max_rank,
                                    double cutoff) {
    svd_result_t* full = tensor_svd(T, left_indices, num_left,
                                    right_indices, num_right);

    // Determine truncation point
    uint32_t keep = full->rank;

    // Truncate by cutoff
    if (cutoff > 0) {
        double max_sv = full->S[0];
        for (uint32_t i = 0; i < full->rank; i++) {
            if (full->S[i] / max_sv < cutoff) {
                keep = i;
                break;
            }
        }
    }

    // Truncate by max rank
    if (max_rank > 0 && keep > max_rank) {
        keep = max_rank;
    }

    // Truncate U, S, Vh
    truncate_svd_result(full, keep);

    return full;
}
```

## Canonical Forms

### Left Canonicalization

Make all tensors left-canonical (A†A = I):

```c
void mps_canonicalize_left(mps_t* mps) {
    for (uint32_t i = 0; i < mps->num_sites - 1; i++) {
        // SVD: A[i] = U * S * V†
        svd_result_t* svd = tensor_svd_site(mps->tensors[i], 'L');

        // A[i] ← U (left-canonical)
        tensor_destroy(mps->tensors[i]);
        mps->tensors[i] = svd->U;

        // A[i+1] ← S * V† * A[i+1]
        tensor_t* SV = tensor_contract_sv(svd->S, svd->Vh);
        tensor_t* new_next = tensor_contract(SV, mps->tensors[i+1], ...);

        tensor_destroy(mps->tensors[i+1]);
        mps->tensors[i+1] = new_next;

        // Update bond dimension
        mps->bond_dims[i] = svd->rank;

        tensor_destroy(SV);
        svd_destroy(svd);
    }

    mps->canonical_form = CANONICAL_LEFT;
}
```

### Right Canonicalization

```c
void mps_canonicalize_right(mps_t* mps) {
    for (int32_t i = mps->num_sites - 1; i > 0; i--) {
        // SVD: A[i] = U * S * V†
        svd_result_t* svd = tensor_svd_site(mps->tensors[i], 'R');

        // A[i] ← V† (right-canonical)
        tensor_destroy(mps->tensors[i]);
        mps->tensors[i] = svd->Vh;

        // A[i-1] ← A[i-1] * U * S
        tensor_t* US = tensor_contract_us(svd->U, svd->S);
        tensor_t* new_prev = tensor_contract(mps->tensors[i-1], US, ...);

        tensor_destroy(mps->tensors[i-1]);
        mps->tensors[i-1] = new_prev;

        mps->bond_dims[i-1] = svd->rank;

        tensor_destroy(US);
        svd_destroy(svd);
    }

    mps->canonical_form = CANONICAL_RIGHT;
}
```

### Mixed Canonical Form

```c
void mps_canonicalize_mixed(mps_t* mps, uint32_t center) {
    // Left-canonicalize sites 0 to center-1
    for (uint32_t i = 0; i < center; i++) {
        canonicalize_site_left(mps, i);
    }

    // Right-canonicalize sites center+1 to n-1
    for (int32_t i = mps->num_sites - 1; i > (int32_t)center; i--) {
        canonicalize_site_right(mps, i);
    }

    mps->canonical_form = CANONICAL_MIXED;
    mps->canonical_center = center;
}
```

## DMRG Implementation

### DMRG Sweep

```c
typedef struct {
    mps_t* state;
    mpo_t* hamiltonian;
    tensor_t** left_envs;   // Left environment tensors
    tensor_t** right_envs;  // Right environment tensors
    uint32_t max_bond_dim;
    double tolerance;
} dmrg_context_t;

double dmrg_sweep(dmrg_context_t* ctx, bool direction_right) {
    double energy = 0;

    if (direction_right) {
        for (uint32_t i = 0; i < ctx->state->num_sites - 1; i++) {
            energy = dmrg_optimize_two_site(ctx, i);
            dmrg_move_center_right(ctx, i);
        }
    } else {
        for (int32_t i = ctx->state->num_sites - 2; i >= 0; i--) {
            energy = dmrg_optimize_two_site(ctx, i);
            dmrg_move_center_left(ctx, i);
        }
    }

    return energy;
}
```

### Two-Site Optimization

```c
double dmrg_optimize_two_site(dmrg_context_t* ctx, uint32_t site) {
    // Merge two site tensors
    tensor_t* theta = tensor_contract(ctx->state->tensors[site],
                                      ctx->state->tensors[site + 1],
                                      ...);

    // Form effective Hamiltonian
    effective_h_t* H_eff = form_effective_hamiltonian(
        ctx->left_envs[site],
        ctx->hamiltonian->tensors[site],
        ctx->hamiltonian->tensors[site + 1],
        ctx->right_envs[site + 1]
    );

    // Solve eigenvalue problem: H_eff |θ⟩ = E |θ⟩
    double energy = lanczos_ground_state(H_eff, theta);

    // SVD and truncate
    svd_result_t* svd = tensor_svd_truncated(
        theta, ...,
        ctx->max_bond_dim,
        ctx->tolerance
    );

    // Update MPS tensors
    tensor_destroy(ctx->state->tensors[site]);
    tensor_destroy(ctx->state->tensors[site + 1]);
    ctx->state->tensors[site] = svd->U;
    ctx->state->tensors[site + 1] = form_right_tensor(svd->S, svd->Vh);

    // Update bond dimension
    ctx->state->bond_dims[site] = svd->rank;

    tensor_destroy(theta);
    effective_h_destroy(H_eff);
    svd_destroy(svd);

    return energy;
}
```

### Lanczos Eigensolver

```c
double lanczos_ground_state(effective_h_t* H, tensor_t* initial) {
    const int max_iter = 100;
    const double tol = 1e-12;

    // Lanczos vectors
    tensor_t** V = malloc((max_iter + 1) * sizeof(tensor_t*));
    double* alpha = calloc(max_iter, sizeof(double));
    double* beta = calloc(max_iter, sizeof(double));

    // Initial vector
    V[0] = tensor_copy(initial);
    tensor_normalize(V[0]);

    for (int j = 0; j < max_iter; j++) {
        // w = H * v_j
        tensor_t* w = apply_effective_h(H, V[j]);

        // Orthogonalize
        alpha[j] = tensor_inner_product_real(V[j], w);
        tensor_axpy(w, -alpha[j], V[j]);

        if (j > 0) {
            tensor_axpy(w, -beta[j-1], V[j-1]);
        }

        beta[j] = tensor_norm(w);

        // Check convergence
        if (beta[j] < tol) break;

        V[j+1] = tensor_copy(w);
        tensor_scale(V[j+1], 1.0 / beta[j]);

        tensor_destroy(w);
    }

    // Diagonalize tridiagonal matrix
    double energy = diagonalize_tridiagonal(alpha, beta, j);

    // Reconstruct ground state
    reconstruct_ground_state(initial, V, j, alpha, beta);

    // Cleanup
    for (int i = 0; i <= j; i++) tensor_destroy(V[i]);
    free(V); free(alpha); free(beta);

    return energy;
}
```

## Environment Tensors

### Left Environment

```c
void dmrg_update_left_env(dmrg_context_t* ctx, uint32_t site) {
    // L[i+1] = L[i] * A[i]† * W[i] * A[i]
    //
    //     L[i]──A[i]──
    //      │     │
    //      ├──W[i]──
    //      │     │
    //     L[i]──A[i]*─
    //

    tensor_t* A = ctx->state->tensors[site];
    tensor_t* W = ctx->hamiltonian->tensors[site];
    tensor_t* L = ctx->left_envs[site];

    // Contract L with A
    tensor_t* temp1 = tensor_contract(L, A, ...);

    // Contract with W
    tensor_t* temp2 = tensor_contract(temp1, W, ...);

    // Contract with A†
    tensor_t* A_dag = tensor_conj(A);
    ctx->left_envs[site + 1] = tensor_contract(temp2, A_dag, ...);

    tensor_destroy(temp1);
    tensor_destroy(temp2);
    tensor_destroy(A_dag);
}
```

### Right Environment

```c
void dmrg_update_right_env(dmrg_context_t* ctx, uint32_t site) {
    // R[i-1] = A[i] * W[i] * A[i]† * R[i]

    tensor_t* A = ctx->state->tensors[site];
    tensor_t* W = ctx->hamiltonian->tensors[site];
    tensor_t* R = ctx->right_envs[site];

    // Similar contraction sequence
    tensor_t* temp1 = tensor_contract(A, R, ...);
    tensor_t* temp2 = tensor_contract(temp1, W, ...);
    tensor_t* A_dag = tensor_conj(A);
    ctx->right_envs[site - 1] = tensor_contract(temp2, A_dag, ...);

    tensor_destroy(temp1);
    tensor_destroy(temp2);
    tensor_destroy(A_dag);
}
```

## Performance Optimization

### Tensor Caching

```c
typedef struct {
    tensor_t** cache;
    uint64_t* keys;
    size_t capacity;
    size_t size;
} tensor_cache_t;

tensor_t* cache_get_or_compute(tensor_cache_t* cache,
                                uint64_t key,
                                tensor_t* (*compute)(void*),
                                void* data) {
    // Check cache
    for (size_t i = 0; i < cache->size; i++) {
        if (cache->keys[i] == key) {
            return tensor_copy(cache->cache[i]);
        }
    }

    // Compute and cache
    tensor_t* result = compute(data);

    if (cache->size < cache->capacity) {
        cache->cache[cache->size] = tensor_copy(result);
        cache->keys[cache->size] = key;
        cache->size++;
    }

    return result;
}
```

### Parallel Contraction

```c
tensor_t* tensor_contract_parallel(tensor_t* A, tensor_t* B, ...) {
    tensor_t* C = tensor_create(...);

    #pragma omp parallel for
    for (uint64_t i = 0; i < C->total_size; i++) {
        // Compute output element C[i]
        Complex sum = {0, 0};

        // Sum over contracted indices
        for (uint64_t j = 0; j < contract_dim; j++) {
            // ...
        }

        C->data[i] = sum;
    }

    return C;
}
```

## Complexity Analysis

| Operation | Time | Memory |
|-----------|------|--------|
| MPS contraction | O(n d χ²) | O(d χ²) |
| SVD (truncated) | O(d χ³) | O(d χ²) |
| DMRG two-site | O(d² χ³ D) | O(d² χ² D) |
| Full DMRG sweep | O(n d² χ³ D) | O(n d χ²) |

Where:
- n = number of sites
- d = local dimension (2 for qubits)
- χ = bond dimension
- D = MPO bond dimension

## See Also

- [DMRG Algorithm](../algorithms/dmrg-algorithm.md) - Algorithm details
- [Tensor Networks Concepts](../concepts/tensor-networks.md) - Theory
- [Tutorial: Tensor Networks](../tutorials/08-tensor-network-simulation.md) - Usage
- [C API: Tensor Network](../api/c/tensor-network.md) - API reference

