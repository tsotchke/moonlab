/**
 * @file tensor_kernels.metal
 * @brief Metal GPU kernels for MPS/tensor network operations
 *
 * High-performance kernels for Matrix Product State simulation:
 * - Tensor contraction for 2-site operations
 * - Gate application to theta tensor
 * - Jacobi SVD for bond truncation
 * - Transfer matrix for expectation values
 *
 * All operations use single precision (float) for Metal compatibility.
 * Optimized for Apple Silicon M1/M2/M3/M4 unified memory architecture.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// SINGLE PRECISION COMPLEX ARITHMETIC
// ============================================================================

// Use float2 for complex numbers (real, imaginary)
typedef float2 complex_t;

/**
 * Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 */
inline complex_t cmul(complex_t a, complex_t b) {
    return complex_t(
        a.x * b.x - a.y * b.y,  // real part
        a.x * b.y + a.y * b.x   // imaginary part
    );
}

/**
 * Complex addition
 */
inline complex_t cadd(complex_t a, complex_t b) {
    return a + b;
}

/**
 * Complex subtraction
 */
inline complex_t csub(complex_t a, complex_t b) {
    return a - b;
}

/**
 * Complex conjugate: (a+bi)* = a-bi
 */
inline complex_t cconj(complex_t z) {
    return complex_t(z.x, -z.y);
}

/**
 * Complex magnitude squared: |a+bi|^2 = a^2 + b^2
 */
inline float cabs2(complex_t z) {
    return z.x * z.x + z.y * z.y;
}

/**
 * Complex magnitude: |a+bi| = sqrt(a^2 + b^2)
 */
inline float cabs(complex_t z) {
    return sqrt(cabs2(z));
}

/**
 * Scalar multiplication of complex number
 */
inline complex_t cscale(complex_t z, float s) {
    return complex_t(z.x * s, z.y * s);
}

/**
 * Complex negation
 */
inline complex_t cneg(complex_t z) {
    return complex_t(-z.x, -z.y);
}

/**
 * Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
 */
inline complex_t cdiv(complex_t a, complex_t b) {
    float denom = cabs2(b);
    if (denom < 1e-30f) {
        return complex_t(0.0f, 0.0f);  // Avoid division by zero
    }
    return complex_t(
        (a.x * b.x + a.y * b.y) / denom,
        (a.y * b.x - a.x * b.y) / denom
    );
}

// ============================================================================
// TENSOR CONTRACTION - 2-Site MPS Operation
// ============================================================================

/**
 * Contract two adjacent MPS tensors into theta tensor
 *
 * theta_{l,p1,p2,r} = sum_m A_{l,p1,m} * B_{m,p2,r}
 *
 * Input tensor layouts (row-major):
 *   A: [chi_l][2][chi_m]   - left MPS tensor
 *   B: [chi_m][2][chi_r]   - right MPS tensor
 *
 * Output tensor layout:
 *   theta: [chi_l][2][2][chi_r] = [chi_l][4][chi_r]
 *
 * Parallelism: Each (l, p1p2, r) output element is independent
 * Threads: chi_l * 4 * chi_r total
 *
 * Performance: O(chi_l * 4 * chi_m * chi_r) FLOPs
 *              Memory bandwidth limited for chi > 64
 */
kernel void tensor_contract_2site(
    device const complex_t* A [[buffer(0)]],      // [chi_l][2][chi_m]
    device const complex_t* B [[buffer(1)]],      // [chi_m][2][chi_r]
    device complex_t* theta [[buffer(2)]],        // [chi_l][4][chi_r]
    constant uint& chi_l [[buffer(3)]],            // Left bond dimension
    constant uint& chi_m [[buffer(4)]],            // Middle bond (contracted)
    constant uint& chi_r [[buffer(5)]],            // Right bond dimension
    uint3 tid [[thread_position_in_grid]]          // (l, p1p2, r)
) {
    uint l = tid.x;      // Left bond index
    uint p1p2 = tid.y;   // Combined physical index (0-3)
    uint r = tid.z;      // Right bond index

    if (l >= chi_l || p1p2 >= 4 || r >= chi_r) return;

    uint p1 = p1p2 / 2;  // Physical index 1 (0 or 1)
    uint p2 = p1p2 % 2;  // Physical index 2 (0 or 1)

    // Strides for tensor indexing
    uint A_stride_l = 2 * chi_m;          // A[l][p1][m] stride for l
    uint A_stride_p = chi_m;              // A[l][p1][m] stride for p1
    uint B_stride_m = 2 * chi_r;          // B[m][p2][r] stride for m
    uint B_stride_p = chi_r;              // B[m][p2][r] stride for p2
    uint theta_stride_l = 4 * chi_r;      // theta[l][p1p2][r] stride for l
    uint theta_stride_p = chi_r;          // theta[l][p1p2][r] stride for p1p2

    // Contract over m index
    complex_t sum = complex_t(0.0f, 0.0f);

    for (uint m = 0; m < chi_m; m++) {
        uint A_idx = l * A_stride_l + p1 * A_stride_p + m;
        uint B_idx = m * B_stride_m + p2 * B_stride_p + r;
        sum = cadd(sum, cmul(A[A_idx], B[B_idx]));
    }

    // Write output
    uint theta_idx = l * theta_stride_l + p1p2 * theta_stride_p + r;
    theta[theta_idx] = sum;
}

// ============================================================================
// GATE APPLICATION - Fused 4x4 Gate to Theta
// ============================================================================

/**
 * Apply 4x4 gate matrix to theta tensor (in-place)
 *
 * theta'_{l,p',r} = sum_{p} G_{p',p} * theta_{l,p,r}
 *
 * where p, p' are combined physical indices (0-3)
 *
 * Input/Output tensor layout:
 *   theta: [chi_l][4][chi_r]
 *
 * Gate layout:
 *   G: [4][4] - row-major 4x4 unitary
 *
 * Parallelism: Each (l, r) pair processes all 4 physical indices
 * Threads: chi_l * chi_r
 */
kernel void apply_gate_theta(
    device complex_t* theta [[buffer(0)]],        // [chi_l][4][chi_r] in-place
    device const complex_t* gate [[buffer(1)]],   // [4][4] gate matrix
    constant uint& chi_l [[buffer(2)]],
    constant uint& chi_r [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]          // (l, r)
) {
    uint l = tid.x;
    uint r = tid.y;

    if (l >= chi_l || r >= chi_r) return;

    uint theta_stride_l = 4 * chi_r;
    uint theta_stride_p = chi_r;
    uint base_idx = l * theta_stride_l + r;

    // Load 4 theta values for this (l, r)
    complex_t theta_old[4];
    for (uint p = 0; p < 4; p++) {
        theta_old[p] = theta[base_idx + p * theta_stride_p];
    }

    // Apply gate: theta_new[p'] = sum_p G[p'][p] * theta_old[p]
    for (uint pp = 0; pp < 4; pp++) {
        complex_t sum = complex_t(0.0f, 0.0f);
        for (uint p = 0; p < 4; p++) {
            sum = cadd(sum, cmul(gate[pp * 4 + p], theta_old[p]));
        }
        theta[base_idx + pp * theta_stride_p] = sum;
    }
}

// ============================================================================
// JACOBI SVD - GPU-Friendly SVD via One-Sided Jacobi
// ============================================================================

/**
 * Compute column norms for SVD initialization
 *
 * Computes ||A[:,j]||^2 for each column j
 */
kernel void compute_column_norms(
    device const complex_t* A [[buffer(0)]],      // [m][n] matrix
    device float* norms [[buffer(1)]],            // [n] output norms squared
    constant uint& m [[buffer(2)]],                // Rows
    constant uint& n [[buffer(3)]],                // Columns
    uint tid [[thread_position_in_grid]]           // Column index
) {
    if (tid >= n) return;

    float sum = 0.0f;
    for (uint i = 0; i < m; i++) {
        sum += cabs2(A[i * n + tid]);
    }
    norms[tid] = sum;
}

/**
 * One Jacobi rotation sweep for SVD
 *
 * For each column pair (i, j), computes and applies Givens rotation
 * to orthogonalize columns of A and accumulate V.
 *
 * A' = A * J     (J is Jacobi rotation)
 * V' = V * J
 *
 * Convergence: ||off-diag(A^H A)|| decreases each sweep
 * Typically converges in O(log(n)) sweeps for well-conditioned matrices
 *
 * Thread assignment: One thread per column pair in parallel
 * Note: Must synchronize between pairs to avoid race conditions
 */
kernel void jacobi_svd_rotation(
    device complex_t* A [[buffer(0)]],            // [m][n] matrix (modified)
    device complex_t* V [[buffer(1)]],            // [n][n] accumulator (modified)
    constant uint& m [[buffer(2)]],                // Rows
    constant uint& n [[buffer(3)]],                // Columns
    constant uint& col_i [[buffer(4)]],            // First column of pair
    constant uint& col_j [[buffer(5)]],            // Second column of pair
    uint tid [[thread_position_in_grid]]           // Row index for parallel update
) {
    // This kernel applies rotation to rows tid for columns (col_i, col_j)
    // Compute A^H A entries for rotation angle
    // a = A[:,i]^H * A[:,i]
    // b = A[:,j]^H * A[:,j]
    // c = A[:,i]^H * A[:,j]

    // Use threadgroup reduction for dot products
    threadgroup float a_shared;
    threadgroup float b_shared;
    threadgroup complex_t c_shared;

    if (tid == 0) {
        a_shared = 0.0f;
        b_shared = 0.0f;
        c_shared = complex_t(0.0f, 0.0f);

        for (uint k = 0; k < m; k++) {
            complex_t ai = A[k * n + col_i];
            complex_t aj = A[k * n + col_j];
            a_shared += cabs2(ai);
            b_shared += cabs2(aj);
            c_shared = cadd(c_shared, cmul(cconj(ai), aj));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float a = a_shared;
    float b = b_shared;
    complex_t c = c_shared;

    // Skip if columns already orthogonal
    if (cabs2(c) < 1e-15f * a * b) return;

    // Compute Jacobi rotation angle
    // tan(2*theta) = 2*|c| / (a - b)
    float c_mag = cabs(c);
    float tau = (b - a) / (2.0f * c_mag);
    float t = (tau >= 0.0f) ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                            : 1.0f / (tau - sqrt(1.0f + tau * tau));
    float cos_theta = 1.0f / sqrt(1.0f + t * t);
    float sin_theta = t * cos_theta;

    // Phase factor: c = |c| * e^{i*phi}
    complex_t phase = cdiv(c, complex_t(c_mag, 0.0f));
    complex_t sin_phase = cmul(complex_t(sin_theta, 0.0f), cconj(phase));

    // Apply rotation to each row
    if (tid < m) {
        complex_t ai = A[tid * n + col_i];
        complex_t aj = A[tid * n + col_j];

        A[tid * n + col_i] = cadd(cscale(ai, cos_theta),
                                   cmul(aj, sin_phase));
        A[tid * n + col_j] = csub(cscale(aj, cos_theta),
                                   cmul(ai, cconj(sin_phase)));
    }

    // Apply rotation to V rows
    if (tid < n) {
        complex_t vi = V[tid * n + col_i];
        complex_t vj = V[tid * n + col_j];

        V[tid * n + col_i] = cadd(cscale(vi, cos_theta),
                                   cmul(vj, sin_phase));
        V[tid * n + col_j] = csub(cscale(vj, cos_theta),
                                   cmul(vi, cconj(sin_phase)));
    }
}

/**
 * Extract singular values from orthogonalized A
 *
 * After Jacobi iterations, A has orthogonal columns
 * sigma_j = ||A[:,j]||
 */
kernel void extract_singular_values(
    device const complex_t* A [[buffer(0)]],      // [m][n] orthogonalized
    device float* sigma [[buffer(1)]],            // [n] singular values
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]           // Column index
) {
    if (tid >= n) return;

    float sum = 0.0f;
    for (uint i = 0; i < m; i++) {
        sum += cabs2(A[i * n + tid]);
    }
    sigma[tid] = sqrt(sum);
}

/**
 * Normalize U columns and apply truncation
 *
 * U[:,j] = A[:,j] / sigma[j]  for j < rank
 */
kernel void normalize_and_truncate_U(
    device const complex_t* A [[buffer(0)]],      // [m][n] orthogonalized
    device complex_t* U [[buffer(1)]],            // [m][rank] output
    device const float* sigma [[buffer(2)]],      // [n] singular values
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& rank [[buffer(5)]],             // Truncation rank
    uint2 tid [[thread_position_in_grid]]          // (row, col)
) {
    uint i = tid.x;  // Row
    uint j = tid.y;  // Column

    if (i >= m || j >= rank) return;

    float s = sigma[j];
    if (s < 1e-7f) {
        U[i * rank + j] = complex_t(0.0f, 0.0f);
    } else {
        U[i * rank + j] = cscale(A[i * n + j], 1.0f / s);
    }
}

/**
 * Truncate V matrix
 *
 * V_trunc = V[:, :rank]
 */
kernel void truncate_V(
    device const complex_t* V [[buffer(0)]],      // [n][n] full V
    device complex_t* V_trunc [[buffer(1)]],      // [n][rank] truncated
    constant uint& n [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]          // (row, col)
) {
    uint i = tid.x;  // Row
    uint j = tid.y;  // Column

    if (i >= n || j >= rank) return;

    V_trunc[i * rank + j] = V[i * n + j];
}

// ============================================================================
// TRANSFER MATRIX - For Expectation Values
// ============================================================================

/**
 * Compute transfer matrix for Z expectation value
 *
 * T_{l',l} = sum_p Z_p * conj(A_{l',p,r}) * A_{l,p,r}
 *
 * where Z_p = +1 for p=0, -1 for p=1 (Pauli Z eigenvalues)
 *
 * This computes the transfer matrix contribution from one MPS site
 * for measuring <Z> at that site.
 *
 * Input:
 *   A: [chi_l][2][chi_r] MPS tensor at measurement site
 *
 * Output:
 *   transfer: [chi_l][chi_l] transfer matrix
 *
 * Note: For canonicalized MPS, chi_l appears on both bra and ket sides
 */
kernel void transfer_matrix_z_single(
    device const complex_t* A [[buffer(0)]],      // [chi_l][2][chi_r]
    device complex_t* transfer [[buffer(1)]],     // [chi_l][chi_l]
    constant uint& chi_l [[buffer(2)]],
    constant uint& chi_r [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]          // (l', l)
) {
    uint lp = tid.x;  // Bra index
    uint l = tid.y;   // Ket index

    if (lp >= chi_l || l >= chi_l) return;

    uint A_stride_l = 2 * chi_r;
    uint A_stride_p = chi_r;

    complex_t sum = complex_t(0.0f, 0.0f);

    // Sum over physical index p and right bond r
    for (uint p = 0; p < 2; p++) {
        float z_val = (p == 0) ? 1.0f : -1.0f;  // Z eigenvalue

        for (uint r = 0; r < chi_r; r++) {
            complex_t a_bra = A[lp * A_stride_l + p * A_stride_p + r];
            complex_t a_ket = A[l * A_stride_l + p * A_stride_p + r];
            sum = cadd(sum, cscale(cmul(cconj(a_bra), a_ket), z_val));
        }
    }

    transfer[lp * chi_l + l] = sum;
}

/**
 * Compute identity transfer matrix (for sites not being measured)
 *
 * T_{l',l} = sum_p conj(A_{l',p,r}) * A_{l,p,r}
 *
 * This is the standard transfer matrix without operator insertion.
 */
kernel void transfer_matrix_identity(
    device const complex_t* A [[buffer(0)]],      // [chi_l][2][chi_r]
    device complex_t* transfer [[buffer(1)]],     // [chi_l][chi_l]
    constant uint& chi_l [[buffer(2)]],
    constant uint& chi_r [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]          // (l', l)
) {
    uint lp = tid.x;
    uint l = tid.y;

    if (lp >= chi_l || l >= chi_l) return;

    uint A_stride_l = 2 * chi_r;
    uint A_stride_p = chi_r;

    complex_t sum = complex_t(0.0f, 0.0f);

    for (uint p = 0; p < 2; p++) {
        for (uint r = 0; r < chi_r; r++) {
            complex_t a_bra = A[lp * A_stride_l + p * A_stride_p + r];
            complex_t a_ket = A[l * A_stride_l + p * A_stride_p + r];
            sum = cadd(sum, cmul(cconj(a_bra), a_ket));
        }
    }

    transfer[lp * chi_l + l] = sum;
}

/**
 * Contract two transfer matrices: T_out = T_left * T_right
 *
 * T_out_{l',l} = sum_m T_left_{l',m} * T_right_{m,l}
 *
 * Used to chain transfer matrices from multiple sites.
 */
kernel void contract_transfer_matrices(
    device const complex_t* T_left [[buffer(0)]],  // [chi][chi]
    device const complex_t* T_right [[buffer(1)]], // [chi][chi]
    device complex_t* T_out [[buffer(2)]],         // [chi][chi]
    constant uint& chi [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]           // (row, col)
) {
    uint i = tid.x;  // Row
    uint j = tid.y;  // Column

    if (i >= chi || j >= chi) return;

    complex_t sum = complex_t(0.0f, 0.0f);
    for (uint k = 0; k < chi; k++) {
        sum = cadd(sum, cmul(T_left[i * chi + k], T_right[k * chi + j]));
    }

    T_out[i * chi + j] = sum;
}

/**
 * Extract trace of transfer matrix (final step of expectation value)
 *
 * result = sum_i T[i][i]
 *
 * For normalized MPS, this gives the expectation value.
 */
kernel void transfer_matrix_trace(
    device const complex_t* T [[buffer(0)]],      // [chi][chi]
    device complex_t* result [[buffer(1)]],       // Single value output
    constant uint& chi [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Parallel reduction for trace
    threadgroup complex_t partial_sums[256];

    complex_t local_sum = complex_t(0.0f, 0.0f);

    // Each thread sums a subset of diagonal elements
    for (uint i = tid; i < chi; i += 256) {
        local_sum = cadd(local_sum, T[i * chi + i]);
    }

    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < 256) {
            partial_sums[tid] = cadd(partial_sums[tid], partial_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        result[0] = partial_sums[0];
    }
}

// ============================================================================
// MPS NORMALIZATION
// ============================================================================

/**
 * Compute squared norm of MPS tensor
 *
 * ||A||^2 = sum_{l,p,r} |A_{l,p,r}|^2
 */
kernel void tensor_norm_squared(
    device const complex_t* A [[buffer(0)]],
    device float* norm_sq [[buffer(1)]],
    constant uint& total_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    threadgroup float partial_sums[256];

    float local_sum = 0.0f;
    for (uint i = tid; i < total_size; i += 256) {
        local_sum += cabs2(A[i]);
    }

    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < 256) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        norm_sq[0] = partial_sums[0];
    }
}

/**
 * Scale tensor by constant factor
 *
 * A *= scale
 */
kernel void tensor_scale(
    device complex_t* A [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& total_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_size) return;
    A[tid] = cscale(A[tid], scale);
}
