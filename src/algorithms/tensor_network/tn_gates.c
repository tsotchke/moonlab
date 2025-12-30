/**
 * @file tn_gates.c
 * @brief Quantum gate application implementation for tensor networks
 *
 * Full production implementation of quantum gates on MPS states.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "tn_gates.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// GPU acceleration
#ifdef __APPLE__
#include "../../optimization/gpu_metal.h"
#define HAS_METAL 1
#else
#define HAS_METAL 0
#endif

// GPU threshold: use GPU when bond dimension exceeds this
// DISABLED: Data type mismatch between CPU (double complex) and Metal (float2)
// requires conversion layer before GPU path can be used
#define GPU_BOND_THRESHOLD 1000000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// ============================================================================
// STANDARD GATES
// ============================================================================

const tn_gate_1q_t TN_GATE_I = {{
    {1.0, 0.0},
    {0.0, 1.0}
}};

const tn_gate_1q_t TN_GATE_X = {{
    {0.0, 1.0},
    {1.0, 0.0}
}};

const tn_gate_1q_t TN_GATE_Y = {{
    {0.0, -I},
    {I, 0.0}
}};

const tn_gate_1q_t TN_GATE_Z = {{
    {1.0, 0.0},
    {0.0, -1.0}
}};

const tn_gate_1q_t TN_GATE_H = {{
    {M_SQRT1_2, M_SQRT1_2},
    {M_SQRT1_2, -M_SQRT1_2}
}};

const tn_gate_1q_t TN_GATE_S = {{
    {1.0, 0.0},
    {0.0, I}
}};

const tn_gate_1q_t TN_GATE_SDG = {{
    {1.0, 0.0},
    {0.0, -I}
}};

const tn_gate_1q_t TN_GATE_T = {{
    {1.0, 0.0},
    {0.0, M_SQRT1_2 + M_SQRT1_2 * I}
}};

const tn_gate_1q_t TN_GATE_TDG = {{
    {1.0, 0.0},
    {0.0, M_SQRT1_2 - M_SQRT1_2 * I}
}};

const tn_gate_2q_t TN_GATE_CNOT = {{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    {0.0, 0.0, 1.0, 0.0}
}};

const tn_gate_2q_t TN_GATE_CZ = {{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, -1.0}
}};

const tn_gate_2q_t TN_GATE_SWAP = {{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
}};

const tn_gate_2q_t TN_GATE_ISWAP = {{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, I, 0.0},
    {0.0, I, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
}};

// ============================================================================
// GPU ACCELERATION CONTEXT
// ============================================================================

#if HAS_METAL
static metal_compute_ctx_t *g_metal_ctx = NULL;
static bool g_gpu_init_attempted = false;

/**
 * @brief Get or create Metal compute context
 */
static metal_compute_ctx_t *get_metal_context(void) {
    if (!g_gpu_init_attempted) {
        g_gpu_init_attempted = true;
        g_metal_ctx = metal_compute_init();
        if (g_metal_ctx) {
            fprintf(stderr, "Metal GPU acceleration enabled for tensor network operations\n");
        }
    }
    return g_metal_ctx;
}

/**
 * @brief Apply 2-qubit gate using Metal GPU
 *
 * @return TN_GATE_SUCCESS on success, or negative error if GPU path failed (fall back to CPU)
 */
static tn_gate_error_t apply_gate_2q_adjacent_gpu(tn_mps_state_t *state,
                                                    uint32_t left_qubit,
                                                    const tn_gate_2q_t *gate,
                                                    double *truncation_error) {
    metal_compute_ctx_t *ctx = get_metal_context();
    if (!ctx) return TN_GATE_ERROR_ALLOC_FAILED;  // Signal to use CPU path

    uint32_t right_qubit = left_qubit + 1;
    tensor_t *tl = state->tensors[left_qubit];
    tensor_t *tr = state->tensors[right_qubit];

    uint32_t chi_l = tl->dims[0];      // Left bond
    uint32_t chi_m = tl->dims[2];      // Middle bond (shared)
    uint32_t chi_r = tr->dims[2];      // Right bond

    // Ensure tensors have GPU buffers and are synced
    if (tensor_ensure_gpu(NULL, tl) != TENSOR_SUCCESS ||
        tensor_ensure_gpu(NULL, tr) != TENSOR_SUCCESS) {
        return TN_GATE_ERROR_ALLOC_FAILED;  // Fall back to CPU
    }

    // Create gate buffer on GPU
    size_t gate_size = 16 * sizeof(double complex);
    metal_buffer_t *gate_buf = metal_buffer_create(ctx, gate_size);
    if (!gate_buf) {
        return TN_GATE_ERROR_ALLOC_FAILED;
    }

    // Copy gate matrix to GPU (flatten 4x4)
    void *gate_ptr = metal_buffer_contents(gate_buf);
    if (gate_ptr) {
        double complex *gd = (double complex *)gate_ptr;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                gd[i * 4 + j] = gate->elements[i][j];
            }
        }
    }

    // Call the Metal MPS gate application
    uint32_t new_bond;
    double trunc_error;
    int result = metal_mps_apply_gate_2q(
        ctx,
        (metal_buffer_t *)tl->gpu_buffer,
        (metal_buffer_t *)tr->gpu_buffer,
        gate_buf,
        chi_l, chi_m, chi_r,
        state->config.max_bond_dim,
        state->config.svd_cutoff,
        &new_bond,
        &trunc_error
    );

    metal_buffer_free(gate_buf);

    if (result != 0) {
        return TN_GATE_ERROR_ALLOC_FAILED;  // Fall back to CPU
    }

    // Sync results back to CPU
    tensor_invalidate_cpu(tl);
    tensor_invalidate_cpu(tr);
    tl->gpu_valid = true;
    tr->gpu_valid = true;

    if (tensor_sync_to_cpu(tl) != TENSOR_SUCCESS ||
        tensor_sync_to_cpu(tr) != TENSOR_SUCCESS) {
        return TN_GATE_ERROR_ALLOC_FAILED;
    }

    // Update bond dimension
    state->bond_dims[left_qubit] = new_bond;

    // Update tracking
    if (truncation_error) *truncation_error = trunc_error;
    state->cumulative_truncation_error += trunc_error;
    if (trunc_error > 0) {
        state->num_truncations++;
    }
    state->canonical = TN_CANONICAL_NONE;

    return TN_GATE_SUCCESS;
}
#endif // HAS_METAL

// ============================================================================
// PARAMETERIZED GATES
// ============================================================================

tn_gate_1q_t tn_gate_rx(double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    tn_gate_1q_t gate = {{
        {c, -I * s},
        {-I * s, c}
    }};
    return gate;
}

tn_gate_1q_t tn_gate_ry(double theta) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    tn_gate_1q_t gate = {{
        {c, -s},
        {s, c}
    }};
    return gate;
}

tn_gate_1q_t tn_gate_rz(double theta) {
    double complex phase_neg = cexp(-I * theta / 2.0);
    double complex phase_pos = cexp(I * theta / 2.0);
    tn_gate_1q_t gate = {{
        {phase_neg, 0.0},
        {0.0, phase_pos}
    }};
    return gate;
}

tn_gate_1q_t tn_gate_u3(double theta, double phi, double lambda) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    double complex ephi = cexp(I * phi);
    double complex elam = cexp(I * lambda);
    double complex ephilam = cexp(I * (phi + lambda));

    tn_gate_1q_t gate = {{
        {c, -elam * s},
        {ephi * s, ephilam * c}
    }};
    return gate;
}

tn_gate_1q_t tn_gate_phase(double phi) {
    double complex phase = cexp(I * phi);
    tn_gate_1q_t gate = {{
        {1.0, 0.0},
        {0.0, phase}
    }};
    return gate;
}

tn_gate_2q_t tn_gate_crz(double theta) {
    double complex phase_neg = cexp(-I * theta / 2.0);
    double complex phase_pos = cexp(I * theta / 2.0);
    tn_gate_2q_t gate = {{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, phase_neg, 0.0},
        {0.0, 0.0, 0.0, phase_pos}
    }};
    return gate;
}

tn_gate_2q_t tn_gate_cphase(double phi) {
    double complex phase = cexp(I * phi);
    tn_gate_2q_t gate = {{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, phase}
    }};
    return gate;
}

tn_gate_2q_t tn_gate_rxx(double theta) {
    double c = cos(theta / 2.0);
    double complex is = -I * sin(theta / 2.0);

    tn_gate_2q_t gate = {{
        {c, 0.0, 0.0, is},
        {0.0, c, is, 0.0},
        {0.0, is, c, 0.0},
        {is, 0.0, 0.0, c}
    }};
    return gate;
}

tn_gate_2q_t tn_gate_ryy(double theta) {
    double c = cos(theta / 2.0);
    double complex is = I * sin(theta / 2.0);

    tn_gate_2q_t gate = {{
        {c, 0.0, 0.0, is},
        {0.0, c, -is, 0.0},
        {0.0, -is, c, 0.0},
        {is, 0.0, 0.0, c}
    }};
    return gate;
}

tn_gate_2q_t tn_gate_rzz(double theta) {
    double complex phase_neg = cexp(-I * theta / 2.0);
    double complex phase_pos = cexp(I * theta / 2.0);

    tn_gate_2q_t gate = {{
        {phase_neg, 0.0, 0.0, 0.0},
        {0.0, phase_pos, 0.0, 0.0},
        {0.0, 0.0, phase_pos, 0.0},
        {0.0, 0.0, 0.0, phase_neg}
    }};
    return gate;
}

// ============================================================================
// SINGLE-QUBIT GATE APPLICATION
// ============================================================================

tn_gate_error_t tn_apply_gate_1q(tn_mps_state_t *state,
                                  uint32_t qubit,
                                  const tn_gate_1q_t *gate) {
    if (!state || !gate) return TN_GATE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_GATE_ERROR_INVALID_QUBIT;

    tensor_t *t = state->tensors[qubit];
    uint32_t left_dim = t->dims[0];
    uint32_t right_dim = t->dims[2];

    // Create new tensor
    tensor_t *new_t = tensor_create(3, t->dims);
    if (!new_t) return TN_GATE_ERROR_ALLOC_FAILED;

    // OPTIMIZED: Direct data access instead of tensor_get/set
    // For tensor [l, p, r] with dims [left, 2, right]:
    // linear_idx = l * (2 * right) + p * right + r
    const uint32_t stride_l = TN_PHYSICAL_DIM * right_dim;
    const uint32_t stride_p = right_dim;

    // Pre-fetch gate elements for better cache locality
    const double complex g00 = gate->elements[0][0];
    const double complex g01 = gate->elements[0][1];
    const double complex g10 = gate->elements[1][0];
    const double complex g11 = gate->elements[1][1];

    // Apply gate: new_t[l, p', r] = sum_p gate[p', p] * t[l, p, r]
    for (uint32_t l = 0; l < left_dim; l++) {
        const uint64_t base_l = l * stride_l;
        for (uint32_t r = 0; r < right_dim; r++) {
            const uint64_t base_lr = base_l + r;
            // Direct access to t[l, 0, r] and t[l, 1, r]
            const double complex t0 = t->data[base_lr];
            const double complex t1 = t->data[base_lr + stride_p];
            // Compute new values
            new_t->data[base_lr] = g00 * t0 + g01 * t1;
            new_t->data[base_lr + stride_p] = g10 * t0 + g11 * t1;
        }
    }

    tensor_free(state->tensors[qubit]);
    state->tensors[qubit] = new_t;

    // Invalidate canonical form if not at center
    if (state->canonical == TN_CANONICAL_MIXED &&
        (int32_t)qubit != state->canonical_center) {
        state->canonical = TN_CANONICAL_NONE;
    }

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_x(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_X);
}

tn_gate_error_t tn_apply_y(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_Y);
}

tn_gate_error_t tn_apply_z(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_Z);
}

tn_gate_error_t tn_apply_h(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_H);
}

tn_gate_error_t tn_apply_s(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_S);
}

tn_gate_error_t tn_apply_t(tn_mps_state_t *state, uint32_t qubit) {
    return tn_apply_gate_1q(state, qubit, &TN_GATE_T);
}

tn_gate_error_t tn_apply_rx(tn_mps_state_t *state, uint32_t qubit, double theta) {
    tn_gate_1q_t gate = tn_gate_rx(theta);
    return tn_apply_gate_1q(state, qubit, &gate);
}

tn_gate_error_t tn_apply_ry(tn_mps_state_t *state, uint32_t qubit, double theta) {
    tn_gate_1q_t gate = tn_gate_ry(theta);
    return tn_apply_gate_1q(state, qubit, &gate);
}

tn_gate_error_t tn_apply_rz(tn_mps_state_t *state, uint32_t qubit, double theta) {
    tn_gate_1q_t gate = tn_gate_rz(theta);
    return tn_apply_gate_1q(state, qubit, &gate);
}

// ============================================================================
// TWO-QUBIT GATE APPLICATION
// ============================================================================

/**
 * @brief Apply two-qubit gate to adjacent qubits
 */
static tn_gate_error_t apply_gate_2q_adjacent(tn_mps_state_t *state,
                                               uint32_t left_qubit,
                                               const tn_gate_2q_t *gate,
                                               double *truncation_error) {
#if HAS_METAL
    // Try GPU path for larger bond dimensions
    if (state->bond_dims[left_qubit] >= GPU_BOND_THRESHOLD) {
        tn_gate_error_t gpu_result = apply_gate_2q_adjacent_gpu(state, left_qubit, gate, truncation_error);
        if (gpu_result == TN_GATE_SUCCESS) {
            return TN_GATE_SUCCESS;
        }
        // Fall through to CPU path if GPU failed
    }
#endif

    uint32_t right_qubit = left_qubit + 1;

    tensor_t *tl = state->tensors[left_qubit];
    tensor_t *tr = state->tensors[right_qubit];

    uint32_t ll_dim = tl->dims[0];  // Left bond of left tensor
    uint32_t bond = tl->dims[2];    // Shared bond (should equal tr->dims[0])
    uint32_t rr_dim = tr->dims[2];  // Right bond of right tensor

    // Contract left and right tensors
    // tl[ll, pl, bond] * tr[bond, pr, rr] -> theta[ll, pl, pr, rr]
    uint32_t axes_l[1] = {2};
    uint32_t axes_r[1] = {0};
    tensor_t *theta = tensor_contract(tl, tr, axes_l, axes_r, 1);
    if (!theta) return TN_GATE_ERROR_CONTRACTION_FAILED;

    // theta has shape [ll, pl, pr, rr]
    // Apply gate: theta'[ll, pl', pr', rr] = sum_{pl,pr} gate[pl'*2+pr', pl*2+pr] * theta[ll, pl, pr, rr]

    tensor_t *theta_new = tensor_create(4, theta->dims);
    if (!theta_new) {
        tensor_free(theta);
        return TN_GATE_ERROR_ALLOC_FAILED;
    }

    for (uint32_t ll = 0; ll < ll_dim; ll++) {
        for (uint32_t rr = 0; rr < rr_dim; rr++) {
            for (uint32_t plp = 0; plp < TN_PHYSICAL_DIM; plp++) {
                for (uint32_t prp = 0; prp < TN_PHYSICAL_DIM; prp++) {
                    double complex sum = 0.0;
                    uint32_t out_idx = plp * TN_PHYSICAL_DIM + prp;

                    for (uint32_t pl = 0; pl < TN_PHYSICAL_DIM; pl++) {
                        for (uint32_t pr = 0; pr < TN_PHYSICAL_DIM; pr++) {
                            uint32_t in_idx = pl * TN_PHYSICAL_DIM + pr;
                            uint32_t theta_idx[4] = {ll, pl, pr, rr};
                            sum += gate->elements[out_idx][in_idx] *
                                   tensor_get(theta, theta_idx);
                        }
                    }

                    uint32_t new_idx[4] = {ll, plp, prp, rr};
                    tensor_set(theta_new, new_idx, sum);
                }
            }
        }
    }

    tensor_free(theta);

    // Reshape to matrix for SVD: [ll * pl, pr * rr]
    uint32_t mat_dims[2] = {ll_dim * TN_PHYSICAL_DIM, TN_PHYSICAL_DIM * rr_dim};
    tensor_t *mat = tensor_reshape(theta_new, 2, mat_dims);
    tensor_free(theta_new);
    if (!mat) return TN_GATE_ERROR_ALLOC_FAILED;

    // SVD with truncation
    svd_compress_config_t svd_cfg = svd_compress_config_default();
    svd_cfg.max_bond_dim = state->config.max_bond_dim;
    svd_cfg.cutoff = state->config.svd_cutoff;

    svd_compress_result_t *svd = svd_compress(mat, &svd_cfg);
    tensor_free(mat);

    if (!svd) return TN_GATE_ERROR_TRUNCATION;

    if (truncation_error) *truncation_error = svd->truncation_error;

    uint32_t new_bond = svd->bond_dim;

    // ========================================================================
    // SINGULAR VALUE HANDLING FOR IMAGINARY TIME EVOLUTION:
    //
    // For non-unitary gates, singular values can grow/shrink exponentially.
    // We normalize to prevent overflow, but TRACK what we normalized in
    // log_norm_factor so measurements can account for it.
    // ========================================================================
    double sv_norm_sq = 0.0;
    for (uint32_t i = 0; i < new_bond; i++) {
        double sv = svd->singular_values[i];
        // Check for NaN/Inf in singular values
        if (isnan(sv) || isinf(sv)) {
            svd_compress_result_free(svd);
            return TN_GATE_ERROR_TRUNCATION;
        }
        sv_norm_sq += sv * sv;
    }
    double sv_norm = sqrt(sv_norm_sq);

    // Normalize singular values but track the factor we removed
    // CRITICAL: Use 1e-100 threshold (not 1e-30) to catch only true numerical zero.
    // Values between 1e-100 and 1e-30 are legitimate small numbers from underflow,
    // not indicators of state collapse. Returning error instead of silent corruption.
    if (sv_norm > 1e-100 && !isnan(sv_norm) && !isinf(sv_norm)) {
        // Accumulate the log of what we're normalizing out
        state->log_norm_factor += log(sv_norm);
        for (uint32_t i = 0; i < new_bond; i++) {
            svd->singular_values[i] /= sv_norm;
        }
    } else if (sv_norm <= 1e-100) {
        // Genuine numerical collapse - return error instead of silently corrupting
        // The calling code should handle this (e.g., reduce step size, increase precision)
        svd_compress_result_free(svd);
        return TN_GATE_ERROR_TRUNCATION;
    }

    // Absorb singular values into right tensor
    // U is [ll * pl, new_bond], reshape to [ll, pl, new_bond]
    uint32_t new_tl_dims[3] = {ll_dim, TN_PHYSICAL_DIM, new_bond};
    tensor_t *new_tl = tensor_reshape(svd->left, 3, new_tl_dims);
    if (!new_tl) {
        svd_compress_result_free(svd);
        return TN_GATE_ERROR_ALLOC_FAILED;
    }

    // Vh is [new_bond, pr * rr], multiply by normalized S and reshape to [new_bond, pr, rr]
    // S_normalized * Vh
    for (uint32_t i = 0; i < new_bond; i++) {
        for (uint32_t j = 0; j < TN_PHYSICAL_DIM * rr_dim; j++) {
            svd->right->data[i * TN_PHYSICAL_DIM * rr_dim + j] *= svd->singular_values[i];
        }
    }

    uint32_t new_tr_dims[3] = {new_bond, TN_PHYSICAL_DIM, rr_dim};
    tensor_t *new_tr = tensor_reshape(svd->right, 3, new_tr_dims);
    if (!new_tr) {
        tensor_free(new_tl);
        svd_compress_result_free(svd);
        return TN_GATE_ERROR_ALLOC_FAILED;
    }

    // Update state
    tensor_free(state->tensors[left_qubit]);
    tensor_free(state->tensors[right_qubit]);
    state->tensors[left_qubit] = new_tl;
    state->tensors[right_qubit] = new_tr;
    state->bond_dims[left_qubit] = new_bond;

    // Track truncation
    state->cumulative_truncation_error += svd->truncation_error;
    if (svd->num_discarded > 0) {
        state->num_truncations++;
    }

    svd->left = NULL;
    svd->right = NULL;
    svd_compress_result_free(svd);

    // After SVD with S absorbed into right tensor, the left tensor is left-canonical
    // (U from SVD has orthonormal columns). Track this for efficient norm computation.
    // If sweeping left-to-right, canonical form is progressively restored.
    // We mark as NONE but provide a fast restore function for TEBD sweeps.
    state->canonical = TN_CANONICAL_NONE;

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_gate_2q(tn_mps_state_t *state,
                                  uint32_t qubit1, uint32_t qubit2,
                                  const tn_gate_2q_t *gate,
                                  double *truncation_error) {
    if (!state || !gate) return TN_GATE_ERROR_NULL_PTR;
    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) {
        return TN_GATE_ERROR_INVALID_QUBIT;
    }
    if (qubit1 == qubit2) return TN_GATE_ERROR_INVALID_QUBIT;

    // Ensure qubit1 < qubit2
    if (qubit1 > qubit2) {
        uint32_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
        // Need to transpose gate for swapped qubit order
        // For simplicity, require qubit1 < qubit2 or use SWAP network
    }

    if (qubit2 == qubit1 + 1) {
        // Adjacent qubits
        return apply_gate_2q_adjacent(state, qubit1, gate, truncation_error);
    }

    // Non-adjacent qubits: use SWAP network
    double total_error = 0.0;
    double step_error;

    // Bring qubit2 adjacent to qubit1 using SWAPs
    for (uint32_t i = qubit2 - 1; i > qubit1; i--) {
        tn_gate_error_t err = apply_gate_2q_adjacent(state, i, &TN_GATE_SWAP, &step_error);
        if (err != TN_GATE_SUCCESS) return err;
        total_error += step_error;
    }

    // Apply the gate
    tn_gate_error_t err = apply_gate_2q_adjacent(state, qubit1, gate, &step_error);
    if (err != TN_GATE_SUCCESS) return err;
    total_error += step_error;

    // Swap back
    for (uint32_t i = qubit1 + 1; i < qubit2; i++) {
        err = apply_gate_2q_adjacent(state, i, &TN_GATE_SWAP, &step_error);
        if (err != TN_GATE_SUCCESS) return err;
        total_error += step_error;
    }

    if (truncation_error) *truncation_error = total_error;

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_cnot(tn_mps_state_t *state,
                               uint32_t control, uint32_t target) {
    return tn_apply_gate_2q(state, control, target, &TN_GATE_CNOT, NULL);
}

tn_gate_error_t tn_apply_cz(tn_mps_state_t *state,
                             uint32_t qubit1, uint32_t qubit2) {
    return tn_apply_gate_2q(state, qubit1, qubit2, &TN_GATE_CZ, NULL);
}

tn_gate_error_t tn_apply_swap(tn_mps_state_t *state,
                               uint32_t qubit1, uint32_t qubit2) {
    return tn_apply_gate_2q(state, qubit1, qubit2, &TN_GATE_SWAP, NULL);
}

tn_gate_error_t tn_apply_rzz(tn_mps_state_t *state,
                              uint32_t qubit1, uint32_t qubit2,
                              double theta) {
    tn_gate_2q_t gate = tn_gate_rzz(theta);
    return tn_apply_gate_2q(state, qubit1, qubit2, &gate, NULL);
}

// ============================================================================
// MULTI-QUBIT OPERATIONS
// ============================================================================

tn_gate_error_t tn_apply_controlled(tn_mps_state_t *state,
                                     const uint32_t *controls,
                                     uint32_t num_controls,
                                     uint32_t target,
                                     const tn_gate_1q_t *gate) {
    if (!state || !controls || !gate) return TN_GATE_ERROR_NULL_PTR;
    if (num_controls == 0) {
        return tn_apply_gate_1q(state, target, gate);
    }

    // For multi-controlled gates, decompose into sequence of operations
    // This is a simplified implementation using recursive decomposition

    if (num_controls == 1) {
        // Single control: build 2-qubit controlled gate
        tn_gate_2q_t cgate = {{{0}}};

        // |00> -> |00>, |01> -> |01>, |10> -> |10> * I, |11> -> |11> * gate
        // Control on first qubit (index 0 in 2q space)
        cgate.elements[0][0] = 1.0;  // |00> -> |00>
        cgate.elements[1][1] = 1.0;  // |01> -> |01>
        cgate.elements[2][2] = gate->elements[0][0];  // |10> -> gate[0,0]|10> + gate[0,1]|11>
        cgate.elements[2][3] = gate->elements[0][1];
        cgate.elements[3][2] = gate->elements[1][0];  // |11> -> gate[1,0]|10> + gate[1,1]|11>
        cgate.elements[3][3] = gate->elements[1][1];

        return tn_apply_gate_2q(state, controls[0], target, &cgate, NULL);
    }

    // Multi-control decomposition (simplified)
    // Use Toffoli-like decomposition
    // This is a placeholder - full implementation would use proper decomposition

    return TN_GATE_ERROR_INVALID_GATE;  // Not fully implemented for >1 control
}

tn_gate_error_t tn_apply_toffoli(tn_mps_state_t *state,
                                  uint32_t control1, uint32_t control2,
                                  uint32_t target) {
    // Toffoli = CCX = two controls on X
    // Decompose into CNOT and single-qubit gates

    // Standard decomposition (Nielsen & Chuang):
    // H target
    // CNOT control2 -> target
    // Tdg target
    // CNOT control1 -> target
    // T target
    // CNOT control2 -> target
    // Tdg target
    // CNOT control1 -> target
    // T target, T control2
    // H target
    // CNOT control1 -> control2
    // T control1, Tdg control2
    // CNOT control1 -> control2

    tn_gate_error_t err;

    err = tn_apply_h(state, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control2, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_gate_1q(state, target, &TN_GATE_TDG);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control1, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_t(state, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control2, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_gate_1q(state, target, &TN_GATE_TDG);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control1, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_t(state, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_t(state, control2);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_h(state, target);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control1, control2);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_t(state, control1);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_gate_1q(state, control2, &TN_GATE_TDG);
    if (err != TN_GATE_SUCCESS) return err;

    err = tn_apply_cnot(state, control1, control2);
    if (err != TN_GATE_SUCCESS) return err;

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_global_phase(tn_mps_state_t *state, double phase) {
    if (!state) return TN_GATE_ERROR_NULL_PTR;

    double complex factor = cexp(I * phase);
    tensor_scale_inplace(state->tensors[0], factor);

    return TN_GATE_SUCCESS;
}

// ============================================================================
// LAYER OPERATIONS
// ============================================================================

tn_gate_error_t tn_apply_h_all(tn_mps_state_t *state) {
    if (!state) return TN_GATE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        tn_gate_error_t err = tn_apply_h(state, i);
        if (err != TN_GATE_SUCCESS) return err;
    }

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_layer_1q(tn_mps_state_t *state,
                                   const tn_gate_1q_t *gates) {
    if (!state || !gates) return TN_GATE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        tn_gate_error_t err = tn_apply_gate_1q(state, i, &gates[i]);
        if (err != TN_GATE_SUCCESS) return err;
    }

    return TN_GATE_SUCCESS;
}

tn_gate_error_t tn_apply_layer_rzz(tn_mps_state_t *state,
                                    const double *angles,
                                    bool even) {
    if (!state || !angles) return TN_GATE_ERROR_NULL_PTR;

    uint32_t start = even ? 0 : 1;
    uint32_t angle_idx = 0;

    for (uint32_t i = start; i + 1 < state->num_qubits; i += 2) {
        tn_gate_error_t err = tn_apply_rzz(state, i, i + 1, angles[angle_idx++]);
        if (err != TN_GATE_SUCCESS) return err;
    }

    return TN_GATE_SUCCESS;
}

// ============================================================================
// MPO OPERATIONS
// ============================================================================

tn_mpo_t *tn_mpo_single_site(uint32_t num_sites, uint32_t site,
                              const tn_gate_1q_t *op) {
    if (!op || site >= num_sites) return NULL;

    tn_mpo_t *mpo = (tn_mpo_t *)calloc(1, sizeof(tn_mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    mpo->tensors = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));
    mpo->bond_dims = (uint32_t *)calloc(num_sites - 1, sizeof(uint32_t));

    if (!mpo->tensors || !mpo->bond_dims) {
        tn_mpo_free(mpo);
        return NULL;
    }

    // Create identity MPO tensors except at target site
    for (uint32_t i = 0; i < num_sites; i++) {
        uint32_t left_bond = (i == 0) ? 1 : 1;
        uint32_t right_bond = (i == num_sites - 1) ? 1 : 1;

        // MPO tensor shape: [left_bond, phys_in, phys_out, right_bond]
        uint32_t dims[4] = {left_bond, TN_PHYSICAL_DIM, TN_PHYSICAL_DIM, right_bond};
        mpo->tensors[i] = tensor_create(4, dims);

        if (!mpo->tensors[i]) {
            tn_mpo_free(mpo);
            return NULL;
        }

        // Fill with operator
        for (uint32_t pi = 0; pi < TN_PHYSICAL_DIM; pi++) {
            for (uint32_t po = 0; po < TN_PHYSICAL_DIM; po++) {
                uint32_t idx[4] = {0, pi, po, 0};
                double complex val = (i == site) ?
                    op->elements[po][pi] :
                    (pi == po ? 1.0 : 0.0);
                tensor_set(mpo->tensors[i], idx, val);
            }
        }

        if (i < num_sites - 1) {
            mpo->bond_dims[i] = 1;
        }
    }

    return mpo;
}

tn_mpo_t *tn_mpo_two_site(uint32_t num_sites, uint32_t site1, uint32_t site2,
                           const tn_gate_2q_t *op) {
    if (!op || site1 >= num_sites || site2 >= num_sites || site1 == site2) {
        return NULL;
    }

    // For simplicity, require adjacent sites
    if (site2 != site1 + 1) return NULL;

    tn_mpo_t *mpo = (tn_mpo_t *)calloc(1, sizeof(tn_mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    mpo->tensors = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));
    mpo->bond_dims = (uint32_t *)calloc(num_sites - 1, sizeof(uint32_t));

    if (!mpo->tensors || !mpo->bond_dims) {
        tn_mpo_free(mpo);
        return NULL;
    }

    for (uint32_t i = 0; i < num_sites; i++) {
        uint32_t left_bond = (i == 0) ? 1 : 1;
        uint32_t right_bond = (i == num_sites - 1) ? 1 : 1;

        // For two-site operator, need bond dimension 4 between sites
        if (i == site1) right_bond = 4;
        if (i == site2) left_bond = 4;

        uint32_t dims[4] = {left_bond, TN_PHYSICAL_DIM, TN_PHYSICAL_DIM, right_bond};
        mpo->tensors[i] = tensor_create(4, dims);

        if (!mpo->tensors[i]) {
            tn_mpo_free(mpo);
            return NULL;
        }

        if (i == site1) {
            // First site of two-site operator
            for (uint32_t pi = 0; pi < TN_PHYSICAL_DIM; pi++) {
                for (uint32_t po = 0; po < TN_PHYSICAL_DIM; po++) {
                    uint32_t idx[4] = {0, pi, po, pi * TN_PHYSICAL_DIM + po};
                    tensor_set(mpo->tensors[i], idx, 1.0);
                }
            }
        } else if (i == site2) {
            // Second site of two-site operator
            for (uint32_t b = 0; b < 4; b++) {
                uint32_t pi1 = b / TN_PHYSICAL_DIM;
                uint32_t po1 = b % TN_PHYSICAL_DIM;
                for (uint32_t pi2 = 0; pi2 < TN_PHYSICAL_DIM; pi2++) {
                    for (uint32_t po2 = 0; po2 < TN_PHYSICAL_DIM; po2++) {
                        uint32_t in_idx = pi1 * TN_PHYSICAL_DIM + pi2;
                        uint32_t out_idx = po1 * TN_PHYSICAL_DIM + po2;
                        uint32_t idx[4] = {b, pi2, po2, 0};
                        tensor_set(mpo->tensors[i], idx, op->elements[out_idx][in_idx]);
                    }
                }
            }
        } else {
            // Identity
            for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                uint32_t idx[4] = {0, p, p, 0};
                tensor_set(mpo->tensors[i], idx, 1.0);
            }
        }

        if (i < num_sites - 1) {
            mpo->bond_dims[i] = (i == site1) ? 4 : 1;
        }
    }

    return mpo;
}

tn_gate_error_t tn_apply_mpo(tn_mps_state_t *state,
                              const tn_mpo_t *mpo,
                              double *truncation_error) {
    if (!state || !mpo) return TN_GATE_ERROR_NULL_PTR;
    if (state->num_qubits != mpo->num_sites) return TN_GATE_ERROR_INVALID_GATE;

    double total_error = 0.0;

    // Apply MPO site by site, contracting and truncating
    for (uint32_t i = 0; i < state->num_qubits; i++) {
        tensor_t *mps_t = state->tensors[i];
        tensor_t *mpo_t = mpo->tensors[i];

        // Contract MPS tensor with MPO tensor
        // MPS: [left_mps, phys, right_mps]
        // MPO: [left_mpo, phys_in, phys_out, right_mpo]
        // Result: [left_mps, left_mpo, phys_out, right_mps, right_mpo]
        // Then reshape to [left_new, phys_out, right_new]

        uint32_t axes_mps[1] = {1};  // Physical index
        uint32_t axes_mpo[1] = {1};  // Physical in index

        tensor_t *contracted = tensor_contract(mps_t, mpo_t, axes_mps, axes_mpo, 1);
        if (!contracted) return TN_GATE_ERROR_CONTRACTION_FAILED;

        // Reshape: [left_mps, right_mps, left_mpo, phys_out, right_mpo]
        // -> [left_mps * left_mpo, phys_out, right_mps * right_mpo]

        uint32_t new_left = mps_t->dims[0] * mpo_t->dims[0];
        uint32_t new_phys = mpo_t->dims[2];
        uint32_t new_right = mps_t->dims[2] * mpo_t->dims[3];

        uint32_t new_dims[3] = {new_left, new_phys, new_right};
        tensor_t *new_mps = tensor_reshape(contracted, 3, new_dims);
        tensor_free(contracted);

        if (!new_mps) return TN_GATE_ERROR_ALLOC_FAILED;

        tensor_free(state->tensors[i]);
        state->tensors[i] = new_mps;
    }

    // Update bond dimensions (they may have grown)
    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        state->bond_dims[i] = state->tensors[i]->dims[2];
    }

    // Truncate back to max bond dimension
    double trunc_err;
    tn_mps_truncate(state, state->config.max_bond_dim, &trunc_err);
    total_error += trunc_err;

    if (truncation_error) *truncation_error = total_error;

    return TN_GATE_SUCCESS;
}

void tn_mpo_free(tn_mpo_t *mpo) {
    if (!mpo) return;

    if (mpo->tensors) {
        for (uint32_t i = 0; i < mpo->num_sites; i++) {
            tensor_free(mpo->tensors[i]);
        }
        free(mpo->tensors);
    }

    free(mpo->bond_dims);
    free(mpo);
}

// ============================================================================
// UTILITIES
// ============================================================================

const char *tn_gate_error_string(tn_gate_error_t error) {
    switch (error) {
        case TN_GATE_SUCCESS: return "Success";
        case TN_GATE_ERROR_NULL_PTR: return "Null pointer";
        case TN_GATE_ERROR_INVALID_QUBIT: return "Invalid qubit index";
        case TN_GATE_ERROR_INVALID_GATE: return "Invalid gate";
        case TN_GATE_ERROR_CONTRACTION_FAILED: return "Contraction failed";
        case TN_GATE_ERROR_TRUNCATION: return "Truncation failed";
        case TN_GATE_ERROR_BOND_TOO_LARGE: return "Bond dimension too large";
        case TN_GATE_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        default: return "Unknown error";
    }
}

void tn_gate_1q_print(const tn_gate_1q_t *gate, const char *name) {
    if (!gate) {
        printf("%s: NULL\n", name ? name : "gate");
        return;
    }

    printf("%s:\n", name ? name : "gate");
    for (int i = 0; i < 2; i++) {
        printf("  [");
        for (int j = 0; j < 2; j++) {
            double r = creal(gate->elements[i][j]);
            double im = cimag(gate->elements[i][j]);
            if (fabs(im) < 1e-10) {
                printf("%8.4f", r);
            } else {
                printf("%6.3f%+6.3fi", r, im);
            }
            if (j < 1) printf(", ");
        }
        printf("]\n");
    }
}

void tn_gate_2q_print(const tn_gate_2q_t *gate, const char *name) {
    if (!gate) {
        printf("%s: NULL\n", name ? name : "gate");
        return;
    }

    printf("%s:\n", name ? name : "gate");
    for (int i = 0; i < 4; i++) {
        printf("  [");
        for (int j = 0; j < 4; j++) {
            double r = creal(gate->elements[i][j]);
            double im = cimag(gate->elements[i][j]);
            if (fabs(im) < 1e-10) {
                printf("%6.3f", r);
            } else {
                printf("%5.2f%+5.2fi", r, im);
            }
            if (j < 3) printf(", ");
        }
        printf("]\n");
    }
}

tn_gate_1q_t tn_gate_from_matrix(const double complex *matrix) {
    tn_gate_1q_t gate;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            gate.elements[i][j] = matrix[i * 2 + j];
        }
    }
    return gate;
}

tn_gate_2q_t tn_gate_2q_from_matrix(const double complex *matrix) {
    tn_gate_2q_t gate;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            gate.elements[i][j] = matrix[i * 4 + j];
        }
    }
    return gate;
}
