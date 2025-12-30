/**
 * @file tn_state.c
 * @brief Tensor network quantum state implementation
 *
 * Full production implementation of Matrix Product State (MPS)
 * for quantum simulation of 50-200+ qubit systems.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "tn_state.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

tn_state_config_t tn_state_config_default(void) {
    tn_state_config_t config = {
        .max_bond_dim = TN_DEFAULT_BOND_DIM,
        .svd_cutoff = SVD_DEFAULT_CUTOFF,
        .max_truncation_error = SVD_DEFAULT_MAX_ERROR,
        .track_truncation = true,
        .auto_canonicalize = true,
        .target_form = TN_CANONICAL_MIXED
    };
    return config;
}

tn_state_config_t tn_state_config_create(uint32_t max_bond, double cutoff) {
    tn_state_config_t config = tn_state_config_default();
    config.max_bond_dim = max_bond;
    config.svd_cutoff = cutoff;
    return config;
}

// ============================================================================
// STATE CREATION
// ============================================================================

tn_mps_state_t *tn_mps_create_zero(uint32_t num_qubits,
                                    const tn_state_config_t *config) {
    if (num_qubits == 0 || num_qubits > TN_MAX_QUBITS) return NULL;

    tn_mps_state_t *state = (tn_mps_state_t *)calloc(1, sizeof(tn_mps_state_t));
    if (!state) return NULL;

    state->num_qubits = num_qubits;

    // Use default config if none provided
    if (config) {
        state->config = *config;
    } else {
        state->config = tn_state_config_default();
    }

    // Allocate tensor array
    state->tensors = (tensor_t **)calloc(num_qubits, sizeof(tensor_t *));
    if (!state->tensors) {
        free(state);
        return NULL;
    }

    // Allocate bond dimension array
    if (num_qubits > 1) {
        state->bond_dims = (uint32_t *)calloc(num_qubits - 1, sizeof(uint32_t));
        if (!state->bond_dims) {
            free(state->tensors);
            free(state);
            return NULL;
        }
    }

    // Create tensors for |00...0> state
    // For product state |0>, each tensor is just [[1], [0]] reshaped to [left, physical, right]
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t left_bond = (i == 0) ? 1 : 1;
        uint32_t right_bond = (i == num_qubits - 1) ? 1 : 1;

        uint32_t dims[3] = {left_bond, TN_PHYSICAL_DIM, right_bond};
        state->tensors[i] = tensor_create(3, dims);

        if (!state->tensors[i]) {
            // Clean up on failure
            for (uint32_t j = 0; j < i; j++) {
                tensor_free(state->tensors[j]);
            }
            free(state->tensors);
            free(state->bond_dims);
            free(state);
            return NULL;
        }

        // Set |0> amplitude
        // Tensor layout: [left_idx, physical_idx, right_idx]
        // For |0> state: tensor[0, 0, 0] = 1, tensor[0, 1, 0] = 0
        uint32_t indices[3] = {0, 0, 0};
        tensor_set(state->tensors[i], indices, 1.0);
    }

    // Set bond dimensions (all 1 for product state)
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        state->bond_dims[i] = 1;
    }

    state->canonical = TN_CANONICAL_LEFT;  // Product state is already canonical
    state->canonical_center = -1;
    state->norm = 1.0;
    state->log_norm_factor = 0.0;
    state->cumulative_truncation_error = 0.0;
    state->num_truncations = 0;

    // Initialize workspace as invalid (will be allocated on first use)
    state->workspace.transfer_buffer = NULL;
    state->workspace.local_buffer = NULL;
    state->workspace.buffer_capacity = 0;
    state->workspace.valid = false;

    return state;
}

tn_mps_state_t *tn_mps_create_basis(uint32_t num_qubits,
                                     uint64_t basis_state,
                                     const tn_state_config_t *config) {
    tn_mps_state_t *state = tn_mps_create_zero(num_qubits, config);
    if (!state) return NULL;

    // Modify each tensor to represent the correct basis state
    for (uint32_t i = 0; i < num_qubits; i++) {
        // Get bit for this qubit (qubit 0 is rightmost bit)
        int bit = (basis_state >> (num_qubits - 1 - i)) & 1;

        // Zero out |0> component, set |bit> component
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};

        if (bit == 0) {
            tensor_set(state->tensors[i], idx0, 1.0);
            tensor_set(state->tensors[i], idx1, 0.0);
        } else {
            tensor_set(state->tensors[i], idx0, 0.0);
            tensor_set(state->tensors[i], idx1, 1.0);
        }
    }

    return state;
}

tn_mps_state_t *tn_mps_create_product(uint32_t num_qubits,
                                       const double complex (*qubit_states)[2],
                                       const tn_state_config_t *config) {
    if (!qubit_states) return NULL;

    tn_mps_state_t *state = tn_mps_create_zero(num_qubits, config);
    if (!state) return NULL;

    // Set each tensor to represent the individual qubit state
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};

        tensor_set(state->tensors[i], idx0, qubit_states[i][0]);
        tensor_set(state->tensors[i], idx1, qubit_states[i][1]);
    }

    return state;
}

tn_mps_state_t *tn_mps_from_statevector(const double complex *amplitudes,
                                         uint32_t num_qubits,
                                         const tn_state_config_t *config) {
    if (!amplitudes || num_qubits == 0 || num_qubits > 32) return NULL;

    uint64_t state_dim = 1ULL << num_qubits;

    tn_mps_state_t *state = (tn_mps_state_t *)calloc(1, sizeof(tn_mps_state_t));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    if (config) {
        state->config = *config;
    } else {
        state->config = tn_state_config_default();
    }

    state->tensors = (tensor_t **)calloc(num_qubits, sizeof(tensor_t *));
    state->bond_dims = (uint32_t *)calloc(num_qubits - 1, sizeof(uint32_t));

    if (!state->tensors || !state->bond_dims) {
        free(state->tensors);
        free(state->bond_dims);
        free(state);
        return NULL;
    }

    // Convert state vector to MPS via successive SVD
    // Start with full state vector as [2, 2^(n-1)] matrix
    // SVD gives U[2, chi] * S * Vh[chi, 2^(n-1)]
    // First tensor is U, reshape remainder and repeat

    // Working copy of amplitudes as tensor
    uint32_t remaining_dims[2] = {TN_PHYSICAL_DIM, (uint32_t)(state_dim / TN_PHYSICAL_DIM)};
    tensor_t *remaining = tensor_create_with_data(2, remaining_dims, amplitudes);
    if (!remaining) {
        free(state->tensors);
        free(state->bond_dims);
        free(state);
        return NULL;
    }

    // SVD config
    svd_compress_config_t svd_config = svd_compress_config_default();
    svd_config.max_bond_dim = state->config.max_bond_dim;
    svd_config.cutoff = state->config.svd_cutoff;

    double total_truncation = 0.0;

    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        // Reshape to [left_bond * physical, right_dims]
        uint32_t left_size = remaining->dims[0];
        uint32_t right_size = remaining->dims[1];

        // Perform SVD
        tensor_svd_result_t *svd = tensor_svd(remaining, svd_config.max_bond_dim,
                                               svd_config.cutoff);
        if (!svd) {
            tensor_free(remaining);
            for (uint32_t j = 0; j < i; j++) {
                tensor_free(state->tensors[j]);
            }
            free(state->tensors);
            free(state->bond_dims);
            free(state);
            return NULL;
        }

        total_truncation += svd->truncation_error;

        // Extract first MPS tensor
        if (i == 0) {
            // First tensor: [1, physical, bond]
            uint32_t tensor_dims[3] = {1, TN_PHYSICAL_DIM, svd->k};
            state->tensors[i] = tensor_create(3, tensor_dims);
            if (state->tensors[i]) {
                // Copy U reshaped
                for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                    for (uint32_t b = 0; b < svd->k; b++) {
                        uint32_t idx[3] = {0, p, b};
                        tensor_set(state->tensors[i], idx, svd->U->data[p * svd->k + b]);
                    }
                }
            }
        } else {
            // Middle tensor: [prev_bond, physical, bond]
            uint32_t prev_bond = state->bond_dims[i - 1];
            uint32_t tensor_dims[3] = {prev_bond, TN_PHYSICAL_DIM, svd->k};
            state->tensors[i] = tensor_create(3, tensor_dims);
            if (state->tensors[i]) {
                // U is [left * physical, k], reshape to [left, physical, k]
                for (uint32_t l = 0; l < prev_bond; l++) {
                    for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                        for (uint32_t b = 0; b < svd->k; b++) {
                            uint32_t idx[3] = {l, p, b};
                            uint32_t u_idx = (l * TN_PHYSICAL_DIM + p) * svd->k + b;
                            tensor_set(state->tensors[i], idx, svd->U->data[u_idx]);
                        }
                    }
                }
            }
        }

        state->bond_dims[i] = svd->k;

        // Prepare remainder: S * Vh, reshaped for next iteration
        uint32_t next_physical_dim = TN_PHYSICAL_DIM;
        uint32_t next_remaining = right_size / next_physical_dim;

        if (i < num_qubits - 2) {
            uint32_t new_dims[2] = {svd->k * next_physical_dim, next_remaining};
            tensor_t *new_remaining = tensor_create(2, new_dims);

            if (new_remaining) {
                // Multiply S into Vh rows and reshape
                for (uint32_t s = 0; s < svd->k; s++) {
                    for (uint32_t col = 0; col < right_size; col++) {
                        double complex val = svd->S[s] * svd->Vh->data[s * right_size + col];
                        // Map col to (physical, remaining)
                        uint32_t p = col / next_remaining;
                        uint32_t r = col % next_remaining;
                        uint32_t new_row = s * next_physical_dim + p;
                        new_remaining->data[new_row * next_remaining + r] = val;
                    }
                }
            }

            tensor_free(remaining);
            remaining = new_remaining;
        } else {
            // Last iteration: remainder becomes last tensor
            tensor_free(remaining);
            remaining = NULL;

            // Last tensor: [bond, physical, 1]
            uint32_t tensor_dims[3] = {svd->k, TN_PHYSICAL_DIM, 1};
            state->tensors[num_qubits - 1] = tensor_create(3, tensor_dims);
            if (state->tensors[num_qubits - 1]) {
                // Vh is [k, right_size], right_size = physical * 1
                for (uint32_t b = 0; b < svd->k; b++) {
                    for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                        uint32_t idx[3] = {b, p, 0};
                        tensor_set(state->tensors[num_qubits - 1], idx,
                                   svd->S[b] * svd->Vh->data[b * TN_PHYSICAL_DIM + p]);
                    }
                }
            }
        }

        tensor_svd_free(svd);
    }

    tensor_free(remaining);

    state->canonical = TN_CANONICAL_LEFT;
    state->canonical_center = -1;
    state->norm = 1.0;
    state->log_norm_factor = 0.0;
    state->cumulative_truncation_error = total_truncation;
    state->num_truncations = num_qubits - 1;

    // Initialize workspace as invalid (will be allocated on first use)
    state->workspace.transfer_buffer = NULL;
    state->workspace.local_buffer = NULL;
    state->workspace.buffer_capacity = 0;
    state->workspace.valid = false;

    // Normalize
    tn_mps_normalize(state);

    return state;
}

tn_mps_state_t *tn_mps_copy(const tn_mps_state_t *state) {
    if (!state) return NULL;

    tn_mps_state_t *copy = (tn_mps_state_t *)calloc(1, sizeof(tn_mps_state_t));
    if (!copy) return NULL;

    copy->num_qubits = state->num_qubits;
    copy->config = state->config;
    copy->canonical = state->canonical;
    copy->canonical_center = state->canonical_center;
    copy->norm = state->norm;
    copy->log_norm_factor = state->log_norm_factor;
    copy->cumulative_truncation_error = state->cumulative_truncation_error;
    copy->num_truncations = state->num_truncations;

    // Workspace is not copied - initialized fresh for each state
    copy->workspace.transfer_buffer = NULL;
    copy->workspace.local_buffer = NULL;
    copy->workspace.buffer_capacity = 0;
    copy->workspace.valid = false;

    copy->tensors = (tensor_t **)calloc(state->num_qubits, sizeof(tensor_t *));
    if (!copy->tensors) {
        free(copy);
        return NULL;
    }

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        copy->tensors[i] = tensor_copy(state->tensors[i]);
        if (!copy->tensors[i]) {
            for (uint32_t j = 0; j < i; j++) {
                tensor_free(copy->tensors[j]);
            }
            free(copy->tensors);
            free(copy);
            return NULL;
        }
    }

    if (state->num_qubits > 1) {
        copy->bond_dims = (uint32_t *)malloc((state->num_qubits - 1) * sizeof(uint32_t));
        if (!copy->bond_dims) {
            for (uint32_t i = 0; i < state->num_qubits; i++) {
                tensor_free(copy->tensors[i]);
            }
            free(copy->tensors);
            free(copy);
            return NULL;
        }
        memcpy(copy->bond_dims, state->bond_dims,
               (state->num_qubits - 1) * sizeof(uint32_t));
    }

    return copy;
}

void tn_mps_free(tn_mps_state_t *state) {
    if (!state) return;

    // Free workspace buffers
    tn_mps_free_workspace(state);

    if (state->tensors) {
        for (uint32_t i = 0; i < state->num_qubits; i++) {
            tensor_free(state->tensors[i]);
        }
        free(state->tensors);
    }

    free(state->bond_dims);
    free(state);
}

// ============================================================================
// STATE PROPERTIES
// ============================================================================

uint32_t tn_mps_num_qubits(const tn_mps_state_t *state) {
    return state ? state->num_qubits : 0;
}

uint32_t tn_mps_bond_dim(const tn_mps_state_t *state, uint32_t bond) {
    if (!state || bond >= state->num_qubits - 1) return 0;
    return state->bond_dims[bond];
}

uint32_t tn_mps_max_bond_dim(const tn_mps_state_t *state) {
    if (!state || state->num_qubits < 2) return 1;

    uint32_t max = 0;
    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        if (state->bond_dims[i] > max) max = state->bond_dims[i];
    }
    return max;
}

const tensor_t *tn_mps_get_tensor(const tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return NULL;
    return state->tensors[qubit];
}

tn_mps_stats_t tn_mps_get_stats(const tn_mps_state_t *state) {
    tn_mps_stats_t stats = {0};
    if (!state) return stats;

    stats.norm = state->norm;
    stats.truncation_error = state->cumulative_truncation_error;
    stats.max_bond_dim = tn_mps_max_bond_dim(state);

    uint64_t total_elements = 0;
    double bond_sum = 0.0;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        if (state->tensors[i]) {
            total_elements += state->tensors[i]->total_size;
        }
    }

    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        bond_sum += state->bond_dims[i];
    }

    stats.total_elements = total_elements;
    stats.memory_bytes = total_elements * sizeof(double complex);
    stats.avg_bond_dim = (state->num_qubits > 1) ?
        bond_sum / (state->num_qubits - 1) : 1.0;

    // Compute entanglement at middle bond
    if (state->num_qubits > 1) {
        uint32_t mid = (state->num_qubits - 1) / 2;
        stats.entanglement_entropy = tn_mps_entanglement_entropy(state, mid);
    }

    return stats;
}

void tn_mps_print_info(const tn_mps_state_t *state) {
    if (!state) {
        printf("MPS State: NULL\n");
        return;
    }

    printf("MPS State:\n");
    printf("  Qubits:           %u\n", state->num_qubits);
    printf("  Max bond dim:     %u\n", tn_mps_max_bond_dim(state));
    printf("  Canonical form:   %d (center: %d)\n",
           state->canonical, state->canonical_center);
    printf("  Norm:             %.6f\n", state->norm);
    printf("  Truncation error: %.6e\n", state->cumulative_truncation_error);
    printf("  Bond dimensions:  [");
    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        printf("%u", state->bond_dims[i]);
        if (i < state->num_qubits - 2) printf(", ");
    }
    printf("]\n");

    tn_mps_stats_t stats = tn_mps_get_stats(state);
    printf("  Memory usage:     %lu bytes\n", (unsigned long)stats.memory_bytes);
    printf("  Entanglement:     %.4f\n", stats.entanglement_entropy);
}

// ============================================================================
// AMPLITUDE ACCESS
// ============================================================================

double complex tn_mps_amplitude(const tn_mps_state_t *state, uint64_t basis_state) {
    if (!state || state->num_qubits == 0) return 0.0;

    // Contract MPS with computational basis state
    // Start from left, contract each tensor with basis bit

    tensor_t *result = NULL;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        int bit = (basis_state >> (state->num_qubits - 1 - i)) & 1;

        // Extract slice for this bit: tensor[:, bit, :]
        const tensor_t *t = state->tensors[i];
        uint32_t left_dim = t->dims[0];
        uint32_t right_dim = t->dims[2];

        tensor_t *slice = tensor_create_matrix(left_dim, right_dim);
        if (!slice) {
            tensor_free(result);
            return 0.0;
        }

        for (uint32_t l = 0; l < left_dim; l++) {
            for (uint32_t r = 0; r < right_dim; r++) {
                uint32_t idx[3] = {l, (uint32_t)bit, r};
                slice->data[l * right_dim + r] = tensor_get(t, idx);
            }
        }

        if (result == NULL) {
            result = slice;
        } else {
            // Contract result with slice
            tensor_t *new_result = tensor_matmul(result, slice);
            tensor_free(result);
            tensor_free(slice);
            result = new_result;
            if (!result) return 0.0;
        }
    }

    double complex amp = 0.0;
    if (result && result->total_size > 0) {
        amp = result->data[0];
    }
    tensor_free(result);

    return amp * state->norm;
}

tn_state_error_t tn_mps_amplitudes(const tn_mps_state_t *state,
                                    const uint64_t *basis_states,
                                    uint32_t num_states,
                                    double complex *amplitudes) {
    if (!state || !basis_states || !amplitudes) return TN_STATE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < num_states; i++) {
        amplitudes[i] = tn_mps_amplitude(state, basis_states[i]);
    }

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_to_statevector(const tn_mps_state_t *state,
                                        double complex *amplitudes) {
    if (!state || !amplitudes) return TN_STATE_ERROR_NULL_PTR;
    if (state->num_qubits > 30) return TN_STATE_ERROR_ALLOC_FAILED;

    uint64_t state_dim = 1ULL << state->num_qubits;

    for (uint64_t i = 0; i < state_dim; i++) {
        amplitudes[i] = tn_mps_amplitude(state, i);
    }

    return TN_STATE_SUCCESS;
}

// ============================================================================
// CANONICAL FORM OPERATIONS
// ============================================================================

tn_state_error_t tn_mps_left_canonicalize(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        tensor_t *t = state->tensors[i];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        // Reshape to [left * phys, right]
        uint32_t mat_dims[2] = {left_dim * phys_dim, right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        // QR decomposition
        tensor_qr_result_t *qr = tensor_qr(mat);
        tensor_free(mat);
        if (!qr) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = qr->Q->dims[1];

        // Reshape Q back to MPS tensor
        uint32_t new_dims[3] = {left_dim, phys_dim, new_bond};
        tensor_t *new_tensor = tensor_reshape(qr->Q, 3, new_dims);
        if (!new_tensor) {
            tensor_qr_free(qr);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // Update tensor
        tensor_free(state->tensors[i]);
        state->tensors[i] = new_tensor;
        state->bond_dims[i] = new_bond;

        // Absorb R into next tensor
        tensor_t *next = state->tensors[i + 1];
        uint32_t next_left = next->dims[0];
        uint32_t next_phys = next->dims[1];
        uint32_t next_right = next->dims[2];

        // Reshape next to [left, phys * right]
        uint32_t next_mat_dims[2] = {next_left, next_phys * next_right};
        tensor_t *next_mat = tensor_reshape(next, 2, next_mat_dims);
        if (!next_mat) {
            tensor_qr_free(qr);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // R @ next_mat
        tensor_t *product = tensor_matmul(qr->R, next_mat);
        tensor_free(next_mat);
        tensor_qr_free(qr);

        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        // Reshape back
        uint32_t next_new_dims[3] = {new_bond, next_phys, next_right};
        tensor_t *next_new = tensor_reshape(product, 3, next_new_dims);
        tensor_free(product);

        if (!next_new) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_free(state->tensors[i + 1]);
        state->tensors[i + 1] = next_new;
    }

    state->canonical = TN_CANONICAL_LEFT;
    state->canonical_center = -1;

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_right_canonicalize(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    for (int i = state->num_qubits - 1; i > 0; i--) {
        tensor_t *t = state->tensors[i];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        // Reshape to [left, phys * right]
        uint32_t mat_dims[2] = {left_dim, phys_dim * right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        // LQ decomposition
        tensor_qr_result_t *lq = tensor_lq(mat);
        tensor_free(mat);
        if (!lq) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = lq->Q->dims[0];

        // Reshape Q back to MPS tensor
        uint32_t new_dims[3] = {new_bond, phys_dim, right_dim};
        tensor_t *new_tensor = tensor_reshape(lq->Q, 3, new_dims);
        if (!new_tensor) {
            tensor_qr_free(lq);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // Update tensor
        tensor_free(state->tensors[i]);
        state->tensors[i] = new_tensor;
        state->bond_dims[i - 1] = new_bond;

        // Absorb L into previous tensor
        tensor_t *prev = state->tensors[i - 1];
        uint32_t prev_left = prev->dims[0];
        uint32_t prev_phys = prev->dims[1];
        uint32_t prev_right = prev->dims[2];

        // Reshape prev to [left * phys, right]
        uint32_t prev_mat_dims[2] = {prev_left * prev_phys, prev_right};
        tensor_t *prev_mat = tensor_reshape(prev, 2, prev_mat_dims);
        if (!prev_mat) {
            tensor_qr_free(lq);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // prev_mat @ L
        tensor_t *product = tensor_matmul(prev_mat, lq->R);  // L is in R field for LQ
        tensor_free(prev_mat);
        tensor_qr_free(lq);

        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        // Reshape back
        uint32_t prev_new_dims[3] = {prev_left, prev_phys, new_bond};
        tensor_t *prev_new = tensor_reshape(product, 3, prev_new_dims);
        tensor_free(product);

        if (!prev_new) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_free(state->tensors[i - 1]);
        state->tensors[i - 1] = prev_new;
    }

    state->canonical = TN_CANONICAL_RIGHT;
    state->canonical_center = -1;

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_mixed_canonicalize(tn_mps_state_t *state, uint32_t center) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;
    if (center >= state->num_qubits) return TN_STATE_ERROR_INVALID_QUBIT_INDEX;

    // Special cases:
    // - center=0: Left loop is empty (correct), right loop runs from n-1 to 1,
    //   absorbing L factors into tensor[0]. tensor[0] becomes the center.
    // - center=n-1: Right loop is empty (correct), left loop runs from 0 to n-2,
    //   absorbing R factors into tensor[n-1]. tensor[n-1] becomes the center.
    // Both edge cases work correctly with the general algorithm.

    // Left-canonicalize from 0 to center-1
    for (uint32_t i = 0; i < center; i++) {
        tensor_t *t = state->tensors[i];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        uint32_t mat_dims[2] = {left_dim * phys_dim, right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_qr_result_t *qr = tensor_qr(mat);
        tensor_free(mat);
        if (!qr) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = qr->Q->dims[1];
        uint32_t new_dims[3] = {left_dim, phys_dim, new_bond};
        tensor_t *new_tensor = tensor_reshape(qr->Q, 3, new_dims);

        if (!new_tensor) {
            tensor_qr_free(qr);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        tensor_free(state->tensors[i]);
        state->tensors[i] = new_tensor;
        state->bond_dims[i] = new_bond;

        // Absorb R into next
        tensor_t *next = state->tensors[i + 1];
        uint32_t next_mat_dims[2] = {next->dims[0], next->dims[1] * next->dims[2]};
        tensor_t *next_mat = tensor_reshape(next, 2, next_mat_dims);

        tensor_t *product = tensor_matmul(qr->R, next_mat);
        tensor_free(next_mat);
        tensor_qr_free(qr);

        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t next_new_dims[3] = {new_bond, next->dims[1], next->dims[2]};
        tensor_t *next_new = tensor_reshape(product, 3, next_new_dims);
        tensor_free(product);

        tensor_free(state->tensors[i + 1]);
        state->tensors[i + 1] = next_new;
    }

    // Right-canonicalize from num_qubits-1 to center+1
    for (int i = state->num_qubits - 1; i > (int)center; i--) {
        tensor_t *t = state->tensors[i];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        uint32_t mat_dims[2] = {left_dim, phys_dim * right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_qr_result_t *lq = tensor_lq(mat);
        tensor_free(mat);
        if (!lq) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = lq->Q->dims[0];
        uint32_t new_dims[3] = {new_bond, phys_dim, right_dim};
        tensor_t *new_tensor = tensor_reshape(lq->Q, 3, new_dims);

        if (!new_tensor) {
            tensor_qr_free(lq);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        tensor_free(state->tensors[i]);
        state->tensors[i] = new_tensor;
        state->bond_dims[i - 1] = new_bond;

        // Absorb L into previous
        tensor_t *prev = state->tensors[i - 1];
        uint32_t prev_mat_dims[2] = {prev->dims[0] * prev->dims[1], prev->dims[2]};
        tensor_t *prev_mat = tensor_reshape(prev, 2, prev_mat_dims);

        tensor_t *product = tensor_matmul(prev_mat, lq->R);
        tensor_free(prev_mat);
        tensor_qr_free(lq);

        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t prev_new_dims[3] = {prev->dims[0], prev->dims[1], new_bond};
        tensor_t *prev_new = tensor_reshape(product, 3, prev_new_dims);
        tensor_free(product);

        tensor_free(state->tensors[i - 1]);
        state->tensors[i - 1] = prev_new;
    }

    state->canonical = TN_CANONICAL_MIXED;
    state->canonical_center = center;

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_move_center(tn_mps_state_t *state, int direction) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;
    if (state->canonical != TN_CANONICAL_MIXED) return TN_STATE_ERROR_INVALID_CONFIG;

    int center = state->canonical_center;
    int new_center = center + direction;

    if (new_center < 0 || new_center >= (int)state->num_qubits) {
        return TN_STATE_ERROR_INVALID_QUBIT_INDEX;
    }

    if (direction > 0) {
        // Move right: QR decompose current center, absorb R into next site
        tensor_t *t = state->tensors[center];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        // Reshape to [left * phys, right]
        uint32_t mat_dims[2] = {left_dim * phys_dim, right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_qr_result_t *qr = tensor_qr(mat);
        tensor_free(mat);
        if (!qr) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = qr->Q->dims[1];

        // Reshape Q back to MPS tensor (now left-isometric)
        uint32_t new_dims[3] = {left_dim, phys_dim, new_bond};
        tensor_t *new_tensor = tensor_reshape(qr->Q, 3, new_dims);
        if (!new_tensor) {
            tensor_qr_free(qr);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        tensor_free(state->tensors[center]);
        state->tensors[center] = new_tensor;
        state->bond_dims[center] = new_bond;

        // Absorb R into next tensor
        tensor_t *next = state->tensors[new_center];
        uint32_t next_left = next->dims[0];
        uint32_t next_phys = next->dims[1];
        uint32_t next_right = next->dims[2];

        // Reshape next to [left, phys * right]
        uint32_t next_mat_dims[2] = {next_left, next_phys * next_right};
        tensor_t *next_mat = tensor_reshape(next, 2, next_mat_dims);
        if (!next_mat) {
            tensor_qr_free(qr);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // R @ next_mat
        tensor_t *product = tensor_matmul(qr->R, next_mat);
        tensor_free(next_mat);
        tensor_qr_free(qr);
        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        // Reshape back to 3D
        uint32_t next_new_dims[3] = {new_bond, next_phys, next_right};
        tensor_t *next_new = tensor_reshape(product, 3, next_new_dims);
        tensor_free(product);
        if (!next_new) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_free(state->tensors[new_center]);
        state->tensors[new_center] = next_new;

    } else {
        // Move left: LQ decompose current center, absorb L into previous site
        tensor_t *t = state->tensors[center];
        uint32_t left_dim = t->dims[0];
        uint32_t phys_dim = t->dims[1];
        uint32_t right_dim = t->dims[2];

        // Reshape to [left, phys * right]
        uint32_t mat_dims[2] = {left_dim, phys_dim * right_dim};
        tensor_t *mat = tensor_reshape(t, 2, mat_dims);
        if (!mat) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_qr_result_t *lq = tensor_lq(mat);
        tensor_free(mat);
        if (!lq) return TN_STATE_ERROR_ALLOC_FAILED;

        uint32_t new_bond = lq->Q->dims[0];

        // Reshape Q back to MPS tensor (now right-isometric)
        uint32_t new_dims[3] = {new_bond, phys_dim, right_dim};
        tensor_t *new_tensor = tensor_reshape(lq->Q, 3, new_dims);
        if (!new_tensor) {
            tensor_qr_free(lq);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        tensor_free(state->tensors[center]);
        state->tensors[center] = new_tensor;
        state->bond_dims[center - 1] = new_bond;

        // Absorb L into previous tensor
        tensor_t *prev = state->tensors[new_center];
        uint32_t prev_left = prev->dims[0];
        uint32_t prev_phys = prev->dims[1];
        uint32_t prev_right = prev->dims[2];

        // Reshape prev to [left * phys, right]
        uint32_t prev_mat_dims[2] = {prev_left * prev_phys, prev_right};
        tensor_t *prev_mat = tensor_reshape(prev, 2, prev_mat_dims);
        if (!prev_mat) {
            tensor_qr_free(lq);
            return TN_STATE_ERROR_ALLOC_FAILED;
        }

        // prev_mat @ L
        tensor_t *product = tensor_matmul(prev_mat, lq->R);  // L is in R field for LQ
        tensor_free(prev_mat);
        tensor_qr_free(lq);
        if (!product) return TN_STATE_ERROR_ALLOC_FAILED;

        // Reshape back to 3D
        uint32_t prev_new_dims[3] = {prev_left, prev_phys, new_bond};
        tensor_t *prev_new = tensor_reshape(product, 3, prev_new_dims);
        tensor_free(product);
        if (!prev_new) return TN_STATE_ERROR_ALLOC_FAILED;

        tensor_free(state->tensors[new_center]);
        state->tensors[new_center] = prev_new;
    }

    state->canonical_center = new_center;
    return TN_STATE_SUCCESS;
}

void tn_mps_mark_canonical_left(tn_mps_state_t *state) {
    if (state) {
        state->canonical = TN_CANONICAL_LEFT;
        state->canonical_center = -1;
    }
}

void tn_mps_mark_canonical_right(tn_mps_state_t *state) {
    if (state) {
        state->canonical = TN_CANONICAL_RIGHT;
        state->canonical_center = -1;
    }
}

// ============================================================================
// NORMALIZATION
// ============================================================================

double tn_mps_norm(const tn_mps_state_t *state) {
    if (!state) return 0.0;

    // Compute <psi|psi> by contracting MPS with its conjugate
    // For left-canonical MPS, norm is just norm of rightmost tensor

    if (state->canonical == TN_CANONICAL_LEFT) {
        return tensor_norm_frobenius(state->tensors[state->num_qubits - 1]);
    }

    // General case: full contraction using transfer matrix method
    // Transfer matrix has indices [left, left_conj, right, right_conj]
    // We contract right bonds with next site's left bonds
    tensor_t *transfer = NULL;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        const tensor_t *t = state->tensors[i];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(transfer);
            return 0.0;
        }

        // Contract physical index: t[l,p,r] * tc[l',p',r'] over p=p'
        // Result has shape [l, r, l', r']
        uint32_t axes_t[1] = {1};
        uint32_t axes_tc[1] = {1};
        tensor_t *local_raw = tensor_contract(t, tc, axes_t, axes_tc, 1);
        tensor_free(tc);

        if (!local_raw) {
            tensor_free(transfer);
            return 0.0;
        }

        // Reorder to [l, l', r, r'] for consistent contraction
        uint32_t perm[4] = {0, 2, 1, 3};
        tensor_t *local = tensor_transpose(local_raw, perm);
        tensor_free(local_raw);

        if (!local) {
            tensor_free(transfer);
            return 0.0;
        }

        if (transfer == NULL) {
            transfer = local;
        } else {
            // Contract transfer[l_first, l_first', r_prev, r_prev'] with
            // local[l_curr, l_curr', r_curr, r_curr'] over r_prev=l_curr, r_prev'=l_curr'
            // Transfer's right bonds are at positions {2, 3}
            // Local's left bonds are at positions {0, 1}
            uint32_t axes_tr[2] = {2, 3};
            uint32_t axes_loc[2] = {0, 1};
            tensor_t *new_transfer = tensor_contract(transfer, local, axes_tr, axes_loc, 2);
            tensor_free(transfer);
            tensor_free(local);
            transfer = new_transfer;
            if (!transfer) return 0.0;
        }
    }

    double norm = 0.0;
    if (transfer && transfer->total_size > 0) {
        // Use absolute value to handle small numerical errors that could make
        // the trace slightly negative
        double norm_sq = creal(transfer->data[0]);
        if (norm_sq < 0.0) {
            // Numerical error - use absolute value if close to zero
            norm_sq = fabs(norm_sq);
        }
        // Check for NaN/Inf
        if (isnan(norm_sq) || isinf(norm_sq)) {
            tensor_free(transfer);
            return 0.0;
        }
        norm = sqrt(norm_sq);
    }
    tensor_free(transfer);

    return norm;
}

tn_state_error_t tn_mps_normalize(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    double norm = tn_mps_norm(state);
    if (norm < 1e-15) return TN_STATE_ERROR_NORMALIZATION;

    // For left-canonical MPS, norm lives in rightmost tensor - scale that one
    // For right-canonical MPS, norm lives in leftmost tensor
    // For non-canonical, scaling any tensor works but we use first by convention
    uint32_t tensor_to_scale;
    if (state->canonical == TN_CANONICAL_LEFT) {
        tensor_to_scale = state->num_qubits - 1;
    } else if (state->canonical == TN_CANONICAL_RIGHT) {
        tensor_to_scale = 0;
    } else {
        tensor_to_scale = 0;
    }

    tensor_scale_inplace(state->tensors[tensor_to_scale], 1.0 / norm);
    state->norm = 1.0;

    return TN_STATE_SUCCESS;
}

// ============================================================================
// WORKSPACE MANAGEMENT
// ============================================================================

tn_state_error_t tn_mps_init_workspace(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    // For small bond dimensions (chi <= 32), we can use optimized in-place computation
    // For larger chi, the chi^4 memory requirement makes it impractical
    // In that case, we fall back to the tensor library implementation
    uint32_t max_chi = state->config.max_bond_dim;

    // Only allocate buffers if chi is small enough (chi^4 < 1M elements = 16MB)
    if (max_chi <= 32) {
        uint64_t buffer_size = (uint64_t)max_chi * max_chi * max_chi * max_chi;

        state->workspace.transfer_buffer = (double complex *)aligned_alloc(64,
            buffer_size * 2 * sizeof(double complex));
        state->workspace.local_buffer = (double complex *)aligned_alloc(64,
            buffer_size * sizeof(double complex));

        if (!state->workspace.transfer_buffer || !state->workspace.local_buffer) {
            tn_mps_free_workspace(state);
            // Not a fatal error - we'll use fallback
            state->workspace.valid = false;
            return TN_STATE_SUCCESS;
        }

        state->workspace.buffer_capacity = buffer_size;
        state->workspace.valid = true;
    } else {
        // For large chi, don't allocate huge buffers - use fallback path
        state->workspace.valid = false;
    }

    return TN_STATE_SUCCESS;
}

void tn_mps_free_workspace(tn_mps_state_t *state) {
    if (!state) return;

    if (state->workspace.transfer_buffer) {
        free(state->workspace.transfer_buffer);
        state->workspace.transfer_buffer = NULL;
    }
    if (state->workspace.local_buffer) {
        free(state->workspace.local_buffer);
        state->workspace.local_buffer = NULL;
    }
    state->workspace.buffer_capacity = 0;
    state->workspace.valid = false;
}

// ============================================================================
// OPTIMIZED NORM COMPUTATION
// ============================================================================

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

/**
 * @brief Compute norm squared using optimized in-place algorithm
 *
 * Key optimizations:
 * 1. Uses pre-allocated workspace buffers to avoid malloc/free per call
 * 2. Computes T[l,r] * conj(T[l',r']) contraction directly using BLAS
 * 3. Avoids tensor library overhead for simple operations
 */
double tn_mps_norm_squared_fast(tn_mps_state_t *state) {
    if (!state) return 0.0;

    // Fast path for left-canonical MPS
    if (state->canonical == TN_CANONICAL_LEFT) {
        const tensor_t *last = state->tensors[state->num_qubits - 1];
        double sum = 0.0;
        for (uint64_t i = 0; i < last->total_size; i++) {
            double complex val = last->data[i];
            sum += creal(val) * creal(val) + cimag(val) * cimag(val);
        }
        return sum;
    }

    // Initialize workspace if needed
    if (!state->workspace.valid) {
        if (tn_mps_init_workspace(state) != TN_STATE_SUCCESS) {
            // Fall back to standard norm
            double n = tn_mps_norm(state);
            return n * n;
        }
    }

    // Get max bond dimension for this state
    uint32_t max_chi = 1;
    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        if (state->bond_dims[i] > max_chi) max_chi = state->bond_dims[i];
    }

    // Check if workspace is large enough
    uint64_t needed = (uint64_t)max_chi * max_chi * max_chi * max_chi;
    if (needed > state->workspace.buffer_capacity) {
        // Reallocate workspace
        tn_mps_free_workspace(state);
        uint32_t old_max = state->config.max_bond_dim;
        state->config.max_bond_dim = max_chi;
        if (tn_mps_init_workspace(state) != TN_STATE_SUCCESS) {
            state->config.max_bond_dim = old_max;
            double n = tn_mps_norm(state);
            return n * n;
        }
        state->config.max_bond_dim = old_max;
    }

    double complex *transfer = state->workspace.transfer_buffer;
    double complex *local = state->workspace.local_buffer;

    // Initialize: transfer = identity for first site's left bond (dim 1)
    uint32_t tr_left = 1;
    uint32_t tr_right = 1;
    transfer[0] = 1.0;

    for (uint32_t site = 0; site < state->num_qubits; site++) {
        const tensor_t *t = state->tensors[site];
        uint32_t l = t->dims[0];
        uint32_t p = t->dims[1];
        uint32_t r = t->dims[2];

        // Compute local tensor: sum over physical index
        // local[l, l', r, r'] = sum_p T[l,p,r] * conj(T[l',p,r'])
        // This is T @ T^H in the (l,r) x (l',r') space after summing over p

        // Reshape T as matrix [l*p, r] and compute [l*p, r] @ [l*p, r]^H = [r, r]
        // But we need [l, l'] x [r, r'] structure

        // More direct: for each p, do outer product contribution
        // local[l,l',r,r'] = sum_p T[l,p,r] * conj(T[l',p,r'])

        uint64_t local_size = (uint64_t)l * l * r * r;
        memset(local, 0, local_size * sizeof(double complex));

        for (uint32_t pi = 0; pi < p; pi++) {
            // T_p is T[:, pi, :] with shape [l, r]
            // We need: local += T_p ⊗ conj(T_p)
            // In index form: local[i,j,k,m] += T[i,pi,k] * conj(T[j,pi,m])

            const double complex *T_p = t->data + pi * r;  // Offset for physical index pi
            uint64_t stride_l = p * r;  // Stride between left indices

            for (uint32_t i = 0; i < l; i++) {
                for (uint32_t j = 0; j < l; j++) {
                    for (uint32_t k = 0; k < r; k++) {
                        double complex tik = t->data[i * stride_l + pi * r + k];
                        for (uint32_t m = 0; m < r; m++) {
                            double complex tjm = t->data[j * stride_l + pi * r + m];
                            // local index: [i, j, k, m] -> i*(l*r*r) + j*(r*r) + k*r + m
                            uint64_t idx = (uint64_t)i * (l * r * r) + j * (r * r) + k * r + m;
                            local[idx] += tik * conj(tjm);
                        }
                    }
                }
            }
        }

        if (site == 0) {
            // First site: transfer = local (l=1, so effectively [r, r'])
            // Copy local to transfer, treating [1,1,r,r'] as [r,r']
            memcpy(transfer, local, local_size * sizeof(double complex));
            tr_left = l;
            tr_right = r;
        } else {
            // Contract: transfer[l_0, l_0', r_prev, r_prev'] * local[l_curr, l_curr', r_curr, r_curr']
            // where r_prev = l_curr (bond dimension match)
            // Sum over r_prev and r_prev' (which equal l_curr and l_curr')
            // Result: [l_0, l_0', r_curr, r_curr']

            // This is a matrix multiplication!
            // Reshape transfer as [l_0*l_0', r_prev*r_prev'] = [tr_left^2, tr_right^2]
            // Reshape local as [l_curr*l_curr', r_curr*r_curr'] = [l^2, r^2]
            // Since tr_right = l (bond dimension), we get:
            // result[l_0^2, r^2] = transfer[l_0^2, l^2] @ local[l^2, r^2]

            uint64_t M = (uint64_t)tr_left * tr_left;
            uint64_t K = (uint64_t)l * l;  // Should equal tr_right^2
            uint64_t N = (uint64_t)r * r;

            // Allocate temporary for result (use part of transfer buffer)
            double complex *result = transfer + state->workspace.buffer_capacity / 2;

            // Use BLAS for matrix multiply
#ifdef __APPLE__
            double complex alpha = 1.0;
            double complex beta = 0.0;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)M, (int)N, (int)K,
                        &alpha, transfer, (int)K,
                        local, (int)N,
                        &beta, result, (int)N);
#else
            // Fallback: naive matmul
            memset(result, 0, M * N * sizeof(double complex));
            for (uint64_t i = 0; i < M; i++) {
                for (uint64_t k = 0; k < K; k++) {
                    double complex a = transfer[i * K + k];
                    for (uint64_t j = 0; j < N; j++) {
                        result[i * N + j] += a * local[k * N + j];
                    }
                }
            }
#endif
            // Copy result back to beginning of transfer buffer
            memcpy(transfer, result, M * N * sizeof(double complex));
            tr_right = r;
        }
    }

    // Final transfer matrix should be [1, 1] (scalar) for properly closed MPS
    // The norm squared is transfer[0]
    return creal(transfer[0]);
}

// ============================================================================
// LAZY NORMALIZATION
// ============================================================================

tn_state_error_t tn_mps_normalize_lazy(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    // Compute norm using fast method
    double norm_sq = tn_mps_norm_squared_fast(state);
    if (norm_sq < 1e-30) return TN_STATE_ERROR_NORMALIZATION;

    double norm = sqrt(norm_sq);

    // Accumulate log(norm) instead of rescaling tensors
    state->log_norm_factor += log(norm);
    state->norm = 1.0;  // Mark as "normalized" from external perspective

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_commit_normalization(tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    if (fabs(state->log_norm_factor) < 1e-15) {
        return TN_STATE_SUCCESS;  // Nothing to commit
    }

    // Apply accumulated normalization factor to first tensor
    double scale = exp(-state->log_norm_factor);
    tensor_scale_inplace(state->tensors[0], scale);

    state->log_norm_factor = 0.0;
    state->norm = 1.0;

    return TN_STATE_SUCCESS;
}

double tn_mps_true_norm(const tn_mps_state_t *state) {
    if (!state) return 0.0;

    double tensor_norm = tn_mps_norm(state);
    return exp(state->log_norm_factor) * tensor_norm;
}

// ============================================================================
// ENTANGLEMENT
// ============================================================================

double tn_mps_entanglement_entropy(const tn_mps_state_t *state, uint32_t bond) {
    if (!state || bond >= state->num_qubits - 1) return 0.0;

    // For mixed-canonical form at this bond, entropy from singular values
    // For general MPS, need to compute Schmidt decomposition

    // Bring to mixed canonical form at bond
    tn_mps_state_t *work = tn_mps_copy(state);
    if (!work) return 0.0;

    tn_state_error_t err = tn_mps_mixed_canonicalize(work, bond);
    if (err != TN_STATE_SUCCESS) {
        tn_mps_free(work);
        return -1.0;  // Negative value indicates error (entropy is always >= 0)
    }

    // SVD the center tensor to get singular values
    tensor_t *center = work->tensors[bond];
    uint32_t left_dim = center->dims[0];
    uint32_t phys_dim = center->dims[1];
    uint32_t right_dim = center->dims[2];

    uint32_t mat_dims[2] = {left_dim * phys_dim, right_dim};
    tensor_t *mat = tensor_reshape(center, 2, mat_dims);

    double entropy = 0.0;
    if (mat) {
        tensor_svd_result_t *svd = tensor_svd(mat, 0, 0.0);
        tensor_free(mat);

        if (svd) {
            entropy = svd_entanglement_entropy(svd->S, svd->k);
            tensor_svd_free(svd);
        }
    }

    tn_mps_free(work);
    return entropy;
}

tn_state_error_t tn_mps_entanglement_spectrum(const tn_mps_state_t *state,
                                               uint32_t bond,
                                               double *spectrum,
                                               uint32_t *num_values) {
    if (!state || !spectrum || !num_values) return TN_STATE_ERROR_NULL_PTR;
    if (bond >= state->num_qubits - 1) return TN_STATE_ERROR_INVALID_QUBIT_INDEX;

    tn_mps_state_t *work = tn_mps_copy(state);
    if (!work) return TN_STATE_ERROR_ALLOC_FAILED;

    tn_state_error_t canon_err = tn_mps_mixed_canonicalize(work, bond);
    if (canon_err != TN_STATE_SUCCESS) {
        tn_mps_free(work);
        return canon_err;
    }

    tensor_t *center = work->tensors[bond];
    uint32_t mat_dims[2] = {center->dims[0] * center->dims[1], center->dims[2]};
    tensor_t *mat = tensor_reshape(center, 2, mat_dims);

    tn_state_error_t result = TN_STATE_ERROR_ALLOC_FAILED;
    if (mat) {
        tensor_svd_result_t *svd = tensor_svd(mat, 0, 0.0);
        tensor_free(mat);

        if (svd) {
            svd_schmidt_coefficients(svd->S, svd->k, spectrum);
            *num_values = svd->k;
            tensor_svd_free(svd);
            result = TN_STATE_SUCCESS;
        }
    }

    tn_mps_free(work);
    return result;
}

bool tn_mps_is_product_state(const tn_mps_state_t *state, double tolerance) {
    if (!state) return true;

    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        if (state->bond_dims[i] > 1) {
            double entropy = tn_mps_entanglement_entropy(state, i);
            if (entropy > tolerance) return false;
        }
    }

    return true;
}

// ============================================================================
// BOND DIMENSION MANAGEMENT
// ============================================================================

tn_state_error_t tn_mps_truncate(tn_mps_state_t *state,
                                  uint32_t max_bond,
                                  double *truncation_error) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;

    double total_error = 0.0;

    // First bring to canonical form
    tn_mps_left_canonicalize(state);

    // Then truncate from right to left
    for (int i = state->num_qubits - 2; i >= 0; i--) {
        double bond_error = 0.0;
        tn_state_error_t err = tn_mps_truncate_bond(state, i, max_bond, &bond_error);
        if (err != TN_STATE_SUCCESS) return err;
        total_error += bond_error;
    }

    if (truncation_error) *truncation_error = total_error;

    state->cumulative_truncation_error += total_error;
    state->num_truncations++;

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_truncate_bond(tn_mps_state_t *state,
                                       uint32_t bond,
                                       uint32_t max_dim,
                                       double *truncation_error) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;
    if (bond >= state->num_qubits - 1) return TN_STATE_ERROR_INVALID_QUBIT_INDEX;

    if (state->bond_dims[bond] <= max_dim) {
        if (truncation_error) *truncation_error = 0.0;
        return TN_STATE_SUCCESS;
    }

    // SVD compress the bond
    svd_compress_config_t config = svd_compress_config_fixed(max_dim);

    svd_compress_result_t *result = svd_compress_bond(
        state->tensors[bond], state->tensors[bond + 1],
        &config, true);

    if (!result) return TN_STATE_ERROR_TRUNCATION;

    // Update tensors
    tensor_free(state->tensors[bond]);
    tensor_free(state->tensors[bond + 1]);

    state->tensors[bond] = result->left;
    state->tensors[bond + 1] = result->right;
    state->bond_dims[bond] = result->bond_dim;

    if (truncation_error) *truncation_error = result->truncation_error;

    result->left = NULL;
    result->right = NULL;
    svd_compress_result_free(result);

    return TN_STATE_SUCCESS;
}

tn_state_error_t tn_mps_grow_bond(tn_mps_state_t *state,
                                   uint32_t bond,
                                   uint32_t new_dim) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;
    if (bond >= state->num_qubits - 1) return TN_STATE_ERROR_INVALID_QUBIT_INDEX;
    if (new_dim < state->bond_dims[bond]) return TN_STATE_ERROR_INVALID_CONFIG;
    if (new_dim == state->bond_dims[bond]) return TN_STATE_SUCCESS;

    // Expand both adjacent tensors
    tensor_t *left = state->tensors[bond];
    tensor_t *right = state->tensors[bond + 1];

    uint32_t old_bond = state->bond_dims[bond];

    // New left tensor with expanded right dimension
    uint32_t new_left_dims[3] = {left->dims[0], left->dims[1], new_dim};
    tensor_t *new_left = tensor_create(3, new_left_dims);
    if (!new_left) return TN_STATE_ERROR_ALLOC_FAILED;

    // Copy existing data
    for (uint32_t l = 0; l < left->dims[0]; l++) {
        for (uint32_t p = 0; p < left->dims[1]; p++) {
            for (uint32_t r = 0; r < old_bond; r++) {
                uint32_t old_idx[3] = {l, p, r};
                uint32_t new_idx[3] = {l, p, r};
                tensor_set(new_left, new_idx, tensor_get(left, old_idx));
            }
        }
    }

    // New right tensor with expanded left dimension
    uint32_t new_right_dims[3] = {new_dim, right->dims[1], right->dims[2]};
    tensor_t *new_right = tensor_create(3, new_right_dims);
    if (!new_right) {
        tensor_free(new_left);
        return TN_STATE_ERROR_ALLOC_FAILED;
    }

    // Copy existing data
    for (uint32_t l = 0; l < old_bond; l++) {
        for (uint32_t p = 0; p < right->dims[1]; p++) {
            for (uint32_t r = 0; r < right->dims[2]; r++) {
                uint32_t old_idx[3] = {l, p, r};
                uint32_t new_idx[3] = {l, p, r};
                tensor_set(new_right, new_idx, tensor_get(right, old_idx));
            }
        }
    }

    tensor_free(state->tensors[bond]);
    tensor_free(state->tensors[bond + 1]);
    state->tensors[bond] = new_left;
    state->tensors[bond + 1] = new_right;
    state->bond_dims[bond] = new_dim;

    return TN_STATE_SUCCESS;
}

// ============================================================================
// OVERLAP AND FIDELITY
// ============================================================================

double complex tn_mps_overlap(const tn_mps_state_t *state1,
                               const tn_mps_state_t *state2) {
    if (!state1 || !state2) return 0.0;
    if (state1->num_qubits != state2->num_qubits) return 0.0;

    // Contract <psi1|psi2> using transfer matrix method
    // Transfer has indices [left1, left1_conj, right1, right2]
    tensor_t *transfer = NULL;

    for (uint32_t i = 0; i < state1->num_qubits; i++) {
        const tensor_t *t1 = state1->tensors[i];
        const tensor_t *t2 = state2->tensors[i];

        tensor_t *t1c = tensor_conj(t1);
        if (!t1c) {
            tensor_free(transfer);
            return 0.0;
        }

        // Contract physical indices: t1c[l1,p,r1] * t2[l2,p',r2] over p=p'
        // Result has shape [l1, r1, l2, r2]
        uint32_t axes_t1[1] = {1};
        uint32_t axes_t2[1] = {1};
        tensor_t *local_raw = tensor_contract(t1c, t2, axes_t1, axes_t2, 1);
        tensor_free(t1c);

        if (!local_raw) {
            tensor_free(transfer);
            return 0.0;
        }

        // Reorder to [l1, l2, r1, r2] for consistent contraction
        uint32_t perm[4] = {0, 2, 1, 3};
        tensor_t *local = tensor_transpose(local_raw, perm);
        tensor_free(local_raw);

        if (!local) {
            tensor_free(transfer);
            return 0.0;
        }

        if (transfer == NULL) {
            transfer = local;
        } else {
            // Contract right bonds with next site's left bonds
            uint32_t axes_tr[2] = {2, 3};
            uint32_t axes_loc[2] = {0, 1};
            tensor_t *new_transfer = tensor_contract(transfer, local, axes_tr, axes_loc, 2);
            tensor_free(transfer);
            tensor_free(local);
            transfer = new_transfer;
            if (!transfer) return 0.0;
        }
    }

    double complex overlap = 0.0;
    if (transfer && transfer->total_size > 0) {
        overlap = transfer->data[0];
    }
    tensor_free(transfer);

    return overlap * state1->norm * state2->norm;
}

double tn_mps_fidelity(const tn_mps_state_t *state1,
                        const tn_mps_state_t *state2) {
    double complex overlap = tn_mps_overlap(state1, state2);
    return cabs(overlap) * cabs(overlap);
}

// ============================================================================
// UTILITIES
// ============================================================================

const char *tn_state_error_string(tn_state_error_t error) {
    switch (error) {
        case TN_STATE_SUCCESS: return "Success";
        case TN_STATE_ERROR_NULL_PTR: return "Null pointer";
        case TN_STATE_ERROR_INVALID_QUBITS: return "Invalid number of qubits";
        case TN_STATE_ERROR_INVALID_QUBIT_INDEX: return "Invalid qubit index";
        case TN_STATE_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case TN_STATE_ERROR_TRUNCATION: return "Truncation error";
        case TN_STATE_ERROR_CONTRACTION_FAILED: return "Contraction failed";
        case TN_STATE_ERROR_NORMALIZATION: return "Normalization failed";
        case TN_STATE_ERROR_INVALID_CONFIG: return "Invalid configuration";
        case TN_STATE_ERROR_ENTANGLEMENT_TOO_HIGH: return "Entanglement too high for bond dimension";
        default: return "Unknown error";
    }
}

uint64_t tn_mps_estimate_memory(uint32_t num_qubits, uint32_t bond_dim) {
    if (num_qubits == 0) return 0;

    // First and last tensors: [1, 2, chi] and [chi, 2, 1]
    uint64_t boundary = 2 * 1 * TN_PHYSICAL_DIM * bond_dim;

    // Middle tensors: [chi, 2, chi]
    uint64_t middle = 0;
    if (num_qubits > 2) {
        middle = (num_qubits - 2) * bond_dim * TN_PHYSICAL_DIM * bond_dim;
    }

    return (boundary + middle) * sizeof(double complex);
}

tn_state_error_t tn_mps_validate(const tn_mps_state_t *state) {
    if (!state) return TN_STATE_ERROR_NULL_PTR;
    if (!state->tensors) return TN_STATE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        if (!state->tensors[i]) return TN_STATE_ERROR_NULL_PTR;

        tensor_t *t = state->tensors[i];
        if (t->rank != 3) return TN_STATE_ERROR_INVALID_CONFIG;
        if (t->dims[1] != TN_PHYSICAL_DIM) return TN_STATE_ERROR_INVALID_CONFIG;

        // Check bond dimension consistency
        if (i == 0 && t->dims[0] != 1) return TN_STATE_ERROR_INVALID_CONFIG;
        if (i == state->num_qubits - 1 && t->dims[2] != 1) return TN_STATE_ERROR_INVALID_CONFIG;

        if (i > 0) {
            if (t->dims[0] != state->tensors[i-1]->dims[2]) {
                return TN_STATE_ERROR_INVALID_CONFIG;
            }
        }
    }

    return TN_STATE_SUCCESS;
}
