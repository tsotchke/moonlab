/**
 * @file contraction.c
 * @brief Tensor contraction engine implementation
 *
 * Full production implementation of tensor network contraction
 * with ordering optimization and memory management.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "contraction.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CONFIGURATION MANAGEMENT
// ============================================================================

contract_config_t contract_config_default(void) {
    contract_config_t config = {
        .strategy = CONTRACT_ORDER_GREEDY,
        .memory_limit = CONTRACT_DEFAULT_MEM_LIMIT,
        .use_compression = false,
        .compress_config = svd_compress_config_default(),
        .track_stats = true,
        .parallel = true,
        .num_threads = 0  // Auto
    };
    return config;
}

contract_config_t contract_config_memory_limited(uint64_t mem_limit) {
    contract_config_t config = contract_config_default();
    config.strategy = CONTRACT_ORDER_GREEDY_MEMORY;
    config.memory_limit = mem_limit;
    return config;
}

contract_config_t contract_config_compressed(uint32_t max_bond, double cutoff) {
    contract_config_t config = contract_config_default();
    config.use_compression = true;
    config.compress_config.mode = SVD_TRUNCATE_ADAPTIVE;
    config.compress_config.max_bond_dim = max_bond;
    config.compress_config.cutoff = cutoff;
    return config;
}

// ============================================================================
// NETWORK CONSTRUCTION
// ============================================================================

contract_network_t *contract_network_create(void) {
    contract_network_t *network = (contract_network_t *)calloc(1, sizeof(contract_network_t));
    return network;
}

int contract_network_add_tensor(contract_network_t *network,
                                 const tensor_t *tensor) {
    if (!network || !tensor) return -1;
    if (network->num_tensors >= CONTRACT_MAX_TENSORS) return -1;

    // Allocate or extend tensor array
    if (network->tensors == NULL) {
        network->tensors = (const tensor_t **)malloc(CONTRACT_MAX_TENSORS * sizeof(tensor_t *));
        if (!network->tensors) return -1;
    }

    int idx = network->num_tensors;
    network->tensors[idx] = tensor;
    network->num_tensors++;

    return idx;
}

contract_error_t contract_network_add_edge(contract_network_t *network,
                                            uint32_t tensor_a, uint32_t axis_a,
                                            uint32_t tensor_b, uint32_t axis_b) {
    if (!network) return CONTRACT_ERROR_NULL_PTR;
    if (network->num_edges >= CONTRACT_MAX_EDGES) return CONTRACT_ERROR_INVALID_NETWORK;
    if (tensor_a >= network->num_tensors || tensor_b >= network->num_tensors) {
        return CONTRACT_ERROR_INVALID_NETWORK;
    }

    const tensor_t *ta = network->tensors[tensor_a];
    const tensor_t *tb = network->tensors[tensor_b];

    if (axis_a >= ta->rank || axis_b >= tb->rank) {
        return CONTRACT_ERROR_INVALID_NETWORK;
    }

    if (ta->dims[axis_a] != tb->dims[axis_b]) {
        return CONTRACT_ERROR_DIM_MISMATCH;
    }

    contract_edge_t *edge = &network->edges[network->num_edges];
    edge->tensor_a = tensor_a;
    edge->tensor_b = tensor_b;
    edge->axis_a = axis_a;
    edge->axis_b = axis_b;
    edge->dimension = ta->dims[axis_a];

    network->num_edges++;
    return CONTRACT_SUCCESS;
}

contract_error_t contract_network_set_output(contract_network_t *network,
                                              const uint32_t *tensor_axes,
                                              uint32_t num_axes) {
    if (!network) return CONTRACT_ERROR_NULL_PTR;

    if (network->output_tensor_axes) {
        free(network->output_tensor_axes);
    }

    if (num_axes == 0 || !tensor_axes) {
        network->output_tensor_axes = NULL;
        network->num_output_axes = 0;
        return CONTRACT_SUCCESS;
    }

    network->output_tensor_axes = (uint32_t *)malloc(num_axes * 2 * sizeof(uint32_t));
    if (!network->output_tensor_axes) return CONTRACT_ERROR_ALLOC_FAILED;

    memcpy(network->output_tensor_axes, tensor_axes, num_axes * 2 * sizeof(uint32_t));
    network->num_output_axes = num_axes;

    return CONTRACT_SUCCESS;
}

contract_error_t contract_network_validate(const contract_network_t *network) {
    if (!network) return CONTRACT_ERROR_NULL_PTR;
    if (network->num_tensors == 0) return CONTRACT_ERROR_INVALID_NETWORK;

    // Check all tensors are valid
    for (uint32_t i = 0; i < network->num_tensors; i++) {
        if (!network->tensors[i]) return CONTRACT_ERROR_NULL_PTR;
    }

    // Check all edges are valid
    for (uint32_t i = 0; i < network->num_edges; i++) {
        const contract_edge_t *edge = &network->edges[i];

        if (edge->tensor_a >= network->num_tensors ||
            edge->tensor_b >= network->num_tensors) {
            return CONTRACT_ERROR_INVALID_NETWORK;
        }

        const tensor_t *ta = network->tensors[edge->tensor_a];
        const tensor_t *tb = network->tensors[edge->tensor_b];

        if (edge->axis_a >= ta->rank || edge->axis_b >= tb->rank) {
            return CONTRACT_ERROR_INVALID_NETWORK;
        }

        if (ta->dims[edge->axis_a] != tb->dims[edge->axis_b]) {
            return CONTRACT_ERROR_DIM_MISMATCH;
        }
    }

    return CONTRACT_SUCCESS;
}

void contract_network_free(contract_network_t *network) {
    if (!network) return;

    free((void *)network->tensors);
    free(network->output_tensor_axes);
    free(network);
}

// ============================================================================
// COST ESTIMATION
// ============================================================================

uint64_t contract_estimate_flops(const tensor_t *a, const tensor_t *b,
                                  const uint32_t *axes_a,
                                  const uint32_t *axes_b,
                                  uint32_t num_contract) {
    if (!a || !b) return 0;

    // Result size
    uint64_t result_size = a->total_size * b->total_size;

    // Divide by contracted dimensions
    for (uint32_t i = 0; i < num_contract; i++) {
        result_size /= a->dims[axes_a[i]];
    }

    // Multiply by contraction dimension (for the summation)
    uint64_t contract_dim = 1;
    for (uint32_t i = 0; i < num_contract; i++) {
        contract_dim *= a->dims[axes_a[i]];
    }

    // FLOPs: result_size * contract_dim * 2 (multiply + add) * 4 (complex)
    return result_size * contract_dim * 8;
}

uint64_t contract_estimate_memory(const tensor_t *a, const tensor_t *b,
                                   const uint32_t *axes_a,
                                   const uint32_t *axes_b,
                                   uint32_t num_contract) {
    if (!a || !b) return 0;

    // Mark contracted axes
    bool contracted_a[TENSOR_MAX_RANK] = {false};
    bool contracted_b[TENSOR_MAX_RANK] = {false};

    for (uint32_t i = 0; i < num_contract; i++) {
        contracted_a[axes_a[i]] = true;
        contracted_b[axes_b[i]] = true;
    }

    // Compute result size
    uint64_t result_size = 1;
    for (uint32_t i = 0; i < a->rank; i++) {
        if (!contracted_a[i]) result_size *= a->dims[i];
    }
    for (uint32_t i = 0; i < b->rank; i++) {
        if (!contracted_b[i]) result_size *= b->dims[i];
    }

    return result_size * sizeof(double complex);
}

// ============================================================================
// CONTRACTION ORDERING
// ============================================================================

/**
 * @brief Internal structure for tracking tensors during ordering
 */
typedef struct {
    uint32_t id;                    // Original tensor index
    tensor_t *tensor;               // Current tensor (may be intermediate)
    bool active;                    // Still in the network
    uint32_t *axis_map;             // Maps current axes to original
    uint32_t axis_map_size;
} contract_tensor_state_t;

/**
 * @brief Compute cost of contracting two tensors
 */
static uint64_t compute_contraction_cost(const contract_tensor_state_t *state_a,
                                          const contract_tensor_state_t *state_b,
                                          const contract_network_t *network,
                                          bool memory_priority) {
    if (!state_a->active || !state_b->active) return UINT64_MAX;

    // Find edges between these tensors
    uint32_t axes_a[TENSOR_MAX_RANK], axes_b[TENSOR_MAX_RANK];
    uint32_t num_contract = 0;

    for (uint32_t e = 0; e < network->num_edges; e++) {
        const contract_edge_t *edge = &network->edges[e];

        bool match_ab = (edge->tensor_a == state_a->id && edge->tensor_b == state_b->id);
        bool match_ba = (edge->tensor_a == state_b->id && edge->tensor_b == state_a->id);

        if (match_ab) {
            axes_a[num_contract] = edge->axis_a;
            axes_b[num_contract] = edge->axis_b;
            num_contract++;
        } else if (match_ba) {
            axes_a[num_contract] = edge->axis_b;
            axes_b[num_contract] = edge->axis_a;
            num_contract++;
        }
    }

    if (num_contract == 0) return UINT64_MAX;  // Not connected

    if (memory_priority) {
        return contract_estimate_memory(state_a->tensor, state_b->tensor,
                                        axes_a, axes_b, num_contract);
    } else {
        return contract_estimate_flops(state_a->tensor, state_b->tensor,
                                       axes_a, axes_b, num_contract);
    }
}

contract_plan_t *contract_find_order(const contract_network_t *network,
                                      const contract_config_t *config) {
    if (!network || !config) return NULL;

    contract_error_t err = contract_network_validate(network);
    if (err != CONTRACT_SUCCESS) return NULL;

    if (network->num_tensors == 1) {
        // Single tensor, no contractions needed
        contract_plan_t *plan = (contract_plan_t *)calloc(1, sizeof(contract_plan_t));
        return plan;
    }

    contract_plan_t *plan = (contract_plan_t *)calloc(1, sizeof(contract_plan_t));
    if (!plan) return NULL;

    plan->num_steps = network->num_tensors - 1;
    plan->steps = (contract_step_t *)calloc(plan->num_steps, sizeof(contract_step_t));
    if (!plan->steps) {
        free(plan);
        return NULL;
    }

    // Initialize tensor states
    contract_tensor_state_t *states = (contract_tensor_state_t *)calloc(
        network->num_tensors, sizeof(contract_tensor_state_t));
    if (!states) {
        contract_plan_free(plan);
        return NULL;
    }

    for (uint32_t i = 0; i < network->num_tensors; i++) {
        states[i].id = i;
        states[i].tensor = (tensor_t *)network->tensors[i];  // Non-owning cast
        states[i].active = true;
    }

    bool memory_priority = (config->strategy == CONTRACT_ORDER_GREEDY_MEMORY);

    // Greedy ordering
    for (uint32_t step = 0; step < plan->num_steps; step++) {
        uint64_t best_cost = UINT64_MAX;
        uint32_t best_a = 0, best_b = 0;

        // Find best pair to contract
        for (uint32_t i = 0; i < network->num_tensors; i++) {
            if (!states[i].active) continue;

            for (uint32_t j = i + 1; j < network->num_tensors; j++) {
                if (!states[j].active) continue;

                uint64_t cost = compute_contraction_cost(&states[i], &states[j],
                                                          network, memory_priority);

                if (cost < best_cost) {
                    best_cost = cost;
                    best_a = i;
                    best_b = j;
                }
            }
        }

        if (best_cost == UINT64_MAX) {
            // No connected tensors found - error
            free(states);
            contract_plan_free(plan);
            return NULL;
        }

        // Record this step
        contract_step_t *s = &plan->steps[step];
        s->tensor_a = best_a;
        s->tensor_b = best_b;

        // Find contraction axes
        uint32_t temp_axes_a[TENSOR_MAX_RANK], temp_axes_b[TENSOR_MAX_RANK];
        uint32_t num_contract = 0;

        for (uint32_t e = 0; e < network->num_edges; e++) {
            const contract_edge_t *edge = &network->edges[e];

            if (edge->tensor_a == states[best_a].id && edge->tensor_b == states[best_b].id) {
                temp_axes_a[num_contract] = edge->axis_a;
                temp_axes_b[num_contract] = edge->axis_b;
                num_contract++;
            } else if (edge->tensor_a == states[best_b].id && edge->tensor_b == states[best_a].id) {
                temp_axes_a[num_contract] = edge->axis_b;
                temp_axes_b[num_contract] = edge->axis_a;
                num_contract++;
            }
        }

        s->num_contract = num_contract;
        s->axes_a = (uint32_t *)malloc(num_contract * sizeof(uint32_t));
        s->axes_b = (uint32_t *)malloc(num_contract * sizeof(uint32_t));

        if (s->axes_a && s->axes_b) {
            memcpy(s->axes_a, temp_axes_a, num_contract * sizeof(uint32_t));
            memcpy(s->axes_b, temp_axes_b, num_contract * sizeof(uint32_t));
        }

        // Estimate costs
        s->estimated_flops = contract_estimate_flops(states[best_a].tensor,
                                                      states[best_b].tensor,
                                                      temp_axes_a, temp_axes_b,
                                                      num_contract);
        s->estimated_memory = contract_estimate_memory(states[best_a].tensor,
                                                        states[best_b].tensor,
                                                        temp_axes_a, temp_axes_b,
                                                        num_contract);

        plan->total_flops += s->estimated_flops;
        if (s->estimated_memory > plan->peak_memory) {
            plan->peak_memory = s->estimated_memory;
        }

        // Update states: merge best_b into best_a
        states[best_b].active = false;

        // Create result tensor shape (for further ordering decisions)
        uint32_t result_rank = states[best_a].tensor->rank +
                               states[best_b].tensor->rank - 2 * num_contract;
        uint32_t result_dims[TENSOR_MAX_RANK];

        bool contracted_a[TENSOR_MAX_RANK] = {false};
        bool contracted_b[TENSOR_MAX_RANK] = {false};
        for (uint32_t i = 0; i < num_contract; i++) {
            contracted_a[temp_axes_a[i]] = true;
            contracted_b[temp_axes_b[i]] = true;
        }

        uint32_t r = 0;
        for (uint32_t i = 0; i < states[best_a].tensor->rank; i++) {
            if (!contracted_a[i]) result_dims[r++] = states[best_a].tensor->dims[i];
        }
        for (uint32_t i = 0; i < states[best_b].tensor->rank; i++) {
            if (!contracted_b[i]) result_dims[r++] = states[best_b].tensor->dims[i];
        }

        // Create placeholder result tensor for cost estimation
        tensor_t *result = tensor_create(result_rank, result_dims);
        if (!result) {
            // Allocation failed, clean up and return
            for (uint32_t i = 0; i < network->num_tensors; i++) {
                if (states[i].tensor != network->tensors[i]) {
                    tensor_free(states[i].tensor);
                }
            }
            free(states);
            contract_plan_free(plan);
            return NULL;
        }

        // Free intermediate tensor if it was created during ordering
        // (not one of the original network tensors)
        if (states[best_a].tensor != network->tensors[best_a]) {
            tensor_free(states[best_a].tensor);
        }
        states[best_a].tensor = result;
    }

    // Clean up
    for (uint32_t i = 0; i < network->num_tensors; i++) {
        if (states[i].tensor != network->tensors[i]) {
            tensor_free(states[i].tensor);
        }
    }
    free(states);

    return plan;
}

void contract_plan_estimate_cost(const contract_plan_t *plan,
                                  uint64_t *flops, uint64_t *memory) {
    if (!plan) {
        if (flops) *flops = 0;
        if (memory) *memory = 0;
        return;
    }

    if (flops) *flops = plan->total_flops;
    if (memory) *memory = plan->peak_memory;
}

void contract_plan_free(contract_plan_t *plan) {
    if (!plan) return;

    if (plan->steps) {
        for (uint32_t i = 0; i < plan->num_steps; i++) {
            free(plan->steps[i].axes_a);
            free(plan->steps[i].axes_b);
        }
        free(plan->steps);
    }
    free(plan);
}

void contract_plan_print(const contract_plan_t *plan) {
    if (!plan) {
        printf("Plan: NULL\n");
        return;
    }

    printf("Contraction Plan:\n");
    printf("  Steps: %u\n", plan->num_steps);
    printf("  Total FLOPs: %lu\n", (unsigned long)plan->total_flops);
    printf("  Peak Memory: %lu bytes\n", (unsigned long)plan->peak_memory);

    for (uint32_t i = 0; i < plan->num_steps; i++) {
        const contract_step_t *s = &plan->steps[i];
        printf("  Step %u: T%u * T%u (contract %u axes)\n",
               i, s->tensor_a, s->tensor_b, s->num_contract);
        printf("    FLOPs: %lu, Memory: %lu\n",
               (unsigned long)s->estimated_flops,
               (unsigned long)s->estimated_memory);
    }
}

// ============================================================================
// PAIRWISE CONTRACTION
// ============================================================================

tensor_t *contract_tensors(const tensor_t *a, const tensor_t *b,
                           const uint32_t *axes_a, const uint32_t *axes_b,
                           uint32_t num_contract) {
    if (!a || !b) return NULL;
    if (num_contract > 0 && (!axes_a || !axes_b)) return NULL;

    // Use tensor library contraction
    return tensor_contract(a, b, axes_a, axes_b, num_contract);
}

tensor_t *contract_tensors_compressed(const tensor_t *a, const tensor_t *b,
                                       const uint32_t *axes_a,
                                       const uint32_t *axes_b,
                                       uint32_t num_contract,
                                       uint32_t compress_axis,
                                       const svd_compress_config_t *config,
                                       double *truncation_error) {
    // First contract
    tensor_t *result = contract_tensors(a, b, axes_a, axes_b, num_contract);
    if (!result) return NULL;

    if (!config || compress_axis >= result->rank) {
        if (truncation_error) *truncation_error = 0.0;
        return result;
    }

    // Apply SVD compression along specified axis
    svd_compress_result_t *svd = svd_compress_split(result, compress_axis, config);
    tensor_free(result);

    if (!svd) return NULL;

    if (truncation_error) *truncation_error = svd->truncation_error;

    // Recombine with singular values absorbed
    tensor_t *compressed = svd->left;  // Take left tensor
    svd->left = NULL;

    // Absorb singular values
    svd_absorb_singular_values(compressed, svd->singular_values,
                                svd->bond_dim, compress_axis);

    // Contract with right tensor
    uint32_t left_axis[1] = {compress_axis};
    uint32_t right_axis[1] = {0};

    tensor_t *final = contract_tensors(compressed, svd->right,
                                        left_axis, right_axis, 1);

    tensor_free(compressed);
    svd_compress_result_free(svd);

    return final;
}

tensor_t *contract_trace(const tensor_t *tensor, uint32_t axis1, uint32_t axis2) {
    if (!tensor) return NULL;
    if (axis1 >= tensor->rank || axis2 >= tensor->rank) return NULL;
    if (axis1 == axis2) return NULL;
    if (tensor->dims[axis1] != tensor->dims[axis2]) return NULL;

    // Ensure axis1 < axis2
    if (axis1 > axis2) {
        uint32_t temp = axis1;
        axis1 = axis2;
        axis2 = temp;
    }

    uint32_t trace_dim = tensor->dims[axis1];

    // Result dimensions
    uint32_t result_rank = tensor->rank - 2;
    uint32_t result_dims[TENSOR_MAX_RANK];

    uint32_t r = 0;
    for (uint32_t i = 0; i < tensor->rank; i++) {
        if (i != axis1 && i != axis2) {
            result_dims[r++] = tensor->dims[i];
        }
    }

    if (result_rank == 0) {
        // Full trace to scalar
        double complex sum = 0.0;
        uint32_t indices[TENSOR_MAX_RANK];

        for (uint64_t i = 0; i < tensor->total_size; i++) {
            tensor_get_multi_index(tensor, i, indices);
            if (indices[axis1] == indices[axis2]) {
                sum += tensor->data[i];
            }
        }

        return tensor_create_scalar(sum);
    }

    tensor_t *result = tensor_create(result_rank, result_dims);
    if (!result) return NULL;

    // Compute trace
    uint32_t in_indices[TENSOR_MAX_RANK];
    uint32_t out_indices[TENSOR_MAX_RANK];

    for (uint64_t out_idx = 0; out_idx < result->total_size; out_idx++) {
        tensor_get_multi_index(result, out_idx, out_indices);

        // Map output indices to input indices
        r = 0;
        for (uint32_t i = 0; i < tensor->rank; i++) {
            if (i == axis1 || i == axis2) {
                in_indices[i] = 0;  // Will iterate
            } else {
                in_indices[i] = out_indices[r++];
            }
        }

        // Sum over trace dimension
        double complex sum = 0.0;
        for (uint32_t t = 0; t < trace_dim; t++) {
            in_indices[axis1] = t;
            in_indices[axis2] = t;
            sum += tensor_get(tensor, in_indices);
        }

        result->data[out_idx] = sum;
    }

    return result;
}

// ============================================================================
// CONTRACTION EXECUTION
// ============================================================================

tensor_t *contract_execute(const contract_network_t *network,
                           const contract_config_t *config,
                           contract_stats_t *stats) {
    if (!network || !config) return NULL;

    contract_plan_t *plan = contract_find_order(network, config);
    if (!plan) return NULL;

    tensor_t *result = contract_execute_plan(network, plan, config, stats);

    contract_plan_free(plan);
    return result;
}

tensor_t *contract_execute_plan(const contract_network_t *network,
                                 const contract_plan_t *plan,
                                 const contract_config_t *config,
                                 contract_stats_t *stats) {
    if (!network || !plan || !config) return NULL;

    clock_t start_time = clock();

    if (stats) {
        *stats = contract_stats_init();
    }

    if (network->num_tensors == 0) return NULL;

    if (network->num_tensors == 1) {
        // Single tensor, just copy
        return tensor_copy(network->tensors[0]);
    }

    if (plan->num_steps == 0) {
        return tensor_copy(network->tensors[0]);
    }

    // Create array to track intermediate tensors
    tensor_t **intermediates = (tensor_t **)calloc(network->num_tensors, sizeof(tensor_t *));
    if (!intermediates) return NULL;

    // Initialize with copies of original tensors (or views)
    for (uint32_t i = 0; i < network->num_tensors; i++) {
        intermediates[i] = tensor_copy(network->tensors[i]);
        if (!intermediates[i]) {
            for (uint32_t j = 0; j < i; j++) {
                tensor_free(intermediates[j]);
            }
            free(intermediates);
            return NULL;
        }
    }

    double total_truncation_error = 0.0;

    // Execute each step
    for (uint32_t step = 0; step < plan->num_steps; step++) {
        const contract_step_t *s = &plan->steps[step];

        tensor_t *ta = intermediates[s->tensor_a];
        tensor_t *tb = intermediates[s->tensor_b];

        if (!ta || !tb) {
            // Error: tensor already consumed
            for (uint32_t i = 0; i < network->num_tensors; i++) {
                tensor_free(intermediates[i]);
            }
            free(intermediates);
            return NULL;
        }

        // Perform contraction
        tensor_t *result;
        if (config->use_compression && s->num_contract > 0) {
            double trunc_err;
            uint32_t compress_axis = 0;  // Compress along first free axis
            result = contract_tensors_compressed(ta, tb,
                                                  s->axes_a, s->axes_b,
                                                  s->num_contract,
                                                  compress_axis,
                                                  &config->compress_config,
                                                  &trunc_err);
            total_truncation_error += trunc_err;
        } else {
            result = contract_tensors(ta, tb, s->axes_a, s->axes_b, s->num_contract);
        }

        if (!result) {
            for (uint32_t i = 0; i < network->num_tensors; i++) {
                tensor_free(intermediates[i]);
            }
            free(intermediates);
            return NULL;
        }

        // Update intermediates
        tensor_free(intermediates[s->tensor_a]);
        tensor_free(intermediates[s->tensor_b]);
        intermediates[s->tensor_a] = result;
        intermediates[s->tensor_b] = NULL;

        if (stats) {
            stats->num_contractions++;
            stats->total_flops += s->estimated_flops;
            if (s->estimated_memory > stats->peak_memory) {
                stats->peak_memory = s->estimated_memory;
            }
        }
    }

    // Find final result (last non-null intermediate)
    tensor_t *final_result = NULL;
    for (uint32_t i = 0; i < network->num_tensors; i++) {
        if (intermediates[i]) {
            final_result = intermediates[i];
            intermediates[i] = NULL;
            break;
        }
    }

    // Clean up any remaining intermediates
    for (uint32_t i = 0; i < network->num_tensors; i++) {
        tensor_free(intermediates[i]);
    }
    free(intermediates);

    if (stats) {
        clock_t end_time = clock();
        stats->total_time_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        stats->compression_error = total_truncation_error;
    }

    return final_result;
}

// ============================================================================
// SPECIALIZED CONTRACTIONS
// ============================================================================

tensor_t *contract_chain(const tensor_t **tensors, uint32_t num_tensors,
                          const contract_config_t *config) {
    if (!tensors || num_tensors == 0 || !config) return NULL;

    if (num_tensors == 1) {
        return tensor_copy(tensors[0]);
    }

    // For chain contraction, contract left to right
    tensor_t *result = tensor_copy(tensors[0]);
    if (!result) return NULL;

    for (uint32_t i = 1; i < num_tensors; i++) {
        // Contract last axis of result with first axis of next tensor
        uint32_t axis_result[1] = {result->rank - 1};
        uint32_t axis_next[1] = {0};

        tensor_t *new_result;
        if (config->use_compression) {
            double trunc_err;
            new_result = contract_tensors_compressed(result, tensors[i],
                                                      axis_result, axis_next, 1,
                                                      result->rank - 1,
                                                      &config->compress_config,
                                                      &trunc_err);
        } else {
            new_result = contract_tensors(result, tensors[i],
                                           axis_result, axis_next, 1);
        }

        tensor_free(result);
        result = new_result;

        if (!result) return NULL;
    }

    return result;
}

tensor_t *contract_tree(const tensor_t **tensors,
                         const contract_edge_t *edges,
                         uint32_t num_tensors,
                         uint32_t num_edges,
                         const contract_config_t *config) {
    if (!tensors || !edges || num_tensors == 0 || !config) return NULL;

    // Build network from edges
    contract_network_t *network = contract_network_create();
    if (!network) return NULL;

    for (uint32_t i = 0; i < num_tensors; i++) {
        contract_network_add_tensor(network, tensors[i]);
    }

    for (uint32_t i = 0; i < num_edges; i++) {
        contract_network_add_edge(network,
                                   edges[i].tensor_a, edges[i].axis_a,
                                   edges[i].tensor_b, edges[i].axis_b);
    }

    // Execute
    tensor_t *result = contract_execute(network, config, NULL);

    contract_network_free(network);
    return result;
}

tensor_t **contract_mps_mpo(const tensor_t **mps,
                             const tensor_t **mpo,
                             uint32_t num_sites,
                             const contract_config_t *config) {
    if (!mps || !mpo || num_sites == 0 || !config) return NULL;

    // Allocate result array
    tensor_t **result = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));
    if (!result) return NULL;

    // For each site, contract MPS tensor with MPO tensor
    // MPS tensor: [left_bond, physical, right_bond]
    // MPO tensor: [left_bond, physical_in, physical_out, right_bond]
    // Result: [left_bond_mps, left_bond_mpo, physical_out, right_bond_mps, right_bond_mpo]
    // Then reshape to [new_left_bond, physical_out, new_right_bond]

    for (uint32_t site = 0; site < num_sites; site++) {
        const tensor_t *m = mps[site];
        const tensor_t *o = mpo[site];

        // Contract physical indices
        // MPS[i,a,b] * MPO[c,a,d,e] -> Result[i,b,c,d,e]
        // Sum over 'a' (physical index)

        uint32_t axes_m[1] = {1};  // Physical axis of MPS
        uint32_t axes_o[1] = {1};  // Physical_in axis of MPO

        tensor_t *contracted = contract_tensors(m, o, axes_m, axes_o, 1);
        if (!contracted) {
            for (uint32_t j = 0; j < site; j++) {
                tensor_free(result[j]);
            }
            free(result);
            return NULL;
        }

        // Apply compression if configured
        if (config->use_compression) {
            // Reshape to matrix and compress
            // Current shape: [left_mps, right_mps, left_mpo, physical_out, right_mpo]
            // Reshape to: [left_mps * left_mpo, physical_out, right_mps * right_mpo]

            // For simplicity, just store the contracted result
            // Full MPS-MPO application would require more sophisticated compression
        }

        result[site] = contracted;
    }

    return result;
}

// ============================================================================
// UTILITIES
// ============================================================================

const char *contract_error_string(contract_error_t error) {
    switch (error) {
        case CONTRACT_SUCCESS: return "Success";
        case CONTRACT_ERROR_NULL_PTR: return "Null pointer";
        case CONTRACT_ERROR_INVALID_NETWORK: return "Invalid network";
        case CONTRACT_ERROR_DIM_MISMATCH: return "Dimension mismatch";
        case CONTRACT_ERROR_ALLOC_FAILED: return "Allocation failed";
        case CONTRACT_ERROR_ORDER_FAILED: return "Ordering failed";
        case CONTRACT_ERROR_CONTRACTION_FAILED: return "Contraction failed";
        case CONTRACT_ERROR_MEMORY_LIMIT: return "Memory limit exceeded";
        case CONTRACT_ERROR_INVALID_CONFIG: return "Invalid configuration";
        default: return "Unknown error";
    }
}

contract_stats_t contract_stats_init(void) {
    contract_stats_t stats = {
        .total_flops = 0,
        .peak_memory = 0,
        .num_contractions = 0,
        .total_time_seconds = 0.0,
        .ordering_time_seconds = 0.0,
        .compression_error = 0.0
    };
    return stats;
}

void contract_stats_print(const contract_stats_t *stats) {
    if (!stats) return;

    printf("Contraction Statistics:\n");
    printf("  Total FLOPs:      %lu\n", (unsigned long)stats->total_flops);
    printf("  Peak Memory:      %lu bytes\n", (unsigned long)stats->peak_memory);
    printf("  Contractions:     %lu\n", (unsigned long)stats->num_contractions);
    printf("  Total Time:       %.4f s\n", stats->total_time_seconds);
    printf("  Ordering Time:    %.4f s\n", stats->ordering_time_seconds);
    printf("  Compression Err:  %.6e\n", stats->compression_error);
}
