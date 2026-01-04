/**
 * @file contraction.h
 * @brief Tensor contraction engine for tensor network operations
 *
 * Provides optimized tensor contraction with:
 * - Pairwise and multi-tensor contraction
 * - Contraction order optimization (greedy and dynamic programming)
 * - Memory-efficient streaming contraction
 * - SIMD-accelerated inner loops
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef CONTRACTION_H
#define CONTRACTION_H

#include "tensor.h"
#include "svd_compress.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/** Maximum tensors in a contraction network */
#define CONTRACT_MAX_TENSORS 64

/** Maximum edges in contraction graph */
#define CONTRACT_MAX_EDGES 256

/** Default memory limit for contraction (bytes) */
#define CONTRACT_DEFAULT_MEM_LIMIT (1ULL << 32)  // 4 GB

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    CONTRACT_SUCCESS = 0,
    CONTRACT_ERROR_NULL_PTR = -1,
    CONTRACT_ERROR_INVALID_NETWORK = -2,
    CONTRACT_ERROR_DIM_MISMATCH = -3,
    CONTRACT_ERROR_ALLOC_FAILED = -4,
    CONTRACT_ERROR_ORDER_FAILED = -5,
    CONTRACT_ERROR_CONTRACTION_FAILED = -6,
    CONTRACT_ERROR_MEMORY_LIMIT = -7,
    CONTRACT_ERROR_INVALID_CONFIG = -8
} contract_error_t;

// ============================================================================
// CONTRACTION SPECIFICATION
// ============================================================================

/**
 * @brief Edge in contraction graph
 *
 * Represents a contracted index between two tensors.
 */
typedef struct {
    uint32_t tensor_a;              /**< First tensor index */
    uint32_t tensor_b;              /**< Second tensor index */
    uint32_t axis_a;                /**< Axis in first tensor */
    uint32_t axis_b;                /**< Axis in second tensor */
    uint32_t dimension;             /**< Size of contracted index */
} contract_edge_t;

/**
 * @brief Contraction network specification
 *
 * Describes the structure of a tensor network to be contracted.
 */
typedef struct {
    uint32_t num_tensors;           /**< Number of tensors */
    uint32_t num_edges;             /**< Number of contracted edges */
    const tensor_t **tensors;       /**< Array of tensor pointers */
    contract_edge_t edges[CONTRACT_MAX_EDGES]; /**< Contraction edges */
    uint32_t *output_tensor_axes;   /**< Which axes appear in output (NULL = trace) */
    uint32_t num_output_axes;       /**< Number of output axes */
} contract_network_t;

/**
 * @brief Contraction step in execution plan
 */
typedef struct {
    uint32_t tensor_a;              /**< First tensor to contract */
    uint32_t tensor_b;              /**< Second tensor to contract */
    uint32_t *axes_a;               /**< Axes of A to contract */
    uint32_t *axes_b;               /**< Axes of B to contract */
    uint32_t num_contract;          /**< Number of contracted axes */
    uint64_t estimated_flops;       /**< Estimated FLOPs for this step */
    uint64_t estimated_memory;      /**< Estimated memory for result */
} contract_step_t;

/**
 * @brief Contraction execution plan
 */
typedef struct {
    uint32_t num_steps;             /**< Number of contraction steps */
    contract_step_t *steps;         /**< Array of steps */
    uint64_t total_flops;           /**< Total estimated FLOPs */
    uint64_t peak_memory;           /**< Peak memory requirement */
} contract_plan_t;

/**
 * @brief Optimization strategy for contraction ordering
 */
typedef enum {
    CONTRACT_ORDER_NAIVE,           /**< Left-to-right order (fast, suboptimal) */
    CONTRACT_ORDER_GREEDY,          /**< Greedy by contraction cost */
    CONTRACT_ORDER_GREEDY_MEMORY,   /**< Greedy prioritizing memory */
    CONTRACT_ORDER_DYNAMIC,         /**< Dynamic programming (optimal for small nets) */
    CONTRACT_ORDER_BRANCH_BOUND     /**< Branch and bound (near-optimal) */
} contract_order_strategy_t;

/**
 * @brief Contraction configuration
 */
typedef struct {
    contract_order_strategy_t strategy; /**< Ordering strategy */
    uint64_t memory_limit;          /**< Maximum intermediate memory */
    bool use_compression;           /**< Apply SVD compression */
    svd_compress_config_t compress_config; /**< Compression settings */
    bool track_stats;               /**< Track contraction statistics */
    bool parallel;                  /**< Use parallel execution */
    uint32_t num_threads;           /**< Thread count (0 = auto) */
} contract_config_t;

/**
 * @brief Contraction execution statistics
 */
typedef struct {
    uint64_t total_flops;           /**< Total FLOPs executed */
    uint64_t peak_memory;           /**< Peak memory used */
    uint64_t num_contractions;      /**< Number of pairwise contractions */
    double total_time_seconds;      /**< Total execution time */
    double ordering_time_seconds;   /**< Time spent on ordering */
    double compression_error;       /**< Total compression error */
} contract_stats_t;

// ============================================================================
// CONFIGURATION MANAGEMENT
// ============================================================================

/**
 * @brief Create default contraction configuration
 *
 * @return Default configuration
 */
contract_config_t contract_config_default(void);

/**
 * @brief Create configuration optimized for memory
 *
 * @param mem_limit Maximum memory in bytes
 * @return Memory-optimized configuration
 */
contract_config_t contract_config_memory_limited(uint64_t mem_limit);

/**
 * @brief Create configuration with SVD compression
 *
 * @param max_bond Maximum bond dimension
 * @param cutoff SVD cutoff
 * @return Compression-enabled configuration
 */
contract_config_t contract_config_compressed(uint32_t max_bond, double cutoff);

// ============================================================================
// NETWORK CONSTRUCTION
// ============================================================================

/**
 * @brief Initialize empty contraction network
 *
 * @return Initialized network
 */
contract_network_t *contract_network_create(void);

/**
 * @brief Add tensor to network
 *
 * @param network Network to modify
 * @param tensor Tensor to add (reference stored, not copied)
 * @return Index of added tensor or -1 on error
 */
int contract_network_add_tensor(contract_network_t *network,
                                 const tensor_t *tensor);

/**
 * @brief Add contraction edge between tensors
 *
 * @param network Network to modify
 * @param tensor_a First tensor index
 * @param axis_a Axis of first tensor
 * @param tensor_b Second tensor index
 * @param axis_b Axis of second tensor
 * @return CONTRACT_SUCCESS or error code
 */
contract_error_t contract_network_add_edge(contract_network_t *network,
                                            uint32_t tensor_a, uint32_t axis_a,
                                            uint32_t tensor_b, uint32_t axis_b);

/**
 * @brief Specify output axes (uncontracted indices)
 *
 * @param network Network to modify
 * @param tensor_axes Array of (tensor_idx, axis) pairs
 * @param num_axes Number of output axes
 * @return CONTRACT_SUCCESS or error code
 */
contract_error_t contract_network_set_output(contract_network_t *network,
                                              const uint32_t *tensor_axes,
                                              uint32_t num_axes);

/**
 * @brief Validate network specification
 *
 * @param network Network to validate
 * @return CONTRACT_SUCCESS if valid
 */
contract_error_t contract_network_validate(const contract_network_t *network);

/**
 * @brief Free network structure
 *
 * @param network Network to free
 */
void contract_network_free(contract_network_t *network);

// ============================================================================
// CONTRACTION ORDERING
// ============================================================================

/**
 * @brief Compute optimal contraction order
 *
 * Analyzes the network and produces an execution plan.
 *
 * @param network Network to analyze
 * @param config Configuration (strategy, memory limit)
 * @return Execution plan or NULL on failure
 */
contract_plan_t *contract_find_order(const contract_network_t *network,
                                      const contract_config_t *config);

/**
 * @brief Estimate cost of a contraction plan
 *
 * @param plan Execution plan
 * @param flops Output: total FLOPs
 * @param memory Output: peak memory
 */
void contract_plan_estimate_cost(const contract_plan_t *plan,
                                  uint64_t *flops, uint64_t *memory);

/**
 * @brief Free execution plan
 *
 * @param plan Plan to free
 */
void contract_plan_free(contract_plan_t *plan);

/**
 * @brief Print execution plan details
 *
 * @param plan Plan to print
 */
void contract_plan_print(const contract_plan_t *plan);

// ============================================================================
// CONTRACTION EXECUTION
// ============================================================================

/**
 * @brief Execute contraction network
 *
 * Contracts all tensors according to the network specification.
 *
 * @param network Network to contract
 * @param config Configuration
 * @param stats Output statistics (can be NULL)
 * @return Contracted result tensor or NULL on failure
 */
tensor_t *contract_execute(const contract_network_t *network,
                           const contract_config_t *config,
                           contract_stats_t *stats);

/**
 * @brief Execute contraction with specific plan
 *
 * @param network Network to contract
 * @param plan Pre-computed execution plan
 * @param config Configuration
 * @param stats Output statistics (can be NULL)
 * @return Contracted result tensor or NULL on failure
 */
tensor_t *contract_execute_plan(const contract_network_t *network,
                                 const contract_plan_t *plan,
                                 const contract_config_t *config,
                                 contract_stats_t *stats);

// ============================================================================
// PAIRWISE CONTRACTION
// ============================================================================

/**
 * @brief Contract two tensors
 *
 * Low-level pairwise contraction with explicit axis specification.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param axes_a Axes of A to contract
 * @param axes_b Axes of B to contract
 * @param num_contract Number of axes to contract
 * @return Contracted tensor or NULL on failure
 */
tensor_t *contract_tensors(const tensor_t *a, const tensor_t *b,
                           const uint32_t *axes_a, const uint32_t *axes_b,
                           uint32_t num_contract);

/**
 * @brief Contract tensors with compression
 *
 * Contracts two tensors and applies SVD truncation to result.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param axes_a Axes of A to contract
 * @param axes_b Axes of B to contract
 * @param num_contract Number of axes to contract
 * @param compress_axis Axis of result to compress
 * @param config Compression configuration
 * @param truncation_error Output: truncation error (can be NULL)
 * @return Contracted and compressed tensor or NULL on failure
 */
tensor_t *contract_tensors_compressed(const tensor_t *a, const tensor_t *b,
                                       const uint32_t *axes_a,
                                       const uint32_t *axes_b,
                                       uint32_t num_contract,
                                       uint32_t compress_axis,
                                       const svd_compress_config_t *config,
                                       double *truncation_error);

/**
 * @brief Contract tensor with itself (trace operation)
 *
 * @param tensor Input tensor
 * @param axis1 First axis to trace
 * @param axis2 Second axis to trace
 * @return Traced tensor or NULL on failure
 */
tensor_t *contract_trace(const tensor_t *tensor, uint32_t axis1, uint32_t axis2);

// ============================================================================
// SPECIALIZED CONTRACTIONS
// ============================================================================

/**
 * @brief Contract chain of tensors (MPS-like)
 *
 * Efficiently contracts A[1] * A[2] * ... * A[n] where each
 * tensor connects to neighbors.
 *
 * @param tensors Array of tensors
 * @param num_tensors Number of tensors
 * @param config Contraction configuration
 * @return Contracted result or NULL on failure
 */
tensor_t *contract_chain(const tensor_t **tensors, uint32_t num_tensors,
                          const contract_config_t *config);

/**
 * @brief Contract tree-structured network
 *
 * Optimized for tree tensor networks (no loops).
 *
 * @param tensors Array of tensors
 * @param edges Edge connectivity (pairs)
 * @param num_tensors Number of tensors
 * @param num_edges Number of edges
 * @param config Configuration
 * @return Contracted result or NULL on failure
 */
tensor_t *contract_tree(const tensor_t **tensors,
                         const contract_edge_t *edges,
                         uint32_t num_tensors,
                         uint32_t num_edges,
                         const contract_config_t *config);

/**
 * @brief Contract MPS with MPO
 *
 * Applies matrix product operator to matrix product state.
 * Commonly used for time evolution and expectation values.
 *
 * @param mps Array of MPS tensors [num_sites]
 * @param mpo Array of MPO tensors [num_sites]
 * @param num_sites Number of sites
 * @param config Configuration (compression important)
 * @return Resulting MPS tensors or NULL on failure
 */
tensor_t **contract_mps_mpo(const tensor_t **mps,
                             const tensor_t **mpo,
                             uint32_t num_sites,
                             const contract_config_t *config);

// ============================================================================
// COST ESTIMATION
// ============================================================================

/**
 * @brief Estimate FLOPs for pairwise contraction
 *
 * @param a First tensor
 * @param b Second tensor
 * @param axes_a Contracted axes of A
 * @param axes_b Contracted axes of B
 * @param num_contract Number of contracted axes
 * @return Estimated FLOPs
 */
uint64_t contract_estimate_flops(const tensor_t *a, const tensor_t *b,
                                  const uint32_t *axes_a,
                                  const uint32_t *axes_b,
                                  uint32_t num_contract);

/**
 * @brief Estimate memory for contraction result
 *
 * @param a First tensor
 * @param b Second tensor
 * @param axes_a Contracted axes of A
 * @param axes_b Contracted axes of B
 * @param num_contract Number of contracted axes
 * @return Estimated memory in bytes
 */
uint64_t contract_estimate_memory(const tensor_t *a, const tensor_t *b,
                                   const uint32_t *axes_a,
                                   const uint32_t *axes_b,
                                   uint32_t num_contract);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *contract_error_string(contract_error_t error);

/**
 * @brief Initialize statistics structure
 *
 * @return Initialized statistics
 */
contract_stats_t contract_stats_init(void);

/**
 * @brief Print statistics summary
 *
 * @param stats Statistics to print
 */
void contract_stats_print(const contract_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif /* CONTRACTION_H */
