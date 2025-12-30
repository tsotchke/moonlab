/**
 * @file config.h
 * @brief Simulator configuration management
 *
 * Runtime configuration for the quantum simulator:
 * - Backend selection (CPU, GPU, distributed)
 * - Optimization settings (SIMD, threading)
 * - Noise model parameters
 * - Memory management
 * - Logging and debugging
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef UTILS_CONFIG_H
#define UTILS_CONFIG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

#define QSIM_CONFIG_MAX_KEY_LEN     64
#define QSIM_CONFIG_MAX_VALUE_LEN   256
#define QSIM_CONFIG_MAX_ENTRIES     128

// ============================================================================
// BACKEND TYPES
// ============================================================================

/**
 * @brief Compute backend type
 */
typedef enum {
    QSIM_BACKEND_AUTO,          /**< Automatic selection */
    QSIM_BACKEND_CPU,           /**< CPU only */
    QSIM_BACKEND_CPU_SIMD,      /**< CPU with SIMD */
    QSIM_BACKEND_GPU_METAL,     /**< Apple Metal */
    QSIM_BACKEND_GPU_OPENCL,    /**< OpenCL */
    QSIM_BACKEND_GPU_VULKAN,    /**< Vulkan compute */
    QSIM_BACKEND_GPU_CUDA,      /**< NVIDIA CUDA */
    QSIM_BACKEND_DISTRIBUTED    /**< MPI distributed */
} qsim_backend_t;

/**
 * @brief SIMD instruction set
 */
typedef enum {
    QSIM_SIMD_NONE,             /**< No SIMD */
    QSIM_SIMD_SSE2,             /**< SSE2 */
    QSIM_SIMD_AVX,              /**< AVX */
    QSIM_SIMD_AVX2,             /**< AVX2 */
    QSIM_SIMD_AVX512,           /**< AVX-512 */
    QSIM_SIMD_NEON,             /**< ARM NEON */
    QSIM_SIMD_SVE               /**< ARM SVE */
} qsim_simd_t;

/**
 * @brief Threading model
 */
typedef enum {
    QSIM_THREADING_NONE,        /**< Single-threaded */
    QSIM_THREADING_OPENMP,      /**< OpenMP */
    QSIM_THREADING_PTHREADS,    /**< POSIX threads */
    QSIM_THREADING_GCD          /**< Grand Central Dispatch (macOS) */
} qsim_threading_t;

/**
 * @brief Log level
 */
typedef enum {
    QSIM_LOG_NONE,              /**< No logging */
    QSIM_LOG_ERROR,             /**< Errors only */
    QSIM_LOG_WARN,              /**< Warnings and errors */
    QSIM_LOG_INFO,              /**< Info, warnings, errors */
    QSIM_LOG_DEBUG,             /**< Debug and above */
    QSIM_LOG_TRACE              /**< All messages */
} qsim_log_level_t;

// ============================================================================
// CONFIGURATION STRUCTURE
// ============================================================================

/**
 * @brief Noise configuration
 */
typedef struct {
    int enabled;                /**< Enable noise simulation */
    double depolarizing_rate;   /**< Depolarizing probability */
    double amplitude_damping;   /**< Amplitude damping rate */
    double phase_damping;       /**< Phase damping rate */
    double t1_time;             /**< T1 relaxation time (µs) */
    double t2_time;             /**< T2 dephasing time (µs) */
    double gate_time;           /**< Gate execution time (µs) */
    double readout_error_0;     /**< P(1|0) readout error */
    double readout_error_1;     /**< P(0|1) readout error */
} qsim_noise_config_t;

/**
 * @brief Memory configuration
 */
typedef struct {
    size_t max_memory;          /**< Maximum memory usage (bytes, 0=unlimited) */
    size_t alignment;           /**< Memory alignment (bytes) */
    int use_huge_pages;         /**< Use huge pages if available */
    int preallocate;            /**< Preallocate state memory */
} qsim_memory_config_t;

/**
 * @brief Threading configuration
 */
typedef struct {
    qsim_threading_t model;     /**< Threading model */
    int num_threads;            /**< Number of threads (0=auto) */
    int thread_affinity;        /**< Enable CPU affinity */
} qsim_thread_config_t;

/**
 * @brief GPU configuration
 */
typedef struct {
    int device_id;              /**< GPU device ID (-1=auto) */
    size_t max_vram;            /**< Maximum VRAM usage */
    int async_transfers;        /**< Enable async memory transfers */
    int kernel_fusion;          /**< Enable kernel fusion */
} qsim_gpu_config_t;

/**
 * @brief Main configuration structure
 */
typedef struct {
    // Backend selection
    qsim_backend_t backend;     /**< Compute backend */
    qsim_simd_t simd;           /**< SIMD instruction set */

    // Limits
    int max_qubits;             /**< Maximum qubits (0=no limit) */
    uint64_t max_state_dim;     /**< Maximum state dimension */

    // Sub-configurations
    qsim_noise_config_t noise;
    qsim_memory_config_t memory;
    qsim_thread_config_t threading;
    qsim_gpu_config_t gpu;

    // Logging
    qsim_log_level_t log_level;
    const char* log_file;       /**< Log file path (NULL=stderr) */

    // Validation
    int validate_states;        /**< Validate state normalization */
    int check_unitarity;        /**< Check gate unitarity */
    double tolerance;           /**< Numerical tolerance */

    // Performance
    int enable_caching;         /**< Cache gate matrices */
    int gate_fusion;            /**< Fuse sequential gates */
    int circuit_optimization;   /**< Optimize circuits */

    // Random number generation
    int use_quantum_rng;        /**< Use quantum RNG for measurements */
    uint64_t seed;              /**< PRNG seed (0=random) */
} qsim_config_t;

// ============================================================================
// GLOBAL CONFIGURATION
// ============================================================================

/**
 * @brief Get global configuration
 *
 * @return Pointer to global configuration
 */
qsim_config_t* qsim_config_global(void);

/**
 * @brief Initialize global configuration with defaults
 *
 * @return 0 on success, -1 on error
 */
int qsim_config_init(void);

/**
 * @brief Cleanup global configuration
 */
void qsim_config_cleanup(void);

// ============================================================================
// CONFIGURATION CREATION
// ============================================================================

/**
 * @brief Create configuration with defaults
 *
 * @return New configuration or NULL on error
 */
qsim_config_t* qsim_config_create(void);

/**
 * @brief Create copy of configuration
 *
 * @param config Configuration to copy
 * @return New configuration or NULL on error
 */
qsim_config_t* qsim_config_copy(const qsim_config_t* config);

/**
 * @brief Destroy configuration
 *
 * @param config Configuration to destroy
 */
void qsim_config_destroy(qsim_config_t* config);

/**
 * @brief Reset configuration to defaults
 *
 * @param config Configuration to reset
 */
void qsim_config_reset(qsim_config_t* config);

// ============================================================================
// CONFIGURATION SETTERS
// ============================================================================

/**
 * @brief Set compute backend
 */
void qsim_config_set_backend(qsim_config_t* config, qsim_backend_t backend);

/**
 * @brief Set SIMD instruction set
 */
void qsim_config_set_simd(qsim_config_t* config, qsim_simd_t simd);

/**
 * @brief Set maximum qubits
 */
void qsim_config_set_max_qubits(qsim_config_t* config, int max_qubits);

/**
 * @brief Set number of threads
 */
void qsim_config_set_threads(qsim_config_t* config, int num_threads);

/**
 * @brief Set log level
 */
void qsim_config_set_log_level(qsim_config_t* config, qsim_log_level_t level);

/**
 * @brief Set log file
 */
void qsim_config_set_log_file(qsim_config_t* config, const char* path);

/**
 * @brief Enable/disable noise simulation
 */
void qsim_config_set_noise_enabled(qsim_config_t* config, int enabled);

/**
 * @brief Set noise model parameters
 */
void qsim_config_set_noise_params(qsim_config_t* config,
                                   double depolarizing,
                                   double amplitude_damping,
                                   double phase_damping);

/**
 * @brief Set thermal relaxation parameters
 */
void qsim_config_set_thermal(qsim_config_t* config,
                             double t1, double t2, double gate_time);

/**
 * @brief Set memory limit
 */
void qsim_config_set_max_memory(qsim_config_t* config, size_t bytes);

/**
 * @brief Set GPU device
 */
void qsim_config_set_gpu_device(qsim_config_t* config, int device_id);

/**
 * @brief Set random seed
 */
void qsim_config_set_seed(qsim_config_t* config, uint64_t seed);

/**
 * @brief Set numerical tolerance
 */
void qsim_config_set_tolerance(qsim_config_t* config, double tolerance);

// ============================================================================
// CONFIGURATION GETTERS
// ============================================================================

/**
 * @brief Get current backend
 */
qsim_backend_t qsim_config_get_backend(const qsim_config_t* config);

/**
 * @brief Get current SIMD setting
 */
qsim_simd_t qsim_config_get_simd(const qsim_config_t* config);

/**
 * @brief Get maximum qubits
 */
int qsim_config_get_max_qubits(const qsim_config_t* config);

/**
 * @brief Get number of threads
 */
int qsim_config_get_threads(const qsim_config_t* config);

/**
 * @brief Check if noise is enabled
 */
int qsim_config_is_noise_enabled(const qsim_config_t* config);

// ============================================================================
// FILE I/O
// ============================================================================

/**
 * @brief Load configuration from file
 *
 * Supports JSON and INI formats.
 *
 * @param path File path
 * @return Loaded configuration or NULL on error
 */
qsim_config_t* qsim_config_load(const char* path);

/**
 * @brief Save configuration to file
 *
 * @param config Configuration to save
 * @param path File path
 * @return 0 on success, -1 on error
 */
int qsim_config_save(const qsim_config_t* config, const char* path);

/**
 * @brief Load configuration from JSON string
 *
 * @param json JSON string
 * @return Loaded configuration or NULL on error
 */
qsim_config_t* qsim_config_from_json(const char* json);

/**
 * @brief Convert configuration to JSON string
 *
 * @param config Configuration
 * @return JSON string (caller must free) or NULL on error
 */
char* qsim_config_to_json(const qsim_config_t* config);

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

/**
 * @brief Load configuration from environment variables
 *
 * Recognized variables:
 * - QSIM_BACKEND: auto, cpu, gpu_metal, gpu_opencl, etc.
 * - QSIM_SIMD: none, sse2, avx, avx2, avx512, neon, sve
 * - QSIM_MAX_QUBITS: integer
 * - QSIM_THREADS: integer (0=auto)
 * - QSIM_LOG_LEVEL: none, error, warn, info, debug, trace
 * - QSIM_LOG_FILE: path
 * - QSIM_NOISE: 0 or 1
 * - QSIM_SEED: integer
 *
 * @param config Configuration to update
 * @return Number of variables loaded
 */
int qsim_config_from_env(qsim_config_t* config);

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * @brief Validate configuration
 *
 * Checks for invalid combinations and unsupported features.
 *
 * @param config Configuration to validate
 * @param error_msg Output buffer for error message (may be NULL)
 * @param error_len Size of error buffer
 * @return 0 if valid, error code otherwise
 */
int qsim_config_validate(const qsim_config_t* config,
                         char* error_msg, size_t error_len);

/**
 * @brief Check if backend is available on this system
 *
 * @param backend Backend to check
 * @return 1 if available, 0 otherwise
 */
int qsim_backend_available(qsim_backend_t backend);

/**
 * @brief Check if SIMD is available on this system
 *
 * @param simd SIMD to check
 * @return 1 if available, 0 otherwise
 */
int qsim_simd_available(qsim_simd_t simd);

// ============================================================================
// AUTO-DETECTION
// ============================================================================

/**
 * @brief Detect best backend for this system
 *
 * @return Recommended backend
 */
qsim_backend_t qsim_detect_backend(void);

/**
 * @brief Detect best SIMD for this system
 *
 * @return Recommended SIMD
 */
qsim_simd_t qsim_detect_simd(void);

/**
 * @brief Detect optimal thread count
 *
 * @return Recommended number of threads
 */
int qsim_detect_threads(void);

/**
 * @brief Detect maximum safe qubits based on available memory
 *
 * @return Maximum recommended qubits
 */
int qsim_detect_max_qubits(void);

// ============================================================================
// STRING CONVERSION
// ============================================================================

/**
 * @brief Convert backend to string
 */
const char* qsim_backend_to_string(qsim_backend_t backend);

/**
 * @brief Parse backend from string
 */
qsim_backend_t qsim_backend_from_string(const char* str);

/**
 * @brief Convert SIMD to string
 */
const char* qsim_simd_to_string(qsim_simd_t simd);

/**
 * @brief Parse SIMD from string
 */
qsim_simd_t qsim_simd_from_string(const char* str);

/**
 * @brief Convert log level to string
 */
const char* qsim_log_level_to_string(qsim_log_level_t level);

/**
 * @brief Parse log level from string
 */
qsim_log_level_t qsim_log_level_from_string(const char* str);

// ============================================================================
// PRESETS
// ============================================================================

/**
 * @brief Load high-performance preset
 *
 * Enables all optimizations, uses GPU if available.
 */
void qsim_config_preset_performance(qsim_config_t* config);

/**
 * @brief Load accuracy preset
 *
 * Enables validation, disables approximations.
 */
void qsim_config_preset_accuracy(qsim_config_t* config);

/**
 * @brief Load low-memory preset
 *
 * Minimizes memory usage, may be slower.
 */
void qsim_config_preset_low_memory(qsim_config_t* config);

/**
 * @brief Load noisy simulation preset
 *
 * Enables realistic noise model based on hardware parameters.
 */
void qsim_config_preset_noisy(qsim_config_t* config,
                              double t1_us, double t2_us,
                              double gate_error);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_CONFIG_H */
