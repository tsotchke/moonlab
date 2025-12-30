/**
 * @file config.c
 * @brief Simulator configuration management implementation
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

#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static qsim_config_t* g_config = NULL;
static int g_initialized = 0;

// ============================================================================
// DEFAULT VALUES
// ============================================================================

static void set_defaults(qsim_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(qsim_config_t));

    // Backend
    config->backend = QSIM_BACKEND_AUTO;
    config->simd = QSIM_SIMD_NONE;

    // Limits
    config->max_qubits = 0;  // No limit
    config->max_state_dim = 0;

    // Noise (disabled by default)
    config->noise.enabled = 0;
    config->noise.depolarizing_rate = 0.001;
    config->noise.amplitude_damping = 0.0001;
    config->noise.phase_damping = 0.0001;
    config->noise.t1_time = 50.0;    // 50 µs
    config->noise.t2_time = 70.0;    // 70 µs
    config->noise.gate_time = 0.02;  // 20 ns
    config->noise.readout_error_0 = 0.01;
    config->noise.readout_error_1 = 0.015;

    // Memory
    config->memory.max_memory = 0;  // Unlimited
    config->memory.alignment = 64;  // Cache line
    config->memory.use_huge_pages = 0;
    config->memory.preallocate = 0;

    // Threading
    config->threading.model = QSIM_THREADING_OPENMP;
    config->threading.num_threads = 0;  // Auto
    config->threading.thread_affinity = 0;

    // GPU
    config->gpu.device_id = -1;  // Auto
    config->gpu.max_vram = 0;
    config->gpu.async_transfers = 1;
    config->gpu.kernel_fusion = 1;

    // Logging
    config->log_level = QSIM_LOG_WARN;
    config->log_file = NULL;

    // Validation
    config->validate_states = 0;
    config->check_unitarity = 0;
    config->tolerance = 1e-10;

    // Performance
    config->enable_caching = 1;
    config->gate_fusion = 1;
    config->circuit_optimization = 0;

    // RNG
    config->use_quantum_rng = 0;
    config->seed = 0;
}

// ============================================================================
// GLOBAL CONFIGURATION
// ============================================================================

qsim_config_t* qsim_config_global(void) {
    if (!g_initialized) {
        qsim_config_init();
    }
    return g_config;
}

int qsim_config_init(void) {
    if (g_initialized) return 0;

    g_config = calloc(1, sizeof(qsim_config_t));
    if (!g_config) return -1;

    set_defaults(g_config);

    // Auto-detect best settings
    g_config->simd = qsim_detect_simd();
    g_config->threading.num_threads = qsim_detect_threads();

    // Load from environment
    qsim_config_from_env(g_config);

    g_initialized = 1;
    return 0;
}

void qsim_config_cleanup(void) {
    if (g_config) {
        free(g_config);
        g_config = NULL;
    }
    g_initialized = 0;
}

// ============================================================================
// CONFIGURATION CREATION
// ============================================================================

qsim_config_t* qsim_config_create(void) {
    qsim_config_t* config = calloc(1, sizeof(qsim_config_t));
    if (!config) return NULL;

    set_defaults(config);
    return config;
}

qsim_config_t* qsim_config_copy(const qsim_config_t* config) {
    if (!config) return NULL;

    qsim_config_t* copy = calloc(1, sizeof(qsim_config_t));
    if (!copy) return NULL;

    memcpy(copy, config, sizeof(qsim_config_t));

    // Deep copy strings
    if (config->log_file) {
        copy->log_file = strdup(config->log_file);
    }

    return copy;
}

void qsim_config_destroy(qsim_config_t* config) {
    if (!config) return;
    if (config == g_config) {
        // Don't destroy global config
        return;
    }
    free(config);
}

void qsim_config_reset(qsim_config_t* config) {
    if (!config) return;
    set_defaults(config);
}

// ============================================================================
// SETTERS
// ============================================================================

void qsim_config_set_backend(qsim_config_t* config, qsim_backend_t backend) {
    if (config) config->backend = backend;
}

void qsim_config_set_simd(qsim_config_t* config, qsim_simd_t simd) {
    if (config) config->simd = simd;
}

void qsim_config_set_max_qubits(qsim_config_t* config, int max_qubits) {
    if (config) config->max_qubits = max_qubits;
}

void qsim_config_set_threads(qsim_config_t* config, int num_threads) {
    if (config) config->threading.num_threads = num_threads;
}

void qsim_config_set_log_level(qsim_config_t* config, qsim_log_level_t level) {
    if (config) config->log_level = level;
}

void qsim_config_set_log_file(qsim_config_t* config, const char* path) {
    if (config) config->log_file = path;
}

void qsim_config_set_noise_enabled(qsim_config_t* config, int enabled) {
    if (config) config->noise.enabled = enabled;
}

void qsim_config_set_noise_params(qsim_config_t* config,
                                   double depolarizing,
                                   double amplitude_damping,
                                   double phase_damping) {
    if (!config) return;
    config->noise.depolarizing_rate = depolarizing;
    config->noise.amplitude_damping = amplitude_damping;
    config->noise.phase_damping = phase_damping;
}

void qsim_config_set_thermal(qsim_config_t* config,
                             double t1, double t2, double gate_time) {
    if (!config) return;
    config->noise.t1_time = t1;
    config->noise.t2_time = t2;
    config->noise.gate_time = gate_time;
}

void qsim_config_set_max_memory(qsim_config_t* config, size_t bytes) {
    if (config) config->memory.max_memory = bytes;
}

void qsim_config_set_gpu_device(qsim_config_t* config, int device_id) {
    if (config) config->gpu.device_id = device_id;
}

void qsim_config_set_seed(qsim_config_t* config, uint64_t seed) {
    if (config) config->seed = seed;
}

void qsim_config_set_tolerance(qsim_config_t* config, double tolerance) {
    if (config) config->tolerance = tolerance;
}

// ============================================================================
// GETTERS
// ============================================================================

qsim_backend_t qsim_config_get_backend(const qsim_config_t* config) {
    return config ? config->backend : QSIM_BACKEND_AUTO;
}

qsim_simd_t qsim_config_get_simd(const qsim_config_t* config) {
    return config ? config->simd : QSIM_SIMD_NONE;
}

int qsim_config_get_max_qubits(const qsim_config_t* config) {
    return config ? config->max_qubits : 0;
}

int qsim_config_get_threads(const qsim_config_t* config) {
    return config ? config->threading.num_threads : 1;
}

int qsim_config_is_noise_enabled(const qsim_config_t* config) {
    return config ? config->noise.enabled : 0;
}

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

int qsim_config_from_env(qsim_config_t* config) {
    if (!config) return 0;

    int count = 0;
    char* val;

    // Backend
    val = getenv("QSIM_BACKEND");
    if (val) {
        config->backend = qsim_backend_from_string(val);
        count++;
    }

    // SIMD
    val = getenv("QSIM_SIMD");
    if (val) {
        config->simd = qsim_simd_from_string(val);
        count++;
    }

    // Max qubits
    val = getenv("QSIM_MAX_QUBITS");
    if (val) {
        config->max_qubits = atoi(val);
        count++;
    }

    // Threads
    val = getenv("QSIM_THREADS");
    if (val) {
        config->threading.num_threads = atoi(val);
        count++;
    }

    // Log level
    val = getenv("QSIM_LOG_LEVEL");
    if (val) {
        config->log_level = qsim_log_level_from_string(val);
        count++;
    }

    // Log file
    val = getenv("QSIM_LOG_FILE");
    if (val) {
        config->log_file = val;
        count++;
    }

    // Noise
    val = getenv("QSIM_NOISE");
    if (val) {
        config->noise.enabled = (atoi(val) != 0);
        count++;
    }

    // Seed
    val = getenv("QSIM_SEED");
    if (val) {
        config->seed = strtoull(val, NULL, 10);
        count++;
    }

    return count;
}

// ============================================================================
// VALIDATION
// ============================================================================

int qsim_config_validate(const qsim_config_t* config,
                         char* error_msg, size_t error_len) {
    if (!config) {
        if (error_msg) snprintf(error_msg, error_len, "NULL configuration");
        return -1;
    }

    // Check backend availability
    if (config->backend != QSIM_BACKEND_AUTO &&
        !qsim_backend_available(config->backend)) {
        if (error_msg) {
            snprintf(error_msg, error_len, "Backend %s not available",
                    qsim_backend_to_string(config->backend));
        }
        return -2;
    }

    // Check SIMD availability
    if (config->simd != QSIM_SIMD_NONE &&
        !qsim_simd_available(config->simd)) {
        if (error_msg) {
            snprintf(error_msg, error_len, "SIMD %s not available",
                    qsim_simd_to_string(config->simd));
        }
        return -3;
    }

    // Check thread count
    if (config->threading.num_threads < 0) {
        if (error_msg) snprintf(error_msg, error_len, "Invalid thread count");
        return -4;
    }

    // Check tolerance
    if (config->tolerance <= 0.0) {
        if (error_msg) snprintf(error_msg, error_len, "Invalid tolerance");
        return -5;
    }

    // Check noise parameters
    if (config->noise.enabled) {
        if (config->noise.t1_time <= 0 || config->noise.t2_time <= 0) {
            if (error_msg) snprintf(error_msg, error_len, "Invalid T1/T2 times");
            return -6;
        }
        if (config->noise.t2_time > 2 * config->noise.t1_time) {
            if (error_msg) snprintf(error_msg, error_len, "T2 cannot exceed 2*T1");
            return -7;
        }
    }

    return 0;
}

// ============================================================================
// AUTO-DETECTION
// ============================================================================

int qsim_backend_available(qsim_backend_t backend) {
    switch (backend) {
        case QSIM_BACKEND_AUTO:
        case QSIM_BACKEND_CPU:
            return 1;

        case QSIM_BACKEND_CPU_SIMD:
            return qsim_detect_simd() != QSIM_SIMD_NONE;

        case QSIM_BACKEND_GPU_METAL:
#ifdef __APPLE__
            return 1;  // Assume Metal available on macOS
#else
            return 0;
#endif

        case QSIM_BACKEND_GPU_OPENCL:
#ifdef QSIM_HAS_OPENCL
            return 1;
#else
            return 0;
#endif

        case QSIM_BACKEND_GPU_VULKAN:
#ifdef QSIM_HAS_VULKAN
            return 1;
#else
            return 0;
#endif

        case QSIM_BACKEND_GPU_CUDA:
#ifdef QSIM_HAS_CUDA
            return 1;
#else
            return 0;
#endif

        case QSIM_BACKEND_DISTRIBUTED:
#ifdef QSIM_HAS_MPI
            return 1;
#else
            return 0;
#endif

        default:
            return 0;
    }
}

int qsim_simd_available(qsim_simd_t simd) {
    switch (simd) {
        case QSIM_SIMD_NONE:
            return 1;

#if defined(__x86_64__) || defined(__i386__)
        case QSIM_SIMD_SSE2:
            return 1;  // Baseline for x86_64

        case QSIM_SIMD_AVX:
        case QSIM_SIMD_AVX2:
        case QSIM_SIMD_AVX512:
            // Would need CPUID check
            return 1;  // Simplified
#endif

#if defined(__aarch64__)
        case QSIM_SIMD_NEON:
            return 1;  // Baseline for ARM64

        case QSIM_SIMD_SVE:
            // Would need hwcap check
            return 0;  // Conservative
#endif

        default:
            return 0;
    }
}

qsim_backend_t qsim_detect_backend(void) {
#ifdef __APPLE__
    return QSIM_BACKEND_GPU_METAL;
#elif defined(QSIM_HAS_CUDA)
    return QSIM_BACKEND_GPU_CUDA;
#elif defined(QSIM_HAS_OPENCL)
    return QSIM_BACKEND_GPU_OPENCL;
#else
    return QSIM_BACKEND_CPU_SIMD;
#endif
}

qsim_simd_t qsim_detect_simd(void) {
#if defined(__x86_64__) || defined(__i386__)
    // Simplified detection - could use CPUID
    #if defined(__AVX512F__)
        return QSIM_SIMD_AVX512;
    #elif defined(__AVX2__)
        return QSIM_SIMD_AVX2;
    #elif defined(__AVX__)
        return QSIM_SIMD_AVX;
    #else
        return QSIM_SIMD_SSE2;
    #endif
#elif defined(__aarch64__)
    return QSIM_SIMD_NEON;
#else
    return QSIM_SIMD_NONE;
#endif
}

int qsim_detect_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#elif defined(__APPLE__)
    int count;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0) {
        return count;
    }
    return 4;
#elif defined(__linux__)
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 4;
#endif
}

int qsim_detect_max_qubits(void) {
    // Get available memory
    size_t mem_bytes = 0;

#ifdef __APPLE__
    size_t size = sizeof(mem_bytes);
    sysctlbyname("hw.memsize", &mem_bytes, &size, NULL, 0);
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    mem_bytes = pages * page_size;
#else
    mem_bytes = 8ULL * 1024 * 1024 * 1024;  // Assume 8 GB
#endif

    // Use 80% of available memory
    mem_bytes = (mem_bytes * 80) / 100;

    // State vector size = 2^n * 16 bytes (complex double)
    int qubits = 0;
    size_t state_size = 16;  // One amplitude

    while (state_size * 2 <= mem_bytes && qubits < 40) {
        state_size *= 2;
        qubits++;
    }

    return qubits;
}

// ============================================================================
// STRING CONVERSION
// ============================================================================

const char* qsim_backend_to_string(qsim_backend_t backend) {
    switch (backend) {
        case QSIM_BACKEND_AUTO:        return "auto";
        case QSIM_BACKEND_CPU:         return "cpu";
        case QSIM_BACKEND_CPU_SIMD:    return "cpu_simd";
        case QSIM_BACKEND_GPU_METAL:   return "gpu_metal";
        case QSIM_BACKEND_GPU_OPENCL:  return "gpu_opencl";
        case QSIM_BACKEND_GPU_VULKAN:  return "gpu_vulkan";
        case QSIM_BACKEND_GPU_CUDA:    return "gpu_cuda";
        case QSIM_BACKEND_DISTRIBUTED: return "distributed";
        default:                       return "unknown";
    }
}

qsim_backend_t qsim_backend_from_string(const char* str) {
    if (!str) return QSIM_BACKEND_AUTO;

    if (strcasecmp(str, "auto") == 0)        return QSIM_BACKEND_AUTO;
    if (strcasecmp(str, "cpu") == 0)         return QSIM_BACKEND_CPU;
    if (strcasecmp(str, "cpu_simd") == 0)    return QSIM_BACKEND_CPU_SIMD;
    if (strcasecmp(str, "gpu_metal") == 0)   return QSIM_BACKEND_GPU_METAL;
    if (strcasecmp(str, "metal") == 0)       return QSIM_BACKEND_GPU_METAL;
    if (strcasecmp(str, "gpu_opencl") == 0)  return QSIM_BACKEND_GPU_OPENCL;
    if (strcasecmp(str, "opencl") == 0)      return QSIM_BACKEND_GPU_OPENCL;
    if (strcasecmp(str, "gpu_vulkan") == 0)  return QSIM_BACKEND_GPU_VULKAN;
    if (strcasecmp(str, "vulkan") == 0)      return QSIM_BACKEND_GPU_VULKAN;
    if (strcasecmp(str, "gpu_cuda") == 0)    return QSIM_BACKEND_GPU_CUDA;
    if (strcasecmp(str, "cuda") == 0)        return QSIM_BACKEND_GPU_CUDA;
    if (strcasecmp(str, "distributed") == 0) return QSIM_BACKEND_DISTRIBUTED;
    if (strcasecmp(str, "mpi") == 0)         return QSIM_BACKEND_DISTRIBUTED;

    return QSIM_BACKEND_AUTO;
}

const char* qsim_simd_to_string(qsim_simd_t simd) {
    switch (simd) {
        case QSIM_SIMD_NONE:   return "none";
        case QSIM_SIMD_SSE2:   return "sse2";
        case QSIM_SIMD_AVX:    return "avx";
        case QSIM_SIMD_AVX2:   return "avx2";
        case QSIM_SIMD_AVX512: return "avx512";
        case QSIM_SIMD_NEON:   return "neon";
        case QSIM_SIMD_SVE:    return "sve";
        default:               return "unknown";
    }
}

qsim_simd_t qsim_simd_from_string(const char* str) {
    if (!str) return QSIM_SIMD_NONE;

    if (strcasecmp(str, "none") == 0)   return QSIM_SIMD_NONE;
    if (strcasecmp(str, "sse2") == 0)   return QSIM_SIMD_SSE2;
    if (strcasecmp(str, "avx") == 0)    return QSIM_SIMD_AVX;
    if (strcasecmp(str, "avx2") == 0)   return QSIM_SIMD_AVX2;
    if (strcasecmp(str, "avx512") == 0) return QSIM_SIMD_AVX512;
    if (strcasecmp(str, "neon") == 0)   return QSIM_SIMD_NEON;
    if (strcasecmp(str, "sve") == 0)    return QSIM_SIMD_SVE;

    return QSIM_SIMD_NONE;
}

const char* qsim_log_level_to_string(qsim_log_level_t level) {
    switch (level) {
        case QSIM_LOG_NONE:  return "none";
        case QSIM_LOG_ERROR: return "error";
        case QSIM_LOG_WARN:  return "warn";
        case QSIM_LOG_INFO:  return "info";
        case QSIM_LOG_DEBUG: return "debug";
        case QSIM_LOG_TRACE: return "trace";
        default:             return "unknown";
    }
}

qsim_log_level_t qsim_log_level_from_string(const char* str) {
    if (!str) return QSIM_LOG_WARN;

    if (strcasecmp(str, "none") == 0)  return QSIM_LOG_NONE;
    if (strcasecmp(str, "error") == 0) return QSIM_LOG_ERROR;
    if (strcasecmp(str, "warn") == 0)  return QSIM_LOG_WARN;
    if (strcasecmp(str, "info") == 0)  return QSIM_LOG_INFO;
    if (strcasecmp(str, "debug") == 0) return QSIM_LOG_DEBUG;
    if (strcasecmp(str, "trace") == 0) return QSIM_LOG_TRACE;

    return QSIM_LOG_WARN;
}

// ============================================================================
// PRESETS
// ============================================================================

void qsim_config_preset_performance(qsim_config_t* config) {
    if (!config) return;

    config->backend = qsim_detect_backend();
    config->simd = qsim_detect_simd();
    config->threading.num_threads = 0;  // Auto
    config->enable_caching = 1;
    config->gate_fusion = 1;
    config->circuit_optimization = 1;
    config->validate_states = 0;
    config->check_unitarity = 0;
    config->memory.use_huge_pages = 1;
    config->gpu.async_transfers = 1;
    config->gpu.kernel_fusion = 1;
}

void qsim_config_preset_accuracy(qsim_config_t* config) {
    if (!config) return;

    config->backend = QSIM_BACKEND_CPU;  // CPU for precision
    config->validate_states = 1;
    config->check_unitarity = 1;
    config->tolerance = 1e-14;
    config->gate_fusion = 0;  // May affect precision
    config->circuit_optimization = 0;
}

void qsim_config_preset_low_memory(qsim_config_t* config) {
    if (!config) return;

    config->backend = QSIM_BACKEND_CPU;
    config->memory.preallocate = 0;
    config->memory.use_huge_pages = 0;
    config->enable_caching = 0;
    config->gate_fusion = 0;
    config->threading.num_threads = 2;  // Reduce thread overhead
}

void qsim_config_preset_noisy(qsim_config_t* config,
                              double t1_us, double t2_us,
                              double gate_error) {
    if (!config) return;

    config->noise.enabled = 1;
    config->noise.t1_time = t1_us;
    config->noise.t2_time = t2_us;
    config->noise.gate_time = 0.02;  // 20 ns typical
    config->noise.depolarizing_rate = gate_error;
    config->noise.amplitude_damping = 0.0;  // Derived from T1
    config->noise.phase_damping = 0.0;      // Derived from T2

    config->use_quantum_rng = 1;  // Use true randomness for noise
}

// ============================================================================
// JSON SERIALIZATION (Simplified)
// ============================================================================

char* qsim_config_to_json(const qsim_config_t* config) {
    if (!config) return NULL;

    // Allocate buffer for JSON
    char* json = malloc(4096);
    if (!json) return NULL;

    snprintf(json, 4096,
        "{\n"
        "  \"backend\": \"%s\",\n"
        "  \"simd\": \"%s\",\n"
        "  \"max_qubits\": %d,\n"
        "  \"threads\": %d,\n"
        "  \"log_level\": \"%s\",\n"
        "  \"noise\": {\n"
        "    \"enabled\": %s,\n"
        "    \"depolarizing_rate\": %.6f,\n"
        "    \"t1_time\": %.2f,\n"
        "    \"t2_time\": %.2f\n"
        "  },\n"
        "  \"tolerance\": %.2e,\n"
        "  \"seed\": %llu\n"
        "}\n",
        qsim_backend_to_string(config->backend),
        qsim_simd_to_string(config->simd),
        config->max_qubits,
        config->threading.num_threads,
        qsim_log_level_to_string(config->log_level),
        config->noise.enabled ? "true" : "false",
        config->noise.depolarizing_rate,
        config->noise.t1_time,
        config->noise.t2_time,
        config->tolerance,
        (unsigned long long)config->seed
    );

    return json;
}

qsim_config_t* qsim_config_from_json(const char* json) {
    if (!json) return NULL;

    // Create default config
    qsim_config_t* config = qsim_config_create();
    if (!config) return NULL;

    // Very simplified JSON parsing
    // In production, use a proper JSON parser

    char* backend = strstr(json, "\"backend\"");
    if (backend) {
        char value[64];
        if (sscanf(backend, "\"backend\": \"%63[^\"]\"", value) == 1) {
            config->backend = qsim_backend_from_string(value);
        }
    }

    char* simd = strstr(json, "\"simd\"");
    if (simd) {
        char value[64];
        if (sscanf(simd, "\"simd\": \"%63[^\"]\"", value) == 1) {
            config->simd = qsim_simd_from_string(value);
        }
    }

    char* max_qubits = strstr(json, "\"max_qubits\"");
    if (max_qubits) {
        int value;
        if (sscanf(max_qubits, "\"max_qubits\": %d", &value) == 1) {
            config->max_qubits = value;
        }
    }

    char* threads = strstr(json, "\"threads\"");
    if (threads) {
        int value;
        if (sscanf(threads, "\"threads\": %d", &value) == 1) {
            config->threading.num_threads = value;
        }
    }

    return config;
}

// ============================================================================
// FILE I/O
// ============================================================================

qsim_config_t* qsim_config_load(const char* path) {
    if (!path) return NULL;

    FILE* fp = fopen(path, "r");
    if (!fp) return NULL;

    // Get file size
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (size <= 0 || size > 1024 * 1024) {  // Max 1MB
        fclose(fp);
        return NULL;
    }

    char* content = malloc(size + 1);
    if (!content) {
        fclose(fp);
        return NULL;
    }

    size_t read = fread(content, 1, size, fp);
    fclose(fp);

    if (read != (size_t)size) {
        free(content);
        return NULL;
    }
    content[size] = '\0';

    qsim_config_t* config = qsim_config_from_json(content);
    free(content);

    return config;
}

int qsim_config_save(const qsim_config_t* config, const char* path) {
    if (!config || !path) return -1;

    char* json = qsim_config_to_json(config);
    if (!json) return -1;

    FILE* fp = fopen(path, "w");
    if (!fp) {
        free(json);
        return -1;
    }

    size_t len = strlen(json);
    size_t written = fwrite(json, 1, len, fp);
    fclose(fp);
    free(json);

    return (written == len) ? 0 : -1;
}
