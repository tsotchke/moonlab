# ==============================================================================
# Quantum Simulator - CMake Build System
# ==============================================================================
#
# Cross-platform build configuration supporting:
# - macOS (Apple Silicon + Intel)
# - Linux (x86_64 + ARM64)
# - Windows (x86_64)
#
# Features:
# - SIMD auto-detection (AVX-512, AVX2, SSE2, NEON, SVE)
# - Optional GPU backends (Metal, OpenCL, Vulkan, cuQuantum)
# - Optional MPI for distributed computing
# - OpenMP parallelization
#
# Build with:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Copyright 2024-2026 tsotchke
# Licensed under the Apache License, Version 2.0
# ==============================================================================

cmake_minimum_required(VERSION 3.20)

# Project definition
project(quantum_simulator
    VERSION 1.0.0
    DESCRIPTION "High-performance quantum computing simulator"
    LANGUAGES C CXX
)

# ==============================================================================
# Build Options
# ==============================================================================

option(QSIM_BUILD_SHARED "Build shared library" ON)
option(QSIM_BUILD_STATIC "Build static library" OFF)
option(QSIM_BUILD_TESTS "Build test suite" ON)
option(QSIM_BUILD_EXAMPLES "Build example programs" ON)
option(QSIM_BUILD_BENCHMARKS "Build benchmarks" ON)

# Feature options
option(QSIM_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(QSIM_ENABLE_METAL "Enable Metal GPU backend (macOS)" ON)
option(QSIM_ENABLE_OPENCL "Enable OpenCL GPU backend" OFF)
option(QSIM_ENABLE_VULKAN "Enable Vulkan GPU backend" OFF)
option(QSIM_ENABLE_CUQUANTUM "Enable NVIDIA cuQuantum backend" OFF)
option(QSIM_ENABLE_MPI "Enable MPI distributed computing" OFF)
option(QSIM_ENABLE_LTO "Enable Link-Time Optimization" ON)

# SIMD options
option(QSIM_ENABLE_AVX512 "Enable AVX-512 (if available)" ON)
option(QSIM_ENABLE_AVX2 "Enable AVX2 (if available)" ON)
option(QSIM_ENABLE_NEON "Enable ARM NEON (if available)" ON)
option(QSIM_ENABLE_SVE "Enable ARM SVE (if available)" ON)

# ==============================================================================
# C Standard and Compiler Settings
# ==============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Position independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Platform Detection
# ==============================================================================

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(QSIM_PLATFORM_MACOS ON)
    set(QSIM_PLATFORM_NAME "macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(QSIM_PLATFORM_LINUX ON)
    set(QSIM_PLATFORM_NAME "Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(QSIM_PLATFORM_WINDOWS ON)
    set(QSIM_PLATFORM_NAME "Windows")
else()
    message(WARNING "Unknown platform: ${CMAKE_SYSTEM_NAME}")
    set(QSIM_PLATFORM_NAME "Unknown")
endif()

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(QSIM_ARCH_X86_64 ON)
    set(QSIM_ARCH_NAME "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(QSIM_ARCH_ARM64 ON)
    set(QSIM_ARCH_NAME "ARM64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|i386)$")
    set(QSIM_ARCH_X86 ON)
    set(QSIM_ARCH_NAME "x86")
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(QSIM_ARCH_NAME "Unknown")
endif()

message(STATUS "Platform: ${QSIM_PLATFORM_NAME} (${QSIM_ARCH_NAME})")

# ==============================================================================
# Compiler Detection and Flags
# ==============================================================================

# Detect compiler
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(QSIM_COMPILER_CLANG ON)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(QSIM_COMPILER_GCC ON)
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(QSIM_COMPILER_MSVC ON)
endif()

# Base warning flags
set(QSIM_WARNING_FLAGS)
if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
    list(APPEND QSIM_WARNING_FLAGS -Wall -Wextra -Wpedantic)
elseif(QSIM_COMPILER_MSVC)
    list(APPEND QSIM_WARNING_FLAGS /W4)
endif()

# Optimization flags
set(QSIM_OPTIMIZE_FLAGS)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
        list(APPEND QSIM_OPTIMIZE_FLAGS -O3 -ffast-math -funroll-loops)

        # Native CPU optimization
        if(NOT CMAKE_CROSSCOMPILING)
            list(APPEND QSIM_OPTIMIZE_FLAGS -march=native)
        endif()
    elseif(QSIM_COMPILER_MSVC)
        list(APPEND QSIM_OPTIMIZE_FLAGS /O2 /fp:fast)
    endif()
endif()

# ==============================================================================
# SIMD Detection
# ==============================================================================

include(CheckCSourceRuns)
include(CheckCCompilerFlag)

# AVX-512 detection
if(QSIM_ARCH_X86_64 AND QSIM_ENABLE_AVX512)
    check_c_compiler_flag("-mavx512f" QSIM_HAS_AVX512F_FLAG)
    if(QSIM_HAS_AVX512F_FLAG)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_c_source_runs("
            #include <immintrin.h>
            int main() {
                __m512d v = _mm512_setzero_pd();
                return 0;
            }
        " QSIM_HAS_AVX512)
        unset(CMAKE_REQUIRED_FLAGS)

        if(QSIM_HAS_AVX512)
            message(STATUS "AVX-512: Available")
            set(QSIM_SIMD_AVX512 ON)
        else()
            message(STATUS "AVX-512: Not supported by CPU")
        endif()
    endif()
endif()

# AVX2 detection
if(QSIM_ARCH_X86_64 AND QSIM_ENABLE_AVX2)
    check_c_compiler_flag("-mavx2" QSIM_HAS_AVX2_FLAG)
    if(QSIM_HAS_AVX2_FLAG)
        set(CMAKE_REQUIRED_FLAGS "-mavx2")
        check_c_source_runs("
            #include <immintrin.h>
            int main() {
                __m256d v = _mm256_setzero_pd();
                return 0;
            }
        " QSIM_HAS_AVX2)
        unset(CMAKE_REQUIRED_FLAGS)

        if(QSIM_HAS_AVX2)
            message(STATUS "AVX2: Available")
            set(QSIM_SIMD_AVX2 ON)
        else()
            message(STATUS "AVX2: Not supported by CPU")
        endif()
    endif()
endif()

# ARM NEON detection (always available on ARM64)
if(QSIM_ARCH_ARM64 AND QSIM_ENABLE_NEON)
    message(STATUS "NEON: Available (ARM64 baseline)")
    set(QSIM_SIMD_NEON ON)
endif()

# ARM SVE detection
if(QSIM_ARCH_ARM64 AND QSIM_ENABLE_SVE AND NOT QSIM_PLATFORM_MACOS)
    check_c_compiler_flag("-march=armv8-a+sve" QSIM_HAS_SVE_FLAG)
    if(QSIM_HAS_SVE_FLAG)
        message(STATUS "SVE: Compiler support available")
        set(QSIM_SIMD_SVE ON)
    endif()
endif()

# ==============================================================================
# Find Dependencies
# ==============================================================================

# Math library
find_library(MATH_LIBRARY m)

# Threads
find_package(Threads REQUIRED)

# OpenMP
if(QSIM_ENABLE_OPENMP)
    # On macOS with Apple Clang, OpenMP requires Homebrew libomp
    if(QSIM_PLATFORM_MACOS AND CMAKE_C_COMPILER_ID MATCHES "Clang")
        # Try Homebrew paths
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE BREW_RESULT
        )

        if(BREW_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
            message(STATUS "OpenMP: Found via Homebrew at ${HOMEBREW_LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_C_INCLUDE_DIR "${HOMEBREW_LIBOMP_PREFIX}/include")

            # Set for find_package
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -I${OpenMP_C_INCLUDE_DIR}")
            set(QSIM_OPENMP_LIBRARIES "${OpenMP_omp_LIBRARY}")
            set(QSIM_OPENMP_INCLUDE_DIR "${OpenMP_C_INCLUDE_DIR}")
            set(QSIM_HAS_OPENMP ON)
        else()
            message(STATUS "OpenMP: Not found (install with: brew install libomp)")
        endif()
    else()
        # Standard OpenMP detection for other platforms
        find_package(OpenMP)
        if(OpenMP_C_FOUND)
            message(STATUS "OpenMP: Found (version ${OpenMP_C_VERSION})")
            set(QSIM_HAS_OPENMP ON)
        else()
            message(STATUS "OpenMP: Not found")
        endif()
    endif()
endif()

# Apple Accelerate Framework
if(QSIM_PLATFORM_MACOS)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate: Found")
        set(QSIM_HAS_ACCELERATE ON)
    endif()
endif()

# Metal (macOS only)
if(QSIM_PLATFORM_MACOS AND QSIM_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        message(STATUS "Metal: Found")
        set(QSIM_HAS_METAL ON)
    else()
        message(STATUS "Metal: Not found")
    endif()
endif()

# Security framework (macOS only - for SecRandomCopyBytes)
if(QSIM_PLATFORM_MACOS)
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        message(STATUS "Security: Found")
        set(QSIM_HAS_SECURITY ON)
    endif()
endif()

# OpenCL
if(QSIM_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL: Found (version ${OpenCL_VERSION_STRING})")
        set(QSIM_HAS_OPENCL ON)
    else()
        message(STATUS "OpenCL: Not found")
    endif()
endif()

# Vulkan
if(QSIM_ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan: Found")
        set(QSIM_HAS_VULKAN ON)
    else()
        message(STATUS "Vulkan: Not found")
    endif()
endif()

# CUDA / cuQuantum
if(QSIM_ENABLE_CUQUANTUM)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA: Found (version ${CUDAToolkit_VERSION})")
            set(QSIM_HAS_CUDA ON)

            # Look for cuQuantum
            find_path(CUQUANTUM_INCLUDE_DIR custatevec.h
                HINTS
                    ${CUQUANTUM_ROOT}/include
                    $ENV{CUQUANTUM_ROOT}/include
                    /usr/local/cuda/include
                    /opt/nvidia/cuquantum/include
            )
            find_library(CUSTATEVEC_LIBRARY custatevec
                HINTS
                    ${CUQUANTUM_ROOT}/lib
                    ${CUQUANTUM_ROOT}/lib64
                    $ENV{CUQUANTUM_ROOT}/lib
                    $ENV{CUQUANTUM_ROOT}/lib64
                    /usr/local/cuda/lib64
                    /opt/nvidia/cuquantum/lib64
            )
            find_library(CUTENSORNET_LIBRARY cutensornet
                HINTS
                    ${CUQUANTUM_ROOT}/lib
                    ${CUQUANTUM_ROOT}/lib64
                    $ENV{CUQUANTUM_ROOT}/lib
                    $ENV{CUQUANTUM_ROOT}/lib64
                    /usr/local/cuda/lib64
                    /opt/nvidia/cuquantum/lib64
            )

            if(CUQUANTUM_INCLUDE_DIR AND CUSTATEVEC_LIBRARY)
                message(STATUS "cuQuantum: Found (cuStateVec)")
                set(QSIM_HAS_CUQUANTUM ON)
                set(QSIM_HAS_CUSTATEVEC ON)
                if(CUTENSORNET_LIBRARY)
                    message(STATUS "cuQuantum: Found (cuTensorNet)")
                    set(QSIM_HAS_CUTENSORNET ON)
                endif()
            else()
                message(STATUS "cuQuantum: Not found (CUDA available, using native kernels)")
            endif()
        else()
            message(STATUS "CUDA: Toolkit not found")
        endif()
    else()
        message(STATUS "CUDA: Compiler not found")
    endif()
endif()

# MPI
if(QSIM_ENABLE_MPI)
    find_package(MPI)
    if(MPI_C_FOUND)
        message(STATUS "MPI: Found")
        set(QSIM_HAS_MPI ON)
    else()
        message(STATUS "MPI: Not found")
    endif()
endif()

# ==============================================================================
# Source Files
# ==============================================================================

set(QSIM_QUANTUM_SOURCES
    src/quantum/state.c
    src/quantum/gates.c
    src/quantum/measurement.c
    src/quantum/entanglement.c
    src/quantum/noise.c
)

set(QSIM_ALGORITHMS_SOURCES
    src/algorithms/grover.c
    src/algorithms/grover_optimizer.c
    src/algorithms/quantum_rng.c
    src/algorithms/bell_tests.c
    src/algorithms/qaoa.c
    src/algorithms/qpe.c
    src/algorithms/vqe.c
)

set(QSIM_OPTIMIZATION_SOURCES
    src/optimization/memory_align.c
    src/optimization/simd_dispatch.c
    src/optimization/simd_ops.c
    src/optimization/stride_gates.c
    src/optimization/parallel_ops.c
    src/optimization/accelerate_ops.c
    src/optimization/gpu/gpu_backend.c
)

# GPU backend sources (conditional)
set(QSIM_GPU_SOURCES)
if(QSIM_ENABLE_OPENCL)
    list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_opencl.c)
endif()
if(QSIM_ENABLE_VULKAN)
    list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_vulkan.c)
endif()
if(QSIM_ENABLE_CUQUANTUM)
    list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_cuda.cu)
    list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_cuquantum.cu)
endif()

set(QSIM_UTILS_SOURCES
    src/utils/entropy.c
    src/utils/config.c
    src/utils/matrix_math.c
)

set(QSIM_APPLICATIONS_SOURCES
    src/applications/health_tests.c
    src/applications/bell_test.c
    src/applications/entropy_pool.c
    src/applications/hardware_entropy.c
    src/applications/qrng.c
)

set(QSIM_PROFILER_SOURCES
    tools/profiler/profiler.c
    tools/profiler/performance_monitor.c
)

# Distributed computing sources (MPI)
set(QSIM_DISTRIBUTED_SOURCES
    src/distributed/mpi_bridge.c
    src/distributed/state_partition.c
    src/distributed/distributed_gates.c
    src/distributed/collective_ops.c
)

# Combine all sources
set(QSIM_ALL_SOURCES
    ${QSIM_QUANTUM_SOURCES}
    ${QSIM_ALGORITHMS_SOURCES}
    ${QSIM_OPTIMIZATION_SOURCES}
    ${QSIM_GPU_SOURCES}
    ${QSIM_UTILS_SOURCES}
    ${QSIM_APPLICATIONS_SOURCES}
    ${QSIM_PROFILER_SOURCES}
    ${QSIM_DISTRIBUTED_SOURCES}
)

# Optional AVX-512 sources
if(QSIM_SIMD_AVX512)
    list(APPEND QSIM_ALL_SOURCES src/optimization/simd_avx512.c)
endif()

# Optional SVE sources
if(QSIM_SIMD_SVE)
    list(APPEND QSIM_ALL_SOURCES src/optimization/simd_sve.c)
endif()

# ==============================================================================
# Library Target
# ==============================================================================

# Create library
if(QSIM_BUILD_SHARED)
    add_library(quantumsim SHARED ${QSIM_ALL_SOURCES})
    set_target_properties(quantumsim PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
elseif(QSIM_BUILD_STATIC)
    add_library(quantumsim STATIC ${QSIM_ALL_SOURCES})
else()
    add_library(quantumsim ${QSIM_ALL_SOURCES})
endif()

# Include directories
target_include_directories(quantumsim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

# Compile options
target_compile_options(quantumsim PRIVATE
    ${QSIM_WARNING_FLAGS}
    ${QSIM_OPTIMIZE_FLAGS}
)

# Link libraries
target_link_libraries(quantumsim
    PRIVATE
        ${MATH_LIBRARY}
        Threads::Threads
)

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

# macOS configuration
if(QSIM_PLATFORM_MACOS)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_MACOS=1)

    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(quantumsim PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(quantumsim PRIVATE HAS_ACCELERATE=1)
    endif()

    if(QSIM_HAS_METAL)
        target_link_libraries(quantumsim PRIVATE
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${COREGRAPHICS_FRAMEWORK}
        )
        target_compile_definitions(quantumsim PRIVATE HAS_METAL=1)
    endif()

    if(QSIM_HAS_SECURITY)
        target_link_libraries(quantumsim PRIVATE ${SECURITY_FRAMEWORK})
        target_compile_definitions(quantumsim PRIVATE HAS_SECURITY=1)
    endif()
endif()

# Linux configuration
if(QSIM_PLATFORM_LINUX)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_LINUX=1)
endif()

# Windows configuration
if(QSIM_PLATFORM_WINDOWS)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_WINDOWS=1)
endif()

# ==============================================================================
# SIMD Configuration
# ==============================================================================

if(QSIM_SIMD_AVX512)
    target_compile_definitions(quantumsim PRIVATE HAS_AVX512=1)
    if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
        set_source_files_properties(
            src/optimization/simd_avx512.c
            PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx512vl"
        )
    endif()
endif()

if(QSIM_SIMD_AVX2)
    target_compile_definitions(quantumsim PRIVATE HAS_AVX2=1)
endif()

if(QSIM_SIMD_NEON)
    target_compile_definitions(quantumsim PRIVATE HAS_NEON=1)
endif()

if(QSIM_SIMD_SVE)
    target_compile_definitions(quantumsim PRIVATE HAS_SVE=1)
    if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
        set_source_files_properties(
            src/optimization/simd_sve.c
            PROPERTIES COMPILE_FLAGS "-march=armv8-a+sve"
        )
    endif()
endif()

# ==============================================================================
# OpenMP Configuration
# ==============================================================================

if(QSIM_HAS_OPENMP)
    if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
        # macOS with Homebrew libomp
        target_link_libraries(quantumsim PRIVATE ${QSIM_OPENMP_LIBRARIES})
        target_include_directories(quantumsim PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
    elseif(TARGET OpenMP::OpenMP_C)
        # Standard OpenMP target
        target_link_libraries(quantumsim PRIVATE OpenMP::OpenMP_C)
    endif()
    target_compile_definitions(quantumsim PRIVATE HAS_OPENMP=1)
endif()

# ==============================================================================
# GPU Backend Configuration
# ==============================================================================

if(QSIM_HAS_OPENCL)
    target_link_libraries(quantumsim PRIVATE OpenCL::OpenCL)
    target_compile_definitions(quantumsim PRIVATE HAS_OPENCL=1)
endif()

if(QSIM_HAS_VULKAN)
    target_link_libraries(quantumsim PRIVATE Vulkan::Vulkan)
    target_compile_definitions(quantumsim PRIVATE HAS_VULKAN=1)
endif()

if(QSIM_HAS_CUDA)
    target_link_libraries(quantumsim PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(quantumsim PRIVATE HAS_CUDA=1)
    target_include_directories(quantumsim PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # Set CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
    set_target_properties(quantumsim PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(QSIM_HAS_CUQUANTUM)
    target_include_directories(quantumsim PRIVATE ${CUQUANTUM_INCLUDE_DIR})
    target_link_libraries(quantumsim PRIVATE ${CUSTATEVEC_LIBRARY})
    target_compile_definitions(quantumsim PRIVATE HAS_CUQUANTUM=1 HAS_CUSTATEVEC=1)
    if(QSIM_HAS_CUTENSORNET)
        target_link_libraries(quantumsim PRIVATE ${CUTENSORNET_LIBRARY})
        target_compile_definitions(quantumsim PRIVATE HAS_CUTENSORNET=1)
    endif()
endif()

if(QSIM_HAS_MPI)
    target_link_libraries(quantumsim PRIVATE MPI::MPI_C)
    target_compile_definitions(quantumsim PRIVATE HAS_MPI=1)
endif()

# ==============================================================================
# Link-Time Optimization
# ==============================================================================

if(QSIM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_error)

    if(lto_supported)
        set_target_properties(quantumsim PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
        message(STATUS "LTO: Enabled")
    else()
        message(STATUS "LTO: Not supported (${lto_error})")
    endif()
endif()

# ==============================================================================
# Test Targets
# ==============================================================================

if(QSIM_BUILD_TESTS)
    enable_testing()

    # Main test executable
    add_executable(qsim_test tests/quantum_sim_test.c)
    target_link_libraries(qsim_test PRIVATE quantumsim)
    add_test(NAME quantum_sim_test COMMAND qsim_test)

    # Health tests
    add_executable(health_tests_test tests/health_tests_test.c)
    target_link_libraries(health_tests_test PRIVATE quantumsim)
    add_test(NAME health_tests COMMAND health_tests_test)

    # Bell test demo
    add_executable(bell_test_demo tests/bell_test_demo.c)
    target_link_libraries(bell_test_demo PRIVATE quantumsim)
    add_test(NAME bell_test COMMAND bell_test_demo)

    # Gate test
    add_executable(gate_test tests/gate_test.c)
    target_link_libraries(gate_test PRIVATE quantumsim)
    add_test(NAME gate_test COMMAND gate_test)

    # Correlation test
    add_executable(correlation_test tests/correlation_test.c)
    target_link_libraries(correlation_test PRIVATE quantumsim)
    add_test(NAME correlation_test COMMAND correlation_test)

    # Unit tests
    add_executable(test_quantum_state tests/unit/test_quantum_state.c)
    target_link_libraries(test_quantum_state PRIVATE quantumsim)
    add_test(NAME unit_quantum_state COMMAND test_quantum_state)

    add_executable(test_quantum_gates tests/unit/test_quantum_gates.c)
    target_link_libraries(test_quantum_gates PRIVATE quantumsim)
    add_test(NAME unit_quantum_gates COMMAND test_quantum_gates)

    add_executable(test_memory_align tests/unit/test_memory_align.c
                   src/optimization/memory_align.c)
    target_link_libraries(test_memory_align PRIVATE ${MATH_LIBRARY} Threads::Threads)
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_memory_align PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_memory_align PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_memory_align COMMAND test_memory_align)

    add_executable(test_simd_dispatch tests/unit/test_simd_dispatch.c
                   src/optimization/simd_dispatch.c)
    target_link_libraries(test_simd_dispatch PRIVATE ${MATH_LIBRARY} Threads::Threads)
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_simd_dispatch PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_simd_dispatch PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_simd_dispatch COMMAND test_simd_dispatch)

    add_executable(test_stride_gates tests/unit/test_stride_gates.c
                   src/optimization/stride_gates.c)
    target_link_libraries(test_stride_gates PRIVATE ${MATH_LIBRARY} Threads::Threads)
    if(QSIM_HAS_OPENMP)
        if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
            target_link_libraries(test_stride_gates PRIVATE ${QSIM_OPENMP_LIBRARIES})
            target_include_directories(test_stride_gates PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
            target_compile_options(test_stride_gates PRIVATE -Xpreprocessor -fopenmp)
        elseif(TARGET OpenMP::OpenMP_C)
            target_link_libraries(test_stride_gates PRIVATE OpenMP::OpenMP_C)
        endif()
        target_compile_definitions(test_stride_gates PRIVATE HAS_OPENMP=1)
    endif()
    add_test(NAME unit_stride_gates COMMAND test_stride_gates)

    # Distributed computing tests (MPI)
    if(QSIM_HAS_MPI)
        add_executable(test_distributed_gates tests/unit/test_distributed_gates.c)
        target_link_libraries(test_distributed_gates PRIVATE quantumsim MPI::MPI_C)
        # Run with 4 MPI processes
        add_test(NAME distributed_gates
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_distributed_gates> ${MPIEXEC_POSTFLAGS})

        add_executable(test_state_partition tests/unit/test_state_partition.c)
        target_link_libraries(test_state_partition PRIVATE quantumsim MPI::MPI_C)
        add_test(NAME state_partition
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_state_partition> ${MPIEXEC_POSTFLAGS})

        add_executable(test_collective_ops tests/unit/test_collective_ops.c)
        target_link_libraries(test_collective_ops PRIVATE quantumsim MPI::MPI_C)
        add_test(NAME collective_ops
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_collective_ops> ${MPIEXEC_POSTFLAGS})

        message(STATUS "MPI Tests: Enabled (will run with 4 processes)")
    endif()

    message(STATUS "Tests: Enabled")
endif()

# ==============================================================================
# Example Targets
# ==============================================================================

if(QSIM_BUILD_EXAMPLES)
    # Grover examples
    add_executable(grover_hash_collision examples/quantum/grover_hash_collision.c)
    target_link_libraries(grover_hash_collision PRIVATE quantumsim)

    add_executable(grover_large_scale_demo examples/quantum/grover_large_scale_demo.c)
    target_link_libraries(grover_large_scale_demo PRIVATE quantumsim)

    add_executable(grover_large_scale_optimized examples/quantum/grover_large_scale_optimized.c)
    target_link_libraries(grover_large_scale_optimized PRIVATE quantumsim)

    add_executable(grover_password_crack examples/quantum/grover_password_crack.c)
    target_link_libraries(grover_password_crack PRIVATE quantumsim)

    add_executable(phase3_phase4_benchmark examples/quantum/phase3_phase4_benchmark.c)
    target_link_libraries(phase3_phase4_benchmark PRIVATE quantumsim)

    # Application examples
    add_executable(vqe_h2_molecule examples/applications/vqe_h2_molecule.c)
    target_link_libraries(vqe_h2_molecule PRIVATE quantumsim)

    add_executable(qaoa_maxcut examples/applications/qaoa_maxcut.c)
    target_link_libraries(qaoa_maxcut PRIVATE quantumsim)

    add_executable(portfolio_optimization examples/applications/portfolio_optimization.c)
    target_link_libraries(portfolio_optimization PRIVATE quantumsim)

    add_executable(tsp_logistics examples/applications/tsp_logistics.c)
    target_link_libraries(tsp_logistics PRIVATE quantumsim)

    message(STATUS "Examples: Enabled")
endif()

# ==============================================================================
# Benchmark Targets
# ==============================================================================

if(QSIM_BUILD_BENCHMARKS)
    add_executable(grover_parallel_benchmark
        examples/quantum/grover_parallel_benchmark.c
        src/optimization/parallel_ops.c
    )
    target_link_libraries(grover_parallel_benchmark PRIVATE quantumsim)
    if(QSIM_HAS_OPENMP)
        if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
            target_link_libraries(grover_parallel_benchmark PRIVATE ${QSIM_OPENMP_LIBRARIES})
            target_include_directories(grover_parallel_benchmark PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
            target_compile_options(grover_parallel_benchmark PRIVATE -Xpreprocessor -fopenmp)
        elseif(TARGET OpenMP::OpenMP_C)
            target_link_libraries(grover_parallel_benchmark PRIVATE OpenMP::OpenMP_C)
        endif()
        target_compile_definitions(grover_parallel_benchmark PRIVATE HAS_OPENMP=1)
    endif()

    message(STATUS "Benchmarks: Enabled")
endif()

# ==============================================================================
# Installation
# ==============================================================================

include(GNUInstallDirs)

# Install library
install(TARGETS quantumsim
    EXPORT quantumsim-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantumsim
    FILES_MATCHING PATTERN "*.h"
)

# Install CMake config
install(EXPORT quantumsim-targets
    FILE quantumsim-targets.cmake
    NAMESPACE quantumsim::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
)

# Generate package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/quantumsim-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
)

# pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/quantumsim.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc
    @ONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Quantum Simulator Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Platform:     ${QSIM_PLATFORM_NAME}")
message(STATUS "Architecture: ${QSIM_ARCH_NAME}")
message(STATUS "Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "SIMD Features:")
message(STATUS "  AVX-512:    ${QSIM_SIMD_AVX512}")
message(STATUS "  AVX2:       ${QSIM_SIMD_AVX2}")
message(STATUS "  NEON:       ${QSIM_SIMD_NEON}")
message(STATUS "  SVE:        ${QSIM_SIMD_SVE}")
message(STATUS "")
message(STATUS "Optional Features:")
message(STATUS "  OpenMP:     ${QSIM_HAS_OPENMP}")
message(STATUS "  Accelerate: ${QSIM_HAS_ACCELERATE}")
message(STATUS "  Metal:      ${QSIM_HAS_METAL}")
message(STATUS "  OpenCL:     ${QSIM_HAS_OPENCL}")
message(STATUS "  Vulkan:     ${QSIM_HAS_VULKAN}")
message(STATUS "  CUDA:       ${QSIM_HAS_CUDA}")
message(STATUS "  cuQuantum:  ${QSIM_HAS_CUQUANTUM}")
message(STATUS "  MPI:        ${QSIM_HAS_MPI}")
message(STATUS "========================================")
message(STATUS "")
