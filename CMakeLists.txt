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

# Read the canonical version string from VERSION.txt so the CMake project
# version, SOVERSION, pkg-config data, and any generated headers stay in
# lock-step with the one file developers actually bump. The file may
# contain MAJOR.MINOR.PATCH optionally followed by a pre-release tag such
# as "-dev" or "-rc1"; CMake's project() accepts only the numeric form, so
# we split the raw string before passing it to project().
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" QSIM_VERSION_RAW)
string(STRIP "${QSIM_VERSION_RAW}" QSIM_VERSION_FULL)
if(NOT QSIM_VERSION_FULL MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)(-[A-Za-z0-9.-]+)?$")
    message(FATAL_ERROR
        "VERSION.txt must be MAJOR.MINOR.PATCH with an optional '-pre' tag; "
        "got '${QSIM_VERSION_FULL}'")
endif()
set(QSIM_VERSION      ${CMAKE_MATCH_1})
set(QSIM_VERSION_PRE  ${CMAKE_MATCH_2})  # includes leading '-' or empty

# Project definition (numeric version only; pre-release suffix is kept
# separately for display / packaging).
project(quantum_simulator
    VERSION ${QSIM_VERSION}
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
option(QSIM_BUILD_JS_DIST "Auto-build the TS/JS dist via pnpm (enables webgpu_unified_smoke on fresh clones)" OFF)

# Feature options
option(QSIM_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(QSIM_ENABLE_METAL "Enable Metal GPU backend (macOS)" ON)
option(QSIM_ENABLE_OPENCL "Enable OpenCL GPU backend" OFF)
option(QSIM_ENABLE_VULKAN "Enable Vulkan GPU backend" OFF)
option(QSIM_ENABLE_CUQUANTUM "Enable NVIDIA cuQuantum backend" OFF)
option(QSIM_ENABLE_MPI "Enable MPI distributed computing" OFF)
option(QSIM_ENABLE_ESHKOL "Link Eshkol GPU backend for fp64-on-Metal (Ozaki-II / SF64)" OFF)
set(QSIM_ESHKOL_ROOT "$ENV{HOME}/Desktop/eshkol" CACHE PATH
    "Root directory of the Eshkol source tree (for headers + libeshkol-static.a).")
option(QSIM_ENABLE_LTO "Enable Link-Time Optimization" ON)
option(QSIM_WERROR     "Treat warnings as errors (excludes -Wpedantic for OpenMP system headers)" OFF)
option(QSIM_ENABLE_SANITIZERS "Build with -fsanitize=address,undefined (for CI / dev)" OFF)

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
    add_compile_definitions(_USE_MATH_DEFINES)
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
    if(QSIM_WERROR)
        # -Werror everywhere our code is responsible. Exclusions:
        #  * -Wpedantic: libomp's <omp.h> is a system header and uses
        #    enumerator values outside the ISO C range.
        #  * -Wdeprecated-declarations: Apple CLAPACK (zheev_, zgesvd_,
        #    zgeqrf_, zungqr_, cblas_zgemm, cblas_zgemv, ...) was
        #    deprecated in macOS 13.3 in favour of ACCELERATE_NEW_LAPACK.
        #    Migration is a Phase 1G task; we still want the warning
        #    visible but not build-breaking.
        list(APPEND QSIM_WARNING_FLAGS
             -Werror
             -Wno-error=pedantic
             -Wno-error=deprecated-declarations)
    endif()
elseif(QSIM_COMPILER_MSVC)
    list(APPEND QSIM_WARNING_FLAGS /W4)
    if(QSIM_WERROR)
        list(APPEND QSIM_WARNING_FLAGS /WX)
    endif()
endif()

# AddressSanitizer + UndefinedBehaviorSanitizer. Opt-in; meant for CI
# and local debug builds. Always paired with -O1 -g for readable reports.
set(QSIM_SANITIZER_FLAGS)
if(QSIM_ENABLE_SANITIZERS AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    list(APPEND QSIM_SANITIZER_FLAGS
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -g
    )
    # Keep -O1 so the sanitizer can keep up without losing debuggability.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND QSIM_SANITIZER_FLAGS -O1)
    endif()
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
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

# Math library. Unix-like platforms need libm explicitly; MSVC provides the
# C math functions from the CRT and has no separate "m" library.
if(QSIM_PLATFORM_WINDOWS)
    set(MATH_LIBRARY "")
else()
    find_library(MATH_LIBRARY m)
endif()

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

set(QSIM_BACKEND_SOURCES
    src/backends/clifford/clifford.c
)

set(QSIM_ALGORITHMS_SOURCES
    src/algorithms/grover.c
    src/algorithms/grover_optimizer.c
    src/algorithms/quantum_rng.c
    src/algorithms/bell_tests.c
    src/algorithms/qaoa.c
    src/algorithms/qpe.c
    src/algorithms/vqe.c
    # Tensor-network stack — MPS / MPO / DMRG / TDVP / skyrmion braiding.
    # Previously present in the tree but not compiled into libquantumsim,
    # which meant every public TN symbol was undefined to consumers and
    # no TN tests could link.
    src/algorithms/tensor_network/tensor.c
    src/algorithms/tensor_network/tn_state.c
    src/algorithms/tensor_network/tn_gates.c
    src/algorithms/tensor_network/tn_measurement.c
    src/algorithms/tensor_network/contraction.c
    src/algorithms/tensor_network/svd_compress.c
    src/algorithms/tensor_network/dmrg.c
    src/algorithms/tensor_network/tdvp.c
    src/algorithms/tensor_network/mpo_2d.c
    src/algorithms/tensor_network/lattice_2d.c
    src/algorithms/tensor_network/skyrmion_braiding.c
    # Chemistry (Jordan-Wigner, UCCSD, molecular Hamiltonians).
    src/algorithms/chemistry/chemistry.c
    # Topological QC: Fibonacci/Ising anyons, fusion trees, surface/toric codes.
    src/algorithms/topological/topological.c
    # Real-space topology: Bianco-Resta local Chern marker (dense reference).
    src/algorithms/topology_realspace/chern_marker.c
    # Real-space topology: matrix-free Chebyshev-KPM Chern marker.
    src/algorithms/topology_realspace/chern_kpm.c
    # Real-space topology: MPO-level Chebyshev / Jackson KPM (P5.16).
    src/algorithms/topology_realspace/mpo_kpm.c
    # Quantum geometric tensor: momentum-space Berry curvature + metric.
    src/algorithms/quantum_geometry/qgt.c
    # Shor-ECDLP resource estimator (logical + FTQC cost model).
    src/algorithms/shor_ecdlp/shor_ecdlp.c
    # Native autograd for parameterized circuits (P5.19).
    src/algorithms/diff/differentiable.c
    # Error mitigation: zero-noise extrapolation (v0.2).
    src/mitigation/zne.c
    # Cryptographic primitives: SHA3 + SHAKE (FIPS 202) -- foundation
    # for PQC (FIPS 203/204/205) and for extractor seeding in DI-QRNG.
    src/crypto/sha3/sha3.c
    # ML-KEM (FIPS 203) polynomial ring + NTT (Kyber primitives).
    src/crypto/mlkem/poly.c
    # ML-KEM-512 KEM: KeyGen / Encaps / Decaps.
    src/crypto/mlkem/mlkem.c
    # AES-256 + NIST SP 800-90A CTR_DRBG (pq-crystals KAT convention).
    src/crypto/aes/aes.c
    src/crypto/drbg/ctr_drbg.c
    # Many-body localization.
    src/algorithms/mbl/mbl.c
    # Visualization — circuit diagrams + Feynman diagrams.
    src/visualization/circuit_diagram.c
    src/visualization/feynman_diagram.c
)

set(QSIM_OPTIMIZATION_SOURCES
    src/optimization/memory_align.c
    src/optimization/simd_dispatch.c
    src/optimization/simd_ops.c
    src/optimization/stride_gates.c
    src/optimization/parallel_ops.c
    src/optimization/accelerate_ops.c
    src/optimization/fusion/fusion.c
    src/optimization/gpu/gpu_backend.c
)
# Backend implementation files are added only when enabled. For Metal we
# also need the Objective-C++ implementation file.
if(QSIM_HAS_METAL OR QSIM_ENABLE_CUDA OR QSIM_ENABLE_OPENCL OR QSIM_ENABLE_VULKAN OR QSIM_ENABLE_CUQUANTUM OR QSIM_ENABLE_WEBGPU)
    if(QSIM_HAS_METAL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/optimization/gpu_metal.mm")
        list(APPEND QSIM_OPTIMIZATION_SOURCES src/optimization/gpu_metal.mm)
    endif()
endif()

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
# The Eshkol GPU bridge compiles regardless of whether Eshkol itself
# is enabled -- the stub path returns MOONLAB_ESHKOL_NOT_BUILT so
# callers can use the API unconditionally. The real (Eshkol-linking)
# path activates only when QSIM_ENABLE_ESHKOL=ON and guards the
# implementation behind #ifdef QSIM_ENABLE_ESHKOL.
list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_eshkol.cpp)

set(QSIM_UTILS_SOURCES
    src/utils/entropy.c
    src/utils/config.c
    src/utils/matrix_math.c
    src/utils/manifest.c
    src/utils/quantum_entropy.c
)

set(QSIM_APPLICATIONS_SOURCES
    src/applications/health_tests.c
    src/applications/bell_test.c
    src/applications/entropy_pool.c
    src/applications/hardware_entropy.c
    src/applications/qrng.c
    src/applications/qrng_di.c
    src/applications/moonlab_qrng_export.c
    src/applications/nist_sp800_22.c
    src/applications/quantum_volume.c
)

set(QSIM_PROFILER_SOURCES
    tools/profiler/profiler.c
    src/utils/performance_monitor.c
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
    ${QSIM_BACKEND_SOURCES}
    ${QSIM_ALGORITHMS_SOURCES}
    ${QSIM_OPTIMIZATION_SOURCES}
    ${QSIM_GPU_SOURCES}
    ${QSIM_UTILS_SOURCES}
    ${QSIM_APPLICATIONS_SOURCES}
    ${QSIM_PROFILER_SOURCES}
)
# Distributed sources are only compilable when MPI is enabled — they
# reference mpi_* helpers declared inside the MPI-gated translation units.
if(QSIM_ENABLE_MPI)
    list(APPEND QSIM_ALL_SOURCES ${QSIM_DISTRIBUTED_SOURCES})
endif()

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
    if(QSIM_PLATFORM_WINDOWS)
        set_target_properties(quantumsim PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
elseif(QSIM_BUILD_STATIC)
    add_library(quantumsim STATIC ${QSIM_ALL_SOURCES})
else()
    add_library(quantumsim ${QSIM_ALL_SOURCES})
endif()

# Eshkol integration: enable the real (non-stub) path of the bridge
# and link libeshkol-static.a with its framework dependencies.
if(QSIM_ENABLE_ESHKOL)
    set(_eshkol_inc   "${QSIM_ESHKOL_ROOT}/inc")
    set(_eshkol_libfn "${QSIM_ESHKOL_ROOT}/build/libeshkol-static.a")
    if(NOT EXISTS "${_eshkol_libfn}")
        message(FATAL_ERROR
            "QSIM_ENABLE_ESHKOL=ON but ${_eshkol_libfn} not found. "
            "Build Eshkol first (cmake -S ${QSIM_ESHKOL_ROOT} -B "
            "${QSIM_ESHKOL_ROOT}/build && cmake --build "
            "${QSIM_ESHKOL_ROOT}/build).")
    endif()
    target_compile_definitions(quantumsim PRIVATE QSIM_ENABLE_ESHKOL=1)
    target_include_directories(quantumsim PRIVATE "${_eshkol_inc}")
    target_link_libraries(quantumsim PRIVATE
        "${_eshkol_libfn}"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
    )
    set_source_files_properties(
        src/optimization/gpu/backends/gpu_eshkol.cpp
        PROPERTIES COMPILE_DEFINITIONS QSIM_ENABLE_ESHKOL=1
    )
    set_property(TARGET quantumsim PROPERTY CXX_STANDARD 17)
    message(STATUS "Eshkol bridge: ON (lib=${_eshkol_libfn})")
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
    ${QSIM_SANITIZER_FLAGS}
)

# Sanitizer runtime needs to be linked into the shared library too.
if(QSIM_ENABLE_SANITIZERS AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    target_link_options(quantumsim PRIVATE ${QSIM_SANITIZER_FLAGS})
    add_compile_options(${QSIM_SANITIZER_FLAGS})
    add_link_options(${QSIM_SANITIZER_FLAGS})
endif()

# Link libraries
target_link_libraries(quantumsim
    PUBLIC
        # libm (-lm on Linux; no-op on macOS where it's in libSystem).
        # Promoted to PUBLIC so every test/example that calls complex
        # math (exp, cabs, ...) via the quantumsim target gets libm
        # transitively -- otherwise the Linux link fails with
        # "undefined reference to exp@@GLIBC_2.29" on any test that
        # includes <math.h> directly without a manual -lm.
        ${MATH_LIBRARY}
    PRIVATE
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

    # LAPACK (zheev_, LAPACKE_zheev, LAPACKE_zgesvd) on Linux.
    # Apple's Accelerate framework provides these on macOS; Linux
    # needs an explicit link against a BLAS/LAPACK provider.  We
    # prefer LAPACKE (row-major wrappers used by collective_ops.c)
    # over the raw Fortran symbols, but link both forms so either
    # entry point resolves.  If neither is available, the build
    # carries on; the affected features will fail at link time for
    # consumers that call them.
    find_library(LAPACKE_LIB NAMES lapacke)
    find_library(LAPACK_LIB NAMES lapack)
    find_library(BLAS_LIB   NAMES openblas blas)
    if(LAPACKE_LIB AND LAPACK_LIB AND BLAS_LIB)
        target_link_libraries(quantumsim PRIVATE
            ${LAPACKE_LIB} ${LAPACK_LIB} ${BLAS_LIB})
        message(STATUS "LAPACK: LAPACKE=${LAPACKE_LIB} "
                       "LAPACK=${LAPACK_LIB} BLAS=${BLAS_LIB}")
    else()
        message(WARNING "LAPACK/LAPACKE not found on Linux; "
                        "entanglement-entropy and MBL modules will fail "
                        "to link their test drivers.  Install "
                        "liblapacke-dev + libopenblas-dev to fix.")
    endif()
endif()

# Windows configuration
if(QSIM_PLATFORM_WINDOWS)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_WINDOWS=1)
    target_link_libraries(quantumsim PRIVATE bcrypt)

    if(QSIM_COMPILER_CLANG)
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} --print-resource-dir
            OUTPUT_VARIABLE QSIM_CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(QSIM_ARCH_X86_64)
            set(QSIM_CLANG_BUILTINS_NAME clang_rt.builtins-x86_64)
        elseif(QSIM_ARCH_ARM64)
            set(QSIM_CLANG_BUILTINS_NAME clang_rt.builtins-aarch64)
        endif()

        if(QSIM_CLANG_BUILTINS_NAME)
            find_library(QSIM_CLANG_BUILTINS
                NAMES ${QSIM_CLANG_BUILTINS_NAME}
                PATHS "${QSIM_CLANG_RESOURCE_DIR}/lib/windows"
                NO_DEFAULT_PATH
            )
        endif()

        if(QSIM_CLANG_BUILTINS)
            target_link_libraries(quantumsim PUBLIC "${QSIM_CLANG_BUILTINS}")
            message(STATUS "Clang builtins: ${QSIM_CLANG_BUILTINS}")
        else()
            message(WARNING "Clang builtins library not found; complex arithmetic may fail to link")
        endif()
    endif()
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

        # Relocate libomp's absolute Homebrew path to @rpath, and build
        # an rpath list that only contains directories that actually
        # exist on this host. This keeps the binary relocatable
        # (@loader_path/../lib is always present) while adding
        # developer-machine fallbacks when present. For container /
        # Docker builds, set QSIM_EXTRA_RPATH to pre-seed additional
        # entries (e.g. "/opt/openmp/lib").
        set(QSIM_RPATH_LIST "@loader_path/../lib")
        foreach(_qsim_candidate_rpath
                "$ENV{HOMEBREW_PREFIX}/opt/libomp/lib"
                "/opt/homebrew/opt/libomp/lib"
                "/usr/local/opt/libomp/lib")
            if(_qsim_candidate_rpath AND EXISTS "${_qsim_candidate_rpath}")
                list(APPEND QSIM_RPATH_LIST "${_qsim_candidate_rpath}")
            endif()
        endforeach()
        if(DEFINED QSIM_EXTRA_RPATH)
            list(APPEND QSIM_RPATH_LIST ${QSIM_EXTRA_RPATH})
        endif()
        set_target_properties(quantumsim PROPERTIES
            BUILD_RPATH   "${QSIM_RPATH_LIST}"
            INSTALL_RPATH "${QSIM_RPATH_LIST}"
        )
        add_custom_command(TARGET quantumsim POST_BUILD
            COMMAND install_name_tool -change "${QSIM_OPENMP_LIBRARIES}"
                    "@rpath/libomp.dylib" "$<TARGET_FILE:quantumsim>"
            VERBATIM
        )
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

    if(QSIM_PLATFORM_WINDOWS AND QSIM_COMPILER_CLANG)
        message(STATUS "LTO: Disabled for Windows Clang")
    elseif(lto_supported)
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

    add_executable(test_constants tests/unit/test_constants.c)
    target_link_libraries(test_constants PRIVATE ${MATH_LIBRARY})
    add_test(NAME unit_constants COMMAND test_constants)

    add_executable(test_correctness_properties tests/unit/test_correctness_properties.c)
    target_link_libraries(test_correctness_properties PRIVATE quantumsim)
    add_test(NAME unit_correctness_properties COMMAND test_correctness_properties)

    add_executable(test_quantum_gates tests/unit/test_quantum_gates.c)
    target_link_libraries(test_quantum_gates PRIVATE quantumsim)
    add_test(NAME unit_quantum_gates COMMAND test_quantum_gates)

    add_executable(test_memory_align tests/unit/test_memory_align.c
                   src/optimization/memory_align.c)
    target_include_directories(test_memory_align PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_memory_align PRIVATE ${MATH_LIBRARY} Threads::Threads ${QSIM_CLANG_BUILTINS})
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_memory_align PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_memory_align PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_memory_align COMMAND test_memory_align)

    add_executable(test_simd_dispatch tests/unit/test_simd_dispatch.c
                   src/optimization/simd_dispatch.c)
    target_include_directories(test_simd_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_simd_dispatch PRIVATE ${MATH_LIBRARY} Threads::Threads ${QSIM_CLANG_BUILTINS})
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_simd_dispatch PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_simd_dispatch PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_simd_dispatch COMMAND test_simd_dispatch)

    add_executable(test_stride_gates tests/unit/test_stride_gates.c
                   src/optimization/stride_gates.c)
    target_include_directories(test_stride_gates PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_stride_gates PRIVATE ${MATH_LIBRARY} Threads::Threads ${QSIM_CLANG_BUILTINS})
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

    # Tensor-network unit test — the 800+ line suite that exercises
    # tensors, SVD, MPS, gate application via tensor networks,
    # measurement, and entanglement. Already in-tree but historically
    # not wired into ctest.
    add_executable(test_tensor_network tests/unit/test_tensor_network.c)
    target_link_libraries(test_tensor_network PRIVATE quantumsim)
    add_test(NAME unit_tensor_network COMMAND test_tensor_network)

    # Comprehensive end-to-end suite: gates, entanglement (von Neumann /
    # Renyi / concurrence), amplitude/phase damping noise, Grover, MPS.
    add_executable(test_comprehensive tests/test_comprehensive.c)
    target_link_libraries(test_comprehensive PRIVATE quantumsim)
    add_test(NAME comprehensive COMMAND test_comprehensive)

    # DMRG ground-state solver regression test.
    add_executable(test_dmrg tests/test_dmrg.c)
    target_link_libraries(test_dmrg PRIVATE quantumsim)
    add_test(NAME dmrg COMMAND test_dmrg)

    # MPS vs exact state-vector drift test — catches divergences as
    # bond dimension or system size grows.
    add_executable(test_mps_vs_exact tests/test_mps_vs_exact.c)
    target_link_libraries(test_mps_vs_exact PRIVATE quantumsim)
    add_test(NAME mps_vs_exact COMMAND test_mps_vs_exact)

    # Time-evolution long-run regression.
    add_executable(test_long_evolution tests/test_long_evolution.c)
    target_link_libraries(test_long_evolution PRIVATE quantumsim)
    add_test(NAME long_evolution COMMAND test_long_evolution)

    # Fast-measurement correctness test.
    add_executable(test_fast_measurement tests/test_fast_measurement.c)
    target_link_libraries(test_fast_measurement PRIVATE quantumsim)
    add_test(NAME fast_measurement COMMAND test_fast_measurement)

    # Measurement-subsystem unit test: probability queries, projective
    # collapse, Bell correlations, expectation values.
    add_executable(test_measurement tests/unit/test_measurement.c)
    target_link_libraries(test_measurement PRIVATE quantumsim)
    add_test(NAME unit_measurement COMMAND test_measurement)

    # Entanglement unit test: product-state zero entropy, Bell-state
    # maximal entropy, fidelity, purity, partial trace = I/2.
    add_executable(test_entanglement tests/unit/test_entanglement.c)
    target_link_libraries(test_entanglement PRIVATE quantumsim)
    add_test(NAME unit_entanglement COMMAND test_entanglement)

    # Noise-channel unit test: involution of bit/phase flips, depolarizing
    # p=0 no-op, norm preservation across every channel.
    add_executable(test_noise tests/unit/test_noise.c)
    target_link_libraries(test_noise PRIVATE quantumsim)
    add_test(NAME unit_noise COMMAND test_noise)

    # Grover algorithm smoke: on 3/4/5 qubits, success probability on the
    # marked state must exceed 0.8 after pi/4 * sqrt(N) iterations.
    add_executable(test_grover tests/unit/test_grover.c)
    target_link_libraries(test_grover PRIVATE quantumsim)
    add_test(NAME unit_grover COMMAND test_grover)

    # VQE H2 ground-state smoke.
    add_executable(test_vqe tests/unit/test_vqe.c)
    target_link_libraries(test_vqe PRIVATE quantumsim)
    add_test(NAME unit_vqe COMMAND test_vqe)

    # QAOA MaxCut smoke.
    add_executable(test_qaoa tests/unit/test_qaoa.c)
    target_link_libraries(test_qaoa PRIVATE quantumsim)
    add_test(NAME unit_qaoa COMMAND test_qaoa)

    # QPE phase-recovery smoke.
    add_executable(test_qpe tests/unit/test_qpe.c)
    target_link_libraries(test_qpe PRIVATE quantumsim)
    add_test(NAME unit_qpe COMMAND test_qpe)

    # Clifford stabilizer backend: H2=I, X, S4=I, Bell, 100-qubit GHZ.
    add_executable(test_clifford tests/unit/test_clifford.c)
    target_link_libraries(test_clifford PRIVATE quantumsim)
    add_test(NAME unit_clifford COMMAND test_clifford)

    # Gate-fusion DAG: random-circuit parity vs gate-by-gate execution.
    add_executable(test_fusion tests/unit/test_fusion.c)
    target_link_libraries(test_fusion PRIVATE quantumsim)
    add_test(NAME unit_fusion COMMAND test_fusion)

    # Quantum Volume harness: HOP above 2/3 on noiseless simulator.
    add_executable(test_quantum_volume tests/unit/test_quantum_volume.c)
    target_link_libraries(test_quantum_volume PRIVATE quantumsim)
    add_test(NAME unit_quantum_volume COMMAND test_quantum_volume)

    # Clifford-backed surface code: syndrome response at d=7,9,15.
    add_executable(test_surface_code_clifford tests/unit/test_surface_code_clifford.c)
    target_link_libraries(test_surface_code_clifford PRIVATE quantumsim)
    add_test(NAME unit_surface_code_clifford COMMAND test_surface_code_clifford)

    # Local Chern marker: QWZ topological vs trivial phases.
    add_executable(test_chern_marker tests/unit/test_chern_marker.c)
    target_link_libraries(test_chern_marker PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_chern_marker COMMAND test_chern_marker)

    # matrix_math ground-truth: matmul, trace, Hermitian check, eigenvalues.
    # Eigenvector correctness is tested ONLY on the real-symmetric path;
    # the complex-Hermitian branch is known unsound (see matrix_math.h).
    add_executable(test_matrix_math tests/unit/test_matrix_math.c)
    target_link_libraries(test_matrix_math PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_matrix_math COMMAND test_matrix_math)

    # Matrix-free KPM Chern marker: parity with dense + large-L demo.
    add_executable(test_chern_kpm tests/unit/test_chern_kpm.c)
    target_link_libraries(test_chern_kpm PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_chern_kpm COMMAND test_chern_kpm)
    if(NOT QSIM_PLATFORM_WINDOWS)
        add_executable(test_mpo_kpm tests/unit/test_mpo_kpm.c)
        target_link_libraries(test_mpo_kpm PRIVATE quantumsim ${MATH_LIBRARY})
        if(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_mpo_kpm PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
    endif()
    add_executable(test_gpu_backend_discovery tests/unit/test_gpu_backend_discovery.c)
    target_link_libraries(test_gpu_backend_discovery PRIVATE quantumsim)
    add_test(NAME gpu_backend_discovery COMMAND test_gpu_backend_discovery)
    add_executable(test_tensor_adversarial tests/unit/test_tensor_adversarial.c)
    target_link_libraries(test_tensor_adversarial PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME tensor_adversarial COMMAND test_tensor_adversarial)
    add_executable(test_differentiable tests/unit/test_differentiable.c)
    target_link_libraries(test_differentiable PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME differentiable COMMAND test_differentiable)

    add_executable(test_bell_variants tests/unit/test_bell_variants.c)
    target_link_libraries(test_bell_variants PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_bell_variants COMMAND test_bell_variants)

    add_executable(test_zne tests/unit/test_zne.c)
    target_link_libraries(test_zne PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_zne COMMAND test_zne)

    add_executable(test_povm tests/unit/test_povm.c)
    target_link_libraries(test_povm PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_povm COMMAND test_povm)

    add_executable(test_mutual_info tests/unit/test_mutual_info.c)
    target_link_libraries(test_mutual_info PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mutual_info COMMAND test_mutual_info)

    add_executable(test_qrng_di tests/unit/test_qrng_di.c)
    target_link_libraries(test_qrng_di PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qrng_di COMMAND test_qrng_di)

    add_executable(test_composite_noise tests/unit/test_composite_noise.c)
    target_link_libraries(test_composite_noise PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_composite_noise COMMAND test_composite_noise)

    add_executable(test_sha3 tests/unit/test_sha3.c)
    target_link_libraries(test_sha3 PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_sha3 COMMAND test_sha3)

    add_executable(test_mlkem_poly tests/unit/test_mlkem_poly.c)
    target_link_libraries(test_mlkem_poly PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem_poly COMMAND test_mlkem_poly)

    add_executable(test_mlkem tests/unit/test_mlkem.c)
    target_link_libraries(test_mlkem PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem COMMAND test_mlkem)

    add_executable(test_aes_drbg tests/unit/test_aes_drbg.c)
    target_link_libraries(test_aes_drbg PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_aes_drbg COMMAND test_aes_drbg)

    add_executable(test_mlkem_nist_kat tests/unit/test_mlkem_nist_kat.c)
    target_link_libraries(test_mlkem_nist_kat PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem_nist_kat COMMAND test_mlkem_nist_kat)
    add_executable(test_manifest tests/unit/test_manifest.c)
    target_link_libraries(test_manifest PRIVATE quantumsim)
    add_test(NAME unit_manifest COMMAND test_manifest)
    if(NOT QSIM_PLATFORM_WINDOWS AND QSIM_PLATFORM_MACOS)
        target_link_libraries(test_mpo_kpm PRIVATE "-framework Accelerate")
    endif()
    if(NOT QSIM_PLATFORM_WINDOWS)
        add_test(NAME unit_mpo_kpm COMMAND test_mpo_kpm)
    endif()

    # Quantum geometric tensor: Berry / Chern via FHS link variables.
    add_executable(test_qgt tests/unit/test_qgt.c)
    target_link_libraries(test_qgt PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qgt COMMAND test_qgt)

    # Shor-ECDLP resource estimator: reproduce Gidney-Drake-Boneh 2026
    # secp256k1 numbers + FTQC overhead model.
    add_executable(test_shor_ecdlp tests/unit/test_shor_ecdlp.c)
    target_link_libraries(test_shor_ecdlp PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_shor_ecdlp COMMAND test_shor_ecdlp)

    # Moonlab-Eshkol bridge parity: moonlab_eshkol_zgemm vs cblas_zgemm.
    # Always built; the test tolerates NOT_BUILT and NO_GPU by design.
    add_executable(test_gpu_eshkol tests/unit/test_gpu_eshkol.cpp)
    set_target_properties(test_gpu_eshkol PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_gpu_eshkol PRIVATE quantumsim)
    if(QSIM_PLATFORM_LINUX AND BLAS_LIB)
        target_link_libraries(test_gpu_eshkol PRIVATE ${BLAS_LIB})
    endif()
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_gpu_eshkol PRIVATE "-framework Accelerate")
    endif()
    add_test(NAME unit_gpu_eshkol COMMAND test_gpu_eshkol)

    # Chemistry module smoke.
    add_executable(test_chemistry tests/unit/test_chemistry.c)
    target_link_libraries(test_chemistry PRIVATE quantumsim)
    add_test(NAME unit_chemistry COMMAND test_chemistry)

    # Topological QC smoke.
    add_executable(test_topological tests/unit/test_topological.c)
    target_link_libraries(test_topological PRIVATE quantumsim)
    add_test(NAME unit_topological COMMAND test_topological)

    # Many-body localization smoke.
    add_executable(test_mbl tests/unit/test_mbl.c)
    target_link_libraries(test_mbl PRIVATE quantumsim)
    add_test(NAME unit_mbl COMMAND test_mbl)

    # SIMD / Accelerate parity against scalar reference.
    add_executable(test_simd_parity tests/unit/test_simd_parity.c)
    target_link_libraries(test_simd_parity PRIVATE quantumsim)
    if(QSIM_HAS_ACCELERATE)
        target_compile_definitions(test_simd_parity PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_simd_parity COMMAND test_simd_parity)

    # Metal GPU kernel parity against CPU (macOS + Metal only).
    add_executable(test_metal_parity tests/unit/test_metal_parity.c)
    target_link_libraries(test_metal_parity PRIVATE quantumsim)
    if(QSIM_HAS_METAL)
        target_compile_definitions(test_metal_parity PRIVATE HAS_METAL=1)
    endif()
    add_test(NAME unit_metal_parity COMMAND test_metal_parity)

    # 2D lattice + MPO-2D smoke (square / triangular / honeycomb).
    add_executable(test_lattice_2d tests/unit/test_lattice_2d.c)
    target_link_libraries(test_lattice_2d PRIVATE quantumsim)
    add_test(NAME unit_lattice_2d COMMAND test_lattice_2d)

    # Skyrmion braid path generators.
    add_executable(test_skyrmion tests/unit/test_skyrmion.c)
    target_link_libraries(test_skyrmion PRIVATE quantumsim)
    add_test(NAME unit_skyrmion COMMAND test_skyrmion)

    # Feynman diagram builder smoke.
    add_executable(test_feynman tests/unit/test_feynman.c)
    target_link_libraries(test_feynman PRIVATE quantumsim)
    add_test(NAME unit_feynman COMMAND test_feynman)

    # Circuit diagram builder smoke.
    add_executable(test_circuit_diagram tests/unit/test_circuit_diagram.c)
    target_link_libraries(test_circuit_diagram PRIVATE quantumsim)
    add_test(NAME unit_circuit_diagram COMMAND test_circuit_diagram)

    # QRNG statistical quality: chi^2 / monobit / serial correlation
    # on 256 KiB drawn through the stable ABI.
    add_executable(test_qrng_statistics tests/unit/test_qrng_statistics.c)
    target_link_libraries(test_qrng_statistics PRIVATE quantumsim)
    add_test(NAME unit_qrng_statistics COMMAND test_qrng_statistics)

    if(QSIM_BUILD_BENCHMARKS)
    # Performance benchmarks (build-only; no ctest entry so CI stays
    # fast). Run directly for local profiling:
    #   ./build/bench_state_operations
    #   ./build/bench_tensor_networks
    add_executable(bench_state_operations tests/performance/bench_state_operations.c)
    target_link_libraries(bench_state_operations PRIVATE quantumsim)
    add_executable(bench_tensor_networks tests/performance/bench_tensor_networks.c)
    target_link_libraries(bench_tensor_networks PRIVATE quantumsim)
    add_executable(bench_fusion tests/performance/bench_fusion.c)
    target_link_libraries(bench_fusion PRIVATE quantumsim)
    add_executable(bench_quantum_volume tests/performance/bench_quantum_volume.c)
    target_link_libraries(bench_quantum_volume PRIVATE quantumsim)
    add_executable(bench_clifford tests/performance/bench_clifford.c)
    target_link_libraries(bench_clifford PRIVATE quantumsim)
    add_executable(bench_chern_kpm tests/performance/bench_chern_kpm.c)
    target_link_libraries(bench_chern_kpm PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_chern_mosaic tests/performance/bench_chern_mosaic.c)
    target_link_libraries(bench_chern_mosaic PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_chern_mosaic_hq tests/performance/bench_chern_mosaic_hq.c)
    target_link_libraries(bench_chern_mosaic_hq PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_tensor_matmul_eshkol tests/performance/bench_tensor_matmul_eshkol.c)
    target_link_libraries(bench_tensor_matmul_eshkol PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_dmrg_workspace tests/performance/bench_dmrg_workspace.c)
    target_link_libraries(bench_dmrg_workspace PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_diff_adjoint tests/performance/bench_diff_adjoint.c)
    target_link_libraries(bench_diff_adjoint PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_hadamard tests/performance/bench_hadamard.c)
    target_link_libraries(bench_hadamard PRIVATE quantumsim ${MATH_LIBRARY})
    if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
        target_link_libraries(bench_hadamard PRIVATE ${QSIM_OPENMP_LIBRARIES})
        target_include_directories(bench_hadamard PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
    elseif(TARGET OpenMP::OpenMP_C)
        target_link_libraries(bench_hadamard PRIVATE OpenMP::OpenMP_C)
    endif()

    add_executable(bench_hadamard_metal tests/performance/bench_hadamard_metal.c)
    target_link_libraries(bench_hadamard_metal PRIVATE quantumsim)
    if(QSIM_HAS_METAL)
        target_compile_definitions(bench_hadamard_metal PRIVATE HAS_METAL=1)
    endif()

    # Optional: Eshkol fp64-on-Metal backend (SF64 / Ozaki-II CRT).
    # Enable via -DQSIM_ENABLE_ESHKOL=ON and -DQSIM_ESHKOL_ROOT=/path/to/eshkol.
    if(QSIM_ENABLE_ESHKOL)
        set(_eshkol_inc   "${QSIM_ESHKOL_ROOT}/inc")
        set(_eshkol_libfn "${QSIM_ESHKOL_ROOT}/build/libeshkol-static.a")
        if(NOT EXISTS "${_eshkol_libfn}")
            message(FATAL_ERROR
                "QSIM_ENABLE_ESHKOL=ON but ${_eshkol_libfn} not found. "
                "Build Eshkol first (cmake -S ${QSIM_ESHKOL_ROOT} -B "
                "${QSIM_ESHKOL_ROOT}/build && cmake --build "
                "${QSIM_ESHKOL_ROOT}/build).")
        endif()
        add_executable(bench_eshkol_gemm tests/performance/bench_eshkol_gemm.cpp)
        set_target_properties(bench_eshkol_gemm PROPERTIES CXX_STANDARD 17)
        target_include_directories(bench_eshkol_gemm PRIVATE "${_eshkol_inc}")
        target_compile_definitions(bench_eshkol_gemm PRIVATE HAS_ESHKOL=1)
        target_link_libraries(bench_eshkol_gemm PRIVATE
            "${_eshkol_libfn}"
            "-framework Accelerate"
            "-framework Metal"
            "-framework MetalPerformanceShaders"
            "-framework Foundation"
            "-framework CoreGraphics"
        )
        message(STATUS "Eshkol: ON (lib=${_eshkol_libfn})")
    endif()
    endif()

    # Python bindings smoke — imports moonlab.core, constructs a Bell
    # state, asserts exact amplitudes. Requires python3 on PATH plus
    # numpy (the smoke test imports it transitively through
    # moonlab.core).  Gate the registration on numpy being importable
    # so CI tiers that don't pre-install Python scientific packages
    # (e.g. the plain macos-arm64 / macos-debug / macos-werror tiers
    # that only brew-install libomp + ninja) don't fail here; the
    # dedicated bindings-smoke job does pip-install numpy first and
    # will still run the tests.
    find_program(PYTHON3_EXECUTABLE python3)
    if(PYTHON3_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON3_EXECUTABLE} -c "import numpy"
            RESULT_VARIABLE _numpy_probe
            OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(PYTHON3_EXECUTABLE AND _numpy_probe EQUAL 0)
        add_test(NAME python_bindings_smoke
                 COMMAND ${PYTHON3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_bindings_smoke.py)
        set_tests_properties(python_bindings_smoke PROPERTIES
            ENVIRONMENT
                "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )

        # Full pytest suite — covers measurement entropy-ctx, algorithms
        # (VQE/QAOA/Grover/BellTest), and state ops.  Gated on pytest
        # being importable; otherwise skipped so bare-Python installs
        # don't break the default build.
        execute_process(
            COMMAND ${PYTHON3_EXECUTABLE} -c "import pytest"
            RESULT_VARIABLE _pytest_probe
            OUTPUT_QUIET ERROR_QUIET)
        if(_pytest_probe EQUAL 0)
            add_test(NAME python_bindings_pytest
                     COMMAND ${PYTHON3_EXECUTABLE} -m pytest -q
                             --deselect tests/test_algorithms.py::TestVQEH2
                             --deselect tests/test_algorithms.py::TestGroverSearch::test_grover_search_medium
                             --deselect tests/test_algorithms.py::TestCHSHTest::test_chsh_high_statistics
                             ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/tests
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python)
            set_tests_properties(python_bindings_pytest PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
            )
        endif()
    endif()

    # Rust bindings smoke — cargo test in bindings/rust/moonlab, only
    # if the crate root and Cargo are present.
    find_program(CARGO_EXECUTABLE cargo)
    if(CARGO_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/rust/moonlab/Cargo.toml")
        add_test(NAME rust_bindings_smoke
                 COMMAND ${CARGO_EXECUTABLE} test --manifest-path
                         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/rust/moonlab/Cargo.toml
                         --no-fail-fast)
        set_tests_properties(rust_bindings_smoke PROPERTIES
            ENVIRONMENT
                "MOONLAB_LIB_DIR=$<TARGET_FILE_DIR:quantumsim>;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )
    endif()

    # WebGPU parity smoke — runs the existing randomized eval script.
    # Requires pnpm, emscripten-built moonlab.wasm, a pnpm-built TS dist,
    # and optionally a WebGPU-capable runtime (Chromium 113+ or Deno 2).
    #
    # We gate on both the script existing AND the compiled dist already
    # being present, because CMake does not own the pnpm build graph;
    # a fresh clone without `pnpm -C bindings/javascript/packages/core
    # build` would otherwise fail this test on first run.
    #
    # With -DQSIM_BUILD_JS_DIST=ON, we invoke the pnpm build ourselves
    # via a custom_target so dist/index.mjs lands on disk before
    # ctest runs.  Default OFF keeps routine builds fast.
    find_program(PNPM_EXECUTABLE pnpm)
    set(_js_dist "${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core/dist/index.mjs")
    if(QSIM_BUILD_JS_DIST AND PNPM_EXECUTABLE)
        add_custom_command(
            OUTPUT ${_js_dist}
            COMMAND ${PNPM_EXECUTABLE} install --ignore-scripts
            COMMAND ${PNPM_EXECUTABLE} run build:ts
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core
            COMMENT "Building TS/JS dist via pnpm (QSIM_BUILD_JS_DIST=ON)"
            VERBATIM
        )
        add_custom_target(js_dist ALL DEPENDS ${_js_dist})
    endif()
    if(PNPM_EXECUTABLE AND EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core/scripts/webgpu-unified-smoke.mjs"
        AND (EXISTS "${_js_dist}" OR QSIM_BUILD_JS_DIST))
        add_test(NAME webgpu_unified_smoke
                 COMMAND ${PNPM_EXECUTABLE} -C
                         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core
                         exec node scripts/webgpu-unified-smoke.mjs)
    endif()

    # Distributed computing tests (MPI). Each test executable is only
    # added if the corresponding source file is present — the MPI test
    # sources are planned per the 0.x release roadmap but not all are
    # in the tree yet. This prevents CMake generate errors in builds
    # that enable MPI before the matching test is written.
    if(QSIM_HAS_MPI)
        set(_mpi_tests_added 0)
        foreach(_mpi_test_name IN ITEMS
                distributed_gates
                state_partition
                collective_ops)
            set(_mpi_test_src "tests/unit/test_${_mpi_test_name}.c")
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_mpi_test_src}")
                add_executable(test_${_mpi_test_name} ${_mpi_test_src})
                target_link_libraries(test_${_mpi_test_name}
                                      PRIVATE quantumsim MPI::MPI_C)
                # --oversubscribe (Open MPI >= 5) or --use-hwthread-cpus
                # lets us run 4 MPI procs on CI runners with fewer than
                # 4 physical cores (macos-14 exposes 3 cores).  The flag
                # is a no-op when there are enough slots.
                add_test(NAME ${_mpi_test_name}
                         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                                 --oversubscribe
                                 ${MPIEXEC_PREFLAGS}
                                 $<TARGET_FILE:test_${_mpi_test_name}>
                                 ${MPIEXEC_POSTFLAGS})
                math(EXPR _mpi_tests_added "${_mpi_tests_added} + 1")
            endif()
        endforeach()
        if(_mpi_tests_added GREATER 0)
            message(STATUS "MPI Tests: Enabled (${_mpi_tests_added} test(s), 4 MPI processes each)")
        else()
            message(STATUS "MPI Tests: None present in tests/unit/ yet")
        endif()
    endif()

    # Downstream-ABI smoke test — dlopens libquantumsim and verifies the
    # symbols declared in src/applications/moonlab_export.h. Only meaningful
    # for shared builds (dlopen needs a dylib/so).
    if(QSIM_BUILD_SHARED AND NOT QSIM_PLATFORM_WINDOWS)
        add_executable(test_moonlab_export_abi tests/abi/test_moonlab_export_abi.c)
        target_link_libraries(test_moonlab_export_abi PRIVATE ${CMAKE_DL_LIBS})
        add_test(NAME abi_moonlab_export
                 COMMAND test_moonlab_export_abi)
        # Make the freshly-built libquantumsim discoverable to the test at
        # run time without requiring `make install`.
        set_tests_properties(abi_moonlab_export PROPERTIES
            ENVIRONMENT
                "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )
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

    add_executable(pqc_qrng_demo examples/applications/pqc_qrng_demo.c)
    target_link_libraries(pqc_qrng_demo PRIVATE quantumsim)

    add_executable(diff_vqe_demo examples/tensor_network/diff_vqe_demo.c)
    target_link_libraries(diff_vqe_demo PRIVATE quantumsim ${MATH_LIBRARY})

    add_executable(quantum_spin_chain examples/tensor_network/quantum_spin_chain.c)
    target_link_libraries(quantum_spin_chain PRIVATE quantumsim)
    if(QSIM_HAS_OPENMP)
        if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
            target_link_libraries(quantum_spin_chain PRIVATE ${QSIM_OPENMP_LIBRARIES})
            target_include_directories(quantum_spin_chain PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
            target_compile_options(quantum_spin_chain PRIVATE -Xpreprocessor -fopenmp)
        elseif(TARGET OpenMP::OpenMP_C)
            target_link_libraries(quantum_spin_chain PRIVATE OpenMP::OpenMP_C)
        endif()
        target_compile_definitions(quantum_spin_chain PRIVATE HAS_OPENMP=1)
    endif()

    add_executable(quantum_critical_point examples/tensor_network/quantum_critical_point.c)
    target_link_libraries(quantum_critical_point PRIVATE quantumsim)

    # Execute a representative subset of examples under ctest so CI
    # catches link-time success followed by runtime crashes. Heavy
    # examples (quantum_spin_chain at 100 qubits, grover_large_scale)
    # are excluded — they're correct but multi-minute runs.
    if(QSIM_BUILD_TESTS)
        add_test(NAME example_vqe_h2_molecule
                 COMMAND vqe_h2_molecule)
        add_test(NAME example_qaoa_maxcut
                 COMMAND qaoa_maxcut)
        add_test(NAME example_grover_hash_collision
                 COMMAND grover_hash_collision)
        add_test(NAME example_quantum_spin_chain_small
                 COMMAND quantum_spin_chain 8 3 16)
        add_test(NAME example_phase3_phase4_benchmark
                 COMMAND phase3_phase4_benchmark 16)
        # quantum_critical_point is a multi-minute finite-size-scaling
        # demo and is deliberately excluded from the CI smoke.
        set_tests_properties(
            example_vqe_h2_molecule
            example_qaoa_maxcut
            example_grover_hash_collision
            example_quantum_spin_chain_small
            example_phase3_phase4_benchmark
            PROPERTIES TIMEOUT 120
        )
    endif()

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

# pkg-config file.  configure_file happens at *configure* time, so
# CMAKE_INSTALL_PREFIX inside the .pc is locked to the cache value
# when `cmake -S . -B build` ran.  If a consumer installs to a
# different prefix via `cmake --install ... --prefix ...`, the .pc
# would point at the wrong location.  Generate it at install time
# instead so the prefix substitution sees the actual destination.
install(CODE "
    set(CMAKE_INSTALL_PREFIX   \"\${CMAKE_INSTALL_PREFIX}\")
    set(CMAKE_INSTALL_LIBDIR   \"${CMAKE_INSTALL_LIBDIR}\")
    set(CMAKE_INSTALL_INCLUDEDIR \"${CMAKE_INSTALL_INCLUDEDIR}\")
    set(PROJECT_DESCRIPTION     \"${PROJECT_DESCRIPTION}\")
    set(PROJECT_VERSION         \"${PROJECT_VERSION}\")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/quantumsim.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc.install
        @ONLY)
")

# ==============================================================================
# Reproducibility manifest build-info header (P5.18)
# ==============================================================================
#
# Capture the current git SHA, dirty state, branch, and build timestamp at
# configure time, then configure_file -> generated header so every consumer
# can self-describe the source tree they came from.  Run again on every
# configure; a full rebuild picks up a new commit's SHA.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_SHA_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD --
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE MOONLAB_GIT_CLEAN_RC
        ERROR_QUIET
    )
    if(MOONLAB_GIT_CLEAN_RC EQUAL 0)
        set(MOONLAB_GIT_DIRTY 0)
    else()
        set(MOONLAB_GIT_DIRTY 1)
    endif()
endif()

if(NOT MOONLAB_GIT_SHA)
    set(MOONLAB_GIT_SHA "unknown")
    set(MOONLAB_GIT_SHA_SHORT "unknown")
    set(MOONLAB_GIT_BRANCH "unknown")
    set(MOONLAB_GIT_DIRTY 0)
endif()

string(TIMESTAMP MOONLAB_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

# Translate CMake booleans into 0/1 macros for the generated header.
foreach(flag
    QSIM_ENABLE_OPENMP QSIM_ENABLE_METAL QSIM_ENABLE_OPENCL QSIM_ENABLE_VULKAN
    QSIM_ENABLE_CUQUANTUM QSIM_ENABLE_MPI QSIM_ENABLE_ESHKOL QSIM_ENABLE_LTO
    QSIM_ENABLE_SANITIZERS QSIM_ENABLE_AVX512 QSIM_ENABLE_AVX2
    QSIM_ENABLE_NEON QSIM_ENABLE_SVE)
    if(${flag})
        set(${flag}_INT 1)
    else()
        set(${flag}_INT 0)
    endif()
endforeach()

# Assemble a human-readable enabled-features list.
set(MOONLAB_ENABLED_FEATURES_LIST "")
foreach(pair
    "QSIM_ENABLE_OPENMP:openmp"
    "QSIM_ENABLE_METAL:metal"
    "QSIM_ENABLE_OPENCL:opencl"
    "QSIM_ENABLE_VULKAN:vulkan"
    "QSIM_ENABLE_CUQUANTUM:cuquantum"
    "QSIM_ENABLE_MPI:mpi"
    "QSIM_ENABLE_ESHKOL:eshkol"
    "QSIM_ENABLE_LTO:lto"
    "QSIM_ENABLE_SANITIZERS:sanitizers"
    "QSIM_ENABLE_AVX512:avx512"
    "QSIM_ENABLE_AVX2:avx2"
    "QSIM_ENABLE_NEON:neon"
    "QSIM_ENABLE_SVE:sve")
    string(REPLACE ":" ";" kv "${pair}")
    list(GET kv 0 flag)
    list(GET kv 1 label)
    if(${flag})
        list(APPEND MOONLAB_ENABLED_FEATURES_LIST "${label}")
    endif()
endforeach()
string(REPLACE ";" "," MOONLAB_ENABLED_FEATURES "${MOONLAB_ENABLED_FEATURES_LIST}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/moonlab_build_info.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/moonlab_build_info.h
    @ONLY
)
target_include_directories(quantumsim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc.install
    RENAME quantumsim.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Quantum Simulator Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:      ${PROJECT_VERSION}${QSIM_VERSION_PRE}")
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
