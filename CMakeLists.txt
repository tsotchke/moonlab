# ==============================================================================
# Quantum Simulator - CMake Build System
# ==============================================================================
#
# Cross-platform build configuration supporting:
# - macOS (Apple Silicon + Intel)
# - Linux (x86_64 + ARM64)
# - Windows (x86_64 + ARM64, ClangCL)
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

# Split MAJOR.MINOR.PATCH for the moonlab_features.h template so callers
# can branch on numeric API versions without re-parsing the string.
string(REPLACE "." ";" _qsim_version_list "${QSIM_VERSION}")
list(GET _qsim_version_list 0 QSIM_VERSION_MAJOR)
list(GET _qsim_version_list 1 QSIM_VERSION_MINOR)
list(GET _qsim_version_list 2 QSIM_VERSION_PATCH)
unset(_qsim_version_list)

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
option(QSIM_BUILD_VISUALIZATION "Build the circuit / Feynman diagram renderer module" ON)
option(QSIM_BUILD_BENCHMARKS "Build benchmarks" ON)
option(QSIM_BUILD_JS_DIST "Auto-build the TS/JS dist via pnpm (enables webgpu_unified_smoke on fresh clones)" OFF)
option(QSIM_PYTHON_WHEEL "Install only the native library into moonlab/.libs for a platform wheel" OFF)

# Feature options
option(QSIM_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(QSIM_ENABLE_METAL "Enable Metal GPU backend (macOS)" ON)
option(QSIM_ENABLE_OPENCL "Enable OpenCL GPU backend" OFF)
option(QSIM_ENABLE_VULKAN "Enable Vulkan GPU backend" OFF)
option(QSIM_ENABLE_CUDA      "Enable moonlab native CUDA backend (Jetson/Tegra + discrete)" OFF)
option(QSIM_ENABLE_CUQUANTUM "Enable NVIDIA cuQuantum backend (implies QSIM_ENABLE_CUDA; not supported on Jetson)" OFF)
option(QSIM_ENABLE_MPI "Enable MPI distributed computing" OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_qsim_control_plane_default OFF)
else()
    set(_qsim_control_plane_default ON)
endif()
option(QSIM_ENABLE_CONTROL_PLANE "Build POSIX TCP control-plane transport and tests" ${_qsim_control_plane_default})
# TLS defaults ON if OpenSSL is detectable at config time: the control-plane
# transport's TLS/mTLS/HMAC-SHA3 path is part of the v1.0 stable ABI, and
# silently dropping back to plaintext on a build that didn't pass the flag
# (the v1.0 release of build_release shipped this way and broke binding TLS
# tests) is worse than a hard error.  Set -DQSIM_ENABLE_TLS=OFF explicitly
# to opt out, or the build will fail-fast with a clear error below.
include(CMakeDependentOption)
if(APPLE AND NOT OPENSSL_ROOT_DIR)
    execute_process(COMMAND brew --prefix openssl@3
                    OUTPUT_VARIABLE _qsim_brew_openssl
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_qsim_brew_openssl)
        set(OPENSSL_ROOT_DIR "${_qsim_brew_openssl}" CACHE PATH "OpenSSL root (Homebrew-probed default)")
    endif()
endif()
find_package(OpenSSL QUIET)
if(OPENSSL_FOUND)
    set(_qsim_tls_default ON)
else()
    set(_qsim_tls_default OFF)
endif()
option(QSIM_ENABLE_TLS "Link OpenSSL for TLS-wrapped control-plane transport (since v0.8.17, default ON when OpenSSL is found)" ${_qsim_tls_default})
option(QSIM_ENABLE_ESHKOL "Link Eshkol GPU backend for fp64-on-Metal (Ozaki-II / SF64)" OFF)
set(QSIM_ESHKOL_ROOT "$ENV{HOME}/Desktop/eshkol" CACHE PATH
    "Root directory of the Eshkol source tree (for headers + libeshkol-static.a).")
option(QSIM_ENABLE_LIBIRREP "Link libirrep for QEC + sector ED + rep theory primitives (since v0.6.0)" OFF)
set(QSIM_LIBIRREP_ROOT "$ENV{LIBIRREP_ROOT}" CACHE PATH
    "Root of a libirrep source/build tree (fallback when find_package + pkg-config both fail).")
option(QSIM_ENABLE_SBNN "Link SbNN for the learned QEC-decoder slot (since v0.7.5)" OFF)
set(QSIM_SBNN_ROOT "$ENV{SBNN_ROOT}" CACHE PATH
    "Root of a SbNN source/build tree.  Expected layout: <root>/include/qec_decoder/qec_decoder.h, <root>/build/libthqcp.{a,dylib,so}.")
option(QSIM_ENABLE_LTO "Enable Link-Time Optimization" ON)
option(QSIM_NATIVE_ARCH "Tune Release builds for the build machine CPU (-march=native). Turn OFF for distributable artifacts so binaries run on any CPU of the target architecture; runtime SIMD dispatch is unaffected." ON)
option(QSIM_FAST_MATH "Enable non-IEEE fast-math optimizations (may change NaN/Inf semantics)" OFF)
option(QSIM_WERROR     "Treat warnings as errors (excludes -Wpedantic for OpenMP system headers, -Wdeprecated-declarations for Apple CLAPACK)" ON)
option(QSIM_ENABLE_SANITIZERS "Build with -fsanitize=address,undefined (for CI / dev)" OFF)
option(QSIM_ENABLE_COVERAGE   "Build with --coverage (gcov-compatible) for CI coverage reporting" OFF)

# v0.3 ABI tightening: when ON, applies -fvisibility=hidden globally so
# only symbols decorated with MOONLAB_API (in src/applications/moonlab_export.h)
# are exported from libquantumsim.dylib.  Defaults OFF for v0.2.x
# because the Python and Rust bindings currently consume ~108 internal
# entry points (gate_*, quantum_*, vqe_*, chern_*, etc.) that are not
# yet in the stable ABI; flipping this to ON without first promoting
# those entries will break the bindings.  The v0.3 cycle's plan is:
# (1) tag every binding-consumed symbol with MOONLAB_API at its decl
# site, (2) flip this default to ON, (3) ship a v1.0 stable ABI.
# Enable here for early-adopter consumers who want to verify their
# usage stays inside the tagged surface.
option(QSIM_HIDDEN_VISIBILITY "Hide non-tagged symbols (forward-compat for v0.3 ABI)" OFF)

# Force the in-tree no-LAPACK QR / SVD fallbacks even on platforms that have
# Accelerate / OpenBLAS / LAPACKE.  Used by a dedicated CI tier so the
# fallback codepaths -- which production WASM and Windows clang-cl builds
# rely on -- are exercised on every PR instead of only when something
# happens to break in the wild.
option(QSIM_FORCE_QR_FALLBACK  "Route tensor_qr through the no-LAPACK Householder fallback" OFF)
option(QSIM_FORCE_SVD_FALLBACK "Route tensor_svd through the no-LAPACK Jacobi fallback" OFF)

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
    # _GNU_SOURCE exposes strptime, asprintf, etc. (POSIX 2001 + GNU
    # extensions).  Required by src/utils/manifest.c on glibc-based
    # systems where the default feature-test macros hide strptime.
    add_compile_definitions(_GNU_SOURCE)
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

include(cmake/compiler_flags.cmake)

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
    if(NOT MATH_LIBRARY)
        # find_library can miss libm in minimal/emulated environments
        # (QEMU cross containers) even though the toolchain links it
        # fine. Fall back to the plain linker name.
        set(MATH_LIBRARY m)
    endif()
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

# Helper macro: link a target against the platform-resolved OpenMP
# runtime.  Replaces 20 copies of an identical conditional block that
# bound macOS Homebrew libomp paths and the standard OpenMP::OpenMP_C
# CMake target, with the matching include-directory dance for macOS.
# Macro (not function) so caller-provided variables interpolate
# correctly into target_link_libraries.
macro(qsim_target_link_openmp tgt)
    if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
        target_link_libraries(${tgt} PRIVATE ${QSIM_OPENMP_LIBRARIES})
        target_include_directories(${tgt} PRIVATE ${QSIM_OPENMP_INCLUDE_DIR})
    elseif(TARGET OpenMP::OpenMP_C)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
    endif()
endmacro()

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
        message(FATAL_ERROR
            "QSIM_ENABLE_OPENCL=ON but OpenCL was not found by find_package. "
            "Install an OpenCL SDK (Khronos headers + ICD loader) or "
            "re-run with -DQSIM_ENABLE_OPENCL=OFF.")
    endif()
endif()

# Vulkan
if(QSIM_ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan: Found")
        set(QSIM_HAS_VULKAN ON)
    else()
        message(FATAL_ERROR
            "QSIM_ENABLE_VULKAN=ON but Vulkan was not found by find_package. "
            "Install the Vulkan SDK (LunarG SDK on Linux/Windows, MoltenVK "
            "on macOS) or re-run with -DQSIM_ENABLE_VULKAN=OFF.")
    endif()
endif()

# CUDA / cuQuantum
# QSIM_ENABLE_CUDA      -- our own state-vector backend.  Works on
#                          Jetson (Tegra194 / Tegra234 / Orin) AND on
#                          discrete PCIe GPUs.  cudaMallocManaged path.
# QSIM_ENABLE_CUQUANTUM -- additionally link NVIDIA's cuStateVec /
#                          cuTensorNet.  Implies CUDA.  Not supported
#                          on Jetson: NVIDIA does not publish a
#                          cuQuantum build for L4T/Tegra, so cmake
#                          will fall back to the moonlab CUDA backend
#                          without cuQuantum if the libraries are
#                          missing on a Tegra host.
if(QSIM_ENABLE_CUDA OR QSIM_ENABLE_CUQUANTUM)
    # Normalize CUDA_ARCHITECTURES before enable_language so cmake
    # picks up the "-real" suffix.  Bare numbers like "72" produce
    # the bracket form code=[compute_72,sm_72] which includes
    # device-LTO IR and triggers `-dlto` at device-link time.
    # On jetpack-nixos that fails ("elfLink linker library load
    # error") because the LTO plugin isn't on nvlink's dlopen path.
    # "-real" suffix tells cmake to emit only SASS for that arch,
    # which avoids LTO entirely.
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_qsim_cuda_arch_real)
        foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(_a MATCHES "^[0-9]+$")
                list(APPEND _qsim_cuda_arch_real "${_a}-real")
            else()
                list(APPEND _qsim_cuda_arch_real "${_a}")
            endif()
        endforeach()
        set(CMAKE_CUDA_ARCHITECTURES ${_qsim_cuda_arch_real} CACHE STRING "" FORCE)
        message(STATUS "CUDA: arch normalized -> ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Pre-seed CMAKE_CUDA_FLAGS with the cuda_runtime.h include dir
    # BEFORE enable_language(CUDA), so the cmake compiler-identification
    # probe finds the header.  nvcc on its own often doesn't pass its
    # toolkit include dir to the host gcc, depending on how the toolkit
    # was packaged:
    #   - jetpack-nixos:  multiple /nix/store subpackages, no auto-include
    #   - Ubuntu / /usr/local/cuda-X.Y:  nvcc inside .../cuda-X.Y/bin
    #     doesn't always self-locate .../cuda-X.Y/include
    # So: probe several common locations and inject -isystem.
    set(_qsim_cuda_runtime_h "")
    # 1. User-provided CUDAToolkit_ROOT (most explicit).
    if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
        set(_qsim_cuda_runtime_h "${CUDAToolkit_ROOT}/include/cuda_runtime.h")
    endif()
    # 2. /usr/local/cuda* (stock Ubuntu / RHEL CUDA install).
    if(NOT _qsim_cuda_runtime_h)
        file(GLOB _qsim_cuda_local_globs "/usr/local/cuda*/include/cuda_runtime.h")
        if(_qsim_cuda_local_globs)
            list(GET _qsim_cuda_local_globs 0 _qsim_cuda_runtime_h)
        endif()
    endif()
    # 3. jetpack-nixos split-toolkit store paths.
    if(NOT _qsim_cuda_runtime_h AND EXISTS "/etc/nv_tegra_release")
        file(GLOB _qsim_cuda_merged_globs
            "/nix/store/*-cuda-merged-*/include/cuda_runtime.h"
            "/nix/store/*-cuda_cudart-*/include/cuda_runtime.h")
        if(_qsim_cuda_merged_globs)
            list(GET _qsim_cuda_merged_globs 0 _qsim_cuda_runtime_h)
        endif()
        set(QSIM_PLATFORM_JETPACK_NIXOS ON)
    endif()
    if(_qsim_cuda_runtime_h)
        get_filename_component(QSIM_CUDA_INCLUDE "${_qsim_cuda_runtime_h}" DIRECTORY)
        message(STATUS "CUDA: pre-seeding cuda_runtime.h from ${QSIM_CUDA_INCLUDE}")
        set(CMAKE_CUDA_FLAGS_INIT "-isystem=${QSIM_CUDA_INCLUDE} ${CMAKE_CUDA_FLAGS_INIT}")
        set(CMAKE_CUDA_FLAGS      "-isystem=${QSIM_CUDA_INCLUDE} ${CMAKE_CUDA_FLAGS}")
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA: Found (version ${CUDAToolkit_VERSION})")
            set(QSIM_HAS_CUDA ON)

            # Look for cuQuantum (optional even when QSIM_ENABLE_CUDA=ON).
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
            elseif(QSIM_ENABLE_CUQUANTUM)
                message(FATAL_ERROR
                    "QSIM_ENABLE_CUQUANTUM=ON and CUDA was found, but the "
                    "cuQuantum headers (custatevec.h) or cuStateVec library "
                    "could not be located.  Set CUQUANTUM_ROOT or "
                    "$ENV{CUQUANTUM_ROOT} to the cuQuantum install root, or "
                    "re-run with -DQSIM_ENABLE_CUQUANTUM=OFF.")
            else()
                message(STATUS "cuQuantum: not found (OK -- only QSIM_ENABLE_CUDA was set; moonlab CUDA backend works without it)")
            endif()
        elseif(QSIM_ENABLE_CUQUANTUM)
            message(FATAL_ERROR
                "QSIM_ENABLE_CUQUANTUM=ON but the CUDA Toolkit was not "
                "found by find_package(CUDAToolkit).  Install the CUDA "
                "Toolkit or re-run with -DQSIM_ENABLE_CUQUANTUM=OFF.")
        else()
            message(WARNING
                "QSIM_ENABLE_CUDA=ON but the CUDA Toolkit was not found.  "
                "Falling back to CPU-only build.")
        endif()
    elseif(QSIM_ENABLE_CUQUANTUM)
        message(FATAL_ERROR
            "QSIM_ENABLE_CUQUANTUM=ON but no CUDA compiler is available. "
            "Install the CUDA Toolkit (which provides nvcc) or re-run "
            "with -DQSIM_ENABLE_CUQUANTUM=OFF.")
    else()
        message(WARNING
            "QSIM_ENABLE_CUDA=ON but no CUDA compiler (nvcc) is available.  "
            "Falling back to CPU-only build.")
    endif()
endif()

# MPI
if(QSIM_ENABLE_MPI)
    find_package(MPI)
    if(MPI_C_FOUND)
        message(STATUS "MPI: Found")
        set(QSIM_HAS_MPI ON)
    else()
        message(FATAL_ERROR
            "QSIM_ENABLE_MPI=ON but MPI was not found by find_package. "
            "Install OpenMPI / MPICH (Linux: apt install libopenmpi-dev, "
            "macOS: brew install open-mpi) or re-run with "
            "-DQSIM_ENABLE_MPI=OFF.")
    endif()
endif()

# TLS via OpenSSL (since v0.8.17).  The Homebrew probe + initial
# detection happen up at the option block so the option default tracks
# OpenSSL availability; here we re-verify and emit the status message.
if(QSIM_ENABLE_TLS)
    find_package(OpenSSL)
    if(OPENSSL_FOUND)
        message(STATUS "OpenSSL: ${OPENSSL_VERSION} at ${OPENSSL_INCLUDE_DIR}")
        set(QSIM_HAS_TLS ON)
    else()
        message(FATAL_ERROR
            "QSIM_ENABLE_TLS=ON but OpenSSL was not found.  "
            "Install via apt install libssl-dev / brew install openssl@3, "
            "or re-run with -DQSIM_ENABLE_TLS=OFF.")
    endif()
endif()

# libirrep -- QEC + rep-theory + sector ED substrate.
# Detection order:
#   1. find_package(libirrep CONFIG)  -- preferred, picks up the installed
#      libirrepTargets that libirrep emits via install(EXPORT ...).
#   2. pkg-config (libirrep.pc).
#   3. QSIM_LIBIRREP_ROOT / LIBIRREP_ROOT pointing at a source tree with a
#      pre-built build/lib/liblibirrep.{a,dylib,so}.
# Mirrors SbNN's IRREP_ENABLE pattern.  When the option is OFF the bridge
# layer compiles down to stub functions that return
# MOONLAB_STATUS_UNSUPPORTED, so consumers can build moonlab without
# libirrep installed.
if(QSIM_ENABLE_LIBIRREP)
    set(QSIM_HAS_LIBIRREP OFF)

    find_package(libirrep CONFIG QUIET)
    if(libirrep_FOUND)
        message(STATUS "libirrep: Found via find_package(libirrep CONFIG)")
        set(QSIM_HAS_LIBIRREP ON)
        set(QSIM_LIBIRREP_TARGET libirrep::libirrep)
    endif()

    if(NOT QSIM_HAS_LIBIRREP)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LIBIRREP_PC QUIET libirrep)
            if(LIBIRREP_PC_FOUND)
                message(STATUS "libirrep: Found via pkg-config (${LIBIRREP_PC_VERSION})")
                add_library(qsim_libirrep_import INTERFACE)
                target_include_directories(qsim_libirrep_import INTERFACE
                    ${LIBIRREP_PC_INCLUDE_DIRS})
                target_link_directories(qsim_libirrep_import INTERFACE
                    ${LIBIRREP_PC_LIBRARY_DIRS})
                target_link_libraries(qsim_libirrep_import INTERFACE
                    ${LIBIRREP_PC_LIBRARIES})
                set(QSIM_HAS_LIBIRREP ON)
                set(QSIM_LIBIRREP_TARGET qsim_libirrep_import)
            endif()
        endif()
    endif()

    if(NOT QSIM_HAS_LIBIRREP AND QSIM_LIBIRREP_ROOT)
        set(_libirrep_inc "${QSIM_LIBIRREP_ROOT}/include")
        find_library(QSIM_LIBIRREP_LIB
            NAMES libirrep irrep
            PATHS
                "${QSIM_LIBIRREP_ROOT}/build/lib"
                "${QSIM_LIBIRREP_ROOT}/lib"
                "${QSIM_LIBIRREP_ROOT}/build"
            NO_DEFAULT_PATH)
        if(EXISTS "${_libirrep_inc}/irrep/irrep.h" AND QSIM_LIBIRREP_LIB)
            message(STATUS "libirrep: Found at ${QSIM_LIBIRREP_ROOT} "
                           "(lib: ${QSIM_LIBIRREP_LIB})")
            add_library(qsim_libirrep_import INTERFACE)
            target_include_directories(qsim_libirrep_import INTERFACE
                "${_libirrep_inc}")
            target_link_libraries(qsim_libirrep_import INTERFACE
                "${QSIM_LIBIRREP_LIB}")
            set(QSIM_HAS_LIBIRREP ON)
            set(QSIM_LIBIRREP_TARGET qsim_libirrep_import)
        endif()
    endif()

    if(NOT QSIM_HAS_LIBIRREP)
        message(FATAL_ERROR
            "QSIM_ENABLE_LIBIRREP=ON but libirrep could not be located.  "
            "Install libirrep (provides libirrepConfig.cmake or libirrep.pc) "
            "or pass -DQSIM_LIBIRREP_ROOT=<path> pointing at a source tree "
            "with build/lib/liblibirrep.{a,dylib,so}.  Re-run with "
            "-DQSIM_ENABLE_LIBIRREP=OFF to skip this dependency.")
    endif()
else()
    set(QSIM_HAS_LIBIRREP OFF)
endif()

# SbNN -- learned QEC decoder + ToricCode + equivariant ML.
# Wires the decoder-bench SBNN slot to qec_decoder_run() via the
# `libthqcp.{a,dylib,so}` SbNN ships in `<root>/build/`.
if(QSIM_ENABLE_SBNN)
    set(QSIM_HAS_SBNN OFF)
    if(QSIM_SBNN_ROOT)
        set(_sbnn_inc "${QSIM_SBNN_ROOT}/include")
        find_library(QSIM_SBNN_LIB
            NAMES thqcp libthqcp
            PATHS
                "${QSIM_SBNN_ROOT}/build"
                "${QSIM_SBNN_ROOT}/build/lib"
                "${QSIM_SBNN_ROOT}/lib"
            NO_DEFAULT_PATH)
        if(EXISTS "${_sbnn_inc}/qec_decoder/qec_decoder.h" AND QSIM_SBNN_LIB)
            message(STATUS "SbNN: Found at ${QSIM_SBNN_ROOT} (lib: ${QSIM_SBNN_LIB})")
            add_library(qsim_sbnn_import INTERFACE)
            target_include_directories(qsim_sbnn_import INTERFACE "${_sbnn_inc}")
            target_link_libraries(qsim_sbnn_import INTERFACE "${QSIM_SBNN_LIB}")
            set(QSIM_HAS_SBNN ON)
            set(QSIM_SBNN_TARGET qsim_sbnn_import)
        endif()
    endif()
    if(NOT QSIM_HAS_SBNN)
        message(FATAL_ERROR
            "QSIM_ENABLE_SBNN=ON but SbNN could not be located.  "
            "Pass -DQSIM_SBNN_ROOT=<path> pointing at a tree with "
            "include/qec_decoder/qec_decoder.h + build/libthqcp.{a,dylib,so}.  "
            "Re-run with -DQSIM_ENABLE_SBNN=OFF to skip this dependency.")
    endif()
else()
    set(QSIM_HAS_SBNN OFF)
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
    src/quantum/noise_mpdo.c
)

set(QSIM_BACKEND_SOURCES
    src/backends/clifford/clifford.c
    src/backends/clifford/pauli_frame.c
)

set(QSIM_ALGORITHMS_SOURCES
    src/algorithms/grover.c
    src/algorithms/grover_optimizer.c
    src/algorithms/quantum_rng.c
    src/algorithms/bell_tests.c
    src/algorithms/qaoa.c
    src/algorithms/qpe.c
    src/algorithms/vqe.c
    src/algorithms/h2_sto3g.c
    src/algorithms/vqe_qng.c
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
    src/algorithms/tensor_network/ca_mps.c
    src/algorithms/tensor_network/ca_mps_var_d.c
    src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.c
    src/algorithms/tensor_network/ca_peps.c
    # Chemistry (Jordan-Wigner, UCCSD, molecular Hamiltonians).
    src/algorithms/chemistry/chemistry.c
    # Topological QC: Fibonacci/Ising anyons, fusion trees, surface/toric codes.
    src/algorithms/topological/topological.c
    # Real-space topology: Bianco-Resta local Chern marker (dense reference).
    src/algorithms/topology_realspace/chern_marker.c
    # Real-space topology: matrix-free Chebyshev-KPM Chern marker.
    src/algorithms/topology_realspace/chern_kpm.c
    # Momentum-space topology: Fukui-Hatsugai-Suzuki link-variable Chern.
    src/algorithms/topology_realspace/chern_fhs.c
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
)

# Visualization: circuit diagrams + Feynman diagrams.  Pure-render
# code, no numerical kernels; downstream consumers (NQS, QGTL) that
# don't need the renderers can configure with
# -DQSIM_BUILD_VISUALIZATION=OFF for a thinner libquantumsim.
if(QSIM_BUILD_VISUALIZATION)
    list(APPEND QSIM_ALGORITHMS_SOURCES
        src/visualization/circuit_diagram.c
        src/visualization/feynman_diagram.c)
endif()

set(QSIM_OPTIMIZATION_SOURCES
    src/optimization/memory_align.c
    src/optimization/simd_dispatch.c
    src/optimization/simd_ops.c
    src/optimization/parallel_ops.c
    src/optimization/accelerate_ops.c
    src/optimization/fusion/fusion.c
    src/optimization/gpu/gpu_backend.c
)
# Backend implementation files are added only when enabled. For Metal we
# also need the Objective-C++ implementation file.
if(QSIM_HAS_METAL OR QSIM_ENABLE_CUDA OR QSIM_ENABLE_OPENCL OR QSIM_ENABLE_VULKAN OR QSIM_ENABLE_CUQUANTUM OR QSIM_ENABLE_WEBGPU)
    if(QSIM_HAS_METAL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/optimization/gpu_metal.mm")
        # Embed the .metal kernel sources into a generated C header so an
        # installed libquantumsim carries its shaders instead of loading them
        # from cwd-relative paths at runtime (which only ever resolves from the
        # source tree, so installed builds and ctest found nothing and the
        # kernels never ran).  MOONLAB_ROOT stays a dev-only override.  The
        # kernel files are configure-dependencies, so editing one re-runs the
        # generation.
        set(_moonlab_metal_kernels "${CMAKE_CURRENT_SOURCE_DIR}/src/optimization/kernels")
        set(_moonlab_metal_hdr "${CMAKE_CURRENT_BINARY_DIR}/include/moonlab_metal_shaders.h")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${_moonlab_metal_kernels}/quantum_kernels.metal"
            "${_moonlab_metal_kernels}/quantum_kernels_batch.metal"
            "${_moonlab_metal_kernels}/tensor_kernels.metal")
        file(READ "${_moonlab_metal_kernels}/quantum_kernels.metal"       _moonlab_metal_q)
        file(READ "${_moonlab_metal_kernels}/quantum_kernels_batch.metal" _moonlab_metal_b)
        file(READ "${_moonlab_metal_kernels}/tensor_kernels.metal"        _moonlab_metal_t)
        file(WRITE "${_moonlab_metal_hdr}"
"/* Auto-generated from the src/optimization/kernels .metal files by CMake. */
#ifndef MOONLAB_METAL_SHADERS_H
#define MOONLAB_METAL_SHADERS_H
static const char MOONLAB_METAL_QUANTUM_KERNELS[] = R\"MOONLABMETAL(
${_moonlab_metal_q}
)MOONLABMETAL\";
static const char MOONLAB_METAL_QUANTUM_KERNELS_BATCH[] = R\"MOONLABMETAL(
${_moonlab_metal_b}
)MOONLABMETAL\";
static const char MOONLAB_METAL_TENSOR_KERNELS[] = R\"MOONLABMETAL(
${_moonlab_metal_t}
)MOONLABMETAL\";
#endif /* MOONLAB_METAL_SHADERS_H */
")
        list(APPEND QSIM_OPTIMIZATION_SOURCES
            src/optimization/gpu_metal.mm
            src/optimization/gpu_metal_svd.c)
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
# gpu_cuda.cu provides the cuda_compute_ctx / cuda_buffer interface
# that tensor.c references whenever HAS_CUDA is defined.  Build it
# under either flag.  gpu_cuquantum.cu is the cuStateVec adapter
# proper -- only under QSIM_ENABLE_CUQUANTUM.
if(QSIM_HAS_CUDA)
    list(APPEND QSIM_GPU_SOURCES src/optimization/gpu/backends/gpu_cuda.cu)
endif()
if(QSIM_ENABLE_CUQUANTUM AND QSIM_HAS_CUDA)
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
    src/utils/moonlab_status.c
    src/utils/token_bucket.c
    src/utils/audit_buffer.c
)

# External-library bridges.  Each translation unit compiles in two modes
# selected at preprocessor time: when MOONLAB_HAS_<LIB>=1 the real
# linkage path is active; otherwise the entry points return
# MOONLAB_STATUS_UNSUPPORTED so callers can treat the dependency as
# optional without conditional compilation at every call site.
set(QSIM_INTEGRATION_SOURCES
    src/integration/libirrep_bridge.c
)

set(QSIM_APPLICATIONS_SOURCES
    src/applications/health_tests.c
    src/applications/bell_test.c
    src/applications/entropy_pool.c
    src/applications/hardware_entropy.c
    src/applications/qrng.c
    src/applications/qrng_di.c
    src/applications/moonlab_qrng_export.c
    src/applications/moonlab_export_lean.c
    src/applications/moonlab_tdvp_export.c
    src/applications/moonlab_vqe_export.c
    src/applications/moonlab_qgtl_backend.c
    src/applications/decoder_bench.c
    src/applications/mwpm_exact.c
    # Vendor-noise emulator backends (IBM Falcon / Rigetti Aspen / IonQ Forte)
    # consume QGTL circuits via the scheduler's backend plug-in surface.
    src/applications/vendor_noise_backend.c
    src/distributed/scheduler.c
    src/applications/nist_sp800_22.c
    src/applications/quantum_volume.c
    # HEP / lattice gauge theory bench primitives.  The 1+1D Z2 LGT Pauli-sum
    # builder powers the var-D x gauge-constraint-absorption demonstration in
    # examples/hep/ and tests/unit/test_z2_lgt_pauli_sum.c.
    src/applications/hep/lattice_z2_1d.c
)
if(QSIM_ENABLE_CONTROL_PLANE)
    list(APPEND QSIM_APPLICATIONS_SOURCES
        # Control plane (since v0.8.7): POSIX-socket TCP transport for the
        # v0.8.3 moonlab-circuit v1 wire format.  v0.8.17 OpenSSL TLS path
        # is gated via #ifdef MOONLAB_HAVE_TLS inside the same TU.
        src/control/control_plane.c
    )
elseif(QSIM_ENABLE_TLS)
    message(STATUS "Control plane: disabled; TLS transport not built")
endif()

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
    ${QSIM_INTEGRATION_SOURCES}
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

# Standalone helper for safe ARM hardware-RNG probing.
add_executable(hw_rng_probe tools/hw_rng_probe.c)

if(QSIM_ENABLE_CONTROL_PLANE)
    # Production control-plane daemon (since v0.10.0).  Bound + run via
    # deploy/docker/docker-compose.yml.
    add_executable(moonlab-control-server tools/moonlab-control-server.c)
endif()

# ==============================================================================
# Library Target
# ==============================================================================

# Create library
if(QSIM_BUILD_SHARED)
    add_library(quantumsim SHARED ${QSIM_ALL_SOURCES})
    # Native SDK installs retain conventional ABI-versioned symlinks. Python
    # wheels need one canonical library file: wheel archives cannot represent
    # those symlinks portably and otherwise expand all three names into full
    # duplicate copies.
    if(NOT QSIM_PYTHON_WHEEL)
        set_target_properties(quantumsim PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
        )
    endif()
    if(QSIM_PLATFORM_WINDOWS)
        set_target_properties(quantumsim PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
elseif(QSIM_BUILD_STATIC)
    add_library(quantumsim STATIC ${QSIM_ALL_SOURCES})
else()
    add_library(quantumsim ${QSIM_ALL_SOURCES})
endif()

add_dependencies(quantumsim hw_rng_probe)
if(QSIM_BUILD_SHARED)
    target_compile_definitions(quantumsim
        PRIVATE MOONLAB_BUILDING_SHARED=1)
endif()
target_compile_definitions(quantumsim PRIVATE
    MOONLAB_HW_RNG_PROBE_PATH=\"$<TARGET_FILE:hw_rng_probe>\"
    MOONLAB_PYMATCHING_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/src/applications/pymatching_bridge.py\"
)

# Test-only knobs that force the in-tree QR / SVD fallback codepaths in
# tensor.c even when LAPACK is available.  Used by a dedicated CI tier
# (tier-fallback) so WASM and Windows clang-cl regressions show up on
# every PR rather than only after a release ships.
if(QSIM_FORCE_QR_FALLBACK)
    target_compile_definitions(quantumsim PRIVATE QSIM_FORCE_QR_FALLBACK=1)
    message(STATUS "Forcing tensor_qr through the no-LAPACK Householder fallback")
endif()
if(QSIM_FORCE_SVD_FALLBACK)
    target_compile_definitions(quantumsim PRIVATE QSIM_FORCE_SVD_FALLBACK=1)
    message(STATUS "Forcing tensor_svd through the no-LAPACK Jacobi fallback")
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

# libirrep bridge -- activate the real linkage path when libirrep was
# located by the detection block earlier.  The bridge translation unit
# itself stays in QSIM_INTEGRATION_SOURCES (compiled unconditionally so
# the API symbols always exist); MOONLAB_HAS_LIBIRREP=1 selects the
# implementation that actually calls into libirrep.
if(QSIM_HAS_SBNN)
    target_compile_definitions(quantumsim PRIVATE MOONLAB_HAS_SBNN=1)
    target_include_directories(quantumsim PRIVATE "${QSIM_SBNN_ROOT}/include")
    # SbNN's libthqcp (the public coupling SDK) doesn't include the
    # qec_decoder + toric_code + kitaev_model object files, so we
    # vendor their TUs directly when QSIM_ENABLE_SBNN=ON.  All three
    # are stdlib-only -- no transitive deps to chase.
    target_sources(quantumsim PRIVATE
        "${QSIM_SBNN_ROOT}/src/qec_decoder/qec_decoder.c"
        "${QSIM_SBNN_ROOT}/src/toric_code.c"
        "${QSIM_SBNN_ROOT}/src/kitaev_model.c"
    )
    message(STATUS "SbNN decoder slot: ON (vendored qec_decoder + toric_code from ${QSIM_SBNN_ROOT})")
endif()

if(QSIM_HAS_LIBIRREP)
    target_compile_definitions(quantumsim PRIVATE MOONLAB_HAS_LIBIRREP=1)
    target_link_libraries(quantumsim PRIVATE ${QSIM_LIBIRREP_TARGET})
    # Bake the libirrep library path into libquantumsim's runtime-rpath
    # so test executables don't need DYLD_LIBRARY_PATH at run time.
    # Only adds an entry when QSIM_LIBIRREP_ROOT pointed at a source
    # tree (the find_package + pkg-config paths handle this through
    # the imported target).
    if(QSIM_LIBIRREP_ROOT)
        set(_libirrep_libdir "${QSIM_LIBIRREP_ROOT}/build/lib")
        if(EXISTS "${_libirrep_libdir}")
            set_property(TARGET quantumsim APPEND PROPERTY
                BUILD_RPATH "${_libirrep_libdir}")
        endif()
    endif()
    message(STATUS "libirrep bridge: ON (target=${QSIM_LIBIRREP_TARGET})")
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

# Windows-only system-header shims for Win32 ports of POSIX names
# (clock_gettime, usleep, pthread_mutex, complex math).  Lives under
# src/compat/ so it does NOT shadow system headers on Linux/macOS,
# which previously produced four #include_next warnings on every
# compile.  On Windows we splice src/compat/ into the include path
# AHEAD of the system path so the shims resolve as <time.h> etc.
if(QSIM_PLATFORM_WINDOWS)
    # Directory-scoped: dlsym-style tests deliberately do not link
    # quantumsim, so a target-scoped PUBLIC include leaves them on UCRT
    # <complex.h> (_Dcomplex prototypes) while their code uses C99
    # _Complex. Every build-tree target needs the shims.
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src/compat)
    target_include_directories(quantumsim BEFORE PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/compat>)
endif()

# Compile options.  C/C++ warning flags must not leak into NVCC: NVCC uses
# `--Werror=<kind>` and interprets GCC's following `-Wno-error=...` token as
# the argument to its own `-Werror`, aborting before kernel compilation.
# Give CUDA both its native all-warnings gate and an explicit host-compiler
# warning gate instead.
foreach(_qsim_warning_flag IN LISTS QSIM_WARNING_FLAGS)
    target_compile_options(quantumsim PRIVATE
        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_qsim_warning_flag}>")
endforeach()
if(QSIM_ENABLE_CUDA AND QSIM_WERROR)
    target_compile_options(quantumsim PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>")
    if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
        # NVCC emits non-standard GCC-style #line markers in its generated
        # host stubs. Enabling host -Wpedantic would turn NVCC's own generated
        # text into an error, so keep every actionable -Wall/-Wextra warning
        # strict without requesting that generated-code-only category.
        target_compile_options(quantumsim PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>")
    endif()
endif()
target_compile_options(quantumsim PRIVATE
    ${QSIM_OPTIMIZE_FLAGS}
    ${QSIM_SANITIZER_FLAGS}
)

# Visibility tightening: if QSIM_HIDDEN_VISIBILITY=ON, hide every
# symbol that is not explicitly tagged with MOONLAB_API.  This is the
# v0.3 forward-compat surface; v0.2.x defaults OFF because the
# bindings still consume internal-only entry points.
if(QSIM_HIDDEN_VISIBILITY AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    target_compile_options(quantumsim PRIVATE
        -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

# Sanitizer runtime needs to be linked into the shared library too.
if(QSIM_ENABLE_SANITIZERS AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    target_link_options(quantumsim PRIVATE ${QSIM_SANITIZER_FLAGS})
    add_compile_options(${QSIM_SANITIZER_FLAGS})
    add_link_options(${QSIM_SANITIZER_FLAGS})
endif()

# gcov-compatible coverage instrumentation; both gcc and clang emit
# .gcno/.gcda files that lcov + genhtml can render.  We disable LTO
# and force -O0 to keep line-level coverage useful (the optimiser
# folds away basic blocks otherwise).
if(QSIM_ENABLE_COVERAGE AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
    message(STATUS "Coverage: gcov instrumentation enabled (-O0)")
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

if(QSIM_HAS_TLS)
    target_link_libraries(quantumsim PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(quantumsim PRIVATE MOONLAB_HAVE_TLS=1)
endif()

if(QSIM_ENABLE_CONTROL_PLANE)
    # Wire the production control-plane daemon binary to quantumsim.
    target_link_libraries(moonlab-control-server PRIVATE quantumsim ${MATH_LIBRARY} Threads::Threads)
    install(TARGETS moonlab-control-server RUNTIME DESTINATION bin)
endif()

# ==============================================================================
# Platform-Specific Configuration
# ==============================================================================

# macOS configuration
if(QSIM_PLATFORM_MACOS)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_MACOS=1)

    # weak_import marks the reference optional for dyld, but ld still needs
    # each symbol explicitly permitted to remain unresolved when the optional
    # backend implementation is not part of this CPU-only build.  -U emits a
    # dynamic-lookup import for only these guarded hooks; definitions in an
    # enabled backend continue to resolve normally.
    set(QSIM_MACHO_OPTIONAL_BACKEND_SYMBOLS
        _qsim_gpu_route_hadamard
        _qsim_gpu_route_pauli_x
        _qsim_gpu_route_cnot
        _qsim_gpu_route_apply_1q_matrix
        _qsim_gpu_route_apply_2q_matrix
        _qsim_gpu_route_mcx
        _qsim_gpu_route_mcz
        _qsim_gpu_route_fredkin
        _moonlab_cuda_state_free
        _moonlab_cuda_apply_1q
        _moonlab_cuda_apply_cnot
        _moonlab_cuda_state_create
        _moonlab_cuda_state_copy_to_host
        _moonlab_cuda_state_copy_from_host
        _moonlab_cuda_runtime_probe_discrete
        _metal_compute_init
        _metal_compute_free
        _metal_buffer_create
        _metal_buffer_contents
        _metal_buffer_free
        _metal_mps_apply_gate_2q
        _metal_mps_expectation_z
    )
    foreach(QSIM_OPTIONAL_BACKEND_SYMBOL IN LISTS QSIM_MACHO_OPTIONAL_BACKEND_SYMBOLS)
        target_link_options(quantumsim PRIVATE "-Wl,-U,${QSIM_OPTIONAL_BACKEND_SYMBOL}")
    endforeach()
    unset(QSIM_OPTIONAL_BACKEND_SYMBOL)
    unset(QSIM_MACHO_OPTIONAL_BACKEND_SYMBOLS)

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
        target_compile_definitions(quantumsim PRIVATE HAS_METAL=1 QSIM_HAS_METAL=1)
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
    # find_library can intermittently miss installed libraries in
    # emulated (QEMU buildx) containers even when the packages are
    # present and the linker resolves them fine — the same failure
    # mode as MATH_LIBRARY above. Fall back per-library to the plain
    # linker names; a genuinely missing provider still fails the
    # library link loudly instead of shipping an underlinked .so.
    if(NOT LAPACKE_LIB)
        set(LAPACKE_LIB lapacke)
    endif()
    if(NOT LAPACK_LIB)
        set(LAPACK_LIB lapack)
    endif()
    if(NOT BLAS_LIB)
        set(BLAS_LIB blas)
    endif()
    target_link_libraries(quantumsim PRIVATE
        ${LAPACKE_LIB} ${LAPACK_LIB} ${BLAS_LIB})
    message(STATUS "LAPACK: LAPACKE=${LAPACKE_LIB} "
                   "LAPACK=${LAPACK_LIB} BLAS=${BLAS_LIB}")
endif()

# Windows configuration
if(QSIM_PLATFORM_WINDOWS)
    target_compile_definitions(quantumsim PRIVATE PLATFORM_WINDOWS=1)
    target_link_libraries(quantumsim PRIVATE bcrypt)

    # Windows builds use the in-tree Householder QR fallback in
    # src/algorithms/tensor_network/tensor.c (no LAPACK link needed).  The
    # fallback is numerically identical to LAPACK Householder QR -- verified
    # against the LAPACKE path on macOS to machine precision (QRA <= 1e-15).
    # Users with OpenBLAS installed can opt in by defining QSIM_HAS_OPENBLAS=1
    # and adding the lapacke/cblas include paths and link library themselves.

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
            # The builtins archive satisfies implementation details inside the
            # DLL. It must not leak as an absolute build-machine path through
            # quantumsim::quantumsim in the installed CMake export.
            target_link_libraries(quantumsim PRIVATE "${QSIM_CLANG_BUILTINS}")
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

# simd_ops.c, entropy.c, hardware_entropy.c on x86_64 use intrinsics
# (SSE3, RDRAND, RDSEED) inside code paths that aren't reliably gated
# on the right __SSE3__/__RDRND__/__RDSEED__ feature macros.  Release
# builds get -march=native which upgrades the baseline transparently;
# Debug / coverage builds use the default x86-64 baseline which
# defines only __SSE2__, so the intrinsics fail to declare.  Force
# the relevant -m flags per-file on x86_64 so these paths compile
# regardless of build type.
if(QSIM_ARCH_X86_64 AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    set_source_files_properties(
        src/optimization/simd_ops.c
        PROPERTIES COMPILE_FLAGS "-msse3"
    )
    set_source_files_properties(
        src/utils/entropy.c
        src/applications/hardware_entropy.c
        PROPERTIES COMPILE_FLAGS "-mrdrnd -mrdseed"
    )
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
        # install_name_tool rewrites a Mach-O dylib's load commands; it fails
        # on a static archive (.a), so only run it for a shared build.
        if(QSIM_BUILD_SHARED)
            add_custom_command(TARGET quantumsim POST_BUILD
                COMMAND install_name_tool -change "${QSIM_OPENMP_LIBRARIES}"
                        "@rpath/libomp.dylib" "$<TARGET_FILE:quantumsim>"
                VERBATIM
            )
        endif()
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
    # config.c / qsim_detect_backend test the QSIM_HAS_* names; keep the
    # unprefixed HAS_* aliases for backward compatibility with older TUs.
    target_compile_definitions(quantumsim PRIVATE HAS_OPENCL=1 QSIM_HAS_OPENCL=1)
endif()

if(QSIM_HAS_VULKAN)
    target_link_libraries(quantumsim PRIVATE Vulkan::Vulkan)
    target_compile_definitions(quantumsim PRIVATE HAS_VULKAN=1 QSIM_HAS_VULKAN=1)
endif()

if(QSIM_HAS_CUDA)
    target_link_libraries(quantumsim PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(quantumsim PRIVATE HAS_CUDA=1 QSIM_HAS_CUDA=1)
    target_include_directories(quantumsim PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # Set CUDA architectures.  72 = Xavier (Volta integrated), 87 = Orin
    # (Ampere integrated).  Other lines: 70 V100, 75 Turing, 80 A100,
    # 86 Ada, 89 RTX 40-series, 90 H100.  Override via
    # -DCMAKE_CUDA_ARCHITECTURES=NN to target a specific device.
    #
    # The "-real" suffix tells cmake to emit only SASS for that arch
    # (no embedded PTX, no LTO-IR).  We need this because the
    # bracket-notation cmake otherwise generates (code=[compute_X,sm_X])
    # triggers `-dlto` in the device-link step, which fails on
    # jetpack-nixos with "nvlink fatal: elfLink linker library load
    # error" because the LTO plugin isn't resolvable from nvlink's
    # dlopen path.  We forfeit forward-JIT compatibility (PTX
    # embedded) for a working build; this is the right trade for
    # in-tree backend kernels where we control the deployment target.
    # Architecture list -- only fires when caller didn't pass one.
    # The bare numbers here would be auto-promoted by the normalization
    # block above when CMAKE_CUDA_ARCHITECTURES is already set; this
    # default path provides "-real" directly.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 70-real 72-real 75-real 80-real 86-real 87-real 89-real 90-real)
    endif()

    # Compile the moonlab CUDA backend.  cuda_tegra_probe.c is plain
    # C (no CUDA dependency in the .c itself; the .h is C-visible),
    # but we still gate it on QSIM_HAS_CUDA because nothing else
    # needs the runtime probe yet.
    target_sources(quantumsim PRIVATE
        src/backends/cuda_tegra_probe.c
        src/backends/cuda_statevec.cu
        src/quantum/state_gpu.cu
    )

    # Separable-compilation policy.
    #
    # CMake 4.x emits `-dlto` (device LTO) at the device-link step
    # whenever CUDA_SEPARABLE_COMPILATION=ON, with no documented
    # off-switch -- not via INTERPROCEDURAL_OPTIMIZATION, not via the
    # "-real" arch suffix.  On jetpack-nixos R35 (CUDA 11.4) the nvlink
    # binary can't dlopen the LTO IR plugin from its nix-store path,
    # so any separable-compilation build fails with "nvlink fatal:
    # elfLink linker library load error".
    #
    # Our backend doesn't require cross-TU device-symbol resolution:
    # each .cu file defines its own __global__ kernels, and the only
    # __constant__ memory block (k_apply_2q_M in cuda_statevec.cu) is
    # referenced only from kernels in the same TU.  So we don't
    # actually need separable compilation, and turning it OFF is the
    # correct platform-aware choice -- not a workaround.
    #
    # If a future kernel needs to share device symbols across TUs,
    # the conditions to turn this back ON are:
    #   - on jetpack-nixos: nvlink's LTO plugin path becomes resolvable
    #     (jetpack-nixos contributes a wrapper, or CUDA 12.6+ removes
    #     the LTO dep at device link)
    #   - on stock linux + CUDA 12+: should already work
    set_target_properties(quantumsim PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )

    # jetpack-nixos rpath wiring (include path was set pre-enable_language).
    # The L4T driver stub libcuda.so.1 lives in nixpkgs's GL-driver compat
    # path.  Add it to BUILD_RPATH so binaries linked against quantumsim
    # find the driver at runtime without LD_LIBRARY_PATH boilerplate.
    if(QSIM_PLATFORM_JETPACK_NIXOS AND EXISTS "/run/opengl-driver/lib/libcuda.so.1")
        message(STATUS "CUDA: jetpack-nixos libcuda stub at /run/opengl-driver/lib")
        set_property(TARGET quantumsim APPEND PROPERTY
            BUILD_RPATH    "/run/opengl-driver/lib")
        set_property(TARGET quantumsim APPEND PROPERTY
            INSTALL_RPATH  "/run/opengl-driver/lib")
    endif()
else()
    # CUDA off: the CUDA translation units that would define the GPU-routing
    # entry points are not compiled, so provide the single strong CPU-only
    # definition here. Must be exactly one TU so it never collides with the
    # CUDA definitions; plain (non-weak) symbols so clang-cl/COFF links it.
    target_sources(quantumsim PRIVATE src/quantum/gpu_route_cpu_fallback.c)
endif()

if(QSIM_HAS_CUQUANTUM)
    target_include_directories(quantumsim PRIVATE ${CUQUANTUM_INCLUDE_DIR})
    target_link_libraries(quantumsim PRIVATE ${CUSTATEVEC_LIBRARY})
    target_compile_definitions(quantumsim PRIVATE
        HAS_CUQUANTUM=1 HAS_CUSTATEVEC=1
        QSIM_HAS_CUQUANTUM=1 QSIM_HAS_CUSTATEVEC=1)
    if(QSIM_HAS_CUTENSORNET)
        target_link_libraries(quantumsim PRIVATE ${CUTENSORNET_LIBRARY})
        target_compile_definitions(quantumsim PRIVATE HAS_CUTENSORNET=1 QSIM_HAS_CUTENSORNET=1)
    endif()
endif()

if(QSIM_HAS_MPI)
    target_link_libraries(quantumsim PRIVATE MPI::MPI_C)
    target_compile_definitions(quantumsim PRIVATE HAS_MPI=1 QSIM_HAS_MPI=1)
endif()

# ==============================================================================
# Generate moonlab_features.h with the resolved feature flags.
# Consumers include this header to branch on what got compiled in,
# avoiding link-time unresolved-symbol errors when calling into a
# subsystem that was disabled at build time.
# ==============================================================================
# For each subsystem, mark MOONLAB_HAS_<NAME> = 1 iff it was actually
# detected and compiled in (QSIM_HAS_* set by find_package or platform
# probe; QSIM_SIMD_* set by the per-arch dispatcher).  Do not consult
# QSIM_ENABLE_* — that's the user-supplied request, not the resolved
# state, and would mark NEON=1 on x86 hosts.
foreach(feat MPI METAL CUDA OPENCL VULKAN WEBGPU CUQUANTUM OPENMP)
    if(QSIM_HAS_${feat})
        set(MOONLAB_HAS_${feat} 1)
    else()
        set(MOONLAB_HAS_${feat} 0)
    endif()
endforeach()
foreach(feat AVX512 AVX2 NEON SVE)
    if(QSIM_SIMD_${feat})
        set(MOONLAB_HAS_${feat} 1)
    else()
        set(MOONLAB_HAS_${feat} 0)
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/applications/moonlab_features.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/moonlab_features.h"
    @ONLY)
target_include_directories(quantumsim PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

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
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
    # Statistical + crypto-adversarial batteries (self-contained subdir).
    add_subdirectory(tests/statistical)
endif()

# ==============================================================================
# Example Targets
# ==============================================================================

if(QSIM_BUILD_EXAMPLES)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/examples.cmake)
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

    # Decoder-zoo shoot-out (since v0.7.8) -- runs all available
    # decoder slots at varying physical error rates, emits JSON.
    add_executable(decoder_shootout benchmarks/decoder_shootout.c)
    target_link_libraries(decoder_shootout PRIVATE quantumsim)

    if(QSIM_ENABLE_CONTROL_PLANE)
        # Control-plane load test (since v1.0.3) -- N concurrent clients
        # against a running moonlab-control-server; throughput + P50/P90/P99
        # latency.  Validates the runbook capacity numbers and the
        # FLEET_DEPLOYMENT.md per-replica budgets.
        add_executable(control_plane_loadtest benchmarks/control_plane_loadtest.c)
        target_link_libraries(control_plane_loadtest
            PRIVATE quantumsim ${MATH_LIBRARY} Threads::Threads)
    endif()
    if(QSIM_HAS_OPENMP)
        qsim_target_link_openmp(grover_parallel_benchmark)
        if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
            target_compile_options(grover_parallel_benchmark PRIVATE -Xpreprocessor -fopenmp)
        endif()
        target_compile_definitions(grover_parallel_benchmark PRIVATE HAS_OPENMP=1)
    endif()

    # v1.0 head-to-head comparison harnesses (since v1.0).  Each emits a
    # JSON file with the moonlab side of one published comparison; the
    # competitor side runs separately -- see docs/benchmarks/v1_comparison.md.
    add_executable(ca_mps_kagome_bench
        benchmarks/v1_comparison/ca_mps_kagome_bench.c)
    target_link_libraries(ca_mps_kagome_bench PRIVATE quantumsim ${MATH_LIBRARY})

    add_executable(surface_threshold_bench
        benchmarks/v1_comparison/surface_threshold_bench.c)
    target_link_libraries(surface_threshold_bench PRIVATE quantumsim ${MATH_LIBRARY})

    add_executable(sv_random_bench
        benchmarks/v1_comparison/sv_random_bench.c)
    target_link_libraries(sv_random_bench PRIVATE quantumsim ${MATH_LIBRARY})

    message(STATUS "Benchmarks: Enabled")
endif()

# ==============================================================================
# Installation
# ==============================================================================

include(GNUInstallDirs)

# Platform wheels carry their native library inside the Python package and
# intentionally omit the C/CMake SDK payload. Normal native installs retain
# the complete export/header/package-config contract below.
if(QSIM_PYTHON_WHEEL)
    install(TARGETS quantumsim
        EXPORT quantumsim-targets
        LIBRARY DESTINATION moonlab/.libs COMPONENT python-wheel
        RUNTIME DESTINATION moonlab/.libs COMPONENT python-wheel
    )
else()
    install(TARGETS quantumsim
        EXPORT quantumsim-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# Install headers
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantumsim
    COMPONENT native-sdk
    FILES_MATCHING PATTERN "*.h"
)

# Stable downstream C ABI convenience include path.  The full source-shaped
# header tree remains under include/quantumsim, while README and FFI consumers
# can include <moonlab/moonlab_export.h>.
install(FILES
    src/applications/moonlab_api.h
    src/applications/moonlab_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/moonlab
    COMPONENT native-sdk
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/moonlab_features.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT native-sdk
)

# Install CMake config
install(EXPORT quantumsim-targets
    FILE quantumsim-targets.cmake
    NAMESPACE quantumsim::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
    COMPONENT native-sdk
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
    COMPONENT native-sdk
)

# pkg-config file.  The template uses pkg-config's ${pcfiledir} so release
# tarballs remain relocatable after extraction.  Compute the relative walk from
# ${libdir}/pkgconfig back to the install prefix; Debian multiarch libdirs such
# as lib/x86_64-linux-gnu need one more ".." than plain lib/pkgconfig.
set(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR "")
set(_qsim_pc_dir "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
string(REPLACE "/" ";" _qsim_pc_dir_parts "${_qsim_pc_dir}")
foreach(_qsim_pc_dir_part IN LISTS _qsim_pc_dir_parts)
    if(_qsim_pc_dir_part STREQUAL "")
        continue()
    endif()
    if(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR STREQUAL "")
        set(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR "..")
    else()
        string(APPEND QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR "/..")
    endif()
endforeach()
if(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR STREQUAL "")
    set(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR ".")
endif()
install(CODE "
    set(CMAKE_INSTALL_PREFIX   \"\${CMAKE_INSTALL_PREFIX}\")
    set(CMAKE_INSTALL_LIBDIR   \"${CMAKE_INSTALL_LIBDIR}\")
    set(CMAKE_INSTALL_INCLUDEDIR \"${CMAKE_INSTALL_INCLUDEDIR}\")
    set(PROJECT_DESCRIPTION     \"${PROJECT_DESCRIPTION}\")
    set(PROJECT_VERSION         \"${PROJECT_VERSION}\")
    set(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR \"${QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR}\")
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
    ${CMAKE_CURRENT_BINARY_DIR}/generated/moonlab_build_info.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT native-sdk
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc.install
    RENAME quantumsim.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT native-sdk
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
