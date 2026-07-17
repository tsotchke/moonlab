# tests/numerical/CMakeLists.txt
#
# Self-contained build for the numerical-edge / uninitialised-memory
# bug-hunting harnesses.
#
# DUAL MODE (mirrors tests/differential):
#   1. Integrated -- the root CMakeLists may add_subdirectory(tests/numerical);
#      target `quantumsim` already exists, so lib-linked harnesses just link it.
#   2. Standalone -- `cmake -S tests/numerical -B <dir>` against a prebuilt
#      libquantumsim (MOONLAB_LIB_DIR / the repo build trees). scripts/run_numerical.sh
#      drives this mode.
#
# Harnesses come in two flavours:
#   * lib-linked (link libquantumsim, exercise the SHIPPING paths):
#       t_rotations, t_qft, t_measure, t_dmrg, t_eigen_lapack, t_svd_lapack
#   * forced-fallback (compile the library source DIRECTLY with the no-LAPACK
#     paths forced, do NOT link the lib -- so the one-sided Jacobi SVD /
#     Householder QR / complex-Hermitian Jacobi eigen fallbacks are exercised
#     even on a LAPACK platform):
#       t_eigen_jacobi   <- src/utils/matrix_math.c  (-U__APPLE__)
#       t_svd_fallback   <- src/algorithms/tensor_network/tensor.c
#                           (-DQSIM_FORCE_SVD_FALLBACK -DQSIM_FORCE_QR_FALLBACK)
#                           + gpu_stubs.c
#
# This file compiles library sources but never edits them.

cmake_minimum_required(VERSION 3.16)

if(NOT TARGET quantumsim)
    project(moonlab_numerical C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

set(NUM_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src")

# --------------------------------------------------------------------------
# Locate libquantumsim for the lib-linked harnesses (standalone mode).
# --------------------------------------------------------------------------
set(NUM_LINK_LIB "")
if(TARGET quantumsim)
    set(NUM_LINK_LIB quantumsim)
    message(STATUS "numerical: linking in-tree quantumsim target")
else()
    set(_num_hints
        "$ENV{MOONLAB_LIB_DIR}"
        "${MOONLAB_LIB_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../build"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../build_release")
    find_library(NUM_QUANTUMSIM NAMES quantumsim libquantumsim
        HINTS ${_num_hints} NO_DEFAULT_PATH)
    if(NOT NUM_QUANTUMSIM)
        find_library(NUM_QUANTUMSIM NAMES quantumsim libquantumsim)
    endif()
    if(NOT NUM_QUANTUMSIM)
        message(FATAL_ERROR
            "libquantumsim not found. Build it first "
            "(cmake -S . -B build && cmake --build build) or set MOONLAB_LIB_DIR.")
    endif()
    message(STATUS "numerical: linking prebuilt ${NUM_QUANTUMSIM}")
    set(NUM_LINK_LIB ${NUM_QUANTUMSIM})
    get_filename_component(NUM_LIBDIR "${NUM_QUANTUMSIM}" DIRECTORY)
endif()

find_library(NUM_M_LIB m)

# helper: a lib-linked harness
function(num_lib_harness name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE ${NUM_LINK_LIB})
    if(NUM_M_LIB)
        target_link_libraries(${name} PRIVATE ${NUM_M_LIB})
    endif()
    if(NUM_LIBDIR)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${NUM_LIBDIR}" INSTALL_RPATH "${NUM_LIBDIR}")
    endif()
endfunction()

num_lib_harness(t_rotations    t_rotations.c)
num_lib_harness(t_qft          t_qft.c)
num_lib_harness(t_measure      t_measure.c)
num_lib_harness(t_dmrg         t_dmrg.c)
num_lib_harness(t_eigen_lapack t_eigen.c)
num_lib_harness(t_svd_lapack   t_svd.c)

# --------------------------------------------------------------------------
# Forced-fallback: complex-Hermitian Jacobi eigen (matrix_math.c, no LAPACK).
# -U__APPLE__ makes MOONLAB_HAVE_LAPACK_ZHEEV undefined -> Jacobi path.
# --------------------------------------------------------------------------
add_library(num_eigen_jacobi_obj OBJECT "${NUM_SRC_ROOT}/utils/matrix_math.c")
target_include_directories(num_eigen_jacobi_obj PRIVATE
    "${NUM_SRC_ROOT}" "${NUM_SRC_ROOT}/utils")
target_compile_options(num_eigen_jacobi_obj PRIVATE -U__APPLE__)

add_executable(t_eigen_jacobi t_eigen.c $<TARGET_OBJECTS:num_eigen_jacobi_obj>)
target_compile_definitions(t_eigen_jacobi PRIVATE EIGEN_JACOBI_BUILD)
if(NUM_M_LIB)
    target_link_libraries(t_eigen_jacobi PRIVATE ${NUM_M_LIB})
endif()

# --------------------------------------------------------------------------
# Forced-fallback: one-sided Jacobi SVD + Householder QR (tensor.c, no LAPACK
# dispatch). GPU stubs report "unavailable" so the CPU paths run.
# --------------------------------------------------------------------------
add_library(num_svd_fallback_obj OBJECT
    "${NUM_SRC_ROOT}/algorithms/tensor_network/tensor.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_stubs.c")
target_include_directories(num_svd_fallback_obj PRIVATE
    "${NUM_SRC_ROOT}"
    "${NUM_SRC_ROOT}/algorithms/tensor_network"
    "${NUM_SRC_ROOT}/optimization"
    "${NUM_SRC_ROOT}/optimization/gpu/backends")
target_compile_definitions(num_svd_fallback_obj PRIVATE
    QSIM_FORCE_SVD_FALLBACK QSIM_FORCE_QR_FALLBACK)
# Silence the Accelerate CLAPACK deprecation noise from the compiled-in source.
target_compile_options(num_svd_fallback_obj PRIVATE -Wno-deprecated-declarations)

add_executable(t_svd_fallback t_svd.c $<TARGET_OBJECTS:num_svd_fallback_obj>)
target_compile_definitions(t_svd_fallback PRIVATE SVD_FALLBACK_BUILD)
if(NUM_M_LIB)
    target_link_libraries(t_svd_fallback PRIVATE ${NUM_M_LIB})
endif()
# tensor.c still #includes Accelerate on Apple; link it to resolve any residual
# LAPACK/BLAS references compiled in outside the forced-fallback code paths.
if(APPLE)
    target_link_libraries(t_svd_fallback PRIVATE "-framework Accelerate")
endif()

# --------------------------------------------------------------------------
# CTest registration (labels: numerical).
# --------------------------------------------------------------------------
enable_testing()
foreach(t t_rotations t_qft t_measure t_dmrg
          t_eigen_lapack t_eigen_jacobi t_svd_lapack t_svd_fallback)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES LABELS "numerical")
endforeach()
