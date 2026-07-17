# tests/differential/CMakeLists.txt
#
# Self-contained build for the cross-backend differential C driver.
#
# DUAL MODE:
#   1. Integrated -- the root CMakeLists adds `add_subdirectory(tests/differential)`
#      (the integrator wires this in separately; do NOT edit the root file). Here
#      the `quantumsim` target already exists, so we just link it and inherit its
#      include directories / OpenMP / compile definitions.
#   2. Standalone -- `cmake -S tests/differential -B <dir>` configured directly by
#      scripts/run_cross_diff.sh against a prebuilt libquantumsim. We locate the
#      library (MOONLAB_LIB_DIR, then the repo build trees) and bake an rpath so
#      the driver finds it at runtime.
#
# The driver includes library headers via source-relative paths
# (`../../src/...`), so no extra include directories are required in either mode.

cmake_minimum_required(VERSION 3.16)

if(NOT TARGET quantumsim)
    project(moonlab_differential C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)

    set(_moonlab_hints
        "$ENV{MOONLAB_LIB_DIR}"
        "${MOONLAB_LIB_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../build"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../build_release")

    find_library(MOONLAB_QUANTUMSIM
        NAMES quantumsim libquantumsim
        HINTS ${_moonlab_hints}
        NO_DEFAULT_PATH)
    if(NOT MOONLAB_QUANTUMSIM)
        find_library(MOONLAB_QUANTUMSIM NAMES quantumsim libquantumsim)
    endif()
    if(NOT MOONLAB_QUANTUMSIM)
        message(FATAL_ERROR
            "libquantumsim not found. Build it first "
            "(cmake -S . -B build && cmake --build build) or set MOONLAB_LIB_DIR.")
    endif()
    message(STATUS "differential: linking prebuilt ${MOONLAB_QUANTUMSIM}")

    add_executable(diff_backends diff_backends.c)
    target_link_libraries(diff_backends PRIVATE ${MOONLAB_QUANTUMSIM})

    get_filename_component(_moonlab_libdir "${MOONLAB_QUANTUMSIM}" DIRECTORY)
    set_target_properties(diff_backends PROPERTIES
        BUILD_RPATH "${_moonlab_libdir}"
        INSTALL_RPATH "${_moonlab_libdir}")
else()
    message(STATUS "differential: linking in-tree quantumsim target")
    add_executable(diff_backends diff_backends.c)
    target_link_libraries(diff_backends PRIVATE quantumsim)
endif()

# Math library (complex.h / cexp etc.) where it is a separate archive.
find_library(DIFF_M_LIB m)
if(DIFF_M_LIB)
    target_link_libraries(diff_backends PRIVATE ${DIFF_M_LIB})
endif()

# The self-test proves the reference oracle catches a corrupted probability;
# it needs no corpus and no external toolchain, so it is always registerable.
enable_testing()
add_test(NAME differential_selftest COMMAND diff_backends --selftest)
set_tests_properties(differential_selftest PROPERTIES LABELS "differential")
