# tests/scaling/CMakeLists.txt
#
# Self-contained build for the Moonlab SCALING / large-n differential drivers.
#
# DUAL MODE (mirrors tests/differential):
#   1. Integrated -- if a `quantumsim` target already exists (root CMake adds
#      this subdir), link it in-tree and inherit its includes / OpenMP / defs.
#   2. Standalone -- `cmake -S tests/scaling -B <dir>` configured directly by
#      scripts/run_scaling.sh against a prebuilt libquantumsim. We locate the
#      library (MOONLAB_LIB_DIR, then the repo build trees) and bake an rpath.
#
# The drivers include library headers via source-relative paths (`../../src/...`),
# so no extra include directories are needed in either mode.

cmake_minimum_required(VERSION 3.16)

set(_scaling_drivers scaling_diff scaling_stab scaling_var_d scaling_dmrg_tdvp)

if(NOT TARGET quantumsim)
    project(moonlab_scaling C)
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
    message(STATUS "scaling: linking prebuilt ${MOONLAB_QUANTUMSIM}")
    get_filename_component(_moonlab_libdir "${MOONLAB_QUANTUMSIM}" DIRECTORY)

    foreach(drv ${_scaling_drivers})
        add_executable(${drv} ${drv}.c)
        target_link_libraries(${drv} PRIVATE ${MOONLAB_QUANTUMSIM})
        set_target_properties(${drv} PROPERTIES
            BUILD_RPATH "${_moonlab_libdir}"
            INSTALL_RPATH "${_moonlab_libdir}")
    endforeach()
else()
    message(STATUS "scaling: linking in-tree quantumsim target")
    foreach(drv ${_scaling_drivers})
        add_executable(${drv} ${drv}.c)
        target_link_libraries(${drv} PRIVATE quantumsim)
    endforeach()
endif()

# Math library (complex.h / cexp etc.) where it is a separate archive.
find_library(SCALING_M_LIB m)
if(SCALING_M_LIB)
    foreach(drv ${_scaling_drivers})
        target_link_libraries(${drv} PRIVATE ${SCALING_M_LIB})
    endforeach()
endif()

# Self-tests need no corpus and no external toolchain, so they are always
# registerable and prove each driver's oracle catches a corrupted value.
enable_testing()
add_test(NAME scaling_diff_selftest      COMMAND scaling_diff      --selftest)
add_test(NAME scaling_stab_selftest      COMMAND scaling_stab      --selftest)
add_test(NAME scaling_var_d_selftest     COMMAND scaling_var_d     --selftest)
add_test(NAME scaling_dmrg_tdvp_selftest COMMAND scaling_dmrg_tdvp --selftest)
# Large-n Clifford structure needs no corpus either; register it directly.
add_test(NAME scaling_stab_large         COMMAND scaling_stab --n 50,100)
set_tests_properties(
    scaling_diff_selftest scaling_stab_selftest scaling_var_d_selftest
    scaling_dmrg_tdvp_selftest scaling_stab_large
    PROPERTIES LABELS "scaling")
