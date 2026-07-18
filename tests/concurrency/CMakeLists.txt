# tests/concurrency/CMakeLists.txt
#
# Self-contained ThreadSanitizer harness build for the Moonlab concurrency
# lane.  It does NOT hook into the root CMakeLists and never edits src/ --
# it compiles the checked-in harnesses under -fsanitize=thread and links them
# against a PRE-BUILT, TSan-instrumented libquantumsim.a that scripts/run_tsan.sh
# produces in a standalone build dir.
#
# Point it at the repo + the two libraries:
#
#   cmake -S tests/concurrency -B build-tsan-conc \
#         -DMOONLAB_ROOT=<repo-root> \
#         -DMOONLAB_TSAN_LIB=<repo>/build-tsan/libquantumsim.a \
#         -DMOONLAB_TSAN_LIB_OMP=<repo>/build-tsan-omp/libquantumsim.a   # optional
#
# The OpenMP harness (conc_grover_gates) is built only when
# MOONLAB_TSAN_LIB_OMP is provided.  All other harnesses use MOONLAB_TSAN_LIB
# (an OpenMP-OFF build gives a clean pthread-only TSan signal).

cmake_minimum_required(VERSION 3.20)
project(moonlab_concurrency C)

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "[conc] ThreadSanitizer needs Clang or GCC; got ${CMAKE_C_COMPILER_ID}")
endif()

# POSIX-only: the harnesses use pthreads directly plus <unistd.h> /
# sched_yield / nanosleep, and ThreadSanitizer targets POSIX threads.  The
# lane is gated off Windows (mirrors the NOT QSIM_PLATFORM_WINDOWS gate on the
# adversarial suites in cmake/tests.cmake); run it on Linux/macOS via
# scripts/run_tsan.sh or the Linux .github/workflows/tsan.yml CI job.
if(WIN32)
    message(FATAL_ERROR "[conc] The concurrency/TSan lane is POSIX-only and is gated off Windows.")
endif()

set(MOONLAB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Moonlab repo root")
set(MOONLAB_TSAN_LIB "" CACHE FILEPATH "Prebuilt TSan libquantumsim.a (OpenMP OFF)")
set(MOONLAB_TSAN_LIB_OMP "" CACHE FILEPATH "Prebuilt TSan libquantumsim.a (OpenMP ON) -- optional")
set(MOONLAB_EXTRA_LIBS "" CACHE STRING "Extra link libs the static lib needs (e.g. openblas;lapack)")
set(MOONLAB_OMP_LIB "" CACHE FILEPATH "OpenMP runtime to link the omp harness (default: auto)")

if(NOT MOONLAB_TSAN_LIB)
    message(FATAL_ERROR "[conc] set -DMOONLAB_TSAN_LIB=<path to TSan libquantumsim.a>")
endif()

find_package(Threads REQUIRED)

set(_tsan_flags -fsanitize=thread -g -O1 -fno-omit-frame-pointer)

# Math / BLAS backend the static library depends on.
if(APPLE)
    set(_math_libs "-framework Accelerate")
else()
    find_library(_libm m)
    set(_math_libs ${_libm})
endif()
if(MOONLAB_EXTRA_LIBS)
    list(APPEND _math_libs ${MOONLAB_EXTRA_LIBS})
endif()

enable_testing()

# --- Non-OpenMP harnesses (link the OpenMP-OFF library) ----------------------
set(_conc_targets
    conc_core_init
    conc_control_plane
    conc_entropy_pool
    conc_audit_buffer
    conc_scheduler
    conc_clifford_measurement)

foreach(_t IN LISTS _conc_targets)
    add_executable(${_t} ${_t}.c)
    target_include_directories(${_t} PRIVATE "${MOONLAB_ROOT}")
    target_compile_options(${_t} PRIVATE ${_tsan_flags})
    target_link_options(${_t} PRIVATE -fsanitize=thread)
    target_link_libraries(${_t} PRIVATE
        "${MOONLAB_TSAN_LIB}" ${_math_libs} Threads::Threads)
endforeach()

# ctest registrations for the configurations that MUST stay clean.
add_test(NAME conc_control_plane_steady    COMMAND conc_control_plane steady)
add_test(NAME conc_entropy_pool_steady     COMMAND conc_entropy_pool  steady)
add_test(NAME conc_entropy_pool_toggle     COMMAND conc_entropy_pool  toggle)
add_test(NAME conc_audit_buffer_mpmc       COMMAND conc_audit_buffer  mpmc)
add_test(NAME conc_scheduler               COMMAND conc_scheduler)
add_test(NAME conc_clifford_measurement    COMMAND conc_clifford_measurement)
set_tests_properties(
    conc_control_plane_steady conc_entropy_pool_steady conc_entropy_pool_toggle
    conc_audit_buffer_mpmc conc_scheduler conc_clifford_measurement
    PROPERTIES LABELS "tsan;clean" TIMEOUT 180
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1:abort_on_error=1")

# Regression configurations for the four fixed concurrency bugs (v1.2).  These
# harnesses drive the exact race / deadlock windows that once fired; they now
# PASS only when the fix holds -- TSan runs with halt_on_error=0 (report, don't
# abort) so any surviving report lands in the log, and a FAIL_REGULAR_EXPRESSION
# fails the test if the race / deadlock text reappears (i.e. a regression).
#   conc_core_init_coldstart      -> config.c lazy-init + simd_ops.c vtable once
#   conc_control_plane_adversarial-> control-plane config setters under cfg_lock
#   conc_audit_buffer_destroy     -> audit_buffer destroy vs in-flight drain
add_test(NAME conc_core_init_coldstart       COMMAND conc_core_init)
add_test(NAME conc_control_plane_adversarial COMMAND conc_control_plane adversarial)
add_test(NAME conc_audit_buffer_destroy      COMMAND conc_audit_buffer destroy)
# abort_on_error=0 stops TSan's Darwin-default SIGABRT so the process exits
# normally and the regexes below govern pass/fail.
set_tests_properties(conc_core_init_coldstart PROPERTIES
    LABELS "tsan;regression" TIMEOUT 180
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=0:abort_on_error=0"
    FAIL_REGULAR_EXPRESSION "data race")
set_tests_properties(conc_control_plane_adversarial PROPERTIES
    LABELS "tsan;regression" TIMEOUT 180
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=0:abort_on_error=0"
    FAIL_REGULAR_EXPRESSION "data race")
# The destroy path retries across rounds; it must complete every round with no
# DEADLOCK watchdog fire (the drain in moonlab_audit_buffer_destroy holds).
set_tests_properties(conc_audit_buffer_destroy PROPERTIES
    LABELS "tsan;regression" TIMEOUT 60
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=0:abort_on_error=0"
    PASS_REGULAR_EXPRESSION "destroy: .* rounds completed"
    FAIL_REGULAR_EXPRESSION "DEADLOCK")

# --- OpenMP harness (link the OpenMP-ON library) -----------------------------
if(MOONLAB_TSAN_LIB_OMP)
    if(MOONLAB_OMP_LIB)
        set(_omp_lib "${MOONLAB_OMP_LIB}")
    elseif(APPLE)
        find_library(_omp_lib omp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(NOT _omp_lib)
            set(_omp_lib "-lomp")
        endif()
    else()
        set(_omp_lib "-lomp")
    endif()

    add_executable(conc_grover_gates conc_grover_gates.c)
    target_include_directories(conc_grover_gates PRIVATE "${MOONLAB_ROOT}")
    target_compile_options(conc_grover_gates PRIVATE ${_tsan_flags})
    target_link_options(conc_grover_gates PRIVATE -fsanitize=thread)
    target_link_libraries(conc_grover_gates PRIVATE
        "${MOONLAB_TSAN_LIB_OMP}" ${_math_libs} Threads::Threads "${_omp_lib}")

    # libomp on dev machines is not Archer-instrumented -> suppress its
    # internal false positives (see tsan.supp).  On an Archer-instrumented
    # libomp (Linux CI) the suppressions are simply unused.
    add_test(NAME conc_grover_gates COMMAND conc_grover_gates all)
    set_tests_properties(conc_grover_gates PROPERTIES
        LABELS "tsan;openmp" TIMEOUT 180
        ENVIRONMENT
        "TSAN_OPTIONS=halt_on_error=1:abort_on_error=1:suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp;OMP_NUM_THREADS=4")
else()
    message(STATUS "[conc] MOONLAB_TSAN_LIB_OMP not set -- skipping conc_grover_gates")
endif()
