# tests/statistical/CMakeLists.txt
#
# Self-contained adversarial statistical + crypto-negative-space lane.
# Wired into the build by the integrator with a single line, placed inside
# the root's QSIM_BUILD_TESTS block AFTER the `quantumsim` target exists:
#
#     add_subdirectory(tests/statistical)
#
# It deliberately depends on nothing from the parent scope except the
# `quantumsim` target itself (whose PUBLIC usage requirements carry the src/
# include dirs and libm).  Sanitizer flags, when the top-level build sets
# -DQSIM_ENABLE_SANITIZERS=ON, propagate here through the root's directory-
# scoped add_compile_options()/add_link_options(), so building the whole tree
# under ASan builds these targets under ASan too -- which is exactly how the
# negative-fuzz target should run in CI.

cmake_minimum_required(VERSION 3.16)

if(NOT TARGET quantumsim)
    message(WARNING
        "tests/statistical: target 'quantumsim' not defined; "
        "skipping statistical battery. Add this subdirectory after the "
        "quantumsim library target.")
    return()
endif()

find_package(Threads)

# Helper: one adversarial target + its ctest registration.
function(qsim_add_statistical_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE quantumsim)
    if(TARGET Threads::Threads)
        target_link_libraries(${name} PRIVATE Threads::Threads)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        LABELS "statistical;crypto"
        TIMEOUT ${ARGV2})
endfunction()

# QRNG statistical battery + GROVER positive control.  Generous timeout: the
# streaming multi-MB check draws real conditioned bytes.
qsim_add_statistical_test(stat_qrng_battery
    ${CMAKE_CURRENT_SOURCE_DIR}/test_qrng_battery.c 600)
# The battery draws several MB of *live* conditioned RNG (non-deterministic)
# and takes ~50s; it belongs in the nightly tier, not the fast PR lane where a
# sustained multi-MB draw under load can transiently under-run the entropy pool.
# The statistical.yml nightly lane runs it; the deep_battery drives it directly.
set_tests_properties(stat_qrng_battery PROPERTIES LABELS "statistical;crypto;long")

# ML-KEM negative / tamper fuzz + shared-secret avalanche (run under ASan in CI).
qsim_add_statistical_test(stat_mlkem_adversarial
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mlkem_adversarial.c 300)

# Constant-time variance measurement (informational unless QSIM_TIMING_STRICT).
qsim_add_statistical_test(stat_mlkem_timing
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mlkem_timing.c 300)

# Entropy health-test adversarial inputs (anti-theater positive controls).
qsim_add_statistical_test(stat_entropy_health_adversarial
    ${CMAKE_CURRENT_SOURCE_DIR}/test_entropy_health_adversarial.c 120)

message(STATUS "tests/statistical: 4 adversarial targets registered "
        "(labels: statistical, crypto)")
