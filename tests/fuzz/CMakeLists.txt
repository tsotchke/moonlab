# tests/fuzz/CMakeLists.txt -- coverage-guided fuzz lane for libquantumsim.
#
# This directory is self-contained.  The integrator wires it in with a
# single line in the root CMakeLists.txt, gated behind the option:
#
#     option(QSIM_ENABLE_FUZZING "Build libFuzzer/AFL++ harnesses" OFF)
#     if(QSIM_ENABLE_FUZZING)
#         add_subdirectory(tests/fuzz)
#     endif()
#
# Nothing here runs unless QSIM_ENABLE_FUZZING is ON, so a normal build
# is completely untouched.
#
# What it produces, per untrusted-input surface:
#   <name>          -- libFuzzer engine build (-fsanitize=fuzzer,address,undefined)
#   <name>_replay   -- no-engine build with a stdin/file driver
#                      (-fsanitize=address,undefined), replays a seed
#                      corpus deterministically; also registered with ctest.
#
# IMPORTANT (instrumentation of the library under test):
#   For the sanitizers and libFuzzer's coverage to see the *library* code
#   where the parsers live (control_plane.c, moonlab_qgtl_backend.c,
#   mlkem.c, config.c, entropy.c, ...), the quantumsim target itself must
#   be built with ASan/UBSan/SanitizerCoverage.  This file adds those
#   options to the existing quantumsim target -- it only *appends* flags
#   (every platform flag the root file set is preserved) and does so only
#   when QSIM_ENABLE_FUZZING is ON.  Modifying a target defined in a
#   parent directory is supported since CMake 3.13; the project requires
#   3.20.  No source file owned by another lane is edited.
#
#   A STATIC quantumsim is recommended for fuzzing (fully hermetic link,
#   no shared-sanitizer-runtime juggling); pass
#       -DQSIM_BUILD_STATIC=ON -DQSIM_BUILD_SHARED=OFF
#   The CI workflow in .github/workflows/fuzz.yml does this.  A SHARED
#   build also works (the sanitizer runtime is linked into the dylib
#   below) but is less portable across CI runners.

if(NOT QSIM_ENABLE_FUZZING)
    return()
endif()

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "[fuzz] QSIM_ENABLE_FUZZING requires Clang (libFuzzer / SanitizerCoverage); "
        "got C compiler '${CMAKE_C_COMPILER_ID}'. Re-run cmake with "
        "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++.")
endif()

message(STATUS "[fuzz] enabling ASan+UBSan+SanitizerCoverage on quantumsim and building fuzz harnesses")

# Sanitizer selection.  address+undefined are the runtime oracles applied
# to both the library under test and the harnesses.
set(QSIM_FUZZ_SANITIZE     "address,undefined")
set(QSIM_FUZZ_COMMON_FLAGS -g -O1 -fno-omit-frame-pointer -fno-sanitize-recover=all)

# Optional: also add libFuzzer coverage instrumentation (SanitizerCoverage)
# to the library so the engine sees edges *inside* quantumsim, not just the
# harness TU.  Default OFF because Apple's system `ld` rejects Homebrew
# clang's SanitizerCoverage relocations in a separately-compiled object
# ("invalid r_symbolnum"); a Linux runner or an lld-equipped toolchain can
# turn it ON for deeper coverage-guided reach.  ASan/UBSan safety checking
# of the library is unaffected either way.
option(QSIM_FUZZ_LIBRARY_COVERAGE
       "Add SanitizerCoverage to quantumsim (needs lld / a capable linker)" OFF)

set(_qsim_lib_fuzz_flags -fsanitize=${QSIM_FUZZ_SANITIZE})
if(QSIM_FUZZ_LIBRARY_COVERAGE)
    set(_qsim_lib_fuzz_flags -fsanitize=fuzzer-no-link,${QSIM_FUZZ_SANITIZE})
endif()

# --- 1. Instrument the library under test ------------------------------------
# ASan+UBSan compile the whole library so heap-overflows / UB inside the
# parsers, crypto, and ABI code are caught -- the harnesses only reach that
# code, they do not contain it.  Appends flags; nothing the root file set is
# removed.
target_compile_options(quantumsim PRIVATE
    ${_qsim_lib_fuzz_flags} ${QSIM_FUZZ_COMMON_FLAGS})
# PRIVATE link option: pulls the sanitizer runtime into libquantumsim
# itself so a SHARED build resolves its ASan/UBSan interceptors; a STATIC
# build ignores it harmlessly.
target_link_options(quantumsim PRIVATE -fsanitize=${QSIM_FUZZ_SANITIZE})

get_target_property(_qsim_type quantumsim TYPE)
message(STATUS "[fuzz] quantumsim built as ${_qsim_type} with -fsanitize=${QSIM_FUZZ_SANITIZE}"
               "${_qsim_cov_note}")
message(STATUS "[fuzz] NOTE for the integrator: build the fuzz configure with "
               "-DQSIM_ENABLE_OPENMP=OFF -DQSIM_ENABLE_METAL=OFF -DQSIM_WERROR=OFF "
               "so the sanitized library links cleanly: the instrumented build "
               "stays free of the OpenMP runtime dependency and the "
               "Objective-C++ framework surface.")

# --- 2. Homebrew-clang libc++ workaround for the libFuzzer runtime -----------
# On Apple silicon the Homebrew clang ships libclang_rt.fuzzer_osx.a built
# against Homebrew's libc++, whose internal symbols (e.g. __hash_memory)
# are absent from the system libc++.  Point the linker at the matching
# libc++ for the fuzzer-engine executables only.
get_filename_component(_qsim_clang_bin "${CMAKE_C_COMPILER}" DIRECTORY)
set(_qsim_libcxx_dir "${_qsim_clang_bin}/../lib/c++")
set(QSIM_FUZZ_ENGINE_LINK_EXTRA "")
if(APPLE AND EXISTS "${_qsim_libcxx_dir}")
    list(APPEND QSIM_FUZZ_ENGINE_LINK_EXTRA
        "-L${_qsim_libcxx_dir}" "-Wl,-rpath,${_qsim_libcxx_dir}")
    message(STATUS "[fuzz] using libc++ from ${_qsim_libcxx_dir} for libFuzzer link")
endif()
# The default Apple linker (ld-prime, Xcode 15+) rejects Homebrew clang's
# SanitizerCoverage relocations with "invalid r_symbolnum"; the classic
# Apple linker accepts them.  Only needed for the engine (coverage) builds
# on macOS -- Linux linkers handle sancov natively.  If a future macOS
# drops -ld_classic, install lld (`brew install lld`) and pass
# -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld instead.
if(APPLE)
    list(APPEND QSIM_FUZZ_ENGINE_LINK_EXTRA "-Wl,-ld_classic")
endif()

# --- 3. Per-surface targets --------------------------------------------------
# name  <->  translation unit.  The name is also the ctest / JSONL trace
# id and the corpus/dict directory basename, so run_fuzz.sh stays a loop.
set(QSIM_FUZZ_SPECS
    "control_plane_protocol_fuzz|fuzz_target_control_plane_protocol.c"
    "circuit_deserialize_fuzz|fuzz_target_circuit_deserialize.c"
    "config_parse_fuzz|fuzz_target_config_parse.c"
    "mlkem_decode_fuzz|fuzz_target_mlkem_decode.c"
    "entropy_input_fuzz|fuzz_target_entropy_input.c"
    "abi_boundary_fuzz|fuzz_target_abi_boundary.c")

foreach(_spec IN LISTS QSIM_FUZZ_SPECS)
    string(REPLACE "|" ";" _pair "${_spec}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _src)

    # -- libFuzzer engine build --
    add_executable(${_name} "${_src}")
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # -UNDEBUG keeps the assert()-based ABI-contract oracles live even
    # under RelWithDebInfo/Release (which define NDEBUG).
    target_compile_options(${_name} PRIVATE
        -fsanitize=fuzzer,${QSIM_FUZZ_SANITIZE} ${QSIM_FUZZ_COMMON_FLAGS} -UNDEBUG)
    target_link_options(${_name} PRIVATE
        -fsanitize=fuzzer,${QSIM_FUZZ_SANITIZE} ${QSIM_FUZZ_ENGINE_LINK_EXTRA})
    target_link_libraries(${_name} PRIVATE quantumsim ${MATH_LIBRARY} Threads::Threads)

    # -- replay build (no engine; stdin/file driver from fuzz_driver.c) --
    add_executable(${_name}_replay "${_src}" fuzz_driver.c)
    target_include_directories(${_name}_replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${_name}_replay PRIVATE
        -fsanitize=${QSIM_FUZZ_SANITIZE} ${QSIM_FUZZ_COMMON_FLAGS} -UNDEBUG)
    target_link_options(${_name}_replay PRIVATE -fsanitize=${QSIM_FUZZ_SANITIZE})
    target_link_libraries(${_name}_replay PRIVATE quantumsim ${MATH_LIBRARY} Threads::Threads)

    # -- ctest: replay the whole seed corpus under the sanitizers --
    add_test(NAME fuzz_replay_${_name}
             COMMAND ${_name}_replay "${CMAKE_CURRENT_SOURCE_DIR}/corpora/${_name}")
    set_tests_properties(fuzz_replay_${_name} PROPERTIES
        LABELS "fuzz"
        TIMEOUT 300
        ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;ASAN_OPTIONS=abort_on_error=1")
endforeach()

message(STATUS "[fuzz] targets: ${QSIM_FUZZ_SPECS}")
