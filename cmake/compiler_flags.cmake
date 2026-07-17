# Compiler detection, diagnostics, sanitizers, and optimization policy.

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(QSIM_COMPILER_CLANG ON)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(QSIM_COMPILER_GCC ON)
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(QSIM_COMPILER_MSVC ON)
endif()

set(QSIM_WARNING_FLAGS)
if(QSIM_COMPILER_CLANG AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # clang-cl: MSVC flag surface. GNU-style -Wall is aliased to MSVC
    # /Wall here, which clang-cl maps to -Weverything (-Wpadded,
    # -Wpre-c11-compat, -Wdocumentation, ... as errors under /WX).
    # /W4 is the clang-cl spelling of -Wall -Wextra.
    list(APPEND QSIM_WARNING_FLAGS /W4)
    if(QSIM_WERROR)
        list(APPEND QSIM_WARNING_FLAGS /WX)
    endif()
    # The UCRT deprecates the POSIX spellings (strdup, strcasecmp, ...) and the
    # non-_s CRT calls in favour of Microsoft-specific names; under /WX those
    # deprecations become hard errors. The POSIX names are the portable ones we
    # build against on every other platform, so silence the CRT nags rather
    # than fork every call site to _strdup and friends.
    add_compile_definitions(_CRT_NONSTDC_NO_WARNINGS _CRT_SECURE_NO_WARNINGS)
elseif(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
    list(APPEND QSIM_WARNING_FLAGS -Wall -Wextra -Wpedantic)
    if(QSIM_WERROR)
        # -Werror everywhere our code is responsible. Demotions are limited to
        # accumulated legacy categories or diagnostics unavailable on older
        # supported compilers; every optional spelling is feature-detected.
        include(CheckCCompilerFlag)
        list(APPEND QSIM_WARNING_FLAGS -Werror)

        list(APPEND QSIM_WARNING_FLAGS
             -Wno-error=pedantic
             -Wno-error=deprecated-declarations
             -Wno-error=unused-parameter
             -Wno-error=unused-variable
             -Wno-error=unused-function
             -Wno-error=unused-but-set-variable
             -Wno-error=address)

        set(_qsim_optional_wno_error
            discarded-qualifiers       # gcc: config.c const-string init
            switch-unreachable         # gcc: legacy default-case fall-through
            ignored-qualifiers         # clang spelling of discarded-qualifiers
            unused-but-set-variable    # gcc 11+: control_plane.c log_peer_buf
            format-truncation          # gcc strict snprintf truncation
            unknown-pragmas            # gcc: tensor.c #pragma STDC FP_CONTRACT
            nan-infinity-disabled      # clang+ffast-math: qaoa.c INFINITY sentinel
            strict-aliasing            # gcc: gpu_vulkan.c pointer punning
            unused-result              # gcc: gpu_vulkan.c fread return value
            overlength-strings         # gcc strict: gpu_opencl.c kernel source > 4095
            missing-braces             # gcc strict: cl_float2 initializer
            implicit-fallthrough       # preventive, in case it appears
            address                    # already in base set on some toolchains
            stringop-truncation        # gcc strict snprintf
        )
        foreach(_wname IN LISTS _qsim_optional_wno_error)
            string(MAKE_C_IDENTIFIER "HAVE_WNO_ERROR_${_wname}" _have_var)
            check_c_compiler_flag("-Wno-error=${_wname}" ${_have_var})
            if(${_have_var})
                list(APPEND QSIM_WARNING_FLAGS "-Wno-error=${_wname}")
            endif()
        endforeach()
    endif()
elseif(QSIM_COMPILER_MSVC)
    list(APPEND QSIM_WARNING_FLAGS /W4)
    if(QSIM_WERROR)
        list(APPEND QSIM_WARNING_FLAGS /WX)
    endif()
endif()

# AddressSanitizer + UndefinedBehaviorSanitizer. Opt-in for CI and development;
# keep enough optimization for usable execution without losing diagnostics.
set(QSIM_SANITIZER_FLAGS)
if(QSIM_ENABLE_SANITIZERS AND (QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC))
    list(APPEND QSIM_SANITIZER_FLAGS
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -g
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND QSIM_SANITIZER_FLAGS -O1)
    endif()
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

set(QSIM_OPTIMIZE_FLAGS)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(QSIM_COMPILER_CLANG OR QSIM_COMPILER_GCC)
        list(APPEND QSIM_OPTIMIZE_FLAGS -O3 -fno-math-errno -ffp-contract=fast -funroll-loops)
        if(QSIM_FAST_MATH)
            list(APPEND QSIM_OPTIMIZE_FLAGS -ffast-math)
            message(WARNING
                "QSIM_FAST_MATH=ON permits non-IEEE transformations. "
                "Release artifacts and correctness validation must leave it OFF.")
        endif()

        # Distributable artifacts use the architecture baseline. Runtime SIMD
        # dispatch remains available even when build-machine tuning is off.
        if(QSIM_NATIVE_ARCH AND NOT CMAKE_CROSSCOMPILING)
            list(APPEND QSIM_OPTIMIZE_FLAGS -march=native)
        endif()
    elseif(QSIM_COMPILER_MSVC)
        list(APPEND QSIM_OPTIMIZE_FLAGS /O2)
        if(QSIM_FAST_MATH)
            list(APPEND QSIM_OPTIMIZE_FLAGS /fp:fast)
            message(WARNING
                "QSIM_FAST_MATH=ON permits non-IEEE transformations. "
                "Release artifacts and correctness validation must leave it OFF.")
        else()
            list(APPEND QSIM_OPTIMIZE_FLAGS /fp:precise)
        endif()
    endif()
endif()
