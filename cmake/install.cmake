# Moonlab install, downstream-consumer metadata, reproducibility header, and CPack.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Platform wheels carry their native library inside the Python package and
# intentionally omit the C/CMake SDK payload. Normal native installs retain
# the complete export/header/package-config contract below.
if(QSIM_PYTHON_WHEEL)
    install(TARGETS quantumsim
        EXPORT quantumsim-targets
        LIBRARY DESTINATION moonlab/.libs COMPONENT python-wheel
        RUNTIME DESTINATION moonlab/.libs COMPONENT python-wheel)
else()
    install(TARGETS quantumsim
        EXPORT quantumsim-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(NOT QSIM_PYTHON_WHEEL)
    install(DIRECTORY src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quantumsim
        COMPONENT native-sdk
        FILES_MATCHING PATTERN "*.h")
    install(FILES src/applications/moonlab_api.h src/applications/moonlab_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/moonlab
        COMPONENT native-sdk)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/moonlab_features.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT native-sdk)

    install(EXPORT quantumsim-targets
        FILE quantumsim-targets.cmake
        NAMESPACE quantumsim::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
        COMPONENT native-sdk)
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/quantumsim-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/quantumsim-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quantumsim
        COMPONENT native-sdk)

    # The template uses ${pcfiledir} so extracted packages stay relocatable. Count
    # the components in libdir/pkgconfig to support both lib and multiarch libdirs.
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
        set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")
        set(CMAKE_INSTALL_LIBDIR \"${CMAKE_INSTALL_LIBDIR}\")
        set(CMAKE_INSTALL_INCLUDEDIR \"${CMAKE_INSTALL_INCLUDEDIR}\")
        set(PROJECT_DESCRIPTION \"${PROJECT_DESCRIPTION}\")
        set(PROJECT_VERSION \"${PROJECT_VERSION}\")
        set(QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR \"${QSIM_PC_PREFIX_FROM_PKGCONFIG_DIR}\")
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/quantumsim.pc.in
            ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc.install
            @ONLY)")
endif()

# Capture the source revision and feature set in an installed build-info header.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_SHA_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MOONLAB_GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain=v1 --untracked-files=all
            -- . ":(exclude)scripts/icc_traces/**"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE MOONLAB_GIT_STATUS_RC
        OUTPUT_VARIABLE MOONLAB_GIT_STATUS
        ERROR_QUIET)
    if(MOONLAB_GIT_STATUS_RC EQUAL 0 AND MOONLAB_GIT_STATUS STREQUAL "")
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
set(MOONLAB_ENABLED_FEATURES_LIST "")
foreach(pair
    "QSIM_ENABLE_OPENMP:openmp" "QSIM_ENABLE_METAL:metal"
    "QSIM_ENABLE_OPENCL:opencl" "QSIM_ENABLE_VULKAN:vulkan"
    "QSIM_ENABLE_CUQUANTUM:cuquantum" "QSIM_ENABLE_MPI:mpi"
    "QSIM_ENABLE_ESHKOL:eshkol" "QSIM_ENABLE_LTO:lto"
    "QSIM_ENABLE_SANITIZERS:sanitizers" "QSIM_ENABLE_AVX512:avx512"
    "QSIM_ENABLE_AVX2:avx2" "QSIM_ENABLE_NEON:neon" "QSIM_ENABLE_SVE:sve")
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
    ${CMAKE_CURRENT_BINARY_DIR}/generated/moonlab_build_info.h @ONLY)
target_include_directories(quantumsim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
if(NOT QSIM_PYTHON_WHEEL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/moonlab_build_info.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT native-sdk)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/quantumsim.pc.install
        RENAME quantumsim.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        COMPONENT native-sdk)
endif()

# Keep a native `cpack` target available in addition to the release wrapper.
set(CPACK_PACKAGE_NAME "moonlab")
set(CPACK_PACKAGE_VENDOR "tsotchke")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME
    "moonlab-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_GENERATOR "TGZ")
include(CPack)
