cmake_minimum_required(VERSION 3.15)
project(moonlab_wasm C)

set(CMAKE_C_STANDARD 11)

# Root of quantum simulator source
set(QSIM_ROOT "${CMAKE_SOURCE_DIR}/../../../../../")
option(MOONLAB_WASM_ENABLE_WEBGPU "Enable WebGPU backend scaffolding for WASM" ON)

# Source files - Core quantum operations
set(QUANTUM_SOURCES
    ${QSIM_ROOT}/src/quantum/state.c
    ${QSIM_ROOT}/src/quantum/gates.c
    ${QSIM_ROOT}/src/quantum/measurement.c
    ${QSIM_ROOT}/src/quantum/entanglement.c
    ${QSIM_ROOT}/src/quantum/noise.c
    # MPDO mixed-state simulator (since 0.3.0; JS binding from 0.4.10).
    ${QSIM_ROOT}/src/quantum/noise_mpdo.c
)

# Source files - Algorithms
set(ALGORITHM_SOURCES
    ${QSIM_ROOT}/src/algorithms/grover.c
    ${QSIM_ROOT}/src/algorithms/grover_optimizer.c
    ${QSIM_ROOT}/src/algorithms/vqe.c
    # Quantum natural gradient (Fubini-Study metric; referenced from vqe.c).
    ${QSIM_ROOT}/src/algorithms/vqe_qng.c
    ${QSIM_ROOT}/src/algorithms/qaoa.c
    ${QSIM_ROOT}/src/algorithms/bell_tests.c
    # Native reverse-mode autograd (referenced unconditionally from vqe.c).
    ${QSIM_ROOT}/src/algorithms/diff/differentiable.c
)

# Source files - Tensor Network solvers (DMRG/TDVP/TEBD helpers, MPO utilities)
set(TENSOR_NETWORK_SOURCES
    ${QSIM_ROOT}/src/algorithms/tensor_network/contraction.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/dmrg.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/lattice_2d.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/mpo_2d.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/skyrmion_braiding.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/svd_compress.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/tdvp.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/tensor.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/tn_gates.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/tn_measurement.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/tn_state.c
    # CA-MPS + var-D + gauge warmstart + CA-PEPS scaffold (since 0.2.1).
    ${QSIM_ROOT}/src/algorithms/tensor_network/ca_mps.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/ca_mps_var_d.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.c
    ${QSIM_ROOT}/src/algorithms/tensor_network/ca_peps.c
)

# Source files - Aaronson-Gottesman tableau backend (since 0.2.1, used by CA-MPS).
set(BACKEND_SOURCES
    ${QSIM_ROOT}/src/backends/clifford/clifford.c
)

# Source files - Topological QC (surface code Clifford-tableau variant,
# since 0.5.14 in WASM).  Pulls in clifford.c + gates.c + matrix_math.c,
# all already in the build above.
set(TOPOLOGICAL_SOURCES
    ${QSIM_ROOT}/src/algorithms/topological/topological.c
)

# Source files - HEP applications (since 0.2.1).
# Note: moonlab_qrng_export.c (the qrng-backed half of the
# v0.2.1 stable ABI) deliberately stays out of the WASM build
# because it pulls in qrng + hardware_entropy (heavy, much of
# it host-only).  The lean half -- abi_version, qwz_chern,
# dmrg_*, ca_mps_var_d_*, ca_mps_gauge_warmstart, z2_lgt_1d_*,
# status_string -- lives in moonlab_export_lean.c and IS
# included below; it pulls in only qgt + dmrg + ca_mps + status,
# all of which are already in the WASM build (since 0.5.0).
#
# moonlab_tdvp_export.c (v0.4.3 adaptive-bond TDVP wrapper) has
# lean deps too: tn_state + dmrg + tdvp.
#
# moonlab_vqe_export.c (v1.1.0 exact-gradient wrapper) depends only
# on vqe.c + diff/, both already compiled into the WASM build.
set(APPLICATION_SOURCES
    ${QSIM_ROOT}/src/applications/hep/lattice_z2_1d.c
    ${QSIM_ROOT}/src/applications/moonlab_export_lean.c
    ${QSIM_ROOT}/src/applications/moonlab_tdvp_export.c
    ${QSIM_ROOT}/src/applications/moonlab_vqe_export.c
    # qgt.c powers moonlab_qwz_chern; topology binding TS layer
    # also needs the lower-level qgt_* surface (since 0.5.0).
    ${QSIM_ROOT}/src/algorithms/quantum_geometry/qgt.c
    # libirrep bridge (since v0.6.0).  WASM build never has libirrep
    # linked, so the TU compiles to the no-op stub path that returns
    # MOONLAB_LIBIRREP_NOT_BUILT from every factory.  JS callers
    # gate on libirrep.is_available() before constructing codes.
    ${QSIM_ROOT}/src/integration/libirrep_bridge.c
    # QGTL ingestion + scheduler (v0.6.6 / v0.7.0).  Both ship into the
    # WASM build as the foundation for browser-runnable cloud-platform
    # demos.  Scheduler uses OpenMP in-process when available, which
    # Emscripten compiles as a no-op (single-threaded fan-out).
    ${QSIM_ROOT}/src/applications/moonlab_qgtl_backend.c
    ${QSIM_ROOT}/src/applications/decoder_bench.c
    # Exact MWPM decoder for the toric code -- referenced by
    # decoder_bench.c via moonlab_mwpm_exact_decode_toric.
    ${QSIM_ROOT}/src/applications/mwpm_exact.c
    # Vendor-noise emulator backends (since v1.1) -- IBM Falcon /
    # Rigetti Aspen / IonQ Forte stochastic-Pauli emulators that
    # plug into the scheduler's backend registry.
    ${QSIM_ROOT}/src/applications/vendor_noise_backend.c
    ${QSIM_ROOT}/src/distributed/scheduler.c
)

# Source files - Utilities
set(UTILS_SOURCES
    ${QSIM_ROOT}/src/utils/entropy.c
    ${QSIM_ROOT}/src/utils/matrix_math.c
    ${QSIM_ROOT}/src/utils/config.c
    ${QSIM_ROOT}/src/utils/moonlab_status.c
    # WASM-only quantum-entropy stack (since 0.5.4): the native
    # hardware_entropy.c pulls in RDRAND / /dev/urandom / jitter
    # paths that don't make sense on wasm32.  hardware_entropy_wasm.c
    # provides the same `entropy_init` / `entropy_get_bytes` /
    # `entropy_free` surface backed by getentropy(3) (which
    # emscripten polyfills via crypto.getRandomValues).  Plus the
    # non-inline `quantum_entropy_ctx_create_hw` / `_destroy`
    # helpers that VQE / QAOA / Bell / Grover ctypes / FFI consumers
    # call through.
    ${QSIM_ROOT}/src/compat/hardware_entropy_wasm.c
    ${QSIM_ROOT}/src/utils/quantum_entropy.c
)

# Source files - Optimization (SIMD operations with scalar fallbacks for WASM)
set(OPTIMIZATION_SOURCES
    ${QSIM_ROOT}/src/optimization/memory_align.c
    ${QSIM_ROOT}/src/optimization/simd_ops.c
    # Single-qubit gate-fusion DAG (since 0.2.0; JS binding from 0.4.7).
    ${QSIM_ROOT}/src/optimization/fusion/fusion.c
    ${QSIM_ROOT}/src/optimization/gpu/gpu_backend.c
    # Eshkol symbols are referenced unconditionally from tensor.c; in
    # the WASM build we don't have the C++ implementation so link the
    # C no-op stub that always reports "not available" and returns
    # MOONLAB_ESHKOL_NOT_BUILT.  Runtime code falls back to CPU.
    ${QSIM_ROOT}/src/optimization/gpu/backends/gpu_eshkol_stub.c
)
if(MOONLAB_WASM_ENABLE_WEBGPU)
    list(APPEND OPTIMIZATION_SOURCES
        ${QSIM_ROOT}/src/optimization/gpu/backends/gpu_webgpu.c
    )
endif()

# Combine all sources
set(ALL_SOURCES
    ${QUANTUM_SOURCES}
    ${ALGORITHM_SOURCES}
    ${TENSOR_NETWORK_SOURCES}
    ${BACKEND_SOURCES}
    ${TOPOLOGICAL_SOURCES}
    ${APPLICATION_SOURCES}
    ${UTILS_SOURCES}
    ${OPTIMIZATION_SOURCES}
    ${CMAKE_SOURCE_DIR}/f2c_stub.c
)

# Create the WASM executable
add_executable(moonlab ${ALL_SOURCES})

# BLAS/LAPACK for tensor-network solvers (OpenBLAS preferred, CLAPACK fallback)
set(OPENBLAS_WASM_ROOT "$ENV{OPENBLAS_WASM_ROOT}")
if(NOT OPENBLAS_WASM_ROOT)
    set(OPENBLAS_WASM_ROOT "${CMAKE_SOURCE_DIR}/build/deps/openblas-wasm")
endif()

if(MOONLAB_WASM_ENABLE_WEBGPU)
    target_compile_definitions(moonlab PRIVATE HAS_WEBGPU=1 QSIM_HAS_WEBGPU=1)
endif()

set(CLAPACK_WASM_ROOT "$ENV{CLAPACK_WASM_ROOT}")
if(NOT CLAPACK_WASM_ROOT)
    set(CLAPACK_WASM_ROOT "${CMAKE_SOURCE_DIR}/build/deps/clapack-wasm")
endif()

if(EXISTS "${OPENBLAS_WASM_ROOT}/libopenblas.a")
    message(STATUS "Using OpenBLAS/LAPACKE from ${OPENBLAS_WASM_ROOT}")
    target_include_directories(moonlab PRIVATE "${OPENBLAS_WASM_ROOT}/include")
    target_link_libraries(moonlab PRIVATE "${OPENBLAS_WASM_ROOT}/libopenblas.a")
    target_compile_definitions(moonlab PRIVATE QSIM_HAS_OPENBLAS=1)
elseif(EXISTS "${CLAPACK_WASM_ROOT}/lib/liblapack.a")
    message(STATUS "Using CLAPACK from ${CLAPACK_WASM_ROOT}")
    target_include_directories(moonlab PRIVATE "${CLAPACK_WASM_ROOT}/include")
    target_link_libraries(moonlab PRIVATE
        "${CLAPACK_WASM_ROOT}/lib/liblapack.a"
        "${CLAPACK_WASM_ROOT}/lib/libblas.a"
        "${CLAPACK_WASM_ROOT}/lib/libf2c.a"
    )
    target_compile_definitions(moonlab PRIVATE QSIM_HAS_CLAPACK=1)
else()
    message(FATAL_ERROR "BLAS/LAPACK not found. Provide OpenBLAS at ${OPENBLAS_WASM_ROOT} or CLAPACK at ${CLAPACK_WASM_ROOT}.")
endif()

# Include paths
target_include_directories(moonlab PRIVATE
    ${QSIM_ROOT}/src
    ${QSIM_ROOT}/src/quantum
    ${QSIM_ROOT}/src/algorithms
    ${QSIM_ROOT}/src/algorithms/tensor_network
    ${QSIM_ROOT}/src/utils
    ${QSIM_ROOT}/src/optimization
)

# Compiler flags
target_compile_options(moonlab PRIVATE
    -O3
    -ffast-math
    -DWASM_BUILD
    -Wno-unused-function
)

if(MOONLAB_WASM_ENABLE_WEBGPU)
    # EM_ASYNC_JS-based WebGPU dispatch uses async/await and requires Asyncify.
    target_link_options(moonlab PRIVATE "SHELL:-sASYNCIFY=1")
    target_link_options(moonlab PRIVATE "SHELL:-sASYNCIFY_STACK_SIZE=8388608")
endif()

# Read exported functions from file and format for Emscripten
# Emscripten expects EXPORTED_FUNCTIONS='["_func1","_func2",...]'
file(READ "${CMAKE_SOURCE_DIR}/exports.txt" EXPORTED_FUNCTIONS)
# Remove comment lines (lines starting with #)
string(REGEX REPLACE "#[^\n]*\n" "" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
# Remove empty lines and normalize whitespace
string(REGEX REPLACE "[ \t]*\n[ \t]*" "\n" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
string(REGEX REPLACE "\n+" "\n" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
# Trim leading/trailing whitespace and newlines
string(STRIP "${EXPORTED_FUNCTIONS}" EXPORTED_FUNCTIONS)
# Convert newlines to ","  (with quotes for JSON array)
string(REGEX REPLACE "\n" "\",\"" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
# Add quotes around the whole thing
set(EXPORTED_FUNCTIONS "\"${EXPORTED_FUNCTIONS}\"")

# Emscripten link flags
set_target_properties(moonlab PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS "\
        -s WASM=1 \
        -s MODULARIZE=1 \
        -s EXPORT_NAME='MoonlabModule' \
        -s EXPORTED_FUNCTIONS='[${EXPORTED_FUNCTIONS}]' \
        -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"getValue\",\"setValue\",\"UTF8ToString\",\"stringToUTF8\",\"HEAP8\",\"HEAPU8\",\"HEAP32\",\"HEAPU32\",\"HEAPF64\",\"addFunction\",\"removeFunction\"]' \
        -s ALLOW_TABLE_GROWTH=1 \
        -s ALLOW_MEMORY_GROWTH=1 \
        -s MAXIMUM_MEMORY=2GB \
        -s INITIAL_MEMORY=67108864 \
        -s STACK_SIZE=5242880 \
        -s NO_EXIT_RUNTIME=1 \
        -s FILESYSTEM=0 \
        -s ENVIRONMENT='web,worker,node' \
        -s SINGLE_FILE=0 \
        --pre-js ${CMAKE_SOURCE_DIR}/pre.js \
        --post-js ${CMAKE_SOURCE_DIR}/post.js \
    "
)

# Copy output to dist
add_custom_command(TARGET moonlab POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../dist
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moonlab> ${CMAKE_SOURCE_DIR}/../dist/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/moonlab.wasm ${CMAKE_SOURCE_DIR}/../dist/
    COMMENT "Copying WASM files to dist/"
)
