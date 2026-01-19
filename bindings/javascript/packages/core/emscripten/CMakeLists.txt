cmake_minimum_required(VERSION 3.15)
project(moonlab_wasm C)

set(CMAKE_C_STANDARD 11)

# Root of quantum simulator source
set(QSIM_ROOT "${CMAKE_SOURCE_DIR}/../../../../../")

# Source files - Core quantum operations
set(QUANTUM_SOURCES
    ${QSIM_ROOT}/src/quantum/state.c
    ${QSIM_ROOT}/src/quantum/gates.c
    ${QSIM_ROOT}/src/quantum/measurement.c
    ${QSIM_ROOT}/src/quantum/entanglement.c
    ${QSIM_ROOT}/src/quantum/noise.c
)

# Source files - Algorithms
set(ALGORITHM_SOURCES
    ${QSIM_ROOT}/src/algorithms/grover.c
    ${QSIM_ROOT}/src/algorithms/grover_optimizer.c
    ${QSIM_ROOT}/src/algorithms/vqe.c
    ${QSIM_ROOT}/src/algorithms/qaoa.c
    ${QSIM_ROOT}/src/algorithms/bell_tests.c
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
)

# Source files - Utilities
set(UTILS_SOURCES
    ${QSIM_ROOT}/src/utils/entropy.c
    ${QSIM_ROOT}/src/utils/matrix_math.c
    ${QSIM_ROOT}/src/utils/config.c
)

# Source files - Optimization (SIMD operations with scalar fallbacks for WASM)
set(OPTIMIZATION_SOURCES
    ${QSIM_ROOT}/src/optimization/simd_ops.c
)

# Combine all sources
set(ALL_SOURCES
    ${QUANTUM_SOURCES}
    ${ALGORITHM_SOURCES}
    ${TENSOR_NETWORK_SOURCES}
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
        -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"getValue\",\"setValue\",\"UTF8ToString\",\"stringToUTF8\",\"HEAP8\",\"HEAPU8\",\"HEAP32\",\"HEAPU32\",\"HEAPF64\"]' \
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
