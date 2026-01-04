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
)

# Source files - Algorithms
set(ALGORITHM_SOURCES
    ${QSIM_ROOT}/src/algorithms/grover.c
    ${QSIM_ROOT}/src/algorithms/grover_optimizer.c
    ${QSIM_ROOT}/src/algorithms/vqe.c
    ${QSIM_ROOT}/src/algorithms/qaoa.c
    ${QSIM_ROOT}/src/algorithms/bell_tests.c
)

# Source files - Utilities
set(UTILS_SOURCES
    ${QSIM_ROOT}/src/utils/entropy.c
    ${QSIM_ROOT}/src/utils/matrix_math.c
)

# Combine all sources
set(ALL_SOURCES
    ${QUANTUM_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILS_SOURCES}
)

# Create the WASM executable
add_executable(moonlab ${ALL_SOURCES})

# Include paths
target_include_directories(moonlab PRIVATE
    ${QSIM_ROOT}/src
    ${QSIM_ROOT}/src/quantum
    ${QSIM_ROOT}/src/algorithms
    ${QSIM_ROOT}/src/utils
)

# Compiler flags
target_compile_options(moonlab PRIVATE
    -O3
    -ffast-math
    -DWASM_BUILD
    -Wno-unused-function
)

# Read exported functions from file
file(READ "${CMAKE_SOURCE_DIR}/exports.txt" EXPORTED_FUNCTIONS)
string(REGEX REPLACE "\n" "," EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
string(REGEX REPLACE ",$" "" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
string(REGEX REPLACE "#[^\n]*," "" EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")

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
