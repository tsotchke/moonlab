# Compiler and flags
CC = gcc
CXX = clang++

# AGGRESSIVE OPTIMIZATION (with numerical stability fixes)
# -Ofast: Maximum optimization
# -flto: Link-time optimization
# -march=native: Use all CPU features
# -ffast-math: Fast FP math (fixed gate code to handle this)
# -funroll-loops: Loop unrolling
CFLAGS = -Wall -Wextra -Ofast -march=native -ffast-math -funroll-loops -fPIC -I.
CFLAGS_METAL = -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops -fPIC -I.

# M2 ULTRA PARALLELIZATION - Phase 1: OpenMP 24-Core Support
OPENMP_FLAGS =
OPENMP_LIBS =

# Detect OpenMP support and configure appropriately
ifeq ($(shell uname),Darwin)
    # macOS: Check if we're using clang (default) or gcc
    CC_VERSION := $(shell $(CC) --version 2>/dev/null | head -n 1)

    # If using clang (default on macOS), use Xcode's OpenMP flags
    ifneq ($(findstring clang,$(CC_VERSION)),)
        # Clang on macOS: Use Homebrew libomp
        # Try multiple locations to handle different Homebrew installations:
        # 1. Apple Silicon native: /opt/homebrew
        # 2. Intel/Rosetta: /usr/local
        # 3. Custom: use brew --prefix libomp
        LIBOMP_PREFIX := $(shell \
            if [ -d "/opt/homebrew/opt/libomp/lib" ]; then \
                echo "/opt/homebrew/opt/libomp"; \
            elif [ -d "/usr/local/opt/libomp/lib" ]; then \
                echo "/usr/local/opt/libomp"; \
            else \
                brew --prefix libomp 2>/dev/null; \
            fi)
        ifneq ($(LIBOMP_PREFIX),)
            OPENMP_FLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
            OPENMP_LIBS = -L$(LIBOMP_PREFIX)/lib -lomp
            CFLAGS += $(OPENMP_FLAGS)
        endif
    else
        # Using actual GCC: standard OpenMP flags
        OPENMP_FLAGS = -fopenmp
        OPENMP_LIBS = -fopenmp
        CFLAGS += $(OPENMP_FLAGS)
    endif
else
    # Linux/other: standard GCC OpenMP
    OPENMP_FLAGS = -fopenmp
    OPENMP_LIBS = -fopenmp
    CFLAGS += $(OPENMP_FLAGS)
endif

# ARM-specific optimizations for ALL M-series (M1/M2/M3/M4/future)
ifeq ($(shell uname -m),arm64)
    # Query actual CPU model for detection
    APPLE_CPU := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    
    # Detect specific M-series for core count (no processor-specific tuning needed)
    ifneq (,$(findstring M1,$(APPLE_CPU)))
        DETECTED_CORES := $(shell sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)
        DETECTED_MODEL = M1
    else ifneq (,$(findstring M2,$(APPLE_CPU)))
        DETECTED_CORES := $(shell sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)
        DETECTED_MODEL = M2
    else ifneq (,$(findstring M3,$(APPLE_CPU)))
        DETECTED_CORES := $(shell sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)
        DETECTED_MODEL = M3
    else ifneq (,$(findstring M4,$(APPLE_CPU)))
        # M4: Use all physical cores (performance + efficiency) for maximum throughput
        DETECTED_CORES := $(shell sysctl -n hw.physicalcpu 2>/dev/null || echo 10)
        DETECTED_MODEL = M4
    else
        # Future M-series or unknown: detect from system
        DETECTED_CORES := $(shell sysctl -n hw.physicalcpu 2>/dev/null || echo 4)
        DETECTED_MODEL = Unknown
    endif
    
    # IMPORTANT: We intentionally do NOT add -mcpu or -mtune flags here!
    # The -march=native flag (already in CFLAGS) automatically optimizes
    # for the current CPU, which works for all M-series (M1/M2/M3/M4/future)
    # without requiring compiler support for specific model names.
    #
    # This approach:
    # - Works with any compiler version
    # - Automatically optimizes for M4 and future chips
    # - Avoids "unsupported argument" errors
    # - Lets the compiler detect best architecture flags
    
    # Set OpenMP thread count to detected performance cores
    export OMP_NUM_THREADS ?= $(DETECTED_CORES)
    export OMP_PROC_BIND ?= close
    export OMP_PLACES ?= cores
    
    $(info [BUILD] Detected Apple Silicon: $(DETECTED_MODEL))
    $(info [BUILD] Optimization: -march=native (auto-detects M-series features))
    $(info [BUILD] Using $(DETECTED_CORES) performance cores for OpenMP)
endif

# Phase 3: Accelerate framework support (macOS only)
ACCELERATE_FLAGS =
SECURITY_FLAGS =
METAL_FLAGS =
ifeq ($(shell uname),Darwin)
    ACCELERATE_FLAGS = -framework Accelerate
    SECURITY_FLAGS = -framework Security
    METAL_FLAGS = -framework Metal -framework Foundation
endif

LDFLAGS = -lm -lpthread -flto $(OPENMP_LIBS) $(ACCELERATE_FLAGS) $(SECURITY_FLAGS) $(METAL_FLAGS)

# Directory structure (MOONLAB REORGANIZATION)
QUANTUM_DIR = src/quantum
ALGORITHMS_DIR = src/algorithms
OPTIMIZATION_DIR = src/optimization
UTILS_DIR = src/utils
APPLICATIONS_DIR = src/applications
PROFILER_DIR = tools/profiler
TEST_DIR = tests
EXAMPLES_DIR = examples

# Include paths for new structure
CFLAGS += -Isrc

# Source files (new organization)
QUANTUM_SRCS = $(wildcard $(QUANTUM_DIR)/*.c)
ALGORITHMS_SRCS = $(wildcard $(ALGORITHMS_DIR)/*.c)
TENSOR_NETWORK_SRCS = $(wildcard $(ALGORITHMS_DIR)/tensor_network/*.c)
OPTIMIZATION_SRCS = $(wildcard $(OPTIMIZATION_DIR)/*.c)
UTILS_SRCS = $(wildcard $(UTILS_DIR)/*.c)
APPLICATIONS_SRCS = $(wildcard $(APPLICATIONS_DIR)/*.c)
PROFILER_SRCS = $(wildcard $(PROFILER_DIR)/*.c)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)

# Object files
QUANTUM_OBJS = $(QUANTUM_SRCS:.c=.o)
ALGORITHMS_OBJS = $(ALGORITHMS_SRCS:.c=.o)
TENSOR_NETWORK_OBJS = $(TENSOR_NETWORK_SRCS:.c=.o)
OPTIMIZATION_OBJS = $(OPTIMIZATION_SRCS:.c=.o)
UTILS_OBJS = $(UTILS_SRCS:.c=.o)
APPLICATIONS_OBJS = $(APPLICATIONS_SRCS:.c=.o)
PROFILER_OBJS = $(PROFILER_SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Combined object files for complete library
ALL_LIB_OBJS = $(QUANTUM_OBJS) $(ALGORITHMS_OBJS) $(TENSOR_NETWORK_OBJS) $(OPTIMIZATION_OBJS) $(UTILS_OBJS) $(APPLICATIONS_OBJS) $(PROFILER_OBJS)

# Add Metal GPU support on macOS
ifeq ($(shell uname),Darwin)
ALL_LIB_OBJS += $(OPTIMIZATION_DIR)/gpu_metal.o
endif

# Output files
LIB = libquantumsim.so
QSIM_TEST = qsim_test
GROVER_PARALLEL_BENCH = grover_parallel_benchmark

# Example executables
EXAMPLE_HASH_COLLISION = examples/quantum/grover_hash_collision
EXAMPLE_LARGE_SCALE = examples/quantum/grover_large_scale_demo
EXAMPLE_OPTIMIZED = examples/quantum/grover_large_scale_optimized
EXAMPLE_PASSWORD = examples/quantum/grover_password_crack
EXAMPLE_METAL_BATCH = examples/quantum/metal_batch_benchmark
EXAMPLE_METAL_GPU = examples/quantum/metal_gpu_benchmark
EXAMPLE_PHASE34 = examples/quantum/phase3_phase4_benchmark
EXAMPLE_VQE_H2 = examples/applications/vqe_h2_molecule
EXAMPLE_QAOA_MAXCUT = examples/applications/qaoa_maxcut
EXAMPLE_PORTFOLIO = examples/applications/portfolio_optimization
EXAMPLE_TSP = examples/applications/tsp_logistics
EXAMPLE_SPIN_CHAIN = examples/tensor_network/quantum_spin_chain
EXAMPLE_CRITICAL_POINT = examples/tensor_network/quantum_critical_point

ALL_EXAMPLES = $(EXAMPLE_HASH_COLLISION) \
               $(EXAMPLE_LARGE_SCALE) \
               $(EXAMPLE_OPTIMIZED) \
               $(EXAMPLE_PASSWORD) \
               $(EXAMPLE_METAL_BATCH) \
               $(EXAMPLE_METAL_GPU) \
               $(EXAMPLE_PHASE34) \
               $(GROVER_PARALLEL_BENCH) \
               $(EXAMPLE_VQE_H2) \
               $(EXAMPLE_QAOA_MAXCUT) \
               $(EXAMPLE_PORTFOLIO) \
               $(EXAMPLE_TSP) \
               $(EXAMPLE_SPIN_CHAIN) \
               $(EXAMPLE_CRITICAL_POINT)

# Test executables
HEALTH_TESTS = tests/health_tests_test
BELL_TEST_DEMO = tests/bell_test_demo
GATE_TEST = tests/gate_test
CORRELATION_TEST = tests/correlation_test

# Unit tests
UNIT_TEST_STATE = tests/unit/test_quantum_state
UNIT_TEST_GATES = tests/unit/test_quantum_gates
UNIT_TEST_MEMORY_ALIGN = tests/unit/test_memory_align
UNIT_TEST_SIMD_DISPATCH = tests/unit/test_simd_dispatch
UNIT_TEST_STRIDE_GATES = tests/unit/test_stride_gates
UNIT_TEST_TENSOR_NETWORK = tests/unit/test_tensor_network

ALL_TESTS = $(QSIM_TEST) $(HEALTH_TESTS) $(BELL_TEST_DEMO) $(GATE_TEST) $(CORRELATION_TEST) \
            $(UNIT_TEST_STATE) $(UNIT_TEST_GATES) $(UNIT_TEST_MEMORY_ALIGN) $(UNIT_TEST_SIMD_DISPATCH) \
            $(UNIT_TEST_STRIDE_GATES) $(UNIT_TEST_TENSOR_NETWORK)

# Phony targets
.PHONY: all clean test tests examples test_examples test_health test_v3 showcase quantum_examples parallel_bench \
        hash_collision large_scale optimized password metal_batch metal_gpu phase34 metal_demos clean_examples \
        test_bell test_gate test_correlation clean_tests unit_tests test_unit coverage

# Main targets
all: $(LIB) $(QSIM_TEST)
	@echo "Running Quantum Simulator v3.0 optimized tests..."
	LD_LIBRARY_PATH=. ./$(QSIM_TEST)

# Build library
$(LIB): $(ALL_LIB_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# ============================================================================
# TEST TARGETS
# ============================================================================

# Build all tests
tests: $(ALL_TESTS)
	@echo "All tests built successfully"

# Quantum Simulator v3 test
test_v3: $(QSIM_TEST)
	@echo "Running Quantum Simulator v3.0 tests..."
	LD_LIBRARY_PATH=. ./$(QSIM_TEST)

$(QSIM_TEST): $(TEST_DIR)/quantum_sim_test.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Health tests (NIST SP 800-90B compliance)
test_health: $(HEALTH_TESTS)
	@echo "Running NIST SP 800-90B health tests..."
	LD_LIBRARY_PATH=. ./$(HEALTH_TESTS)

$(HEALTH_TESTS): $(TEST_DIR)/health_tests_test.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Bell test demonstration
test_bell: $(BELL_TEST_DEMO)
	@echo "Running Bell inequality test demonstration..."
	LD_LIBRARY_PATH=. ./$(BELL_TEST_DEMO)

$(BELL_TEST_DEMO): $(TEST_DIR)/bell_test_demo.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Gate correctness test
test_gate: $(GATE_TEST)
	@echo "Running quantum gate correctness tests..."
	LD_LIBRARY_PATH=. ./$(GATE_TEST)

$(GATE_TEST): $(TEST_DIR)/gate_test.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Correlation test
test_correlation: $(CORRELATION_TEST)
	@echo "Running entanglement correlation tests..."
	LD_LIBRARY_PATH=. ./$(CORRELATION_TEST)

$(CORRELATION_TEST): $(TEST_DIR)/correlation_test.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Unit tests
unit_tests: $(UNIT_TEST_STATE) $(UNIT_TEST_GATES) $(UNIT_TEST_MEMORY_ALIGN) $(UNIT_TEST_SIMD_DISPATCH) $(UNIT_TEST_STRIDE_GATES) $(UNIT_TEST_TENSOR_NETWORK)
	@echo "All unit tests built successfully"

test_unit: unit_tests
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║              RUNNING UNIT TEST SUITE                      ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "=== Memory Alignment Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_MEMORY_ALIGN)
	@echo ""
	@echo "=== SIMD Dispatch Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_SIMD_DISPATCH)
	@echo ""
	@echo "=== Stride Gates Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_STRIDE_GATES)
	@echo ""
	@echo "=== Quantum State Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_STATE)
	@echo ""
	@echo "=== Quantum Gates Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_GATES)
	@echo ""
	@echo "=== Tensor Network Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_TENSOR_NETWORK)
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║         UNIT TESTS COMPLETED                              ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"

$(UNIT_TEST_STATE): $(TEST_DIR)/unit/test_quantum_state.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(UNIT_TEST_GATES): $(TEST_DIR)/unit/test_quantum_gates.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(UNIT_TEST_MEMORY_ALIGN): $(TEST_DIR)/unit/test_memory_align.o $(OPTIMIZATION_DIR)/memory_align.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(UNIT_TEST_SIMD_DISPATCH): $(TEST_DIR)/unit/test_simd_dispatch.o $(OPTIMIZATION_DIR)/simd_dispatch.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(UNIT_TEST_STRIDE_GATES): $(TEST_DIR)/unit/test_stride_gates.o $(OPTIMIZATION_DIR)/stride_gates.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(UNIT_TEST_TENSOR_NETWORK): $(TEST_DIR)/unit/test_tensor_network.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Run all tests
test: tests
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║         RUNNING ALL QUANTUM SIMULATOR TESTS               ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "=== Quantum RNG v3.0 Tests ==="
	LD_LIBRARY_PATH=. ./$(QSIM_TEST)
	@echo ""
	@echo "=== Health Tests ==="
	LD_LIBRARY_PATH=. ./$(HEALTH_TESTS)
	@echo ""
	@echo "=== Bell Test Demo ==="
	-LD_LIBRARY_PATH=. ./$(BELL_TEST_DEMO)
	@echo ""
	@echo "=== Gate Tests ==="
	LD_LIBRARY_PATH=. ./$(GATE_TEST)
	@echo ""
	@echo "=== Correlation Tests ==="
	LD_LIBRARY_PATH=. ./$(CORRELATION_TEST)
	@echo ""
	@echo "=== Unit Tests ==="
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_MEMORY_ALIGN)
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_SIMD_DISPATCH)
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_STRIDE_GATES)
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_STATE)
	LD_LIBRARY_PATH=. ./$(UNIT_TEST_GATES)
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║         ALL TESTS COMPLETED                               ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"

# Parallel Grover benchmark (M2 Ultra Phase 1)
parallel_bench: $(GROVER_PARALLEL_BENCH)
	@echo "Running M2 Ultra parallel Grover benchmark..."
	@echo "This tests 24-core parallelization targeting 20-30x speedup"
	LD_LIBRARY_PATH=. ./$(GROVER_PARALLEL_BENCH)

$(GROVER_PARALLEL_BENCH): $(EXAMPLES_DIR)/quantum/grover_parallel_benchmark.o $(OPTIMIZATION_DIR)/parallel_ops.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# ============================================================================
# QUANTUM EXAMPLES - All 8 demonstrations
# ============================================================================

# Build all examples
examples: $(ALL_EXAMPLES)
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║         ALL QUANTUM EXAMPLES BUILT SUCCESSFULLY           ║"
	@echo "╠═══════════════════════════════════════════════════════════╣"
	@echo "║                                                           ║"
	@echo "║  Run examples:                                            ║"
	@echo "║    ./$(EXAMPLE_HASH_COLLISION)                            ║"
	@echo "║    ./$(EXAMPLE_LARGE_SCALE)                               ║"
	@echo "║    ./$(EXAMPLE_OPTIMIZED)                                 ║"
	@echo "║    ./$(EXAMPLE_PASSWORD)                                  ║"
	@echo "║    ./$(EXAMPLE_METAL_BATCH)                               ║"
	@echo "║    ./$(EXAMPLE_METAL_GPU)                                 ║"
	@echo "║    ./$(EXAMPLE_PHASE34)                                   ║"
	@echo "║    ./$(GROVER_PARALLEL_BENCH)                             ║"
	@echo "║    ./$(EXAMPLE_VQE_H2)                                    ║"
	@echo "║    ./$(EXAMPLE_QAOA_MAXCUT)                               ║"
	@echo "║    ./$(EXAMPLE_PORTFOLIO)                                 ║"
	@echo "║    ./$(EXAMPLE_TSP)                                       ║"
	@echo "║                                                           ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""

# Individual example targets
hash_collision: $(EXAMPLE_HASH_COLLISION)
	@echo "Built: $<"

large_scale: $(EXAMPLE_LARGE_SCALE)
	@echo "Built: $<"

optimized: $(EXAMPLE_OPTIMIZED)
	@echo "Built: $<"

password: $(EXAMPLE_PASSWORD)
	@echo "Built: $<"

metal_batch: $(EXAMPLE_METAL_BATCH)
	@echo "Built: $<"

metal_gpu: $(EXAMPLE_METAL_GPU)
	@echo "Built: $<"

phase34: $(EXAMPLE_PHASE34)
	@echo "Built: $<"

# Build only Metal GPU examples
metal_demos: $(EXAMPLE_METAL_BATCH) $(EXAMPLE_METAL_GPU)
	@echo "Metal GPU examples built successfully"

# Standard C examples (no Metal)
$(EXAMPLE_HASH_COLLISION): $(EXAMPLES_DIR)/quantum/grover_hash_collision.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_LARGE_SCALE): $(EXAMPLES_DIR)/quantum/grover_large_scale_demo.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_OPTIMIZED): $(EXAMPLES_DIR)/quantum/grover_large_scale_optimized.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_PASSWORD): $(EXAMPLES_DIR)/quantum/grover_password_crack.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_PHASE34): $(EXAMPLES_DIR)/quantum/phase3_phase4_benchmark.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# VQE example
$(EXAMPLE_VQE_H2): $(EXAMPLES_DIR)/applications/vqe_h2_molecule.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# QAOA examples
$(EXAMPLE_QAOA_MAXCUT): $(EXAMPLES_DIR)/applications/qaoa_maxcut.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Portfolio optimization example
$(EXAMPLE_PORTFOLIO): $(EXAMPLES_DIR)/applications/portfolio_optimization.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# TSP logistics example
$(EXAMPLE_TSP): $(EXAMPLES_DIR)/applications/tsp_logistics.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Tensor network spin chain example
$(EXAMPLE_SPIN_CHAIN): $(EXAMPLES_DIR)/tensor_network/quantum_spin_chain.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Run tensor network spin chain demo
spin_chain: $(EXAMPLE_SPIN_CHAIN)
	@echo "Running 100-qubit quantum spin chain simulation..."
	LD_LIBRARY_PATH=. ./$(EXAMPLE_SPIN_CHAIN)

# Quantum critical point demo (200 qubits, CFT verification)
$(EXAMPLE_CRITICAL_POINT): $(EXAMPLES_DIR)/tensor_network/quantum_critical_point.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Run quantum critical point demo
critical_point: $(EXAMPLE_CRITICAL_POINT)
	@echo "Running 200-qubit quantum critical point analysis..."
	LD_LIBRARY_PATH=. ./$(EXAMPLE_CRITICAL_POINT)

# Metal examples (require Objective-C++ and Metal framework)
# Build with gcc (standard linker) to avoid clang++/gcc LTO incompatibility
$(EXAMPLE_METAL_BATCH): $(EXAMPLES_DIR)/quantum/metal_batch_benchmark.o $(OPTIMIZATION_DIR)/gpu_metal.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) -framework Metal -framework Foundation -framework CoreGraphics -lstdc++

$(EXAMPLE_METAL_GPU): $(EXAMPLES_DIR)/quantum/metal_gpu_benchmark.o $(OPTIMIZATION_DIR)/gpu_metal.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) -framework Metal -framework Foundation -framework CoreGraphics -lstdc++

# Compile Metal bridge (Objective-C++) - use CFLAGS_METAL (no LTO) for compatibility
$(OPTIMIZATION_DIR)/gpu_metal.o: $(OPTIMIZATION_DIR)/gpu_metal.mm
	clang++ -c -o $@ $< $(CFLAGS_METAL) $(OPENMP_FLAGS) -framework Metal -framework Foundation

# Clean examples
clean_examples:
	rm -f $(ALL_EXAMPLES)
	rm -f $(EXAMPLES_DIR)/quantum/*.o
	rm -f $(EXAMPLES_DIR)/applications/*.o
	@echo "Examples cleaned"

# Clean tests
clean_tests:
	rm -f $(ALL_TESTS)
	rm -f $(TEST_DIR)/*.o
	rm -f $(TEST_DIR)/unit/*.o
	@echo "Tests cleaned"

# Test coverage (requires gcov)
coverage:
	@echo "Building with coverage instrumentation..."
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS) --coverage" LDFLAGS="$(LDFLAGS) --coverage" tests
	@echo "Running tests..."
	@$(MAKE) test
	@echo "Generating coverage report..."
	@gcov src/quantum/*.c src/algorithms/*.c
	@echo "Coverage data generated. Use 'lcov' for HTML reports."

# Clean all
clean: clean_examples clean_tests
	rm -f $(GROVER_PARALLEL_BENCH)
	find . -name "*.o" -delete
	find . -name "*.so" -delete
	@echo "Build artifacts cleaned"

# Dependencies
%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Core dependencies (updated for new structure)
$(QUANTUM_OBJS): $(wildcard $(QUANTUM_DIR)/*.h)
$(ALGORITHMS_OBJS): $(wildcard $(ALGORITHMS_DIR)/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(TENSOR_NETWORK_OBJS): $(wildcard $(ALGORITHMS_DIR)/tensor_network/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(OPTIMIZATION_OBJS): $(wildcard $(OPTIMIZATION_DIR)/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(UTILS_OBJS): $(wildcard $(UTILS_DIR)/*.h)
$(APPLICATIONS_OBJS): $(wildcard $(APPLICATIONS_DIR)/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(PROFILER_OBJS): $(wildcard $(PROFILER_DIR)/*.h)
