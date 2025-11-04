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
        LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
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
ifeq ($(shell uname),Darwin)
    ACCELERATE_FLAGS = -framework Accelerate
endif

LDFLAGS = -lm -lpthread -flto $(OPENMP_LIBS) $(ACCELERATE_FLAGS)

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
OPTIMIZATION_SRCS = $(wildcard $(OPTIMIZATION_DIR)/*.c)
UTILS_SRCS = $(wildcard $(UTILS_DIR)/*.c)
APPLICATIONS_SRCS = $(wildcard $(APPLICATIONS_DIR)/*.c)
PROFILER_SRCS = $(wildcard $(PROFILER_DIR)/*.c)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)

# Object files
QUANTUM_OBJS = $(QUANTUM_SRCS:.c=.o)
ALGORITHMS_OBJS = $(ALGORITHMS_SRCS:.c=.o)
OPTIMIZATION_OBJS = $(OPTIMIZATION_SRCS:.c=.o)
UTILS_OBJS = $(UTILS_SRCS:.c=.o)
APPLICATIONS_OBJS = $(APPLICATIONS_SRCS:.c=.o)
PROFILER_OBJS = $(PROFILER_SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Combined object files for complete library
ALL_LIB_OBJS = $(QUANTUM_OBJS) $(ALGORITHMS_OBJS) $(OPTIMIZATION_OBJS) $(UTILS_OBJS) $(APPLICATIONS_OBJS) $(PROFILER_OBJS)

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

ALL_EXAMPLES = $(EXAMPLE_HASH_COLLISION) \
               $(EXAMPLE_LARGE_SCALE) \
               $(EXAMPLE_OPTIMIZED) \
               $(EXAMPLE_PASSWORD) \
               $(EXAMPLE_METAL_BATCH) \
               $(EXAMPLE_METAL_GPU) \
               $(EXAMPLE_PHASE34) \
               $(GROVER_PARALLEL_BENCH)

# Phony targets
.PHONY: all clean test examples test_examples test_health test_thread_safety test_v3 showcase quantum_examples parallel_bench \
        hash_collision large_scale optimized password metal_batch metal_gpu phase34 metal_demos clean_examples

# Main targets
all: $(LIB) $(QSIM_TEST)
	@echo "Running Quantum Simulator v3.0 optimized tests..."
	LD_LIBRARY_PATH=. ./$(QSIM_TEST)

# Library builds
$(LIB): $(CORE_OBJS) $(ENTROPY_OBJS) $(HEALTH_OBJS) $(PROFILING_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# Health tests (NIST SP 800-90B compliance)
test_health: $(HEALTH_TESTS)
	@echo "Running NIST SP 800-90B health tests..."
	LD_LIBRARY_PATH=. ./$(HEALTH_TESTS)

$(HEALTH_TESTS): $(TEST_DIR)/health_tests_test.o $(HEALTH_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)


# Quantum Simulator v3 test
test_v3: $(QSIM_TEST)
	@echo "Running Quantum Simulator v3.0 tests..."
	LD_LIBRARY_PATH=. ./$(QSIM_TEST)

$(QSIM_TEST): $(TEST_DIR)/quantum_sim_test.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

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
	@echo "Examples cleaned"

# Clean
clean: clean_examples
	rm -f $(QSIM_TEST) $(GROVER_PARALLEL_BENCH)
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
$(OPTIMIZATION_OBJS): $(wildcard $(OPTIMIZATION_DIR)/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(UTILS_OBJS): $(wildcard $(UTILS_DIR)/*.h)
$(APPLICATIONS_OBJS): $(wildcard $(APPLICATIONS_DIR)/*.h) $(wildcard $(QUANTUM_DIR)/*.h)
$(PROFILER_OBJS): $(wildcard $(PROFILER_DIR)/*.h)
