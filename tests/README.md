# Moonlab Quantum Simulator - Test Suite

Comprehensive test infrastructure for production-ready quantum simulation.

## Overview

The Moonlab test suite provides extensive coverage across:
- **Unit Tests**: Core quantum operations (state management, gates)
- **Integration Tests**: Full algorithm implementations (VQE, QAOA, QPE, Grover)
- **System Tests**: QRNG, Bell tests, health checks
- **Python Bindings**: Cross-platform Python interface testing
- **Performance Tests**: Benchmarks and profiling

**Current Coverage**: Target 80%+ (automated reporting via CI/CD)

## Quick Start

### Run All Tests
```bash
make test
```

### Run Unit Tests Only
```bash
make test_unit
```

### Generate Coverage Report
```bash
./tools/test_coverage.sh
```
Opens HTML coverage report in browser (macOS) showing line-by-line coverage.

## Test Structure

```
tests/
├── unit/                          # Unit tests (NEW)
│   ├── test_quantum_state.c      # State management (25+ tests)
│   └── test_quantum_gates.c      # Gate operations (24+ tests)
├── integration/                   # Integration tests (planned)
│   ├── test_vqe.c
│   ├── test_qaoa.c
│   └── test_qpe.c
├── quantum_sim_test.c            # Comprehensive QRNG tests
├── health_tests_test.c           # NIST SP 800-90B compliance
├── bell_test_demo.c              # Bell inequality verification
├── gate_test.c                   # Basic gate correctness
└── correlation_test.c            # Entanglement correlation
```

## Unit Tests

### Quantum State Management (`test_quantum_state.c`)
**Coverage**: 80%+ of [`state.c`](../src/quantum/state.c)

Tests include:
- ✅ Initialization & memory management (various sizes, edge cases)
- ✅ State cloning & reset
- ✅ Normalization checking & enforcement
- ✅ Entropy, purity, fidelity calculations
- ✅ Entanglement measures (Bell states, product states)
- ✅ Measurement history tracking
- ✅ Error handling (NULL pointers, invalid parameters)
- ✅ Large state support (16 qubits, 65K dimensions)

**Run**:
```bash
./tests/unit/test_quantum_state
```

### Quantum Gates (`test_quantum_gates.c`)
**Coverage**: 80%+ of [`gates.c`](../src/quantum/gates.c)

Tests include:
- ✅ Pauli gates (X, Y, Z)
- ✅ Hadamard (superposition creation)
- ✅ Phase gates (S, T, custom phases)
- ✅ Rotation gates (RX, RY, RZ)
- ✅ Two-qubit gates (CNOT, CZ, SWAP)
- ✅ Three-qubit gates (Toffoli, Fredkin)
- ✅ Bell state preparation (H + CNOT)
- ✅ GHZ state (3-qubit entanglement)
- ✅ Gate properties (unitarity, reversibility)
- ✅ Error handling

**Run**:
```bash
./tests/unit/test_quantum_gates
```

## Integration Tests

### QRNG System Test (`quantum_sim_test.c`)
Comprehensive Quantum RNG v3.0 validation:
- Quantum engine integration
- Layered entropy architecture
- Bell tests (10-20x faster)
- Grover sampling APIs
- Continuous monitoring
- ARM hardware entropy

**Run**:
```bash
./qsim_test
```

### Health Tests (`health_tests_test.c`)
NIST SP 800-90B compliance:
- Repetition Count Test (RCT)
- Adaptive Proportion Test (APT)
- Startup validation
- Statistical analysis

**Run**:
```bash
./tests/health_tests_test
```

### Bell Inequality Test (`bell_test_demo.c`)
Proves genuine quantum behavior:
- CHSH test execution
- Bell inequality violation
- Entanglement verification
- Expected CHSH ≈ 2.828

**Run**:
```bash
./tests/bell_test_demo
```

## CI/CD Pipeline

### GitHub Actions Workflows
Located in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

**Automated on every push/PR:**

1. **macOS Build & Test** (primary platform)
   - Build with Apple Silicon optimizations
   - Run all unit + integration tests
   - Generate coverage reports
   - Upload to Codecov

2. **Linux Build & Test**
   - Multi-distro validation
   - Valgrind memory leak detection
   - Performance benchmarks

3. **Python Bindings Test**
   - Python 3.8, 3.9, 3.10, 3.11
   - Cross-platform validation
   - PyTorch integration tests

4. **Code Quality**
   - cppcheck static analysis
   - clang-tidy checks
   - Security vulnerability scanning

### Running CI Locally

**Full test suite**:
```bash
make clean && make && make test
```

**With coverage**:
```bash
./tools/test_coverage.sh
```

**Memory leak check** (requires Valgrind):
```bash
valgrind --leak-check=full ./tests/unit/test_quantum_state
```

## Test Coverage Goals

| Module | Current | Target | Status |
|--------|---------|--------|--------|
| Quantum State | 80%+ | 80% | ✅ |
| Quantum Gates | 80%+ | 80% | ✅ |
| VQE Algorithm | - | 80% | 🔄 In Progress |
| QAOA Algorithm | - | 80% | 🔄 In Progress |
| QPE Algorithm | - | 80% | 🔄 In Progress |
| Grover Algorithm | - | 80% | 🔄 In Progress |
| Python Bindings | - | 70% | 📋 Planned |

## Writing New Tests

### Unit Test Template

```c
#include "../../src/module/header.h"
#include <stdio.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("[%d] Testing: %s... ", tests_run, name); \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("✓ PASS\n"); \
        return 1; \
    } while(0)

int test_feature_name() {
    TEST_START("Feature description");
    
    // Setup
    // ...
    
    // Test assertions
    assert(condition);
    
    // Cleanup
    // ...
    
    TEST_PASS();
}

int main(void) {
    test_feature_name();
    // More tests...
    
    printf("Tests: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
```

### Add to Build System

Edit [`Makefile`](../Makefile):
```makefile
# Add test executable
NEW_TEST = tests/unit/test_new_feature

# Add to ALL_TESTS
ALL_TESTS = ... $(NEW_TEST)

# Build rule
$(NEW_TEST): $(TEST_DIR)/unit/test_new_feature.o $(ALL_LIB_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
```

## Performance Testing

### Benchmarks
```bash
# Run performance benchmarks
make benchmarks

# Specific benchmarks
./examples/applications/vqe_h2_molecule
./examples/applications/qaoa_maxcut
./examples/quantum/grover_parallel_benchmark
```

### Profiling
```bash
# Build with profiling
make CFLAGS="-pg" clean all

# Run and generate gprof report
./tests/unit/test_quantum_state
gprof ./tests/unit/test_quantum_state gmon.out > analysis.txt
```

## Continuous Improvement

### Coverage Targets by Release

- **v1.0 (March 2026)**: 80% core modules
- **v1.1**: 85% including algorithms
- **v2.0**: 90% full system

### Quality Metrics

- ✅ All tests must pass before merge
- ✅ No memory leaks (Valgrind clean)
- ✅ Coverage maintained/improved
- ✅ Performance benchmarks stable
- ✅ Documentation updated

## Troubleshooting

### Build Issues
```bash
# Clean rebuild
make clean && make tests unit_tests

# Check dependencies
brew install libomp lcov  # macOS
apt install libomp-dev lcov  # Linux
```

### Test Failures
1. Check error messages in test output
2. Run specific test: `./tests/unit/test_name`
3. Use debugger: `lldb ./tests/unit/test_name`
4. Check CI logs for platform-specific issues

### Coverage Issues
```bash
# Ensure gcov is installed
gcov --version

# Manual coverage check
./tools/test_coverage.sh

# View detailed HTML report
open coverage_report/index.html
```

## Contributing

1. Write tests for all new features
2. Maintain 80%+ coverage
3. Run full test suite before committing
4. Add tests to CI/CD pipeline
5. Document test purpose and expectations

## Resources

- [Google Test Best Practices](https://google.github.io/googletest/)
- [NIST SP 800-90B](https://csrc.nist.gov/publications/detail/sp/800-90b/final)
- [Quantum Testing Strategies](../docs/testing-strategies.md)
- [CI/CD Documentation](../.github/workflows/README.md)

---

**Questions?** See [`docs/DEVELOPMENT_STATUS.md`](../docs/DEVELOPMENT_STATUS.md) or open an issue.