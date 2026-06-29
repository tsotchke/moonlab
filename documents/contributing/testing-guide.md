# Archived Moonlab Documentation: Testing Guide

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Testing Guide

How to write and run tests for Moonlab.

## Overview

Moonlab uses a multi-tier testing strategy:

1. **Unit Tests**: Test individual functions and components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Verify performance requirements
4. **Fuzz Tests**: Find edge cases and crashes

## Running Tests

### Quick Start

[archived fence delimiter: ```bash]
# Build with tests
cmake -DBUILD_TESTS=ON ..
make

# Run all tests
make test

# Or use ctest directly
ctest --output-on-failure
[archived fence delimiter: ```]

### Specific Test Suites

[archived fence delimiter: ```bash]
# C tests
./bin/test_quantum_gates
./bin/test_quantum_state
./bin/test_algorithms

# Python tests
pytest bindings/python/tests/

# Rust tests
cd bindings/rust/moonlab && cargo test
[archived fence delimiter: ```]

### Verbose Output

[archived fence delimiter: ```bash]
# CTest verbose
ctest --verbose

# Single test verbose
./bin/test_quantum_gates --verbose

# Python with details
pytest -v bindings/python/tests/
[archived fence delimiter: ```]

## Test Organization

### Directory Structure

[archived fence delimiter: ```]
tests/
├── unit/
│   ├── test_quantum_state.c
│   ├── test_quantum_gates.c
│   ├── test_measurement.c
│   ├── test_entanglement.c
│   └── test_tensor.c
├── integration/
│   ├── test_grover.c
│   ├── test_vqe.c
│   └── test_dmrg.c
├── performance/
│   ├── bench_gates.c
│   └── bench_algorithms.c
└── fuzz/
    └── fuzz_state.c

bindings/python/tests/
├── test_core.py
├── test_algorithms.py
└── test_ml.py

bindings/rust/moonlab/tests/
└── integration_tests.rs
[archived fence delimiter: ```]

## Writing C Tests

### Test Framework

We use a minimal custom test framework. Basic structure:

[archived fence delimiter: ```c]
#include "test_framework.h"
#include "quantum_sim.h"

/**
 * Test quantum state creation.
 */
TEST(test_state_create) {
    quantum_state_t* state = quantum_state_init(4);

    ASSERT_NOT_NULL(state);
    ASSERT_EQ(quantum_state_num_qubits(state), 4);

    // State should be initialized to |0000⟩
    double complex* amps = quantum_state_get_amplitudes(state);
    ASSERT_COMPLEX_EQ(amps[0], 1.0 + 0.0 * I);
    for (int i = 1; i < 16; i++) {
        ASSERT_COMPLEX_EQ(amps[i], 0.0 + 0.0 * I);
    }

    quantum_state_free(state);
}

/**
 * Test Hadamard gate.
 */
TEST(test_hadamard_gate) {
    quantum_state_t* state = quantum_state_init(1);

    gate_hadamard(state, 0);

    double complex* amps = quantum_state_get_amplitudes(state);
    double inv_sqrt2 = 1.0 / sqrt(2.0);

    ASSERT_COMPLEX_NEAR(amps[0], inv_sqrt2, 1e-10);
    ASSERT_COMPLEX_NEAR(amps[1], inv_sqrt2, 1e-10);

    quantum_state_free(state);
}

/**
 * Main test runner.
 */
int main(void) {
    TEST_INIT();

    RUN_TEST(test_state_create);
    RUN_TEST(test_hadamard_gate);

    TEST_SUMMARY();
    return TEST_RESULT();
}
[archived fence delimiter: ```]

### Assertion Macros

| Macro | Description |
|-------|-------------|
| `ASSERT_TRUE(expr)` | Expression is true |
| `ASSERT_FALSE(expr)` | Expression is false |
| `ASSERT_EQ(a, b)` | Values are equal |
| `ASSERT_NE(a, b)` | Values are not equal |
| `ASSERT_NULL(ptr)` | Pointer is NULL |
| `ASSERT_NOT_NULL(ptr)` | Pointer is not NULL |
| `ASSERT_NEAR(a, b, tol)` | Values within tolerance |
| `ASSERT_COMPLEX_EQ(a, b)` | Complex values equal |
| `ASSERT_COMPLEX_NEAR(a, b, tol)` | Complex values within tolerance |
| `ASSERT_ARRAY_EQ(a, b, n)` | Arrays equal |

### Testing Gates

[archived fence delimiter: ```c]
/**
 * Verify gate unitarity.
 */
TEST(test_gate_unitary) {
    // Create superposition
    quantum_state_t* state = quantum_state_init(2);
    gate_hadamard(state, 0);
    gate_hadamard(state, 1);

    double norm_before = quantum_state_norm(state);

    // Apply various gates
    gate_pauli_x(state, 0);
    gate_pauli_y(state, 1);
    gate_cnot(state, 0, 1);
    gate_t(state, 0);

    double norm_after = quantum_state_norm(state);

    // Norm should be preserved
    ASSERT_NEAR(norm_before, norm_after, 1e-10);

    quantum_state_free(state);
}

/**
 * Verify gate inverse.
 */
TEST(test_gate_inverse) {
    quantum_state_t* state = quantum_state_init(2);

    // Random initial state
    gate_rx(state, 0, 1.234);
    gate_ry(state, 1, 2.345);
    gate_cnot(state, 0, 1);

    // Save amplitudes
    double complex* initial = copy_amplitudes(state);

    // Apply and unapply gate
    gate_hadamard(state, 0);
    gate_hadamard(state, 0);  // H is self-inverse

    // Should return to initial state
    double complex* final = quantum_state_get_amplitudes(state);
    for (int i = 0; i < 4; i++) {
        ASSERT_COMPLEX_NEAR(initial[i], final[i], 1e-10);
    }

    free(initial);
    quantum_state_free(state);
}
[archived fence delimiter: ```]

### Testing Algorithms

[archived fence delimiter: ```c]
/**
 * Test Grover's algorithm finds correct answer.
 */
TEST(test_grover_search) {
    int n_qubits = 4;
    int target = 7;

    // Run multiple trials
    int successes = 0;
    int trials = 100;

    for (int t = 0; t < trials; t++) {
        grover_solver_t* solver = grover_create(n_qubits);
        grover_set_target(solver, target);

        grover_result_t result = grover_search(solver);

        if (result.found == target) {
            successes++;
        }

        grover_destroy(solver);
    }

    // Should succeed most of the time (theoretical ~96%)
    double success_rate = (double)successes / trials;
    ASSERT_TRUE(success_rate > 0.9);
}
[archived fence delimiter: ```]

## Writing Python Tests

### Test Structure

[archived fence delimiter: ```python]
"""Tests for quantum state operations."""

import pytest
import numpy as np
from moonlab import QuantumState

class TestQuantumState:
    """Tests for QuantumState class."""

    def test_create(self):
        """Test state creation."""
        state = QuantumState(4)

        assert state.num_qubits == 4
        assert len(state.amplitudes) == 16
        assert state.amplitudes[0] == 1.0
        assert all(state.amplitudes[1:] == 0)

    def test_hadamard(self):
        """Test Hadamard gate."""
        state = QuantumState(1)
        state.h(0)

        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_allclose(state.amplitudes, expected, rtol=1e-10)

    def test_bell_state(self):
        """Test Bell state creation."""
        state = QuantumState(2)
        state.h(0)
        state.cnot(0, 1)

        # Only |00⟩ and |11⟩ should have amplitude
        probs = state.probabilities
        assert pytest.approx(probs[0], abs=1e-10) == 0.5  # |00⟩
        assert pytest.approx(probs[1], abs=1e-10) == 0.0  # |01⟩
        assert pytest.approx(probs[2], abs=1e-10) == 0.0  # |10⟩
        assert pytest.approx(probs[3], abs=1e-10) == 0.5  # |11⟩

    def test_normalization(self):
        """Test state remains normalized after operations."""
        state = QuantumState(4)

        # Apply random circuit
        for i in range(4):
            state.h(i)
        for i in range(3):
            state.cnot(i, i+1)
        for i in range(4):
            state.rx(i, np.random.uniform(0, 2*np.pi))

        assert pytest.approx(sum(state.probabilities), abs=1e-10) == 1.0


class TestMeasurement:
    """Tests for measurement operations."""

    def test_measure_deterministic(self):
        """Test measurement of computational basis state."""
        state = QuantumState(2)
        state.x(0)
        state.x(1)  # Prepare |11⟩

        result = state.measure_all()
        assert result == 3  # Binary 11

    def test_measure_statistics(self):
        """Test measurement follows Born rule."""
        state = QuantumState(1)
        state.h(0)

        # Sample many times
        counts = {0: 0, 1: 0}
        for _ in range(10000):
            s = QuantumState(1)
            s.h(0)
            result = s.measure(0)
            counts[result] += 1

        # Should be approximately 50/50
        assert 4500 < counts[0] < 5500
        assert 4500 < counts[1] < 5500
[archived fence delimiter: ```]

### Fixtures

[archived fence delimiter: ```python]
@pytest.fixture
def bell_state():
    """Create a Bell state for testing."""
    state = QuantumState(2)
    state.h(0)
    state.cnot(0, 1)
    return state

@pytest.fixture
def ghz_state():
    """Create a GHZ state for testing."""
    state = QuantumState(3)
    state.h(0)
    state.cnot(0, 1)
    state.cnot(0, 2)
    return state

def test_entanglement_entropy(bell_state):
    """Test entanglement entropy of Bell state."""
    entropy = bell_state.entanglement_entropy(qubit=0)
    assert pytest.approx(entropy, abs=0.01) == 1.0
[archived fence delimiter: ```]

### Parametrized Tests

[archived fence delimiter: ```python]
@pytest.mark.parametrize("n_qubits", [1, 2, 4, 8, 10])
def test_state_size(n_qubits):
    """Test states of various sizes."""
    state = QuantumState(n_qubits)
    assert len(state.amplitudes) == 2 ** n_qubits

@pytest.mark.parametrize("gate,expected", [
    ("x", [0, 1]),
    ("h", [1/np.sqrt(2), 1/np.sqrt(2)]),
    ("z", [1, 0]),
])
def test_single_qubit_gates(gate, expected):
    """Test single qubit gates."""
    state = QuantumState(1)
    getattr(state, gate)(0)
    np.testing.assert_allclose(state.amplitudes, expected, rtol=1e-10)
[archived fence delimiter: ```]

### Slow Tests

[archived fence delimiter: ```python]
@pytest.mark.slow
def test_large_circuit():
    """Test large circuit (slow)."""
    state = QuantumState(20)
    for _ in range(100):
        for q in range(20):
            state.h(q)
    assert pytest.approx(sum(state.probabilities)) == 1.0
[archived fence delimiter: ```]

Run with: `pytest -m "not slow"` to skip slow tests.

## Writing Rust Tests

[archived fence delimiter: ```rust]
#[cfg(test)]
mod tests {
    use moonlab::QuantumState;

    #[test]
    fn test_state_creation() {
        let state = QuantumState::new(4);
        assert_eq!(state.num_qubits(), 4);
    }

    #[test]
    fn test_hadamard() {
        let mut state = QuantumState::new(1);
        state.h(0);

        let amps = state.amplitudes();
        let expected = 1.0 / 2.0_f64.sqrt();

        assert!((amps[0].re - expected).abs() < 1e-10);
        assert!((amps[1].re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut state = QuantumState::new(2);
        state.h(0);
        state.cnot(0, 1);

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
}
[archived fence delimiter: ```]

## Performance Tests

### Benchmarking

[archived fence delimiter: ```c]
#include "benchmark.h"
#include "quantum_sim.h"

BENCHMARK(bench_hadamard_20_qubits) {
    quantum_state_t* state = quantum_state_init(20);

    BENCH_START();
    for (int i = 0; i < 1000; i++) {
        gate_hadamard(state, 0);
    }
    BENCH_END();

    quantum_state_free(state);
}

BENCHMARK(bench_cnot_20_qubits) {
    quantum_state_t* state = quantum_state_init(20);
    gate_hadamard(state, 0);

    BENCH_START();
    for (int i = 0; i < 1000; i++) {
        gate_cnot(state, 0, 1);
    }
    BENCH_END();

    quantum_state_free(state);
}
[archived fence delimiter: ```]

### Python Benchmarks

[archived fence delimiter: ```python]
import pytest

@pytest.mark.benchmark
def test_gate_performance(benchmark):
    """Benchmark gate application."""
    from moonlab import QuantumState

    state = QuantumState(20)

    def apply_gates():
        for _ in range(100):
            state.h(0)
            state.cnot(0, 1)

    result = benchmark(apply_gates)
    assert result.stats.mean < 0.1  # Should complete in <100ms
[archived fence delimiter: ```]

## Fuzz Testing

### Structure

[archived fence delimiter: ```c]
#include "fuzz_framework.h"
#include "quantum_sim.h"

/**
 * Fuzz test for gate applications.
 */
FUZZ_TARGET(fuzz_gates) {
    // Get fuzzed input
    uint8_t* data = FUZZ_DATA;
    size_t size = FUZZ_SIZE;

    if (size < 4) return;

    // Extract parameters from fuzz data
    int n_qubits = (data[0] % 10) + 1;  // 1-10 qubits
    int gate_type = data[1] % 10;
    int qubit1 = data[2] % n_qubits;
    int qubit2 = data[3] % n_qubits;

    quantum_state_t* state = quantum_state_init(n_qubits);

    // Apply gate based on type
    switch (gate_type) {
        case 0: gate_hadamard(state, qubit1); break;
        case 1: gate_pauli_x(state, qubit1); break;
        case 2: gate_pauli_y(state, qubit1); break;
        case 3: gate_pauli_z(state, qubit1); break;
        case 4:
            if (qubit1 != qubit2) {
                gate_cnot(state, qubit1, qubit2);
            }
            break;
        // ... more cases
    }

    // Verify state is still valid
    double norm = quantum_state_norm(state);
    ASSERT(isfinite(norm));
    ASSERT(fabs(norm - 1.0) < 0.01);

    quantum_state_free(state);
}
[archived fence delimiter: ```]

## Code Coverage

### Generate Coverage Report

[archived fence delimiter: ```bash]
# Build with coverage
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="--coverage" \
      -DCMAKE_CXX_FLAGS="--coverage" \
      ..
make

# Run tests
ctest

# Generate report
gcovr --html --html-details -o coverage.html
[archived fence delimiter: ```]

### Python Coverage

[archived fence delimiter: ```bash]
pytest --cov=moonlab --cov-report=html bindings/python/tests/
open htmlcov/index.html
[archived fence delimiter: ```]

## CI Integration

Tests run automatically on GitHub Actions. See `.github/workflows/test.yml`:

[archived fence delimiter: ```yaml]
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Build
        run: |
          mkdir build && cd build
          cmake -DBUILD_TESTS=ON ..
          make -j4

      - name: Run C Tests
        run: cd build && ctest --output-on-failure

      - name: Run Python Tests
        run: pytest bindings/python/tests/ -v
[archived fence delimiter: ```]

## See Also

- [Development Setup](development-setup.md) - Environment configuration
- [Code Style](code-style.md) - Coding standards
- [Contributing](index.md) - Contribution process

```
