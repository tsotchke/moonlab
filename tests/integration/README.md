# Integration Tests

Comprehensive integration tests for the Moonlab Quantum Simulator algorithms.

## Test Files

| File | Description | Key Tests |
|------|-------------|-----------|
| `test_vqe_integration.c` | VQE algorithm tests | Hamiltonian construction, ansatz preparation, gradient computation, H2 chemical accuracy |
| `test_qaoa_integration.c` | QAOA algorithm tests | Graph encoding, MaxCut cost, parameter optimization, approximation ratio |
| `test_grover_integration.c` | Grover search tests | Oracle, diffusion, optimal iterations, multi-target search, measurement |
| `test_qpe_integration.c` | QPE algorithm tests | QFT, inverse QFT, phase estimation, precision scaling, Hamiltonian simulation |

## Building

```bash
# Build all integration tests
make integration-tests

# Or build individually
gcc -o test_vqe tests/integration/test_vqe_integration.c \
    src/quantum/*.c src/algorithms/vqe.c src/utils/*.c \
    -I. -lm -O2

gcc -o test_qaoa tests/integration/test_qaoa_integration.c \
    src/quantum/*.c src/algorithms/qaoa.c src/utils/*.c \
    -I. -lm -O2

gcc -o test_grover tests/integration/test_grover_integration.c \
    src/quantum/*.c src/algorithms/grover.c src/utils/*.c \
    -I. -lm -O2

gcc -o test_qpe tests/integration/test_qpe_integration.c \
    src/quantum/*.c src/algorithms/qpe.c src/utils/*.c \
    -I. -lm -O2
```

## Running Tests

```bash
# Run all integration tests
./test_vqe
./test_qaoa
./test_grover
./test_qpe

# Run with verbose output
./test_vqe --verbose
```

## Test Coverage

### VQE Tests
- **Hamiltonian Construction**: Verifies Pauli decomposition, Hermiticity, H2 encoding
- **Ansatz Preparation**: Hardware-efficient and UCCSD ansatz states
- **Energy Evaluation**: Expectation values, variational bounds
- **Gradient Computation**: Parameter shift rule, finite difference verification
- **Optimizer Convergence**: ADAM, BFGS, COBYLA convergence
- **Chemical Accuracy**: H2 ground state energy within 1.6 mHa
- **Potential Surface**: Bond length scanning
- **Noise Robustness**: Depolarizing noise tolerance

### QAOA Tests
- **Graph Encoding**: Vertices, edges, Ising model conversion
- **MaxCut Cost**: Bitstring evaluation, partition counting
- **Initial State**: Uniform superposition preparation
- **Cost Unitary**: Problem Hamiltonian application
- **Mixer Unitary**: X-rotation mixing
- **QAOA Layers**: Combined cost + mixer operations
- **Parameter Optimization**: p=1, p=2 optimization quality
- **Multiple Graphs**: Triangle, square, K4, star graphs
- **Solution Verification**: Cut value validation

### Grover Tests
- **Initial Superposition**: Uniform distribution
- **Oracle**: Single and multi-target phase flip
- **Diffusion Operator**: Amplitude amplification
- **Optimal Iterations**: O(sqrt(N)) scaling
- **Full Search**: End-to-end search success
- **Measurement**: Statistical success rate
- **Over-iteration**: Probability oscillation
- **Speedup Verification**: Quadratic advantage

### QPE Tests
- **QFT Correctness**: Transform properties
- **Inverse QFT**: Reversibility
- **Controlled Phase**: Gate application
- **Standard Gates**: T, S, Z gate phases
- **Precision Scaling**: Error vs bits
- **Success Probability**: Exact phase estimation
- **2-Qubit Unitary**: CNOT eigenvalues
- **Iterative QPE**: Single-qubit implementation
- **Hamiltonian Simulation**: Time evolution phase
- **Phase Kickback**: Controlled-U mechanism

## Expected Output

```
================================================================================
VQE Integration Tests
================================================================================

    PASS: Hamiltonian Construction
    PASS: Ansatz State Preparation
    PASS: Energy Expectation Value
    PASS: Parameter Shift Gradient
    PASS: Optimizer Convergence
      VQE energy: -1.137284 Ha
      FCI energy: -1.137284 Ha
      Error: 0.0000 mHa
    PASS: H2 Chemical Accuracy
    PASS: H2 Potential Energy Surface
    PASS: UCCSD Ansatz
    PASS: Noise Robustness
    PASS: Different Optimizers

================================================================================
Results: 10 passed, 0 failed (2.34 seconds)
================================================================================
```

## Adding New Tests

1. Create a new test function following the pattern:
```c
void test_new_feature(void) {
    printf("  Test: New Feature\n");

    // Setup
    // ...

    // Test assertions
    TEST_ASSERT(condition, "Description");
    TEST_ASSERT_NEAR(actual, expected, tolerance, "Description");

    // Cleanup
    // ...

    TEST_PASS("New Feature");
}
```

2. Add the test call in `main()`:
```c
test_new_feature();
```

3. Run and verify the test passes.

## Performance Notes

- VQE tests: ~2-5 seconds (depends on optimizer iterations)
- QAOA tests: ~1-3 seconds
- Grover tests: ~0.5-1 second
- QPE tests: ~0.5-1 second

For faster development iteration, individual tests can be run by modifying `main()` to call only specific test functions.
