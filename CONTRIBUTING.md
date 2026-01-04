# Contributing to MoonLab Quantum Simulator

Thank you for your interest in contributing to MoonLab Quantum Simulator! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@moonlab.io.

## Getting Started

### Prerequisites

- **C compiler**: GCC 10+ or Clang 12+
- **Make** or **CMake 3.20+**
- **OpenMP**: For parallelization
- **macOS**: Xcode Command Line Tools (for Metal GPU support)
- **Linux**: OpenCL or Vulkan SDK (optional, for GPU support)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/tsotchke/moonlab.git
cd moonlab

# Build the library and tests
make all

# Run tests
make test
```

## Development Setup

### macOS (Recommended for Metal GPU)

```bash
# Install dependencies
brew install libomp lcov

# Build with Metal support
make ENABLE_METAL=1 all
```

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential libomp-dev lcov valgrind

# Build
make all
```

### Running Examples

```bash
# Build and run examples
make examples
./examples/quantum/grover_parallel_demo
./examples/quantum/vqe_demo
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/grover-optimization` - New features
- `fix/cnot-gate-normalization` - Bug fixes
- `docs/api-reference` - Documentation
- `perf/simd-hadamard` - Performance improvements
- `refactor/gate-abstraction` - Code refactoring

### Development Workflow

1. **Fork** the repository
2. **Create a branch** from `develop`
3. **Make changes** with tests
4. **Run tests locally**: `make test`
5. **Submit a pull request** to `develop`

## Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) for consistent commit messages:

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `security` | Security improvements |

### Examples

```bash
feat(gates): add controlled-RZ gate implementation

fix(state): correct normalization after measurement

perf(simd): optimize Hadamard gate with AVX-512

docs(api): add examples for VQE configuration
```

### Breaking Changes

For breaking changes, add `!` after the type or add `BREAKING CHANGE:` in the footer:

```bash
feat(api)!: rename quantum_state_create to qs_state_create

BREAKING CHANGE: All state creation calls must be updated.
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `make test`
2. **Check code style**: `make lint` (if available)
3. **Update documentation** for API changes
4. **Add tests** for new functionality

### PR Requirements

- [ ] Tests pass on all platforms (CI will verify)
- [ ] Code coverage does not decrease
- [ ] No new compiler warnings
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

### Review Process

1. **Automated checks** run on every PR
2. **Code review** by at least one maintainer
3. **Address feedback** with new commits (don't force push)
4. **Squash and merge** when approved

## Code Style

### C Code Style

```c
/**
 * @brief Apply a single-qubit gate to the quantum state.
 *
 * @param state Pointer to quantum state
 * @param gate Gate matrix (2x2 complex)
 * @param qubit Target qubit index
 * @return QS_SUCCESS on success, error code otherwise
 *
 * @stability stable
 * @since v1.0.0
 */
int qs_apply_gate(qs_state_t *state, const complex_t gate[2][2], int qubit) {
    // Validate inputs
    if (state == NULL) {
        return QS_ERROR_INVALID_STATE;
    }
    if (qubit < 0 || qubit >= state->num_qubits) {
        return QS_ERROR_INVALID_QUBIT;
    }

    // Implementation
    uint64_t stride = 1ULL << qubit;
    // ...

    return QS_SUCCESS;
}
```

### Style Guidelines

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters max
- **Braces**: K&R style (opening brace on same line)
- **Naming**:
  - Functions: `snake_case` with prefix (`qs_`, `qg_`, etc.)
  - Types: `snake_case_t` suffix
  - Constants: `UPPER_SNAKE_CASE`
  - Local variables: `snake_case`
- **Comments**: Doxygen-style for public APIs

### Header Guards

```c
#ifndef QUANTUM_STATE_H
#define QUANTUM_STATE_H

// ... content ...

#endif // QUANTUM_STATE_H
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_quantum_state.c
│   └── test_quantum_gates.c
├── integration/             # Integration tests
│   └── test_algorithms.c
└── benchmarks/              # Performance tests
    └── bench_grover.c
```

### Writing Tests

```c
void test_hadamard_creates_superposition(void) {
    quantum_state_t *state = quantum_state_init(1);

    apply_hadamard(state, 0);

    // |0⟩ → (|0⟩ + |1⟩)/√2
    double expected = 1.0 / sqrt(2.0);
    assert(fabs(creal(state->amplitudes[0]) - expected) < 1e-10);
    assert(fabs(creal(state->amplitudes[1]) - expected) < 1e-10);

    quantum_state_free(state);
}
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make unit_tests

# Run with coverage
make coverage

# Run with Valgrind (Linux)
valgrind --leak-check=full ./tests/quantum_sim_test
```

## Documentation

### API Documentation

Use Doxygen-style comments:

```c
/**
 * @file state.h
 * @brief Quantum state management functions.
 *
 * This module provides functions for creating, manipulating,
 * and measuring quantum states.
 */

/**
 * @brief Create a new quantum state.
 *
 * Initializes a quantum state with all qubits in the |0⟩ state.
 *
 * @param num_qubits Number of qubits (1-32)
 * @return Pointer to new state, or NULL on error
 *
 * @example
 * @code
 * qs_state_t *state = qs_state_create(4, NULL);
 * if (state == NULL) {
 *     // Handle error
 * }
 * // Use state...
 * qs_state_destroy(state);
 * @endcode
 *
 * @see qs_state_destroy
 * @stability stable
 * @since v1.0.0
 */
qs_state_t *qs_state_create(int num_qubits, qs_config_t *config);
```

### Generating Documentation

```bash
# Generate HTML documentation
doxygen Doxyfile

# View in browser
open docs/api/html/index.html
```

## Questions?

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: support@tsotchke.ai

Thank you for contributing!
