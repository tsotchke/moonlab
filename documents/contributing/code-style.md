# Code Style Guide

Coding conventions and formatting standards for Moonlab contributions.

## Overview

Moonlab follows a consistent C coding style optimized for readability, performance, and maintainability. This guide covers the conventions used throughout the codebase.

## Formatting

### Indentation and Spacing

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters maximum (soft limit)
- **Blank lines**: One between logical sections, two between function definitions

```c
// GOOD: Proper spacing
qs_error_t gate_hadamard(quantum_state_t *state, int qubit) {
    if (!state || !state->amplitudes) return QS_ERROR_INVALID_STATE;
    if (!check_qubit_valid(state, qubit)) return QS_ERROR_INVALID_QUBIT;

    const uint64_t stride = 1ULL << qubit;
    const uint64_t block_size = stride << 1;

    for (uint64_t base = 0; base < state->state_dim; base += block_size) {
        for (uint64_t i = 0; i < stride; i++) {
            const uint64_t idx0 = base + i;
            const uint64_t idx1 = idx0 + stride;

            const complex_t amp0 = state->amplitudes[idx0];
            const complex_t amp1 = state->amplitudes[idx1];

            state->amplitudes[idx0] = (amp0 + amp1) * QC_SQRT2_INV;
            state->amplitudes[idx1] = (amp0 - amp1) * QC_SQRT2_INV;
        }
    }

    return QS_SUCCESS;
}
```

### Braces

Use K&R style (opening brace on same line):

```c
// GOOD: K&R style
if (condition) {
    do_something();
} else {
    do_other();
}

for (size_t i = 0; i < n; i++) {
    process(i);
}

// GOOD: Single-line statements may omit braces for simple returns
if (!state) return QS_ERROR_INVALID_STATE;

// BAD: Allman style (opening brace on new line)
if (condition)
{
    do_something();
}
```

### Pointer and Reference Alignment

Attach `*` to the variable name:

```c
// GOOD
quantum_state_t *state;
const complex_t *amplitudes;
void *buffer;

// BAD
quantum_state_t* state;
quantum_state_t * state;
```

## Naming Conventions

### Functions

Use `snake_case` with module prefix:

```c
// Pattern: module_action_object()
qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits);
qs_error_t quantum_state_free(quantum_state_t *state);
qs_error_t gate_hadamard(quantum_state_t *state, int qubit);
qs_error_t gate_cnot(quantum_state_t *state, int control, int target);
grover_result_t grover_search(quantum_state_t *state, const grover_config_t *config);
```

### Variables

Use `snake_case`:

```c
// GOOD
size_t num_qubits;
uint64_t state_dim;
complex_t amplitude;
double entanglement_entropy;

// BAD
size_t numQubits;      // camelCase
uint64_t StateDim;     // PascalCase
complex_t amp;         // Too abbreviated (unless obvious from context)
```

### Constants and Macros

Use `UPPER_SNAKE_CASE`:

```c
#define MAX_QUBITS 32
#define MAX_STATE_DIM (1ULL << MAX_QUBITS)
#define QC_PI 3.14159265358979323846
#define QC_SQRT2_INV 0.7071067811865476
```

### Types

Use `snake_case` with `_t` suffix:

```c
typedef double _Complex complex_t;

typedef enum {
    QS_SUCCESS = 0,
    QS_ERROR_INVALID_QUBIT = -1,
    QS_ERROR_INVALID_STATE = -2,
    QS_ERROR_NOT_NORMALIZED = -3,
    QS_ERROR_OUT_OF_MEMORY = -4
} qs_error_t;

typedef struct {
    size_t num_qubits;
    size_t state_dim;
    complex_t *amplitudes;
} quantum_state_t;
```

### Static/Private Functions

Prefix with `static` and optionally use descriptive names:

```c
static inline int check_qubit_valid(const quantum_state_t *state, int qubit) {
    return (qubit >= 0 && qubit < (int)state->num_qubits);
}

static inline uint64_t flip_bit(uint64_t n, int bit_pos) {
    return n ^ (1ULL << bit_pos);
}
```

## File Organization

### Header Files

```c
#ifndef MODULE_NAME_H
#define MODULE_NAME_H

// 1. System includes
#include <stdint.h>
#include <stddef.h>
#include <complex.h>

// 2. Type definitions
typedef double _Complex complex_t;

/**
 * @file module_name.h
 * @brief Brief description of the module
 *
 * Detailed description including:
 * - Key features
 * - Usage notes
 * - Performance characteristics
 */

// 3. Constants/Macros
#define MAX_VALUE 100

// 4. Type declarations
typedef enum { ... } error_code_t;
typedef struct { ... } data_t;

// 5. Function prototypes (grouped by category)

// ============================================================================
// SECTION HEADER
// ============================================================================

/**
 * @brief Brief function description
 *
 * Detailed description with mathematical notation if applicable.
 *
 * @param param1 Description of parameter
 * @param param2 Description of parameter
 * @return Description of return value
 */
return_type function_name(param1_type param1, param2_type param2);

#endif // MODULE_NAME_H
```

### Source Files

```c
// 1. Corresponding header
#include "module_name.h"

// 2. Other project headers
#include "../utils/constants.h"
#include "../optimization/simd_ops.h"

// 3. System headers
#include <math.h>
#include <stdlib.h>
#include <string.h>

// 4. Platform-specific headers
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// 5. Local constants
#define LOCAL_CONSTANT 42

// ============================================================================
// SECTION NAME
// ============================================================================

// 6. Static helper functions
static inline int helper_function(...) { ... }

// 7. Public function implementations
qs_error_t public_function(...) { ... }
```

## Documentation

### Doxygen Comments

Use Doxygen-style comments for public APIs:

```c
/**
 * @brief Initialize quantum state in |0...0⟩
 *
 * Allocates and initializes a quantum state vector with AMX-aligned memory
 * for optimal M2 Ultra performance. State is initialized to |0...0⟩.
 *
 * Memory Requirements:
 * - 20 qubits: 16 MB
 * - 28 qubits: 4.3 GB
 * - 32 qubits: 68.7 GB
 *
 * @param state Pointer to quantum state structure (must not be NULL)
 * @param num_qubits Number of qubits (1-32)
 * @return QS_SUCCESS on success, error code on failure
 *
 * @note Uses Accelerate framework on macOS for memory alignment
 * @warning Large qubit counts require significant memory
 *
 * @code
 * quantum_state_t state;
 * qs_error_t err = quantum_state_init(&state, 10);
 * if (err != QS_SUCCESS) {
 *     // Handle error
 * }
 * @endcode
 */
qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits);
```

### Inline Comments

Use `//` for inline comments explaining non-obvious logic:

```c
qs_error_t grover_diffusion(quantum_state_t *state) {
    // Diffusion operator: D = 2|s⟩⟨s| - I
    // where |s⟩ = H⊗ⁿ|0⟩ⁿ (equal superposition)

    // Step 1: Apply Hadamard to all qubits
    for (size_t i = 0; i < state->num_qubits; i++) {
        qs_error_t err = gate_hadamard(state, i);
        if (err != QS_SUCCESS) return err;
    }

    // Step 2: Phase flip all states except |0...0⟩
    // SIMD OPTIMIZED: Use vectorized negation (8-16x faster on ARM NEON)
    if (state->state_dim > 1) {
        simd_negate(&state->amplitudes[1], state->state_dim - 1);
    }

    // Step 3: Apply Hadamard again
    for (size_t i = 0; i < state->num_qubits; i++) {
        qs_error_t err = gate_hadamard(state, i);
        if (err != QS_SUCCESS) return err;
    }

    return QS_SUCCESS;
}
```

### Section Headers

Use decorated section headers for logical groupings:

```c
// ============================================================================
// SINGLE-QUBIT GATES
// ============================================================================

qs_error_t gate_pauli_x(...) { ... }
qs_error_t gate_pauli_y(...) { ... }
qs_error_t gate_pauli_z(...) { ... }
qs_error_t gate_hadamard(...) { ... }

// ============================================================================
// TWO-QUBIT GATES
// ============================================================================

qs_error_t gate_cnot(...) { ... }
qs_error_t gate_cz(...) { ... }
```

## Error Handling

### Error Codes

Always return error codes rather than using assertions for recoverable errors:

```c
qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits) {
    // Validate input
    if (!state) return QS_ERROR_INVALID_STATE;
    if (num_qubits == 0 || num_qubits > MAX_QUBITS) {
        return QS_ERROR_INVALID_QUBIT;
    }

    // Allocate memory
    state->amplitudes = calloc(state->state_dim, sizeof(complex_t));
    if (!state->amplitudes) {
        return QS_ERROR_OUT_OF_MEMORY;
    }

    return QS_SUCCESS;
}
```

### Error Propagation

Check and propagate errors immediately:

```c
qs_error_t complex_operation(quantum_state_t *state) {
    qs_error_t err;

    err = gate_hadamard(state, 0);
    if (err != QS_SUCCESS) return err;

    err = gate_cnot(state, 0, 1);
    if (err != QS_SUCCESS) return err;

    return QS_SUCCESS;
}
```

### Guard Clauses

Use early returns for validation:

```c
// GOOD: Guard clauses at the start
qs_error_t gate_hadamard(quantum_state_t *state, int qubit) {
    if (!state || !state->amplitudes) return QS_ERROR_INVALID_STATE;
    if (!check_qubit_valid(state, qubit)) return QS_ERROR_INVALID_QUBIT;

    // Main logic here...
    return QS_SUCCESS;
}

// BAD: Deeply nested validation
qs_error_t gate_hadamard(quantum_state_t *state, int qubit) {
    if (state) {
        if (state->amplitudes) {
            if (check_qubit_valid(state, qubit)) {
                // Main logic here...
            }
        }
    }
    return QS_ERROR_INVALID_STATE;
}
```

## Performance Patterns

### Use `const` Liberally

Mark immutable values as `const`:

```c
// GOOD: const for values that don't change
const uint64_t stride = 1ULL << qubit;
const uint64_t block_size = stride << 1;
const complex_t amp0 = state->amplitudes[idx0];
```

### Use `static inline` for Small Helpers

```c
static inline int check_qubit_valid(const quantum_state_t *state, int qubit) {
    return (qubit >= 0 && qubit < (int)state->num_qubits);
}

static inline uint64_t flip_bit(uint64_t n, int bit_pos) {
    return n ^ (1ULL << bit_pos);
}
```

### Platform-Specific Code

Use preprocessor guards for SIMD and platform-specific optimizations:

```c
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON optimized path
    for (uint64_t base = 0; base < state->state_dim; base += block_size) {
        // Vectorized processing...
    }
#elif defined(__AVX2__)
    // x86 AVX2 optimized path
    __m256d scale = _mm256_set1_pd(inv_sqrt2);
    // ...
#else
    // Scalar fallback
    for (uint64_t i = 0; i < num_pairs; i++) {
        // Standard processing...
    }
#endif
```

### Loop Optimization

Use stride-based indexing for cache efficiency:

```c
// GOOD: Stride-based access pattern
const uint64_t stride = 1ULL << qubit;
const uint64_t block_size = stride << 1;

for (uint64_t base = 0; base < state->state_dim; base += block_size) {
    for (uint64_t i = 0; i < stride; i++) {
        const uint64_t idx0 = base + i;
        const uint64_t idx1 = idx0 + stride;
        // Process pair...
    }
}

// BAD: Bit-checking in inner loop
for (uint64_t i = 0; i < state->state_dim; i++) {
    if (get_bit(i, qubit) == 0) {  // Expensive bit check
        uint64_t j = flip_bit(i, qubit);
        // Process pair...
    }
}
```

## Python Style

Python bindings follow PEP 8 with these additions:

```python
"""
Module docstring with description.
"""

import numpy as np
from typing import Optional, List, Tuple

class QuantumState:
    """Quantum state representation.

    Attributes:
        num_qubits: Number of qubits in the system.
        amplitudes: Complex amplitude vector.
    """

    def __init__(self, num_qubits: int) -> None:
        """Initialize quantum state.

        Args:
            num_qubits: Number of qubits (1-32).

        Raises:
            ValueError: If num_qubits is out of range.
        """
        if not 1 <= num_qubits <= 32:
            raise ValueError(f"num_qubits must be 1-32, got {num_qubits}")

        self.num_qubits = num_qubits
        self._state = _create_state(num_qubits)

    def h(self, qubit: int) -> 'QuantumState':
        """Apply Hadamard gate.

        Args:
            qubit: Target qubit index.

        Returns:
            Self for method chaining.
        """
        _gate_hadamard(self._state, qubit)
        return self
```

## Clang-Format Configuration

The project uses clang-format for automatic formatting:

```yaml
# .clang-format
---
Language: Cpp
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 100
BreakBeforeBraces: Attach
AllowShortIfStatementsOnASingleLine: Always
AllowShortLoopsOnASingleLine: false
AllowShortFunctionsOnASingleLine: Inline
PointerAlignment: Right
SpaceAfterCStyleCast: false
SpaceBeforeParens: ControlStatements
IndentCaseLabels: true
AlignConsecutiveMacros: true
AlignConsecutiveDeclarations: false
AlignTrailingComments: true
SortIncludes: false
...
```

### Running clang-format

```bash
# Format a single file
clang-format -i src/quantum/gates.c

# Format all source files
find src -name '*.c' -o -name '*.h' | xargs clang-format -i

# Check formatting without modifying
clang-format --dry-run -Werror src/quantum/gates.c
```

## Git Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code refactoring |
| `docs` | Documentation |
| `test` | Tests |
| `chore` | Build, CI, dependencies |

### Examples

```
feat(gates): add parameterized rotation gates

Implement RX, RY, RZ gates with arbitrary angle parameters.
Uses SIMD optimization for rotation matrix computation.

Closes #42
```

```
fix(state): correct memory alignment on M1

State vector was not aligned to 64-byte boundaries,
causing suboptimal AMX utilization.

Fixes #78
```

## See Also

- [Development Setup](development-setup.md) - Environment configuration
- [Testing Guide](testing-guide.md) - Testing practices
- [Documentation Guide](documentation-guide.md) - Writing documentation
