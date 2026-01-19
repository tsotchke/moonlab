# Debugging Guide

Diagnose and fix common issues in quantum simulations.

## Debug Mode

### Enabling Debug Output

```c
// Compile with debug symbols
// make DEBUG=1

// Enable runtime debugging
#include "utils/config.h"

quantum_config_set("debug.level", "verbose");
quantum_config_set("debug.log_file", "quantum_debug.log");

// Or via environment
// export MOONLAB_DEBUG=2
// export MOONLAB_LOG_FILE=debug.log
```

### Debug Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Off | No debug output |
| 1 | Error | Errors only |
| 2 | Warning | Errors + warnings |
| 3 | Info | General information |
| 4 | Verbose | Detailed tracing |
| 5 | Trace | Full execution trace |

## Common Issues

### 1. State Not Normalized

**Symptom**: Measurement probabilities don't sum to 1, or NaN values appear.

**Diagnosis**:
```c
double norm = quantum_state_norm(state);
printf("State norm: %.15f\n", norm);

if (fabs(norm - 1.0) > 1e-10) {
    printf("WARNING: State not normalized!\n");
}
```

**Causes**:
- Applying non-unitary operations
- Numerical precision loss after many gates
- Bug in custom gate implementation

**Solutions**:
```c
// Renormalize periodically
quantum_state_normalize(state);

// Check gate unitarity
bool is_unitary = gate_check_unitary(my_gate, 1e-10);
if (!is_unitary) {
    fprintf(stderr, "Gate is not unitary!\n");
}
```

### 2. Incorrect Measurement Results

**Symptom**: Measurement outcomes don't match expected probabilities.

**Diagnosis**:
```c
// Print state amplitudes
quantum_state_print(state);

// Compare expected vs actual probabilities
double* probs = quantum_state_probabilities(state);
for (size_t i = 0; i < state->dimension; i++) {
    printf("|%zu⟩: %.6f\n", i, probs[i]);
}
```

**Causes**:
- Wrong qubit ordering (little-endian vs big-endian)
- Gate applied to wrong qubit
- State not properly initialized

**Solutions**:
```c
// Verify qubit indexing convention
// Moonlab uses little-endian: |q_{n-1}...q_1 q_0⟩
// Index i = q_0 * 2^0 + q_1 * 2^1 + ... + q_{n-1} * 2^{n-1}

// Example: |101⟩ in 3-qubit system
// q_0 = 1, q_1 = 0, q_2 = 1
// Index = 1*1 + 0*2 + 1*4 = 5

// Verify with specific state
quantum_state_init(state, 3);
gate_x(state, 0);  // Flip q_0
gate_x(state, 2);  // Flip q_2
// State should be |101⟩ = index 5
printf("Amplitude at |101⟩: %.6f\n", cabs(state->amplitudes[5]));
```

### 3. Memory Errors

**Symptom**: Segmentation fault, corrupted data, or valgrind errors.

**Diagnosis**:
```bash
# Run with valgrind
valgrind --leak-check=full ./my_quantum_app

# Run with AddressSanitizer
make SANITIZE=address
./my_quantum_app
```

**Common Memory Issues**:

```c
// WRONG: Accessing freed state
quantum_state_t* state = quantum_state_create(10);
quantum_state_destroy(state);
gate_hadamard(state, 0);  // Use after free!

// WRONG: Buffer overflow
quantum_state_t* state = quantum_state_create(5);
gate_hadamard(state, 10);  // Qubit index out of bounds!

// CORRECT: Bounds checking
if (qubit < state->num_qubits) {
    gate_hadamard(state, qubit);
} else {
    fprintf(stderr, "Qubit %d out of range\n", qubit);
}
```

**Solutions**:
```c
// Enable bounds checking
quantum_config_set("safety.bounds_check", "true");

// Use safe wrappers
int result = gate_hadamard_safe(state, qubit);
if (result != QUANTUM_SUCCESS) {
    handle_error(result);
}
```

### 4. Performance Issues

**Symptom**: Simulation runs much slower than expected.

**Diagnosis**:
```c
#include "tools/profiler/profiler.h"

profiler_start();

// Your simulation code
for (int i = 0; i < 1000; i++) {
    gate_hadamard(state, 0);
}

profiler_stop();
profiler_report(stdout);
```

**Common Performance Issues**:

| Issue | Symptom | Solution |
|-------|---------|----------|
| No SIMD | 4-8x slower | Compile with `-march=native` |
| No parallelism | Uses 1 core | Enable OpenMP: `make OMP=1` |
| Memory thrashing | Slow I/O | Reduce qubit count |
| GPU not used | CPU bound | Enable Metal: `make METAL=1` |

**Solutions**:
```c
// Check optimization status
quantum_print_capabilities();
// Output:
// SIMD: NEON enabled
// Threads: 8 (OpenMP)
// GPU: Metal (M2 Ultra)

// Force GPU usage
quantum_config_set("gpu.enabled", "true");
quantum_config_set("gpu.threshold", "16");  // Use GPU for ≥16 qubits
```

### 5. Entanglement Issues

**Symptom**: Bell state doesn't show expected correlations.

**Diagnosis**:
```c
// Create Bell state
quantum_state_init(state, 2);
gate_hadamard(state, 0);
gate_cnot(state, 0, 1);

// Check entanglement entropy
double entropy = quantum_entanglement_entropy(state, 0);
printf("Entanglement entropy: %.6f\n", entropy);
// Should be ln(2) ≈ 0.693 for maximally entangled

// Verify Bell state amplitudes
printf("|00⟩: %.6f\n", cabs(state->amplitudes[0]));  // ~0.707
printf("|01⟩: %.6f\n", cabs(state->amplitudes[1]));  // ~0.0
printf("|10⟩: %.6f\n", cabs(state->amplitudes[2]));  // ~0.0
printf("|11⟩: %.6f\n", cabs(state->amplitudes[3]));  // ~0.707
```

### 6. CHSH Test Failures

**Symptom**: Bell test doesn't reach theoretical maximum of 2√2.

**Diagnosis**:
```c
bell_test_result_t result = bell_test_chsh(state, 0, 1, 10000, NULL, &entropy);

printf("CHSH Value: %.4f\n", result.chsh_value);
printf("Statistical error: %.4f\n", result.error);
printf("Shots: %zu\n", result.num_shots);
```

**Causes**:
- Too few measurement shots (statistical noise)
- State preparation errors
- Wrong measurement angles

**Solutions**:
```c
// Increase shots for better statistics
bell_test_chsh(state, 0, 1, 100000, NULL, &entropy);

// Use optimal measurement settings
bell_test_settings_t settings = {
    .alice_angles = {0.0, M_PI/2},      // 0° and 90°
    .bob_angles = {M_PI/4, -M_PI/4}     // 45° and -45°
};
bell_test_chsh(state, 0, 1, 10000, &settings, &entropy);
```

## Debugging Tools

### State Visualization

```c
// ASCII visualization
quantum_state_print_bars(state);
// Output:
// |00⟩ ████████████████ 0.500
// |01⟩                  0.000
// |10⟩                  0.000
// |11⟩ ████████████████ 0.500

// Export for external visualization
quantum_state_export_json(state, "state.json");
```

### Circuit Tracing

```c
// Enable gate-by-gate logging
quantum_config_set("trace.gates", "true");

gate_hadamard(state, 0);
// [TRACE] H(0): |0⟩ → (|0⟩+|1⟩)/√2

gate_cnot(state, 0, 1);
// [TRACE] CNOT(0,1): (|00⟩+|10⟩)/√2 → (|00⟩+|11⟩)/√2
```

### Assertion Macros

```c
#include "utils/debug.h"

// Verify state properties
QUANTUM_ASSERT_NORMALIZED(state);
QUANTUM_ASSERT_QUBIT_RANGE(state, qubit);
QUANTUM_ASSERT_UNITARY(gate_matrix, dim);

// Custom assertions
QUANTUM_ASSERT(condition, "Error message: %s", details);
```

## Python Debugging

```python
import moonlab
from moonlab.debug import enable_debug, StateInspector

# Enable debug mode
moonlab.set_debug_level(3)

# Inspect state
state = QuantumState(4)
state.h(0)
state.cnot(0, 1)

inspector = StateInspector(state)
inspector.print_amplitudes()
inspector.print_probabilities()
inspector.check_normalization()
inspector.plot_state()  # Matplotlib visualization
```

## GDB Integration

```bash
# Compile with debug symbols
make DEBUG=1

# Run under GDB
gdb ./my_quantum_app

# Useful GDB commands
(gdb) break quantum_state_apply_gate
(gdb) run
(gdb) print *state
(gdb) print state->amplitudes[0]
(gdb) call quantum_state_print(state)
```

### Custom GDB Pretty Printers

```python
# ~/.gdbinit
python
import gdb

class QuantumStatePrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        n = int(self.val['num_qubits'])
        dim = int(self.val['dimension'])
        return f"QuantumState({n} qubits, dim={dim})"

def quantum_lookup(val):
    if str(val.type) == 'quantum_state_t':
        return QuantumStatePrinter(val)
    return None

gdb.pretty_printers.append(quantum_lookup)
end
```

## Logging Best Practices

```c
#include "utils/logging.h"

// Structured logging
LOG_INFO("Applying gate", "gate=H", "qubit=%d", qubit);
LOG_DEBUG("State amplitude", "index=%zu", "amp=(%.6f, %.6f)", i, creal(amp), cimag(amp));
LOG_ERROR("Gate failed", "error=%d", error_code);

// Log to file
quantum_log_init("simulation.log", LOG_LEVEL_DEBUG);

// Periodic state dumps
if (iteration % 1000 == 0) {
    char filename[64];
    snprintf(filename, sizeof(filename), "state_%06d.bin", iteration);
    quantum_state_save(state, filename);
}
```

## See Also

- [Troubleshooting](../troubleshooting.md)
- [Error Codes Reference](../reference/error-codes.md)
- [Configuration Options](../reference/configuration-options.md)
