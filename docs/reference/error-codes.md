# Error Codes Reference

Complete catalog of Moonlab error codes and their resolutions.

## Error Code Format

Error codes follow the pattern: `QSIM_ERR_<CATEGORY>_<SPECIFIC>`

Categories:
- `MEM`: Memory allocation errors
- `ARG`: Invalid argument errors
- `STATE`: State-related errors
- `GATE`: Gate operation errors
- `ALGO`: Algorithm errors
- `GPU`: GPU/Metal errors
- `IO`: Input/output errors
- `MPI`: Distributed computing errors

## Error Code Tables

### Memory Errors (0x0100 - 0x01FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0100 | `QSIM_ERR_MEM_ALLOC` | Memory allocation failed | Reduce qubit count or free memory |
| 0x0101 | `QSIM_ERR_MEM_ALIGN` | Memory alignment failed | Check alignment requirements |
| 0x0102 | `QSIM_ERR_MEM_GPU` | GPU memory allocation failed | Reduce state size or use CPU |
| 0x0103 | `QSIM_ERR_MEM_POOL` | Memory pool exhausted | Increase pool size or release objects |
| 0x0104 | `QSIM_ERR_MEM_MAP` | Memory mapping failed | Check system limits |

### Argument Errors (0x0200 - 0x02FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0200 | `QSIM_ERR_ARG_NULL` | Null pointer argument | Check pointer validity |
| 0x0201 | `QSIM_ERR_ARG_RANGE` | Argument out of range | Verify value bounds |
| 0x0202 | `QSIM_ERR_ARG_QUBIT` | Invalid qubit index | Qubit must be < num_qubits |
| 0x0203 | `QSIM_ERR_ARG_SIZE` | Invalid size parameter | Check size is positive |
| 0x0204 | `QSIM_ERR_ARG_TYPE` | Invalid type parameter | Check valid enum values |
| 0x0205 | `QSIM_ERR_ARG_CONFLICT` | Conflicting arguments | Review parameter combination |

### State Errors (0x0300 - 0x03FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0300 | `QSIM_ERR_STATE_INVALID` | Invalid quantum state | Check state was properly created |
| 0x0301 | `QSIM_ERR_STATE_DESTROYED` | State already destroyed | Don't use after destroy |
| 0x0302 | `QSIM_ERR_STATE_SIZE` | State size mismatch | Verify qubit counts match |
| 0x0303 | `QSIM_ERR_STATE_NORM` | State not normalized | Call normalize() |
| 0x0304 | `QSIM_ERR_STATE_QUBIT_MAX` | Too many qubits | Max 32 for state vector |
| 0x0305 | `QSIM_ERR_STATE_MEASURED` | State already measured | Create new state or reset |

### Gate Errors (0x0400 - 0x04FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0400 | `QSIM_ERR_GATE_INVALID` | Invalid gate | Check gate creation |
| 0x0401 | `QSIM_ERR_GATE_QUBIT_DUP` | Duplicate qubit in gate | Qubits must be distinct |
| 0x0402 | `QSIM_ERR_GATE_CONTROL` | Invalid control qubit | Control must differ from target |
| 0x0403 | `QSIM_ERR_GATE_MATRIX` | Invalid gate matrix | Matrix must be 2x2 or 4x4 |
| 0x0404 | `QSIM_ERR_GATE_UNITARY` | Gate not unitary | U†U must equal I |
| 0x0405 | `QSIM_ERR_GATE_PARAMS` | Invalid gate parameters | Check parameter values |

### Algorithm Errors (0x0500 - 0x05FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0500 | `QSIM_ERR_ALGO_CONVERGE` | Algorithm did not converge | Increase iterations |
| 0x0501 | `QSIM_ERR_ALGO_ORACLE` | Invalid oracle function | Check oracle implementation |
| 0x0502 | `QSIM_ERR_ALGO_HAMILTONIAN` | Invalid Hamiltonian | Verify Hamiltonian is Hermitian |
| 0x0503 | `QSIM_ERR_ALGO_ANSATZ` | Invalid ansatz | Check ansatz parameters |
| 0x0504 | `QSIM_ERR_ALGO_OPTIMIZER` | Optimizer error | Try different optimizer |
| 0x0505 | `QSIM_ERR_ALGO_BOND_DIM` | Bond dimension too small | Increase max_bond_dim |

### GPU Errors (0x0600 - 0x06FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0600 | `QSIM_ERR_GPU_UNAVAIL` | GPU not available | Check GPU support |
| 0x0601 | `QSIM_ERR_GPU_INIT` | GPU initialization failed | Verify Metal is installed |
| 0x0602 | `QSIM_ERR_GPU_DEVICE` | GPU device error | Check device status |
| 0x0603 | `QSIM_ERR_GPU_SHADER` | Shader compilation failed | Report as bug |
| 0x0604 | `QSIM_ERR_GPU_BUFFER` | GPU buffer error | Reduce buffer size |
| 0x0605 | `QSIM_ERR_GPU_QUEUE` | Command queue error | Reset GPU context |
| 0x0606 | `QSIM_ERR_GPU_TIMEOUT` | GPU operation timed out | Reduce problem size |

### I/O Errors (0x0700 - 0x07FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0700 | `QSIM_ERR_IO_FILE` | File operation failed | Check file path and permissions |
| 0x0701 | `QSIM_ERR_IO_FORMAT` | Invalid file format | Verify file format |
| 0x0702 | `QSIM_ERR_IO_VERSION` | Incompatible file version | Update Moonlab or convert file |
| 0x0703 | `QSIM_ERR_IO_CORRUPT` | File data corrupted | Recreate file |
| 0x0704 | `QSIM_ERR_IO_PERMISSION` | Permission denied | Check file permissions |

### MPI Errors (0x0800 - 0x08FF)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 0x0800 | `QSIM_ERR_MPI_INIT` | MPI initialization failed | Check MPI installation |
| 0x0801 | `QSIM_ERR_MPI_COMM` | MPI communication error | Check network |
| 0x0802 | `QSIM_ERR_MPI_RANK` | Invalid MPI rank | Verify rank values |
| 0x0803 | `QSIM_ERR_MPI_SIZE` | Invalid partition size | Check process count |
| 0x0804 | `QSIM_ERR_MPI_SYNC` | Synchronization error | Check for deadlocks |

## Error Handling

### C API

```c
#include "quantum_sim.h"

// Check return code
qsim_error_t err = quantum_state_h(state, qubit);
if (err != QSIM_SUCCESS) {
    const char* msg = qsim_error_message(err);
    fprintf(stderr, "Error 0x%04X: %s\n", err, msg);
}

// Get last error
err = qsim_get_last_error();
if (err != QSIM_SUCCESS) {
    printf("Last error: %s\n", qsim_error_message(err));
    qsim_clear_error();
}

// Custom error handler
void my_error_handler(qsim_error_t err, const char* file, int line) {
    fprintf(stderr, "Error at %s:%d - %s\n", file, line, qsim_error_message(err));
}
qsim_set_error_handler(my_error_handler);
```

### Python API

```python
from moonlab import QuantumState, MoonlabError

try:
    state = QuantumState(40)  # Too many qubits
except MoonlabError as e:
    print(f"Error {e.code:04X}: {e.message}")
    # Error 0304: Maximum qubit count exceeded

# Specific exception types
from moonlab.errors import (
    MemoryError,
    QubitIndexError,
    GateError,
    GPUError
)

try:
    state.h(100)  # Invalid qubit
except QubitIndexError as e:
    print(f"Invalid qubit: {e}")
```

### Rust API

```rust
use moonlab::{QuantumState, Error, ErrorKind};

fn main() -> Result<(), Error> {
    let state = QuantumState::new(4)?;

    match state.h(10) {
        Ok(_) => println!("Gate applied"),
        Err(e) => {
            match e.kind() {
                ErrorKind::QubitIndex => println!("Invalid qubit"),
                ErrorKind::Memory => println!("Memory error"),
                _ => println!("Other error: {}", e),
            }
        }
    }

    Ok(())
}
```

## Common Error Scenarios

### Memory Allocation Failed

```
Error 0x0100: Memory allocation failed

Cause: Insufficient memory for state vector.
State with N qubits requires 16 × 2^N bytes.

Solutions:
1. Reduce qubit count
2. Close other applications
3. Use tensor network methods for large systems
4. Enable swap (not recommended for performance)
```

### Invalid Qubit Index

```
Error 0x0202: Invalid qubit index

Cause: Qubit index >= num_qubits

Example:
  state = QuantumState(4)  # Qubits 0, 1, 2, 3
  state.h(4)  # ERROR: qubit 4 doesn't exist

Solution: Use qubit indices 0 to n-1
```

### GPU Not Available

```
Error 0x0600: GPU not available

Causes:
1. macOS version < 12.0
2. Running in VM without GPU passthrough
3. Built without GPU support
4. No Metal-capable GPU

Check:
  from moonlab import gpu_diagnose
  gpu_diagnose()

Solutions:
1. Update macOS
2. Use CPU backend: set_backend('cpu')
3. Rebuild with -DENABLE_GPU=ON
```

### Algorithm Did Not Converge

```
Error 0x0500: Algorithm did not converge

Cause: VQE/QAOA optimization failed to find minimum.

Possible issues:
1. Learning rate too high/low
2. Not enough iterations
3. Poor initial parameters
4. Ansatz not expressive enough

Solutions:
1. Adjust optimizer settings:
   vqe = VQE(learning_rate=0.01, max_iterations=500)
2. Try different optimizer:
   vqe = VQE(optimizer='COBYLA')
3. Use multiple random initializations
4. Increase ansatz depth
```

### State Not Normalized

```
Error 0x0303: State not normalized

Cause: Numerical errors accumulated over many operations.

Check:
  norm = state.norm()
  print(f"Norm: {norm}")  # Should be 1.0

Solutions:
1. Call normalize():
   state.normalize()
2. Enable auto-normalization:
   configure(auto_normalize=True)
3. Use higher precision:
   configure(precision='double')
```

## Debugging Tips

### Enable Verbose Errors

```python
from moonlab import configure
configure(error_verbosity='detailed')
```

### Error Logging

```bash
export MOONLAB_LOG_LEVEL=DEBUG
python your_script.py 2>&1 | tee debug.log
```

### Core Dumps (C)

```bash
# Enable core dumps
ulimit -c unlimited

# Run program
./bin/your_program

# Analyze with lldb
lldb ./bin/your_program -c /cores/core.xxxxx
(lldb) bt  # Backtrace
```

## See Also

- [Troubleshooting](../troubleshooting.md) - Common issues
- [FAQ](../faq.md) - Frequently asked questions
- [API Reference](../api/index.md) - Function documentation

