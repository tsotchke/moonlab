# Error Codes Reference

Moonlab's C functions signal failure through small negative integer error codes. The core
simulation surface uses `qs_error_t`; the distributed backend defines its own per-subsystem
enums. All codes are `int`-compatible, and `0` always means success.

## Core: `qs_error_t`

Defined in `src/quantum/state.h`. Returned by state-management and gate functions.

| Code | Value | Meaning |
|------|-------|---------|
| `QS_SUCCESS` | 0 | Operation succeeded |
| `QS_ERROR_INVALID_QUBIT` | -1 | Qubit index out of range |
| `QS_ERROR_INVALID_STATE` | -2 | Invalid state (NULL or uninitialized) |
| `QS_ERROR_NOT_NORMALIZED` | -3 | State is not normalized (e.g. zero norm) |
| `QS_ERROR_OUT_OF_MEMORY` | -4 | Memory allocation failed |
| `QS_ERROR_INVALID_DIMENSION` | -5 | Dimension mismatch |
| `QS_ERROR_INVALID_PARAM` | -6 | Invalid non-qubit argument |
| `QS_ERROR_NOT_SUPPORTED` | -7 | Feature not compiled in (e.g. CUDA path on a CPU-only build) |
| `QS_ERROR_DRIVER` | -8 | Backend/driver failure (GPU, etc.) |

```c
qs_error_t err = quantum_state_init(&state, num_qubits);
if (err != QS_SUCCESS) {
    fprintf(stderr, "init failed: %d\n", err);
    return 1;
}
```

The `QS_*` names are stable across the v0.x/v1.x series. New in-tree modules may use the
centralised `moonlab_status_t` registry (`src/utils/moonlab_status.h`), whose values are
compatible with these codes.

## Distributed: MPI bridge (`mpi_bridge_error_t`)

Defined in `src/distributed/mpi_bridge.h`. `mpi_bridge_error_string(code)` returns a
human-readable message.

| Code | Value |
|------|-------|
| `MPI_BRIDGE_SUCCESS` | 0 |
| `MPI_BRIDGE_ERROR_INIT` | -1 |
| `MPI_BRIDGE_ERROR_COMM` | -2 |
| `MPI_BRIDGE_ERROR_PARTITION` | -3 |
| `MPI_BRIDGE_ERROR_ALLOC` | -4 |
| `MPI_BRIDGE_ERROR_SYNC` | -5 |
| `MPI_BRIDGE_ERROR_NOT_SUPPORTED` | -6 |
| `MPI_BRIDGE_ERROR_TIMEOUT` | -7 |

## Distributed: gates (`dist_gate_error_t`)

Defined in `src/distributed/distributed_gates.h`. `dist_gate_error_string(code)` maps to text.

| Code | Value |
|------|-------|
| `DIST_GATE_SUCCESS` | 0 |
| `DIST_GATE_ERROR_INVALID_QUBIT` | -1 |
| `DIST_GATE_ERROR_COMM` | -2 |
| `DIST_GATE_ERROR_ALLOC` | -3 |
| `DIST_GATE_ERROR_NOT_INITIALIZED` | -4 |
| `DIST_GATE_ERROR_INVALID_MATRIX` | -5 |

## Distributed: partitioning (`partition_error_t`)

Defined in `src/distributed/state_partition.h`. `partition_error_string(code)` maps to text.

| Code | Value |
|------|-------|
| `PARTITION_SUCCESS` | 0 |
| `PARTITION_ERROR_INVALID_QUBITS` | -1 |
| `PARTITION_ERROR_ALLOC` | -2 |
| `PARTITION_ERROR_MPI` | -3 |
| `PARTITION_ERROR_INDEX_RANGE` | -4 |
| `PARTITION_ERROR_NOT_INITIALIZED` | -5 |

## Distributed: collective operations (`collective_error_t`)

Defined in `src/distributed/collective_ops.h`. `collective_error_string(code)` maps to text.

| Code | Value |
|------|-------|
| `COLLECTIVE_SUCCESS` | 0 |
| `COLLECTIVE_ERROR_INVALID_ARG` | -1 |
| `COLLECTIVE_ERROR_MPI` | -2 |
| `COLLECTIVE_ERROR_ALLOC` | -3 |
| `COLLECTIVE_ERROR_NOT_INITIALIZED` | -4 |
| `COLLECTIVE_ERROR_INVALID_QUBIT` | -5 |

## Python

The Python bindings raise exceptions rather than returning codes:
`moonlab.core.QuantumError` for core failures, and
`moonlab.distributed.MpiUnavailableError` when the distributed engine is not present in the
loaded library (a non-MPI build).

## See Also

- [Troubleshooting](../troubleshooting.md)
- [Configuration Options](configuration-options.md)
- [C API: Quantum State](../api/c/quantum-state.md)
