# Moonlab error codes — discovery catalog

Moonlab uses per-module error enums (`*_error_t`) rather than a single
unified status type. This is the C idiom: each module owns its
contract, downstream callers handle each enum where they call into it.
This document is the central registry for every enum in the tree so
contributors can find them without grepping.

**Convention.** Every Moonlab error enum:
- has a zero member named `*_SUCCESS` (or `*_OK` in a few legacy
  modules — flagged below)
- uses negative values for error states
- is declared in the module's public header next to the functions
  that return it
- is **module-scoped**: do not compare a value of one enum type to
  another; the integer codes overlap by design

## Index

| Module                                    | Enum                          | Header |
|-------------------------------------------|-------------------------------|--------|
| Distributed: state partition              | `partition_error_t`           | `src/distributed/state_partition.h` |
| Distributed: gate dispatch                | `dist_gate_error_t`           | `src/distributed/distributed_gates.h` |
| Distributed: collectives                  | `collective_error_t`          | `src/distributed/collective_ops.h` |
| Distributed: MPI bridge                   | `mpi_bridge_error_t`          | `src/distributed/mpi_bridge.h` |
| GPU dispatch                              | `gpu_error_t`                 | `src/optimization/gpu/gpu_backend.h` |
| GPU: Eshkol fp64-on-Metal                 | `moonlab_eshkol_status_t`     | `src/optimization/gpu/backends/gpu_eshkol.h` |
| Stabilizer tableau                        | `clifford_error_t`            | `src/backends/clifford/clifford.h` |
| Tensor library                            | `tensor_error_t`              | `src/algorithms/tensor_network/tensor.h` |
| Tensor: contraction planner               | `contract_error_t`            | `src/algorithms/tensor_network/contraction.h` |
| Tensor: SVD compression                   | `svd_compress_error_t`        | `src/algorithms/tensor_network/svd_compress.h` |
| Tensor: MPS state                         | `tn_state_error_t`            | `src/algorithms/tensor_network/tn_state.h` |
| Tensor: MPS gates                         | `tn_gate_error_t`             | `src/algorithms/tensor_network/tn_gates.h` |
| Tensor: MPS measurement                   | `tn_measure_error_t`          | `src/algorithms/tensor_network/tn_measurement.h` |
| Tensor: CA-MPS                            | `ca_mps_error_t`              | `src/algorithms/tensor_network/ca_mps.h` |

## Detail

### `partition_error_t` (distributed/state_partition.h)
| Code | Name |
|------|------|
| `0` | `PARTITION_SUCCESS` |
| `-1` | `PARTITION_ERROR_INVALID_QUBITS` |
| `-2` | `PARTITION_ERROR_ALLOC` |
| `-3` | `PARTITION_ERROR_MPI` |
| `-4` | `PARTITION_ERROR_INDEX_RANGE` |
| `-5` | `PARTITION_ERROR_NOT_INITIALIZED` |

### `dist_gate_error_t` (distributed/distributed_gates.h)
| Code | Name |
|------|------|
| `0` | `DIST_GATE_SUCCESS` |
| `-1` | `DIST_GATE_ERROR_INVALID_QUBIT` |
| `-2` | `DIST_GATE_ERROR_COMM` |
| `-3` | `DIST_GATE_ERROR_ALLOC` |
| `-4` | `DIST_GATE_ERROR_NOT_INITIALIZED` |
| `-5` | `DIST_GATE_ERROR_INVALID_MATRIX` |

### `collective_error_t` (distributed/collective_ops.h)
| Code | Name |
|------|------|
| `0` | `COLLECTIVE_SUCCESS` |
| `-1` | `COLLECTIVE_ERROR_INVALID_ARG` |
| `-2` | `COLLECTIVE_ERROR_MPI` |
| `-3` | `COLLECTIVE_ERROR_ALLOC` |
| `-4` | `COLLECTIVE_ERROR_NOT_INITIALIZED` |
| `-5` | `COLLECTIVE_ERROR_INVALID_QUBIT` |

### `mpi_bridge_error_t` (distributed/mpi_bridge.h)
| Code | Name |
|------|------|
| `0` | `MPI_BRIDGE_SUCCESS` |
| `-1` | `MPI_BRIDGE_ERROR_INIT` |
| `-2` | `MPI_BRIDGE_ERROR_COMM` |
| `-3` | `MPI_BRIDGE_ERROR_PARTITION` |
| `-4` | `MPI_BRIDGE_ERROR_ALLOC` |
| `-5` | `MPI_BRIDGE_ERROR_SYNC` |
| `-6` | `MPI_BRIDGE_ERROR_NOT_SUPPORTED` |
| `-7` | `MPI_BRIDGE_ERROR_TIMEOUT` |

### `gpu_error_t` (optimization/gpu/gpu_backend.h)

Live in the unified GPU dispatch layer; specific backend codes
(Metal / WebGPU / OpenCL / Vulkan / CUDA / cuQuantum) all map into
this domain.

### `moonlab_eshkol_status_t` (optimization/gpu/backends/gpu_eshkol.h)

Eshkol fp64-on-Metal interop status. **Legacy naming:** uses
`MOONLAB_ESHKOL_OK` (not `_SUCCESS`).  See header for the full
list of precision-tier and status codes.

### `clifford_error_t` (backends/clifford/clifford.h)

Stabilizer tableau backend.  See header — codes for invalid qubit
indices, allocation failure, and post-measurement projection
failures.

### `tensor_error_t` (algorithms/tensor_network/tensor.h)

Generic tensor library: dimension mismatches, out-of-range axes,
allocation failure.

### `contract_error_t` (algorithms/tensor_network/contraction.h)

Tensor-network contraction planner.

### `svd_compress_error_t` (algorithms/tensor_network/svd_compress.h)

SVD-based bond compression.

### `tn_state_error_t` (algorithms/tensor_network/tn_state.h)

MPS state lifecycle, canonicalization, truncation.

### `tn_gate_error_t` (algorithms/tensor_network/tn_gates.h)

MPS gate application.

### `tn_measure_error_t` (algorithms/tensor_network/tn_measurement.h)

MPS-state measurement.

### `ca_mps_error_t` (algorithms/tensor_network/ca_mps.h)

Clifford-Assisted MPS.

## Adding a new error enum

If you write a new module that needs a return code:

1. Add `MODULE_SUCCESS = 0` as the zero member.
2. Use negative integers for failure codes.
3. Pick a `MODULE_ERROR_*` prefix that is short and unique
   (avoid collisions with the table above when grepping).
4. Add the enum to the table above so future contributors find it.
5. Do **not** add a code to a sibling enum just because the value
   "fits" — error codes are scoped to the module that returns them.

## What is *not* here

- `*_result_t` types in algorithm headers (`vqe_result_t`,
  `qaoa_result_t`, `lanczos_result_t`, ...) — those are **structs**
  carrying success metadata, not enums.  Look in the per-algorithm
  header.
- ABI-level diagnostics on the stable surface are exposed via the
  function-specific return values documented in
  `src/applications/moonlab_export.h`.
