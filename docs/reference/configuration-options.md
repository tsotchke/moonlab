# Configuration options reference

Moonlab's runtime configuration is owned by the `qsim_config_t`
struct declared in
[`src/utils/config.h`](../../src/utils/config.h).  This document
records the public fields, the typed setter/getter functions, the
environment variables parsed by `qsim_config_from_env`, and the
preset helpers.  Every option name on this page maps directly to a
declaration in `src/utils/config.h`; nothing else is part of the
public surface.

## Overview

Configuration flows from highest to lowest priority:

1. **API**: direct field assignment or typed setters on a
   `qsim_config_t *` (`src/utils/config.h:255-323`).
2. **JSON / INI**: `qsim_config_load(path)` and
   `qsim_config_from_json(json)` (`src/utils/config.h:366-383`).
3. **Environment**: `qsim_config_from_env(config)` reads the
   `QSIM_*` variables listed below (`src/utils/config.h:413`).
4. **Defaults**: applied by `qsim_config_create` /
   `qsim_config_reset` (`src/utils/config.h:228, 250`).

A process-global configuration exists for convenience -- see
`qsim_config_global`, `qsim_config_init`, `qsim_config_cleanup`
(`src/utils/config.h:205-217`).  Validation routines
(`qsim_config_validate`, `qsim_backend_available`,
`qsim_simd_available`) flag invalid combinations against the
current host.

## Backend selection

| Field | Type | Setter | Values |
|-------|------|--------|--------|
| `config->backend` | `qsim_backend_t` | `qsim_config_set_backend(config, backend)` | `QSIM_BACKEND_AUTO`, `QSIM_BACKEND_CPU`, `QSIM_BACKEND_CPU_SIMD`, `QSIM_BACKEND_GPU_METAL`, `QSIM_BACKEND_GPU_OPENCL`, `QSIM_BACKEND_GPU_VULKAN`, `QSIM_BACKEND_GPU_CUDA`, `QSIM_BACKEND_DISTRIBUTED`, `QSIM_BACKEND_GPU_WEBGPU` |

Enumeration declared at `src/utils/config.h:53-63`.  Use
`qsim_backend_available(backend)` to test whether the current build
links the corresponding backend before forcing it.

| Field | Type | Setter | Values |
|-------|------|--------|--------|
| `config->simd` | `qsim_simd_t` | `qsim_config_set_simd(config, simd)` | `QSIM_SIMD_NONE`, `QSIM_SIMD_SSE2`, `QSIM_SIMD_AVX`, `QSIM_SIMD_AVX2`, `QSIM_SIMD_AVX512`, `QSIM_SIMD_NEON`, `QSIM_SIMD_SVE` |

Enumeration declared at `src/utils/config.h:68-76`.  Auto-detection
via `qsim_detect_simd()` picks the highest level the host supports.

## Limits

| Field | Type | Setter | Notes |
|-------|------|--------|-------|
| `config->max_qubits`    | `int`      | `qsim_config_set_max_qubits(config, n)` | `0` means "no library-imposed cap" (the state-vector backend still caps at `MOONLAB_MAX_QUBITS = 32`). |
| `config->max_state_dim` | `uint64_t` | direct assignment                       | Manual cap on $2^N$, overrides `max_qubits` when smaller.  |

## Threading

`config->threading` is a `qsim_thread_config_t`
(`src/utils/config.h:131-136`):

| Field | Type | Notes |
|-------|------|-------|
| `model`           | `qsim_threading_t` | `QSIM_THREADING_NONE`, `QSIM_THREADING_OPENMP`, `QSIM_THREADING_PTHREADS`, `QSIM_THREADING_GCD` |
| `num_threads`     | `int`              | `0` => auto-detect via `qsim_detect_threads()` |
| `thread_affinity` | `int`              | non-zero enables CPU pinning where supported |

Setter shortcut: `qsim_config_set_threads(config, n)`
(`src/utils/config.h:274`).

## GPU

`config->gpu` is a `qsim_gpu_config_t`
(`src/utils/config.h:141-146`):

| Field | Type | Notes |
|-------|------|-------|
| `device_id`       | `int`     | `-1` = auto-select |
| `max_vram`        | `size_t`  | byte cap; 0 = unlimited |
| `async_transfers` | `int`     | enable async host/device copies |
| `kernel_fusion`   | `int`     | enable Metal/WebGPU kernel fusion |

Setter shortcut: `qsim_config_set_gpu_device(config, id)`
(`src/utils/config.h:313`).

## Noise

`config->noise` is a `qsim_noise_config_t`
(`src/utils/config.h:107-117`):

| Field | Type | Notes |
|-------|------|-------|
| `enabled`            | `int`    | non-zero to apply noise after every gate |
| `depolarizing_rate`  | `double` | per-gate depolarising probability |
| `amplitude_damping`  | `double` | per-gate amplitude-damping rate |
| `phase_damping`      | `double` | per-gate phase-damping rate |
| `t1_time`, `t2_time` | `double` | T1 / T2 relaxation times (µs) |
| `gate_time`          | `double` | gate execution time (µs) |
| `readout_error_0`    | `double` | P(1\|0) measurement error |
| `readout_error_1`    | `double` | P(0\|1) measurement error |

Helpers: `qsim_config_set_noise_enabled`,
`qsim_config_set_noise_params(depol, ad, pd)`,
`qsim_config_set_thermal(t1, t2, gate_time)`
(`src/utils/config.h:289-303`).  The MPDO simulator
(`src/quantum/noise_mpdo.h`) uses its own configuration struct; see
[mpdo-api.md](mpdo-api.md).

## Memory

`config->memory` is a `qsim_memory_config_t`
(`src/utils/config.h:122-127`):

| Field | Type | Notes |
|-------|------|-------|
| `max_memory`     | `size_t` | byte cap on host RAM; `0` = unlimited |
| `alignment`      | `size_t` | state-vector alignment in bytes |
| `use_huge_pages` | `int`    | request transparent huge pages |
| `preallocate`    | `int`    | reserve state memory at init time |

Setter shortcut: `qsim_config_set_max_memory(config, bytes)`
(`src/utils/config.h:308`).

## Algorithm parameters

`config->algorithm` is a `qsim_algorithm_config_t`
(`src/utils/config.h:151-156`):

| Field | Type | Default | Purpose |
|-------|------|--------:|---------|
| `max_measurements`            | `size_t` | 1024 | circular buffer cap on measurement history |
| `jacobi_max_iter`             | `int`    |  100 | Jacobi eigenvalue iteration cap |
| `jacobi_tolerance`            | `double` | 1e-12 | Jacobi convergence threshold |
| `grover_analysis_max_qubits`  | `int`    |   12 | upper bound for Grover analysis kernels |

## Validation and tolerance

Top-level fields in `qsim_config_t` (`src/utils/config.h:182-189`):

| Field | Type | Notes |
|-------|------|-------|
| `validate_states`     | `int`    | check state-norm invariants per op |
| `check_unitarity`     | `int`    | verify `U†U = I` for incoming gate matrices |
| `tolerance`           | `double` | numerical-comparison threshold |
| `enable_caching`      | `int`    | cache gate matrices keyed by signature |
| `gate_fusion`         | `int`    | fuse adjacent commuting gates |
| `circuit_optimization`| `int`    | run circuit-level optimisations before execution |

Setter shortcut: `qsim_config_set_tolerance(config, tol)`
(`src/utils/config.h:323`).

## Logging

| Field | Type | Setter | Values |
|-------|------|--------|--------|
| `config->log_level` | `qsim_log_level_t` | `qsim_config_set_log_level(config, level)` | `QSIM_LOG_NONE`, `QSIM_LOG_ERROR`, `QSIM_LOG_WARN`, `QSIM_LOG_INFO`, `QSIM_LOG_DEBUG`, `QSIM_LOG_TRACE` |
| `config->log_file`  | `const char *`     | `qsim_config_set_log_file(config, path)`   | `NULL` => stderr |

Enumeration declared at `src/utils/config.h:91-98`.  The
`qsim_log_level_to_string` / `qsim_log_level_from_string` helpers
(`src/utils/config.h:507-512`) translate to and from the canonical
strings used by `QSIM_LOG_LEVEL`.

## Random-number generation

| Field | Type | Setter | Notes |
|-------|------|--------|-------|
| `config->use_quantum_rng` | `int`      | direct                              | route measurement randomness through the Bell-verified QRNG |
| `config->seed`            | `uint64_t` | `qsim_config_set_seed(config, seed)` | `0` => seed from system entropy   |

QRNG hardware-init can be skipped via the `MOONLAB_SKIP_HW_ENTROPY`
environment variable (see [Environment variables](#environment-variables)).

## Environment variables

`qsim_config_from_env(config)` (`src/utils/config.h:413`, body in
`src/utils/config.c:299-348`) reads the following.  Unset variables
leave the corresponding field at its current value.

| Variable             | Mapped field                  | Parser                                 |
|----------------------|-------------------------------|----------------------------------------|
| `QSIM_BACKEND`       | `config->backend`             | `qsim_backend_from_string`             |
| `QSIM_SIMD`          | `config->simd`                | `qsim_simd_from_string`                |
| `QSIM_MAX_QUBITS`    | `config->max_qubits`          | base-10 integer                        |
| `QSIM_THREADS`       | `config->threading.num_threads` | base-10 integer (`0` = auto)         |
| `QSIM_LOG_LEVEL`     | `config->log_level`           | `qsim_log_level_from_string`           |
| `QSIM_LOG_FILE`      | `config->log_file`            | filesystem path                        |
| `QSIM_NOISE`         | `config->noise.enabled`       | `0` / `1`                              |
| `QSIM_SEED`          | `config->seed`                | base-10 `uint64_t`                     |

Additional environment variables consulted elsewhere in the tree
(outside `qsim_config_from_env`):

| Variable                          | Read by                                | Purpose |
|-----------------------------------|----------------------------------------|---------|
| `MOONLAB_SKIP_HW_ENTROPY`         | `src/applications/qrng.c`              | bypass hardware-entropy probe at QRNG init (CI / aarch64). |
| `MOONLAB_TENSOR_GPU_THRESHOLD_MUL`| `src/algorithms/tensor_network/tn_gates.c` | scale the tensor-GPU dispatch crossover. |
| `MOONLAB_BENCH_N`                 | `src/utils/bench_stats.h:81` (`bench_stats_n_runs`) | timing-replica count for benchmark harnesses. |
| `MOONLAB_LIB_DIR`                 | `bindings/python/moonlab/core.py:38` and `bindings/rust/moonlab/build.rs:15` | directory containing the built `libquantumsim` library; the Python and Rust bindings dlopen / link against it. |

## File I/O

`qsim_config_load(path)` (`src/utils/config.h:366`) supports JSON
and INI inputs (the docstring lists both; INI parsing lives next to
the JSON path in `src/utils/config.c`).  `qsim_config_save` and
`qsim_config_to_json` are the round-trip counterparts.  There is no
TOML loader and no `~/.moonlab/config.toml` lookup.

## Presets

`src/utils/config.h:518-545` declares four preset helpers that
stamp a known-good configuration over an existing struct:

| Preset                              | Purpose                                                    |
|-------------------------------------|------------------------------------------------------------|
| `qsim_config_preset_performance`    | enables all optimisations, picks the GPU backend if present |
| `qsim_config_preset_accuracy`       | enables validation, disables approximations                 |
| `qsim_config_preset_low_memory`     | minimises memory at the cost of speed                       |
| `qsim_config_preset_noisy(t1, t2, gate_error)` | populates the noise sub-config from device numbers |

## Worked example (C)

```c
#include "utils/config.h"

qsim_config_t *cfg = qsim_config_create();

qsim_config_preset_performance(cfg);
qsim_config_set_threads(cfg, 8);
qsim_config_set_log_level(cfg, QSIM_LOG_INFO);

qsim_config_from_env(cfg);              /* environment overrides */

char err[128];
if (qsim_config_validate(cfg, err, sizeof err) != 0) {
    fprintf(stderr, "bad config: %s\n", err);
    return 1;
}

/* ... run simulation against cfg ... */

qsim_config_destroy(cfg);
```

## Language bindings

The Python (`bindings/python/moonlab/`) and Rust
(`bindings/rust/moonlab/`) bindings do not currently expose
`qsim_config_t` directly.  Configuration is controlled through:

- the environment variables above (preferred for Python / Rust
  callers since they affect the loaded `libquantumsim` at init), and
- module-specific dataclasses such as `moonlab.tdvp.TdvpConfig`,
  `moonlab.mpdo.MpdoConfig`, and the equivalents in the Rust
  surface, which pass typed structs into the algorithm modules.

When the bindings need to influence the global C config they call
`qsim_config_global` and the typed setters via ctypes / FFI rather
than wrapping the struct opaquely.

## See also

- [Error codes](error-codes.md) -- meaning of the `int` return
  codes from `qsim_config_*` functions.
- [TDVP API](tdvp-api.md), [QGT API](qgt-api.md),
  [MPDO API](mpdo-api.md) -- module-level config structs that layer
  on top of `qsim_config_t`.
