# Configuration Options

Runtime configuration for the Moonlab C core lives in `qsim_config_t`
(`src/utils/config.h`): backend selection, SIMD, threading, noise model, memory, logging,
and validation. Obtain the global config with `qsim_config_global()` (after
`qsim_config_init()`), or create a standalone one with `qsim_config_create()`.

```c
#include "utils/config.h"

qsim_config_t* cfg = qsim_config_create();
qsim_config_set_backend(cfg, QSIM_BACKEND_AUTO);
qsim_config_set_threads(cfg, 0);          // 0 = auto-detect
qsim_config_set_log_level(cfg, QSIM_LOG_INFO);
qsim_config_destroy(cfg);
```

## Backends (`qsim_backend_t`)

| Value | Backend |
|-------|---------|
| `QSIM_BACKEND_AUTO` | Automatic selection |
| `QSIM_BACKEND_CPU` | CPU only |
| `QSIM_BACKEND_CPU_SIMD` | CPU with SIMD |
| `QSIM_BACKEND_GPU_METAL` | Apple Metal |
| `QSIM_BACKEND_GPU_OPENCL` | OpenCL |
| `QSIM_BACKEND_GPU_VULKAN` | Vulkan compute |
| `QSIM_BACKEND_GPU_CUDA` | NVIDIA CUDA |
| `QSIM_BACKEND_DISTRIBUTED` | MPI distributed |
| `QSIM_BACKEND_GPU_WEBGPU` | WebGPU (WASM/browser) |

`qsim_backend_available(backend)` reports whether a backend is usable on the current system;
`qsim_detect_backend()` returns the recommended one.

## SIMD (`qsim_simd_t`)

`QSIM_SIMD_NONE`, `QSIM_SIMD_SSE2`, `QSIM_SIMD_AVX`, `QSIM_SIMD_AVX2`, `QSIM_SIMD_AVX512`,
`QSIM_SIMD_NEON`, `QSIM_SIMD_SVE`. Use `qsim_detect_simd()` to pick the best available.

## Threading (`qsim_threading_t`)

`QSIM_THREADING_NONE`, `QSIM_THREADING_OPENMP`, `QSIM_THREADING_PTHREADS`,
`QSIM_THREADING_GCD` (macOS). `num_threads = 0` auto-detects (`qsim_detect_threads()`).

## Log Levels (`qsim_log_level_t`)

`QSIM_LOG_NONE`, `QSIM_LOG_ERROR`, `QSIM_LOG_WARN`, `QSIM_LOG_INFO`, `QSIM_LOG_DEBUG`,
`QSIM_LOG_TRACE`. Set `log_file` to a path (or NULL for stderr).

## Noise Model (`qsim_noise_config_t`)

| Field | Meaning |
|-------|---------|
| `enabled` | Enable noise simulation |
| `depolarizing_rate` | Depolarizing probability |
| `amplitude_damping` | Amplitude damping rate |
| `phase_damping` | Phase damping rate |
| `t1_time` / `t2_time` | T1 relaxation / T2 dephasing time (µs) |
| `gate_time` | Gate execution time (µs) |
| `readout_error_0` / `readout_error_1` | Readout error P(1 given 0) / P(0 given 1) |

Set via `qsim_config_set_noise_enabled`, `qsim_config_set_noise_params`, and
`qsim_config_set_thermal`.

## Limits and Validation

| Field | Meaning |
|-------|---------|
| `max_qubits` | Maximum qubits (0 = no explicit limit) |
| `max_state_dim` | Maximum state dimension |
| `validate_states` | Validate state normalization |
| `check_unitarity` | Check gate unitarity |
| `tolerance` | Numerical tolerance |
| `enable_caching` / `gate_fusion` / `circuit_optimization` | Performance toggles |
| `use_quantum_rng` / `seed` | RNG source and PRNG seed (0 = random) |

`qsim_detect_max_qubits()` estimates the maximum safe qubit count from available memory.

## Environment Variables

`qsim_config_from_env(cfg)` reads these and returns the number applied:

| Variable | Effect |
|----------|--------|
| `QSIM_BACKEND` | `auto`, `cpu`, `gpu_metal`, `gpu_opencl`, ... |
| `QSIM_SIMD` | `none`, `sse2`, `avx`, `avx2`, `avx512`, `neon`, `sve` |
| `QSIM_MAX_QUBITS` | integer |
| `QSIM_THREADS` | integer (0 = auto) |
| `QSIM_LOG_LEVEL` | `none`, `error`, `warn`, `info`, `debug`, `trace` |
| `QSIM_LOG_FILE` | path |
| `QSIM_NOISE` | `0` or `1` |
| `QSIM_SEED` | integer |

Separately, the Python and Rust bindings honor `MOONLAB_LIB_DIR` (directory containing the
built `libquantumsim`) and `MOONLAB_LIB` (full path) to locate the shared library.

## Presets

| Preset | Effect |
|--------|--------|
| `qsim_config_preset_performance(cfg)` | Enable all optimizations, use GPU if available |
| `qsim_config_preset_accuracy(cfg)` | Enable validation, disable approximations |
| `qsim_config_preset_low_memory(cfg)` | Minimize memory usage |
| `qsim_config_preset_noisy(cfg, t1_us, t2_us, gate_error)` | Realistic hardware noise model |

## File I/O

`qsim_config_load(path)` / `qsim_config_save(cfg, path)` (JSON or INI), and
`qsim_config_from_json(str)` / `qsim_config_to_json(cfg)`.

## See Also

- [C API: Config](../api/c/config.md)
- [Error Codes](error-codes.md)
- [Performance Tuning](../guides/performance-tuning.md)
