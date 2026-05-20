# Cross-language parity matrix

Coverage of moonlab capabilities across the four bindings as of v1.0.

| Symbol | Meaning |
|--------|---------|
| ✅     | Full parity with the C surface |
| ◐      | Partial -- see footnote |
| ✗      | Not exposed in this binding (intentional) |

## Quantum core

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| State vector + gates (≤32 qubits)       | ✅  | `core.py`   | `state.rs`        | `quantum-state.ts`|
| Circuit builder                         | ✅  | `core.py`   | `state.rs`        | `circuit.ts`      |
| Bell-pair primitive + CHSH + Mermin     | ✅  | `algorithms.py` | `bell.rs`     | `bell.ts`         |
| Grover search                           | ✅  | `algorithms.py` | `grover.rs`   | `grover.ts`       |
| VQE + native autograd                   | ✅  | `algorithms.py` | `vqe.rs`      | `vqe.ts`          |
| QAOA                                    | ✅  | `algorithms.py` | `qaoa.rs`     | `qaoa.ts`         |
| Clifford tableau backend                | ✅  | `clifford.py` | `clifford.rs`   | `clifford.ts`     |
| Gate-fusion DAG                         | ✅  | `fusion.py` | `fusion.rs`       | `fusion.ts`       |

## Tensor networks

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| MPS state + SVD truncation              | ✅  | `tdvp.py`   | `tdvp.rs`         | `tensor-network.ts` ⁵ |
| Two-site adaptive TDVP (entropy-PID)    | ✅  | `tdvp.py`   | `tdvp.rs`         | `tdvp.ts`         |
| MPDO noise simulator                    | ✅  | `mpdo.py`   | `mpdo.rs`         | `mpdo.ts`         |
| DMRG (variational ground state)         | ✅  | `dmrg.py`   | `dmrg.rs`         | `tensor-network.ts` ¹ |
| CA-MPS (Clifford-assisted MPS)          | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts`       |
| CA-MPS variational-D (Clifford search)  | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts`       |
| CA-MPS Born-rule sampling (since v0.10) | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts`       |
| CA-PEPS 2D                              | ✅  | `ca_peps.py`| `ca_peps.rs`      | `ca-peps.ts`      |

## Quantum geometry + topology

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Berry curvature grid integrator         | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Chern number (FHS + parallel-transport) | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Real-space Bianco-Resta Chern marker    | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Haldane / Kane-Mele / BHZ / Kitaev / Hofstadter models | ✅ | `topology.py` | `topology.rs` | `topology.ts` |
| Kane-Mele full Rashba (since v0.10)     | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Z_2 invariant (block-Chern + Wilson-loop, since v0.10) | ✅ | `topology.py` | `topology.rs` | `topology.ts` |
| Quasicrystal Chern mosaic (Bianco-Resta KPM, L=O(300)) | ✅  | `topology.py` | `topology.rs` ⁶ | ✗ ⁶             |

## Topological QC + QEC

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Surface code (rotated, d × d)           | ✅  | `surface_code.py` | `surface_code.rs` | `surface-code.ts`|
| Threshold sweep harness                 | ✅  | `surface_code.py` | `surface_code.rs` | `surface-code.ts`|
| Decoder zoo (MWPM, MWPM-exact, BP, SBNN, PyMatching) | ✅ | `decoder.py` | `decoder.rs` | `decoder.ts` |
| QGTL circuit ingestion                  | ✅  | `qgtl.py`   | `qgtl.rs`         | `qgtl.ts`         |
| libirrep QEC zoo bridge (opt-in)        | ✅ ⁷ | `libirrep_qec.py` | `libirrep_qec.rs` | `libirrep-qec.ts` |
| Z_2 1+1D lattice gauge theory           | ✅  | `ca_mps.py` | `z2_lgt.rs`      | `ca-mps.ts`       |

## Distributed + cloud

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Distributed scheduler MVP               | ✅  | `scheduler.py` | `scheduler.rs` | `scheduler.ts`    |
| MPI state-vector partitioning           | ✅ ⁸ | ✗          | ✗                | ✗                |
| Control plane (line protocol)           | ✅  | `control_plane.py` | `control_plane.rs` | `control-plane.ts` ⁹ |
| Control plane TLS / mTLS                | ✅  | `control_plane.py` | `control_plane.rs` | `control-plane.ts` |
| Control plane HMAC-SHA3 auth            | ✅  | `control_plane.py` | `control_plane.rs` | `control-plane.ts` |
| Prometheus exporter sidecar             | ✅ ¹⁰| -          | -                 | -                 |
| WebSocket gateway (since v1.0)          | ✅ ¹¹| -          | -                 | -                 |
| Docker production stack                 | ✅ ¹²| -          | -                 | -                 |

## Crypto + RNG

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| ML-KEM-512 / 768 / 1024 (FIPS 203)      | ✅  | `algorithms.py` | ✗            | ✗                |
| Bell-verified QRNG                      | ✅  | `algorithms.py` | ✗            | ✗                |
| SHA3 / AES / DRBG primitives            | ✅  | ✗          | ✗                | ✗                |

## GPU + acceleration

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Apple Metal backend                     | ✅  | ✗          | ✗                | ✗                |
| Apple AMX / SME                         | ✅  | ✗          | ✗                | ✗                |
| WebGPU (browser + Deno)                 | ✅  | ✗          | ✗                | `webgpu.ts`       |
| CUDA / OpenCL / Vulkan / cuQuantum      | ◐ ¹³| ✗          | ✗                | ✗                |
| Eshkol fp64-on-Metal interop            | ✅  | ✗          | ✗                | ✗                |

## Footnotes

¹ All four bindings expose the v0.10.0 stable-ABI DMRG energy entries
(`moonlab_dmrg_tfim_energy`, `moonlab_dmrg_heisenberg_energy`).  JS
additionally exposes `dmrgTFIMGroundState` from `tensor-network.ts`
which returns the converged MPS handle plus a result struct via the
internal `_dmrg_tfim_ground_state` entry.  Two-site DMRG with caller-
supplied MPO is still C-only -- drop to `dmrg_ground_state` via FFI
for that.

³ All four bindings expose the variational-D Clifford search: JS as
`varDRun` in `ca-mps.ts`, Python as `ca_mps.var_d_run_v2`, Rust as
`moonlab::ca_mps::var_d_run`.  The TS layer marshals the
`ca_mps_var_d_alt_config_t` + `ca_mps_var_d_alt_result_t` structs into
the WASM heap by hand (offsets pinned in the source).

⁴ `moonlab_ca_mps_sample_z` reaches all four bindings (C + Python +
Rust + JS) on the v1.0 line.  JS calls `CaMps.sampleZ(numSamples,
randomValues)` against the rebuilt WASM blob; integration test in
`src/__tests__/ca-mps.integration.test.ts` checks Bell + GHZ_4 support.

⁵ JS binds the MPS state through `tensor-network.ts` but the
adaptive-bond TDVP driver is in the separate `tdvp.ts` module.

⁶ The Bianco-Resta Chern marker (`chern_kpm_*` API) is exposed in all
three high-level bindings.  The current ceiling on the sparse-stencil
backend is L = O(300) (~90 000 sites); the C-side
`tests/performance/bench_chern_mosaic_hq.c` harness drives this directly
and ships the canonical PRL-reproduction artefacts.  No Python or Rust
driver currently re-implements the full PPM-emitting + manifest pipeline.
JS is `✗` because the chern_kpm symbols are not in the WASM exports list
(the kernel is heavy and not a browser-target).

⁷ libirrep bridge requires `-DQSIM_ENABLE_LIBIRREP=ON` at build time.
Without it the bridge entries return `MOONLAB_LIBIRREP_NOT_BUILT = -201`
deliberately (caller can always call -- not a stub).

⁸ MPI state-vector requires `-DQSIM_ENABLE_MPI=ON` and an MPI runtime
(OpenMPI / MPICH).  No binding language exposes the MPI bridge because
none of the runtimes (CPython, Rust, V8) cleanly co-host an MPI rank.

⁹ The JS control-plane client is Node-only -- browsers can't open raw
TCP sockets.  Use the WebSocket gateway for browser-side clients
(see deploy/docker/ or footnote ¹¹).

¹⁰ `tools/exporter/moonlab_control_exporter.py` is a Python
HTTP-bridge sidecar.  Exposed as a Docker image
`moonlab/control-exporter:0.10.0`; see `deploy/docker/`.

¹¹ `tools/gateway/moonlab_websocket_gateway.py` (since v1.0)
exposes the line protocol via WebSocket so browsers can submit
circuits.  Same Python image is used in `deploy/docker/`.

¹² The compose stack in `deploy/docker/docker-compose.yml` builds and
runs the control plane + exporter + Prometheus.

¹³ CUDA / OpenCL / Vulkan / cuQuantum compile under their respective
`QSIM_ENABLE_*` opt-in flags but the moonlab CI does not currently run
on machines with those SDKs.  The code paths are present (1000+ LOC
each) but smoke-tested manually only.

## v1.0 parity gaps acknowledged

The following parity gaps will NOT close in v1.0; they are listed
here for transparency:

- **JS Node-only control-plane client.**  Browsers must go through the
  WebSocket gateway.
- **MPI bindings.**  MPI state-vector is C-only because no high-level
  runtime co-hosts an MPI rank cleanly.
- **Crypto bindings in Rust / JS.**  ML-KEM and the QRNG are C-only.
  These can be added in v1.1 if there's demand; they're not
  cloud-platform-critical because the control plane handles auth via
  HMAC-SHA3 at a layer above.
- **GPU backends in non-C bindings.**  The Metal / CUDA / OpenCL /
  Vulkan / cuQuantum / Eshkol kernels run inside the C library; you
  drive them implicitly when the Python / Rust binding calls into a
  simulator function that picks the active backend.  No language-level
  toggle is exposed because the backend choice is a C-side runtime
  decision.
