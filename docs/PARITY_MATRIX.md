# Cross-language parity matrix

Coverage of Moonlab capabilities across the four bindings as of v1.2.0.

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
| MPS state + SVD truncation              | ✅  | `tdvp.py`   | `tdvp.rs`         | `tensor-network.ts` ⁴ |
| Two-site adaptive TDVP (entropy-PID)    | ✅  | `tdvp.py`   | `tdvp.rs`         | `tdvp.ts`         |
| MPDO noise simulator                    | ✅  | `mpdo.py`   | `mpdo.rs`         | `mpdo.ts`         |
| DMRG (variational ground state)         | ✅  | `dmrg.py`   | `dmrg.rs`         | `tensor-network.ts` ¹ |
| CA-MPS (Clifford-assisted MPS)          | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts`       |
| CA-MPS variational-D (Clifford search)  | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts` ²     |
| CA-MPS Born-rule sampling (since v0.10) | ✅  | `ca_mps.py` | `ca_mps.rs`       | `ca-mps.ts` ³     |
| CA-PEPS 2D                              | ✅  | `ca_peps.py`| `ca_peps.rs`      | `ca-peps.ts`      |

## Quantum geometry + topology

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Berry curvature grid integrator         | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Chern number (FHS + parallel-transport) | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Real-space Bianco-Resta Chern marker    | ✅  | `topology.py` | `topology.rs`   | `topology.ts`     |
| Haldane / Kane-Mele / BHZ / Kitaev / Hofstadter models | ✅ | `topology.py` | `topology.rs` | `topology.ts` |
| Kane-Mele full Rashba (since v0.10)     | ◐ ¹⁴ | `topology.py` | `topology.rs`  | `topology.ts`     |
| Z_2 invariant (block-Chern + Wilson-loop, since v0.10) | ✅ | `topology.py` | `topology.rs` | `topology.ts` |
| Quasicrystal Chern mosaic (Bianco-Resta KPM, L=O(300)) | ✅  | `topology.py` | `topology.rs` ⁵ | ✗ ⁵             |

## Topological QC + QEC

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Surface code (rotated, d × d)           | ✅  | `surface_code.py` | `surface_code.rs` | `surface-code.ts`|
| Threshold sweep harness                 | ✅  | `surface_code.py` | `surface_code.rs` | `surface-code.ts`|
| Decoder zoo (GREEDY, MWPM-exact, SBNN, LIBIRREP_SS, PyMatching) | ✅ ¹⁵ | `decoder.py` | `decoder.rs` | `decoder.ts` |
| QGTL circuit ingestion                  | ✅  | `qgtl.py`   | `qgtl.rs`         | `qgtl.ts`         |
| libirrep QEC zoo bridge (opt-in)        | ✅ ⁶ | `libirrep_qec.py` | `libirrep_qec.rs` | `libirrep-qec.ts` |
| Z_2 1+1D lattice gauge theory           | ✅  | `ca_mps.py` | `z2_lgt.rs`      | `ca-mps.ts`       |

## Distributed + cloud

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Distributed scheduler MVP               | ✅  | `scheduler.py` | `scheduler.rs` | `scheduler.ts`    |
| MPI state-vector partitioning           | ✅ ⁷ | ✗          | ✗                | ✗                |
| Control plane (line protocol)           | ✅  | `control_plane.py` | `control_plane.rs` | `control-plane.ts` ⁸ |
| Control plane TLS / mTLS                | ✅  | `control_plane.py` | `control_plane.rs` ⁹ | `control-plane.ts` ⁸ |
| Control plane HMAC-SHA3 auth            | ✅  | `control_plane.py` | `control_plane.rs` | `control-plane.ts` ⁸ |
| Control plane admission hook (since v1.0.3) | ✅ | `control_plane.py` | ✗ ⁹          | ✗                |
| Token-bucket rate limiter (`utils/token_bucket.h`) | ✅ | `token_bucket.py` | `token_bucket.rs` | ✗    |
| Audit buffer (`utils/audit_buffer.h`)   | ✅  | ✗          | ✗                | ✗                |
| Prometheus exporter sidecar             | ✅ ¹⁰| -          | -                 | -                 |
| WebSocket gateway (since v1.0)          | ✅ ¹¹| -          | -                 | -                 |
| Docker production stack                 | ✅ ¹²| -          | -                 | -                 |

## Differentiable physics + ML

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| VQE exact gradient (`moonlab_vqe_gradient`, ABI 0.4.0) | ✅ | ✗ | ◐ ¹⁶      | ✗                |
| Native reverse-mode autograd circuit (`differentiable.h`) | ✅ | `diff.py` | ✗       | ✗                |
| Feynman diagram rendering (ASCII/SVG/TikZ) | ✅ | `visualization/feynman.py` | `feynman.rs` | ✗    |
| QSVM / QuantumKernel / QuantumPCA       | ✅ (primitives) | `ml.py` | ✗       | ✗                |
| PyTorch quantum layer (parameter-shift autograd) | ✅ (primitives) | `torch_layer.py` | ✗ | ✗          |
| WASM Grover + classical H2 VQE package  | -- | -- | --                | `@moonlab/quantum-algorithms` ¹⁷ |
| Canvas2D / WebGL quantum visualizations | -- | -- | --                | `@moonlab/quantum-viz`         |

## Crypto + RNG

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| ML-KEM-512 / 768 / 1024 (FIPS 203)      | ✅  | `crypto.mlkem` | ✗             | ✗                |
| Conditioned hybrid RNG + assurance status | ✅  | `crypto.mlkem` | ✗           | ✗                |
| QRNG live status (`moonlab_qrng_get_status`, ABI 0.5.0) | ✅ | `crypto.mlkem.qrng_status()` | ✗ | ✗   |
| SHA3 / AES / DRBG primitives            | ✅  | ✗          | ✗                | ✗                |

## GPU + acceleration

| Capability                              | C   | Python      | Rust              | JS                |
|-----------------------------------------|-----|-------------|-------------------|-------------------|
| Apple Metal backend                     | ✅  | ✗          | ✗                | ✗                |
| Apple AMX (via Accelerate)              | ✅  | ✗          | ✗                | ✗                |
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

² All four bindings expose the variational-D Clifford search: JS as
`varDRun` in `ca-mps.ts`, Python as `ca_mps.var_d_run_v2`, Rust as
`moonlab::ca_mps::var_d_run`.  The TS layer marshals the
`ca_mps_var_d_alt_config_t` + `ca_mps_var_d_alt_result_t` structs into
the WASM heap by hand (offsets pinned in the source).

³ `moonlab_ca_mps_sample_z` reaches all four bindings (C + Python +
Rust + JS) on the v1.0 line.  JS calls `CaMps.sampleZ(numSamples,
randomValues)` against the rebuilt WASM blob; integration test in
`src/__tests__/ca-mps.integration.test.ts` checks Bell + GHZ_4 support.

⁴ JS binds the MPS state through `tensor-network.ts` but the
adaptive-bond TDVP driver is in the separate `tdvp.ts` module.

⁵ The Bianco-Resta Chern marker (`chern_kpm_*` API) is exposed in both
C++ and its Python and Rust bindings.  The current ceiling on the
sparse-stencil backend is L = O(300) (~90 000 sites); the C-side
`tests/performance/bench_chern_mosaic_hq.c` harness drives this directly
and ships the canonical PRL-reproduction artefacts.  No Python or Rust
driver currently re-implements the full PPM-emitting + manifest pipeline
(hence Rust is marked with this footnote rather than a bare ✅).  JS is
`✗` because the `chern_kpm_*` symbols are not in the WASM exports list
(the kernel is heavy and not a browser-target).

⁶ libirrep bridge requires `-DQSIM_ENABLE_LIBIRREP=ON` at build time.
Without it the bridge entries return `MOONLAB_LIBIRREP_NOT_BUILT = -201`
deliberately (caller can always call -- not a stub).

⁷ MPI state-vector requires `-DQSIM_ENABLE_MPI=ON` and an MPI runtime
(OpenMPI / MPICH).  No binding language exposes the MPI bridge because
none of the runtimes (CPython, Rust, V8) cleanly co-host an MPI rank.

⁸ The JS control-plane client is Node-only -- browsers can't open raw
TCP sockets, and the client-side implementation (including its TLS /
mTLS / HMAC transport) has no server-hosting counterpart in JS at all.
Use the WebSocket gateway for browser-side clients (see deploy/docker/
or footnote ¹¹).

⁹ Rust wires the control-plane server's general lifecycle
(`ControlPlaneServer::run` / `shutdown` / `set_rate_limit` /
`set_request_timeout` / `set_max_concurrent`), client-side TLS and mTLS
submission (`submit_circuit_tls`, `submit_circuit_mtls`), and HMAC-SHA3
auth (`hmac_sha3_256`, `submit_circuit_auth_tenant`) -- all real, not
stubs. What it does **not** yet bind is the *server*-side TLS listener
config (C's `moonlab_control_server_use_tls`) or the admission hook
(C's `moonlab_control_server_set_admission_hook`); both are Python-only
today (`ControlPlaneServer.use_tls()`, `.set_admission_hook()`).
Server-side Rust wrappers for both are in progress this cycle.

¹⁰ `tools/exporter/moonlab_control_exporter.py` is a Python
HTTP-bridge sidecar.  Exposed as a Docker image
`moonlab/control-exporter:0.10.0`; see `deploy/docker/`.

¹¹ `tools/gateway/moonlab_websocket_gateway.py` (since v1.0)
exposes the line protocol via WebSocket so browsers can submit
circuits.  Same Python image is used in `deploy/docker/`.

¹² The compose stack in `deploy/docker/docker-compose.yml` builds and
runs the control plane + exporter + Prometheus.

¹³ CUDA / OpenCL / Vulkan / cuQuantum compile under their respective
`QSIM_ENABLE_*` opt-in flags, but **none has a dedicated hosted-CI lane**
today -- `.github/workflows/ci.yml`'s `unix-matrix` job only builds the
`linux-x64-lite`, `linux-x64-mpi`, `linux-x64-asan`, `macos-arm64-lite`,
and `macos-arm64-mpi` configurations, none of which sets a GPU
`QSIM_ENABLE_*` flag. The native CUDA state-vector backend (moonlab's
own kernels, not cuQuantum) is validated out-of-band on a real Jetson
Xavier via `scripts/run_mesh_release_smoke.sh` before every release;
`.github/workflows/ci-jetson.yml` is wired for the same purpose inside
GitHub Actions but is `workflow_dispatch`-only with no self-hosted
runner currently enrolled. OpenCL, Vulkan, and cuQuantum have compile
coverage only (they build as part of local/manual `QSIM_ENABLE_*`
testing) and zero automated correctness coverage in CI as of this
writing. Apple Metal builds and is exercised locally on macOS hosts but
isn't part of hosted CI either.

¹⁴ `qgt_z2_invariant_pfaffian` (since v0.10.0) computes the Z_2
invariant for any 4-band, time-reversal-symmetric model where S_z is
not conserved, explicitly including non-zero Rashba coupling. However
the only Kane-Mele model constructor in the header,
`qgt_model_kane_mele`, still hard-rejects `lambda_r != 0` and returns
NULL (`qgt.h:587`) -- the invariant machinery supports full Rashba, but
there is currently no built-in way to construct a full-Rashba Kane-Mele
system to feed it. Full Rashba is functional only for custom
hand-built `qgt_system_n_t` instances, not the convenience constructor.

¹⁵ `decoder_bench.h` defines five dispatcher slots:
`MOONLAB_DECODER_GREEDY` and `MOONLAB_DECODER_MWPM_EXACT` are built-in;
`MOONLAB_DECODER_SBNN`, `MOONLAB_DECODER_LIBIRREP_SS`, and
`MOONLAB_DECODER_PYMATCHING` require an external library and return
`MOONLAB_DECODER_NOT_BUILT` (-401) when it isn't linked. There is no
belief-propagation (BP) decoder in the current tree. All four bindings
expose the same five-slot dispatcher uniformly, so the binding-parity
claim (✅) holds independent of which slots are actually built in a
given configuration.

¹⁶ `moonlab_vqe_gradient` is allowlisted in `moonlab-sys`'s bindgen
build (the raw FFI symbol is callable), but the high-level `moonlab`
Rust crate does not yet expose a safe wrapper for it in `vqe.rs`.

¹⁷ `@moonlab/quantum-algorithms` is a lean WASM package built on
`@moonlab/quantum-core`: a real WASM-backed `Grover` class, and an
H2-only `VQE` class that runs a classical (non-quantum-state) grid
search over a closed-form single-parameter H2 ansatz. It does not
implement QAOA.

## Current parity gaps

The following parity gaps are listed for transparency:

- **JS Node-only control-plane client.**  Browsers must go through the
  WebSocket gateway.
- **MPI bindings.**  MPI state-vector is C-only because no high-level
  runtime co-hosts an MPI rank cleanly.
- **Crypto bindings in Rust / JS.**  ML-KEM and the conditioned hybrid RNG
  are available through C and Python, but are not yet exposed by the Rust or
  JavaScript bindings.  The control plane's HMAC-SHA3 authentication remains
  independent of these convenience APIs.
- **GPU backends in non-C bindings.**  The Metal / CUDA / OpenCL /
  Vulkan / cuQuantum / Eshkol kernels run inside the C library; you
  drive them implicitly when the Python / Rust binding calls into a
  simulator function that picks the active backend.  No language-level
  toggle is exposed because the backend choice is a C-side runtime
  decision.
- **Rust control-plane server TLS + admission hook.**  Client-side TLS/
  mTLS/HMAC and the general server lifecycle are wired in Rust; the
  server-side TLS listener config and the per-tenant admission hook are
  Python-only today (see footnote ⁹). Rust server-side wrappers for both
  are in progress.
- **`moonlab_vqe_gradient` in Rust.**  Declared in the `moonlab-sys` FFI
  layer but has no safe high-level wrapper yet (see footnote ¹⁶).
