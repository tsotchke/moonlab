# MoonLab documentation

**Current release:** 1.2.0 (2026-07-18)
**Stable C ABI:** 0.6.0
**Supported release platforms:** Linux x86-64/ARM64, macOS Apple
Silicon/Intel, and Windows x64/ARM64

This is the canonical entry point for MoonLab documentation. The root
[`README.md`](../README.md) is the product overview; this page routes to the
documents that describe the current build, API, architecture, and operations.

## Start here

| Goal | Document |
|---|---|
| Build and run a first circuit | [Getting started](getting-started.md) |
| Build or consume MoonLab on Windows | [Windows guide](WINDOWS.md) |
| Understand CI and release artifacts | [CI/CD pipelines](CI_CD.md) |
| See what changed in 1.2.0 | [v1.2.0 release notes](release/v1.2.0-release-notes.md) |
| Use the supported binary interface | [Stable ABI](STABLE_ABI.md) |
| Choose build flags | [Configuration options](reference/configuration-options.md) |
| Understand the implementation | [Architecture](../ARCHITECTURE.md) |
| Understand platform design commitments | [Platform specification](../PLATFORM.md) |

## Current v1.2 surface

MoonLab 1.2 adds bounded CUDA/MPI sharding beyond 32 qubits, promotes the
binding-consumed public surface under hidden visibility, and advances the
stable ABI to 0.6.0. States created with `quantum_state_create_gpu()` retain the
same gate API used by CPU states, while distributed gates exchange bounded
chunks rather than allocating full remote shards.

The distributable CPU library remains the broad compatibility target. Official
release archives are built with native-CPU tuning disabled and contain the
shared library, public headers, CMake package metadata, pkg-config metadata,
license, changelog, and README. Windows archives are native ZIP packages built
with ClangCL because MSVC's C frontend does not implement the C99 complex
arithmetic used by the simulator.

## Tutorials

- [Tutorial index](tutorials/README.md)
- [First circuit and Bell pair](tutorials/getting_started.md)
- [MPDO noisy-circuit simulation](tutorials/mpdo_noise.md)
- [Topological band structure](tutorials/topological_band_structure.md)
- [Adaptive-bond TDVP](tutorials/adaptive_bond_tdvp.md)
- [Cross-binding research workflow](tutorials/research_workflow.md)

## API and integration reference

- [Gate reference](reference/gate-reference.md)
- [Error codes](reference/error-codes.md)
- [QGT API](reference/qgt-api.md)
- [MPDO API](reference/mpdo-api.md)
- [TDVP API](reference/tdvp-api.md)
- [Binding parity matrix](PARITY_MATRIX.md)
- [libirrep and SbNN integration](INTEGRATION_libirrep_SbNN.md)

Language-specific guides live with their packages:

- [Python](../bindings/python/README.md)
- [Rust](../documents/api/rust/index.md)
- [JavaScript/WASM](../bindings/javascript/README.md)

## Deployment and operations

- [Control plane](CONTROL_PLANE.md)
- [Docker deployment](../deploy/docker/README.md)
- [Helm chart](../deploy/helm/moonlab/README.md)
- [WebSocket gateway](../tools/gateway/README.md)
- [Stable ABI](STABLE_ABI.md)

## Documentation lifecycle

Release notes and current guides describe shipping behavior. Files under
`docs/release/` for older versions and dated audit reports are historical
records; version numbers in them intentionally remain pinned to the release
they audited. `PLATFORM.md` is a prescriptive design contract and labels
historical roadmap language explicitly. `CHANGELOG.md` is the authoritative
version-by-version ledger.
