# CI/CD pipelines

**Applies to:** MoonLab 1.1.0 and later

MoonLab uses GitHub Actions for routine validation, Jetson hardware coverage,
and tagged releases. The Windows jobs follow the native packaging design used
by Eshkol: Visual Studio generators with ClangCL, per-architecture hosted
runners, install-tree ZIP staging, and a consumer test before upload.

## Workflows

| Workflow | File | Trigger | Purpose |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Pushes and pull requests | Native Unix/Windows builds, WASM build + full JS binding test suite, Python wheel build + install smoke, Docker, and hygiene checks |
| Linux compatibility | `.github/workflows/linux-compatibility.yml` | Relevant pushes/PRs, tags through the release caller, manual | Clean-distro source build, tests, package, and external-consumer verification |
| Jetson CI | `.github/workflows/ci-jetson.yml` | Manual dispatch only (no self-hosted runner is currently enrolled) | CUDA and ARM64 validation on the self-hosted Jetson runner; routine Jetson coverage happens out-of-band via `scripts/run_mesh_release_smoke.sh` |
| Release | `.github/workflows/release.yml` | Manual candidate dispatch; tags matching `v*` promote one certified candidate | Build and seal unpublished packages, then verify exact-byte promotion before publication |
| differential | `.github/workflows/differential.yml` | Nightly cron + manual | Cross-backend / cross-binding differential against a numpy reference oracle |
| fuzz | `.github/workflows/fuzz.yml` | Nightly cron + manual | Seed-corpus replay under ASan+UBSan plus per-surface libFuzzer soak |
| numerical | `.github/workflows/numerical.yml` | Nightly cron + manual | Numerical-edge harnesses, valgrind memcheck uninit sweep, MSan fallback legs |
| scaling | `.github/workflows/scaling.yml` | Nightly cron + manual | Large-n cross-backend differential with quarantined known divergences |
| Statistical Adversarial | `.github/workflows/statistical.yml` | Nightly cron + manual | QRNG battery, ML-KEM negative fuzz, entropy-health adversarial (fast ASan + deep lanes) |
| tsan | `.github/workflows/tsan.yml` | PRs touching concurrency surfaces, nightly cron, manual | ThreadSanitizer pthread-core and OpenMP/Archer concurrency lanes |

GPU execution and multi-host MPI are NOT covered by GitHub-hosted CI: no
hosted runner has a GPU this project targets, and multi-host MPI needs real
fleet endpoints. Those surfaces are validated out-of-band on the release
mesh (`scripts/run_mesh_release_smoke.sh`, `scripts/run_mpi_sharded_gpu.sh`)
and their evidence is bound into the release certificate; hosted CI covers
only the CPU fallbacks and the GPU/MPI code that compiles everywhere.

GitLab also runs the repository's `.gitlab-ci.yml`; GitHub Actions is the
source of the public release assets described here.

## Routine CI matrix

The Unix matrix builds:

| Lane | Runner | Variant |
|---|---|---|
| `linux-x64-lite` | `ubuntu-22.04` | CPU baseline |
| `linux-x64-mpi` | `ubuntu-22.04` | MPI enabled |
| `linux-x64-asan` | `ubuntu-22.04` | ASAN + UBSAN |
| `macos-arm64-lite` | `macos-14` | Apple Silicon CPU/Metal discovery |
| `macos-arm64-mpi` | `macos-14` | Apple Silicon + MPI |

The Windows matrix builds:

| Lane | Runner | CMake platform | Toolset |
|---|---|---|---|
| `windows-x64` | `windows-latest` | `x64` | `ClangCL` |
| `windows-arm64` | `windows-11-arm` | `ARM64` | `ClangCL` |

Windows jobs run the same six-stage driver used for releases:
`scripts/build_windows_artifact.ps1`. The smoke set covers core state/gates,
measurement, aligned memory, reproducibility manifests, and stable ABI loading.
The job then verifies a completely separate CMake consumer against the staged
ZIP, catching missing DLL/import-library/header/export metadata before upload.

Additional CI jobs build the Emscripten/WASM target and run the complete JS
binding test suite against the freshly built module (the vitest unit lane plus
the 200+ test integration lane), build and install-smoke the Python wheel
through the same scikit-build-core backend the release wheels use, smoke the
Debian release container, and reject private markers or credential-shaped
strings in the public tree.

The unix matrix's ctest invocation runs the registered unit and integration
tests without label filtering (minus the `long`/`memory_heavy` exclusions),
which includes the bit-packed Clifford tableau tests (`unit_clifford`,
`unit_clifford_rowsum`), the Pauli-frame batch sampler (`unit_pauli_frame`),
the union-find decoder (`unit_uf_decoder`), and the Python binding pytest
suite.

The Linux compatibility workflow runs the universal source-build driver on
Ubuntu 22.04/24.04/26.04 and Debian 12/13 on both hosted amd64 and ARM64
runners. Each of those ten canonical lanes binds the clean Git source identity,
resolved container-image digest, positive all-pass/zero-skip CTest counts, and
hashes for its build/test logs, install tree, package, and both external
consumer checks. A fail-closed aggregate requires exactly those ten profiles
to describe the identical source before reporting `PASS`.

Additional clean-family smokes cover AlmaLinux 8/9/10, current Fedora, current
Arch, current openSUSE Tumbleweed, and current Alpine; NixOS is verified on the
physical ARM64 Jetson mesh lane. These rows represent the mainstream package-
manager/libc families rather than claiming that every downstream derivative is
a distinct ABI. Every row disables native CPU tuning and fast-math, builds,
runs the bounded release labels including `health_tests`, creates the install-
tree archive, and verifies separate CMake and pkg-config consumers. A tagged
release cannot create its draft until this reusable workflow and its evidence
aggregate pass.

## Release workflow

Release is a two-phase candidate-and-promotion protocol. A manual
`workflow_dispatch` for an exact version builds and consumer-tests the complete
22-artifact release inventory without reading publication credentials or
mutating GitHub Releases, PyPI, npm, crates.io, or Homebrew. The candidate run
also executes the full hosted platform and Linux-compatibility matrix, then
seals the exact run ID, head SHA, artifact identities, sizes, and SHA-256
digests in `moonlab-release-candidate.json`.

After the local release oracle, MPI/mesh evidence, and external certificate are
green, creating and pushing the annotated release tag is a separate explicit
approval boundary. Its annotation must contain exactly these two lines:

```text
Moonlab-Release-Candidate-Run: <positive GitHub Actions run ID>
Moonlab-Release-Candidate-Head: <40-character lowercase commit SHA>
```

The tag must target the same head. The tag-triggered promotion path resolves
only that run, verifies the repository, workflow, dispatch event, head,
conclusion, complete job-name set, candidate manifest, and all 22 artifact
hashes, then re-uploads the verified bytes for downstream publication. Missing
or mismatched evidence is fatal; promotion never falls back to rebuilding.

Because Cargo has no supported option to publish a supplied, pre-certified
`.crate` archive, crates.io publication goes through the audited exact-byte
uploader `scripts/publish_crate_exact_bytes.py` (design and protocol audit in
`docs/release/rust-crate-exact-byte-publish.md`, tests in
`scripts/test_publish_crate_exact_bytes.py`). The `publication-readiness` job
fails closed if that uploader is absent, so no registry can be mutated through
a path that would re-package the certified bytes. With readiness green, the
exact tested wheels and npm tarballs are published, followed by the three
certified Rust crates in dependency order: `moonlab-sys`, `moonlab`, then
`moonlab-tui`. Stable tags then update and test Homebrew. The GitHub Release
leaves draft state only after all required registries succeed. Numbered
`alpha`, `beta`, or `rc` versions use the npm `next` tag and do not update
Homebrew.

### Native archives

| Asset pattern | Platform | Format |
|---|---|---|
| `moonlab-<tag>-linux-x64.tar.gz` | Linux x86-64 | tar.gz |
| `moonlab-<tag>-linux-arm64.tar.gz` | Linux ARM64 | tar.gz |
| `moonlab-<tag>-macos-arm64.tar.gz` | macOS Apple Silicon | tar.gz |
| `moonlab-<tag>-macos-x64.tar.gz` | macOS Intel | tar.gz |
| `moonlab-<tag>-windows-x64.zip` | Windows x64 | ZIP |
| `moonlab-<tag>-windows-arm64.zip` | Windows ARM64 | ZIP |

The Unix and Windows archives are generated from `cmake --install`, not by
copying a hand-maintained subset of build outputs. They therefore include the
same exported `quantumsim::quantumsim` target, public header tree, build-info
headers, pkg-config metadata, and project documents.

### Other release outputs

- Debian x86-64 and ARM64 packages
- Self-contained Python wheels for Linux, macOS, and Windows on x64/ARM64,
  uploaded with `PYPI_API_TOKEN`
- Five exact-tested JavaScript/WASM tarballs uploaded with `NPM_TOKEN`
- Three SDK-tested Rust crates uploaded with `CARGO_REGISTRY_TOKEN`
- Homebrew formula updates for stable tags with `HOMEBREW_TAP_TOKEN`

Missing credentials are release failures; publication is never silently
skipped. The repository secret names are exact and case-sensitive.

The Rust low-level crate discovers installed headers and libraries through
`quantumsim.pc`. Custom relocatable SDKs can instead set
`MOONLAB_INCLUDE_DIR` and `MOONLAB_LIB_DIR`; bindgen writes only to Cargo's
`OUT_DIR`, so registry sources remain read-only.

## Reproduce the gates locally

Unix package and consumer gate:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DQSIM_NATIVE_ARCH=OFF -DQSIM_BUILD_EXAMPLES=OFF \
  -DQSIM_BUILD_BENCHMARKS=OFF
cmake --build build-release -j
scripts/package_release_artifact.sh \
  --build-dir build-release --output dist/moonlab-local.tar.gz
scripts/verify_release_package.sh \
  --package dist/moonlab-local.tar.gz
```

Windows package and consumer gate:

```powershell
.\scripts\build_windows_artifact.ps1 `
  -Arch x64 `
  -BuildDir build-windows-x64 `
  -Output .\dist\moonlab-local-windows-x64.zip
```

## Maintenance rules

- Add Windows packaging behavior to the shared PowerShell driver rather than
  duplicating it between `ci.yml` and `release.yml`.
- Keep `QSIM_NATIVE_ARCH=OFF` for distributable binaries.
- Keep `QSIM_FAST_MATH=OFF` for release and correctness validation; enabling it
  explicitly permits non-IEEE NaN/Inf transformations.
- Preserve the external-consumer gate whenever install/export paths change.
- Run `python3 scripts/version_tool.py check` before tagging. Use
  `scripts/sync-versions.sh 1.2.0-rc.1` to synchronize a numbered prerelease;
  the tool maps it to the PEP 440 version `1.2.0rc1` for PyPI.
- A new release architecture is complete only when routine CI and tagged
  release jobs both build it and the documentation lists its artifact name.
