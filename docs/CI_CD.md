# CI/CD pipelines

**Applies to:** MoonLab 1.1.0 and later

MoonLab uses GitHub Actions for routine validation, Jetson hardware coverage,
and tagged releases. The Windows jobs follow the native packaging design used
by Eshkol: Visual Studio generators with ClangCL, per-architecture hosted
runners, install-tree ZIP staging, and a consumer test before upload.

## Workflows

| Workflow | File | Trigger | Purpose |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Pushes and pull requests | Native Unix/Windows builds, WASM, Docker, and hygiene checks |
| Jetson CI | `.github/workflows/ci-jetson.yml` | Workflow-defined hardware trigger | CUDA and ARM64 validation on the self-hosted Jetson runner |
| Release | `.github/workflows/release.yml` | Tags matching `v*` | Build and attach platform packages, Debian packages, and publish bindings |

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

Additional CI jobs build the Emscripten/WASM target, smoke the Debian release
container, and reject private markers or credential-shaped strings in the
public tree.

## Release workflow

A pushed tag matching `v*` first runs a fail-closed preflight: the tag must
match every version surface; PyPI, npm, and crates.io credentials must exist;
and stable tags also require the Homebrew tap credential. Platform artifacts
are built and consumer-tested into private workflow storage. Only after every
artifact passes does the workflow create a fully populated **draft** GitHub
Release.

The exact tested wheels and npm tarballs are then published. Rust crates are
verified against the packaged native SDK and published in dependency order:
`moonlab-sys`, `moonlab`, then `moonlab-tui`. Stable tags also build, audit,
install, and test the generated formula before pushing it to
`tsotchke/homebrew-moonlab`. The GitHub Release leaves draft state only after
all required registries succeed. Tags ending in numbered `alpha`, `beta`, or
`rc` identifiers are pre-releases and publish npm packages under `next`; they
do not update Homebrew.

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
