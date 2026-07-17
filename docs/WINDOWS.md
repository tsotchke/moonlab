# Windows builds and release packages

**Applies to:** MoonLab 1.1.0 and later
**Architectures:** x64 and ARM64
**Compiler:** ClangCL with a Visual Studio 2022-or-newer generator

MoonLab uses C99 complex arithmetic. The MSVC C frontend does not implement
that language surface, so native Windows builds must use the Visual Studio
ClangCL toolset. The official CI and release jobs enforce this contract on both
x64 and ARM64.

## Use a release ZIP

Download the archive matching the machine architecture:

- `moonlab-<tag>-windows-x64.zip`
- `moonlab-<tag>-windows-arm64.zip`

After extraction the package is a relocatable CMake prefix:

```text
bin/quantumsim.dll
lib/quantumsim.lib
lib/cmake/quantumsim/
lib/pkgconfig/quantumsim.pc
include/moonlab/
include/quantumsim/
README.md
CHANGELOG.md
LICENSE
```

Point `CMAKE_PREFIX_PATH` at the extracted directory and link the exported
target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_moonlab_program C)

find_package(quantumsim CONFIG REQUIRED)
add_executable(my_moonlab_program main.c)
target_link_libraries(my_moonlab_program PRIVATE quantumsim::quantumsim)
```

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL `
  -DCMAKE_PREFIX_PATH=C:\sdk\moonlab
cmake --build build --config Release
$env:PATH = "C:\sdk\moonlab\bin;$env:PATH"
.\build\Release\my_moonlab_program.exe
```

Use `-A ARM64` on native Windows ARM64. Keep the package `bin` directory on
`PATH` while running a dynamically linked consumer.

## Build from source

Install Visual Studio 2022 or newer with C++ build tools, CMake, and the
ClangCL component. From a PowerShell prompt at the repository root:

```powershell
.\scripts\build_windows_artifact.ps1 `
  -Arch x64 `
  -BuildDir build-windows-x64 `
  -Output .\dist\moonlab-local-windows-x64.zip
```

For ARM64, pass `-Arch arm64`. The driver:

1. selects a supported Visual Studio CMake generator;
2. configures a portable shared-library build with ClangCL;
3. builds the native library and test targets;
4. runs state, gate, measurement, memory, manifest, and stable-ABI smokes;
5. stages the CMake install tree into a ZIP; and
6. builds and runs a separate consumer against the extracted package.

The equivalent configure command is:

```powershell
cmake -S . -B build-windows-x64 `
  -G "Visual Studio 17 2022" -A x64 -T ClangCL `
  -DQSIM_BUILD_SHARED=ON `
  -DQSIM_BUILD_EXAMPLES=OFF `
  -DQSIM_BUILD_BENCHMARKS=OFF `
  -DQSIM_ENABLE_OPENMP=OFF `
  -DQSIM_ENABLE_CONTROL_PLANE=OFF `
  -DQSIM_ENABLE_TLS=OFF `
  -DQSIM_NATIVE_ARCH=OFF `
  -DQSIM_ENABLE_LTO=OFF
cmake --build build-windows-x64 --config Release --parallel
ctest --test-dir build-windows-x64 -C Release --output-on-failure
```

## Platform behavior

- Windows uses `BCryptGenRandom` for operating-system entropy.
- Tensor QR and SVD use the in-tree Householder/Jacobi fallbacks in official
  packages, so OpenBLAS is not a release dependency.
- The POSIX socket control-plane transport is disabled by default on Windows.
- Official Windows packages are CPU-only. CUDA, MPI, OpenCL, and Vulkan are
  opt-in source builds and are not part of the hosted Windows artifact gate.
- OpenMP is disabled in official packages to avoid redistributing a compiler
  runtime DLL. The core library remains multi-platform and consumers may enable
  OpenMP in their own source build.

## CI/CD artifacts

Every push covered by `ci.yml` and every pull request builds
`moonlab-ci-windows-x64` and `moonlab-ci-windows-arm64` workflow artifacts. A
tag matching `v*` publishes the versioned ZIPs as GitHub Release assets. Both
paths call the same `scripts/build_windows_artifact.ps1` driver, which keeps CI
and release package contents identical.

See [CI/CD pipelines](CI_CD.md) for the complete matrix and release flow.

## Troubleshooting

- **MSVC reports missing complex functions:** the build was configured with
  the default MSVC toolset. Reconfigure with `-T ClangCL`.
- **`quantumsim.dll` is not found:** add the extracted package's `bin`
  directory to `PATH` before starting the consumer.
- **CMake cannot find `quantumsim`:** set `CMAKE_PREFIX_PATH` to the extracted
  package root, not its `lib` directory.
- **Wrong machine type:** use the x64 package on x64 Windows and the ARM64
  package on native ARM64 Windows; the archives are not universal binaries.
