[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Output,

    [ValidateSet('x64', 'arm64')]
    [string]$Arch = 'x64',
    [string]$BuildDir = 'build-windows',
    [string]$Configuration = 'Release',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildPath = if ([IO.Path]::IsPathRooted($BuildDir)) {
    [IO.Path]::GetFullPath($BuildDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $BuildDir))
}
$platform = if ($Arch -eq 'arm64') { 'ARM64' } else { 'x64' }

$cmakeHelp = (cmake --help) -join "`n"

# Common configure arguments; the generator is chosen by the fallback loop
# below. cmake --help LISTS every generator it knows, but a listed VS version
# may have no installed instance on a given runner (notably the ARM runner
# lists 'Visual Studio 18 2026' but only ships an older toolset). Try each
# candidate for real and keep the first that configures, cleaning the build
# tree between attempts so a partial cache from a failed generator never leaks.
$commonArgs = @(
    '-S', $root,
    '-B', $buildPath,
    '-A', $platform,
    '-T', 'ClangCL',
    '-DQSIM_BUILD_SHARED=ON',
    '-DQSIM_BUILD_STATIC=OFF',
    '-DQSIM_BUILD_TESTS=ON',
    '-DQSIM_BUILD_EXAMPLES=OFF',
    '-DQSIM_BUILD_BENCHMARKS=OFF',
    '-DQSIM_ENABLE_OPENMP=OFF',
    '-DQSIM_ENABLE_CONTROL_PLANE=OFF',
    '-DQSIM_ENABLE_TLS=OFF',
    '-DQSIM_NATIVE_ARCH=OFF',
    '-DQSIM_ENABLE_LTO=OFF',
    '-DQSIM_WERROR=ON'
)

$generator = $null
foreach ($candidate in @('Visual Studio 18 2026', 'Visual Studio 17 2022')) {
    if (-not ($cmakeHelp -match [regex]::Escape($candidate))) { continue }
    Write-Host "Trying generator '$candidate', ClangCL, platform $platform"
    if (Test-Path $buildPath) { Remove-Item -Recurse -Force $buildPath }
    & cmake '-G' $candidate @commonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Configured with '$candidate'"
        $generator = $candidate
        break
    }
    Write-Host "Generator '$candidate' did not configure (exit $LASTEXITCODE); trying next"
}
if (-not $generator) {
    throw 'Moonlab configure failed: no installed Visual Studio generator worked'
}

& cmake --build $buildPath --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Moonlab build failed with exit code $LASTEXITCODE"
}

if (-not $SkipTests) {
    $env:MOONLAB_SKIP_HW_ENTROPY = '1'
    & ctest --test-dir $buildPath -C $Configuration `
        --output-on-failure --timeout 300 `
        -R '^(unit_quantum_state|unit_quantum_gates|unit_measurement|unit_memory_align|unit_manifest|abi_moonlab_export)$'
    if ($LASTEXITCODE -ne 0) {
        throw "Moonlab Windows smoke tests failed with exit code $LASTEXITCODE"
    }
}

& (Join-Path $PSScriptRoot 'package_release_artifact.ps1') `
    -BuildDir $buildPath -Configuration $Configuration -Output $Output
if ($LASTEXITCODE -ne 0) {
    throw "Moonlab packaging failed with exit code $LASTEXITCODE"
}

& (Join-Path $PSScriptRoot 'verify_release_package.ps1') `
    -Package $Output -Platform $platform -Generator $generator
if ($LASTEXITCODE -ne 0) {
    throw "Moonlab package verification failed with exit code $LASTEXITCODE"
}
