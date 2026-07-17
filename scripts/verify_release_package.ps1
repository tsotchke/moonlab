[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Package,

    [ValidateSet('x64', 'ARM64')]
    [string]$Platform = 'x64',
    [string]$Generator,
    [string]$WorkDir,
    [switch]$KeepWorkDir
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packagePath = if ([IO.Path]::IsPathRooted($Package)) {
    [IO.Path]::GetFullPath($Package)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Package))
}
if (-not (Test-Path $packagePath)) {
    throw "release package not found: $packagePath"
}

if ([string]::IsNullOrWhiteSpace($Generator)) {
    $cmakeHelp = (cmake --help) -join "`n"
    foreach ($candidate in @('Visual Studio 18 2026', 'Visual Studio 17 2022')) {
        if ($cmakeHelp -match [regex]::Escape($candidate)) {
            $Generator = $candidate
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($Generator)) {
        throw 'No supported Visual Studio CMake generator was found'
    }
}

$ownsWorkDir = [string]::IsNullOrWhiteSpace($WorkDir)
if ($ownsWorkDir) {
    $WorkDir = Join-Path ([IO.Path]::GetTempPath()) (
        'moonlab-release-consumer-{0}-{1}' -f $PID, [guid]::NewGuid().ToString('N'))
} elseif (-not [IO.Path]::IsPathRooted($WorkDir)) {
    $WorkDir = Join-Path $root $WorkDir
}
$workPath = [IO.Path]::GetFullPath($WorkDir)

if (Test-Path $workPath) {
    Remove-Item -LiteralPath $workPath -Recurse -Force
}
$prefix = Join-Path $workPath 'prefix'
$consumerDir = Join-Path $workPath 'consumer'
$consumerBuild = Join-Path $workPath 'build'
New-Item -ItemType Directory -Force -Path $prefix, $consumerDir | Out-Null

try {
    Expand-Archive -LiteralPath $packagePath -DestinationPath $prefix -Force

    $required = @(
        'bin\quantumsim.dll',
        'lib\quantumsim.lib',
        'include\moonlab\moonlab_export.h',
        'include\moonlab_features.h',
        'include\moonlab_build_info.h',
        'include\quantumsim\algorithms\tensor_network\ca_mps.h',
        'lib\cmake\quantumsim\quantumsim-config.cmake'
    )
    foreach ($relativePath in $required) {
        if (-not (Test-Path (Join-Path $prefix $relativePath))) {
            throw "release package is missing consumer entry: $relativePath"
        }
    }

    # Installed targets must not retain a compiler-runtime or build-tree path
    # from the hosted runner. Relative _IMPORT_PREFIX entries are expected;
    # drive-qualified paths make the ZIP unusable on a different machine.
    $targetMetadata = Get-ChildItem `
        -Path (Join-Path $prefix 'lib\cmake\quantumsim') `
        -Filter 'quantumsim-targets*.cmake' -File
    foreach ($metadataFile in $targetMetadata) {
        $absolutePathLeak = Select-String `
            -LiteralPath $metadataFile.FullName `
            -Pattern '[A-Za-z]:[\\/]' -Quiet
        if ($absolutePathLeak) {
            throw "installed CMake metadata contains a drive-qualified path: $($metadataFile.FullName)"
        }
    }

    @'
cmake_minimum_required(VERSION 3.20)
project(moonlab_release_consumer C)

find_package(quantumsim CONFIG REQUIRED)
if(NOT TARGET quantumsim::quantumsim)
    message(FATAL_ERROR "quantumsim::quantumsim was not exported")
endif()
if(NOT DEFINED MOONLAB_EXPECT_PREFIX)
    message(FATAL_ERROR "MOONLAB_EXPECT_PREFIX is required")
endif()
file(REAL_PATH "${MOONLAB_EXPECT_PREFIX}/include" expected_include)
file(REAL_PATH "${QUANTUMSIM_INCLUDE_DIRS}" actual_include)
if(NOT actual_include STREQUAL expected_include)
    message(FATAL_ERROR "non-relocatable include path: ${actual_include}")
endif()

add_executable(moonlab_release_consumer consumer.c)
target_link_libraries(moonlab_release_consumer PRIVATE quantumsim::quantumsim)
'@ | Set-Content -LiteralPath (Join-Path $consumerDir 'CMakeLists.txt') -Encoding utf8

    @'
#include <stdio.h>
#include <moonlab/moonlab_export.h>
#include <moonlab_features.h>

int main(void) {
    int major = -1;
    int minor = -1;
    int patch = -1;
    moonlab_abi_version(&major, &minor, &patch);
    if (major < 0 || minor < 0 || patch < 0) {
        return 1;
    }
    printf("moonlab release consumer OK %d.%d.%d features=%s\n",
           major, minor, patch, MOONLAB_VERSION_STRING);
    return 0;
}
'@ | Set-Content -LiteralPath (Join-Path $consumerDir 'consumer.c') -Encoding utf8

    & cmake -S $consumerDir -B $consumerBuild `
        -G $Generator -A $Platform -T ClangCL `
        "-DCMAKE_PREFIX_PATH=$prefix" `
        "-DMOONLAB_EXPECT_PREFIX=$prefix"
    if ($LASTEXITCODE -ne 0) {
        throw "consumer configure failed with exit code $LASTEXITCODE"
    }
    & cmake --build $consumerBuild --config Release --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "consumer build failed with exit code $LASTEXITCODE"
    }

    $consumerExe = Join-Path $consumerBuild 'Release\moonlab_release_consumer.exe'
    $env:PATH = "$(Join-Path $prefix 'bin');$env:PATH"
    & $consumerExe
    if ($LASTEXITCODE -ne 0) {
        throw "consumer smoke failed with exit code $LASTEXITCODE"
    }

    Write-Host "[verify-release] package consumer verification passed: $packagePath"
} finally {
    if ($ownsWorkDir -and -not $KeepWorkDir -and (Test-Path $workPath)) {
        Remove-Item -LiteralPath $workPath -Recurse -Force
    } elseif (Test-Path $workPath) {
        Write-Host "[verify-release] keeping work directory: $workPath"
    }
}
