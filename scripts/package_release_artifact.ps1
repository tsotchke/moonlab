[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$BuildDir = 'build-windows',
    [string]$Configuration = 'Release',
    [string]$StagingDir,
    [switch]$KeepStaging
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildPath = if ([IO.Path]::IsPathRooted($BuildDir)) {
    [IO.Path]::GetFullPath($BuildDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $BuildDir))
}
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Output))
}

if (-not (Test-Path (Join-Path $buildPath 'CMakeCache.txt'))) {
    throw "CMake build directory is not configured: $buildPath"
}

$ownsStaging = [string]::IsNullOrWhiteSpace($StagingDir)
if ($ownsStaging) {
    $StagingDir = Join-Path ([IO.Path]::GetTempPath()) (
        'moonlab-package-{0}-{1}' -f $PID, [guid]::NewGuid().ToString('N'))
} elseif (-not [IO.Path]::IsPathRooted($StagingDir)) {
    $StagingDir = Join-Path $root $StagingDir
}
$stagingPath = [IO.Path]::GetFullPath($StagingDir)

if (Test-Path $stagingPath) {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingPath | Out-Null

try {
    & cmake --install $buildPath --config $Configuration --prefix $stagingPath
    if ($LASTEXITCODE -ne 0) {
        throw "cmake --install failed with exit code $LASTEXITCODE"
    }

    foreach ($doc in @('README.md', 'LICENSE', 'CHANGELOG.md')) {
        $source = Join-Path $root $doc
        if (Test-Path $source) {
            Copy-Item -LiteralPath $source -Destination $stagingPath
        }
    }

    $required = @(
        'include\moonlab\moonlab_export.h',
        'include\moonlab\moonlab_api.h',
        'include\moonlab_features.h',
        'include\moonlab_build_info.h',
        'include\quantumsim\applications\moonlab_export.h',
        'include\quantumsim\applications\moonlab_api.h',
        'include\quantumsim\algorithms\tensor_network\ca_mps.h',
        'lib\cmake\quantumsim\quantumsim-config.cmake',
        'lib\pkgconfig\quantumsim.pc',
        'README.md',
        'LICENSE'
    )
    foreach ($relativePath in $required) {
        if (-not (Test-Path (Join-Path $stagingPath $relativePath))) {
            throw "release package is missing required entry: $relativePath"
        }
    }

    $dlls = @(Get-ChildItem -Path (Join-Path $stagingPath 'bin') -Filter 'quantumsim*.dll' -File -ErrorAction SilentlyContinue)
    $importLibraries = @(Get-ChildItem -Path (Join-Path $stagingPath 'lib') -Filter 'quantumsim*.lib' -File -ErrorAction SilentlyContinue)
    if ($dlls.Count -eq 0 -or $importLibraries.Count -eq 0) {
        throw 'release package must contain a quantumsim DLL and import library'
    }

    $outputDirectory = Split-Path -Parent $outputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    if (Test-Path $outputPath) {
        Remove-Item -LiteralPath $outputPath -Force
    }
    Compress-Archive -Path (Join-Path $stagingPath '*') -DestinationPath $outputPath -CompressionLevel Optimal
    Write-Output $outputPath
} finally {
    if ($ownsStaging -and -not $KeepStaging -and (Test-Path $stagingPath)) {
        Remove-Item -LiteralPath $stagingPath -Recurse -Force
    } elseif (Test-Path $stagingPath) {
        Write-Host "[package-release] keeping staging directory: $stagingPath"
    }
}
