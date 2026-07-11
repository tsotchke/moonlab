#!/usr/bin/env bash
# Package a CMake-built Moonlab tree using the install manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
OUTPUT=""
KEEP_STAGING="${MOONLAB_PACKAGE_KEEP_STAGING:-0}"
STAGING_DIR="${MOONLAB_PACKAGE_STAGING_DIR:-}"

usage() {
    cat <<EOF
Usage: $0 --output <tar.gz> [--build-dir <build-dir>]

Packages an existing CMake build by running:
  cmake --install <build-dir> --prefix <staging>

The resulting archive includes the installed library, CMake package files,
all installed public headers, README.md, and LICENSE.

Environment:
  MOONLAB_PACKAGE_STAGING_DIR   Reuse this staging directory.
  MOONLAB_PACKAGE_KEEP_STAGING  Keep temporary staging directory when set to 1.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    echo "--output is required" >&2
    usage >&2
    exit 2
fi

BUILD_DIR="$(cd "$ROOT" && cd "$BUILD_DIR" && pwd)"
case "$OUTPUT" in
    /*)
        mkdir -p "$(dirname "$OUTPUT")"
        ;;
    *)
        OUTPUT="$ROOT/$OUTPUT"
        mkdir -p "$(dirname "$OUTPUT")"
        ;;
esac

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "CMake build directory not configured: $BUILD_DIR" >&2
    exit 2
fi

if [[ -z "$STAGING_DIR" ]]; then
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-package.XXXXXX")"
    if [[ "$KEEP_STAGING" != "1" ]]; then
        trap 'rm -rf "$STAGING_DIR"' EXIT
    fi
else
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"
fi

cmake --install "$BUILD_DIR" --prefix "$STAGING_DIR"
cp "$ROOT/README.md" "$ROOT/LICENSE" "$STAGING_DIR/"

required=(
    "lib"
    "include/moonlab/moonlab_export.h"
    "include/moonlab/moonlab_api.h"
    "include/moonlab_features.h"
    "include/moonlab_build_info.h"
    "include/quantumsim/applications/moonlab_export.h"
    "include/quantumsim/applications/moonlab_api.h"
    "include/quantumsim/algorithms/tensor_network/ca_mps.h"
    "lib/cmake/quantumsim/quantumsim-config.cmake"
    "lib/pkgconfig/quantumsim.pc"
    "README.md"
    "LICENSE"
)

for rel in "${required[@]}"; do
    if [[ ! -e "$STAGING_DIR/$rel" ]]; then
        echo "release package missing required entry: $rel" >&2
        exit 1
    fi
done

if ! compgen -G "$STAGING_DIR/lib/libquantumsim.*" >/dev/null; then
    echo "release package missing libquantumsim artifact" >&2
    exit 1
fi

tar -czf "$OUTPUT" -C "$STAGING_DIR" .
echo "$OUTPUT"
