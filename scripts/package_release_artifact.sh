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

# The installed macOS dylib references @rpath/libomp.dylib so a consumer's
# own OpenMP runtime can satisfy it (issue #27).  A redistributable tarball
# has to resolve that reference on a machine with no Homebrew, so it also
# carries libomp beside libquantumsim.  The component is EXCLUDE_FROM_ALL so
# that only the tarball gets it -- a plain `cmake --install` into a system
# prefix must not shadow the package manager's libomp.
if [[ "$(uname -s)" == "Darwin" ]]; then
    cmake --install "$BUILD_DIR" --prefix "$STAGING_DIR" --component openmp-runtime
fi

cp "$ROOT/README.md" "$ROOT/LICENSE" "$STAGING_DIR/"

# CMAKE_INSTALL_LIBDIR is not always "lib".  GNUInstallDirs picks lib64 on
# plenty of 64-bit hosts, and every path below has to follow the tree that
# was actually installed rather than assume one.  Locate it by the package
# config file the install always emits under <libdir>/cmake/quantumsim.
qs_config="$(find "$STAGING_DIR" -maxdepth 5 \
    -path '*/cmake/quantumsim/quantumsim-config.cmake' -print -quit)"
if [[ -z "$qs_config" ]]; then
    echo "release package missing quantumsim-config.cmake under any libdir" >&2
    exit 1
fi
libdir_abs="${qs_config%/cmake/quantumsim/quantumsim-config.cmake}"
LIBDIR="${libdir_abs#"$STAGING_DIR"/}"

required=(
    "$LIBDIR"
    "include/moonlab/moonlab_export.h"
    "include/moonlab/moonlab_api.h"
    "include/moonlab_features.h"
    "include/moonlab_build_info.h"
    "include/quantumsim/applications/moonlab_export.h"
    "include/quantumsim/applications/moonlab_api.h"
    "include/quantumsim/algorithms/tensor_network/ca_mps.h"
    "$LIBDIR/cmake/quantumsim/quantumsim-config.cmake"
    "$LIBDIR/pkgconfig/quantumsim.pc"
    "README.md"
    "LICENSE"
)

for rel in "${required[@]}"; do
    if [[ ! -e "$STAGING_DIR/$rel" ]]; then
        echo "release package missing required entry: $rel" >&2
        exit 1
    fi
done

if ! compgen -G "$STAGING_DIR/$LIBDIR/libquantumsim.*" >/dev/null; then
    echo "release package missing libquantumsim artifact under $LIBDIR" >&2
    exit 1
fi

# Every @rpath dependency of the shipped dylib must resolve inside the
# package: INSTALL_RPATH is loader-relative only, so nothing outside the
# package libdir can satisfy it on a clean machine.  Catch a missing bundled
# runtime here rather than in the consumer's dyld abort.
if [[ "$(uname -s)" == "Darwin" ]] && command -v otool >/dev/null; then
    # A redistributed third-party runtime ships with its license text.
    if [[ -f "$STAGING_DIR/$LIBDIR/libomp.dylib" \
          && ! -f "$STAGING_DIR/share/licenses/libomp/LICENSE.TXT" ]]; then
        echo "release package bundles libomp without share/licenses/libomp/LICENSE.TXT" >&2
        exit 1
    fi
    for dylib in "$STAGING_DIR"/"$LIBDIR"/libquantumsim.*.dylib; do
        [[ -f "$dylib" && ! -L "$dylib" ]] || continue
        while read -r dep; do
            leaf="${dep#@rpath/}"
            if [[ ! -e "$STAGING_DIR/$LIBDIR/$leaf" ]]; then
                echo "release package dylib references $dep but $LIBDIR/$leaf is absent" >&2
                exit 1
            fi
        done < <(otool -L "$dylib" | awk '/^\t@rpath\//{print $1}' \
                    | grep -v '^@rpath/libquantumsim\.')
    done
fi

tar -czf "$OUTPUT" -C "$STAGING_DIR" .
echo "$OUTPUT"
