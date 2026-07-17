#!/usr/bin/env bash
# Verify that a Debian package contains a usable Moonlab install tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE=""
WORK_DIR="${MOONLAB_VERIFY_DEB_WORK_DIR:-}"
KEEP_WORK_DIR="${MOONLAB_VERIFY_KEEP_WORK_DIR:-0}"
SKIP_RUN="${MOONLAB_VERIFY_DEB_SKIP_RUN:-0}"

usage() {
    cat <<EOF
Usage: $0 --package <moonlab.deb> [--work-dir <dir>] [--keep-work-dir] [--skip-run]

Extracts a Debian package, verifies the installed Moonlab headers and package
metadata, builds a tiny external CMake consumer against quantumsim::quantumsim,
and runs it unless --skip-run is set.  When pkg-config is available, it also
compiles the same consumer through the installed quantumsim.pc.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --package)
            PACKAGE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --keep-work-dir)
            KEEP_WORK_DIR=1
            shift
            ;;
        --skip-run)
            SKIP_RUN=1
            shift
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

if [[ -z "$PACKAGE" ]]; then
    echo "--package is required" >&2
    usage >&2
    exit 2
fi

case "$PACKAGE" in
    /*) ;;
    *) PACKAGE="$ROOT/$PACKAGE" ;;
esac
if [[ ! -f "$PACKAGE" ]]; then
    echo "Debian package not found: $PACKAGE" >&2
    exit 2
fi
if ! command -v dpkg-deb >/dev/null; then
    echo "dpkg-deb is required to verify Debian packages" >&2
    exit 2
fi

if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-deb-consumer.XXXXXX")"
else
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
fi

if [[ "$KEEP_WORK_DIR" != "1" ]]; then
    trap 'rm -rf "$WORK_DIR"' EXIT
else
    echo "[verify-deb] keeping work dir: $WORK_DIR"
fi

PREFIX="$WORK_DIR/root"
CONSUMER_DIR="$WORK_DIR/consumer"
CMAKE_BUILD_DIR="$WORK_DIR/cmake-build"
mkdir -p "$PREFIX" "$CONSUMER_DIR"

dpkg-deb --info "$PACKAGE" >/dev/null
dpkg-deb --extract "$PACKAGE" "$PREFIX"

required=(
    "usr/include/moonlab/moonlab_export.h"
    "usr/include/moonlab/moonlab_api.h"
    "usr/include/moonlab_features.h"
    "usr/include/moonlab_build_info.h"
    "usr/include/quantumsim/algorithms/tensor_network/ca_mps.h"
)

for rel in "${required[@]}"; do
    if [[ ! -e "$PREFIX/$rel" ]]; then
        echo "Debian package missing consumer entry: $rel" >&2
        exit 1
    fi
done

cmake_config="$(find "$PREFIX/usr" \
    -path '*/cmake/quantumsim/quantumsim-config.cmake' \
    -type f -print -quit)"
pkg_config="$(find "$PREFIX/usr" -path '*/pkgconfig/quantumsim.pc' \
    -type f -print -quit)"
shared_lib="$(find "$PREFIX/usr" -name 'libquantumsim.so*' \
    \( -type f -o -type l \) -print -quit)"

if [[ -z "$cmake_config" ]]; then
    echo "Debian package missing quantumsim CMake config" >&2
    exit 1
fi
if [[ -z "$pkg_config" ]]; then
    echo "Debian package missing quantumsim pkg-config file" >&2
    exit 1
fi
if [[ -z "$shared_lib" ]]; then
    echo "Debian package missing libquantumsim shared library" >&2
    exit 1
fi

PKG_CONFIG_DIR="$(dirname "$pkg_config")"
LIB_DIR="$(dirname "$shared_lib")"

# ldd resolution check: confirm every shared library libquantumsim.so
# actually NEEDs (libgomp, liblapacke, libopenblas/libblas, libssl, ...)
# resolves on this host, i.e. the Depends: field in DEBIAN/control is
# not missing a runtime package. Deliberately run with no
# LD_LIBRARY_PATH override so this reflects what `apt install
# moonlab.deb` gets on a plain system, not the CMake-consumer run
# below (which does set LD_LIBRARY_PATH). ldd is Linux-specific
# (dpkg packages are a Linux concept in the first place); on macOS
# this step degrades to a no-op and the rest of the script's
# structural checks still run.
if [[ "$(uname -s)" == "Linux" ]]; then
    if command -v ldd >/dev/null; then
        echo "[verify-deb] checking ldd resolution for $shared_lib"
        ldd_out="$(ldd "$shared_lib" 2>&1)" || {
            echo "$ldd_out" >&2
            echo "ldd failed to process the extracted libquantumsim.so" >&2
            exit 1
        }
        echo "$ldd_out"
        if unresolved="$(grep -E 'not found' <<<"$ldd_out")"; then
            echo "libquantumsim.so has unresolved dynamic dependencies:" >&2
            echo "$unresolved" >&2
            echo "DEBIAN/control's Depends: field is likely missing a runtime package." >&2
            exit 1
        fi
    else
        echo "[verify-deb] ldd not available; skipping dynamic-dependency resolution check" >&2
    fi
else
    echo "[verify-deb] ldd resolution check requires Linux; running structural checks only on $(uname -s)"
fi

cat >"$CONSUMER_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(moonlab_deb_consumer C)

find_package(quantumsim CONFIG REQUIRED)

if(NOT TARGET quantumsim::quantumsim)
    message(FATAL_ERROR "quantumsim::quantumsim target was not exported")
endif()

if(NOT DEFINED MOONLAB_EXPECT_PREFIX)
    message(FATAL_ERROR "MOONLAB_EXPECT_PREFIX is required")
endif()

file(REAL_PATH "${MOONLAB_EXPECT_PREFIX}/include" expected_include)
file(REAL_PATH "${QUANTUMSIM_INCLUDE_DIRS}" actual_include)
if(NOT actual_include STREQUAL expected_include)
    message(FATAL_ERROR
        "QUANTUMSIM_INCLUDE_DIRS is not relocatable: "
        "${actual_include} != ${expected_include}")
endif()

foreach(header
        moonlab/moonlab_export.h
        moonlab/moonlab_api.h
        moonlab_features.h
        moonlab_build_info.h
        quantumsim/algorithms/tensor_network/ca_mps.h)
    if(NOT EXISTS "${QUANTUMSIM_INCLUDE_DIRS}/${header}")
        message(FATAL_ERROR "missing installed header: ${header}")
    endif()
endforeach()

add_executable(moonlab_deb_consumer consumer.c)
target_link_libraries(moonlab_deb_consumer PRIVATE quantumsim::quantumsim)
EOF

cat >"$CONSUMER_DIR/consumer.c" <<'EOF'
#include <complex.h>
#include <stdio.h>

#include <moonlab/moonlab_export.h>
#include <moonlab_features.h>
#include <quantumsim/algorithms/tensor_network/ca_mps.h>

int main(void) {
    int major = -1;
    int minor = -1;
    int patch = -1;
    moonlab_abi_version(&major, &minor, &patch);
    if (major < 0 || minor < 0 || patch < 0) {
        fprintf(stderr, "invalid ABI version %d.%d.%d\n", major, minor, patch);
        return 1;
    }

    moonlab_ca_mps_t* state = 0;
    (void)state;
    printf("moonlab deb consumer OK %d.%d.%d features=%s\n",
           major, minor, patch, MOONLAB_VERSION_STRING);
    return 0;
}
EOF

cmake -S "$CONSUMER_DIR" -B "$CMAKE_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PREFIX/usr" \
    -DMOONLAB_EXPECT_PREFIX="$PREFIX/usr"
cmake --build "$CMAKE_BUILD_DIR" -j

run_with_library_path() {
    local exe="$1"
    if [[ "$SKIP_RUN" == "1" ]]; then
        echo "[verify-deb] skip run: $exe"
        return 0
    fi

    case "$(uname -s)" in
        Darwin)
            DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}" "$exe"
            ;;
        Linux)
            LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" "$exe"
            ;;
        *)
            "$exe"
            ;;
    esac
}

run_with_library_path "$CMAKE_BUILD_DIR/moonlab_deb_consumer"

if command -v pkg-config >/dev/null && command -v cc >/dev/null; then
    export PKG_CONFIG_PATH="$PKG_CONFIG_DIR"
    pkg-config --exists quantumsim

    read -r -a pkg_cflags <<<"$(pkg-config --cflags quantumsim)"
    read -r -a pkg_libs <<<"$(pkg-config --libs quantumsim)"
    cc "${pkg_cflags[@]}" "$CONSUMER_DIR/consumer.c" \
        -o "$WORK_DIR/pkg-config-consumer" "${pkg_libs[@]}"
    run_with_library_path "$WORK_DIR/pkg-config-consumer"
else
    echo "[verify-deb] pkg-config or cc not available; skipping pkg-config consumer"
fi

echo "[verify-deb] package consumer verification passed: $PACKAGE"
