#!/usr/bin/env bash
# Verify that a release tarball works for an external CMake/pkg-config consumer.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE=""
WORK_DIR="${MOONLAB_VERIFY_WORK_DIR:-}"
KEEP_WORK_DIR="${MOONLAB_VERIFY_KEEP_WORK_DIR:-0}"

usage() {
    cat <<EOF
Usage: $0 --package <moonlab-release.tar.gz> [--work-dir <dir>] [--keep-work-dir]

Extracts a release tarball, configures a tiny external CMake project with
find_package(quantumsim CONFIG), includes the public Moonlab ABI headers, links
against quantumsim::quantumsim, and runs the resulting executable.  When
pkg-config is available, it also compiles and runs the same consumer through
lib/pkgconfig/quantumsim.pc.
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
    echo "release package not found: $PACKAGE" >&2
    exit 2
fi

if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-release-consumer.XXXXXX")"
else
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
fi

if [[ "$KEEP_WORK_DIR" != "1" ]]; then
    trap 'rm -rf "$WORK_DIR"' EXIT
else
    echo "[verify-release] keeping work dir: $WORK_DIR"
fi

PREFIX="$WORK_DIR/prefix"
CONSUMER_DIR="$WORK_DIR/consumer"
CMAKE_BUILD_DIR="$WORK_DIR/cmake-build"
mkdir -p "$PREFIX" "$CONSUMER_DIR"

tar -xzf "$PACKAGE" -C "$PREFIX"

required=(
    "include/moonlab/moonlab_export.h"
    "include/moonlab/moonlab_api.h"
    "include/moonlab_features.h"
    "include/moonlab_build_info.h"
    "include/quantumsim/algorithms/tensor_network/ca_mps.h"
    "lib/cmake/quantumsim/quantumsim-config.cmake"
    "lib/pkgconfig/quantumsim.pc"
)

for rel in "${required[@]}"; do
    if [[ ! -e "$PREFIX/$rel" ]]; then
        echo "release package missing consumer entry: $rel" >&2
        exit 1
    fi
done

cat >"$CONSUMER_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(moonlab_release_consumer C)

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

add_executable(moonlab_release_consumer consumer.c)
target_link_libraries(moonlab_release_consumer PRIVATE quantumsim::quantumsim)
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
    printf("moonlab release consumer OK %d.%d.%d features=%s\n",
           major, minor, patch, MOONLAB_VERSION_STRING);
    return 0;
}
EOF

cmake -S "$CONSUMER_DIR" -B "$CMAKE_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DMOONLAB_EXPECT_PREFIX="$PREFIX"
cmake --build "$CMAKE_BUILD_DIR" -j

run_with_library_path() {
    local exe="$1"
    case "$(uname -s)" in
        Darwin)
            DYLD_LIBRARY_PATH="$PREFIX/lib:${DYLD_LIBRARY_PATH:-}" "$exe"
            ;;
        Linux)
            LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" "$exe"
            ;;
        *)
            "$exe"
            ;;
    esac
}

run_with_library_path "$CMAKE_BUILD_DIR/moonlab_release_consumer"

if command -v pkg-config >/dev/null && command -v cc >/dev/null; then
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
    pkg-config --exists quantumsim

    read -r -a pkg_cflags <<<"$(pkg-config --cflags quantumsim)"
    read -r -a pkg_libs <<<"$(pkg-config --libs quantumsim)"
    cc "${pkg_cflags[@]}" "$CONSUMER_DIR/consumer.c" \
        -o "$WORK_DIR/pkg-config-consumer" "${pkg_libs[@]}"
    run_with_library_path "$WORK_DIR/pkg-config-consumer"
else
    echo "[verify-release] pkg-config or cc not available; skipping pkg-config consumer"
fi

echo "[verify-release] package consumer verification passed: $PACKAGE"
