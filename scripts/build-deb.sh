#!/usr/bin/env bash
# Build a Debian package from the CMake install manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-0.1.0}"
BUILD_DIR="${MOONLAB_DEB_BUILD_DIR:-$PROJECT_ROOT/build-deb}"
PKG_ROOT="${MOONLAB_DEB_STAGING_DIR:-$PROJECT_ROOT/build/deb-root}"

echo "========================================"
echo "  Building Moonlab DEB Package"
echo "  Version: $VERSION"
echo "========================================"

# Detect architecture
ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
case "$ARCH" in
    aarch64|arm64)
        ARCH="arm64"
        ;;
    x86_64|amd64)
        ARCH="amd64"
        ;;
esac

echo "Architecture: $ARCH"

if ! command -v dpkg-deb >/dev/null; then
    echo "dpkg-deb is required to build Debian packages" >&2
    exit 2
fi

echo "Configuring with CMake..."
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DQSIM_BUILD_SHARED=ON \
    -DQSIM_BUILD_TESTS=OFF \
    -DQSIM_BUILD_EXAMPLES=OFF \
    -DQSIM_BUILD_BENCHMARKS=OFF

echo "Building..."
cmake --build "$BUILD_DIR" --parallel --target quantumsim moonlab-control-server

echo "Staging install tree..."
rm -rf "$PKG_ROOT"
cmake --install "$BUILD_DIR" --prefix "$PKG_ROOT/usr"

install -d "$PKG_ROOT/DEBIAN" "$PKG_ROOT/usr/share/doc/moonlab"
install -m 0644 "$PROJECT_ROOT/README.md" "$PROJECT_ROOT/LICENSE" \
    "$PKG_ROOT/usr/share/doc/moonlab/"

installed_size="$(du -sk "$PKG_ROOT/usr" | awk '{print $1}')"
cat >"$PKG_ROOT/DEBIAN/control" <<EOF
Package: moonlab
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: tsotchke <noreply@github.com>
Installed-Size: ${installed_size}
Depends: libc6 (>= 2.31), libstdc++6, libomp5 | libomp5-14, liblapacke, libopenblas0 | libblas3, libssl3 | libssl1.1
Description: High-performance quantum computing simulator
 Moonlab provides the libquantumsim runtime, stable C ABI headers,
 CMake package config, pkg-config metadata, and the moonlab control server.
EOF

required=(
    "$PKG_ROOT/usr/include/moonlab/moonlab_export.h"
    "$PKG_ROOT/usr/include/moonlab/moonlab_api.h"
    "$PKG_ROOT/usr/include/moonlab_features.h"
    "$PKG_ROOT/usr/include/moonlab_build_info.h"
)

for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "DEB staging tree missing required entry: ${path#$PKG_ROOT/}" >&2
        exit 1
    fi
done

cmake_config="$(find "$PKG_ROOT/usr" \
    -path '*/cmake/quantumsim/quantumsim-config.cmake' \
    -type f -print -quit)"
if [[ -z "$cmake_config" ]]; then
    echo "DEB staging tree missing quantumsim CMake config" >&2
    exit 1
fi

pkg_config="$(find "$PKG_ROOT/usr" -path '*/pkgconfig/quantumsim.pc' \
    -type f -print -quit)"
if [[ -z "$pkg_config" ]]; then
    echo "DEB staging tree missing quantumsim pkg-config file" >&2
    exit 1
fi

shared_lib="$(find "$PKG_ROOT/usr" -name 'libquantumsim.so*' \
    \( -type f -o -type l \) -print -quit)"
if [[ -z "$shared_lib" ]]; then
    echo "DEB staging tree missing libquantumsim shared library" >&2
    exit 1
fi

FINAL_NAME="$PROJECT_ROOT/moonlab_${VERSION}_${ARCH}.deb"
rm -f "$FINAL_NAME"

echo "Creating DEB package..."
dpkg-deb --build --root-owner-group "$PKG_ROOT" "$FINAL_NAME"

if [[ "${MOONLAB_DEB_VERIFY:-1}" == "1" ]]; then
    "$PROJECT_ROOT/scripts/verify_deb_package.sh" --package "$FINAL_NAME"
fi

echo ""
echo "========================================"
echo "  Package created: $FINAL_NAME"
echo "========================================"
ls -la "$FINAL_NAME"
