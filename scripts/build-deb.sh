#!/usr/bin/env bash
# Build a Debian package from the CMake install manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-1.2.0}"
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
    -DQSIM_BUILD_BENCHMARKS=OFF \
    -DQSIM_NATIVE_ARCH=OFF

echo "Building..."
cmake --build "$BUILD_DIR" --parallel --target quantumsim moonlab-control-server

echo "Staging install tree..."
rm -rf "$PKG_ROOT"
cmake --install "$BUILD_DIR" --prefix "$PKG_ROOT/usr"

install -d "$PKG_ROOT/DEBIAN" "$PKG_ROOT/usr/share/doc/moonlab"
install -m 0644 "$PROJECT_ROOT/README.md" "$PROJECT_ROOT/LICENSE" \
    "$PKG_ROOT/usr/share/doc/moonlab/"

# Hand-maintained fallback for hosts without dpkg-dev (e.g. building on
# macOS for local testing). Verified against the live Debian/Ubuntu
# archives (madison + Launchpad) rather than guessed: the LAPACKE C
# wrapper library ships as unversioned "liblapacke" on every current
# suite (bullseye through sid/noble) -- there is no "liblapacke3"
# package. ("liblapack3" is a real, different package: the base
# Fortran LAPACK library, not the C interface moonlab links.)
STATIC_DEPENDS='libc6 (>= 2.31), libstdc++6, libgomp1, liblapacke, libopenblas0 | libblas3, libssl3 | libssl1.1'

# Derive the real Depends: line from the staged binaries' actual ELF
# NEEDED entries via dpkg-shlibdeps when available. This is the correct
# way to compute a Debian package's runtime dependencies -- it reads
# the installed shlibs database on the build host instead of a
# hand-maintained guess that silently drifts as distro package names
# change across releases. Falls back to STATIC_DEPENDS when
# dpkg-shlibdeps isn't installed or the derivation fails for any reason.
depends_line() {
    if ! command -v dpkg-shlibdeps >/dev/null 2>&1; then
        echo "$STATIC_DEPENDS"
        return
    fi

    local scratch
    scratch="$(mktemp -d "${PROJECT_ROOT}/.moonlab-shlibdeps.XXXXXX")"

    # dpkg-shlibdeps needs debian/control on-disk (relative to its cwd)
    # to know which binary package it is computing shlibs:Depends for,
    # and to exclude that package from its own dependency list.
    mkdir -p "$scratch/debian"
    cat >"$scratch/debian/control" <<'CONTROL'
Source: moonlab

Package: moonlab
Architecture: any
Depends: ${shlibs:Depends}
Description: High-performance quantum computing simulator
CONTROL

    local libs=()
    while IFS= read -r -d '' f; do
        libs+=("$f")
    done < <(find "$PKG_ROOT/usr/lib" "$PKG_ROOT/usr/bin" \
        -type f \( -name '*.so*' -o -perm -u+x \) -print0 2>/dev/null)

    local derived=""
    if [[ ${#libs[@]} -gt 0 ]]; then
        derived="$(cd "$scratch" && dpkg-shlibdeps -O --ignore-missing-info \
            -Tdebian/moonlab.substvars "${libs[@]}" 2>/dev/null \
            | sed -n 's/^shlibs:Depends=//p')" || derived=""
    fi
    rm -rf -- "$scratch"

    if [[ -n "$derived" ]]; then
        echo "$derived"
    else
        echo "$STATIC_DEPENDS"
    fi
}

DEPENDS="$(depends_line)"
echo "Depends: $DEPENDS"

installed_size="$(du -sk "$PKG_ROOT/usr" | awk '{print $1}')"
cat >"$PKG_ROOT/DEBIAN/control" <<EOF
Package: moonlab
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: tsotchke <noreply@github.com>
Installed-Size: ${installed_size}
Depends: ${DEPENDS}
Description: High-performance quantum computing simulator
 Moonlab provides the libquantumsim runtime, stable C ABI headers,
 CMake package config, pkg-config metadata, and the moonlab control server.
EOF

# ldconfig must run after (un)installing a shared library so the
# dynamic linker's cache picks up libquantumsim.so's new SONAME entry;
# without this, a fresh install can fail to resolve the library until
# something else happens to invalidate the ld.so cache.
cat >"$PKG_ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
fi
exit 0
EOF
chmod 0755 "$PKG_ROOT/DEBIAN/postinst"

cat >"$PKG_ROOT/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
fi
exit 0
EOF
chmod 0755 "$PKG_ROOT/DEBIAN/postrm"

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
PACKAGE_TMP="$(mktemp "${PROJECT_ROOT}/.moonlab-deb.XXXXXX")"
trap 'rm -f -- "$PACKAGE_TMP"' EXIT

echo "Creating DEB package..."
dpkg-deb --build --root-owner-group "$PKG_ROOT" "$PACKAGE_TMP"

if [[ "${MOONLAB_DEB_VERIFY:-1}" == "1" ]]; then
    "$PROJECT_ROOT/scripts/verify_deb_package.sh" --package "$PACKAGE_TMP"
fi

# Install only a package that was built and verified successfully. rename(2)
# makes replacement atomic when PROJECT_ROOT is on one filesystem and does not
# follow a pre-existing destination symlink.
mv -f -- "$PACKAGE_TMP" "$FINAL_NAME"
trap - EXIT

echo ""
echo "========================================"
echo "  Package created: $FINAL_NAME"
echo "========================================"
ls -la "$FINAL_NAME"
