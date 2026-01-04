#!/bin/bash
# Build Debian package for moonlab
# Usage: ./scripts/build-deb.sh <VERSION>
#
# Prerequisites:
# - cmake and ninja-build installed
# - Project must be built first (make clean && make)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

VERSION="${1:-0.1.0}"

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

# Create build directory if needed
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DQSIM_BUILD_SHARED=ON \
    -DQSIM_BUILD_STATIC=ON

# Build
echo "Building..."
cmake --build . --parallel

# Create package with CPack
echo "Creating DEB package..."
cpack -G DEB -D CPACK_PACKAGE_VERSION="${VERSION}"

# Move to project root with standardized name
cd "$PROJECT_ROOT"
DEB_FILE=$(find build -name "*.deb" -type f | head -1)

if [[ -n "$DEB_FILE" ]]; then
    FINAL_NAME="moonlab_${VERSION}_${ARCH}.deb"
    mv "$DEB_FILE" "$FINAL_NAME"
    echo ""
    echo "========================================"
    echo "  Package created: $FINAL_NAME"
    echo "========================================"
    ls -la "$FINAL_NAME"
else
    echo "ERROR: No .deb file generated"
    exit 1
fi
