#!/usr/bin/env bash
set -euo pipefail

# Build CLAPACK (C LAPACK + CBLAS + f2c) for Emscripten and stage artifacts under:
#   bindings/javascript/packages/core/emscripten/build/deps/clapack-wasm
#
# Requirements:
#   - emcmake/emmake/emcc in PATH (emsdk activated)
#   - curl, tar
#
# Notes:
#   - This builds the C-translated LAPACK (no Fortran needed).
#   - Outputs: lib/libclapack.a, lib/libblas.a, lib/libf2c.a, include/*.h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/build/deps/clapack-wasm"

CLAPACK_URL="${CLAPACK_URL:-https://www.netlib.org/clapack/clapack-3.2.1-CMAKE.tgz}"
CLAPACK_TGZ="${DEPS_DIR}/clapack.tgz"

echo "==> Preparing deps dir at ${DEPS_DIR}"
rm -rf "${DEPS_DIR}"
mkdir -p "${DEPS_DIR}"

echo "==> Downloading CLAPACK bundle"
curl -fL --retry 3 "${CLAPACK_URL}" -o "${CLAPACK_TGZ}"

echo "==> Extracting"
tar -xzf "${CLAPACK_TGZ}" -C "${DEPS_DIR}"
rm -f "${CLAPACK_TGZ}"

SRC_DIR="$(find "${DEPS_DIR}" -maxdepth 1 -type d ! -path "${DEPS_DIR}" -iname 'clapack-*' | head -n 1)"
if [ -z "${SRC_DIR}" ]; then
  echo "ERROR: CLAPACK source directory not found after extraction."
  exit 1
fi

# Patch missing stdio include in xerbla.c (needed for emcc/clang)
XERBLA_C="${SRC_DIR}/BLAS/SRC/xerbla.c"
if [ -f "${XERBLA_C}" ] && ! grep -q "stdio.h" "${XERBLA_C}"; then
  echo "==> Patching xerbla.c to include stdio.h"
  sed -i '1i #include <stdio.h>' "${XERBLA_C}"
fi

# Disable CLAPACK test builds (they fail under wasm-ld due to duplicate symbols)
ROOT_CMAKE="${SRC_DIR}/CMakeLists.txt"
BLAS_CMAKE="${SRC_DIR}/BLAS/CMakeLists.txt"
if [ -f "${ROOT_CMAKE}" ]; then
  sed -i 's/^add_subdirectory(TESTING)/# add_subdirectory(TESTING)/' "${ROOT_CMAKE}"
  sed -i 's/^enable_testing()/# enable_testing()/' "${ROOT_CMAKE}"
fi
if [ -f "${BLAS_CMAKE}" ]; then
  sed -i 's/^add_subdirectory(TESTING)/# add_subdirectory(TESTING)/' "${BLAS_CMAKE}"
fi

BUILD_DIR="${DEPS_DIR}/build"
echo "==> Configuring with emcmake"
emcmake cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF

echo "==> Building"
emmake make -C "${BUILD_DIR}" -j"$(nproc)"

echo "==> Staging libs/headers"
LIB_DIR="${DEPS_DIR}/lib"
INCLUDE_DIR="${DEPS_DIR}/include"
mkdir -p "${LIB_DIR}" "${INCLUDE_DIR}"

find "${BUILD_DIR}" -name 'libclapack.a' -o -name 'liblapack.a' -o -name 'libblas.a' -o -name 'libf2c.a' | while read -r lib; do
  cp -f "${lib}" "${LIB_DIR}/"
done

# Copy headers from common CLAPACK locations
for hdr_dir in "${SRC_DIR}/INCLUDE" "${SRC_DIR}/F2CLIBS" "${SRC_DIR}/CBLAS/include" "${SRC_DIR}/SRC"; do
  if [ -d "${hdr_dir}" ]; then
    find "${hdr_dir}" -maxdepth 1 -name '*.h' -exec cp -f {} "${INCLUDE_DIR}/" \;
  fi
done

echo "==> Done. Contents:"
ls -l "${LIB_DIR}"
ls -l "${INCLUDE_DIR}" | head -n 20

cat <<'EOF'
If the build succeeds, CMake will auto-detect CLAPACK at:
  bindings/javascript/packages/core/emscripten/build/deps/clapack-wasm

You can override with:
  export CLAPACK_WASM_ROOT=/path/to/clapack-wasm
EOF
