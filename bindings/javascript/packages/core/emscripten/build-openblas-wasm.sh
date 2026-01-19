#!/usr/bin/env bash
set -euo pipefail

# Build/download OpenBLAS + LAPACKE for Emscripten and place artifacts under
#   bindings/javascript/packages/core/emscripten/build/deps/openblas-wasm
#
# This script assumes:
#   - Emscripten env is active (emcc/emar available)
#   - curl, unzip are available
#   - You have network access to fetch the prebuilt archive
#
# Override OPENBLAS_WASM_URL to point at a different prebuilt bundle.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${SCRIPT_DIR}/build/deps/openblas-wasm"

OPENBLAS_WASM_URL="${OPENBLAS_WASM_URL:-https://github.com/pschatzmann/openblas-wasm/releases/latest/download/openblas-wasm.zip}"

echo "==> Preparing deps dir at ${DEPS_DIR}"
rm -rf "${DEPS_DIR}"
mkdir -p "${DEPS_DIR}"

TMP_ARCHIVE="${DEPS_DIR}/openblas-wasm.zip"
echo "==> Downloading OpenBLAS/LAPACKE bundle"
if ! curl -fL --retry 3 "${OPENBLAS_WASM_URL}" -o "${TMP_ARCHIVE}"; then
  echo "ERROR: download failed from ${OPENBLAS_WASM_URL}"
  exit 1
fi

echo "==> Validating archive"
FILESIZE=$(stat -c%s "${TMP_ARCHIVE}")
if [ "${FILESIZE}" -lt 1024 ]; then
  echo "ERROR: downloaded archive is too small (${FILESIZE} bytes) - likely a 404/HTML. Check OPENBLAS_WASM_URL."
  exit 1
fi

echo "==> Unpacking"
unzip -q "${TMP_ARCHIVE}" -d "${DEPS_DIR}"
rm -f "${TMP_ARCHIVE}"

echo "==> Contents:"
ls -l "${DEPS_DIR}"

cat <<'EOF'
Done.

If the Moonlab WASM link fails to find BLAS/LAPACKE:
  - Set OPENBLAS_WASM_ROOT to point at the extracted directory (default: build/deps/openblas-wasm)
  - Ensure the bundle includes libopenblas.a (with LAPACK/LAPACKE symbols) and cblas/lapacke headers.
  - Re-run the WASM build, e.g.:
      cd bindings/javascript
      pnpm --filter @moonlab/quantum-core build
EOF
