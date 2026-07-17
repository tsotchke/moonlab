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
# OpenBLAS-WASM has no maintained, canonical prebuilt release Moonlab can
# pin by default: the previously-hardcoded default URL
# (github.com/pschatzmann/openblas-wasm/releases/latest) 404s -- that
# repository no longer exists. build-clapack-wasm.sh (CLAPACK from
# netlib, sha256-pinned) is the supported, CI-exercised BLAS/LAPACK path
# for the WASM build (see emscripten/CMakeLists.txt, which tries
# OpenBLAS first and falls back to CLAPACK). This script is opt-in for
# anyone with their own trusted OpenBLAS-WASM artifact.
#
# Both OPENBLAS_WASM_URL and OPENBLAS_WASM_SHA256 must be set explicitly
# -- there is no default URL, so this script never silently pins (or
# silently fails against) an unverifiable source. Compute the expected
# digest once download succeeds with `shasum -a 256 <archive>`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPS_DIR="${SCRIPT_DIR}/build/deps/openblas-wasm"

if [ -z "${OPENBLAS_WASM_URL:-}" ]; then
  echo "ERROR: OPENBLAS_WASM_URL is not set." >&2
  echo "  There is no maintained default OpenBLAS-WASM prebuilt release to" >&2
  echo "  fetch (the previous default, pschatzmann/openblas-wasm, no longer" >&2
  echo "  exists on GitHub). Point OPENBLAS_WASM_URL at your own trusted" >&2
  echo "  artifact and set OPENBLAS_WASM_SHA256 to its sha256, or use" >&2
  echo "  build-clapack-wasm.sh instead -- that is the supported, CI-" >&2
  echo "  exercised BLAS/LAPACK path for the WASM build." >&2
  exit 1
fi

if [ -z "${OPENBLAS_WASM_SHA256:-}" ]; then
  echo "ERROR: OPENBLAS_WASM_SHA256 is not set." >&2
  echo "  Download the archive once, compute 'shasum -a 256 <file>', and" >&2
  echo "  pass the digest via OPENBLAS_WASM_SHA256 so this script can" >&2
  echo "  verify it on every run rather than trusting the URL blindly." >&2
  exit 1
fi

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
# `stat -c%s` is GNU coreutils syntax; macOS/BSD `stat` needs `-f%z`.
# Try GNU first and fall back to BSD so this runs on both.
FILESIZE=$(stat -c%s "${TMP_ARCHIVE}" 2>/dev/null || stat -f%z "${TMP_ARCHIVE}")
if [ "${FILESIZE}" -lt 1024 ]; then
  echo "ERROR: downloaded archive is too small (${FILESIZE} bytes) - likely a 404/HTML. Check OPENBLAS_WASM_URL."
  exit 1
fi

echo "==> Verifying sha256"
ACTUAL_SHA256=$(shasum -a 256 "${TMP_ARCHIVE}" | awk '{print $1}')
if [ "${ACTUAL_SHA256}" != "${OPENBLAS_WASM_SHA256}" ]; then
  echo "ERROR: sha256 mismatch for ${TMP_ARCHIVE}"
  echo "  expected: ${OPENBLAS_WASM_SHA256}"
  echo "  actual:   ${ACTUAL_SHA256}"
  rm -f "${TMP_ARCHIVE}"
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
