#!/usr/bin/env bash
# Run the reproducible-benchmark corpus and emit one JSON manifest per
# bench into an output directory.
#
# Usage:
#   tools/bench/run_corpus.sh [OUTPUT_DIR]
#
# If OUTPUT_DIR is not given, defaults to
#   artifacts/bench-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)
# so back-to-back runs from the same checkout land in distinct dirs.
#
# Each bench writes a pretty-printed JSON manifest capturing git SHA,
# build info, host info, and per-row timings (with stddev/min/max for
# benches that support MOONLAB_BENCH_N).
#
# Environment overrides:
#   MOONLAB_BENCH_N    number of timing replicas (default 5 for matmul,
#                      3 for chern).
#   BUILD_DIR          path to the cmake build directory (default: build).

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "error: BUILD_DIR=$BUILD_DIR does not exist; run cmake first" >&2
    exit 1
fi

OUT_DIR="${1:-}"
if [[ -z "$OUT_DIR" ]]; then
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    OUT_DIR="artifacts/bench-${TS}-${SHA}"
fi
mkdir -p "$OUT_DIR"

echo "==> bench corpus run"
echo "    build dir : $BUILD_DIR"
echo "    out dir   : $OUT_DIR"
echo "    replicas  : ${MOONLAB_BENCH_N:-default (5 / 3)}"
echo

run_bench () {
    local name="$1"; shift
    local bin="$BUILD_DIR/$name"
    if [[ ! -x "$bin" ]]; then
        echo "    skip $name (not built)"
        return
    fi
    local manifest="$OUT_DIR/${name}.manifest.json"
    echo "    run $name  -> $manifest"
    local stdout_log="$OUT_DIR/${name}.stdout.txt"
    MOONLAB_MANIFEST_OUT="$manifest" "$bin" "$@" > "$stdout_log" 2>&1 || {
        echo "      FAILED (see $stdout_log)"
        return 1
    }
}

run_bench bench_tensor_matmul_eshkol
run_bench bench_chern_kpm
run_bench bench_dmrg_workspace
# Chern mosaic HQ also takes CLI flags; run a default-topological
# config to match the canonical.
if [[ -x "$BUILD_DIR/bench_chern_mosaic_hq" ]]; then
    MOONLAB_CHERN_OUT_CSV="$OUT_DIR/chern_mosaic.csv" \
    MOONLAB_CHERN_OUT_PPM="$OUT_DIR/chern_mosaic.ppm" \
    MOONLAB_MANIFEST_OUT="$OUT_DIR/bench_chern_mosaic_hq.manifest.json" \
    "$BUILD_DIR/bench_chern_mosaic_hq" --L 64 --n 4 --V0 0.3 --Q 0.8976 \
        > "$OUT_DIR/bench_chern_mosaic_hq.stdout.txt" 2>&1 || echo "      chern_mosaic_hq FAILED"
    echo "    run bench_chern_mosaic_hq"
fi

echo
echo "==> done.  $(ls -1 $OUT_DIR/*.manifest.json 2>/dev/null | wc -l | tr -d ' ') manifests in $OUT_DIR"
ls -la "$OUT_DIR"
