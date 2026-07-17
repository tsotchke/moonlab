#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist/npm}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

for package in core algorithms viz react vue; do
    (
        cd "$ROOT/bindings/javascript/packages/$package"
        pnpm pack --pack-destination "$OUT"
    )
done

echo "Packed npm artifacts in $OUT"
