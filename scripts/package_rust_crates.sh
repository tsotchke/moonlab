#!/usr/bin/env bash
# Package the Moonlab Rust crates into certified .crate artifacts.
#
# Produces the exact .crate tarballs that the release publishes byte for byte,
# in dependency order (moonlab-sys, moonlab, moonlab-tui), together with a
# SHA256SUMS file that certifies each artifact.  scripts/publish_crate_exact_bytes.py
# later ships these exact bytes and re-verifies them against SHA256SUMS.
#
# moonlab and moonlab-tui depend on siblings that are not yet on crates.io.
# `cargo package` would refuse to resolve their path+version dependencies, so a
# throwaway `--config patch.crates-io.<sibling>.path=...` points the resolver at
# the local sibling.  Cargo strips this patch from the packaged manifest, so the
# rewritten registry dependency (e.g. moonlab-sys = "1.2.0") is what ships.
#
# Usage:   scripts/package_rust_crates.sh <output-dir>
# Env:     MOONLAB_ALLOW_DIRTY=1  add --allow-dirty (local trees with edits)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_ROOT="$ROOT/bindings/rust"
OUT="${1:?usage: package_rust_crates.sh <output-dir>}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

PACKAGE_FLAGS=(package --no-verify)
if [[ "${MOONLAB_ALLOW_DIRTY:-0}" == "1" ]]; then
    PACKAGE_FLAGS+=(--allow-dirty)
fi

package_one() {
    local crate="$1"
    shift
    local patch_flags=("$@")
    (
        cd "$RUST_ROOT/$crate"
        rm -f target/package/*.crate 2>/dev/null || true
        cargo "${PACKAGE_FLAGS[@]}" ${patch_flags[@]+"${patch_flags[@]}"}
        cp target/package/*.crate "$OUT/"
    )
    echo "packaged $crate"
}

package_one moonlab-sys
package_one moonlab \
    --config "patch.crates-io.moonlab-sys.path=\"$RUST_ROOT/moonlab-sys\""
package_one moonlab-tui \
    --config "patch.crates-io.moonlab-sys.path=\"$RUST_ROOT/moonlab-sys\"" \
    --config "patch.crates-io.moonlab.path=\"$RUST_ROOT/moonlab\""

(
    cd "$OUT"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- *.crate > SHA256SUMS
    else
        shasum -a 256 -- *.crate > SHA256SUMS
    fi
)

echo "Certified Rust crates in $OUT:"
cat "$OUT/SHA256SUMS"
