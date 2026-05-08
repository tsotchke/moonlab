#!/usr/bin/env bash
# tools/check_binding_versions.sh
#
# Verifies every binding manifest's version field matches VERSION.txt
# at the repo root.  Run as a CI gate; exits non-zero on any mismatch.
#
# Checked files (relative to repo root):
#   bindings/javascript/package.json
#   bindings/javascript/packages/core/package.json
#   bindings/javascript/packages/algorithms/package.json
#   bindings/javascript/packages/react/package.json
#   bindings/javascript/packages/vue/package.json
#   bindings/javascript/packages/viz/package.json
#   bindings/python/pyproject.toml
#   bindings/rust/moonlab/Cargo.toml
#   bindings/rust/moonlab-tui/Cargo.toml
#   bindings/rust/moonlab-sys/Cargo.toml
#
# bindings/python/setup.py reads VERSION.txt at install time, so it is
# auto-synced and we do not check it.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPECTED="$(cat "$ROOT/VERSION.txt" | tr -d '\n')"
if [[ -z "$EXPECTED" ]]; then
    echo "ERROR: VERSION.txt is empty" >&2
    exit 2
fi

# (file, regex-to-extract-version)
declare -a CHECKS=(
    'bindings/javascript/package.json|"version":\s*"([^"]+)"'
    'bindings/javascript/packages/core/package.json|"version":\s*"([^"]+)"'
    'bindings/javascript/packages/algorithms/package.json|"version":\s*"([^"]+)"'
    'bindings/javascript/packages/react/package.json|"version":\s*"([^"]+)"'
    'bindings/javascript/packages/vue/package.json|"version":\s*"([^"]+)"'
    'bindings/javascript/packages/viz/package.json|"version":\s*"([^"]+)"'
    'bindings/python/pyproject.toml|^version = "([^"]+)"'
    'bindings/rust/moonlab/Cargo.toml|^version = "([^"]+)"'
    'bindings/rust/moonlab-tui/Cargo.toml|^version = "([^"]+)"'
    'bindings/rust/moonlab-sys/Cargo.toml|^version = "([^"]+)"'
)

failures=0
for spec in "${CHECKS[@]}"; do
    file="${spec%%|*}"
    pattern="${spec#*|}"
    full="$ROOT/$file"
    if [[ ! -f "$full" ]]; then
        echo "MISSING $file"
        failures=$((failures + 1))
        continue
    fi
    actual="$(grep -m1 -oE "$pattern" "$full" | head -1 | sed -E "s/.*\"([^\"]+)\".*/\1/")"
    if [[ -z "$actual" ]]; then
        echo "FAIL $file: no version field matched pattern"
        failures=$((failures + 1))
        continue
    fi
    if [[ "$actual" == "$EXPECTED" ]]; then
        printf '  OK    %-60s %s\n' "$file" "$actual"
    else
        printf '  FAIL  %-60s %s != VERSION.txt(%s)\n' "$file" "$actual" "$EXPECTED"
        failures=$((failures + 1))
    fi
done

if [[ $failures -ne 0 ]]; then
    echo
    echo "$failures binding manifest(s) out of sync with VERSION.txt=$EXPECTED" >&2
    echo "Bump them, then re-run this script." >&2
    exit 1
fi
echo
echo "All binding manifests in sync with VERSION.txt=$EXPECTED."
