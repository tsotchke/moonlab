#!/bin/bash
# Synchronize version across all project files
# Usage: ./scripts/sync-versions.sh <VERSION>
#
# Updates:
# - VERSION.txt file (root) - named .txt to avoid C++ <version> header conflict
# - CMakeLists.txt
# - bindings/python/setup.py
# - bindings/rust/*/Cargo.toml
# - bindings/javascript/package.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <VERSION>"
    echo "Example: $0 0.2.0"
    exit 1
fi

VERSION="$1"

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "ERROR: Invalid version format. Expected semver (e.g., 0.2.0 or 0.2.0-beta.1)"
    exit 1
fi

echo "========================================"
echo "  Syncing version to: $VERSION"
echo "========================================"

# Update VERSION.txt file (named to avoid conflict with C++ <version> header on case-insensitive filesystems)
echo "Updating VERSION.txt..."
echo "$VERSION" > VERSION.txt

# Update CMakeLists.txt
if [[ -f "CMakeLists.txt" ]]; then
    echo "Updating CMakeLists.txt..."
    sed -i.bak -E "s/project\(quantum_sim VERSION [0-9]+\.[0-9]+\.[0-9]+/project(quantum_sim VERSION $VERSION/" CMakeLists.txt
    rm -f CMakeLists.txt.bak
fi

# Update Python setup.py
if [[ -f "bindings/python/setup.py" ]]; then
    echo "Updating bindings/python/setup.py..."
    sed -i.bak -E "s/version=['\"][^'\"]+['\"]/version=\"$VERSION\"/" bindings/python/setup.py
    rm -f bindings/python/setup.py.bak
fi

# Update Python pyproject.toml if it exists
if [[ -f "bindings/python/pyproject.toml" ]]; then
    echo "Updating bindings/python/pyproject.toml..."
    sed -i.bak -E "s/^version = ['\"][^'\"]+['\"]/version = \"$VERSION\"/" bindings/python/pyproject.toml
    rm -f bindings/python/pyproject.toml.bak
fi

# Update Rust Cargo.toml files
for cargo_file in bindings/rust/*/Cargo.toml; do
    if [[ -f "$cargo_file" ]]; then
        echo "Updating $cargo_file..."
        sed -i.bak -E "s/^version = ['\"][^'\"]+['\"]/version = \"$VERSION\"/" "$cargo_file"
        rm -f "${cargo_file}.bak"
    fi
done

# Update JavaScript package.json files
if [[ -f "bindings/javascript/package.json" ]]; then
    echo "Updating bindings/javascript/package.json..."
    # Use jq if available, otherwise sed
    if command -v jq &> /dev/null; then
        jq ".version = \"$VERSION\"" bindings/javascript/package.json > bindings/javascript/package.json.tmp
        mv bindings/javascript/package.json.tmp bindings/javascript/package.json
    else
        sed -i.bak -E "s/\"version\": \"[^\"]+\"/\"version\": \"$VERSION\"/" bindings/javascript/package.json
        rm -f bindings/javascript/package.json.bak
    fi
fi

# Update individual JS packages
for pkg_file in bindings/javascript/packages/*/package.json; do
    if [[ -f "$pkg_file" ]]; then
        echo "Updating $pkg_file..."
        if command -v jq &> /dev/null; then
            jq ".version = \"$VERSION\"" "$pkg_file" > "${pkg_file}.tmp"
            mv "${pkg_file}.tmp" "$pkg_file"
        else
            sed -i.bak -E "s/\"version\": \"[^\"]+\"/\"version\": \"$VERSION\"/" "$pkg_file"
            rm -f "${pkg_file}.bak"
        fi
    fi
done

echo ""
echo "========================================"
echo "  Version synced to $VERSION"
echo "========================================"
echo ""
echo "Updated files:"
echo "  - VERSION.txt"
[[ -f "CMakeLists.txt" ]] && echo "  - CMakeLists.txt"
[[ -f "bindings/python/setup.py" ]] && echo "  - bindings/python/setup.py"
[[ -f "bindings/python/pyproject.toml" ]] && echo "  - bindings/python/pyproject.toml"
for f in bindings/rust/*/Cargo.toml; do [[ -f "$f" ]] && echo "  - $f"; done
[[ -f "bindings/javascript/package.json" ]] && echo "  - bindings/javascript/package.json"
for f in bindings/javascript/packages/*/package.json; do [[ -f "$f" ]] && echo "  - $f"; done
echo ""
echo "Don't forget to commit these changes!"
