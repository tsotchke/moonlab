#!/usr/bin/env bash
# Synchronize every Moonlab release version surface.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <VERSION>" >&2
    echo "Example: $0 1.2.0-rc.1" >&2
    exit 2
fi

exec python3 "$SCRIPT_DIR/version_tool.py" set "$1"
