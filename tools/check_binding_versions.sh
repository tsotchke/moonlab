#!/usr/bin/env bash
# Compatibility entry point for the canonical multi-ecosystem version gate.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/version_tool.py" check
