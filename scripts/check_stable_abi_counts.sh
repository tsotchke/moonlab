#!/usr/bin/env bash
# stable_abi_counts evidence producer: the per-header symbol counts in the
# "Symbol catalog by module" table of docs/STABLE_ABI.md must equal the number
# of MOONLAB_API declarations actually present in those headers.
#
# The table went stale twice before this check existed -- vqe.h read 30 against
# 33 real declarations through the whole v1.2.0 cycle -- because nothing tied
# the prose to the source. A wrong count is not cosmetic: the table is what a
# downstream consumer reads to size the surface they are binding against.
#
# Exit 0 iff every header named in the table has the count the table claims.
# Prints each mismatch as "header: doc=N actual=M".

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - "$@" <<'PY'
import re
import sys
from pathlib import Path

DOC = Path("docs/STABLE_ABI.md")
if not DOC.is_file():
    print(f"FAIL: {DOC} not found")
    sys.exit(1)

# Rows look like:  | `src/algorithms/vqe.h`  |  35  |
ROW = re.compile(r"^\|\s*`([^`]+\.h)`\s*\|\s*(\d+)\s*\|")

claimed = {}
for line in DOC.read_text(encoding="utf-8").splitlines():
    m = ROW.match(line)
    if m:
        claimed[m.group(1)] = int(m.group(2))

if not claimed:
    print("FAIL: no per-header symbol counts found in docs/STABLE_ABI.md")
    sys.exit(1)

# Count MOONLAB_API declarations the same way the sweep does: a declaration
# starts at a line beginning with MOONLAB_API and runs to the first ';', so
# wrapped signatures count once, not once per line.
def count_decls(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="ignore").split("\n")
    n = i = 0
    while i < len(lines):
        if lines[i].startswith("MOONLAB_API"):
            joined, j = lines[i], i
            while ";" not in joined and j + 1 < len(lines):
                j += 1
                joined += " " + lines[j].strip()
            n += 1
            i = j + 1
        else:
            i += 1
    return n

bad = []
missing = []
for header, want in sorted(claimed.items()):
    p = Path(header)
    if not p.is_file():
        missing.append(header)
        continue
    got = count_decls(p)
    if got != want:
        bad.append((header, want, got))

if missing:
    print("FAIL: docs/STABLE_ABI.md names headers that do not exist:")
    for h in missing:
        print(f"  {h}")

if bad:
    print("FAIL: docs/STABLE_ABI.md symbol counts disagree with the headers:")
    for header, want, got in bad:
        print(f"  {header}: doc={want} actual={got}")

if bad or missing:
    sys.exit(1)

print(f"PASS: {len(claimed)} header symbol counts match docs/STABLE_ABI.md")
sys.exit(0)
PY
