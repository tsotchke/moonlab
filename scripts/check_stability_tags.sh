#!/usr/bin/env bash
# stability_tags_complete evidence producer: every MOONLAB_API declaration in a
# public header must carry an @stability tier in its doxygen block.
#
# The tiers are defined normatively in docs/STABLE_ABI.md: stable (frozen,
# breaks only at a major), evolving (signature stable within the minor series),
# beta (may change in any release). A public symbol with no tier makes no
# statement about what a consumer may rely on, which is the state the whole
# surface was in before v1.2.1 -- 7 of 1244 declarations were tagged.
#
# This is a ratchet. Once the surface is fully tagged, a new public symbol
# cannot land untagged without failing here, which is the only thing that keeps
# the sweep from decaying the moment someone adds an API.
#
# Exit 0 iff every MOONLAB_API declaration is tagged. Prints each offender as
# "path:line symbol".

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - "$@" <<'PY'
import os
import re
import sys

SYM = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DATA = re.compile(r"\bextern\s+const\s+[A-Za-z_][A-Za-z0-9_]*\s+"
                  r"([A-Za-z_][A-Za-z0-9_]*)\s*;")
VALID = {"stable", "evolving", "beta"}
TAG = re.compile(r"@stability\s+(\S+)")


def decl_symbol(joined: str):
    m = DATA.search(joined)
    if m:
        return m.group(1)
    m = SYM.search(joined[len("MOONLAB_API"):])
    return m.group(1) if m else None


untagged = []
bad_tier = []
total = 0

for dirpath, _, files in os.walk("src"):
    for f in sorted(files):
        if not f.endswith(".h"):
            continue
        path = os.path.join(dirpath, f)
        lines = open(path, encoding="utf-8", errors="ignore").read().split("\n")
        i = 0
        while i < len(lines):
            if not lines[i].startswith("MOONLAB_API"):
                i += 1
                continue
            joined, j = lines[i], i
            while ";" not in joined and j + 1 < len(lines):
                j += 1
                joined += " " + lines[j].strip()
            sym = decl_symbol(joined)
            if not sym:
                i = j + 1
                continue
            total += 1

            # Walk up past blank lines; the block must close on the line above
            # and must be a doxygen /** block, not a banner comment.
            k = i - 1
            while k >= 0 and lines[k].strip() == "":
                k -= 1
            found = None
            if k >= 0 and lines[k].strip().endswith("*/"):
                start = k
                while start >= 0 and "/*" not in lines[start]:
                    start -= 1
                if start >= 0 and "/**" in lines[start]:
                    for b in range(start, k + 1):
                        m = TAG.search(lines[b])
                        if m:
                            found = m.group(1).rstrip("*/ ").strip()
                            break
            if found is None:
                untagged.append((path, i + 1, sym))
            elif found not in VALID:
                bad_tier.append((path, i + 1, sym, found))
            i = j + 1

if untagged:
    print(f"FAIL: {len(untagged)} MOONLAB_API declaration(s) carry no "
          f"@stability tier:")
    for path, line, sym in untagged:
        print(f"  {path}:{line} {sym}")
    print("\nAdd one of: stable / evolving / beta. See the tier table in "
          "docs/STABLE_ABI.md;")
    print("a new symbol that has not shipped in a tagged release is `beta`.")

if bad_tier:
    print(f"FAIL: {len(bad_tier)} declaration(s) carry a tier that is not "
          f"stable/evolving/beta:")
    for path, line, sym, tier in bad_tier:
        print(f"  {path}:{line} {sym} -> '{tier}'")

if untagged or bad_tier:
    sys.exit(1)

print(f"PASS: all {total} MOONLAB_API declarations carry a valid "
      f"@stability tier")
sys.exit(0)
PY
