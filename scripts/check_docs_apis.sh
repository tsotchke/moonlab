#!/usr/bin/env bash
# docs_apis_exist evidence producer: every C API named in a ```c code fence in
# README.md or docs/**/*.md must resolve to a real identifier somewhere in src/.
# A fictional name (e.g. vqe_minimize, qaoa_maxcut) appears nowhere in src/ and
# fails the check. This is a floor, not a type-checker: it catches invented
# symbols, which is the failure class the 2026-07-17 audit found in the README.
#
# Exit 0 iff every extracted C API resolves. Prints each unresolved symbol with
# the file it was cited in.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - "$@" <<'PY'
import os, re, subprocess, sys

# C-API namespaces Moonlab actually uses. An identifier that starts with one of
# these and is call-shaped (followed by "(") is treated as a load-bearing API.
PREFIXES = (
    "quantum_", "qsim_", "tn_", "dmrg_", "tdvp_", "vqe_", "qaoa_", "grover_",
    "qpe_", "mpo_", "ca_mps_", "ca_peps_", "moonlab_", "surface_code_",
    "toric_code_", "braid_", "anyon_", "fusion_tree_", "chern_", "qgt_", "jw_",
    "uccsd_", "molecule_", "xxz_", "mbl_", "bell_", "gate_", "measurement_",
    "clifford_", "topological_", "skyrmion_", "topo_", "entropy_", "qrng_",
    "mlkem_", "fermion_", "lattice_", "noise_", "mpdo_", "chern_kpm_",
)
# Call-shaped tokens that are language/library builtins or documentation
# placeholders (moonlab_foo/_v2 illustrate the ABI versioning convention in
# STABLE_ABI.md), not real Moonlab APIs.
IGNORE = {"gate_t", "measurement_t", "moonlab_foo", "moonlab_foo_v2"}

def c_fences(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    # ```c ... ``` and ```cpp ... ```
    for m in re.finditer(r"```(?:c|cpp|C)\n(.*?)```", text, re.S):
        yield m.group(1)

docs = ["README.md"]
for root, _, files in os.walk("docs"):
    for f in files:
        if f.endswith(".md"):
            docs.append(os.path.join(root, f))

# Build the set of identifiers that occur anywhere in src/ (headers+sources).
# One grep of all call-shaped/def identifiers is enough to answer "does this
# name exist at all"; a fictional API is absent entirely.
src_syms = set()
for root, _, files in os.walk("src"):
    for f in files:
        if f.endswith((".h", ".c", ".cpp", ".mm", ".cu", ".hpp")):
            p = os.path.join(root, f)
            try:
                t = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for tok in re.findall(r"\b[a-z_][a-z0-9_]*\b", t):
                src_syms.add(tok)

unresolved = {}   # symbol -> set(doc files)
for d in docs:
    for block in c_fences(d):
        for m in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(", block):
            sym = m.group(1)
            if sym in IGNORE:
                continue
            if not sym.startswith(PREFIXES):
                continue
            if sym not in src_syms:
                unresolved.setdefault(sym, set()).add(d)

if unresolved:
    print("FAIL: documented C APIs that resolve to no symbol in src/:")
    for sym in sorted(unresolved):
        print(f"  {sym:40s} cited in: {', '.join(sorted(unresolved[sym]))}")
    sys.exit(1)

print("PASS: every documented C API resolves to a real src/ symbol")
sys.exit(0)
PY
