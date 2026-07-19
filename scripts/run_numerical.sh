#!/usr/bin/env bash
#
# scripts/run_numerical.sh
#
# Numerical-edge + uninitialised-memory bug-hunting lane for Moonlab.
#
# Builds (or reuses) libquantumsim, builds the self-contained harnesses in
# tests/numerical, runs them, and emits JSONL events:
#
#   {"kind":"moonlab_numerical","name":"numerical_edge_clean","status":"PASS|FAIL",...}
#   {"kind":"moonlab_numerical","name":"uninit_clean","status":"PASS|FAIL",...}
#
# numerical_edge_clean : PASS iff every numerical harness reports fails=0.
# uninit_clean         : PASS iff, under heap poisoning (macOS MallocScribble /
#                        Linux MALLOC_PERTURB_), no harness gains a NEW failure
#                        vs its clean-run baseline and none crash -- i.e. no
#                        uninitialised-read propagation surfaced.  Full
#                        uninitialised-value detection (valgrind memcheck / MSan)
#                        runs in .github/workflows/numerical.yml on Linux, where
#                        those tools exist; they are unavailable on macOS arm64.
#
# The script edits nothing under src/; it configures standalone build dirs.
#
# Usage: scripts/run_numerical.sh [--max-qubits N] [--lib-dir DIR] [--out FILE]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_DIR="$REPO_ROOT/tests/numerical"
WORK="$REPO_ROOT/build-numerical"
LIB_WORK="$REPO_ROOT/build-numerical-lib"
LOG_DIR="$WORK/logs"
TRACES_DIR="$REPO_ROOT/scripts/icc_traces"
LIB_DIR=""
OUT=""
MAX_QUBITS="${NUMERIC_MAX_QUBITS:-22}"

while [ $# -gt 0 ]; do
  case "$1" in
    --max-qubits) MAX_QUBITS="$2"; shift 2 ;;
    --lib-dir)    LIB_DIR="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

require_build_path() {
  local resolved
  resolved="$(resolve_path "$1")" || return 1
  case "$resolved" in
    "$REPO_ROOT"/build-*) printf '%s\n' "$resolved" ;;
    *) printf 'numerical writable path must resolve under %s/build-*: %s\n' "$REPO_ROOT" "$resolved" >&2; return 1 ;;
  esac
}

WORK="$(require_build_path "$WORK")" || exit 2
LIB_WORK="$(require_build_path "$LIB_WORK")" || exit 2
LOG_DIR="$(require_build_path "$LOG_DIR")" || exit 2
TRACES_DIR="$(resolve_path "$TRACES_DIR")" || exit 2
if [ "$TRACES_DIR" != "$REPO_ROOT/scripts/icc_traces" ]; then
  echo "numerical trace directory must resolve to $REPO_ROOT/scripts/icc_traces" >&2
  exit 2
fi

mkdir -p "$WORK" "$LOG_DIR" "$TRACES_DIR"
[ -n "$OUT" ] || OUT="$TRACES_DIR/moonlab_numerical.jsonl"
OUT="$(resolve_path "$OUT")" || exit 2
case "$OUT" in
  "$TRACES_DIR"/*|"$WORK"/*) ;;
  *) echo "--out must resolve under $TRACES_DIR or $WORK" >&2; exit 2 ;;
esac
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).write_text("", encoding="utf-8")
PY

SOURCE_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || exit 2
IFS=$'\t' read -r SOURCE_GIT_HEAD SOURCE_GIT_TREE SOURCE_DIRTY SOURCE_FINGERPRINT \
  < <(python3 - "$SOURCE_IDENTITY_JSON" <<'PY'
import json
import sys
identity = json.loads(sys.argv[1])
print("\t".join((
    identity["git_head"], identity["git_tree"], str(identity["dirty"]).lower(),
    identity["source_fingerprint"],
)))
PY
)

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

sha256_file() {
  python3 - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

write_hash_manifest() { # manifest path, followed by artifact paths
  python3 - "$@" <<'PY'
import hashlib
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
rows = []
for raw in sys.argv[2:]:
    path = Path(raw)
    if path.is_file():
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
payload = ("\n".join(sorted(rows)) + "\n").encode()
manifest.write_bytes(payload)
print(hashlib.sha256(payload).hexdigest())
PY
}

DIAGNOSTICS_SHA256=""
LIB_ARTIFACT=""
LIB_ARTIFACT_SHA256=""

emit() { # name verdict extra_json_object; status and value stay in sync
  local generated_at
  generated_at="$(now_iso)"
  python3 - "$OUT" "$1" "$2" "$generated_at" "$3" \
      "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" \
      "$WORK" "$DIAGNOSTICS_SHA256" "$LIB_ARTIFACT" "$LIB_ARTIFACT_SHA256" <<'PY'
import json
import sys

(
    output, name, verdict, generated_at, extra_json, git_head, git_tree, dirty,
    source_fingerprint, build_dir, diagnostics_sha256, library_path, library_sha256,
) = sys.argv[1:]
event = {
    "kind": "moonlab_numerical",
    "name": name,
    "status": verdict,
    "value": verdict,
    "timestamp": generated_at,
    "generated_at": generated_at,
    "git_head": git_head,
    "git_tree": git_tree,
    "dirty": dirty == "true",
    "source_fingerprint": source_fingerprint,
    "build_dir": build_dir,
}
if diagnostics_sha256:
    event["diagnostics_manifest_sha256"] = diagnostics_sha256
if library_path:
    event["library_path"] = library_path
    event["library_sha256"] = library_sha256
event.update(json.loads(extra_json))
with open(output, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

# --------------------------------------------------------------------------
# 1. Locate or build libquantumsim.
# --------------------------------------------------------------------------
find_lib() {
  local d
  for d in "$LIB_DIR"; do
    [ -n "$d" ] || continue
    for candidate in "$d/libquantumsim.dylib" "$d/libquantumsim.so" "$d/libquantumsim.a"; do
      if [ -f "$candidate" ]; then echo "$d"; return 0; fi
    done
  done
  return 1
}

if ! LIB_DIR="$(find_lib)"; then
  echo "--- building fresh lane-owned CPU-only Release library ---"
  cmake -S "$REPO_ROOT" -B "$LIB_WORK" --fresh -DCMAKE_BUILD_TYPE=Release \
        -DQSIM_BUILD_TESTS=OFF -DQSIM_BUILD_EXAMPLES=OFF \
        -DQSIM_BUILD_BENCHMARKS=OFF -DQSIM_ENABLE_METAL=OFF \
        2>&1 | tee "$LOG_DIR/lib-configure.log"
  cfg_rc=${PIPESTATUS[0]}
  if [ "$cfg_rc" -ne 0 ]; then echo "FATAL: lib configure failed"; exit 3; fi
  cmake --build "$LIB_WORK" --target quantumsim --parallel 2 \
        2>&1 | tee "$LOG_DIR/lib-build.log"
  build_rc=${PIPESTATUS[0]}
  if [ "$build_rc" -ne 0 ]; then echo "FATAL: lib build failed"; exit 3; fi
  LIB_DIR="$LIB_WORK"
fi
LIB_DIR="$(resolve_path "$LIB_DIR")" || exit 3
echo "libquantumsim: $LIB_DIR"
export MOONLAB_LIB_DIR="$LIB_DIR"
export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"

# --------------------------------------------------------------------------
# 2. Build the harnesses (standalone, -j2).
# --------------------------------------------------------------------------
echo "--- building numerical harnesses ---"
cmake -S "$NUM_DIR" -B "$WORK/cmake" --fresh -DCMAKE_BUILD_TYPE=Release \
      -DMOONLAB_LIB_DIR="$LIB_DIR" 2>&1 | tee "$LOG_DIR/harness-configure.log"
cfg_rc=${PIPESTATUS[0]}
if [ "$cfg_rc" -ne 0 ]; then echo "FATAL: harness configure failed"; exit 4; fi
cmake --build "$WORK/cmake" --parallel 2 2>&1 | tee "$LOG_DIR/harness-build.log"
build_rc=${PIPESTATUS[0]}
if [ "$build_rc" -ne 0 ]; then echo "FATAL: harness build failed"; exit 4; fi

LIB_ARTIFACT="$(sed -n 's/^NUM_QUANTUMSIM:FILEPATH=//p' "$WORK/cmake/CMakeCache.txt" | tail -1)"
if [ -z "$LIB_ARTIFACT" ] || [ ! -f "$LIB_ARTIFACT" ]; then
  echo "FATAL: configured numerical library artifact is missing" >&2
  exit 4
fi
LIB_ARTIFACT="$(resolve_path "$LIB_ARTIFACT")" || exit 4
LIB_ARTIFACT_SHA256="$(sha256_file "$LIB_ARTIFACT")" || exit 4
DIAGNOSTICS_SHA256="$(write_hash_manifest "$LOG_DIR/build-diagnostics.sha256" \
  "$LOG_DIR/lib-configure.log" "$LOG_DIR/lib-build.log" \
  "$LOG_DIR/harness-configure.log" "$LOG_DIR/harness-build.log")" || exit 4

HARNESSES=(t_rotations t_qft t_measure t_dmrg t_eigen_lapack t_eigen_jacobi t_svd_lapack t_svd_fallback)

run_one() { # phase binary -> prints "checks fails rc"
  local phase="$1" bin="$WORK/cmake/$2" log="$LOG_DIR/$1-$2.log"
  local rc summ checks fails
  if [ "$phase" = "poison" ]; then
    env MallocPreScribble=1 MallocScribble=1 MallocGuardEdges=1 MALLOC_PERTURB_=170 \
      "$bin" 2>&1 | tee "$log" >/dev/null
  else
    "$bin" 2>&1 | tee "$log" >/dev/null
  fi
  rc=${PIPESTATUS[0]}
  summ="$(grep -o 'checks=[0-9]* fails=[0-9]*' "$log" | tail -1)"
  checks="$(printf '%s' "$summ" | sed -n 's/.*checks=\([0-9]*\).*/\1/p')"
  fails="$(printf '%s' "$summ" | sed -n 's/.*fails=\([0-9]*\).*/\1/p')"
  [ -n "$checks" ] || checks=-1
  [ -n "$fails" ] || fails=-1
  echo "$checks $fails $rc"
}

# --------------------------------------------------------------------------
# 3. Clean run (numerical_edge_clean).
# --------------------------------------------------------------------------
echo "--- clean run ---"
export NUMERIC_MAX_QUBITS="$MAX_QUBITS"
TOTAL_CHECKS=0; TOTAL_FAILS=0; ANY_CRASH=0
declare -a BASE_FAILS
i=0
DETAIL=""
for h in "${HARNESSES[@]}"; do
  read -r c f rc <<<"$(run_one clean "$h")"
  BASE_FAILS[$i]=$f
  echo "  $h: checks=$c fails=$f rc=$rc"
  [ "$c" -ge 0 ] && TOTAL_CHECKS=$((TOTAL_CHECKS + c)) || ANY_CRASH=1
  [ "$f" -ge 0 ] && TOTAL_FAILS=$((TOTAL_FAILS + f)) || ANY_CRASH=1
  # a harness intentionally returns its fail count as exit code; only treat
  # missing-summary (rc with no parsed counts) as a crash.
  if [ "$c" -lt 0 ]; then ANY_CRASH=1; fi
  DETAIL="$DETAIL{\"harness\":\"$h\",\"checks\":$c,\"fails\":$f}"
  [ $((i+1)) -lt ${#HARNESSES[@]} ] && DETAIL="$DETAIL,"
  i=$((i+1))
done

EDGE_STATUS="PASS"
[ "$TOTAL_FAILS" -eq 0 ] && [ "$ANY_CRASH" -eq 0 ] || EDGE_STATUS="FAIL"
emit "numerical_edge_clean" "$EDGE_STATUS" \
  "{\"total_checks\":$TOTAL_CHECKS,\"total_fails\":$TOTAL_FAILS,\"crash\":$ANY_CRASH,\"harnesses\":[$DETAIL],\"run_logs_manifest_sha256\":\"$(write_hash_manifest "$LOG_DIR/clean-run-logs.sha256" "$LOG_DIR"/clean-*.log)\"}"
echo "numerical_edge_clean: $EDGE_STATUS (checks=$TOTAL_CHECKS fails=$TOTAL_FAILS crash=$ANY_CRASH)"

# --------------------------------------------------------------------------
# 4. Heap-poisoned run (uninit_clean).
#    macOS: MallocPreScribble(0xAA fresh)/MallocScribble(0x55 freed).
#    Linux: MALLOC_PERTURB_ fills allocations with a nonzero pattern.
# --------------------------------------------------------------------------
echo "--- heap-poisoned (uninit) run ---"
UNINIT_NEWFAILS=0; UNINIT_CRASH=0
i=0
for h in "${HARNESSES[@]}"; do
  read -r c f rc <<<"$(run_one poison "$h")"
  local_base="${BASE_FAILS[$i]}"
  echo "  $h: fails=$f (baseline=$local_base) rc=$rc"
  if [ "$c" -lt 0 ]; then UNINIT_CRASH=1; fi
  if [ "$f" -ge 0 ] && [ "$local_base" -ge 0 ] && [ "$f" -gt "$local_base" ]; then
    UNINIT_NEWFAILS=$((UNINIT_NEWFAILS + (f - local_base)))
    echo "    NEW failures under poisoning: $((f - local_base)) (possible uninitialised read)"
  fi
  i=$((i+1))
done

UNINIT_STATUS="PASS"
[ "$UNINIT_NEWFAILS" -eq 0 ] && [ "$UNINIT_CRASH" -eq 0 ] || UNINIT_STATUS="FAIL"
emit "uninit_clean" "$UNINIT_STATUS" \
  "{\"new_failures_under_poison\":$UNINIT_NEWFAILS,\"crash\":$UNINIT_CRASH,\"tool\":\"heap-poison\",\"note\":\"valgrind/MSan run in CI (Linux); unavailable on macOS arm64\",\"run_logs_manifest_sha256\":\"$(write_hash_manifest "$LOG_DIR/poison-run-logs.sha256" "$LOG_DIR"/poison-*.log)\"}"
echo "uninit_clean: $UNINIT_STATUS (new_failures=$UNINIT_NEWFAILS crash=$UNINIT_CRASH)"

echo "--- JSONL: $OUT ---"
cat "$OUT"

# Non-zero exit if either gate failed (for CI signalling).
[ "$EDGE_STATUS" = "PASS" ] && [ "$UNINIT_STATUS" = "PASS" ]
