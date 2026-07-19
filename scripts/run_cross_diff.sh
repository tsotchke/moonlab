#!/usr/bin/env bash
#
# run_cross_diff.sh -- orchestrate the Moonlab cross-backend / cross-binding
# differential and emit a JSONL trace.
#
# Pipeline:
#   1. generate the seeded corpus (independent numpy reference oracle pinned in).
#   2. build the C differential driver from tests/differential (standalone CMake)
#      against a prebuilt libquantumsim.
#   3. prove the reference oracle catches a corrupted probability (self-test).
#   4. run the C cross-backend differential (dense / tn_mps / Clifford / GPU).
#   5. run whichever bindings are available (Python always; Rust and JS on the
#      full/nightly profile), each SKIPPING cleanly with a logged reason if its
#      toolchain or library is missing.
#   6. aggregate into scripts/icc_traces/moonlab_differential.jsonl with
#      kind:"moonlab_differential" events:
#        cross_backend_differential, cross_binding_python, reference_oracle_agreement
#          -> release-blocking
#        cross_binding_rust, cross_binding_js
#          -> medium/nightly
#      plus an umbrella event recording how many bindings were actually
#      exercised (a silently-empty run cannot masquerade as green).
#
# Exit status is nonzero on any FAIL of a leg that ran. A SKIP is not a FAIL.
#
# Usage:
#   scripts/run_cross_diff.sh [--profile quick|full] [--rust] [--js] [--gpu]
#                             [--seed N] [--lib-dir DIR]
#
#   quick (default): qubits 2,3,4,6,8 depths 2,8  -- PR-suitable, tn exact/fast.
#   full            : qubits 2,3,4,6,8,10,12 depths 2,8,32 -- nightly; the exact
#                     tn_mps leg is capped at n<=10 for wall-clock (dense-vs-
#                     reference remains the absolute anchor at every n).

set -uo pipefail

# --------------------------------------------------------------------------
# Locate repo + paths
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIFF_DIR="$REPO_ROOT/tests/differential"
WORK="$REPO_ROOT/build/differential"
TRACES_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACES_DIR/moonlab_differential.jsonl"
QUARANTINE="$DIFF_DIR/KNOWN_DIVERGENCES.txt"

# --------------------------------------------------------------------------
# Arguments / profile
# --------------------------------------------------------------------------
PROFILE="quick"
RUN_RUST=0
RUN_JS=0
GPU_FLAG=""
# Empty => use gen_diff_corpus's canonical default seed (0xB16B00B5). The
# quarantine case ids in KNOWN_DIVERGENCES.txt are tied to that default, so we do
# NOT override it unless the caller explicitly passes --seed.
SEED=""
LIB_DIR="${MOONLAB_LIB_DIR:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --rust) RUN_RUST=1; shift ;;
    --js) RUN_JS=1; shift ;;
    --gpu) GPU_FLAG="--gpu"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --lib-dir) LIB_DIR="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# tn_mps is an EXACT second oracle across the whole corpus (n<=12, every family
# and depth): it reproduces the dense statevector and the independent numpy
# reference to ~1e-13. The former n>=10 divergences were three real, now-fixed
# bugs: the reversed-CNOT 2q transpose (control>target), the interior-bond
# 2q-gate norm collapse, and the silent float32 Metal SVD (exact 2q gates now
# stay on the CPU LAPACK double-precision path; the lossy GPU kernel is opt-in
# via MOONLAB_TN_GPU_LOSSY). KNOWN_DIVERGENCES.txt is therefore empty; any entry
# added back is a live open bug, never a tolerance escape hatch.
case "$PROFILE" in
  quick) QUBITS="2,3,4,6,8";        DEPTHS="2,8";    TN_MAX_N=12 ;;
  full)  QUBITS="2,3,4,6,8,10,12";  DEPTHS="2,8,32"; TN_MAX_N=12; RUN_RUST=1; RUN_JS=1 ;;
  *) echo "unknown profile: $PROFILE (want quick|full)" >&2; exit 2 ;;
esac

echo "=== moonlab cross-diff: profile=$PROFILE qubits=$QUBITS depths=$DEPTHS tn_max_n=$TN_MAX_N ==="

mkdir -p "$WORK" "$TRACES_DIR"
# Invalidate saved evidence before any current-tree prerequisite can fail.
: > "$TRACE_FILE"

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

# --------------------------------------------------------------------------
# Locate the prebuilt libquantumsim (for the standalone CMake build + bindings)
# --------------------------------------------------------------------------
if [ -z "$LIB_DIR" ]; then
  for d in "$REPO_ROOT/build" "$REPO_ROOT/build_release"; do
    if ls "$d"/libquantumsim.* >/dev/null 2>&1; then LIB_DIR="$d"; break; fi
  done
fi
if [ -z "$LIB_DIR" ] || ! ls "$LIB_DIR"/libquantumsim.* >/dev/null 2>&1; then
  echo "FATAL: libquantumsim not found. Build it first:" >&2
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build" >&2
  echo "  (or pass --lib-dir DIR / set MOONLAB_LIB_DIR)" >&2
  exit 3
fi
if [ ! -f "$LIB_DIR/CMakeCache.txt" ] || \
   ! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=$REPO_ROOT" "$LIB_DIR/CMakeCache.txt"; then
  echo "FATAL: $LIB_DIR is not a CMake build of the current source checkout" >&2
  exit 3
fi
if ! cmake --build "$LIB_DIR" --target quantumsim --parallel 2 \
    >"$WORK/library_build.out" 2>"$WORK/library_build.err"; then
  echo "FATAL: current-source libquantumsim build failed" >&2
  cat "$WORK/library_build.err" >&2
  exit 3
fi
export MOONLAB_LIB_DIR="$LIB_DIR"
# Runtime loader path for the driver + ctypes/rust bindings.
export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "libquantumsim: $LIB_DIR"

# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------
: > "$WORK/events.jsonl"   # accumulate this run's events, then append to trace

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# sanitize a string for embedding inside a JSON double-quoted value
json_escape() { printf '%s' "$1" | tr -d '\n\r' | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# emit_event <name> <status> <tier> <value_json>
emit_event() {
  local name="$1" status="$2" tier="$3" value="$4"
  printf '{"kind":"moonlab_differential","name":"%s","status":"%s","tier":"%s","value":%s,"timestamp":"%s","generated_at":"%s","git_head":"%s","git_tree":"%s","dirty":%s,"source_fingerprint":"%s","corpus_txt_sha256":"%s","corpus_json_sha256":"%s"}\n' \
    "$name" "$status" "$tier" "$value" "$(now_iso)" "$(now_iso)" \
    "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" \
    "${CORPUS_TXT_SHA256:-pending}" "${CORPUS_JSON_SHA256:-pending}" >> "$WORK/events.jsonl"
}

# Track overall gate outcome (release-blocking legs only) + exercise accounting.
GATE_FAIL=0
BINDINGS_EXERCISED=0
BINDINGS_SKIPPED=0

# --------------------------------------------------------------------------
# 1. Generate corpus
# --------------------------------------------------------------------------
echo "--- generating corpus ---"
if [ -n "$SEED" ]; then
  GEN_OK=1
  python3 "$SCRIPT_DIR/gen_diff_corpus.py" --out-dir "$WORK/corpus" \
      --seed "$SEED" --qubits "$QUBITS" --depths "$DEPTHS" || GEN_OK=0
else
  GEN_OK=1
  python3 "$SCRIPT_DIR/gen_diff_corpus.py" --out-dir "$WORK/corpus" \
      --qubits "$QUBITS" --depths "$DEPTHS" || GEN_OK=0
fi
if [ "$GEN_OK" = "0" ]; then
  echo "FATAL: corpus generation failed" >&2
  exit 4
fi
CORPUS_TXT="$WORK/corpus/corpus.txt"
CORPUS_JSON="$WORK/corpus/corpus.json"
IFS=$'\t' read -r CORPUS_TXT_SHA256 CORPUS_JSON_SHA256 \
  < <(python3 - "$CORPUS_TXT" "$CORPUS_JSON" <<'PY'
import hashlib
from pathlib import Path
import sys
print("\t".join(hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sys.argv[1:]))
PY
)

# --------------------------------------------------------------------------
# 2. Build the C differential driver (standalone CMake against prebuilt lib)
# --------------------------------------------------------------------------
echo "--- building C differential driver ---"
if ! cmake -UMOONLAB_QUANTUMSIM -S "$DIFF_DIR" -B "$WORK/cmake" -DCMAKE_BUILD_TYPE=Release \
      -DMOONLAB_LIB_DIR="$LIB_DIR" >/dev/null 2>"$WORK/cmake_cfg.err"; then
  echo "FATAL: cmake configure failed" >&2; cat "$WORK/cmake_cfg.err" >&2; exit 5
fi
if ! cmake --build "$WORK/cmake" --parallel 2 >/dev/null 2>"$WORK/cmake_build.err"; then
  echo "FATAL: cmake build failed" >&2; cat "$WORK/cmake_build.err" >&2; exit 5
fi
DRIVER="$WORK/cmake/diff_backends"
[ -x "$DRIVER" ] || { echo "FATAL: driver not built at $DRIVER" >&2; exit 5; }

# --------------------------------------------------------------------------
# 3. Reference-oracle self-test (proves a corrupted probability is caught)
# --------------------------------------------------------------------------
echo "--- reference oracle self-test ---"
if "$DRIVER" --selftest | tee "$WORK/selftest.out" | grep -q "DIFF_SELFTEST status=PASS"; then
  echo "self-test: PASS"
else
  echo "self-test: FAIL" >&2
  GATE_FAIL=1
fi

# --------------------------------------------------------------------------
# 4. C cross-backend differential
# --------------------------------------------------------------------------
echo "--- C cross-backend differential ---"
"$DRIVER" --tn-max-n "$TN_MAX_N" --quarantine "$QUARANTINE" $GPU_FLAG "$CORPUS_TXT" \
  > "$WORK/c_diff.out" 2> "$WORK/c_diff.err" || true
cat "$WORK/c_diff.err" >&2 || true

parse_kv() { grep "$1" "$WORK/c_diff.out" | grep -oE "$2=[-0-9]+" | head -1 | cut -d= -f2; }
get_result() {  # get_result <name> <key>
  grep "DIFF_RESULT name=$1 " "$WORK/c_diff.out" | grep -oE "$2=[A-Za-z0-9_-]+" | head -1 | cut -d= -f2
}

CB_STATUS=$(get_result cross_backend_differential status); CB_STATUS=${CB_STATUS:-FAIL}
CB_CHECKS=$(get_result cross_backend_differential checks); CB_CHECKS=${CB_CHECKS:-0}
CB_FAILED=$(get_result cross_backend_differential failed); CB_FAILED=${CB_FAILED:-0}
CB_QUAR=$(get_result cross_backend_differential quarantined); CB_QUAR=${CB_QUAR:-0}
CB_SKIP=$(get_result cross_backend_differential skipped); CB_SKIP=${CB_SKIP:-0}

RO_STATUS=$(get_result reference_oracle_agreement status); RO_STATUS=${RO_STATUS:-FAIL}
RO_CHECKS=$(get_result reference_oracle_agreement checks); RO_CHECKS=${RO_CHECKS:-0}
RO_FAILED=$(get_result reference_oracle_agreement failed); RO_FAILED=${RO_FAILED:-0}
RO_QUAR=$(get_result reference_oracle_agreement quarantined); RO_QUAR=${RO_QUAR:-0}
RO_SKIP=$(get_result reference_oracle_agreement skipped); RO_SKIP=${RO_SKIP:-0}

EX_DENSE=$(grep "DIFF_EXERCISED backend=dense" "$WORK/c_diff.out" | grep -oE "cases=[0-9]+" | cut -d= -f2); EX_DENSE=${EX_DENSE:-0}
EX_TN=$(grep "DIFF_EXERCISED backend=tn_mps" "$WORK/c_diff.out" | grep -oE "cases=[0-9]+" | head -1 | cut -d= -f2); EX_TN=${EX_TN:-0}
EX_CLIFF=$(grep "DIFF_EXERCISED backend=clifford" "$WORK/c_diff.out" | grep -oE "cases=[0-9]+" | cut -d= -f2); EX_CLIFF=${EX_CLIFF:-0}
EX_GPU=$(grep "DIFF_EXERCISED backend=gpu" "$WORK/c_diff.out" | grep -oE "cases=[0-9]+" | cut -d= -f2); EX_GPU=${EX_GPU:-0}
GPU_REASON=$(grep "DIFF_EXERCISED backend=gpu" "$WORK/c_diff.out" | sed -E 's/.*reason="([^"]*)".*/\1/')
CASES=$(grep "DIFF_CASES parsed=" "$WORK/c_diff.out" | grep -oE "parsed=[0-9]+" | cut -d= -f2); CASES=${CASES:-0}

echo "cross_backend=$CB_STATUS  reference_oracle=$RO_STATUS  (dense=$EX_DENSE tn_mps=$EX_TN clifford=$EX_CLIFF gpu=$EX_GPU)"

emit_event cross_backend_differential "$CB_STATUS" release \
  "{\"checks\":$CB_CHECKS,\"failed\":$CB_FAILED,\"quarantined\":$CB_QUAR,\"skipped\":$CB_SKIP,\"cases\":$CASES,\"backends\":{\"dense\":$EX_DENSE,\"tn_mps\":$EX_TN,\"clifford\":$EX_CLIFF,\"gpu\":$EX_GPU}}"
emit_event reference_oracle_agreement "$RO_STATUS" release \
  "{\"checks\":$RO_CHECKS,\"failed\":$RO_FAILED,\"quarantined\":$RO_QUAR,\"skipped\":$RO_SKIP,\"cases\":$CASES}"

[ "$CB_STATUS" = "PASS" ] || GATE_FAIL=1
[ "$RO_STATUS" = "PASS" ] || GATE_FAIL=1
# The C driver exercises backends, not a language binding.  Keep binding
# accounting honest so a skipped Python leg cannot make the umbrella green.

# --------------------------------------------------------------------------
# 5a. Python cross-binding (release-blocking)
# --------------------------------------------------------------------------
echo "--- Python cross-binding ---"
python3 "$DIFF_DIR/test_diff_python.py" "$CORPUS_JSON" > "$WORK/py.out" 2> "$WORK/py.err" || true
cat "$WORK/py.err" >&2 || true
PY_LINE=$(grep "DIFF_PY_RESULT" "$WORK/py.out" | head -1)
PY_STATUS=$(printf '%s' "$PY_LINE" | grep -oE "status=[A-Z]+" | cut -d= -f2); PY_STATUS=${PY_STATUS:-SKIP}
PY_CASES=$(printf '%s' "$PY_LINE" | grep -oE "cases=[0-9]+" | cut -d= -f2); PY_CASES=${PY_CASES:-0}
PY_FAILED=$(printf '%s' "$PY_LINE" | grep -oE "failed=[0-9]+" | cut -d= -f2); PY_FAILED=${PY_FAILED:-0}
PY_REASON=$(printf '%s' "$PY_LINE" | sed -E 's/.*reason="([^"]*)".*/\1/'); PY_REASON=$(json_escape "${PY_REASON:-}")
echo "python: $PY_STATUS ($PY_CASES cases, $PY_FAILED failed)"
emit_event cross_binding_python "$PY_STATUS" release \
  "{\"cases\":$PY_CASES,\"failed\":$PY_FAILED,\"reason\":\"$PY_REASON\"}"
if [ "$PY_STATUS" = "SKIP" ]; then BINDINGS_SKIPPED=$((BINDINGS_SKIPPED + 1));
else BINDINGS_EXERCISED=$((BINDINGS_EXERCISED + 1)); fi
[ "$PY_STATUS" = "PASS" ] || GATE_FAIL=1

# --------------------------------------------------------------------------
# 5b. Rust cross-binding (medium/nightly)
# --------------------------------------------------------------------------
if [ "$RUN_RUST" = "1" ]; then
  echo "--- Rust cross-binding ---"
  if command -v cargo >/dev/null 2>&1; then
    cargo run --quiet --release \
         --manifest-path "$DIFF_DIR/rust/Cargo.toml" -- "$CORPUS_TXT" \
         > "$WORK/rust.out" 2> "$WORK/rust.err" || true
    RUST_LINE=$(grep "DIFF_RUST_RESULT" "$WORK/rust.out" | head -1)
    if [ -n "$RUST_LINE" ]; then
      RUST_STATUS=$(printf '%s' "$RUST_LINE" | grep -oE "status=[A-Z]+" | cut -d= -f2)
      RUST_CASES=$(printf '%s' "$RUST_LINE" | grep -oE "cases=[0-9]+" | cut -d= -f2)
      RUST_FAILED=$(printf '%s' "$RUST_LINE" | grep -oE "failed=[0-9]+" | cut -d= -f2)
      RUST_REASON="ok"
    else
      RUST_STATUS="SKIP"; RUST_CASES=0; RUST_FAILED=0
      RUST_REASON="cargo build/run failed (see build/differential/rust.err)"
      head -5 "$WORK/rust.err" >&2 || true
    fi
  else
    RUST_STATUS="SKIP"; RUST_CASES=0; RUST_FAILED=0; RUST_REASON="cargo not installed"
  fi
  RUST_CASES=${RUST_CASES:-0}; RUST_FAILED=${RUST_FAILED:-0}
  echo "rust: $RUST_STATUS ($RUST_CASES cases, $RUST_FAILED failed) $RUST_REASON"
  emit_event cross_binding_rust "$RUST_STATUS" medium \
    "{\"cases\":$RUST_CASES,\"failed\":$RUST_FAILED,\"reason\":\"$(json_escape "$RUST_REASON")\"}"
  if [ "$RUST_STATUS" = "SKIP" ]; then BINDINGS_SKIPPED=$((BINDINGS_SKIPPED + 1));
  else BINDINGS_EXERCISED=$((BINDINGS_EXERCISED + 1)); fi
  [ "$RUST_STATUS" = "FAIL" ] && GATE_FAIL=1
fi

# --------------------------------------------------------------------------
# 5c. JS/WASM cross-binding (medium/nightly)
# --------------------------------------------------------------------------
if [ "$RUN_JS" = "1" ]; then
  echo "--- JS/WASM cross-binding ---"
  if command -v node >/dev/null 2>&1; then
    node "$DIFF_DIR/diff_js.mjs" "$CORPUS_JSON" > "$WORK/js.out" 2> "$WORK/js.err" || true
    JS_LINE=$(grep "DIFF_JS_RESULT" "$WORK/js.out" | head -1)
    JS_STATUS=$(printf '%s' "$JS_LINE" | grep -oE "status=[A-Z]+" | cut -d= -f2); JS_STATUS=${JS_STATUS:-SKIP}
    JS_CASES=$(printf '%s' "$JS_LINE" | grep -oE "cases=[0-9]+" | cut -d= -f2); JS_CASES=${JS_CASES:-0}
    JS_FAILED=$(printf '%s' "$JS_LINE" | grep -oE "failed=[0-9]+" | cut -d= -f2); JS_FAILED=${JS_FAILED:-0}
    JS_REASON=$(printf '%s' "$JS_LINE" | sed -E 's/.*reason="([^"]*)".*/\1/'); JS_REASON=${JS_REASON:-unknown}
  else
    JS_STATUS="SKIP"; JS_CASES=0; JS_FAILED=0; JS_REASON="node not installed"
  fi
  echo "js: $JS_STATUS ($JS_CASES cases, $JS_FAILED failed) $JS_REASON"
  emit_event cross_binding_js "$JS_STATUS" medium \
    "{\"cases\":$JS_CASES,\"failed\":$JS_FAILED,\"reason\":\"$(json_escape "$JS_REASON")\"}"
  if [ "$JS_STATUS" = "SKIP" ]; then BINDINGS_SKIPPED=$((BINDINGS_SKIPPED + 1));
  else BINDINGS_EXERCISED=$((BINDINGS_EXERCISED + 1)); fi
  [ "$JS_STATUS" = "FAIL" ] && GATE_FAIL=1
fi

# --------------------------------------------------------------------------
# 6. Umbrella event + append to the trace file
# --------------------------------------------------------------------------
FINAL_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
if [ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
  GATE_FAIL=1
  python3 - "$WORK/events.jsonl" "$SOURCE_FINGERPRINT" "$FINAL_FINGERPRINT" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
events = []
for line in path.read_text(encoding="utf-8").splitlines():
    event = json.loads(line)
    event["status"] = "FAIL"
    event["provenance_error"] = f"source changed during lane: start={sys.argv[2]} end={sys.argv[3] or 'unknown'}"
    events.append(event)
path.write_text("".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")
PY
fi
UMBRELLA_STATUS="PASS"; [ "$GATE_FAIL" = "0" ] || UMBRELLA_STATUS="FAIL"
emit_event moonlab_differential "$UMBRELLA_STATUS" release \
  "{\"profile\":\"$PROFILE\",\"bindings_exercised\":$BINDINGS_EXERCISED,\"bindings_skipped\":$BINDINGS_SKIPPED,\"corpus_cases\":$CASES,\"gpu\":\"$(json_escape "${GPU_REASON:-not enabled}")\"}"

# A silently-empty run must never look green: the C driver + at least one binding
# must have actually exercised cases.
CORPUS_VALIDATION_STATUS="$UMBRELLA_STATUS"
if [ "$CASES" -lt 1 ] || [ "$BINDINGS_EXERCISED" -lt 1 ]; then
  CORPUS_VALIDATION_STATUS="FAIL"
fi
CORPUS_VALIDATION_VALUE='{"artifacts":["'"$(json_escape "$CORPUS_TXT")"'","'"$(json_escape "$CORPUS_JSON")"'"],"cases":'"$CASES"',"snippet":"corpus artifacts consumed: '"$(json_escape "$CORPUS_TXT")"', '"$(json_escape "$CORPUS_JSON")"'"}'
emit_event corpus_artifacts_validated "$CORPUS_VALIDATION_STATUS" release "$CORPUS_VALIDATION_VALUE"

cat "$WORK/events.jsonl" >> "$TRACE_FILE"
echo "--- wrote $(wc -l < "$WORK/events.jsonl" | tr -d ' ') events to $TRACE_FILE ---"
cat "$WORK/events.jsonl"

if [ "$BINDINGS_EXERCISED" -lt 1 ] || [ "$CASES" -lt 1 ]; then
  echo "FATAL: no cases/bindings exercised -- refusing to report green" >&2
  exit 6
fi

if [ "$GATE_FAIL" = "0" ]; then
  echo "=== cross-diff PASS (profile=$PROFILE, bindings exercised=$BINDINGS_EXERCISED, skipped=$BINDINGS_SKIPPED) ==="
  exit 0
else
  echo "=== cross-diff FAIL (profile=$PROFILE) ===" >&2
  exit 1
fi
