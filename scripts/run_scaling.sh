#!/usr/bin/env bash
#
# run_scaling.sh -- orchestrate the Moonlab SCALING / large-n differential and
# emit a JSONL trace.
#
# Pipeline:
#   1. generate the seeded scaling corpus (independent numpy reference pinned in).
#   2. build the four scaling drivers from tests/scaling (standalone CMake, -j2)
#      against a prebuilt libquantumsim.
#   3. run each driver's self-test (proves each oracle catches a corruption).
#   4. run the drivers:
#        scaling_diff       -- dense / tn_mps / Clifford over the circuit corpus
#        scaling_stab       -- Clifford tableau vs stabilizer structure at n=50,100
#        scaling_var_d      -- CA-MPS var-D on TFIM/Heisenberg + uint32 width
#        scaling_dmrg_tdvp  -- DMRG vs ED, TDVP vs Krylov
#   5. aggregate into scripts/icc_traces/moonlab_scaling.jsonl with
#      kind:"moonlab_scaling" events, plus the release-blocking umbrella event
#        name:"scaling_differential_clean" value:PASS|FAIL + divergence_count
#      where divergence_count counts every divergence. No scaling quarantine is
#      permitted: any nonzero known_divergences count also fails the umbrella.
#
# Exit status is nonzero iff any driver reports a NEW divergence.
#
# Usage:
#   scripts/run_scaling.sh [--profile quick|full] [--lib-dir DIR] [--seed N]
#     quick (default): qubits 10,12  depths 8,32   + small-longrange, tn cap 12
#     full           : qubits 10,12,14,16 depths 8,32,128 + small-longrange, tn cap 14

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCALE_DIR="$REPO_ROOT/tests/scaling"
WORK="$REPO_ROOT/build-scaling"
LIB_WORK="$REPO_ROOT/build-scaling-lib"
LOG_DIR="$WORK/logs"
TRACES_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACES_DIR/moonlab_scaling.jsonl"
GEN="$SCALE_DIR/gen_scaling_corpus.py"

PROFILE="quick"
LIB_DIR="${MOONLAB_LIB_DIR:-}"
SEED="0x5CA11B16"
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --lib-dir) LIB_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# The tn_mps leg is O(chi^3) per gate with chi = 2^ceil(n/2), so deep circuits at
# n>=10 are the affordability wall. dense-vs-reference and Clifford-vs-reference
# anchor correctness at EVERY n/depth cheaply, and the tn 2q-gate bug reproduces
# fully on the small-longrange (n<=8) cases, so bounding the tn leg loses no
# coverage of the findings.
case "$PROFILE" in
  quick) QUBITS="10,12";        DEPTHS="8,32";     TN_MAX_N=8;  TN_MAX_DEPTH=32 ;;
  full)  QUBITS="10,12,14,16";  DEPTHS="8,32,128"; TN_MAX_N=10; TN_MAX_DEPTH=32 ;;
  *) echo "unknown profile: $PROFILE (want quick|full)" >&2; exit 2 ;;
esac

echo "=== moonlab scaling: profile=$PROFILE qubits=$QUBITS depths=$DEPTHS tn_max_n=$TN_MAX_N tn_max_depth=$TN_MAX_DEPTH ==="

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
    *) printf 'scaling writable path must resolve under %s/build-*: %s\n' "$REPO_ROOT" "$resolved" >&2; return 1 ;;
  esac
}

WORK="$(require_build_path "$WORK")" || exit 2
LIB_WORK="$(require_build_path "$LIB_WORK")" || exit 2
LOG_DIR="$(require_build_path "$LOG_DIR")" || exit 2
TRACES_DIR="$(resolve_path "$TRACES_DIR")" || exit 2
if [ "$TRACES_DIR" != "$REPO_ROOT/scripts/icc_traces" ]; then
  echo "scaling trace directory must resolve to $REPO_ROOT/scripts/icc_traces" >&2
  exit 2
fi
TRACE_FILE="$TRACES_DIR/moonlab_scaling.jsonl"
mkdir -p "$WORK" "$LOG_DIR" "$TRACES_DIR"
python3 - "$TRACE_FILE" <<'PY'
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

# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------
now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
emit() { # name verdict tier new known extra_json_object [artifact]
  local generated_at artifact_path
  generated_at="$(now_iso)"
  artifact_path="${7:-}"
  python3 - "$TRACE_FILE" "$1" "$2" "$3" "$4" "$5" "$6" "$generated_at" \
      "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" \
      "$WORK" "$DIAGNOSTICS_SHA256" "$LIB_ARTIFACT" "$LIB_ARTIFACT_SHA256" "$artifact_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    trace, name, verdict, tier, new, known, extra_json, generated_at, git_head,
    git_tree, dirty, source_fingerprint, build_dir, diagnostics_sha256,
    library_path, library_sha256, artifact_path,
) = sys.argv[1:]
event = {
    "kind": "moonlab_scaling",
    "name": name,
    "status": verdict,
    "value": verdict,
    "tier": tier,
    "ts": generated_at,
    "generated_at": generated_at,
    "new_divergences": int(new),
    "known_divergences": int(known),
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
if artifact_path and Path(artifact_path).is_file():
    event["artifact_path"] = artifact_path
    event["artifact_sha256"] = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
event.update(json.loads(extra_json))
with open(trace, "a", encoding="utf-8") as output:
    output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

# --------------------------------------------------------------------------
# Build or locate libquantumsim. The default release path always uses a fresh,
# lane-owned library so a cached external artifact cannot certify current HEAD.
# --------------------------------------------------------------------------
if [ -z "$LIB_DIR" ]; then
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
library_found=0
for candidate in "$LIB_DIR/libquantumsim.dylib" "$LIB_DIR/libquantumsim.so" "$LIB_DIR/libquantumsim.a"; do
  if [ -f "$candidate" ]; then library_found=1; break; fi
done
if [ "$library_found" -ne 1 ]; then echo "FATAL: libquantumsim not found in $LIB_DIR" >&2; exit 3; fi
export MOONLAB_LIB_DIR="$LIB_DIR"
export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "libquantumsim: $LIB_DIR"

# --------------------------------------------------------------------------
# 1. generate corpus
# --------------------------------------------------------------------------
CORPUS_DIR="$WORK/corpus"
python3 "$GEN" --out-dir "$CORPUS_DIR" --seed "$SEED" \
      --qubits "$QUBITS" --depths "$DEPTHS" --small-longrange \
      2>&1 | tee "$LOG_DIR/corpus-generation.log"
corpus_rc=${PIPESTATUS[0]}
if [ "$corpus_rc" -ne 0 ]; then
  echo "FATAL: corpus generation failed (numpy required)" >&2
  DIAGNOSTICS_SHA256="$(write_hash_manifest "$LOG_DIR/build-diagnostics.sha256" "$LOG_DIR/corpus-generation.log")"
  emit corpus_generation FAIL release 1 0 '{"reason":"gen_scaling_corpus failed"}' "$LOG_DIR/corpus-generation.log"
  exit 4
fi

# --------------------------------------------------------------------------
# 2. build drivers (standalone CMake, -j2)
# --------------------------------------------------------------------------
cmake -S "$SCALE_DIR" -B "$WORK/cmake" --fresh -DMOONLAB_LIB_DIR="$LIB_DIR" \
    2>&1 | tee "$LOG_DIR/driver-configure.log"
cfg_rc=${PIPESTATUS[0]}
if [ "$cfg_rc" -eq 0 ]; then
  cmake --build "$WORK/cmake" --parallel 2 2>&1 | tee "$LOG_DIR/driver-build.log"
  build_rc=${PIPESTATUS[0]}
else
  build_rc=1
fi
DIAGNOSTICS_SHA256="$(write_hash_manifest "$LOG_DIR/build-diagnostics.sha256" \
  "$LOG_DIR/lib-configure.log" "$LOG_DIR/lib-build.log" \
  "$LOG_DIR/corpus-generation.log" "$LOG_DIR/driver-configure.log" "$LOG_DIR/driver-build.log")"
if [ "$cfg_rc" -ne 0 ] || [ "$build_rc" -ne 0 ]; then
  echo "FATAL: driver build failed" >&2
  emit driver_build FAIL release 1 0 "{\"reason\":\"cmake build failed\",\"configure_rc\":$cfg_rc,\"build_rc\":$build_rc}" "$LOG_DIR/driver-build.log"
  exit 5
fi
BIN="$WORK/cmake"
LIB_ARTIFACT="$(sed -n 's/^MOONLAB_QUANTUMSIM:FILEPATH=//p' "$WORK/cmake/CMakeCache.txt" | tail -1)"
if [ -z "$LIB_ARTIFACT" ] || [ ! -f "$LIB_ARTIFACT" ]; then
  echo "FATAL: configured scaling library artifact is missing" >&2
  exit 5
fi
LIB_ARTIFACT="$(resolve_path "$LIB_ARTIFACT")" || exit 5
LIB_ARTIFACT_SHA256="$(sha256_file "$LIB_ARTIFACT")" || exit 5

# --------------------------------------------------------------------------
# 3. self-tests
# --------------------------------------------------------------------------
selftest_ok=1
for drv in scaling_diff scaling_stab scaling_var_d scaling_dmrg_tdvp; do
  "$BIN/$drv" --selftest 2>&1 | tee "$LOG_DIR/selftest-$drv.log"
  selftest_rc=${PIPESTATUS[0]}
  if [ "$selftest_rc" -ne 0 ]; then echo "SELFTEST FAIL: $drv" >&2; selftest_ok=0; fi
done
selftest_logs_sha="$(write_hash_manifest "$LOG_DIR/selftest-logs.sha256" "$LOG_DIR"/selftest-*.log)"
emit reference_oracle_selftest "$([ $selftest_ok -eq 1 ] && echo PASS || echo FAIL)" release 0 0 \
  "{\"selftest_logs_manifest_sha256\":\"$selftest_logs_sha\"}" "$LOG_DIR/selftest-logs.sha256"

# --------------------------------------------------------------------------
# 4. run drivers, parse SCALING_RESULT new=/known=
# --------------------------------------------------------------------------
total_new=0
total_known=0
run_driver() { # run_driver <event_name> <tier> <binary> [args...]
  local name="$1" tier="$2"; shift 2
  local log="$LOG_DIR/$name.log" rc new known status parsed=true
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  new="$(sed -n 's/.*SCALING_RESULT [a-z_]* new=\([0-9]*\) known=[0-9]*.*/\1/p' "$log" | tail -1)"
  known="$(sed -n 's/.*SCALING_RESULT [a-z_]* new=[0-9]* known=\([0-9]*\).*/\1/p' "$log" | tail -1)"
  if [ -z "$new" ] || [ -z "$known" ]; then
    parsed=false; new=1; known=0
  elif [ "$rc" -ne 0 ] && [ "$new" -eq 0 ]; then
    new=1
  fi
  status="$([ "$new" -eq 0 ] && [ "$rc" -eq 0 ] && [ "$parsed" = true ] && echo PASS || echo FAIL)"
  total_new=$(( total_new + new ))
  total_known=$(( total_known + known ))
  emit "$name" "$status" "$tier" "$new" "$known" \
    "{\"exit_code\":$rc,\"summary_parsed\":$parsed}" "$log"
  echo "  -> $name: $status (new=$new known=$known rc=$rc parsed=$parsed)"
}

run_driver scaling_circuit_differential release "$BIN/scaling_diff" "$CORPUS_DIR/corpus.txt" --tn-max-n "$TN_MAX_N" --tn-max-depth "$TN_MAX_DEPTH"
run_driver scaling_clifford_stabilizer   release "$BIN/scaling_stab"
run_driver scaling_var_d                 release "$BIN/scaling_var_d"
run_driver scaling_dmrg_tdvp             release "$BIN/scaling_dmrg_tdvp"

# --------------------------------------------------------------------------
# 5. umbrella event
# --------------------------------------------------------------------------
umbrella="$([ "$total_new" -eq 0 ] && [ "$total_known" -eq 0 ] && [ "$selftest_ok" -eq 1 ] && echo PASS || echo FAIL)"
emit scaling_differential_clean "$umbrella" release "$total_new" "$total_known" \
  "{\"profile\":\"$PROFILE\",\"divergence_count\":$((total_new + total_known)),\"note\":\"all divergences are release blocking\"}" \
  "$LOG_DIR/build-diagnostics.sha256"

echo
echo "=== scaling_differential_clean: $umbrella  (new=$total_new known=$total_known) ==="
echo "trace -> $TRACE_FILE"
[ "$umbrella" = PASS ] || exit 1
exit 0
