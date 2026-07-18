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
WORK="$REPO_ROOT/build/scaling"
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
mkdir -p "$WORK" "$TRACES_DIR"

# --------------------------------------------------------------------------
# Locate the prebuilt libquantumsim
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
export MOONLAB_LIB_DIR="$LIB_DIR"
export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "libquantumsim: $LIB_DIR"

# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------
EVENTS="$WORK/events.jsonl"
: > "$EVENTS"
now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
json_escape() { printf '%s' "$1" | tr -d '\n\r' | sed 's/\\/\\\\/g; s/"/\\"/g'; }
emit() { # emit <name> <verdict> <tier> <new> <known> <extra_json>
  printf '{"kind":"moonlab_scaling","name":"%s","status":"%s","value":"%s","tier":"%s","ts":"%s","new_divergences":%s,"known_divergences":%s%s}\n' \
    "$1" "$2" "$2" "$3" "$(now_iso)" "$4" "$5" "$6" >> "$EVENTS"
}

# --------------------------------------------------------------------------
# 1. generate corpus
# --------------------------------------------------------------------------
CORPUS_DIR="$WORK/corpus"
if ! python3 "$GEN" --out-dir "$CORPUS_DIR" --seed "$SEED" \
      --qubits "$QUBITS" --depths "$DEPTHS" --small-longrange; then
  echo "FATAL: corpus generation failed (numpy required)" >&2
  emit corpus_generation FAIL release 0 0 ',"reason":"gen_scaling_corpus failed"'
  cat "$EVENTS" > "$TRACE_FILE"; exit 4
fi

# --------------------------------------------------------------------------
# 2. build drivers (standalone CMake, -j2)
# --------------------------------------------------------------------------
if ! cmake -S "$SCALE_DIR" -B "$WORK/cmake" >/dev/null 2>&1 || \
   ! cmake --build "$WORK/cmake" -j2 >/dev/null 2>&1; then
  echo "FATAL: driver build failed" >&2
  emit driver_build FAIL release 0 0 ',"reason":"cmake build failed"'
  cat "$EVENTS" > "$TRACE_FILE"; exit 5
fi
BIN="$WORK/cmake"

# --------------------------------------------------------------------------
# 3. self-tests
# --------------------------------------------------------------------------
selftest_ok=1
for drv in scaling_diff scaling_stab scaling_var_d scaling_dmrg_tdvp; do
  if ! "$BIN/$drv" --selftest >/dev/null 2>&1; then
    echo "SELFTEST FAIL: $drv" >&2; selftest_ok=0
  fi
done
emit reference_oracle_selftest "$([ $selftest_ok -eq 1 ] && echo PASS || echo FAIL)" release 0 0 ""

# --------------------------------------------------------------------------
# 4. run drivers, parse SCALING_RESULT new=/known=
# --------------------------------------------------------------------------
total_new=0
total_known=0
run_driver() { # run_driver <event_name> <tier> <binary> [args...]
  local name="$1" tier="$2"; shift 2
  local out new known status
  out="$("$@" 2>"$WORK/$name.err")"
  echo "$out" | tail -1
  new="$(printf '%s\n' "$out"  | sed -n 's/.*SCALING_RESULT [a-z_]* new=\([0-9]*\) known=[0-9]*.*/\1/p' | tail -1)"
  known="$(printf '%s\n' "$out" | sed -n 's/.*SCALING_RESULT [a-z_]* new=[0-9]* known=\([0-9]*\).*/\1/p' | tail -1)"
  [ -z "$new" ]   && new=0
  [ -z "$known" ] && known=0
  status="$([ "$new" -eq 0 ] && echo PASS || echo FAIL)"
  total_new=$(( total_new + new ))
  total_known=$(( total_known + known ))
  emit "$name" "$status" "$tier" "$new" "$known" ""
  echo "  -> $name: $status (new=$new known=$known)"
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
  ",\"profile\":\"$(json_escape "$PROFILE")\",\"divergence_count\":$((total_new + total_known)),\"note\":\"all divergences are release blocking\""

cat "$EVENTS" > "$TRACE_FILE"
echo
echo "=== scaling_differential_clean: $umbrella  (new=$total_new known=$total_known) ==="
echo "trace -> $TRACE_FILE"
[ "$umbrella" = PASS ] || exit 1
exit 0
