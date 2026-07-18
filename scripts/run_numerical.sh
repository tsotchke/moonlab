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
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_DIR="$REPO_ROOT/tests/numerical"
WORK="$REPO_ROOT/build/numerical"
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

mkdir -p "$WORK" "$TRACES_DIR"
[ -n "$OUT" ] || OUT="$TRACES_DIR/moonlab_numerical.jsonl"
: > "$OUT"
now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

emit() { # name status extra_json
  printf '{"kind":"moonlab_numerical","name":"%s","status":"%s","timestamp":"%s"%s}\n' \
    "$1" "$2" "$(now_iso)" "$3" >> "$OUT"
}

# --------------------------------------------------------------------------
# 1. Locate or build libquantumsim.
# --------------------------------------------------------------------------
find_lib() {
  local d
  for d in "$LIB_DIR" "$REPO_ROOT/build" "$REPO_ROOT/build_release" \
           "$REPO_ROOT/build/macos-arm64-release" "$REPO_ROOT/build/linux-x86_64-release"; do
    [ -n "$d" ] || continue
    if ls "$d"/libquantumsim.* >/dev/null 2>&1; then echo "$d"; return 0; fi
  done
  return 1
}

if ! LIB_DIR="$(find_lib)"; then
  echo "--- libquantumsim not found; building CPU-only Release ---"
  cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" -DCMAKE_BUILD_TYPE=Release \
        -DQSIM_BUILD_TESTS=OFF -DQSIM_BUILD_EXAMPLES=OFF \
        -DQSIM_BUILD_BENCHMARKS=OFF -DQSIM_ENABLE_METAL=OFF \
        >"$WORK/lib_cfg.log" 2>&1 || { echo "FATAL: lib configure failed"; cat "$WORK/lib_cfg.log"; exit 3; }
  cmake --build "$REPO_ROOT/build" --target quantumsim -j2 \
        >"$WORK/lib_build.log" 2>&1 || { echo "FATAL: lib build failed"; tail -40 "$WORK/lib_build.log"; exit 3; }
  LIB_DIR="$REPO_ROOT/build"
fi
echo "libquantumsim: $LIB_DIR"
export MOONLAB_LIB_DIR="$LIB_DIR"
export DYLD_LIBRARY_PATH="$LIB_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"

# --------------------------------------------------------------------------
# 2. Build the harnesses (standalone, -j2).
# --------------------------------------------------------------------------
echo "--- building numerical harnesses ---"
cmake -S "$NUM_DIR" -B "$WORK/cmake" -DCMAKE_BUILD_TYPE=Release \
      -DMOONLAB_LIB_DIR="$LIB_DIR" >"$WORK/cmake_cfg.log" 2>&1 \
  || { echo "FATAL: harness configure failed"; cat "$WORK/cmake_cfg.log"; exit 4; }
cmake --build "$WORK/cmake" --parallel 2 >"$WORK/cmake_build.log" 2>&1 \
  || { echo "FATAL: harness build failed"; tail -40 "$WORK/cmake_build.log"; exit 4; }

HARNESSES="t_rotations t_qft t_measure t_dmrg t_eigen_lapack t_eigen_jacobi t_svd_lapack t_svd_fallback"

run_one() { # binary -> prints "checks fails rc"
  local bin="$WORK/cmake/$1"
  local out rc summ checks fails
  out="$("$bin" 2>&1)"; rc=$?
  summ="$(printf '%s\n' "$out" | grep -o 'checks=[0-9]* fails=[0-9]*' | tail -1)"
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
for h in $HARNESSES; do
  read -r c f rc <<<"$(run_one "$h")"
  BASE_FAILS[$i]=$f
  echo "  $h: checks=$c fails=$f rc=$rc"
  [ "$c" -ge 0 ] && TOTAL_CHECKS=$((TOTAL_CHECKS + c)) || ANY_CRASH=1
  [ "$f" -ge 0 ] && TOTAL_FAILS=$((TOTAL_FAILS + f)) || ANY_CRASH=1
  # a harness intentionally returns its fail count as exit code; only treat
  # missing-summary (rc with no parsed counts) as a crash.
  if [ "$c" -lt 0 ]; then ANY_CRASH=1; fi
  DETAIL="$DETAIL{\"harness\":\"$h\",\"checks\":$c,\"fails\":$f}"
  [ $((i+1)) -lt $(echo $HARNESSES | wc -w) ] && DETAIL="$DETAIL,"
  i=$((i+1))
done

EDGE_STATUS="PASS"
[ "$TOTAL_FAILS" -eq 0 ] && [ "$ANY_CRASH" -eq 0 ] || EDGE_STATUS="FAIL"
emit "numerical_edge_clean" "$EDGE_STATUS" \
  ",\"total_checks\":$TOTAL_CHECKS,\"total_fails\":$TOTAL_FAILS,\"crash\":$ANY_CRASH,\"harnesses\":[$DETAIL]"
echo "numerical_edge_clean: $EDGE_STATUS (checks=$TOTAL_CHECKS fails=$TOTAL_FAILS crash=$ANY_CRASH)"

# --------------------------------------------------------------------------
# 4. Heap-poisoned run (uninit_clean).
#    macOS: MallocPreScribble(0xAA fresh)/MallocScribble(0x55 freed).
#    Linux: MALLOC_PERTURB_ fills allocations with a nonzero pattern.
# --------------------------------------------------------------------------
echo "--- heap-poisoned (uninit) run ---"
export MallocPreScribble=1 MallocScribble=1 MallocGuardEdges=1
export MALLOC_PERTURB_=170
UNINIT_NEWFAILS=0; UNINIT_CRASH=0
i=0
for h in $HARNESSES; do
  read -r c f rc <<<"$(run_one "$h")"
  local_base="${BASE_FAILS[$i]}"
  echo "  $h: fails=$f (baseline=$local_base) rc=$rc"
  if [ "$c" -lt 0 ]; then UNINIT_CRASH=1; fi
  if [ "$f" -ge 0 ] && [ "$local_base" -ge 0 ] && [ "$f" -gt "$local_base" ]; then
    UNINIT_NEWFAILS=$((UNINIT_NEWFAILS + (f - local_base)))
    echo "    NEW failures under poisoning: $((f - local_base)) (possible uninitialised read)"
  fi
  i=$((i+1))
done
unset MallocPreScribble MallocScribble MallocGuardEdges MALLOC_PERTURB_

UNINIT_STATUS="PASS"
[ "$UNINIT_NEWFAILS" -eq 0 ] && [ "$UNINIT_CRASH" -eq 0 ] || UNINIT_STATUS="FAIL"
emit "uninit_clean" "$UNINIT_STATUS" \
  ",\"new_failures_under_poison\":$UNINIT_NEWFAILS,\"crash\":$UNINIT_CRASH,\"tool\":\"heap-poison\",\"note\":\"valgrind/MSan run in CI (Linux); unavailable on macOS arm64\""
echo "uninit_clean: $UNINIT_STATUS (new_failures=$UNINIT_NEWFAILS crash=$UNINIT_CRASH)"

echo "--- JSONL: $OUT ---"
cat "$OUT"

# Non-zero exit if either gate failed (for CI signalling).
[ "$EDGE_STATUS" = "PASS" ] && [ "$UNINIT_STATUS" = "PASS" ]
