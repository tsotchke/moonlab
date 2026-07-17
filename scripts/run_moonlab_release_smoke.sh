#!/usr/bin/env bash
# Whole-system release smoke: the evidence producer for the moonlab_smoke
# trace events that .icc/completion-oracles.yaml (moonlab-release-readiness)
# gates on. Each check writes one JSON line to scripts/icc_traces/moonlab_smoke.jsonl
# with kind "moonlab_smoke", a name matching an oracle criterion, and value
# PASS or FAIL. A missing test/tool yields FAIL (honest: the gate stays red
# until the owning lane lands the check), never a silent skip.
#
# Usage:
#   scripts/run_moonlab_release_smoke.sh                # use existing $BUILD_DIR, run checks
#   QSIM_SMOKE_BUILD=1 scripts/run_moonlab_release_smoke.sh   # (re)configure+build first
#   BUILD_DIR=build QSIM_SMOKE_ASAN=1 ...               # also run the sanitizer lane
#
# Exit code: 0 iff every high-severity check PASSed. The ICC gate reads the
# trace file, not the exit code, but CI can use the exit code directly.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
ASAN_DIR="${ASAN_DIR:-build-asan-smoke}"
HIDDEN_DIR="${HIDDEN_DIR:-build-hidden-smoke}"
EX_DIR="${EX_DIR:-build-examples-smoke}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_smoke.jsonl"
JOBS="${QSIM_SMOKE_JOBS:-2}"          # gentle by default; machine may be fragile
ICC="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"

mkdir -p "$TRACE_DIR"
: > "$TRACE"
FAILS=0

emit() {   # emit <name> <PASS|FAIL> <detail>
  local name="$1" value="$2" detail="${3:-}"
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="${detail//\"/\'}"
  printf '{"kind":"moonlab_smoke","name":"%s","value":"%s","ts":"%s","detail":"%s"}\n' \
    "$name" "$value" "$ts" "$detail" >> "$TRACE"
  if [ "$value" = "PASS" ]; then
    printf '  [PASS] %-26s %s\n' "$name" "$detail"
  else
    printf '  [FAIL] %-26s %s\n' "$name" "$detail"
    FAILS=$((FAILS + 1))
  fi
}

need_build() {
  if [ ! -d "$BUILD_DIR" ] || [ "${QSIM_SMOKE_BUILD:-0}" = "1" ]; then
    echo "== configuring + building $BUILD_DIR (-j$JOBS) =="
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_BUILD_TESTS=ON >/dev/null 2>&1 || return 1
    cmake --build "$BUILD_DIR" --parallel "$JOBS" >/dev/null 2>&1 || return 1
  fi
  return 0
}

# --- full_suite_green -------------------------------------------------------
check_full_suite() {
  if ! need_build; then emit full_suite_green FAIL "build failed"; return; fi
  local out; out="$(ctest --test-dir "$BUILD_DIR" -LE "long|memory_heavy" \
                     --timeout 400 -j"$JOBS" 2>&1)"
  local notrun; notrun="$(printf '%s' "$out" | grep -c 'Not Run')"
  if printf '%s' "$out" | grep -q '100% tests passed' && [ "$notrun" -eq 0 ]; then
    emit full_suite_green PASS "$(printf '%s' "$out" | grep -Eo '[0-9]+ tests passed' | head -1)"
  else
    emit full_suite_green FAIL "failures or Not Run present: $(printf '%s' "$out" | grep -E 'failed|Not Run' | head -1)"
  fi
}

# --- asan_ubsan_clean -------------------------------------------------------
check_asan() {
  if [ "${QSIM_SMOKE_ASAN:-1}" != "1" ]; then
    emit asan_ubsan_clean FAIL "ASan lane not run (QSIM_SMOKE_ASAN != 1)"; return
  fi
  cmake -S . -B "$ASAN_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DQSIM_ENABLE_SANITIZERS=ON -DQSIM_BUILD_TESTS=ON >/dev/null 2>&1 \
    || { emit asan_ubsan_clean FAIL "configure failed"; return; }
  cmake --build "$ASAN_DIR" --parallel "$JOBS" >/dev/null 2>&1 \
    || { emit asan_ubsan_clean FAIL "build failed"; return; }
  # LeakSanitizer is unsupported on Darwin; detect_leaks=1 aborts every test at
  # init there. Match CI (detect_leaks=0): the gate is for UAF/OOB/UB, not leaks.
  local out; out="$(ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
      ctest --test-dir "$ASAN_DIR" -L "core|tn|ca_mps|algorithms" \
      -LE "long|memory_heavy" --timeout 600 -j"$JOBS" 2>&1)"
  if printf '%s' "$out" | grep -q '100% tests passed'; then
    emit asan_ubsan_clean PASS "core|tn|ca_mps|algorithms clean under ASan+UBSan"
  else
    emit asan_ubsan_clean FAIL "$(printf '%s' "$out" | grep -E 'failed|runtime error|ERROR' | head -1)"
  fi
}

# --- ci_all_green -----------------------------------------------------------
check_ci() {
  if ! command -v gh >/dev/null 2>&1; then emit ci_all_green FAIL "gh not available"; return; fi
  local sha; sha="$(git rev-parse HEAD)"
  local concl; concl="$(gh run list --repo tsotchke/moonlab --commit "$sha" \
      --workflow CI --limit 1 --json conclusion -q '.[0].conclusion' 2>/dev/null)"
  if [ "$concl" = "success" ]; then
    emit ci_all_green PASS "CI success on $sha"
  else
    emit ci_all_green FAIL "CI conclusion='${concl:-none}' on $sha"
  fi
}

# --- audit-finding gates: run the lane's regression test by regex ----------
# PASS iff at least one matching test exists AND all matching tests pass.
check_ctest_gate() {  # <event_name> <ctest -R regex>
  local name="$1" rx="$2"
  if ! need_build; then emit "$name" FAIL "build failed"; return; fi
  local listed; listed="$(ctest --test-dir "$BUILD_DIR" -N -R "$rx" 2>/dev/null | grep -c 'Test #')"
  if [ "${listed:-0}" -eq 0 ]; then
    emit "$name" FAIL "no test matches /$rx/ yet (owning lane not landed)"; return
  fi
  local out; out="$(ctest --test-dir "$BUILD_DIR" -R "$rx" --timeout 400 -j"$JOBS" 2>&1)"
  if printf '%s' "$out" | grep -q '100% tests passed'; then
    emit "$name" PASS "$listed test(s) matching /$rx/ pass"
  else
    emit "$name" FAIL "$(printf '%s' "$out" | grep -E 'failed' | head -1)"
  fi
}

# --- zero_phantom_api / zero_odr_collision via ICC --------------------------
check_phantom() {
  local n; n="$("$ICC" phantom-api --repo moonlab 2>/dev/null \
      | python3 -c 'import json,sys;print(json.load(sys.stdin).get("phantom_count","?"))' 2>/dev/null)"
  if [ "$n" = "0" ]; then emit zero_phantom_api PASS "0 phantom declarations"
  else emit zero_phantom_api FAIL "phantom_count=${n:-error}"; fi
}
check_odr() {
  local n; n="$("$ICC" odr-audit --repo moonlab 2>/dev/null \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("summary",{}).get("high","?"))' 2>/dev/null)"
  if [ "$n" = "0" ]; then emit zero_odr_collision PASS "0 high-severity ODR collisions"
  else emit zero_odr_collision FAIL "high-severity collisions=${n:-error}"; fi
}

# --- hidden_visibility_abi --------------------------------------------------
check_hidden_visibility() {
  cmake -S . -B "$HIDDEN_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DQSIM_HIDDEN_VISIBILITY=ON -DQSIM_BUILD_TESTS=ON >/dev/null 2>&1 \
    || { emit hidden_visibility_abi FAIL "configure failed"; return; }
  cmake --build "$HIDDEN_DIR" --parallel "$JOBS" >/dev/null 2>&1 \
    || { emit hidden_visibility_abi FAIL "build failed"; return; }
  local out; out="$(ctest --test-dir "$HIDDEN_DIR" -L "abi|bindings" --timeout 300 -j"$JOBS" 2>&1)"
  if printf '%s' "$out" | grep -q '100% tests passed'; then
    emit hidden_visibility_abi PASS "abi|bindings green under hidden visibility"
  else
    emit hidden_visibility_abi FAIL "$(printf '%s' "$out" | grep -E 'failed|Not Run' | head -1)"
  fi
}

# --- docs_apis_exist --------------------------------------------------------
# Delegates to the docs lane's checker if present; otherwise FAIL (not landed).
check_docs_apis() {
  if [ -x scripts/check_docs_apis.sh ]; then
    if scripts/check_docs_apis.sh >/dev/null 2>&1; then
      emit docs_apis_exist PASS "all documented APIs resolve to headers"
    else
      emit docs_apis_exist FAIL "documented API(s) do not resolve"
    fi
  else
    emit docs_apis_exist FAIL "scripts/check_docs_apis.sh not present (docs lane not landed)"
  fi
}

# --- examples_all_build -----------------------------------------------------
check_examples() {
  cmake -S . -B "$EX_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DQSIM_BUILD_EXAMPLES=ON -DQSIM_BUILD_TESTS=OFF >/dev/null 2>&1 \
    || { emit examples_all_build FAIL "configure failed"; return; }
  if cmake --build "$EX_DIR" --parallel "$JOBS" >/dev/null 2>&1; then
    emit examples_all_build PASS "all registered examples compile"
  else
    emit examples_all_build FAIL "an example failed to build"
  fi
}

# --- binding_versions_synced ------------------------------------------------
check_versions() {
  if [ -f scripts/version_tool.py ]; then
    if python3 scripts/version_tool.py check --tag "v$(cat VERSION.txt)" >/dev/null 2>&1 \
       || python3 scripts/version_tool.py check >/dev/null 2>&1; then
      emit binding_versions_synced PASS "version_tool check passed"
    else
      emit binding_versions_synced FAIL "version_tool check failed"
    fi
  else
    emit binding_versions_synced FAIL "scripts/version_tool.py not present"
  fi
}

echo "== Moonlab release smoke -> $TRACE =="
check_full_suite
check_asan
check_ci
check_ctest_gate gpu_host_sync_contract   'gpu_sync|gpu_contract|gpu_measure'
check_ctest_gate tdvp_projector_splitting 'tdvp_projector|tdvp_real_time|tdvp_validation'
check_phantom
check_odr
check_hidden_visibility
check_docs_apis
check_examples
check_versions
check_ctest_gate bindings_suites_green    'python_bindings|rust_bindings|js_'
check_ctest_gate mpi_sharded_gpu_works    'sharded_gpu|multigpu|partition_gpu'
# Must match a test that proves DI certification is consumed by the delivered
# byte stream (bell_epoch_certified reflects real per-epoch certification), not
# the pre-existing unit_qrng_di which only exercises the isolated DI math.
check_ctest_gate qrng_certification_wired 'qrng_bell_certified_output|qrng_certified_delivery|bell_epoch_certified'
check_ctest_gate mlkem_official_kat       'mlkem_acvp|mlkem_official'

echo "== $FAILS high-severity check(s) FAILing =="
[ "$FAILS" -eq 0 ]
