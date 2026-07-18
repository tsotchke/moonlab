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
  # Configure once (cache reused), then ALWAYS run the incremental build so a
  # stale $BUILD_DIR never yields false test failures against pre-fix binaries
  # (the trap that made full_suite_green report 2/137 while a clean build is
  # 179/179). QSIM_SMOKE_BUILD=1 forces a clean reconfigure.
  if [ "${QSIM_SMOKE_BUILD:-0}" = "1" ]; then rm -rf "$BUILD_DIR"; fi
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "== configuring $BUILD_DIR (-j$JOBS) =="
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_BUILD_TESTS=ON >/dev/null 2>&1 || return 1
  fi
  echo "== building $BUILD_DIR (-j$JOBS, incremental) =="
  cmake --build "$BUILD_DIR" --parallel "$JOBS" >/dev/null 2>&1 || return 1
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

# --- mpi_sharded_gpu_works -------------------------------------------------
# The small CTest is the regression guard; it cannot establish the release
# claim by itself.  PASS also requires a trace from run_mpi_sharded_gpu.sh
# proving N>32, at least two physical GPU endpoints, and at least one real
# partition-crossing halo swap on this exact commit.
check_mpi_sharded_gpu() {
  if ! need_build; then emit mpi_sharded_gpu_works FAIL "build failed"; return; fi

  local rx='mpi_sharded_gpu_ghz|sharded_gpu|multigpu|partition_gpu'
  local listed; listed="$(ctest --test-dir "$BUILD_DIR" -N -R "$rx" 2>/dev/null | grep -c 'Test #')"
  if [ "${listed:-0}" -eq 0 ]; then
    emit mpi_sharded_gpu_works FAIL "no MPI+CUDA sharding regression test is registered"
    return
  fi

  local out; out="$(ctest --test-dir "$BUILD_DIR" -R "$rx" --timeout 400 -j"$JOBS" 2>&1)"
  if ! printf '%s' "$out" | grep -q '100% tests passed'; then
    emit mpi_sharded_gpu_works FAIL "routine MPI+CUDA sharding regression failed"
    return
  fi

  local fleet_trace="$TRACE_DIR/moonlab_mpi_gpu.jsonl"
  if [ ! -f "$fleet_trace" ]; then
    emit mpi_sharded_gpu_works FAIL "routine test passed; N>32 fleet trace is absent"
    return
  fi

  local sha evidence
  sha="$(git rev-parse HEAD)"
  if evidence="$(python3 - "$fleet_trace" "$sha" <<'PY'
import json
import sys

trace_path, expected_sha = sys.argv[1:]
event = None
with open(trace_path, encoding="utf-8") as trace:
    for line in trace:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (candidate.get("kind") == "moonlab_mpi_gpu" and
                candidate.get("name") == "mpi_sharded_gpu_works"):
            event = candidate

if event is None:
    print("no mpi_sharded_gpu_works event")
    raise SystemExit(1)
if event.get("value", event.get("status")) != "PASS":
    print("latest fleet event is not PASS")
    raise SystemExit(1)
if event.get("commit_sha") != expected_sha:
    print("fleet event is for a different commit")
    raise SystemExit(1)

checks = {
    "n": int(event.get("n", 0)) > 32,
    "ranks": int(event.get("ranks", 0)) >= 2,
    "gpu_endpoints": int(event.get("gpu_endpoints", 0)) >= 2,
    "halo_swaps": int(event.get("halo_swaps", 0)) >= 1,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("fleet event lacks exact evidence: " + ",".join(failed))
    raise SystemExit(1)

print("N={n} ranks={ranks} GPUs={gpu_endpoints} halo_swaps={halo_swaps}".format(**event))
PY
)"; then
    emit mpi_sharded_gpu_works PASS "$listed routine test(s) pass; fleet $evidence"
  else
    emit mpi_sharded_gpu_works FAIL "routine test passed; ${evidence:-fleet evidence invalid}"
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

# --- all_discovered_bugs_closed -------------------------------------------
# The v1.2.0 bug-closure campaign gate (.icc/BUG_CLOSURE_CAMPAIGN.md): every
# adversarial-lane allowlist must have zero active (non-comment, non-blank)
# entries. A live quarantine entry is an OPEN bug; the release is blocked until
# all are removed (each removal paired with a fails-before/passes-after test).
check_quarantines_empty() {
  local total=0 detail=""
  local f
  for f in tests/oracle/KNOWN_FAILURES.txt \
           tests/differential/KNOWN_DIVERGENCES.txt \
           tests/scaling/KNOWN_DIVERGENCES.txt; do
    [ -f "$f" ] || continue
    # grep -c always prints a count (0 when nothing matches) but exits 1 on zero
    # matches; a `|| echo 0` would append a second 0 and break the arithmetic.
    local n; n="$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' "$f" 2>/dev/null)"; n="${n:-0}"
    total=$((total + n))
    [ "$n" -gt 0 ] && detail="$detail ${f##*/}=$n"
  done
  # fuzz crash-seed quarantine dirs (a genuine crash the owning lane must fix)
  local cp; cp="$(find tests/fuzz -type d -name 'crashes-pending' 2>/dev/null \
                  -exec sh -c 'ls -A "$1" 2>/dev/null | grep -q . && echo x' _ {} \; | wc -l | tr -d ' ')"
  total=$((total + cp))
  [ "${cp:-0}" -gt 0 ] && detail="$detail fuzz-crashes-pending=$cp"
  if [ "$total" -eq 0 ]; then
    emit all_discovered_bugs_closed PASS "every adversarial allowlist is empty"
  else
    emit all_discovered_bugs_closed FAIL "open quarantined bugs:$detail"
  fi
}

# --- deep-hunt lanes: relay their trace events into the gate ----------------
# tsan/numerical/scaling emit their own kind:"moonlab_*" JSONL; the release gate
# requires the umbrella clean events. FAIL (with reason) if a lane's trace is
# absent so a lane that never ran cannot masquerade as green.
check_deep_hunt() {
  local kind="$1" name="$2" tf="$TRACE_DIR/$3"
  if [ ! -f "$tf" ]; then
    emit "$name" FAIL "no $kind trace ($3) -- run the lane's script"; return
  fi
  # Parse JSON rather than depending on field order or whether a producer calls
  # its verdict "value" (tsan/numerical) or "status" (scaling).
  local v; v="$(python3 - "$tf" "$name" <<'PY'
import json
import sys

verdict = ""
with open(sys.argv[1], encoding="utf-8") as trace:
    for line in trace:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("name") == sys.argv[2]:
            verdict = event.get("value", event.get("status", ""))
print(verdict)
PY
)"
  if [ "$v" = "PASS" ]; then emit "$name" PASS "$kind lane clean"
  else emit "$name" FAIL "$kind lane value=${v:-none}"; fi
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
check_mpi_sharded_gpu
# Must match a test that proves DI certification is consumed by the delivered
# byte stream (bell_epoch_certified reflects real per-epoch certification), not
# the pre-existing unit_qrng_di which only exercises the isolated DI math.
check_ctest_gate qrng_certification_wired 'qrng_bell_certified_output|qrng_certified_delivery|bell_epoch_certified'
check_ctest_gate mlkem_official_kat       'mlkem_acvp|mlkem_official'

# v1.2.0 bug-closure campaign gates
check_quarantines_empty
check_deep_hunt tsan       tsan_clean                moonlab_tsan.jsonl
check_deep_hunt numerical  numerical_edge_clean      moonlab_numerical.jsonl
check_deep_hunt numerical  uninit_clean              moonlab_numerical.jsonl
check_deep_hunt scaling    scaling_differential_clean moonlab_scaling.jsonl

echo "== $FAILS high-severity check(s) FAILing =="
[ "$FAILS" -eq 0 ]
