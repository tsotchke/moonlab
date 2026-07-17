#!/usr/bin/env bash
#
# run_fuzz.sh -- driver for the Moonlab coverage-guided fuzz lane.
#
# Two modes:
#
#   ./scripts/run_fuzz.sh [replay]
#       Build the <target>_replay binaries and replay every seed corpus
#       through them under AddressSanitizer + UndefinedBehaviorSanitizer.
#       Fast (< ~5 min), deterministic, CI-usable.  Emits one JSONL trace
#       event per surface (kind "moonlab_fuzz") plus an umbrella
#       fuzz_corpus_clean event to scripts/icc_traces/moonlab_fuzz.jsonl.
#       Exits nonzero if any seed crashes a sanitizer.
#
#   ./scripts/run_fuzz.sh soak <seconds>
#       Build the libFuzzer engine binaries and run an actual campaign per
#       target for <seconds> each, seeded from the corpus.  New crash /
#       OOM / timeout artifacts are quarantined under
#       corpora/<target>/crashes-pending/ and reported.  For nightly /
#       workflow_dispatch CI, not PR CI.
#
# The library under test (quantumsim) is compiled with ASan+UBSan by
# tests/fuzz/CMakeLists.txt, so the sanitizers see the parser / crypto /
# ABI code, not just the harness.
#
# Requires Clang.  On macOS the Homebrew clang + a libc++/-ld_classic
# workaround are handled by tests/fuzz/CMakeLists.txt.
#
# Integration note: tests/fuzz is wired in by the root CMakeLists via
#   if(QSIM_ENABLE_FUZZING) add_subdirectory(tests/fuzz) endif()
# If that hook is not yet present, this script appends it to a scratch
# copy of CMakeLists.txt for the duration of the run and restores the
# file on exit -- it never leaves the tree modified.

set -u

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-replay}"

BUILD_DIR="${QSIM_FUZZ_BUILD_DIR:-build-fuzz}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_fuzz.jsonl"
JOBS="${QSIM_FUZZ_JOBS:-2}"          # gentle by default; machine may be fragile
CORPORA="tests/fuzz/corpora"
DICTS="tests/fuzz/dicts"

# Surfaces == executable / corpus / trace-name basenames.
TARGETS=(
  control_plane_protocol_fuzz
  circuit_deserialize_fuzz
  config_parse_fuzz
  mlkem_decode_fuzz
  entropy_input_fuzz
  abi_boundary_fuzz
)

# --- compiler selection -----------------------------------------------------
# Prefer an explicit CC/CXX; else a Homebrew clang on macOS (its libFuzzer
# runtime is complete); else plain clang/clang++.
pick_clang() {
  if [ -n "${CC:-}" ] && [ -n "${CXX:-}" ]; then
    FUZZ_CC="$CC"; FUZZ_CXX="$CXX"; return
  fi
  for pfx in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
    if [ -x "$pfx/bin/clang" ]; then
      FUZZ_CC="$pfx/bin/clang"; FUZZ_CXX="$pfx/bin/clang++"; return
    fi
  done
  FUZZ_CC="clang"; FUZZ_CXX="clang++"
}
pick_clang

# --- sanitizer runtime options ----------------------------------------------
# macOS LeakSanitizer reports only one-time ObjC/dyld/libc init allocations
# here (not real leaks), so leak detection is disabled on Darwin and left
# on elsewhere where LSan is reliable.
case "$(uname -s)" in
  Darwin) DETECT_LEAKS=0 ;;
  *)      DETECT_LEAKS="${QSIM_FUZZ_DETECT_LEAKS:-1}" ;;
esac
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=${DETECT_LEAKS}:print_summary=1:handle_abort=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
# So the replay/engine binaries find the instrumented libquantumsim dylib.
export DYLD_LIBRARY_PATH="$REPO_ROOT/$BUILD_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$REPO_ROOT/$BUILD_DIR:${LD_LIBRARY_PATH:-}"

mkdir -p "$TRACE_DIR"

# --- self-wiring of the add_subdirectory hook (restored on exit) ------------
HOOK_ADDED=0
CMAKE_BACKUP=""
maybe_wire_hook() {
  if grep -q "add_subdirectory(tests/fuzz)" CMakeLists.txt; then
    return 0                     # integrator already wired it
  fi
  CMAKE_BACKUP="$(mktemp)"
  cp CMakeLists.txt "$CMAKE_BACKUP"
  cat >> CMakeLists.txt <<'HOOK'

# --- run_fuzz.sh self-wiring shim (auto-removed after the run) ---------------
option(QSIM_ENABLE_FUZZING "Build libFuzzer/AFL++ fuzz harnesses" OFF)
if(QSIM_ENABLE_FUZZING)
    add_subdirectory(tests/fuzz)
endif()
HOOK
  HOOK_ADDED=1
  echo "[run_fuzz] note: temporarily appended add_subdirectory(tests/fuzz) to CMakeLists.txt (restored on exit)"
}
cleanup() {
  if [ "$HOOK_ADDED" = "1" ] && [ -n "$CMAKE_BACKUP" ] && [ -f "$CMAKE_BACKUP" ]; then
    cp "$CMAKE_BACKUP" CMakeLists.txt
    rm -f "$CMAKE_BACKUP"
    echo "[run_fuzz] restored CMakeLists.txt"
  fi
}
trap cleanup EXIT INT TERM

# --- configure -------------------------------------------------------------
configure() {
  maybe_wire_hook
  echo "[run_fuzz] configuring $BUILD_DIR with clang=$FUZZ_CC"
  cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DQSIM_ENABLE_FUZZING=ON \
    -DQSIM_BUILD_TESTS=OFF \
    -DQSIM_WERROR=OFF \
    -DQSIM_ENABLE_METAL=OFF \
    -DQSIM_ENABLE_OPENMP=OFF \
    -DQSIM_ENABLE_TLS=OFF \
    -DCMAKE_C_COMPILER="$FUZZ_CC" \
    -DCMAKE_CXX_COMPILER="$FUZZ_CXX" \
    "$@" >/dev/null
}

# --- JSONL trace emit -------------------------------------------------------
FAILS=0
emit() {   # emit <name> <PASS|FAIL> <detail>
  local name="$1" value="$2" detail="${3:-}"
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="${detail//\"/\'}"
  detail="${detail//$'\n'/ }"
  printf '{"kind":"moonlab_fuzz","name":"%s","value":"%s","ts":"%s","detail":"%s"}\n' \
    "$name" "$value" "$ts" "$detail" >> "$TRACE"
  if [ "$value" = "PASS" ]; then
    printf '  [PASS] %-30s %s\n' "$name" "$detail"
  else
    printf '  [FAIL] %-30s %s\n' "$name" "$detail"
    FAILS=$((FAILS + 1))
  fi
}

# ============================================================================
# REPLAY MODE
# ============================================================================
run_replay() {
  : > "$TRACE"
  configure || { emit fuzz_corpus_clean FAIL "cmake configure failed"; return 1; }

  echo "[run_fuzz] building replay targets (-j$JOBS)"
  local build_targets=()
  for t in "${TARGETS[@]}"; do build_targets+=("${t}_replay"); done
  if ! cmake --build "$BUILD_DIR" --target "${build_targets[@]}" -j"$JOBS" >/dev/null 2>"$BUILD_DIR/replay_build.err"; then
    emit fuzz_corpus_clean FAIL "replay build failed (see $BUILD_DIR/replay_build.err)"
    return 1
  fi

  local overall=0
  for t in "${TARGETS[@]}"; do
    local bin="$BUILD_DIR/tests/fuzz/${t}_replay"
    local corpus="$CORPORA/$t"
    if [ ! -x "$bin" ]; then emit "$t" FAIL "missing binary $bin"; overall=1; continue; fi

    # Top-level seeds only: crashes-pending/ is excluded from the gate.
    local seeds=()
    while IFS= read -r -d '' f; do seeds+=("$f"); done \
      < <(find "$corpus" -maxdepth 1 -type f -print0 2>/dev/null)
    if [ "${#seeds[@]}" -eq 0 ]; then emit "$t" FAIL "no seeds in $corpus"; overall=1; continue; fi

    local log="$BUILD_DIR/replay_${t}.log"
    if "$bin" "${seeds[@]}" >"$log" 2>&1; then
      emit "$t" PASS "$(( ${#seeds[@]} )) seeds clean under ASan/UBSan"
    else
      local why; why="$(grep -m1 -iE 'runtime error|ERROR: AddressSanitizer|SUMMARY:' "$log" | head -c 160)"
      emit "$t" FAIL "sanitizer abort: ${why:-see $log}"
      overall=1
    fi
  done

  if [ "$overall" -eq 0 ]; then
    emit fuzz_corpus_clean PASS "all ${#TARGETS[@]} surfaces replayed clean"
  else
    emit fuzz_corpus_clean FAIL "one or more surfaces crashed on the seed corpus"
  fi
  return "$overall"
}

# ============================================================================
# SOAK MODE
# ============================================================================
run_soak() {
  local secs="${1:-300}"
  local only="${2:-}"
  : > "$TRACE"

  # Optional single-target filter so CI can fan a matrix job per surface.
  if [ -n "$only" ]; then
    local found=0
    for t in "${TARGETS[@]}"; do [ "$t" = "$only" ] && found=1; done
    if [ "$found" -ne 1 ]; then
      echo "unknown target '$only'; valid: ${TARGETS[*]}" >&2
      return 2
    fi
    TARGETS=("$only")
  fi

  configure || { emit fuzz_corpus_clean FAIL "cmake configure failed"; return 1; }

  echo "[run_fuzz] building engine targets (-j$JOBS)"
  if ! cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS" >/dev/null 2>"$BUILD_DIR/engine_build.err"; then
    emit fuzz_corpus_clean FAIL "engine build failed (see $BUILD_DIR/engine_build.err)"
    return 1
  fi

  local overall=0
  for t in "${TARGETS[@]}"; do
    local bin="$BUILD_DIR/tests/fuzz/$t"
    local corpus="$CORPORA/$t"
    local quarantine="$corpus/crashes-pending"
    mkdir -p "$quarantine"
    if [ ! -x "$bin" ]; then emit "$t" FAIL "missing engine binary $bin"; overall=1; continue; fi

    local dict_arg=()
    [ -f "$DICTS/$t.dict" ] && dict_arg=("-dict=$DICTS/$t.dict")

    echo "[run_fuzz] soaking $t for ${secs}s"
    local log="$BUILD_DIR/soak_${t}.log"
    # -rss_limit_mb caps the documented state-vector amplification so a
    # single large-qubit request is reported rather than OOM-killing CI.
    "$bin" -max_total_time="$secs" -timeout=25 \
      -rss_limit_mb=4096 -malloc_limit_mb=2048 -max_len=65536 \
      -artifact_prefix="$quarantine/" \
      "${dict_arg[@]}" "$corpus" >"$log" 2>&1 || true

    # New artifacts == findings.
    local arts
    arts="$(find "$quarantine" -maxdepth 1 -type f \( -name 'crash-*' -o -name 'oom-*' -o -name 'timeout-*' -o -name 'leak-*' \) 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$arts" -eq 0 ]; then
      emit "$t" PASS "no new crashes in ${secs}s ($(grep -Eo '#[0-9]+' "$log" | tail -1 || echo '?') execs)"
    else
      emit "$t" FAIL "$arts artifact(s) quarantined in $quarantine"
      overall=1
    fi
  done

  if [ "$overall" -eq 0 ]; then
    emit fuzz_corpus_clean PASS "soak clean across ${#TARGETS[@]} surfaces (${secs}s each)"
  else
    emit fuzz_corpus_clean FAIL "soak surfaced crash artifacts (see corpora/*/crashes-pending)"
  fi
  return "$overall"
}

# ============================================================================
case "$MODE" in
  replay)
    run_replay; rc=$?
    ;;
  soak)
    run_soak "${2:-300}" "${3:-}"; rc=$?
    ;;
  *)
    echo "usage: $0 [replay | soak <seconds> [target]]" >&2
    exit 2
    ;;
esac

echo
echo "[run_fuzz] trace written to $TRACE ; failures: $FAILS"
exit "$rc"
