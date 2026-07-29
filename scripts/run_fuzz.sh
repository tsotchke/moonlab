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
#       Either mode emits fuzz_corpus_clean exactly once.  The umbrella
#       verdict is withheld until after the end-of-lane source check, so the
#       trace can never carry two contradictory values for one check name.
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
# Exact-source release evidence requires that hook to be present in the source
# snapshot; this runner refuses transient self-wiring.

set -u

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-replay}"

BUILD_DIR="${QSIM_FUZZ_BUILD_DIR:-$REPO_ROOT/build-fuzz}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_fuzz.jsonl"
JOBS="${QSIM_FUZZ_JOBS:-2}"          # gentle by default; machine may be fragile
CORPORA="tests/fuzz/corpora"
DICTS="tests/fuzz/dicts"

# Every path this runner writes must remain in an ignored build tree or the
# canonical trace directory.  Environment overrides are conveniences, not
# authority to truncate arbitrary files.
BUILD_DIR="$(python3 - "$BUILD_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
case "$BUILD_DIR" in
  "$REPO_ROOT/build"|"$REPO_ROOT"/build-*|"$REPO_ROOT"/build_*) ;;
  *) echo "QSIM_FUZZ_BUILD_DIR must be a build*, build-*, or build_* path directly under $REPO_ROOT" >&2; exit 2 ;;
esac
TRACE_DIR="$(python3 - "$TRACE_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
if [ "$TRACE_DIR" != "$REPO_ROOT/scripts/icc_traces" ]; then
  echo "MOONLAB_TRACE_DIR must resolve to $REPO_ROOT/scripts/icc_traces" >&2
  exit 2
fi
TRACE="$TRACE_DIR/moonlab_fuzz.jsonl"

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
export DYLD_LIBRARY_PATH="$BUILD_DIR:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

mkdir -p "$TRACE_DIR"

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
# Worktree state over exactly the paths the fingerprint covers (tracked
# modifications + non-ignored untracked files, minus the trace directory).  If
# the end-of-lane fingerprint moves, diffing this names the offending paths
# instead of leaving two opaque hashes to be reverse-engineered.
SOURCE_STATE="$(git status --porcelain=v1 --untracked-files=all \
  -- . ':(exclude)scripts/icc_traces/**' | sort)"

CORPUS_SHA256="$(python3 - "$REPO_ROOT/$CORPORA" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(p for p in root.rglob("*") if p.is_file() and "crashes-pending" not in p.parts):
    rel = path.relative_to(root).as_posix().encode()
    data = path.read_bytes()
    digest.update(len(rel).to_bytes(8, "big"))
    digest.update(rel)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
PY
)"

# --- require the committed add_subdirectory hook ---------------------------
maybe_wire_hook() {
  if grep -q "add_subdirectory(tests/fuzz)" CMakeLists.txt; then
    return 0                     # integrator already wired it
  fi
  echo "[run_fuzz] exact-source evidence requires the committed tests/fuzz CMake hook" >&2
  return 1
}

# --- configure -------------------------------------------------------------
configure() {
  maybe_wire_hook || return 1
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
  python3 - "$TRACE" "$name" "$value" "$ts" "$detail" \
      "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$CORPUS_SHA256" <<'PY'
import json
import sys
trace, name, value, generated_at, detail, git_head, git_tree, dirty, fingerprint, corpus_sha256 = sys.argv[1:]
event = {
    "kind": "moonlab_fuzz",
    "name": name,
    "value": value,
    "generated_at": generated_at,
    "ts": generated_at,
    "detail": detail.replace("\n", " "),
    "git_head": git_head,
    "git_tree": git_tree,
    "dirty": dirty == "true",
    "source_fingerprint": fingerprint,
    "corpus_sha256": corpus_sha256,
}
with open(trace, "a", encoding="utf-8") as output:
    output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
  if [ "$value" = "PASS" ]; then
    printf '  [PASS] %-30s %s\n' "$name" "$detail"
  else
    printf '  [FAIL] %-30s %s\n' "$name" "$detail"
    FAILS=$((FAILS + 1))
  fi
}

# The umbrella verdict is recorded here and emitted once, at the very end of
# the run, after the source-drift check has had its say.  A lane that reported
# fuzz_corpus_clean twice with different values would be reporting two answers
# to one question.
UMBRELLA_VALUE=""
UMBRELLA_DETAIL=""
umbrella() {   # umbrella <PASS|FAIL> <detail>
  UMBRELLA_VALUE="$1"
  UMBRELLA_DETAIL="$2"
}

# ============================================================================
# REPLAY MODE
# ============================================================================
run_replay() {
  : > "$TRACE"
  configure || { umbrella FAIL "cmake configure failed"; return 1; }

  echo "[run_fuzz] building replay targets (-j$JOBS)"
  local build_targets=()
  for t in "${TARGETS[@]}"; do build_targets+=("${t}_replay"); done
  if ! cmake --build "$BUILD_DIR" --target "${build_targets[@]}" -j"$JOBS" >/dev/null 2>"$BUILD_DIR/replay_build.err"; then
    umbrella FAIL "replay build failed (see $BUILD_DIR/replay_build.err)"
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
    umbrella PASS "all ${#TARGETS[@]} surfaces replayed clean"
  else
    umbrella FAIL "one or more surfaces crashed on the seed corpus"
  fi
  return "$overall"
}

# ============================================================================
# SOAK MODE
# ============================================================================
run_soak() {
  local secs="${1:-300}"
  local only="${2:-}"

  # Optional single-target filter so CI can fan a matrix job per surface.
  # Validated before the trace is truncated: a usage error must not destroy
  # the previous run's evidence, and it produces no verdict of its own.
  if [ -n "$only" ]; then
    local found=0
    for t in "${TARGETS[@]}"; do [ "$t" = "$only" ] && found=1; done
    if [ "$found" -ne 1 ]; then
      echo "unknown target '$only'; valid: ${TARGETS[*]}" >&2
      return 2
    fi
    TARGETS=("$only")
  fi

  : > "$TRACE"
  configure || { umbrella FAIL "cmake configure failed"; return 1; }

  echo "[run_fuzz] building engine targets (-j$JOBS)"
  if ! cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS" >/dev/null 2>"$BUILD_DIR/engine_build.err"; then
    umbrella FAIL "engine build failed (see $BUILD_DIR/engine_build.err)"
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
    umbrella PASS "soak clean across ${#TARGETS[@]} surfaces (${secs}s each)"
  else
    umbrella FAIL "soak surfaced crash artifacts (see corpora/*/crashes-pending)"
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

# Usage errors ran no lane, so there is nothing to certify either way.
if [ "$rc" -eq 2 ]; then
  exit 2
fi

# --- end-of-lane source check ----------------------------------------------
# The evidence above is only worth what the snapshot it was measured on is
# worth, so the fingerprint has to still be the one the run started from.
# libFuzzer's own output is not source and is ignored (.gitignore: 40-hex
# corpus units and crashes-pending/), so a clean soak keeps this stable.
FINAL_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
if [ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
  FINAL_STATE="$(git status --porcelain=v1 --untracked-files=all \
    -- . ':(exclude)scripts/icc_traces/**' | sort)"
  DRIFT="$(diff <(printf '%s\n' "$SOURCE_STATE") <(printf '%s\n' "$FINAL_STATE") \
    | grep -E '^[<>]' | head -8 | tr '\n' ' ')"
  umbrella FAIL "source changed during lane: start=$SOURCE_FINGERPRINT end=${FINAL_FINGERPRINT:-unknown} drift=${DRIFT:-<no worktree status delta; content changed in place>}"
  rc=1
fi

# The one and only umbrella verdict for this run.  Silence and a green
# verdict on a red exit code are both failures to report, so they are failures.
if [ -z "$UMBRELLA_VALUE" ]; then
  umbrella FAIL "lane produced no verdict"
  rc=1
elif [ "$rc" -ne 0 ] && [ "$UMBRELLA_VALUE" = "PASS" ]; then
  umbrella FAIL "lane exited $rc with a PASS verdict recorded: $UMBRELLA_DETAIL"
fi
emit fuzz_corpus_clean "$UMBRELLA_VALUE" "$UMBRELLA_DETAIL"

echo
echo "[run_fuzz] trace written to $TRACE ; failures: $FAILS"
exit "$rc"
