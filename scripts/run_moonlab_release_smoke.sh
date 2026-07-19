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
cd "$REPO_ROOT" || exit 2

# Print the shared canonical identity for the exact source snapshot under test.
source_identity() {
  python3 "$REPO_ROOT/scripts/moonlab_source_identity.py" --repo-root "$REPO_ROOT"
}

if [ "${1:-}" = "--source-identity" ]; then
  source_identity
  exit 0
fi

SOURCE_IDENTITY_JSON="$(source_identity)" || exit 2
IFS=$'\t' read -r SOURCE_GIT_HEAD SOURCE_GIT_TREE SOURCE_DIRTY SOURCE_FINGERPRINT SOURCE_CAPTURED_AT \
  < <(python3 - "$SOURCE_IDENTITY_JSON" <<'PY'
import json
import sys
identity = json.loads(sys.argv[1])
print("\t".join(str(identity[key]).lower() if key == "dirty" else str(identity[key]) for key in (
    "git_head", "git_tree", "dirty", "source_fingerprint", "captured_at"
)))
PY
)

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-moonlab-release-smoke}"
ASAN_DIR="${ASAN_DIR:-$REPO_ROOT/build-asan-smoke}"
HIDDEN_DIR="${HIDDEN_DIR:-$REPO_ROOT/build-hidden-smoke}"
EX_DIR="${EX_DIR:-$REPO_ROOT/build-examples-smoke}"
LOG_DIR="${MOONLAB_SMOKE_LOG_DIR:-${BUILD_DIR}-logs}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_smoke.jsonl"
JOBS="${QSIM_SMOKE_JOBS:-2}"          # gentle by default; machine may be fragile
ICC="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"
LOCK_DIR="${MOONLAB_SMOKE_LOCK_DIR:-${BUILD_DIR}.moonlab-smoke.lock}"
LIBIRREP_SOURCE="${MOONLAB_LIBIRREP_SOURCE:-$REPO_ROOT/../libirrep}"
LIBIRREP_ROOT=""
LIBIRREP_ARTIFACT_SHA256=""
LIBIRREP_SOURCE_FINGERPRINT=""
LIBIRREP_SOURCE_GIT_HEAD=""
LIBIRREP_SOURCE_GIT_TREE=""
LIBIRREP_SOURCE_DIRTY="false"
LIBIRREP_WORKTREE_DIRTY="true"
LIBIRREP_BUILD_LOG_SHA256=""
LIBIRREP_STAGE_STATUS=0

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
    "$REPO_ROOT/build"|"$REPO_ROOT"/build-*|"$REPO_ROOT"/build_*) printf '%s\n' "$resolved" ;;
    *) printf 'release-smoke writable path must resolve under a build*, build-*, or build_* tree: %s\n' "$resolved" >&2; return 1 ;;
  esac
}

# Validate every environment-influenced writable path before the first mkdir or
# truncation.  Trace evidence has one canonical repository location.
BUILD_DIR="$(require_build_path "$BUILD_DIR")" || exit 2
ASAN_DIR="$(require_build_path "$ASAN_DIR")" || exit 2
HIDDEN_DIR="$(require_build_path "$HIDDEN_DIR")" || exit 2
EX_DIR="$(require_build_path "$EX_DIR")" || exit 2
LOG_DIR="$(require_build_path "$LOG_DIR")" || exit 2
LOCK_DIR="$(require_build_path "$LOCK_DIR")" || exit 2
LIBIRREP_SOURCE="$(resolve_path "$LIBIRREP_SOURCE")" || exit 2
TRACE_DIR="$(resolve_path "$TRACE_DIR")" || exit 2
if [ "$TRACE_DIR" != "$REPO_ROOT/scripts/icc_traces" ]; then
  printf 'MOONLAB_TRACE_DIR must resolve to %s\n' "$REPO_ROOT/scripts/icc_traces" >&2
  exit 2
fi
TRACE="$TRACE_DIR/moonlab_smoke.jsonl"

mkdir -p "$TRACE_DIR" "$LOG_DIR"
FAILS=0

# CMake and CTest are not safe when two release-smoke invocations mutate the
# same build tree.  A dedicated default build plus an atomic lock prevents the
# transient 16/170 and 3/7 collision failures seen during the release hunt.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'Moonlab release smoke lock already held: %s\n' "$LOCK_DIR" >&2
  exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
# Only the lock owner may invalidate or append the canonical smoke trace.
: > "$TRACE"

emit() {   # emit <name> <PASS|FAIL> <detail> [artifact_path]
  local name="$1" value="$2" detail="${3:-}" artifact_path="${4:-}"
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$TRACE" "$name" "$value" "$ts" "$detail" \
      "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$artifact_path" \
      "$LIBIRREP_SOURCE" "$LIBIRREP_ROOT" "$LIBIRREP_ARTIFACT_SHA256" "$LIBIRREP_SOURCE_FINGERPRINT" \
      "$LIBIRREP_SOURCE_GIT_HEAD" "$LIBIRREP_SOURCE_GIT_TREE" "$LIBIRREP_SOURCE_DIRTY" \
      "$LIBIRREP_WORKTREE_DIRTY" "$LIBIRREP_BUILD_LOG_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    trace, name, value, generated_at, detail, git_head, git_tree, dirty,
    fingerprint, artifact_path, libirrep_source, libirrep_root, libirrep_hash,
    libirrep_fingerprint, libirrep_head, libirrep_tree, libirrep_dirty,
    libirrep_worktree_dirty, libirrep_build_log_hash,
) = sys.argv[1:]
event = {
    "kind": "moonlab_smoke",
    "name": name,
    "value": value,
    "generated_at": generated_at,
    "ts": generated_at,
    "detail": detail,
    "git_head": git_head,
    "git_tree": git_tree,
    "dirty": dirty == "true",
    "source_fingerprint": fingerprint,
}
if libirrep_hash:
    event["libirrep_source"] = libirrep_source
    event["libirrep_staged_root"] = libirrep_root
    event["libirrep_artifact_sha256"] = libirrep_hash
    event["libirrep_source_fingerprint"] = libirrep_fingerprint
    event["libirrep_git_head"] = libirrep_head
    event["libirrep_git_tree"] = libirrep_tree
    event["libirrep_dirty"] = libirrep_dirty == "true"
    event["libirrep_worktree_dirty"] = libirrep_worktree_dirty == "true"
    event["libirrep_build_log_sha256"] = libirrep_build_log_hash
if artifact_path:
    artifact = Path(artifact_path)
    if artifact.is_file():
        event["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        event["artifact_path"] = artifact_path
with open(trace, "a", encoding="utf-8") as output:
    output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
  if [ "$value" = "PASS" ]; then
    printf '  [PASS] %-26s %s\n' "$name" "$detail"
  else
    printf '  [FAIL] %-26s %s\n' "$name" "$detail"
    FAILS=$((FAILS + 1))
  fi
}

save_log() { # save_log <path> <captured-output>
  mkdir -p "$(dirname "$1")"
  printf '%s\n' "$2" > "$1"
}

ctest_log_green() { # 100% passed is insufficient: CTest calls exit-77 a skip.
  grep -q '100% tests passed' "$1" \
    && ! grep -Eq 'Not Run|\*\*\*Skipped|The following tests did not run:' "$1"
}

ctest_log_detail() { # concise exact names; the hashed full log remains available.
  python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
failed = []
skipped = []
section = None
for line in lines:
    if line.startswith("The following tests FAILED:"):
        section = failed
        continue
    if line.startswith("The following tests did not run:"):
        section = skipped
        continue
    match = re.match(r"\s*\d+\s+-\s+(.+?)(?:\s+\(|$)", line)
    if section is not None and match:
        section.append(match.group(1).strip())
detail = []
if failed:
    detail.append("failed=" + ",".join(failed))
if skipped:
    detail.append("skipped=" + ",".join(skipped))
if not detail:
    for line in lines:
        if re.search(r"failed|Not Run|runtime error|ERROR", line, re.I):
            detail.append(line.strip())
            break
print("; ".join(detail) or "CTest did not report a clean complete matrix")
PY
}

stage_libirrep() {
  # Build the sibling read-only: all generated state lives below this smoke's
  # validated ignored build tree.  A source-content fingerprint selects a fresh
  # stage root, so deleted/renamed headers or objects cannot leak from an older
  # sibling snapshot.
  if [ "$LIBIRREP_STAGE_STATUS" -eq 1 ]; then return 0; fi
  if [ "$LIBIRREP_STAGE_STATUS" -lt 0 ]; then return 1; fi
  if ! git -C "$LIBIRREP_SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "== required libirrep sibling is not a Git worktree: $LIBIRREP_SOURCE ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi

  LIBIRREP_SOURCE_GIT_HEAD="$(git -C "$LIBIRREP_SOURCE" rev-parse 'HEAD^{commit}')" || {
    LIBIRREP_STAGE_STATUS=-1; return 1;
  }
  LIBIRREP_SOURCE_GIT_TREE="$(git -C "$LIBIRREP_SOURCE" rev-parse 'HEAD^{tree}')" || {
    LIBIRREP_STAGE_STATUS=-1; return 1;
  }
  if ! [[ "$LIBIRREP_SOURCE_GIT_HEAD" =~ ^[0-9a-f]{40,64}$ ]] || \
     ! [[ "$LIBIRREP_SOURCE_GIT_TREE" =~ ^[0-9a-f]{40,64}$ ]]; then
    echo "== libirrep HEAD/tree identity is invalid ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi
  if [ -n "$(git -C "$LIBIRREP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]; then
    LIBIRREP_WORKTREE_DIRTY=true
  else
    LIBIRREP_WORKTREE_DIRTY=false
  fi
  # The staged source is a Git archive of the captured commit, never the
  # sibling's mutable worktree.  Thus libirrep_dirty=false is a property of the
  # actual build input even when libirrep_worktree_dirty truthfully reports
  # unrelated local edits.
  LIBIRREP_SOURCE_DIRTY=false

  local stage_base archive_source archive_marker stage_build stage_log artifact candidate marker_value
  stage_base="$BUILD_DIR/libirrep-stage/$LIBIRREP_SOURCE_GIT_TREE"
  archive_source="$stage_base/source"
  archive_marker="$stage_base/archive.provenance"
  stage_build="$stage_base/build"
  LIBIRREP_ROOT="$stage_base/root"
  stage_log="$stage_base/build.log"
  stage_base="$(require_build_path "$stage_base")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  archive_source="$(require_build_path "$archive_source")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  archive_marker="$(require_build_path "$archive_marker")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  stage_build="$(require_build_path "$stage_build")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  LIBIRREP_ROOT="$(require_build_path "$LIBIRREP_ROOT")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  stage_log="$(require_build_path "$stage_log")" || { LIBIRREP_STAGE_STATUS=-1; return 1; }

  marker_value="$LIBIRREP_SOURCE_GIT_HEAD $LIBIRREP_SOURCE_GIT_TREE"
  if [ -f "$archive_marker" ]; then
    if [ "$(tr -d '\r\n' < "$archive_marker")" != "$marker_value" ]; then
      echo "== existing libirrep archive marker does not match captured HEAD/tree ==" >&2
      LIBIRREP_STAGE_STATUS=-1
      return 1
    fi
  else
    if [ -e "$archive_source" ]; then
      echo "== incomplete libirrep archive stage exists without provenance marker: $archive_source ==" >&2
      LIBIRREP_STAGE_STATUS=-1
      return 1
    fi
    mkdir -p "$archive_source"
    if ! git -C "$LIBIRREP_SOURCE" archive --format=tar "$LIBIRREP_SOURCE_GIT_HEAD" \
         | tar -xf - -C "$archive_source"; then
      echo "== failed to extract clean libirrep Git archive ==" >&2
      LIBIRREP_STAGE_STATUS=-1
      return 1
    fi
    printf '%s\n' "$marker_value" > "$archive_marker"
  fi

  if [ ! -f "$archive_source/CMakeLists.txt" ] || \
     [ ! -f "$archive_source/include/irrep/irrep.h" ]; then
    echo "== committed libirrep archive lacks required build inputs ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi
  LIBIRREP_SOURCE_FINGERPRINT="$(python3 - "$archive_source" <<'PY'
import hashlib
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1]).resolve()
inputs = [root / "CMakeLists.txt", root / "VERSION"]
inputs.extend(sorted((root / "src").glob("*.c")))
inputs.extend(sorted((root / "include").rglob("*.h")))
if not inputs or any(not path.is_file() for path in inputs):
    raise SystemExit("libirrep compile inputs are incomplete")
digest = hashlib.sha256()
for path in sorted(inputs):
    rel = path.relative_to(root).as_posix().encode()
    info = path.lstat()
    payload = path.read_bytes()
    digest.update(len(rel).to_bytes(8, "big"))
    digest.update(rel)
    digest.update(f"{stat.S_IMODE(info.st_mode):04o}".encode())
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
print(digest.hexdigest())
PY
)" || {
    echo "== unable to fingerprint committed libirrep archive ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  }
  if ! [[ "$LIBIRREP_SOURCE_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]]; then
    echo "== invalid libirrep source fingerprint ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi

  mkdir -p "$stage_base" "$LIBIRREP_ROOT/build/lib"
  echo "== staging clean libirrep archive: head=$LIBIRREP_SOURCE_GIT_HEAD tree=$LIBIRREP_SOURCE_GIT_TREE =="
  cmake -S "$archive_source" -B "$stage_build" --fresh \
      -DCMAKE_BUILD_TYPE=Release -DLIBIRREP_BUILD_TESTS=OFF \
      -DLIBIRREP_BUILD_EXAMPLES=OFF -DLIBIRREP_BUILD_BENCHMARKS=OFF \
      >"$stage_log" 2>&1 || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  cmake --build "$stage_build" --clean-first --parallel "$JOBS" --target libirrep \
      >>"$stage_log" 2>&1 || { LIBIRREP_STAGE_STATUS=-1; return 1; }

  artifact=""
  for candidate in "$stage_build/liblibirrep.a" "$stage_build/lib/liblibirrep.a" \
                   "$stage_build/liblibirrep.dylib" "$stage_build/lib/liblibirrep.dylib" \
                   "$stage_build/liblibirrep.so" "$stage_build/lib/liblibirrep.so"; do
    if [ -f "$candidate" ]; then artifact="$candidate"; break; fi
  done
  if [ -z "$artifact" ]; then
    echo "== staged libirrep build produced no supported library artifact ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi
  cmake -E copy_directory "$archive_source/include" "$LIBIRREP_ROOT/include" \
      >>"$stage_log" 2>&1 || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  cmake -E copy "$artifact" "$LIBIRREP_ROOT/build/lib/$(basename "$artifact")" \
      >>"$stage_log" 2>&1 || { LIBIRREP_STAGE_STATUS=-1; return 1; }
  artifact="$LIBIRREP_ROOT/build/lib/$(basename "$artifact")"

  IFS=$'\t' read -r LIBIRREP_ARTIFACT_SHA256 LIBIRREP_BUILD_LOG_SHA256 \
    < <(python3 - "$artifact" "$stage_log" <<'PY'
import hashlib
from pathlib import Path
import sys
print("\t".join(hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sys.argv[1:]))
PY
)
  if ! [[ "$LIBIRREP_ARTIFACT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
     ! [[ "$LIBIRREP_BUILD_LOG_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "== staged libirrep artifact/log hashing failed ==" >&2
    LIBIRREP_STAGE_STATUS=-1
    return 1
  fi
  LIBIRREP_STAGE_STATUS=1
  return 0
}

need_build() {
  # Configure once (cache reused), then ALWAYS run the incremental build so a
  # stale $BUILD_DIR never yields false test failures against pre-fix binaries
  # (the trap that made full_suite_green report 2/137 while a clean build is
  # 179/179). QSIM_SMOKE_BUILD=1 forces CMake's non-destructive fresh
  # reconfigure; the script never recursively removes an override path.
  stage_libirrep || return 1
  local fresh=0 configure=0
  if [ "${QSIM_SMOKE_BUILD:-0}" = "1" ]; then fresh=1; fi
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$fresh" -eq 1 ]; then configure=1; fi
  if [ -f "$BUILD_DIR/CMakeCache.txt" ] && ! grep -q '^QSIM_ENABLE_LIBIRREP:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"; then configure=1; fi
  if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    local cached_libirrep_root
    cached_libirrep_root="$(sed -n 's/^QSIM_LIBIRREP_ROOT:PATH=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
    if [ "$cached_libirrep_root" != "$LIBIRREP_ROOT" ] || ! grep -q '^QSIM_LIBIRREP_LIB:FILEPATH=' "$BUILD_DIR/CMakeCache.txt"; then
      configure=1
      fresh=1
    fi
  fi
  if [ "$configure" -eq 1 ]; then
    echo "== configuring $BUILD_DIR (-j$JOBS) =="
    local -a configure_cmd=(cmake -S . -B "$BUILD_DIR")
    if [ "$fresh" -eq 1 ]; then configure_cmd+=(--fresh); fi
    "${configure_cmd[@]}" -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_BUILD_TESTS=ON -DQSIM_ENABLE_LIBIRREP=ON \
      -DQSIM_LIBIRREP_ROOT="$LIBIRREP_ROOT" >/dev/null 2>&1 || return 1
  fi
  echo "== building $BUILD_DIR (-j$JOBS, incremental) =="
  cmake --build "$BUILD_DIR" --parallel "$JOBS" >/dev/null 2>&1 || return 1
  local libirrep_artifact
  libirrep_artifact="$(sed -n 's/^QSIM_LIBIRREP_LIB:FILEPATH=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
  if [ -z "$libirrep_artifact" ] || [ ! -f "$libirrep_artifact" ]; then
    echo "== enabled libirrep artifact is not recorded in CMakeCache.txt ==" >&2
    return 1
  fi
  local configured_libirrep_hash
  configured_libirrep_hash="$(python3 - "$libirrep_artifact" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)" || return 1
  if [ "$configured_libirrep_hash" != "$LIBIRREP_ARTIFACT_SHA256" ]; then
    echo "== configured libirrep artifact differs from staged attested artifact ==" >&2
    return 1
  fi
  return 0
}

# --- full_suite_green -------------------------------------------------------
check_full_suite() {
  if ! need_build; then emit full_suite_green FAIL "build failed"; return; fi
  local out log="$LOG_DIR/full-suite.log"
  out="$(ctest --test-dir "$BUILD_DIR" -LE "long|memory_heavy" \
                     --timeout 400 -j"$JOBS" --output-on-failure 2>&1)"
  save_log "$log" "$out"
  if ctest_log_green "$log"; then
    emit full_suite_green PASS "complete matrix passed with no skip or Not Run" "$log"
  else
    emit full_suite_green FAIL "$(ctest_log_detail "$log")" "$log"
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
  local out log="$LOG_DIR/asan-ubsan.log"; out="$(ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
      ctest --test-dir "$ASAN_DIR" -L "core|tn|ca_mps|algorithms" \
      -LE "long|memory_heavy" --timeout 600 -j"$JOBS" --output-on-failure 2>&1)"
  save_log "$log" "$out"
  if ctest_log_green "$log"; then
    emit asan_ubsan_clean PASS "core|tn|ca_mps|algorithms clean under ASan+UBSan" "$log"
  else
    emit asan_ubsan_clean FAIL "$(ctest_log_detail "$log")" "$log"
  fi
}

# --- ci_all_green -----------------------------------------------------------
check_ci() {
  if [ "$SOURCE_DIRTY" = "true" ]; then
    emit ci_all_green FAIL "hosted CI cannot attest the current dirty source fingerprint"
    return
  fi
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
  local out log="$LOG_DIR/${name}.log"; out="$(ctest --test-dir "$BUILD_DIR" -R "$rx" --timeout 400 -j"$JOBS" --output-on-failure 2>&1)"
  save_log "$log" "$out"
  if ctest_log_green "$log"; then
    emit "$name" PASS "$listed test(s) matching /$rx/ pass" "$log"
  else
    emit "$name" FAIL "$(ctest_log_detail "$log")" "$log"
  fi
}

# --- mpi_sharded_gpu_works -------------------------------------------------
# The small CTest is the regression guard; it cannot establish the release
# claim by itself.  If registered, it must pass.  PASS also requires a trace
# from run_mpi_sharded_gpu.sh proving the exact N=33, four-rank, two-host
# (2+2 ranks) fleet topology on this commit.  The fleet trace must also prove
# that the same topology passed its N=12 preflight before the exact run.
check_mpi_sharded_gpu() {
  if ! need_build; then emit mpi_sharded_gpu_works FAIL "build failed"; return; fi

  local rx='mpi_sharded_gpu_ghz|sharded_gpu|multigpu|partition_gpu'
  local listed; listed="$(ctest --test-dir "$BUILD_DIR" -N -R "$rx" 2>/dev/null | grep -c 'Test #')"
  local routine_tests_registered=0
  local routine_local_pass=0
  if [ "${listed:-0}" -gt 0 ]; then
    routine_tests_registered=1
    local out; out="$(ctest --test-dir "$BUILD_DIR" -R "$rx" --timeout 400 -j"$JOBS" 2>&1)"
    if printf '%s' "$out" | grep -q '100% tests passed'; then
      routine_local_pass=1
    else
      emit mpi_sharded_gpu_works FAIL "routine MPI+CUDA sharding regression failed"
      return
    fi
  fi

  local fleet_trace="$TRACE_DIR/moonlab_mpi_gpu.jsonl"
  if [ ! -f "$fleet_trace" ]; then
    if [ "$routine_tests_registered" -eq 1 ]; then
      emit mpi_sharded_gpu_works FAIL "routine test passed; exact fleet trace is absent"
    else
      emit mpi_sharded_gpu_works FAIL "routine test absent; exact fleet trace is absent"
    fi
    return
  fi

  local evidence
  if evidence="$(python3 - "$fleet_trace" "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$routine_tests_registered" "$routine_local_pass" <<'PY'
import json
import sys

trace_path, expected_sha, expected_tree, expected_dirty, expected_fingerprint, routine_tests_registered, routine_local_pass = sys.argv[1:]
expected_dirty = expected_dirty == "true"
routine_tests_registered = int(routine_tests_registered)
routine_local_pass = routine_local_pass == "1"
routine_test_passed = False

event = None
with open(trace_path, encoding="utf-8") as trace:
    for line in trace:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            candidate.get("kind") == "moonlab_mpi_gpu"
            and candidate.get("name") == "mpi_sharded_gpu_works"
            and candidate.get("git_tree") == expected_tree
            and candidate.get("source_fingerprint") == expected_fingerprint
            and candidate.get("dirty") is expected_dirty
        ):
            event = candidate

if event is None:
    print("no mpi_sharded_gpu_works event for this exact source fingerprint")
    raise SystemExit(1)
if event.get("value", event.get("status")) != "PASS":
    print("latest fleet event is not PASS")
    raise SystemExit(1)

checks = {
    "git_head_present": isinstance(event.get("git_head"), str) and len(event["git_head"]) >= 40,
    "git_tree_exact": event.get("git_tree") == expected_tree,
    "dirty_exact": event.get("dirty") is expected_dirty,
    "source_fingerprint_exact": event.get("source_fingerprint") == expected_fingerprint,
    "generated_at_present": bool(event.get("generated_at", event.get("ts"))),
    "executable_sha256_present": isinstance(event.get("executable_sha256"), str) and len(event["executable_sha256"]) == 64,
    "exact_n": int(event.get("n", 0)) == 33,
    "exact_ranks": int(event.get("ranks", 0)) == 4,
    "exact_hosts": int(event.get("hosts", 0)) == 2,
    "exact_host_slots": int(event.get("host_slots_2x2", 0)) == 1,
    "exact_gpu_endpoints": int(event.get("gpu_endpoints", 0)) >= 2,
    "exact_halo_swaps": int(event.get("halo_swaps", 0)) >= 1,
    "routine_n": int(event.get("routine_n", 0)) == 12,
    "routine_ranks": int(event.get("routine_ranks", 0)) == 4,
    "routine_hosts": int(event.get("routine_hosts", 0)) == 2,
    "routine_host_slots": int(event.get("routine_host_slots_2x2", 0)) == 1,
    "routine_gpu_endpoints": int(event.get("routine_gpu_endpoints", 0)) >= 2,
    "routine_halo_swaps": int(event.get("routine_halo_swaps", 0)) >= 1,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("fleet event lacks exact evidence: " + ",".join(failed))
    raise SystemExit(1)

if isinstance(event.get("routine_test_passed"), bool):
    routine_test_passed = event.get("routine_test_passed")
elif event.get("routine_test_passed") is not None:
    routine_test_passed = str(event.get("routine_test_passed")).strip().lower() in ("1", "true", "yes", "on")

if not routine_test_passed:
    print("fleet N=12 routine preflight is false/missing")
    raise SystemExit(1)
if routine_tests_registered and not routine_local_pass:
    print("routine local regression test did not pass")
    raise SystemExit(1)

print("N={n} ranks={ranks} hosts={hosts} slots=2+2 GPUs={gpu_endpoints} halo_swaps={halo_swaps} routine_test_passed={routine_pass}".format(
    routine_pass=("true" if routine_test_passed else "false"),
    **event
))
PY
 )"; then
    if [ "$routine_tests_registered" -eq 1 ]; then
      emit mpi_sharded_gpu_works PASS "$listed routine test(s) pass; fleet $evidence"
    else
      emit mpi_sharded_gpu_works PASS "routine test absent locally; fleet $evidence"
    fi
  else
    emit mpi_sharded_gpu_works FAIL "${evidence:-fleet evidence invalid}"
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
  local out log="$LOG_DIR/hidden-visibility.log"; out="$(ctest --test-dir "$HIDDEN_DIR" -L "abi|bindings" --timeout 300 -j"$JOBS" --output-on-failure 2>&1)"
  save_log "$log" "$out"
  if ctest_log_green "$log"; then
    emit hidden_visibility_abi PASS "abi|bindings green under hidden visibility" "$log"
  else
    emit hidden_visibility_abi FAIL "$(ctest_log_detail "$log")" "$log"
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
  local result v reason
  result="$(python3 - "$tf" "$name" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" <<'PY'
import json
import sys

trace_path, name, expected_tree, expected_dirty, expected_fingerprint = sys.argv[1:]
expected_dirty = expected_dirty == "true"
matched = None
with open(trace_path, encoding="utf-8") as trace:
    for line in trace:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("name") == name:
            matched = event
if matched is None:
    print("FAIL\tmissing-event")
elif matched.get("git_tree") != expected_tree:
    print("FAIL\tgit-tree-mismatch")
elif matched.get("dirty") is not expected_dirty:
    print("FAIL\tdirty-state-mismatch")
elif matched.get("source_fingerprint") != expected_fingerprint:
    print("FAIL\tsource-fingerprint-mismatch")
elif not matched.get("generated_at", matched.get("ts", matched.get("timestamp"))):
    print("FAIL\tgenerated-at-missing")
else:
    verdict = matched.get("value", matched.get("status", ""))
    print(f"{verdict}\texact-source")
PY
)"
  IFS=$'\t' read -r v reason <<<"$result"
  if [ "$v" = "PASS" ]; then emit "$name" PASS "$kind lane clean ($reason)"
  else emit "$name" FAIL "$kind lane value=${v:-none} provenance=${reason:-invalid}"; fi
}

check_source_stable() {
  local final_json final_fingerprint
  final_json="$(source_identity)" || { emit source_identity_stable FAIL "unable to recapture source identity"; return; }
  final_fingerprint="$(python3 - "$final_json" 2>/dev/null <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["source_fingerprint"])
PY
)"
  if [ "$final_fingerprint" = "$SOURCE_FINGERPRINT" ]; then
    emit source_identity_stable PASS "source fingerprint remained $SOURCE_FINGERPRINT"
  else
    emit source_identity_stable FAIL "source changed during smoke: start=$SOURCE_FINGERPRINT end=${final_fingerprint:-unknown}"
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
check_source_stable

echo "== $FAILS high-severity check(s) FAILing =="
[ "$FAILS" -eq 0 ]
