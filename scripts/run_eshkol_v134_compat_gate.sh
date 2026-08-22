#!/usr/bin/env bash
# Build the published Eshkol v1.3.4 quantum consumer against this exact
# Moonlab source tree and run the seven published quantum probes. A PASS is
# written only after every command and every explicit PASS marker succeeds.
# The Eshkol worktree is never configured or written: the tag is archived into
# this invocation's ignored Moonlab build directory first.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT" || exit 2

ESH_TAG="v1.3.4-evolve"
EXPECTED_ESH_COMMIT="694c31798f3f89d55015492bffd027c01951f7bf"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE="$TRACE_DIR/moonlab_eshkol_compatibility.jsonl"
SOURCE_IDENTITY_SCRIPT="$REPO_ROOT/scripts/moonlab_source_identity.py"
ESH_REPO="${ESHKOL_REPO:-}"
if [ -z "$ESH_REPO" ]; then
  if [ -e "$HOME/Desktop/eshkol/.git" ]; then ESH_REPO="$HOME/Desktop/eshkol"
  elif [ -e "$HOME/eshkol/.git" ]; then ESH_REPO="$HOME/eshkol"
  else ESH_REPO="$HOME/Desktop/eshkol"
  fi
fi
BUILD_BASE="${MOONLAB_ESHKOL_COMPAT_BUILD_DIR:-${BUILD_DIR:-$REPO_ROOT/build-eshkol-compat}}"
JOBS="${MOONLAB_ESHKOL_COMPAT_JOBS:-${ESHKOL_JOBS:-2}}"
RUN_DIR=""
LOG_DIR=""
SOURCE_IDENTITY_JSON=""
SOURCE_BINDING=""
ESH_COMMIT=""
FAIL_EMITTED=0
EVIDENCE_STARTED=0
LAST_FAILURE="compatibility gate did not complete"

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

valid_build_base() {
  local resolved
  resolved="$(resolve_path "$1")" || return 1
  case "$resolved" in
    "$REPO_ROOT/build"|"$REPO_ROOT"/build-*|"$REPO_ROOT"/build_*) printf '%s\n' "$resolved";;
    *) printf 'compatibility build path must resolve under %s/build, build-*, or build_*: %s\n' "$REPO_ROOT" "$resolved" >&2; return 1;;
  esac
}

capture_source_identity() {
  SOURCE_IDENTITY_JSON="$1/source-identity.json"
  python3 "$SOURCE_IDENTITY_SCRIPT" --repo-root "$REPO_ROOT" >"$SOURCE_IDENTITY_JSON"
  SOURCE_BINDING="$(python3 - "$SOURCE_IDENTITY_JSON" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = ("git_head", "git_tree", "dirty", "source_fingerprint")
if any(key not in data for key in required):
    raise SystemExit("source identity is missing a required field")
if data["dirty"] is not False:
    raise SystemExit("Moonlab source identity is dirty")
for key in ("git_head", "git_tree"):
    value = str(data[key])
    if not (40 <= len(value) <= 64 and all(c in "0123456789abcdef" for c in value)):
        raise SystemExit(f"invalid source identity {key}")
fingerprint = str(data["source_fingerprint"])
if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
    raise SystemExit("invalid source identity fingerprint")
print("\t".join(str(data[key]) for key in required))
PY
)"
}

source_is_clean() {
  [ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all -- . ':(exclude)scripts/icc_traces/**')" ]
}

emit_event() {
  local status="$1" detail="$2"
  [ -n "$RUN_DIR" ] || RUN_DIR="$REPO_ROOT/build-eshkol-compat/preflight"
  mkdir -p "$TRACE_DIR" "$RUN_DIR" 2>/dev/null || return 1
  python3 - "$TRACE" "$status" "$detail" "$SOURCE_BINDING" "$ESH_TAG" "$ESH_COMMIT" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
trace, status, detail, binding, tag, commit = sys.argv[1:]
parts = binding.split("\t") if binding else ["", "", "", ""]
event = {
    "kind": "moonlab_eshkol_compatibility",
    "name": "eshkol_v134_quantum_consumer",
    "status": status,
    "value": status,
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "detail": detail,
    "git_head": parts[0],
    "git_tree": parts[1],
    "dirty": parts[2].lower() == "true",
    "source_fingerprint": parts[3],
    "moonlab_git_head": parts[0],
    "moonlab_git_tree": parts[1],
    "moonlab_dirty": parts[2].lower() == "true",
    "moonlab_source_fingerprint": parts[3],
    "eshkol_tag": tag,
    "eshkol_commit": commit,
}
path = Path(trace)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as output:
    output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

fail() {
  LAST_FAILURE="$1"
  if [ "$EVIDENCE_STARTED" -eq 1 ] && [ "$FAIL_EMITTED" -eq 0 ]; then
    FAIL_EMITTED=1
    emit_event FAIL "$LAST_FAILURE" || true
  fi
  printf 'eshkol v1.3.4 compatibility gate: FAIL: %s\n' "$LAST_FAILURE" >&2
  exit 1
}

# EXIT invokes this indirectly through trap; keep the shellcheck contract clear.
# shellcheck disable=SC2329
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$EVIDENCE_STARTED" -eq 1 ] && [ "$FAIL_EMITTED" -eq 0 ]; then
    FAIL_EMITTED=1
    emit_event FAIL "$LAST_FAILURE (exit $rc)" || true
  fi
  return "$rc"
}
trap on_exit EXIT

[ -f "$SOURCE_IDENTITY_SCRIPT" ] || fail "missing Moonlab source identity helper"
source_is_clean || fail "Moonlab source worktree is not clean"
BUILD_BASE="$(valid_build_base "$BUILD_BASE")" || fail "invalid compatibility build directory"
case "$JOBS" in ''|*[!0-9]*) fail "jobs must be an integer in the range 1..4";; esac
if [ "$JOBS" -lt 1 ]; then JOBS=1; elif [ "$JOBS" -gt 4 ]; then JOBS=4; fi

# This unique run directory and its logs are ignored by Moonlab's build rule.
RUN_DIR="$BUILD_BASE/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR" || fail "unable to create run directory"
capture_source_identity "$RUN_DIR" || fail "unable to capture clean Moonlab source identity"
: > "$TRACE" || fail "unable to reset compatibility evidence"
EVIDENCE_STARTED=1

git -C "$ESH_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "ESHKOL_REPO is not a Git worktree: $ESH_REPO"
git -C "$ESH_REPO" show-ref --verify --quiet "refs/tags/$ESH_TAG" || fail "$ESH_TAG is not a published tag ref in $ESH_REPO"
ESH_COMMIT="$(git -C "$ESH_REPO" rev-parse --verify "refs/tags/$ESH_TAG^{commit}" 2>/dev/null)" || fail "unable to resolve $ESH_TAG to a commit"
[ "$ESH_COMMIT" = "$EXPECTED_ESH_COMMIT" ] || fail "$ESH_TAG resolves to $ESH_COMMIT, expected $EXPECTED_ESH_COMMIT"

ESH_SOURCE="$RUN_DIR/eshkol-source"
ESH_BUILD="$RUN_DIR/eshkol-build"
mkdir -p "$ESH_SOURCE" "$ESH_BUILD" || fail "unable to create archived Eshkol directories"
reject_error_output() {
  local log="$1" label="$2"
  grep -q 'ERROR' "$log" && fail "$label emitted ERROR output; log=$log"
}
# git archive reads the pinned commit object but never checks out or writes the
# user's (possibly dirty) Eshkol worktree.
git -C "$ESH_REPO" archive --format=tar "$ESH_COMMIT" | tar -xf - -C "$ESH_SOURCE" >"$LOG_DIR/archive.log" 2>&1 || fail "unable to archive $ESH_TAG at $ESH_COMMIT"
[ ! -s "$LOG_DIR/archive.log" ] || reject_error_output "$LOG_DIR/archive.log" archive
[ -f "$ESH_SOURCE/CMakeLists.txt" ] || fail "archived Eshkol source is incomplete"

run_logged() {
  local log="$1"
  shift
  "$@" >"$log" 2>&1 || {
    local rc=$?
    LAST_FAILURE="command failed ($rc): $*; log=$log"
    return "$rc"
  }
}

run_logged "$LOG_DIR/configure.log" cmake -S "$ESH_SOURCE" -B "$ESH_BUILD" \
  -DESHKOL_QUANTUM_ENABLED=ON \
  -DFETCHCONTENT_SOURCE_DIR_MOONLAB="$REPO_ROOT" \
  -DESHKOL_GPU_ENABLED=OFF \
  -DQSIM_ENABLE_METAL=OFF \
  -DQSIM_ENABLE_CUDA=OFF \
  -DQSIM_ENABLE_CUQUANTUM=OFF \
  -DQSIM_ENABLE_OPENCL=OFF \
  -DQSIM_ENABLE_VULKAN=OFF \
  -DQSIM_ENABLE_WEBGPU=OFF \
  -DCMAKE_BUILD_TYPE=Release || fail "$LAST_FAILURE"
reject_error_output "$LOG_DIR/configure.log" configure
run_logged "$LOG_DIR/build.log" cmake --build "$ESH_BUILD" --target eshkol-run stdlib --parallel "$JOBS" || fail "$LAST_FAILURE"
reject_error_output "$LOG_DIR/build.log" build

ESH_RUN="$ESH_BUILD/eshkol-run"
[ -x "$ESH_RUN" ] || fail "built eshkol-run is missing: $ESH_RUN"
export ESHKOL_LIB_DIR="$ESH_BUILD"
export ESHKOL_PATH="$ESH_SOURCE/lib"
export ESHKOL_JIT_CACHE_DIR="$RUN_DIR/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR" || fail "unable to create JIT cache"
export DYLD_LIBRARY_PATH="$ESH_BUILD:$ESH_BUILD/_deps/moonlab-build${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ESH_BUILD:$ESH_BUILD/_deps/moonlab-build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TESTS=(quantum_smoke_test bell_chsh_test quantum_surface_coverage_test vqe_test vqe_ad_test vqe_ad_adversarial pqc_mlkem_test)
for test_name in "${TESTS[@]}"; do
  test_source="$ESH_SOURCE/tests/quantum/$test_name.esk"
  test_log="$LOG_DIR/$test_name.log"
  [ -f "$test_source" ] || fail "published quantum test is missing: $test_name"
  run_logged "$test_log" "$ESH_RUN" -r "$test_source" || fail "$LAST_FAILURE"
  grep -Eq '(^|[^[:alpha:]])PASS([^[:alpha:]]|$)' "$test_log" || fail "$test_name did not emit an explicit PASS marker; log=$test_log"
  reject_error_output "$test_log" "$test_name"
  grep -q 'FAIL' "$test_log" && fail "$test_name emitted FAIL output; log=$test_log"
done

FINAL_IDENTITY_JSON="$RUN_DIR/source-identity-final.json"
python3 "$SOURCE_IDENTITY_SCRIPT" --repo-root "$REPO_ROOT" >"$FINAL_IDENTITY_JSON" || fail "unable to recapture Moonlab source identity"
FINAL_BINDING="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("dirty") is not False:
    raise SystemExit("Moonlab source became dirty")
print("\t".join(str(data.get(key, "")) for key in ("git_head", "git_tree", "dirty", "source_fingerprint")))
PY
)" || fail "Moonlab source identity became dirty"
[ "$FINAL_BINDING" = "$SOURCE_BINDING" ] || fail "Moonlab source identity changed during compatibility build"

emit_event PASS "all seven published Eshkol quantum consumer tests passed" || fail "unable to emit PASS evidence"
FAIL_EMITTED=1
printf 'eshkol v1.3.4 compatibility gate: PASS (trace: %s)\n' "$TRACE"
exit 0
