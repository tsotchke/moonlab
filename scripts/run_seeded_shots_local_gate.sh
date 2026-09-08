#!/usr/bin/env bash
# Fail-closed local evidence producer for the Moonlab v1.2.1 seeded-SHOTS
# contract.  Release evidence is emitted only from a clean, stable source tree;
# ordinary dirty-worktree development runs should use the focused commands
# listed in the ICC task plan instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${QSIM_SEEDED_SHOTS_BUILD_DIR:-$REPO_ROOT/build-seeded-shots}"
FUZZ_BUILD_DIR="${QSIM_SEEDED_SHOTS_FUZZ_BUILD_DIR:-$REPO_ROOT/build-seeded-shots-fuzz}"
TRACE="$REPO_ROOT/scripts/icc_traces/moonlab_seeded_shots.jsonl"
JOBS="${QSIM_SEEDED_SHOTS_JOBS:-4}"

case "$BUILD_DIR" in
  "$REPO_ROOT"/build-*) ;;
  *) echo "QSIM_SEEDED_SHOTS_BUILD_DIR must be a build-* path under $REPO_ROOT" >&2; exit 2 ;;
esac
case "$FUZZ_BUILD_DIR" in
  "$REPO_ROOT"/build-*) ;;
  *) echo "QSIM_SEEDED_SHOTS_FUZZ_BUILD_DIR must be a build-* path under $REPO_ROOT" >&2; exit 2 ;;
esac
case "$JOBS" in
  1|2|3|4) ;;
  *) echo "QSIM_SEEDED_SHOTS_JOBS must be in [1,4]" >&2; exit 2 ;;
esac

SOURCE_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)"
IFS=$'\t' read -r SOURCE_GIT_HEAD SOURCE_GIT_TREE SOURCE_DIRTY SOURCE_FINGERPRINT \
  < <(python3 - "$SOURCE_IDENTITY_JSON" <<'PY'
import json
import sys
identity = json.loads(sys.argv[1])
print("\t".join((
    identity["git_head"], identity["git_tree"],
    str(identity["dirty"]).lower(), identity["source_fingerprint"],
)))
PY
)

if [ "$SOURCE_DIRTY" != "false" ]; then
  echo "seeded-SHOTS evidence requires a clean worktree; no trace was changed" >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/scripts/icc_traces"
: > "$TRACE"
LANE_STARTED=1
LANE_COMPLETE=0

emit() {
  python3 - "$1" "$2" "$3" "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" \
    "$SOURCE_FINGERPRINT" >> "$TRACE" <<'PY'
import datetime
import json
import sys
name, status, detail, head, tree, fingerprint = sys.argv[1:]
print(json.dumps({
    "kind": "moonlab_seeded_shots",
    "name": name,
    "status": status,
    "value": status,
    "detail": detail,
    "git_head": head,
    "git_tree": tree,
    "dirty": False,
    "source_fingerprint": fingerprint,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, sort_keys=True))
PY
}

on_exit() {
  status=$?
  trap - EXIT
  if [ "$LANE_STARTED" -eq 1 ] && [ "$LANE_COMPLETE" -eq 0 ]; then
    emit local_bit_replay FAIL "local gate aborted before completion (exit $status)"
  fi
  exit "$status"
}
trap on_exit EXIT

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQSIM_BUILD_TESTS=ON \
  -DQSIM_BUILD_EXAMPLES=OFF \
  -DQSIM_BUILD_BENCHMARKS=OFF
cmake --build "$BUILD_DIR" --target test_control_plane_shots -j"$JOBS"
ctest --test-dir "$BUILD_DIR" \
  -R '^integration_control_plane_shots$' --output-on-failure

cmake -S "$REPO_ROOT" -B "$FUZZ_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DQSIM_ENABLE_FUZZING=ON \
  -DQSIM_BUILD_TESTS=OFF \
  -DQSIM_BUILD_EXAMPLES=OFF \
  -DQSIM_BUILD_BENCHMARKS=OFF \
  -DQSIM_WERROR=OFF \
  -DQSIM_ENABLE_OPENMP=OFF \
  -DQSIM_ENABLE_METAL=OFF
cmake --build "$FUZZ_BUILD_DIR" \
  --target control_plane_protocol_fuzz_replay -j"$JOBS"

FUZZ_BIN="$FUZZ_BUILD_DIR/tests/fuzz/control_plane_protocol_fuzz_replay"
SEEDS=()
while IFS= read -r -d '' seed; do SEEDS+=("$seed"); done \
  < <(find "$REPO_ROOT/tests/fuzz/corpora/control_plane_protocol_fuzz" \
      -maxdepth 1 -type f -print0)
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "seeded-SHOTS fuzz corpus is empty" >&2
  exit 2
fi
ASAN_OPTIONS=abort_on_error=1:detect_leaks=0:print_summary=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
DYLD_LIBRARY_PATH="$FUZZ_BUILD_DIR" \
  "$FUZZ_BIN" "${SEEDS[@]}"

SOURCE_END_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)"
SOURCE_END_FINGERPRINT="$(python3 - "$SOURCE_END_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["source_fingerprint"])
PY
)"
if [ "$SOURCE_END_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
  echo "source changed during seeded-SHOTS gate" >&2
  exit 2
fi

emit local_bit_replay PASS \
  "explicit and server-assigned seeds replay byte-identically in the native integration test"
emit protocol_fuzz_clean PASS \
  "control-plane protocol corpus clean under ASan/UBSan, including seeded frames"
LANE_COMPLETE=1
