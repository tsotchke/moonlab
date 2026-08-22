#!/usr/bin/env bash
# Fail-closed two-target seeded-SHOTS replay evidence gate.
#
# The source archive is always git archive HEAD.  Each target gets a fresh,
# never-reused stage path; existing stage paths are an error, not cleanup work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_seeded_shots_mesh_gate.sh [target target]

Runs the seeded_shots_replay_probe on exactly two distinct mesh targets.
Targets default to "atlas old-donkey" and may be supplied as positional
arguments or MOONLAB_SEEDED_SHOTS_TARGETS.  The mesh config supplies target
kinds and staging hooks.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SOURCE_IDENTITY_JSON="$(bash "$ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || {
    echo "unable to read canonical source identity" >&2
    exit 2
}
IFS=$'\t' read -r SOURCE_GIT_HEAD SOURCE_GIT_TREE SOURCE_DIRTY SOURCE_FINGERPRINT \
    < <(python3 - "$SOURCE_IDENTITY_JSON" <<'PY'
import json
import sys
identity = json.loads(sys.argv[1])
print("\t".join((identity["git_head"], identity["git_tree"],
                  str(identity["dirty"]).lower(), identity["source_fingerprint"])))
PY
)
if [[ "$SOURCE_DIRTY" != "false" ]]; then
    echo "seeded-SHOTS mesh evidence requires a clean source tree" >&2
    exit 2
fi

TARGETS_TEXT="${MOONLAB_SEEDED_SHOTS_TARGETS:-atlas old-donkey}"
if [[ $# -gt 0 ]]; then
    TARGETS_TEXT="$*"
fi
read -r -a TARGETS <<<"$TARGETS_TEXT"
if [[ "${#TARGETS[@]}" -ne 2 || "${TARGETS[0]}" == "${TARGETS[1]}" ]]; then
    echo "seeded-SHOTS mesh gate requires exactly two distinct targets" >&2
    exit 2
fi
for target in "${TARGETS[@]}"; do
    if [[ ! "$target" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "invalid mesh target label" >&2
        exit 2
    fi
done

JOBS="${MOONLAB_SEEDED_SHOTS_JOBS:-1}"
case "$JOBS" in 1|2|3|4) ;; *) echo "MOONLAB_SEEDED_SHOTS_JOBS must be in [1,4]" >&2; exit 2 ;; esac
TIMEOUT_SECS="${MOONLAB_SEEDED_SHOTS_TIMEOUT:-900}"
if [[ ! "$TIMEOUT_SECS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MOONLAB_SEEDED_SHOTS_TIMEOUT must be a positive integer" >&2
    exit 2
fi

MESH_ROOT="${MOONLAB_MESH_ROOT:-}"
if [[ -z "$MESH_ROOT" ]]; then
    for candidate in "$HOME/computer_mesh" "$HOME/Desktop/computer_mesh" "$HOME/src/computer_mesh"; do
        if [[ -d "$candidate" ]]; then MESH_ROOT="$candidate"; break; fi
    done
fi
MESH_ROOT="${MESH_ROOT:-$HOME/computer_mesh}"
MESH_BIN="${MOONLAB_MESH_BIN:-$MESH_ROOT/bin/mesh}"
MESH_CONFIG="${MOONLAB_MESH_CONFIG:-$MESH_ROOT/config}"
if [[ ! -f "$MESH_CONFIG" && -f "$MESH_ROOT/moonlab-mesh-smoke.conf" ]]; then
    MESH_CONFIG="$MESH_ROOT/moonlab-mesh-smoke.conf"
fi
if [[ -f "$MESH_CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$MESH_CONFIG"
fi

RUN_ID="${MOONLAB_SEEDED_SHOTS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_FINGERPRINT:0:12}}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid seeded-SHOTS run id" >&2
    exit 2
fi
ARTIFACT_PARENT="$ROOT/build/seeded-shots-mesh"
ARTIFACT_DIR="$ARTIFACT_PARENT/$RUN_ID"
mkdir -p "$ARTIFACT_PARENT"
if ! mkdir "$ARTIFACT_DIR" 2>/dev/null; then
    echo "seeded-SHOTS run path already exists: refusing reuse" >&2
    exit 2
fi
mkdir "$ARTIFACT_DIR/logs"

TRACE="$ROOT/scripts/icc_traces/moonlab_seeded_shots.jsonl"
mkdir -p "$ROOT/scripts/icc_traces"
SOURCE_ARCHIVE="$ARTIFACT_DIR/source-head.tar"
short_head="${SOURCE_GIT_HEAD:0:12}"
SOURCE_FINGERPRINT_SHORT="${SOURCE_FINGERPRINT:0:12}"
STARTED=1
EVENT_EMITTED=0

emit_event() {
    local value="$1" detail="$2" include_targets="${3:-0}"
    local event
    event="$(python3 - "$value" "$detail" "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" \
        "$SOURCE_FINGERPRINT" "$include_targets" "${TARGETS[0]}" "${TARGETS[1]}" <<'PY'
import datetime as dt
import json
import sys
value, detail, head, tree, fingerprint, include_targets, first, second = sys.argv[1:]
event = {
    "kind": "moonlab_seeded_shots",
    "name": "cross_host_bit_replay",
    "value": value,
    "status": value,
    "detail": detail,
    "git_head": head,
    "git_tree": tree,
    "dirty": False,
    "source_fingerprint": fingerprint,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
}
if include_targets == "1":
    event["targets"] = [first, second]
print(json.dumps(event, sort_keys=True, separators=(",", ":")))
PY
    )"
    printf '%s\n' "$event" >>"$TRACE"
    printf '%s\n' "$event"
    EVENT_EMITTED=1
}

# shellcheck disable=SC2329  # invoked indirectly by the EXIT trap
on_exit() {
    local rc=$?
    trap - EXIT
    if [[ "$STARTED" -eq 1 && "$EVENT_EMITTED" -eq 0 ]]; then
        emit_event FAIL "seeded-SHOTS mesh gate failed after start" 0 >/dev/null || true
    fi
    exit "$rc"
}
trap on_exit EXIT

if [[ ! -x "$MESH_BIN" && "$(declare -F mesh_target_kind >/dev/null; echo $?)" -ne 0 ]]; then
    echo "mesh executable not found" >&2
    exit 2
fi

git archive --format=tar --output="$SOURCE_ARCHIVE" HEAD

quote_sh() {
    local value="$1"
    value=${value//\'/\'\\\'\'}
    printf "'%s'" "$value"
}

run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM "$TIMEOUT_SECS" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout --signal=TERM "$TIMEOUT_SECS" "$@"
    else
        python3 - "$TIMEOUT_SECS" "$@" <<'PY'
import subprocess
import sys
try:
    raise SystemExit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    raise SystemExit(124)
PY
    fi
}

run_logged() {
    local log="$1"
    shift
    if ! run_bounded "$@" >"$log" 2>&1; then
        echo "seeded-SHOTS mesh operation failed; see artifact log" >&2
        return 1
    fi
}

run_probe_logged() {
    local log="$1"
    local output="$2"
    shift 2
    if ! run_bounded "$@" >"$output" 2>"$log"; then
        echo "seeded-SHOTS replay probe failed; see artifact log" >&2
        return 1
    fi
}

target_kind() {
    if declare -F mesh_target_kind >/dev/null; then
        mesh_target_kind "$1"
    else
        printf '%s\n' posix
    fi
}

stage_base() {
    if declare -F mesh_target_stage_dir >/dev/null; then
        mesh_target_stage_dir "$1"
    else
        printf '%s\n' "/tmp/moonlab-seeded-shots-$short_head"
    fi
}

cmake_flags() {
    if declare -F mesh_target_cmake_flags >/dev/null; then
        mesh_target_cmake_flags "$1"
    else
        printf '%s\n' ""
    fi
}

write_remote_runner() {
    local target="$1" stage="$2" archive="$3" output="$4" flags="$5" out="$6"
    local q_stage q_archive q_output
    printf -v q_stage '%q' "$stage"
    printf -v q_archive '%q' "$archive"
    printf -v q_output '%q' "$output"
    cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
stage=$q_stage
archive=$q_archive
output=$q_output
cmake -S "\$stage" -B "\$stage/build" -DCMAKE_BUILD_TYPE=Release \\
  -DQSIM_BUILD_TESTS=ON -DQSIM_ENABLE_CONTROL_PLANE=ON \\
  -DQSIM_BUILD_EXAMPLES=OFF -DQSIM_BUILD_BENCHMARKS=OFF $flags
cmake --build "\$stage/build" --target seeded_shots_replay_probe --parallel $JOBS
"\$stage/build/seeded_shots_replay_probe" >"\$output"
EOF
    chmod +x "$out"
    printf '%s\n' "$target" >/dev/null
}

run_target() {
    local target="$1" kind base stage archive remote_script remote_output flags_text
    local -a local_flags=()
    local local_output="$ARTIFACT_DIR/out-$target.txt"
    kind="$(target_kind "$target")"
    if [[ "$kind" == "local" ]]; then
        stage="$ARTIFACT_DIR/stage-$target"
        if ! mkdir "$stage"; then
            echo "local seeded-SHOTS stage already exists" >&2
            return 1
        fi
        tar -xf "$SOURCE_ARCHIVE" -C "$stage"
        flags_text="$(cmake_flags "$target")"
        if [[ -n "$flags_text" ]]; then
            read -r -a local_flags <<<"$flags_text"
        fi
        run_logged "$ARTIFACT_DIR/logs/$target.log" \
            cmake -S "$stage" -B "$stage/build" -DCMAKE_BUILD_TYPE=Release \
            -DQSIM_BUILD_TESTS=ON -DQSIM_ENABLE_CONTROL_PLANE=ON \
            -DQSIM_BUILD_EXAMPLES=OFF -DQSIM_BUILD_BENCHMARKS=OFF \
            "${local_flags[@]}" || return 1
        run_logged "$ARTIFACT_DIR/logs/$target-build.log" \
            cmake --build "$stage/build" --target seeded_shots_replay_probe \
            --parallel "$JOBS" || return 1
        run_probe_logged "$ARTIFACT_DIR/logs/$target-probe.log" "$local_output" \
            "$stage/build/seeded_shots_replay_probe" || return 1
    else
        [[ -x "$MESH_BIN" ]] || { echo "mesh executable not found" >&2; return 1; }
        base="$(stage_base "$target")"
        [[ -n "$base" && "$base" != *$'\n'* ]] || { echo "mesh stage hook returned no path" >&2; return 1; }
        stage="${base%/}/moonlab-seeded-shots-${RUN_ID}-${SOURCE_FINGERPRINT_SHORT}-${target}"
        archive="/tmp/moonlab-seeded-shots-${RUN_ID}-${SOURCE_FINGERPRINT_SHORT}-${target}.tar"
        remote_script="/tmp/moonlab-seeded-shots-${RUN_ID}-${SOURCE_FINGERPRINT_SHORT}-${target}.sh"
        remote_output="${stage}/probe.out"
        remote_check="if [ -e $(quote_sh "$stage") ] || [ -e $(quote_sh "$archive") ] || [ -e $(quote_sh "$remote_script") ]; then exit 73; fi"
        run_logged "$ARTIFACT_DIR/logs/$target-preflight.log" \
            "$MESH_BIN" exec "$target" "$remote_check" || return 1
        run_logged "$ARTIFACT_DIR/logs/$target-upload.log" \
            scp -q -o BatchMode=yes -o ConnectTimeout=20 \
            "$SOURCE_ARCHIVE" "$target:$archive" || return 1
        flags_text="$(cmake_flags "$target")"
        write_remote_runner "$target" "$stage" "$archive" "$remote_output" "$flags_text" \
            "$ARTIFACT_DIR/$target-runner.sh"
        run_logged "$ARTIFACT_DIR/logs/$target-script-upload.log" \
            scp -q -o BatchMode=yes -o ConnectTimeout=20 \
            "$ARTIFACT_DIR/$target-runner.sh" "$target:$remote_script" || return 1
        run_logged "$ARTIFACT_DIR/logs/$target-run.log" \
            "$MESH_BIN" exec "$target" \
            "mkdir -p $(quote_sh "$base") && mkdir $(quote_sh "$stage") && tar -xf $(quote_sh "$archive") -C $(quote_sh "$stage") && bash $(quote_sh "$remote_script")" || return 1
        run_logged "$ARTIFACT_DIR/logs/$target-download.log" \
            scp -q -o BatchMode=yes -o ConnectTimeout=20 \
            "$target:$remote_output" "$local_output" || return 1
    fi
    [[ "$(wc -l <"$local_output" | tr -d ' ')" == 1 ]] || {
        echo "seeded-SHOTS probe did not emit exactly one line" >&2
        return 1
    }
    return 0
}

for i in 0 1; do
    run_target "${TARGETS[$i]}" || exit 1
done

if ! cmp -s "$ARTIFACT_DIR/out-${TARGETS[0]}.txt" "$ARTIFACT_DIR/out-${TARGETS[1]}.txt"; then
    emit_event FAIL "target probe outputs differ" 0 >/dev/null
    exit 1
fi

SOURCE_END_JSON="$(bash "$ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || exit 1
SOURCE_END_FINGERPRINT="$(python3 - "$SOURCE_END_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["source_fingerprint"])
PY
)"
if [[ "$SOURCE_END_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]]; then
    emit_event FAIL "source changed during seeded-SHOTS mesh gate" 0 >/dev/null
    exit 1
fi

emit_event PASS "two target probes returned byte-identical ordered outcomes" 1
STARTED=0
exit 0
