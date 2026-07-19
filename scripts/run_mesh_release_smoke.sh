#!/usr/bin/env bash
# Run a bounded Moonlab release smoke across a computer mesh.
#
# The script stages the current working tree, not just HEAD, so it can validate
# pre-commit release candidates. Logs are written under build/mesh-release-smoke.
#
# All node-specific facts (target names, stage directories, CMake flags,
# evidence labels) live in an external mesh config sourced at startup, so this
# script carries no fleet topology. Point MOONLAB_MESH_CONFIG at your config,
# or place moonlab-mesh-smoke.conf in the mesh checkout root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MOONLAB_MESH_ROOT:-}" ]]; then
    for candidate in "$HOME/computer_mesh" "$HOME/Desktop/computer_mesh" "$HOME/src/computer_mesh"; do
        if [[ -d "$candidate" ]]; then
            MOONLAB_MESH_ROOT="$candidate"
            break
        fi
    done
fi
MESH_ROOT="${MOONLAB_MESH_ROOT:-$HOME/computer_mesh}"
MESH_BIN="${MOONLAB_MESH_BIN:-$MESH_ROOT/bin/mesh}"
MESH_CONFIG="${MOONLAB_MESH_CONFIG:-$MESH_ROOT/moonlab-mesh-smoke.conf}"

# The config may define MOONLAB_MESH_TARGETS plus any of the mesh_target_*
# override functions documented in usage().
if [[ -f "$MESH_CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$MESH_CONFIG"
fi

TARGETS="${MOONLAB_MESH_TARGETS:-}"
EXPECTED_TARGETS="${MOONLAB_MESH_EXPECTED_TARGETS:-5}"
JOBS="${MOONLAB_MESH_JOBS:-4}"
LABELS="${MOONLAB_MESH_LABELS:-core|abi|gpu}"
EXCLUDE_LABELS="${MOONLAB_MESH_EXCLUDE_LABELS:-long|memory_heavy}"
EXCLUDE_TESTS="${MOONLAB_MESH_EXCLUDE_TESTS:-n34_to_n40|hofstadter_sweep|kagome_ed_n18}"
REMOTE_BASE="${MOONLAB_MESH_REMOTE_BASE:-moonlab-mesh-smoke}"
RUN_ID="${MOONLAB_MESH_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_DIR="${MOONLAB_MESH_ARTIFACT_DIR:-$ROOT/build/mesh-release-smoke/$RUN_ID}"
CONNECT_TIMEOUT="${MOONLAB_MESH_CONNECT_TIMEOUT:-20}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT" -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

usage() {
    cat <<EOF
Usage: $0 [target...]

Runs a bounded release smoke on the mesh targets. Targets come from the
command line, MOONLAB_MESH_TARGETS, or the mesh config file.

Environment:
  MOONLAB_MESH_CONFIG           Mesh config file. Default: \$MESH_ROOT/moonlab-mesh-smoke.conf
  MOONLAB_MESH_TARGETS          Space-separated target list.
  MOONLAB_MESH_EXPECTED_TARGETS Exact distinct target count. Default: $EXPECTED_TARGETS
  MOONLAB_MESH_ROOT             Mesh checkout. Default: \$HOME/computer_mesh
  MOONLAB_MESH_JOBS             Build/test parallelism. Default: $JOBS
  MOONLAB_MESH_LABELS           CTest -L regex. Default: $LABELS
  MOONLAB_MESH_EXCLUDE_LABELS   CTest -LE regex. Default: $EXCLUDE_LABELS
  MOONLAB_MESH_EXCLUDE_TESTS    CTest -E regex. Default: $EXCLUDE_TESTS
  MOONLAB_MESH_ARTIFACT_DIR     Local log/snapshot directory.
  MOONLAB_MESH_REMOTE_BASE      Remote staging directory name.
  MOONLAB_MESH_CONNECT_TIMEOUT  SSH/SCP connect timeout. Default: $CONNECT_TIMEOUT

Config file hooks (bash functions, all optional, called with the target name):
  mesh_target_kind          Echo local | posix | windows. Default: posix.
  mesh_target_stage_dir     Echo the staging directory for a posix target.
                            Default: /tmp/\$REMOTE_BASE-<head>.
  mesh_target_cmake_flags   Echo extra CMake flags (e.g. -DQSIM_ENABLE_CUDA=ON
                            -DCMAKE_CUDA_ARCHITECTURES=72). Default: none.
  mesh_target_evidence_label Echo a human label for the runtime-evidence file.
                            Default: the target name.

Notes:
  - A "local" target runs on this machine from the same staged snapshot.
  - Posix targets fall back to a nix flake dev shell (.#cuda) when cmake is
    not on PATH and nix is available.
  - Windows targets stage under \$env:USERPROFILE\\src and use Ninja.
  - icc_runtime_evidence.txt is emitted for readiness --trace-file.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    TARGETS="$*"
fi

if [[ -z "$TARGETS" ]]; then
    echo "no targets: pass targets or set MOONLAB_MESH_TARGETS (see --help)" >&2
    exit 2
fi

read -r -a target_array <<<"$TARGETS"
unique_target_count="$(printf '%s\n' "${target_array[@]}" | sort -u | wc -l | tr -d ' ')"
if [[ ! "$EXPECTED_TARGETS" =~ ^[1-9][0-9]*$ ]] || \
   [[ "${#target_array[@]}" -ne "$EXPECTED_TARGETS" ]] || \
   [[ "$unique_target_count" -ne "$EXPECTED_TARGETS" ]]; then
    echo "release mesh requires exactly $EXPECTED_TARGETS distinct targets (got ${#target_array[@]} entries, $unique_target_count distinct)" >&2
    exit 2
fi

mkdir -p "$ARTIFACT_DIR/logs" "$ARTIFACT_DIR/scripts"

if [[ ! -x "$MESH_BIN" ]]; then
    echo "mesh binary not found: $MESH_BIN" >&2
    exit 2
fi

git_head="$(git -C "$ROOT" rev-parse HEAD)"
short_head="${git_head:0:12}"
SOURCE_IDENTITY_JSON="$(bash "$ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)"
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
snapshot_id="$short_head-${SOURCE_FINGERPRINT:0:12}"
source_tar="$ARTIFACT_DIR/moonlab-source-$snapshot_id.tar"
# Keep existing external config hooks that key stage directories from
# $short_head collision-safe for distinct dirty snapshots of the same commit.
short_head="$snapshot_id"

create_source_snapshot() {
    echo "[mesh-smoke] snapshot: $source_tar"
    python3 - "$ROOT" "$source_tar" <<'PY'
import os
import pathlib
import subprocess
import sys
import tarfile

root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
proc = subprocess.run(
    [
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", ".", ":(exclude)scripts/icc_traces/**",
    ],
    cwd=root,
    check=True,
    stdout=subprocess.PIPE,
)
paths = [p.decode() for p in proc.stdout.split(b"\0") if p]
with tarfile.open(out, "w") as tar:
    for rel in paths:
        path = root / rel
        if not path.exists() or path.is_dir():
            continue
        tar.add(path, arcname=rel, recursive=False)
PY
}

run_and_log() {
    local label="$1"
    local logfile="$2"
    shift 2
    echo "[mesh-smoke] $label"
    set +e
    "$@" >"$logfile" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "[mesh-smoke] $label failed; log: $logfile" >&2
        tail -80 "$logfile" >&2 || true
        return "$rc"
    fi
    return 0
}

target_kind() {
    local target="$1"
    if declare -F mesh_target_kind >/dev/null; then
        mesh_target_kind "$target"
    else
        printf '%s\n' "posix"
    fi
}

posix_stage_dir() {
    local target="$1"
    if declare -F mesh_target_stage_dir >/dev/null; then
        local dir
        dir="$(mesh_target_stage_dir "$target")"
        if [[ -n "$dir" ]]; then
            printf '%s\n' "$dir"
            return
        fi
    fi
    if [[ "$(target_kind "$target")" == "local" ]]; then
        printf '%s\n' "$ARTIFACT_DIR/$target-src"
    else
        printf '%s\n' "/tmp/$REMOTE_BASE-$short_head"
    fi
}

target_cmake_flags() {
    local target="$1"
    if declare -F mesh_target_cmake_flags >/dev/null; then
        mesh_target_cmake_flags "$target"
    else
        printf '%s\n' ""
    fi
}

write_posix_smoke_script() {
    local target="$1"
    local repo_dir="$2"
    local flags="$3"
    local out="$4"
    cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail

repo_dir='$repo_dir'
build_dir="\$repo_dir/build-mesh-release-smoke"
jobs='$JOBS'
labels='$LABELS'
exclude_labels='$EXCLUDE_LABELS'
exclude_tests='$EXCLUDE_TESTS'
target='$target'

run_body() {
    cmake -S "\$repo_dir" -B "\$build_dir" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DQSIM_WERROR=ON \\
        -DQSIM_NATIVE_ARCH=OFF \\
        -DQSIM_FAST_MATH=OFF \\
        -DQSIM_BUILD_BENCHMARKS=OFF \\
        -DQSIM_BUILD_EXAMPLES=OFF \\
        $flags
    cmake --build "\$build_dir" --parallel "\$jobs"
    ctest --test-dir "\$build_dir" --output-on-failure --timeout 600 \\
        -LE "\$exclude_labels" \\
        -E "\$exclude_tests" \\
        -L "\$labels" \\
        -j "\$jobs"
}

# Hosts without cmake on PATH (e.g. NixOS Jetson nodes) fall back to the
# repo's flake dev shell when nix is available.
if [[ "\${1:-}" != "--inside-dev-shell" ]]; then
    nix_bin="\$(command -v nix || true)"
    if [[ -z "\$nix_bin" && -x /run/current-system/sw/bin/nix ]]; then
        nix_bin=/run/current-system/sw/bin/nix
    fi
    if ! command -v cmake >/dev/null 2>&1 && [[ -n "\$nix_bin" ]]; then
        exec "\$nix_bin" --extra-experimental-features 'nix-command flakes' \\
            develop "\$repo_dir#cuda" --command bash "\$0" --inside-dev-shell
    fi
fi

echo "target=\$target"
uname -srmo || true
cmake --version | head -1
ctest --version | head -1
run_body
EOF
    chmod +x "$out"
}

run_local_target() {
    local target="$1"
    local repo_dir
    repo_dir="$(posix_stage_dir "$target")"
    rm -rf "$repo_dir"
    mkdir -p "$repo_dir"
    tar -xf "$source_tar" -C "$repo_dir"

    local script="$ARTIFACT_DIR/scripts/$target-smoke.sh"
    write_posix_smoke_script "$target" "$repo_dir" "$(target_cmake_flags "$target")" "$script"
    run_and_log "$target smoke" "$ARTIFACT_DIR/logs/$target.smoke.log" bash "$script"
}

run_posix_target() {
    local target="$1"
    local repo_dir remote_tar remote_script local_script
    repo_dir="$(posix_stage_dir "$target")"
    remote_tar="/tmp/moonlab-source-$snapshot_id.tar"
    remote_script="/tmp/moonlab-mesh-smoke-$snapshot_id.sh"
    local_script="$ARTIFACT_DIR/scripts/$target-smoke.sh"

    write_posix_smoke_script "$target" "$repo_dir" "$(target_cmake_flags "$target")" "$local_script"

    run_and_log "$target upload source" "$ARTIFACT_DIR/logs/$target.upload-source.log" \
        scp "${SSH_OPTS[@]}" -q "$source_tar" "$target:$remote_tar"
    run_and_log "$target upload script" "$ARTIFACT_DIR/logs/$target.upload-script.log" \
        scp "${SSH_OPTS[@]}" -q "$local_script" "$target:$remote_script"
    run_and_log "$target stage" "$ARTIFACT_DIR/logs/$target.stage.log" \
        "$MESH_BIN" exec "$target" "rm -rf '$repo_dir' && mkdir -p '$repo_dir' && tar -xf '$remote_tar' -C '$repo_dir'"
    run_and_log "$target smoke" "$ARTIFACT_DIR/logs/$target.smoke.log" \
        "$MESH_BIN" exec "$target" "bash '$remote_script'"
}

write_windows_smoke_script() {
    local out="$1"
    cat >"$out" <<EOF
\$ErrorActionPreference = 'Stop'
\$Repo = Join-Path \$env:USERPROFILE 'src\\moonlab-mesh-smoke-$snapshot_id'
\$Tar = Join-Path \$PSScriptRoot 'moonlab-source-$snapshot_id.tar'
\$Build = Join-Path \$Repo 'build-mesh-release-smoke'
\$SharedBuild = Join-Path \$Repo 'build-mesh-release-smoke-shared'
\$Jobs = '$JOBS'
\$Labels = '$LABELS'
\$ExcludeLabels = '$EXCLUDE_LABELS'
\$ExcludeTests = '$EXCLUDE_TESTS'

function Run-Native {
    param(
        [Parameter(Mandatory=\$true)]
        [string]\$Exe,
        [Parameter(Mandatory=\$true)]
        [string[]]\$NativeArgs
    )
    & \$Exe @NativeArgs
    if (\$LASTEXITCODE -ne 0) {
        throw "\$Exe failed with exit code \$LASTEXITCODE"
    }
}

Write-Host "target=windows"
Remove-Item -LiteralPath \$Repo -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path \$Repo | Out-Null
Run-Native 'tar' @('-xf', \$Tar, '-C', \$Repo)

# Toolchain discovery: prefer an existing cmake/ninja on PATH, then the
# conventional MSYS2 ucrt64 + CMake install locations.
\$Msys = Join-Path \$env:USERPROFILE 'src\\msys64\\ucrt64\\bin'
\$CMake = 'C:\\Program Files\\CMake\\bin'
if (Test-Path \$Msys) { \$env:PATH = "\$Msys;\$env:PATH" }
if (Test-Path \$CMake) { \$env:PATH = "\$CMake;\$env:PATH" }
Run-Native 'cmake' @('--version')
Run-Native 'ctest' @('--version')

# First prove the Windows shared DLL exports the stable downstream ABI.  The
# broader in-tree C tests use static linkage below because many older tests
# call unannotated internal helpers and otherwise trip MinGW data auto-import
# pseudo-relocations unrelated to public ABI consumers.
Run-Native 'cmake' @(
    '-S', \$Repo,
    '-B', \$SharedBuild,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DQSIM_WERROR=ON',
    '-DQSIM_NATIVE_ARCH=OFF',
    '-DQSIM_FAST_MATH=OFF',
    '-DQSIM_BUILD_BENCHMARKS=OFF',
    '-DQSIM_BUILD_EXAMPLES=OFF',
    '-DQSIM_ENABLE_CUDA=OFF',
    '-DQSIM_BUILD_SHARED=ON',
    '-DQSIM_BUILD_STATIC=OFF'
)
Run-Native 'cmake' @('--build', \$SharedBuild, '--target', 'quantumsim', 'test_moonlab_export_abi', '--parallel', \$Jobs)
Run-Native 'ctest' @(
    '--test-dir', \$SharedBuild,
    '--output-on-failure',
    '--timeout', '600',
    '-L', 'abi',
    '-j', \$Jobs
)

Run-Native 'cmake' @(
    '-S', \$Repo,
    '-B', \$Build,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DQSIM_WERROR=ON',
    '-DQSIM_NATIVE_ARCH=OFF',
    '-DQSIM_FAST_MATH=OFF',
    '-DQSIM_BUILD_BENCHMARKS=OFF',
    '-DQSIM_BUILD_EXAMPLES=OFF',
    '-DQSIM_ENABLE_CUDA=OFF',
    '-DQSIM_BUILD_SHARED=OFF',
    '-DQSIM_BUILD_STATIC=ON'
)
Run-Native 'cmake' @('--build', \$Build, '--parallel', \$Jobs)
Run-Native 'ctest' @(
    '--test-dir', \$Build,
    '--output-on-failure',
    '--timeout', '600',
    '-LE', \$ExcludeLabels,
    '-E', \$ExcludeTests,
    '-L', \$Labels,
    '-j', \$Jobs
)
EOF
}

run_windows_target() {
    local target="$1"
    local local_script="$ARTIFACT_DIR/scripts/$target-smoke.ps1"
    write_windows_smoke_script "$local_script"
    run_and_log "$target upload source" "$ARTIFACT_DIR/logs/$target.upload-source.log" \
        scp "${SSH_OPTS[@]}" -q "$source_tar" "$target:moonlab-source-$snapshot_id.tar"
    run_and_log "$target upload script" "$ARTIFACT_DIR/logs/$target.upload-script.log" \
        scp "${SSH_OPTS[@]}" -q "$local_script" "$target:moonlab-mesh-smoke-$snapshot_id.ps1"
    run_and_log "$target smoke" "$ARTIFACT_DIR/logs/$target.smoke.log" \
        ssh "${SSH_OPTS[@]}" "$target" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
            -File "moonlab-mesh-smoke-$snapshot_id.ps1"
}

target_evidence_label() {
    local target="$1"
    if declare -F mesh_target_evidence_label >/dev/null; then
        local label
        label="$(mesh_target_evidence_label "$target")"
        if [[ -n "$label" ]]; then
            printf '%s\n' "$label"
            return
        fi
    fi
    printf '%s\n' "$target"
}

write_runtime_evidence() {
    local out="$1"
    {
        printf 'Moonlab mesh release smoke completed done.\n'
        printf 'Source snapshot: %s.\n' "$git_head"
        printf 'Source git tree: %s.\n' "$SOURCE_GIT_TREE"
        printf 'Source dirty: %s.\n' "$SOURCE_DIRTY"
        printf 'Source fingerprint: %s.\n' "$SOURCE_FINGERPRINT"
        printf 'Source snapshot SHA-256: %s.\n' "$SOURCE_TAR_SHA256"
        printf 'Expected distinct targets: %s.\n' "$EXPECTED_TARGETS"
        printf 'Run id: %s.\n' "$RUN_ID"
        printf 'Labels: %s.\n' "${LABELS//|/, }"
        printf 'Excluded labels: %s.\n' "${EXCLUDE_LABELS//|/, }"
        printf 'Excluded tests: %s.\n' "${EXCLUDE_TESTS//|/, }"

        while IFS=$'\t' read -r target status log; do
            [[ "$target" == "target" ]] && continue
            [[ -z "$target" ]] && continue

            local label kind
            label="$(target_evidence_label "$target")"
            kind="$(target_kind "$target")"
            if [[ "$kind" == "windows" && "$status" == "PASS" ]]; then
                printf '%s shared ABI DLL lane PASS done.\n' "$label"
                printf '%s static core/abi/gpu lane PASS done.\n' "$label"
            elif [[ "$status" == "PASS" ]]; then
                printf '%s lane PASS done.\n' "$label"
            else
                printf '%s lane FAIL review log %s.\n' "$label" "$log"
            fi
        done <"$summary"

        if [[ "$rc" -eq 0 ]]; then
            printf 'All release smoke targets completed cleanly.\n'
        else
            printf 'One or more release smoke targets require review.\n'
        fi
    } >"$out"
}

create_source_snapshot
SOURCE_TAR_SHA256="$(python3 - "$source_tar" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

summary="$ARTIFACT_DIR/summary.tsv"
runtime_evidence="$ARTIFACT_DIR/icc_runtime_evidence.txt"
printf 'target\tstatus\tlog\n' >"$summary"
rc=0

for target in $TARGETS; do
    echo "[mesh-smoke] target: $target"
    kind="$(target_kind "$target")"
    case "$kind" in
        local)
            runner=run_local_target
            ;;
        windows)
            runner=run_windows_target
            ;;
        *)
            runner=run_posix_target
            ;;
    esac
    if "$runner" "$target"; then
        printf '%s\tPASS\t%s\n' "$target" "$ARTIFACT_DIR/logs/$target.smoke.log" >>"$summary"
    else
        printf '%s\tFAIL\t%s\n' "$target" "$ARTIFACT_DIR/logs/$target.smoke.log" >>"$summary"
        rc=1
    fi
done

FINAL_IDENTITY_JSON="$(bash "$ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
if [[ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]]; then
    echo "[mesh-smoke] source changed during run: start=$SOURCE_FINGERPRINT end=${FINAL_FINGERPRINT:-unknown}" >&2
    rc=1
fi

artifact_manifest="$ARTIFACT_DIR/artifact_manifest.json"
python3 - "$summary" "$artifact_manifest" "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$SOURCE_TAR_SHA256" "$EXPECTED_TARGETS" "$rc" <<'PY'
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

summary_path, output_path, git_head, git_tree, dirty, fingerprint, snapshot_sha256, expected_targets, run_rc = sys.argv[1:]
targets = []
for line in Path(summary_path).read_text(encoding="utf-8").splitlines()[1:]:
    if not line.strip():
        continue
    target, status, log_name = line.split("\t")
    log = Path(log_name)
    targets.append({
        "target": target,
        "status": status,
        "log": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest() if log.is_file() else None,
    })
event = {
    "schema": "moonlab.mesh_release_smoke.v1",
    "kind": "moonlab_mesh",
    "name": "mesh_release_smoke_green",
    "value": "PASS" if int(run_rc) == 0 and len(targets) == int(expected_targets) and all(item["status"] == "PASS" for item in targets) else "FAIL",
    "git_head": git_head,
    "git_tree": git_tree,
    "dirty": dirty == "true",
    "source_fingerprint": fingerprint,
    "source_snapshot_sha256": snapshot_sha256,
    "expected_target_count": int(expected_targets),
    "target_count": len(targets),
    "targets": targets,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
Path(output_path).write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

write_runtime_evidence "$runtime_evidence"
echo "[mesh-smoke] summary: $summary"
echo "[mesh-smoke] icc runtime evidence: $runtime_evidence"
echo "[mesh-smoke] artifact manifest: $artifact_manifest"
column -t -s $'\t' "$summary" || cat "$summary"
exit "$rc"
