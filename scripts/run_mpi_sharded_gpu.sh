#!/usr/bin/env bash
# Build and run the exact MPI+CUDA release proof, then emit the commit-bound
# moonlab_mpi_gpu trace consumed by run_moonlab_release_smoke.sh.
#
# Typical multi-host invocation (paths must exist on every MPI host):
#   MOONLAB_MPI_GPU_COMMIT_SHA=<sha> \
#   MOONLAB_MPI_GPU_EXECUTABLE=/shared/path/test_mpi_sharded_gpu_ghz \
#   MOONLAB_MPI_GPU_EXECUTABLE_SHA256=<sha256> \
#   MOONLAB_MPI_GPU_HOSTS='gpu-a:2,gpu-b:2' \
#   MOONLAB_MPI_GPU_RANKS=4 scripts/run_mpi_sharded_gpu.sh
#
# Without MOONLAB_MPI_GPU_EXECUTABLE, the script configures a fresh temporary
# CUDA+MPI build.  CUDACXX and CMAKE_CUDA_ARCHITECTURES may be supplied for a
# non-default CUDA toolkit.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

TRACE_DIR="${MOONLAB_TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_mpi_gpu.jsonl"
EXPECTED_N=33
EXPECTED_RANKS=4
N="${MOONLAB_MPI_GPU_N:-$EXPECTED_N}"
ROUTINE_N=12
RANKS="${MOONLAB_MPI_GPU_RANKS:-$EXPECTED_RANKS}"
HOSTS="${MOONLAB_MPI_GPU_HOSTS:-}"
MPIEXEC="${MOONLAB_MPI_GPU_MPIEXEC:-mpirun}"
CMAKE_BIN="${MOONLAB_MPI_GPU_CMAKE:-cmake}"
EXECUTABLE="${MOONLAB_MPI_GPU_EXECUTABLE:-}"
EXPECTED_EXECUTABLE_SHA256="${MOONLAB_MPI_GPU_EXECUTABLE_SHA256:-}"
COMMIT_SHA="${MOONLAB_MPI_GPU_COMMIT_SHA:-}"

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
TRACE="$TRACE_DIR/moonlab_mpi_gpu.jsonl"

# These are executed as quoted array element zero (never eval'd), but require
# them to resolve to actual executables before creating any build/log output.
if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "CMake executable is unavailable: $CMAKE_BIN" >&2
  exit 2
fi
if ! command -v "$MPIEXEC" >/dev/null 2>&1; then
  echo "MPI launcher is unavailable: $MPIEXEC" >&2
  exit 2
fi

mkdir -p "$TRACE_DIR"

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || {
  echo "cannot determine repository commit" >&2
  exit 2
}
if [ -z "$COMMIT_SHA" ]; then
  COMMIT_SHA="$HEAD_SHA"
fi
if [ -n "$(git status --porcelain=v1 --untracked-files=all -- . ':(exclude)scripts/icc_traces/**')" ]; then
  echo "refusing fleet attestation from a dirty source worktree" >&2
  exit 2
fi
if [ "$COMMIT_SHA" != "$HEAD_SHA" ]; then
  echo "requested commit $COMMIT_SHA does not match checked-out HEAD $HEAD_SHA" >&2
  exit 2
fi

if ! [[ "$COMMIT_SHA" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
  echo "invalid commit SHA: $COMMIT_SHA" >&2
  exit 2
fi

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
if [ "$SOURCE_DIRTY" != "false" ]; then
  echo "source identity unexpectedly reports dirty=true" >&2
  exit 2
fi
: > "$TRACE"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -ne "$EXPECTED_N" ]; then
  echo "MOONLAB_MPI_GPU_N must be exactly $EXPECTED_N" >&2
  exit 2
fi
if ! [[ "$RANKS" =~ ^[0-9]+$ ]] || [ "$RANKS" -ne "$EXPECTED_RANKS" ]; then
  echo "MOONLAB_MPI_GPU_RANKS must be exactly $EXPECTED_RANKS" >&2
  exit 2
fi
if ! [[ "$HOSTS" =~ ^[^,[:space:]:]+:2,[^,[:space:]:]+:2$ ]]; then
  echo "MOONLAB_MPI_GPU_HOSTS must be exactly two host names with two slots each (host-a:2,host-b:2)" >&2
  exit 2
fi
IFS=',' read -r HOST_ENTRY_A HOST_ENTRY_B <<<"$HOSTS"
if [ "${HOST_ENTRY_A%:2}" = "${HOST_ENTRY_B%:2}" ]; then
  echo "MOONLAB_MPI_GPU_HOSTS must name two distinct hosts" >&2
  exit 2
fi

case " ${MOONLAB_MPI_GPU_MPI_ARGS:-} " in
  *" -n "*|*" -np "*|*" --np "*|*" --host "*|*" -H "*|*" --hostfile "*|*" -hostfile "*|*" --machinefile "*|*" --map-by "*)
    echo "MOONLAB_MPI_GPU_MPI_ARGS must not override the attested rank/host topology" >&2
    exit 2
    ;;
esac

if [ -z "$EXECUTABLE" ]; then
  LOG_DIR="$(mktemp -d "/tmp/moonlab-mpi-gpu.XXXXXX")"
  BUILD_LOG="$LOG_DIR/build.log"
  echo "== fresh CUDA+MPI build: $LOG_DIR =="
  BUILD_RC=0
  "$CMAKE_BIN" -S . -B "$LOG_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_ENABLE_CUDA=ON -DQSIM_ENABLE_MPI=ON \
      -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=OFF >"$BUILD_LOG" 2>&1 || BUILD_RC=$?
  if [ "$BUILD_RC" -eq 0 ]; then
    "$CMAKE_BIN" --build "$LOG_DIR" --target test_mpi_sharded_gpu_ghz \
        --parallel "${QSIM_MPI_GPU_BUILD_JOBS:-4}" >>"$BUILD_LOG" 2>&1 || BUILD_RC=$?
  fi
  EXECUTABLE="$LOG_DIR/test_mpi_sharded_gpu_ghz"
else
  LOG_DIR="$(mktemp -d "/tmp/moonlab-mpi-gpu.XXXXXX")"
  BUILD_RC=0
fi

if [ ! -x "$EXECUTABLE" ]; then
  EXECUTABLE_SHA256=""
else
  EXECUTABLE_SHA256="$(python3 - "$EXECUTABLE" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
fi
if [ -n "${MOONLAB_MPI_GPU_EXECUTABLE:-}" ]; then
  if ! [[ "$EXPECTED_EXECUTABLE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "MOONLAB_MPI_GPU_EXECUTABLE_SHA256 is required for a supplied executable" >&2
    exit 2
  fi
  if [ "$EXECUTABLE_SHA256" != "${EXPECTED_EXECUTABLE_SHA256,,}" ]; then
    echo "supplied executable SHA-256 does not match MOONLAB_MPI_GPU_EXECUTABLE_SHA256" >&2
    exit 2
  fi
else
  EXPECTED_EXECUTABLE_SHA256="$EXECUTABLE_SHA256"
fi

ROUTINE_LOG="$LOG_DIR/routine-${ROUTINE_N}.log"
EXACT_LOG="$LOG_DIR/exact-${N}.log"
HASH_LOG="$LOG_DIR/executable-sha256.log"

run_rank_hash_preflight() {
  : > "$HASH_LOG"
  if [ "$BUILD_RC" -ne 0 ] || [ ! -x "$EXECUTABLE" ] || ! command -v "$MPIEXEC" >/dev/null 2>&1; then
    echo "executable hash preflight prerequisites are unavailable" >>"$HASH_LOG"
    return 127
  fi
  local -a hash_cmd=("$MPIEXEC" -n "$RANKS" --host "$HOSTS")
  if [ -n "${MOONLAB_MPI_GPU_MPI_ARGS:-}" ]; then
    local -a mpi_extra=()
    read -r -a mpi_extra <<<"$MOONLAB_MPI_GPU_MPI_ARGS"
    hash_cmd+=( "${mpi_extra[@]}" )
  fi
  hash_cmd+=(sh -c '
path=$1
host=$(hostname)
if command -v sha256sum >/dev/null 2>&1; then
  line=$(sha256sum "$path")
  digest=${line%% *}
elif command -v shasum >/dev/null 2>&1; then
  line=$(shasum -a 256 "$path")
  digest=${line%% *}
else
  exit 127
fi
printf "MOONLAB_MPI_EXECUTABLE_SHA256 host=%s sha256=%s\\n" "$host" "$digest"
' _ "$EXECUTABLE")
  "${hash_cmd[@]}" >"$HASH_LOG" 2>&1
}

run_rank_hash_preflight
HASH_RC=$?
cat "$HASH_LOG"

run_mpi_gpu() {
  local run_n="$1"
  local log="$2"
  local rc=127

  : > "$log"
  if [ "$BUILD_RC" -ne 0 ]; then
    [ -n "${BUILD_LOG:-}" ] && cat "$BUILD_LOG" >>"$log"
    echo "CUDA+MPI build failed (rc=$BUILD_RC); skipping run for N=$run_n" >>"$log"
    return 127
  fi
  if [ "$HASH_RC" -ne 0 ]; then
    cat "$HASH_LOG" >>"$log"
    echo "rank-local executable SHA-256 preflight failed (rc=$HASH_RC)" >>"$log"
    return 126
  fi

  if [ ! -x "$EXECUTABLE" ]; then
    echo "executable is absent or not executable: $EXECUTABLE" >>"$log"
    return 127
  fi

  if ! command -v "$MPIEXEC" >/dev/null 2>&1; then
    echo "MPI launcher is unavailable: $MPIEXEC" >>"$log"
    return 127
  fi

  local -a mpi_cmd=("$MPIEXEC" -n "$RANKS")
  if [ -n "$HOSTS" ]; then
    mpi_cmd+=(--host "$HOSTS")
  fi
  if [ -n "${MOONLAB_MPI_GPU_MPI_ARGS:-}" ]; then
    read -r -a mpi_extra <<<"$MOONLAB_MPI_GPU_MPI_ARGS"
    mpi_cmd+=( "${mpi_extra[@]}" )
  fi
  mpi_cmd+=( "$EXECUTABLE" "$run_n" )

  echo "== MPI+CUDA fleet proof: N=$run_n ranks=$RANKS hosts=${HOSTS:-local} =="
  "${mpi_cmd[@]}" >>"$log" 2>&1
  rc=$?
  return "$rc"
}

run_mpi_gpu "$ROUTINE_N" "$ROUTINE_LOG"
ROUTINE_RC=$?
if [ "$ROUTINE_RC" -eq 0 ] &&
   grep -Eq "MOONLAB_MPI_SHARDED_GPU PASS n=$ROUTINE_N ranks=$EXPECTED_RANKS hosts=2 host_slots_2x2=1 gpu_endpoints=([2-9]|[1-9][0-9]+) halo_swaps=([1-9][0-9]*) " "$ROUTINE_LOG"; then
  run_mpi_gpu "$N" "$EXACT_LOG"
  EXACT_RC=$?
else
  : > "$EXACT_LOG"
  echo "routine N=$ROUTINE_N topology preflight failed; exact N=$N run skipped" >>"$EXACT_LOG"
  EXACT_RC=125
fi

cat "$ROUTINE_LOG"
cat "$EXACT_LOG"

FINAL_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
ROUTINE_LOG_SHA256="$(python3 - "$ROUTINE_LOG" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
EXACT_LOG_SHA256="$(python3 - "$EXACT_LOG" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

python3 - "$ROUTINE_LOG" "$EXACT_LOG" "$HASH_LOG" "$TRACE" "$COMMIT_SHA" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$FINAL_FINGERPRINT" "$EXECUTABLE_SHA256" "$ROUTINE_LOG_SHA256" "$EXACT_LOG_SHA256" "$ROUTINE_N" "$N" "$ROUTINE_RC" "$EXACT_RC" "$HASH_RC" <<'PY'
import datetime as dt
import json
import re
import sys

(
    routine_log,
    exact_log,
    hash_log,
    trace_path,
    commit_sha,
    git_tree,
    dirty,
    source_fingerprint,
    final_fingerprint,
    executable_sha256,
    routine_log_sha256,
    exact_log_sha256,
    routine_n,
    exact_n,
    routine_rc,
    exact_rc,
    hash_rc,
) = sys.argv[1:]
routine_n = int(routine_n)
exact_n = int(exact_n)
routine_rc = int(routine_rc)
exact_rc = int(exact_rc)
hash_rc = int(hash_rc)
dirty = dirty == "true"

PATTERN = re.compile(
    r"MOONLAB_MPI_SHARDED_GPU (?P<marker>PASS|FAIL) "
    r"n=(?P<n>\d+) ranks=(?P<ranks>\d+) hosts=(?P<hosts>\d+) "
    r"host_slots_2x2=(?P<host_slots_2x2>\d+) "
    r"gpu_endpoints=(?P<gpu_endpoints>\d+) "
    r"halo_swaps=(?P<halo_swaps>\d+) local_qubits=(?P<local_qubits>\d+)"
)


def parse_event(log_path, target_n):
    match_result = None
    with open(log_path, encoding="utf-8", errors="replace") as source:
        for match in PATTERN.finditer(source.read()):
            data = match.groupdict()
            n = int(data["n"])
            if n != target_n:
                continue
            match_result = {
                "marker": data["marker"],
                "n": n,
                "ranks": int(data["ranks"]),
                "hosts": int(data["hosts"]),
                "host_slots_2x2": int(data["host_slots_2x2"]),
                "gpu_endpoints": int(data["gpu_endpoints"]),
                "halo_swaps": int(data["halo_swaps"]),
                "local_qubits": int(data["local_qubits"]),
            }
    return match_result


routine_event = parse_event(routine_log, routine_n)
exact_event = parse_event(exact_log, exact_n)

hash_pattern = re.compile(r"MOONLAB_MPI_EXECUTABLE_SHA256 host=(\S+) sha256=([0-9a-fA-F]{64})")
with open(hash_log, encoding="utf-8", errors="replace") as source:
    rank_hashes = hash_pattern.findall(source.read())
rank_hash_hosts = {host for host, _ in rank_hashes}
rank_hash_values = {value.lower() for _, value in rank_hashes}

requirements = {
    "source_clean": not dirty,
    "source_unchanged": source_fingerprint == final_fingerprint,
    "executable_sha256_present": bool(re.fullmatch(r"[0-9a-f]{64}", executable_sha256)),
    "rank_hash_preflight_exit_zero": hash_rc == 0,
    "rank_hash_count_exact": len(rank_hashes) == 4,
    "rank_hash_hosts_multiple": len(rank_hash_hosts) >= 2,
    "rank_hashes_exact": rank_hash_values == {executable_sha256},
    "routine_exit_zero": routine_rc == 0,
    "routine_marker_pass": routine_event is not None and routine_event["marker"] == "PASS",
    "exact_exit_zero": exact_rc == 0,
    "exact_marker_pass": exact_event is not None and exact_event["marker"] == "PASS",
}

fields = {
    "run_rc": exact_rc,
    "exact_run_rc": exact_rc,
    "routine_run_rc": routine_rc,
    "routine_n": routine_n,
    "exact_commit_sha": commit_sha,
    "git_head": commit_sha,
    "git_tree": git_tree,
    "dirty": dirty,
    "source_fingerprint": source_fingerprint,
    "executable_sha256": executable_sha256,
    "routine_log_sha256": routine_log_sha256,
    "exact_log_sha256": exact_log_sha256,
    "rank_hash_count": len(rank_hashes),
    "rank_hash_host_count": len(rank_hash_hosts),
    "routine_test_passed": requirements["routine_exit_zero"] and requirements["routine_marker_pass"],
}
if routine_event is not None:
    fields.update(
        {
            "routine_marker": routine_event["marker"],
            "routine_ranks": routine_event["ranks"],
            "routine_hosts": routine_event["hosts"],
            "routine_host_slots_2x2": routine_event["host_slots_2x2"],
            "routine_gpu_endpoints": routine_event["gpu_endpoints"],
            "routine_halo_swaps": routine_event["halo_swaps"],
        }
    )
else:
    fields.update(
        {
            "routine_marker": "MISSING",
            "routine_ranks": 0,
            "routine_hosts": 0,
            "routine_host_slots_2x2": 0,
            "routine_gpu_endpoints": 0,
            "routine_halo_swaps": 0,
        }
    )

if exact_event is not None:
    fields.update(
        {
            "n": exact_event["n"],
            "ranks": exact_event["ranks"],
            "hosts": exact_event["hosts"],
            "host_slots_2x2": exact_event["host_slots_2x2"],
            "gpu_endpoints": exact_event["gpu_endpoints"],
            "halo_swaps": exact_event["halo_swaps"],
            "local_qubits": exact_event["local_qubits"],
            "exact_marker": exact_event["marker"],
        }
    )
else:
    fields.update(
        {
            "n": 0,
            "ranks": 0,
            "hosts": 0,
            "host_slots_2x2": 0,
            "gpu_endpoints": 0,
            "halo_swaps": 0,
            "local_qubits": 0,
            "exact_marker": "MISSING",
        }
    )

requirements.update(
    {
        "routine_n_exact": routine_event is not None and routine_event["n"] == 12,
        "routine_ranks_exact": fields["routine_ranks"] == 4,
        "routine_hosts_exact": fields["routine_hosts"] == 2,
        "routine_host_slots_exact": fields["routine_host_slots_2x2"] == 1,
        "routine_multiple_gpu_endpoints": fields["routine_gpu_endpoints"] >= 2,
        "routine_halo_swap_exercised": fields["routine_halo_swaps"] >= 1,
        "exact_n_exact": fields["n"] == 33,
        "exact_ranks_exact": fields["ranks"] == 4,
        "exact_hosts_exact": fields["hosts"] == 2,
        "exact_host_slots_exact": fields["host_slots_2x2"] == 1,
        "exact_multiple_gpu_endpoints": fields["gpu_endpoints"] >= 2,
        "exact_halo_swap_exercised": fields["halo_swaps"] >= 1,
        "routine_fields_present": routine_event is not None,
        "exact_fields_present": exact_event is not None,
    }
)

failed = [name for name, passed in requirements.items() if not passed]
value = "PASS" if not failed else "FAIL"
event = {
    "kind": "moonlab_mpi_gpu",
    "name": "mpi_sharded_gpu_works",
    "value": value,
    "commit_sha": commit_sha,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    **fields,
    "detail": "exact sharded MPI+CUDA fleet proof" if not failed else ",".join(failed),
}
event["ts"] = event["generated_at"]
with open(trace_path, "w", encoding="utf-8") as trace:
    trace.write(json.dumps(event, separators=(",", ":")) + "\n")
print(json.dumps(event, indent=2, sort_keys=True))
raise SystemExit(0 if value == "PASS" else 1)
PY
