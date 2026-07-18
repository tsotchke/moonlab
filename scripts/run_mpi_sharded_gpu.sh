#!/usr/bin/env bash
# Build and run the exact MPI+CUDA release proof, then emit the commit-bound
# moonlab_mpi_gpu trace consumed by run_moonlab_release_smoke.sh.
#
# Typical multi-host invocation (paths must exist on every MPI host):
#   MOONLAB_MPI_GPU_COMMIT_SHA=<sha> \
#   MOONLAB_MPI_GPU_EXECUTABLE=/shared/path/test_mpi_sharded_gpu_ghz \
#   MOONLAB_MPI_GPU_HOSTS='gpu-a:2,gpu-b:2' \
#   MOONLAB_MPI_GPU_RANKS=4 scripts/run_mpi_sharded_gpu.sh
#
# Without MOONLAB_MPI_GPU_EXECUTABLE, the script configures a fresh temporary
# CUDA+MPI build.  CUDACXX and CMAKE_CUDA_ARCHITECTURES may be supplied for a
# non-default CUDA toolkit.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

TRACE_DIR="${MOONLAB_TRACE_DIR:-scripts/icc_traces}"
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
COMMIT_SHA="${MOONLAB_MPI_GPU_COMMIT_SHA:-}"

mkdir -p "$TRACE_DIR"
: > "$TRACE"

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || {
  echo "cannot determine repository commit" >&2
  exit 2
}
if [ -z "$COMMIT_SHA" ]; then
  COMMIT_SHA="$HEAD_SHA"
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing fleet attestation from a dirty tracked worktree" >&2
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
  LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-mpi-gpu.XXXXXX")"
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
  LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-mpi-gpu.XXXXXX")"
  BUILD_RC=0
fi

ROUTINE_LOG="$LOG_DIR/routine-${ROUTINE_N}.log"
EXACT_LOG="$LOG_DIR/exact-${N}.log"

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

python3 - "$ROUTINE_LOG" "$EXACT_LOG" "$TRACE" "$COMMIT_SHA" "$ROUTINE_N" "$N" "$ROUTINE_RC" "$EXACT_RC" <<'PY'
import datetime as dt
import json
import re
import sys

routine_log, exact_log, trace_path, commit_sha, routine_n, exact_n, routine_rc, exact_rc = sys.argv[1:]
routine_n = int(routine_n)
exact_n = int(exact_n)
routine_rc = int(routine_rc)
exact_rc = int(exact_rc)

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

requirements = {
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
    "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    **fields,
    "detail": "exact sharded MPI+CUDA fleet proof" if not failed else ",".join(failed),
}
with open(trace_path, "w", encoding="utf-8") as trace:
    trace.write(json.dumps(event, separators=(",", ":")) + "\n")
print(json.dumps(event, indent=2, sort_keys=True))
raise SystemExit(0 if value == "PASS" else 1)
PY
