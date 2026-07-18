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
cd "$REPO_ROOT"

TRACE_DIR="${MOONLAB_TRACE_DIR:-scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_mpi_gpu.jsonl"
N="${MOONLAB_MPI_GPU_N:-33}"
RANKS="${MOONLAB_MPI_GPU_RANKS:-4}"
HOSTS="${MOONLAB_MPI_GPU_HOSTS:-}"
MPIEXEC="${MOONLAB_MPI_GPU_MPIEXEC:-mpirun}"
CMAKE_BIN="${MOONLAB_MPI_GPU_CMAKE:-cmake}"
EXECUTABLE="${MOONLAB_MPI_GPU_EXECUTABLE:-}"
COMMIT_SHA="${MOONLAB_MPI_GPU_COMMIT_SHA:-}"

mkdir -p "$TRACE_DIR"
: > "$TRACE"

if [ -z "$COMMIT_SHA" ]; then
  COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null)" || {
    echo "cannot determine commit; set MOONLAB_MPI_GPU_COMMIT_SHA" >&2
    exit 2
  }
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "refusing fleet attestation from a dirty tracked worktree" >&2
    exit 2
  fi
fi

if ! [[ "$COMMIT_SHA" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
  echo "invalid commit SHA: $COMMIT_SHA" >&2
  exit 2
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -le 32 ]; then
  echo "MOONLAB_MPI_GPU_N must be an integer greater than 32" >&2
  exit 2
fi
if ! [[ "$RANKS" =~ ^[0-9]+$ ]] || [ "$RANKS" -lt 2 ]; then
  echo "MOONLAB_MPI_GPU_RANKS must be at least 2" >&2
  exit 2
fi

LOG=""
BUILD_RC=0
if [ -z "$EXECUTABLE" ]; then
  BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moonlab-mpi-gpu.XXXXXX")"
  LOG="$BUILD_DIR/fleet-proof.log"
  echo "== fresh CUDA+MPI build: $BUILD_DIR =="
  "$CMAKE_BIN" -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_ENABLE_CUDA=ON -DQSIM_ENABLE_MPI=ON \
      -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=OFF >"$LOG" 2>&1 || BUILD_RC=$?
  if [ "$BUILD_RC" -eq 0 ]; then
    "$CMAKE_BIN" --build "$BUILD_DIR" --target test_mpi_sharded_gpu_ghz \
        --parallel "${QSIM_MPI_GPU_BUILD_JOBS:-4}" >>"$LOG" 2>&1 || BUILD_RC=$?
  fi
  EXECUTABLE="$BUILD_DIR/test_mpi_sharded_gpu_ghz"
else
  LOG="$(mktemp "${TMPDIR:-/tmp}/moonlab-mpi-gpu-run.XXXXXX.log")"
fi

RUN_RC="$BUILD_RC"
if [ "$BUILD_RC" -eq 0 ]; then
  if [ ! -x "$EXECUTABLE" ]; then
    echo "executable is absent or not executable: $EXECUTABLE" >>"$LOG"
    RUN_RC=127
  elif ! command -v "$MPIEXEC" >/dev/null 2>&1; then
    echo "MPI launcher is unavailable: $MPIEXEC" >>"$LOG"
    RUN_RC=127
  else
    mpi_cmd=("$MPIEXEC" -n "$RANKS")
    if [ -n "$HOSTS" ]; then
      mpi_cmd+=(--host "$HOSTS")
    fi
    if [ -n "${MOONLAB_MPI_GPU_MPI_ARGS:-}" ]; then
      read -r -a mpi_extra <<<"$MOONLAB_MPI_GPU_MPI_ARGS"
      mpi_cmd+=("${mpi_extra[@]}")
    fi
    mpi_cmd+=("$EXECUTABLE" "$N")

    echo "== MPI+CUDA fleet proof: N=$N ranks=$RANKS hosts=${HOSTS:-local} =="
    "${mpi_cmd[@]}" >>"$LOG" 2>&1
    RUN_RC=$?
  fi
fi

cat "$LOG"

python3 - "$LOG" "$TRACE" "$COMMIT_SHA" "$RUN_RC" <<'PY'
import datetime as dt
import json
import re
import sys

log_path, trace_path, commit_sha, raw_rc = sys.argv[1:]
run_rc = int(raw_rc)
with open(log_path, encoding="utf-8", errors="replace") as source:
    output = source.read()

pattern = re.compile(
    r"MOONLAB_MPI_SHARDED_GPU PASS "
    r"n=(?P<n>\d+) ranks=(?P<ranks>\d+) hosts=(?P<hosts>\d+) "
    r"gpu_endpoints=(?P<gpu_endpoints>\d+) "
    r"halo_swaps=(?P<halo_swaps>\d+) local_qubits=(?P<local_qubits>\d+)"
)
matches = list(pattern.finditer(output))
fields = {
    key: int(value)
    for key, value in (matches[-1].groupdict().items() if matches else [])
}

requirements = {
    "process_exit_zero": run_rc == 0,
    "marker_present": bool(matches),
    "n_gt_32": fields.get("n", 0) > 32,
    "multiple_ranks": fields.get("ranks", 0) >= 2,
    "multiple_gpu_endpoints": fields.get("gpu_endpoints", 0) >= 2,
    "halo_swap_exercised": fields.get("halo_swaps", 0) >= 1,
}
failed = [name for name, passed in requirements.items() if not passed]
value = "PASS" if not failed else "FAIL"
event = {
    "kind": "moonlab_mpi_gpu",
    "name": "mpi_sharded_gpu_works",
    "value": value,
    "commit_sha": commit_sha,
    "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "run_rc": run_rc,
    **fields,
    "detail": "exact sharded MPI+CUDA fleet proof" if not failed else ",".join(failed),
}
with open(trace_path, "w", encoding="utf-8") as trace:
    trace.write(json.dumps(event, separators=(",", ":")) + "\n")
print(json.dumps(event, indent=2, sort_keys=True))
raise SystemExit(0 if value == "PASS" else 1)
PY
