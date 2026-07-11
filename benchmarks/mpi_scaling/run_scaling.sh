#!/usr/bin/env bash
# Moonlab v1.0 MPI scaling harness.
#
# Drives examples/distributed/large_state_random_circuit at a grid of
# (N, ranks) points and emits a JSON report.  Requires an MPI build
# (`cmake -DQSIM_ENABLE_MPI=ON ...`) and a working `mpirun`.
#
# Usage:
#   ./benchmarks/mpi_scaling/run_scaling.sh \
#       --binary  build/examples/distributed/large_state_random_circuit \
#       --depth   8 \
#       --seed    42 \
#       --output  bench/mpi_scaling.json
#
# Default grid: N in {32, 34, 36}; ranks in {1, 4, 16}.  Adjust via
# --qubits / --ranks if you have a different cluster shape.

set -euo pipefail

BINARY=""
DEPTH=8
SEED=42
OUTPUT="mpi_scaling.json"
QUBITS=(32 34 36)
RANKS=(1 4 16)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)  BINARY="$2";        shift 2 ;;
        --depth)   DEPTH="$2";         shift 2 ;;
        --seed)    SEED="$2";          shift 2 ;;
        --output)  OUTPUT="$2";        shift 2 ;;
        --qubits)  IFS=, read -r -a QUBITS <<< "$2"; shift 2 ;;
        --ranks)   IFS=, read -r -a RANKS  <<< "$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$BINARY" ]]; then
    BINARY="build/examples/distributed/large_state_random_circuit"
fi
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not built or not executable" >&2
    echo "       Rebuild with -DQSIM_ENABLE_MPI=ON first." >&2
    exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "error: mpirun not in PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
HOST_LABEL=${MOONLAB_BENCH_HOST_LABEL:-redacted}
if [ "${MOONLAB_BENCH_INCLUDE_HOSTNAME:-0}" = "1" ]; then
  HOST_LABEL=$(hostname)
fi
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    printf '{\n'
    printf '  "harness":  "moonlab/mpi-scaling/v1.0",\n'
    printf '  "host":     "%s",\n' "$HOST_LABEL"
    printf '  "date":     "%s",\n' "$DATE"
    printf '  "depth":    %d,\n' "$DEPTH"
    printf '  "seed":     %d,\n' "$SEED"
    printf '  "binary":   "%s",\n' "$BINARY"
    printf '  "points":   [\n'

    first=1
    for n in "${QUBITS[@]}"; do
        for r in "${RANKS[@]}"; do
            echo "[mpi_scaling] N=$n ranks=$r ..." >&2
            t0=$(date +%s.%N)
            out=$(mpirun --oversubscribe -n "$r" "$BINARY" "$n" "$DEPTH" "$SEED" 2>&1 || true)
            t1=$(date +%s.%N)
            wall_s=$(echo "$t0 $t1" | awk '{printf "%.3f", $2 - $1}')
            sim_s=$(echo "$out" | sed -n 's/.*simulation time:[ ]*\([0-9.]*\) s.*/\1/p' | head -1)
            norm=$(echo "$out" | sed -n 's/.*global L2 norm:[ ]*\([0-9.]*\).*/\1/p' | head -1)
            sim_s=${sim_s:-NaN}
            norm=${norm:-NaN}

            if [[ $first -eq 1 ]]; then first=0; else printf '    ,\n'; fi
            printf '    {"N": %d, "ranks": %d, "wall_s": %s, "sim_s": %s, "norm": %s}' \
                   "$n" "$r" "$wall_s" "$sim_s" "$norm"
        done
    done

    printf '\n  ]\n'
    printf '}\n'
} > "$OUTPUT"

echo "[mpi_scaling] results -> $OUTPUT" >&2
