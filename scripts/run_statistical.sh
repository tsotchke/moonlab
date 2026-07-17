#!/usr/bin/env bash
#
# run_statistical.sh -- build and run the Moonlab adversarial statistical +
# crypto-negative-space lane, and emit ICC evidence as JSONL.
#
# Emits one object per named result to scripts/icc_traces/moonlab_statistical.jsonl
# with kind "moonlab_statistical".  Names:
#     qrng_statistical_battery     (release-blocking)
#     qrng_bias_positive_control   (release-blocking)
#     mlkem_negative_fuzz          (release-blocking)
#     mlkem_avalanche              (release-blocking)
#     entropy_health_rejects_bad   (release-blocking)
#     constant_time_variance       (informational unless QSIM_TIMING_STRICT=1)
#
# The gating field on each object records whether that result blocks a release.
# The script exits non-zero iff a gating (gating=1) result reported FAIL, or a
# target crashed.  A non-strict constant_time_variance FAIL is reported but
# never gates.
#
# Ownership: this lane adds ONLY new files.  The four test targets live in
# tests/statistical/, wired into the build by the integrator with
#     add_subdirectory(tests/statistical)
# inside the root CMakeLists' QSIM_BUILD_TESTS block.  Until that lands, set
# STAT_INJECT_HOOK=1 and this script temporarily appends the hook to the root
# CMakeLists for the duration of the build and restores it on exit (nothing is
# committed).  Once the permanent hook exists the injection is skipped
# (idempotent).
#
# Environment knobs:
#   BUILD_DIR           build tree (default: <repo>/build-statistical)
#   BUILD_TYPE          CMake build type (default: Release)
#   CMAKE_EXTRA_FLAGS   extra configure flags (e.g. sanitizers)
#   JOBS                parallel build jobs (default: 2 -- machine-fragile safe)
#   STAT_TARGETS        space list from {battery mlkem timing health}
#                       (default: all four)
#   STAT_INJECT_HOOK    1 to temporarily wire add_subdirectory (pre-integration)
#   QSIM_BATTERY_BYTES / QSIM_STREAM_BYTES / QSIM_GROVER_BYTES
#   QSIM_TIMING_REPS / QSIM_TIMING_STRICT
#   MOONLAB_SKIP_HW_ENTROPY (recommended =1 on hosted CI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-statistical}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
TRACE_DIR="${REPO_ROOT}/scripts/icc_traces"
TRACE_FILE="${TRACE_DIR}/moonlab_statistical.jsonl"
STAT_TARGETS="${STAT_TARGETS:-battery mlkem timing health}"

mkdir -p "${TRACE_DIR}"

# --- map logical target names to CMake target + binary path ----------------
target_cmake() {
    case "$1" in
        battery) echo "stat_qrng_battery" ;;
        mlkem)   echo "stat_mlkem_adversarial" ;;
        timing)  echo "stat_mlkem_timing" ;;
        health)  echo "stat_entropy_health_adversarial" ;;
        *) echo "" ;;
    esac
}

CMAKE_TARGETS=()
for t in ${STAT_TARGETS}; do
    ct="$(target_cmake "${t}")"
    if [ -z "${ct}" ]; then
        echo "run_statistical: unknown STAT_TARGETS entry '${t}'" >&2
        exit 2
    fi
    CMAKE_TARGETS+=("${ct}")
done

# --- optional temporary wiring of the subdirectory -------------------------
CMAKE_FILE="${REPO_ROOT}/CMakeLists.txt"
INJECTED=0
restore_cmakelists() {
    if [ "${INJECTED}" = "1" ] && [ -f "${CMAKE_FILE}.stat.bak" ]; then
        mv -f "${CMAKE_FILE}.stat.bak" "${CMAKE_FILE}"
        INJECTED=0
    fi
}
trap restore_cmakelists EXIT

if [ "${STAT_INJECT_HOOK:-0}" = "1" ] \
   && ! grep -q 'add_subdirectory(tests/statistical)' "${CMAKE_FILE}"; then
    echo "run_statistical: temporarily wiring add_subdirectory(tests/statistical)"
    cp "${CMAKE_FILE}" "${CMAKE_FILE}.stat.bak"
    INJECTED=1
    {
        echo ""
        echo "# TEMP (run_statistical.sh STAT_INJECT_HOOK): removed on script exit"
        echo "if(TARGET quantumsim AND NOT TARGET stat_qrng_battery)"
        echo "    add_subdirectory(tests/statistical)"
        echo "endif()"
    } >> "${CMAKE_FILE}"
fi

# --- configure + build ------------------------------------------------------
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    # shellcheck disable=SC2086
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        ${CMAKE_EXTRA_FLAGS:-}
fi

cmake --build "${BUILD_DIR}" --target "${CMAKE_TARGETS[@]}" -j"${JOBS}"

# Build is done; restore the CMakeLists now so the tree is pristine for the
# rest of the run (binaries are already linked).
restore_cmakelists

# --- run + emit JSONL -------------------------------------------------------
: > "${TRACE_FILE}"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
OVERALL_FAIL=0

emit_from_output() {
    # Reads RESULT lines on stdin, appends one JSONL object per result.
    local line name value gating stats ts
    while IFS= read -r line; do
        case "${line}" in
            "RESULT name="*)
                name="$(printf '%s' "${line}"   | sed -n 's/^RESULT name=\([^ ]*\) .*/\1/p')"
                value="$(printf '%s' "${line}"  | sed -n 's/^RESULT name=[^ ]* value=\([^ ]*\) .*/\1/p')"
                gating="$(printf '%s' "${line}" | sed -n 's/^RESULT name=[^ ]* value=[^ ]* gating=\([^ ]*\) stats=.*/\1/p')"
                stats="$(printf '%s' "${line}"  | sed -n 's/^RESULT name=[^ ]* value=[^ ]* gating=[^ ]* stats=//p')"
                [ -n "${gating}" ] || gating=0
                [ -n "${stats}" ] || stats='{}'
                ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                printf '{"kind":"moonlab_statistical","name":"%s","value":"%s","gating":%s,"stats":%s,"commit":"%s","ts":"%s"}\n' \
                    "${name}" "${value}" "${gating}" "${stats}" "${COMMIT}" "${ts}" >> "${TRACE_FILE}"
                if [ "${value}" = "FAIL" ] && [ "${gating}" = "1" ]; then
                    echo "run_statistical: GATING FAIL -> ${name}" >&2
                    OVERALL_FAIL=1
                fi
                ;;
        esac
    done
}

run_one() {
    local logical="$1"
    local ct binpath out rc
    ct="$(target_cmake "${logical}")"
    binpath="${BUILD_DIR}/tests/statistical/${ct}"
    if [ ! -x "${binpath}" ]; then
        echo "run_statistical: missing binary ${binpath}" >&2
        OVERALL_FAIL=1
        return
    fi
    echo "=========================================================="
    echo "== ${ct}"
    echo "=========================================================="
    set +e
    out="$("${binpath}")"
    rc=$?
    set -e
    printf '%s\n' "${out}"
    emit_from_output <<< "${out}"
    # Our binaries exit non-zero only on a gating failure (a non-strict
    # timing FAIL exits 0); a crash (>=128) is always a failure.
    if [ "${rc}" -ne 0 ]; then
        echo "run_statistical: ${ct} exited ${rc}" >&2
        OVERALL_FAIL=1
    fi
}

for t in ${STAT_TARGETS}; do
    run_one "${t}"
done

echo "=========================================================="
echo "JSONL evidence written to ${TRACE_FILE}"
cat "${TRACE_FILE}"
echo "=========================================================="

if [ "${OVERALL_FAIL}" -ne 0 ]; then
    echo "run_statistical: FAIL (a release-blocking result failed or a target crashed)" >&2
    exit 1
fi
echo "run_statistical: PASS"
