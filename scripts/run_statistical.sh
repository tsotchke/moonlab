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
# inside the root CMakeLists' QSIM_BUILD_TESTS block. Exact-source release
# evidence requires that hook in the source snapshot; transient injection is
# intentionally refused.
#
# Environment knobs:
#   BUILD_DIR           build tree (default: <repo>/build-statistical)
#   BUILD_TYPE          CMake build type (default: Release)
#   CMAKE_EXTRA_FLAGS   extra configure flags (e.g. sanitizers)
#   JOBS                parallel build jobs (default: 2 -- machine-fragile safe)
#   STAT_TARGETS        space list from {battery mlkem timing health}
#                       (default: all four)
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
# Invalidate saved PASS evidence before any current-tree prerequisite can fail.
: > "${TRACE_FILE}"

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

if ! grep -q 'add_subdirectory(tests/statistical)' "${CMAKE_FILE}"; then
    echo "run_statistical: exact-source evidence requires the committed tests/statistical CMake hook" >&2
    exit 2
fi

SOURCE_IDENTITY_JSON="$(bash "${REPO_ROOT}/scripts/run_moonlab_release_smoke.sh" --source-identity)" || exit 2
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

# --- configure + build ------------------------------------------------------
# Always reconfigure so the cache is demonstrably attached to this checkout
# and current CMake/source changes participate in the evidence build.
# shellcheck disable=SC2086
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    ${CMAKE_EXTRA_FLAGS:-}

cmake --build "${BUILD_DIR}" --target "${CMAKE_TARGETS[@]}" -j"${JOBS}"

# Build is done; restore the CMakeLists now so the tree is pristine for the
# rest of the run (binaries are already linked).
restore_cmakelists

# --- run + emit JSONL -------------------------------------------------------
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
                printf '{"kind":"moonlab_statistical","name":"%s","value":"%s","gating":%s,"stats":%s,"git_head":"%s","git_tree":"%s","dirty":%s,"source_fingerprint":"%s","artifact_sha256":"%s","generated_at":"%s","ts":"%s"}\n' \
                    "${name}" "${value}" "${gating}" "${stats}" \
                    "${SOURCE_GIT_HEAD}" "${SOURCE_GIT_TREE}" "${SOURCE_DIRTY}" "${SOURCE_FINGERPRINT}" \
                    "${BINARY_SHA256}" "${ts}" "${ts}" >> "${TRACE_FILE}"
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
    local ct binpath out rc before_count after_count expected_count
    ct="$(target_cmake "${logical}")"
    binpath="${BUILD_DIR}/tests/statistical/${ct}"
    if [ ! -x "${binpath}" ]; then
        echo "run_statistical: missing binary ${binpath}" >&2
        OVERALL_FAIL=1
        return
    fi
    BINARY_SHA256="$(python3 - "${binpath}" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
    before_count="$(wc -l < "${TRACE_FILE}" | tr -d ' ')"
    echo "=========================================================="
    echo "== ${ct}"
    echo "=========================================================="
    set +e
    out="$("${binpath}")"
    rc=$?
    set -e
    printf '%s\n' "${out}"
    emit_from_output <<< "${out}"
    after_count="$(wc -l < "${TRACE_FILE}" | tr -d ' ')"
    case "${logical}" in
        battery|mlkem) expected_count=2 ;;
        timing|health) expected_count=1 ;;
    esac
    if [ $((after_count - before_count)) -ne "${expected_count}" ]; then
        echo "run_statistical: ${ct} emitted $((after_count - before_count)) result(s), expected ${expected_count}" >&2
        OVERALL_FAIL=1
    fi
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

FINAL_IDENTITY_JSON="$(bash "${REPO_ROOT}/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
if [ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
    OVERALL_FAIL=1
    python3 - "$TRACE_FILE" "$SOURCE_FINGERPRINT" "$FINAL_FINGERPRINT" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
events = []
for line in path.read_text(encoding="utf-8").splitlines():
    event = json.loads(line)
    event["value"] = "FAIL"
    event["provenance_error"] = f"source changed during lane: start={sys.argv[2]} end={sys.argv[3] or 'unknown'}"
    events.append(event)
path.write_text("".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events), encoding="utf-8")
PY
fi

echo "=========================================================="
echo "JSONL evidence written to ${TRACE_FILE}"
cat "${TRACE_FILE}"
echo "=========================================================="

if [ "${OVERALL_FAIL}" -ne 0 ]; then
    echo "run_statistical: FAIL (a release-blocking result failed or a target crashed)" >&2
    exit 1
fi
echo "run_statistical: PASS"
