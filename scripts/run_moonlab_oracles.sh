#!/usr/bin/env bash
# run_moonlab_oracles.sh -- drive the Moonlab adversarial oracle matrix
# (.swarm/ADVERSARIAL_TESTING_CAMPAIGN.md) and translate each pillar's result
# into an ICC runtime trace event.
#
# For each pillar it emits one JSON-L line to
# scripts/icc_traces/moonlab_oracles.jsonl:
#
#   {"kind":"moonlab_oracle","name":"<pillar_event>","value":"PASS"|"FAIL",
#    "total":N,"failed":F,"xfail":K,"xpass":S,"snippet":"...","confidence":0.95}
#
# The six events this runner owns are exactly the ones the integrator's
# moonlab-adversarial-matrix target requires, plus a corpus artifact
# validation evidence event:
#   backend_differential_oracle   (P1)
#   gradient_oracle               (P2)
#   measurement_statistics_oracle (P3)
#   edge_matrix_oracle            (P5)
#   property_invariants_oracle    (P6)
#   corpus_artifacts_validated    (artifact contract evidence)
#
# The oracle binaries are KNOWN_FAILURES-aware: a binary exits 0 when its only
# failures are allowlisted in tests/oracle/KNOWN_FAILURES.txt (XFAIL). This
# runner exits nonzero iff any pillar has a non-allowlisted FAIL.
#
# Usage:
#   scripts/run_moonlab_oracles.sh              # regen corpus, build, run, emit
#   SEED=12345 scripts/run_moonlab_oracles.sh   # different corpus seed
#   SKIP_BUILD=1 scripts/run_moonlab_oracles.sh # reuse the existing build
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-moonlab-oracles}"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/moonlab_oracles.jsonl"
KNOWN_FAILURES="$REPO_ROOT/tests/oracle/KNOWN_FAILURES.txt"
SEED="${SEED:-20260717}"
JOBS="${JOBS:-2}"

BUILD_DIR="$(python3 - "$BUILD_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
case "$BUILD_DIR" in
    "$REPO_ROOT/build"|"$REPO_ROOT"/build-*|"$REPO_ROOT"/build_*) ;;
    *) echo "BUILD_DIR must resolve to a build*, build-*, or build_* path under $REPO_ROOT" >&2; exit 2 ;;
esac
ORACLE_LOG_DIR="$BUILD_DIR/moonlab-oracle-logs"
CONFIGURE_LOG="$ORACLE_LOG_DIR/configure.log"
BUILD_LOG="$ORACLE_LOG_DIR/build.log"
CTEST_LOG="$ORACLE_LOG_DIR/ctest.log"

mkdir -p "$TRACE_DIR" "$ORACLE_LOG_DIR"
# Clear saved evidence before any generation/configure/build step can fail.
# A failed current-tree attempt must never leave an older PASS consumable.
: > "$TRACE_FILE"

# Single-threaded BLAS: the oracle circuits are small, and thread fan-out adds
# far more spin overhead than it saves, so this keeps the full run fast.
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MOONLAB_ORACLE_KNOWN_FAILURES="$KNOWN_FAILURES"

# Pillar table: <ctest_name>|<binary>|<event_name>
PILLARS=(
  "oracle_backend_differential|test_backend_differential|backend_differential_oracle"
  "oracle_gradient|test_gradient_oracle|gradient_oracle"
  "oracle_measurement_statistics|test_measurement_oracle|measurement_statistics_oracle"
  "oracle_edge_matrix|test_edge_matrix|edge_matrix_oracle"
  "oracle_property_invariants|test_property_invariants|property_invariants_oracle"
)

echo "== Moonlab adversarial oracle matrix =="
echo "   corpus seed : $SEED"
echo "   trace file  : $TRACE_FILE"

# 1. Regenerate the corpus as a pure function of the seed.
python3 "$REPO_ROOT/scripts/gen_circuit_corpus.py" --seed "$SEED" \
        --out-dir "$REPO_ROOT/tests/oracle/corpus" || {
    echo "run_moonlab_oracles.sh: corpus generation failed" >&2
    exit 2
}

SOURCE_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || {
    echo "run_moonlab_oracles.sh: source identity capture failed" >&2
    exit 2
}
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

# 2. Build the oracle targets (so a regenerated corpus takes effect).
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "run_moonlab_oracles.sh: SKIP_BUILD cannot produce exact-source release evidence" >&2
    exit 2
else
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
            >"$CONFIGURE_LOG" 2>&1 || {
            echo "run_moonlab_oracles.sh: cmake configure failed (see $CONFIGURE_LOG)" >&2
            exit 2
        }
    fi
    cmake --build "$BUILD_DIR" --parallel "$JOBS" \
        --target test_backend_differential test_gradient_oracle \
                 test_measurement_oracle test_edge_matrix test_property_invariants \
        >"$BUILD_LOG" 2>&1 || {
        echo "run_moonlab_oracles.sh: oracle build failed (see $BUILD_LOG)" >&2
        exit 2
    }
fi

# 3. Run the oracle-labelled tests via ctest and capture their output.
( cd "$BUILD_DIR" && ctest -L oracle -V --timeout 150 ) >"$CTEST_LOG" 2>&1
CTEST_RC=$?
echo "   ctest -L oracle exit: $CTEST_RC"

FINAL_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)" || FINAL_IDENTITY_JSON="{}"
FINAL_FINGERPRINT="$(python3 - "$FINAL_IDENTITY_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("source_fingerprint", ""))
PY
)"
SOURCE_CHANGED=0
if [ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
    echo "run_moonlab_oracles.sh: source changed during the lane" >&2
    SOURCE_CHANGED=1
fi

# 4. Translate each pillar's ORACLE_SUMMARY line into a trace event.
overall_fail=0
CIRCUIT_CORPUS_PATH="$REPO_ROOT/tests/oracle/corpus/circuit_corpus.json"
CIRCUIT_CORPUS_REL_PATH="tests/oracle/corpus/circuit_corpus.json"
printf '%-34s %-6s %s\n' "PILLAR" "VALUE" "counts"
for row in "${PILLARS[@]}"; do
    IFS='|' read -r _tname _bin event <<< "$row"
    line="$(grep -E "ORACLE_SUMMARY pillar=${event} " "$CTEST_LOG" | tail -1)"
    if [ -z "$line" ]; then
        # Pillar produced no summary (crash/timeout): hard FAIL.
        value="FAIL"; total=0; failed=-1; xfail=0; xpass=0
        snippet="no ORACLE_SUMMARY captured (crash or timeout)"
        overall_fail=1
    else
        total=$(sed -n 's/.*total=\([0-9]*\).*/\1/p'  <<< "$line")
        failed=$(sed -n 's/.*failed=\([0-9]*\).*/\1/p' <<< "$line")
        xfail=$(sed -n 's/.*xfail=\([0-9]*\).*/\1/p'   <<< "$line")
        xpass=$(sed -n 's/.*xpass=\([0-9]*\).*/\1/p'   <<< "$line")
        if [ "${failed:-1}" -eq 0 ]; then
            value="PASS"
        else
            value="FAIL"; overall_fail=1
        fi
        snippet="${failed} non-allowlisted fail(s), ${xfail} quarantined (XFAIL), ${xpass} stale (XPASS) of ${total} probes"
    fi
    if [ "$SOURCE_CHANGED" -ne 0 ]; then
        value="FAIL"
        snippet="source changed during lane: start=$SOURCE_FINGERPRINT end=${FINAL_FINGERPRINT:-unknown}"
        overall_fail=1
    fi
    generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%-34s %-6s total=%s failed=%s xfail=%s xpass=%s\n' \
        "$event" "$value" "$total" "$failed" "$xfail" "$xpass"
    printf '{"kind":"moonlab_oracle","name":"%s","value":"%s","total":%s,"failed":%s,"xfail":%s,"xpass":%s,"seed":%s,"snippet":"%s","confidence":0.95,"git_head":"%s","git_tree":"%s","dirty":%s,"source_fingerprint":"%s","generated_at":"%s"}\n' \
        "$event" "$value" "${total:-0}" "${failed:-0}" "${xfail:-0}" "${xpass:-0}" "$SEED" "$snippet" \
        "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$generated_at" \
        >> "$TRACE_FILE"
done

if [ "$CTEST_RC" -ne 0 ]; then
    overall_fail=1
fi
if [ "$overall_fail" -ne 0 ]; then
    ORACLE_CORPUS_STATUS="FAIL"
elif [ ! -f "$CIRCUIT_CORPUS_PATH" ]; then
    ORACLE_CORPUS_STATUS="FAIL"
elif [ "$SOURCE_CHANGED" -ne 0 ]; then
    ORACLE_CORPUS_STATUS="FAIL"
else
    ORACLE_CORPUS_STATUS="PASS"
fi
CORPUS_SHA256="$(python3 - "$CIRCUIT_CORPUS_PATH" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"kind":"moonlab_oracle","name":"corpus_artifacts_validated","value":"%s","seed":%s,"snippet":"circuit_corpus artifact consumed: %s","artifacts":["%s"],"artifact_sha256":"%s","confidence":0.95,"git_head":"%s","git_tree":"%s","dirty":%s,"source_fingerprint":"%s","generated_at":"%s"}\n' \
    "$ORACLE_CORPUS_STATUS" "$SEED" "$CIRCUIT_CORPUS_REL_PATH" "$CIRCUIT_CORPUS_REL_PATH" "$CORPUS_SHA256" \
    "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" "$generated_at" \
    >> "$TRACE_FILE"

echo "   wrote $(wc -l < "$TRACE_FILE" | tr -d ' ') events to $TRACE_FILE"

if [ "$overall_fail" -ne 0 ]; then
    echo "run_moonlab_oracles.sh: at least one pillar has a non-allowlisted FAIL" >&2
    exit 1
fi
echo "run_moonlab_oracles.sh: all pillars green (modulo quarantined XFAILs)"
exit 0
