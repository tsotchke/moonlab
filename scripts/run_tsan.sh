#!/usr/bin/env bash
#
# run_tsan.sh -- driver for the Moonlab data-race / concurrency lane.
#
# Builds two TSan-instrumented static libquantumsim.a (OpenMP OFF for a clean
# pthread-only signal; OpenMP ON to exercise the intra-gate / fan-out parallel
# regions), compiles the checked-in harnesses under tests/concurrency via its
# self-contained CMakeLists, runs each under ThreadSanitizer, classifies the
# reports, and emits one JSONL trace event per subsystem plus an umbrella
# tsan_clean event to scripts/icc_traces/moonlab_tsan.jsonl.
#
# It never edits src/, cmake/tests.cmake, CMakeLists.txt, or ci.yml -- the
# TSan flags are injected into private, throwaway build dirs (build-tsan/,
# build-tsan-omp/, build-tsan-conc/).
#
# Exit code: 0 if the SHOULD-BE-CLEAN surface is race-free (regardless of the
# diagnostic findings, which are expected); nonzero if a clean-expected
# harness regressed into a race.
#
# Env:
#   QSIM_TSAN_JOBS=<n>     parallel build jobs (default 2)
#   QSIM_TSAN_FORCE=1      rebuild the libraries even if present
#   QSIM_TSAN_NO_OMP=1     skip the OpenMP library + conc_grover_gates
#
# Requires Clang (TSan / -fsanitize=thread).  If unavailable, prints a notice
# and points at the Linux CI job in .github/workflows/tsan.yml.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JOBS="${QSIM_TSAN_JOBS:-2}"
CC_BIN="${CC:-clang}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_tsan.jsonl"
LOG_DIR="$REPO_ROOT/build-tsan-conc/logs"
LIB_OFF="$REPO_ROOT/build-tsan/libquantumsim.a"
LIB_ON="$REPO_ROOT/build-tsan-omp/libquantumsim.a"
SUPP="$REPO_ROOT/tests/concurrency/tsan.supp"

tsan_options_for_run() { # $1=suppression enabled (0|1)
    local opts="halt_on_error=0"
    [ "$1" = "1" ] && opts="$opts:suppressions=$SUPP"
    printf '%s\n' "$opts"
}

run_result_is_clean() { # $1=unique race sites, $2=process exit status
    [ "$1" -eq 0 ] && [ "$2" -eq 0 ]
}

# The OpenMP grover harness needs a TSan-instrumented (Archer) libomp. On a
# host whose libomp is stock (not Archer-aware), the binary is killed by a
# fatal signal during OpenMP+TSan initialisation before producing any output
# (SIGILL/132 on Ubuntu clang + system libomp). That is a host toolchain gap,
# not a moonlab race: detect it (exit >= 128 with an empty run log) so grover
# can be skipped rather than failing the whole lane. Grover still runs where an
# Archer libomp is present (macOS Homebrew, CI's openmp-archer job).
grover_omp_runtime_broken() { # $1=process exit status  $2=run log path
    [ "$1" -ge 128 ] && [ ! -s "$2" ]
}

# Side-effect-free probes used by the focused producer-contract regression.
# They intentionally run before trace truncation, source capture, or toolchain
# discovery so the test does not need ThreadSanitizer.
case "${1:-}" in
    --internal-print-suppression-options)
        tsan_options_for_run 1
        exit 0
        ;;
    --internal-classify-run-result)
        if [ "$#" -ne 3 ] || ! [[ "$2" =~ ^[0-9]+$ && "$3" =~ ^[0-9]+$ ]]; then
            echo "usage: $0 --internal-classify-run-result RACES EXIT_STATUS" >&2
            exit 2
        fi
        if run_result_is_clean "$2" "$3"; then
            echo PASS
            exit 0
        fi
        echo FAIL
        exit 1
        ;;
esac

resolve_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

require_build_path() {
    local resolved
    resolved="$(resolve_path "$1")" || return 1
    case "$resolved" in
        "$REPO_ROOT"/build-*) printf '%s\n' "$resolved" ;;
        *) printf 'TSan writable path must resolve under %s/build-*: %s\n' "$REPO_ROOT" "$resolved" >&2; return 1 ;;
    esac
}

TRACE_DIR="$(resolve_path "$TRACE_DIR")" || exit 2
if [ "$TRACE_DIR" != "$REPO_ROOT/scripts/icc_traces" ]; then
    echo "MOONLAB_TRACE_DIR must resolve to $REPO_ROOT/scripts/icc_traces" >&2
    exit 2
fi
TRACE="$TRACE_DIR/moonlab_tsan.jsonl"
LOG_DIR="$(require_build_path "$LOG_DIR")" || exit 2
case "$JOBS" in ''|*[!0-9]*|0) echo "QSIM_TSAN_JOBS must be a positive integer" >&2; exit 2 ;; esac

mkdir -p "$TRACE_DIR" "$LOG_DIR"
python3 - "$TRACE" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).write_text("", encoding="utf-8")
PY

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

sha256_file() {
    python3 - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

write_hash_manifest() { # manifest path, followed by artifact paths
    python3 - "$@" <<'PY'
import hashlib
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
rows = []
for raw in sys.argv[2:]:
    path = Path(raw)
    if path.is_file():
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
payload = ("\n".join(sorted(rows)) + "\n").encode()
manifest.write_bytes(payload)
print(hashlib.sha256(payload).hexdigest())
PY
}

DIAGNOSTICS_SHA256=""

emit() {  # name verdict races snippet confidence [extra_json_object] [artifact]
    local generated_at extra_json artifact_path
    generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    extra_json="${6:-{}}"
    artifact_path="${7:-}"
    python3 - "$TRACE" "$1" "$2" "$3" "$4" "$5" "$extra_json" "$generated_at" \
        "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" "$SOURCE_DIRTY" "$SOURCE_FINGERPRINT" \
        "$DIAGNOSTICS_SHA256" "$LOG_DIR" "$artifact_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    trace, name, verdict, races, snippet, confidence, extra_json, generated_at,
    git_head, git_tree, dirty, source_fingerprint, diagnostics_sha256, log_dir,
    artifact_path,
) = sys.argv[1:]
event = {
    "kind": "moonlab_tsan",
    "name": name,
    "value": verdict,
    "status": verdict,
    "races": int(races),
    "snippet": snippet,
    "confidence": float(confidence),
    "generated_at": generated_at,
    "git_head": git_head,
    "git_tree": git_tree,
    "dirty": dirty == "true",
    "source_fingerprint": source_fingerprint,
    "build_dir": str(Path(log_dir).parent),
}
if diagnostics_sha256:
    event["diagnostics_manifest_sha256"] = diagnostics_sha256
if artifact_path and Path(artifact_path).is_file():
    event["artifact_path"] = artifact_path
    event["artifact_sha256"] = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
event.update(json.loads(extra_json))
with open(trace, "a", encoding="utf-8") as output:
    output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

# --- 0a. Windows gate --------------------------------------------------------
# The lane is POSIX-only (pthreads / unistd / sched_yield) and TSan targets
# POSIX threads, so it is gated off Windows -- mirrors the NOT
# QSIM_PLATFORM_WINDOWS gate on the adversarial suites in cmake/tests.cmake.
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        echo "[tsan] concurrency lane is POSIX-only; gated off Windows."
        emit "tsan_clean" "SKIP" 0 "POSIX-only lane, gated off Windows" 0.2
        exit 0 ;;
esac

# --- 0. TSan availability probe ---------------------------------------------
probe_c="$(mktemp -t tsanprobe.XXXXXX).c"
printf 'int main(void){return 0;}\n' > "$probe_c"
if ! "$CC_BIN" -fsanitize=thread "$probe_c" -o "${probe_c%.c}.bin" >/dev/null 2>&1; then
    echo "[tsan] ThreadSanitizer is NOT available in '$CC_BIN' on this host."
    echo "[tsan] Falling back to the documented Linux-CI TSan job:"
    echo "       .github/workflows/tsan.yml (clang + Archer-instrumented libomp)."
    rm -f "$probe_c" "${probe_c%.c}.bin"
    emit "tsan_clean" "SKIP" 0 "ThreadSanitizer unavailable on this host; see .github/workflows/tsan.yml" 0.2
    exit 0
fi
rm -f "$probe_c" "${probe_c%.c}.bin"
echo "[tsan] using $("$CC_BIN" --version | head -1)"

# --- 1. Build the TSan-instrumented libraries -------------------------------
build_lib() {  # $1=build_dir  $2=openmp(ON|OFF)
    local dir="$1" omp="$2"
    # A stale libquantumsim.a would make this evidence test PRE-FIX code and lie
    # (a fixed race reported as still-racy, or a new race hidden by an old-clean
    # archive). Never short-circuit on the archive's mere existence. Configure
    # once (cache reused), then ALWAYS run the incremental build so CMake's own
    # dependency tracking rebuilds any object whose source changed since the last
    # run. QSIM_TSAN_FORCE=1 forces a clean reconfigure from scratch.
    if [ "${QSIM_TSAN_FORCE:-0}" = "1" ]; then
        rm -rf "$dir"
    fi
    if [ ! -f "$dir/CMakeCache.txt" ]; then
        echo "[tsan] configuring $dir (OpenMP=$omp) ..."
        CC="$CC_BIN" CXX="${CXX:-clang++}" cmake -S . -B "$dir" -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=Debug \
            -DQSIM_BUILD_STATIC=ON -DQSIM_BUILD_SHARED=OFF \
            -DQSIM_BUILD_TESTS=OFF -DQSIM_BUILD_EXAMPLES=OFF \
            -DQSIM_BUILD_BENCHMARKS=OFF -DQSIM_BUILD_VISUALIZATION=OFF \
            -DQSIM_ENABLE_OPENMP="$omp" -DQSIM_ENABLE_METAL=OFF -DQSIM_WERROR=OFF \
            -DQSIM_ENABLE_TLS=OFF -DQSIM_ENABLE_CONTROL_PLANE=ON \
            -DQSIM_ENABLE_LTO=OFF -DQSIM_NATIVE_ARCH=OFF \
            -DCMAKE_C_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer" \
            -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer" \
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
            -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread" >"$LOG_DIR/cfg_$(basename "$dir").log" 2>&1 \
            || { echo "[tsan] configure failed for $dir; see $LOG_DIR/cfg_$(basename "$dir").log"; return 1; }
    fi
    echo "[tsan] building $dir/libquantumsim.a -j$JOBS (incremental) ..."
    cmake --build "$dir" --target quantumsim -j"$JOBS" >"$LOG_DIR/build_$(basename "$dir").log" 2>&1 \
        || { echo "[tsan] build failed for $dir; see $LOG_DIR/build_$(basename "$dir").log"; return 1; }
}

build_lib build-tsan OFF || exit 2
WITH_OMP=1
if [ "${QSIM_TSAN_NO_OMP:-0}" = "1" ]; then
    WITH_OMP=0
elif ! build_lib build-tsan-omp ON; then
    echo "[tsan] OpenMP library build failed; continuing without conc_grover_gates."
    WITH_OMP=0
fi

# --- 2. Configure + build the harnesses -------------------------------------
CONC_CMAKE_ARGS=(
    -S tests/concurrency -B build-tsan-conc -G "Unix Makefiles"
    -DMOONLAB_ROOT="$REPO_ROOT"
    -DMOONLAB_TSAN_LIB="$LIB_OFF"
)
if [ "$WITH_OMP" = "1" ]; then
    CONC_CMAKE_ARGS+=( -DMOONLAB_TSAN_LIB_OMP="$LIB_ON" )
fi
# On non-macOS the static lib needs its BLAS/LAPACK backend at harness link
# time (Accelerate is auto-linked on Apple).  Match the CI toolchain
# (libopenblas-dev liblapacke-dev); override with QSIM_TSAN_EXTRA_LIBS.
if [ "$(uname -s)" != "Darwin" ]; then
    CONC_CMAKE_ARGS+=( -DMOONLAB_EXTRA_LIBS="${QSIM_TSAN_EXTRA_LIBS:-openblas;lapack;lapacke}" )
fi
CC="$CC_BIN" CXX="${CXX:-clang++}" cmake "${CONC_CMAKE_ARGS[@]}" \
    >"$LOG_DIR/cfg_conc.log" 2>&1 || { echo "[tsan] harness configure failed; see $LOG_DIR/cfg_conc.log"; exit 2; }
cmake --build build-tsan-conc -j"$JOBS" \
    >"$LOG_DIR/build_conc.log" 2>&1 || { echo "[tsan] harness build failed; see $LOG_DIR/build_conc.log"; exit 2; }

BIN=build-tsan-conc

# --- 3. Run harnesses + classify --------------------------------------------
# distinct_races <logfile>: count unique "SUMMARY: data race" sites.
distinct_races() { grep "SUMMARY: ThreadSanitizer: data race" "$1" 2>/dev/null | sort -u | wc -l | tr -d ' '; }

json_escape() { sed 's/\\/\\\\/g; s/"/\\"/g' <<<"$1"; }

# The authoritative emit() is defined once near the top of this script: it binds
# every event to git_head/git_tree/dirty/source_fingerprint so the release
# smoke's check_deep_hunt relay can verify the tsan lane ran on the exact source
# tree. A second, provenance-less emit() used to shadow it here, which made every
# relay fail with git-tree-mismatch; it has been removed so all calls carry
# provenance.

run_one() {  # $1=name $2=supp(0|1) $3=binary $4...=args -> echoes "races exit"
    local name="$1" supp="$2" bin="$3"; shift 3
    local log="$LOG_DIR/run_${name}.log"
    local opts run_rc races
    opts="$(tsan_options_for_run "$supp")"
    TSAN_OPTIONS="$opts" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
        "$BIN/$bin" "$@" >"$log" 2>&1
    run_rc=$?
    printf '%s\n' "$run_rc" > "$log.exit"
    races="$(distinct_races "$log")"
    printf '%s %s\n' "$races" "$run_rc"
}

echo "[tsan] running harnesses ..."
FAILS=0
REAL_RACES=0

# ---- Clean-expected surface (must be race-free) ----
for spec in \
    "control_plane_steady|0|conc_control_plane|steady" \
    "entropy_pool_steady|0|conc_entropy_pool|steady" \
    "entropy_pool_toggle|0|conc_entropy_pool|toggle" \
    "audit_buffer_mpmc|0|conc_audit_buffer|mpmc" \
    "scheduler|0|conc_scheduler|" \
    "clifford_measurement|0|conc_clifford_measurement|" ; do
    IFS='|' read -r nm supp binp arg <<<"$spec"
    # shellcheck disable=SC2086
    read -r r run_rc <<<"$(run_one "$nm" "$supp" "$binp" $arg)"
    if run_result_is_clean "$r" "$run_rc"; then
        emit "$nm" "PASS" 0 "clean under TSan (no data race)" 0.9
        echo "  [PASS] $nm  (0 races)"
    else
        emit "$nm" "FAIL" "$r" "UNEXPECTED race or harness failure in a clean-expected harness (exit=$run_rc)" 0.9
        echo "  [FAIL] $nm  ($r races, exit=$run_rc) <-- regression"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+r))
    fi
done

# OpenMP harness: clean once libomp-internal frames are suppressed.
if [ "$WITH_OMP" = "1" ]; then
    read -r r run_rc <<<"$(run_one "grover_gates_omp" 1 "conc_grover_gates" all)"
    read -r raw raw_rc <<<"$(run_one "grover_gates_omp_raw" 0 "conc_grover_gates" all)"
    # The unsuppressed diagnostic normally exits nonzero when it reports the
    # Homebrew libomp annotation gaps. It is valid only if it either completed
    # cleanly or actually emitted a race summary; a crash with zero summaries
    # must not turn into PASS.
    raw_completed=0
    if [ "$raw_rc" -eq 0 ] || [ "$raw" -gt 0 ]; then raw_completed=1; fi
    if grover_omp_runtime_broken "$run_rc" "$LOG_DIR/run_grover_gates_omp.log" \
       && grover_omp_runtime_broken "$raw_rc" "$LOG_DIR/run_grover_gates_omp_raw.log"; then
        emit "grover_gates_omp" "SKIP" 0 \
          "OpenMP+TSan runtime unavailable on this host: libomp is not TSan/Archer-instrumented, so the harness is killed by a signal during OpenMP+TSan init (exit=$run_rc, no output). Grover is exercised where Archer libomp is present (macOS Homebrew, CI openmp-archer)." 0.3
        echo "  [SKIP] grover_gates_omp  (OpenMP+TSan/Archer runtime not functional on this host; exit=$run_rc)"
    elif run_result_is_clean "$r" "$run_rc" && [ "$raw_completed" -eq 1 ]; then
        emit "grover_gates_omp" "PASS" 0 \
          "OpenMP core clean after excluding un-annotated libomp frames (raw=$raw libomp false positives); entropy isolation + disjoint fan-out verified" 0.75
        echo "  [PASS] grover_gates_omp  (0 real, $raw libomp-runtime false positives filtered; exit=$run_rc raw_exit=$raw_rc)"
    else
        emit "grover_gates_omp" "FAIL" "$r" "moonlab-owned race or harness failure survived libomp suppression (exit=$run_rc raw_exit=$raw_rc raw_races=$raw)" 0.75
        echo "  [FAIL] grover_gates_omp  ($r races, exit=$run_rc raw=$raw raw_exit=$raw_rc)"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+r))
    fi
else
    emit "grover_gates_omp" "SKIP" 0 "OpenMP library not built on this host" 0.3
fi

# ---- Regression checks for the four fixed concurrency bugs (v1.2) ----
# Each harness drives the exact race / deadlock window that once fired; a
# A nonzero race count or process exit means a REGRESSION, so PASS iff clean.
regress() {  # $1=name $2=races $3=exit status $4=snippet
    if run_result_is_clean "$2" "$3"; then
        emit "$1" "PASS" 0 "$4" 0.9
        echo "  [PASS] $1  (0 races: fix holds)"
    else
        emit "$1" "FAIL" "$2" "REGRESSION or harness failure (exit=$3) -- $4" 0.95
        echo "  [FAIL] $1  ($2 races, exit=$3) <-- regression"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+$2))
    fi
}

read -r r run_rc <<<"$(run_one "core_init_lazy_init" 0 "conc_core_init")"
regress "core_init_lazy_init" "$r" "$run_rc" \
  "cold-start lazy init of process globals is now race-free: qsim_config_global double-checked-locking (config.c) + simd_dispatch_init_once pthread_once (simd_ops.c)"

read -r r run_rc <<<"$(run_one "control_plane_config_fields" 0 "conc_control_plane" adversarial)"
regress "control_plane_config_fields" "$r" "$run_rc" \
  "server config setters/readers now synchronised: admission_hook+ctx pair and max_concurrent/request_timeout under cfg_lock, rate_rps atomic"

# Audit-buffer destroy vs in-flight push/pop: with the in_flight drain the
# destroy no longer wedges; watchdog prints DEADLOCK + exits 7 on a regression.
adlog="$LOG_DIR/run_audit_destroy.log"
TSAN_OPTIONS="halt_on_error=0" "$BIN/conc_audit_buffer" destroy >"$adlog" 2>&1
ad_exit=$?
ad_races="$(distinct_races "$adlog")"
ad_deadlock=0
if [ "$ad_exit" -eq 7 ] || grep -q "DEADLOCK:" "$adlog"; then ad_deadlock=1; fi
if [ "$ad_exit" -ne 0 ] || [ "$ad_races" -ne 0 ] || [ "$ad_deadlock" -ne 0 ]; then
    emit "audit_buffer_destroy_deadlock" "FAIL" 1 \
      "REGRESSION or harness failure -- destroy() vs in-flight push/pop (exit=$ad_exit races=$ad_races)" 0.95
    echo "  [FAIL] audit_buffer_destroy_deadlock  (exit=$ad_exit races=$ad_races) <-- regression"
    FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+ad_races+ad_deadlock))
else
    emit "audit_buffer_destroy_deadlock" "PASS" 0 \
      "destroy() drains in_flight before destroying the mutex; 16 rounds completed with no wedge" 0.9
    echo "  [PASS] audit_buffer_destroy_deadlock  (0 deadlocks: drain holds)"
fi

# --- 3b. Bind the lane's evidence by content --------------------------------
# Every other release lane emits a content hash so the release certificate can
# verify what was actually tested; the concurrency lane emitted none, which made
# validate_release_certificate.py reject it ("declared artifact hashes are not
# all content-verified"). Manifest the TSan-instrumented libraries plus the
# per-run logs and attach the digest to the umbrella verdict via the existing
# diagnostics_manifest_sha256 field.
DIAG_MANIFEST="$LOG_DIR/diagnostics-manifest.sha256"
DIAGNOSTICS_SHA256="$(write_hash_manifest "$DIAG_MANIFEST" \
    "$LIB_OFF" "$LIB_ON" \
    "$LOG_DIR"/run_*.log "$LOG_DIR"/run_*.log.exit "$LOG_DIR"/build_*.log 2>/dev/null || true)"

# --- 4. Umbrella verdict ----------------------------------------------------
if [ "$REAL_RACES" -eq 0 ] && [ "$FAILS" -eq 0 ]; then
    emit "tsan_clean" "PASS" 0 "no data races or deadlocks across the concurrency lane" 0.9
    echo "[tsan] RESULT: PASS (0 real races)"
    exit 0
fi
emit "tsan_clean" "FAIL" "$REAL_RACES" \
  "concurrency lane found race/deadlock sites or harness failures. Clean-surface failures: $FAILS" 0.9
echo "[tsan] RESULT: FAIL ($REAL_RACES real race/deadlock sites; $FAILS clean-surface regressions)"
echo "[tsan] trace -> $TRACE ; per-run logs -> $LOG_DIR ; findings -> tests/concurrency/FINDINGS.md"
# A clean-surface regression is a hard failure; the known diagnostic findings
# alone do not fail the script (they are the point of the lane).
[ "$FAILS" -gt 0 ] && exit 1
exit 0
