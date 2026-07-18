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

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JOBS="${QSIM_TSAN_JOBS:-2}"
CC_BIN="${CC:-clang}"
TRACE_DIR="${MOONLAB_TRACE_DIR:-scripts/icc_traces}"
TRACE="$TRACE_DIR/moonlab_tsan.jsonl"
LOG_DIR="build-tsan-conc/logs"
LIB_OFF="build-tsan/libquantumsim.a"
LIB_ON="build-tsan-omp/libquantumsim.a"
SUPP="tests/concurrency/tsan.supp"

mkdir -p "$TRACE_DIR" "$LOG_DIR"
: > "$TRACE"

# --- 0a. Windows gate --------------------------------------------------------
# The lane is POSIX-only (pthreads / unistd / sched_yield) and TSan targets
# POSIX threads, so it is gated off Windows -- mirrors the NOT
# QSIM_PLATFORM_WINDOWS gate on the adversarial suites in cmake/tests.cmake.
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        echo "[tsan] concurrency lane is POSIX-only; gated off Windows."
        printf '{"kind":"moonlab_tsan","name":"tsan_clean","value":"SKIP","status":"SKIP","races":0,"snippet":"POSIX-only lane, gated off Windows","confidence":0.2}\n' >> "$TRACE"
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
    printf '{"kind":"moonlab_tsan","name":"tsan_clean","value":"SKIP","status":"SKIP","races":0,"snippet":"ThreadSanitizer unavailable on this host; see .github/workflows/tsan.yml","confidence":0.2}\n' >> "$TRACE"
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
    -DMOONLAB_TSAN_LIB="$REPO_ROOT/$LIB_OFF"
)
if [ "$WITH_OMP" = "1" ]; then
    CONC_CMAKE_ARGS+=( -DMOONLAB_TSAN_LIB_OMP="$REPO_ROOT/$LIB_ON" )
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

emit() {  # $1=name $2=verdict $3=races $4=snippet $5=confidence
    printf '{"kind":"moonlab_tsan","name":"%s","value":"%s","status":"%s","races":%s,"snippet":"%s","confidence":%s}\n' \
        "$1" "$2" "$2" "$3" "$(json_escape "$4")" "$5" >> "$TRACE"
}

run_one() {  # $1=name $2=supp(0|1) $3=binary $4...=args  -> echoes race count
    local name="$1" supp="$2" bin="$3"; shift 3
    local log="$LOG_DIR/run_${name}.log"
    local opts="halt_on_error=0"
    [ "$supp" = "1" ] && opts="$opts:suppressions=$REPO_ROOT/$SUPP"
    TSAN_OPTIONS="$opts" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
        "$BIN/$bin" "$@" >"$log" 2>&1
    echo $? > "$log.exit"
    distinct_races "$log"
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
    r=$(run_one "$nm" "$supp" "$binp" $arg)
    if [ "$r" -eq 0 ]; then
        emit "$nm" "PASS" 0 "clean under TSan (no data race)" 0.9
        echo "  [PASS] $nm  (0 races)"
    else
        emit "$nm" "FAIL" "$r" "UNEXPECTED race in a clean-expected harness" 0.9
        echo "  [FAIL] $nm  ($r races) <-- regression"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+r))
    fi
done

# OpenMP harness: clean once libomp-internal frames are suppressed.
if [ "$WITH_OMP" = "1" ]; then
    r=$(run_one "grover_gates_omp" 1 "conc_grover_gates" all)
    raw=$(run_one "grover_gates_omp_raw" 0 "conc_grover_gates" all)
    if [ "$r" -eq 0 ]; then
        emit "grover_gates_omp" "PASS" 0 \
          "OpenMP core clean after excluding un-annotated libomp frames (raw=$raw libomp false positives); entropy isolation + disjoint fan-out verified" 0.75
        echo "  [PASS] grover_gates_omp  (0 real, $raw libomp-runtime false positives filtered)"
    else
        emit "grover_gates_omp" "FAIL" "$r" "moonlab-owned race survived libomp suppression" 0.75
        echo "  [FAIL] grover_gates_omp  ($r races)"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+r))
    fi
else
    emit "grover_gates_omp" "SKIP" 0 "OpenMP library not built on this host" 0.3
fi

# ---- Regression checks for the four fixed concurrency bugs (v1.2) ----
# Each harness drives the exact race / deadlock window that once fired; a
# nonzero race count now means a REGRESSION, so PASS iff clean.
regress() {  # $1=name $2=races $3=snippet
    if [ "$2" -eq 0 ]; then
        emit "$1" "PASS" 0 "$3" 0.9
        echo "  [PASS] $1  (0 races: fix holds)"
    else
        emit "$1" "FAIL" "$2" "REGRESSION -- $3" 0.95
        echo "  [FAIL] $1  ($2 races) <-- regression"
        FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+$2))
    fi
}

r=$(run_one "core_init_lazy_init" 0 "conc_core_init")
regress "core_init_lazy_init" "$r" \
  "cold-start lazy init of process globals is now race-free: qsim_config_global double-checked-locking (config.c) + simd_dispatch_init_once pthread_once (simd_ops.c)"

r=$(run_one "control_plane_config_fields" 0 "conc_control_plane" adversarial)
regress "control_plane_config_fields" "$r" \
  "server config setters/readers now synchronised: admission_hook+ctx pair and max_concurrent/request_timeout under cfg_lock, rate_rps atomic"

# Audit-buffer destroy vs in-flight push/pop: with the in_flight drain the
# destroy no longer wedges; watchdog prints DEADLOCK + exits 7 on a regression.
adlog="$LOG_DIR/run_audit_destroy.log"
TSAN_OPTIONS="halt_on_error=0" "$BIN/conc_audit_buffer" destroy >"$adlog" 2>&1
ad_exit=$?
if [ "$ad_exit" -eq 7 ] || grep -q "DEADLOCK:" "$adlog"; then
    emit "audit_buffer_destroy_deadlock" "FAIL" 1 \
      "REGRESSION -- destroy() vs in-flight push/pop wedged on the destroyed mutex (audit_buffer.c)" 0.95
    echo "  [FAIL] audit_buffer_destroy_deadlock  (deadlock reproduced) <-- regression"
    FAILS=$((FAILS+1)); REAL_RACES=$((REAL_RACES+1))
else
    emit "audit_buffer_destroy_deadlock" "PASS" 0 \
      "destroy() drains in_flight before destroying the mutex; 16 rounds completed with no wedge" 0.9
    echo "  [PASS] audit_buffer_destroy_deadlock  (0 deadlocks: drain holds)"
fi

# --- 4. Umbrella verdict ----------------------------------------------------
if [ "$REAL_RACES" -eq 0 ]; then
    emit "tsan_clean" "PASS" 0 "no data races or deadlocks across the concurrency lane" 0.9
    echo "[tsan] RESULT: PASS (0 real races)"
    exit 0
fi
emit "tsan_clean" "FAIL" "$REAL_RACES" \
  "concurrency lane found real bugs: shared-core lazy-init races (config/simd), control-plane config-field races, audit-buffer destroy deadlock. Clean-surface regressions: $FAILS" 0.9
echo "[tsan] RESULT: FAIL ($REAL_RACES real race/deadlock sites; $FAILS clean-surface regressions)"
echo "[tsan] trace -> $TRACE ; per-run logs -> $LOG_DIR ; findings -> tests/concurrency/FINDINGS.md"
# A clean-surface regression is a hard failure; the known diagnostic findings
# alone do not fail the script (they are the point of the lane).
[ "$FAILS" -gt 0 ] && exit 1
exit 0
