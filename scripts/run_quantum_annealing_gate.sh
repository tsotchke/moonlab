#!/usr/bin/env bash
# Clean-tree evidence producer for Moonlab v1.2.1 quantum annealing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${QSIM_ANNEAL_BUILD_DIR:-$REPO_ROOT/build-quantum-annealing}"
FUZZ_BUILD_DIR="${QSIM_ANNEAL_FUZZ_BUILD_DIR:-$REPO_ROOT/build-quantum-annealing-fuzz}"
TRACE="$REPO_ROOT/scripts/icc_traces/moonlab_quantum_annealing.jsonl"
JOBS="${QSIM_ANNEAL_JOBS:-4}"
PYTHON_BIN="${QSIM_ANNEAL_PYTHON:-python3}"

case "$BUILD_DIR" in "$REPO_ROOT"/build-*) ;; *) exit 2 ;; esac
case "$FUZZ_BUILD_DIR" in "$REPO_ROOT"/build-*) ;; *) exit 2 ;; esac
case "$JOBS" in 1|2|3|4) ;; *) echo "QSIM_ANNEAL_JOBS must be in [1,4]" >&2; exit 2 ;; esac

SOURCE_IDENTITY_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)"
IFS=$'\t' read -r SOURCE_GIT_HEAD SOURCE_GIT_TREE SOURCE_DIRTY SOURCE_FINGERPRINT \
  < <("$PYTHON_BIN" - "$SOURCE_IDENTITY_JSON" <<'PY'
import json, sys
x = json.loads(sys.argv[1])
print("\t".join((x["git_head"], x["git_tree"],
                 str(x["dirty"]).lower(), x["source_fingerprint"])))
PY
)
if [ "$SOURCE_DIRTY" != "false" ]; then
  echo "quantum-annealing evidence requires a clean worktree; no trace was changed" >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/scripts/icc_traces"
: > "$TRACE"
COMPLETE=0
emit() {
  "$PYTHON_BIN" - "$1" "$2" "$3" "$SOURCE_GIT_HEAD" "$SOURCE_GIT_TREE" \
    "$SOURCE_FINGERPRINT" >> "$TRACE" <<'PY'
import datetime, json, sys
name, status, detail, head, tree, fingerprint = sys.argv[1:]
print(json.dumps({
    "kind": "moonlab_quantum_annealing", "name": name,
    "status": status, "value": status, "detail": detail,
    "git_head": head, "git_tree": tree, "dirty": False,
    "source_fingerprint": fingerprint,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, sort_keys=True))
PY
}
on_exit() {
  status=$?
  trap - EXIT
  if [ "$COMPLETE" -eq 0 ]; then
    emit rk4_oracle FAIL "annealing gate aborted before completion (exit $status)"
  fi
  exit "$status"
}
trap on_exit EXIT

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=ON -DQSIM_BUILD_BENCHMARKS=OFF
cmake --build "$BUILD_DIR" -j"$JOBS" --target \
  test_quantum_annealing test_moonlab_export_abi quantum_annealing_demo
ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R '^(unit_quantum_annealing|abi_moonlab_export|example_quantum_annealing)$'
emit rk4_oracle PASS "second-order anneal agrees with independent 100000-step RK4 oracle"
emit qubo_ising_parity PASS "asymmetric full-matrix QUBO parity holds on every three-bit state"
emit abi_080 PASS "ABI 0.8.0 Ising and QUBO one-shots resolve and pass analytic smoke"

MOONLAB_LIB_DIR="$BUILD_DIR" PYTHONPATH="$REPO_ROOT/bindings/python" \
  "$PYTHON_BIN" -m pytest bindings/python/tests/test_annealing.py -q
MOONLAB_LIB_DIR="$BUILD_DIR" MOONLAB_INCLUDE_DIR="$REPO_ROOT/src" \
DYLD_LIBRARY_PATH="$BUILD_DIR" LD_LIBRARY_PATH="$BUILD_DIR" \
  cargo test --manifest-path bindings/rust/Cargo.toml -p moonlab annealing
pnpm --dir bindings/javascript/packages/core build:ts
QSIM_WASM_JOBS="$JOBS" pnpm --dir bindings/javascript/packages/core build:wasm
pnpm --dir bindings/javascript/packages/core test -- \
  --run src/__tests__/annealing.test.ts
pnpm --dir bindings/javascript/packages/core exec vitest run \
  --config vitest.integration.config.ts \
  src/__tests__/annealing.integration.test.ts
emit binding_parity PASS "Python, Rust, and JavaScript annealing suites pass"

cmake -S "$REPO_ROOT" -B "$FUZZ_BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug \
  -DQSIM_ENABLE_FUZZING=ON -DQSIM_BUILD_TESTS=OFF \
  -DQSIM_BUILD_EXAMPLES=OFF -DQSIM_BUILD_BENCHMARKS=OFF \
  -DQSIM_WERROR=OFF -DQSIM_ENABLE_OPENMP=OFF -DQSIM_ENABLE_METAL=OFF
cmake --build "$FUZZ_BUILD_DIR" -j"$JOBS" --target abi_boundary_fuzz_replay
ASAN_OPTIONS=abort_on_error=1:detect_leaks=0:print_summary=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
DYLD_LIBRARY_PATH="$FUZZ_BUILD_DIR" LD_LIBRARY_PATH="$FUZZ_BUILD_DIR" \
  "$FUZZ_BUILD_DIR/tests/fuzz/abi_boundary_fuzz_replay" \
  "$REPO_ROOT/tests/fuzz/corpora/abi_boundary_fuzz/anneal.bin"
emit asan_ubsan_clean PASS "bounded annealing ABI fuzz seed clean under ASan/UBSan"

SOURCE_END_JSON="$(bash "$REPO_ROOT/scripts/run_moonlab_release_smoke.sh" --source-identity)"
SOURCE_END_FINGERPRINT="$("$PYTHON_BIN" - "$SOURCE_END_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["source_fingerprint"])
PY
)"
if [ "$SOURCE_END_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]; then
  echo "source changed during quantum-annealing gate" >&2
  exit 2
fi
COMPLETE=1
