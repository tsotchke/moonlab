#!/bin/bash
# Run all tests for moonlab quantum simulator
# Usage: ./scripts/run_all_tests.sh [--verbose]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

VERBOSE=""
if [[ "$1" == "--verbose" ]]; then
    VERBOSE="1"
fi

echo "========================================"
echo "  Moonlab Quantum Simulator Test Suite"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local name="$1"
    local command="$2"

    echo -n "Running $name... "

    if [[ -n "$VERBOSE" ]]; then
        echo ""
        if eval "$command"; then
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${RED}FAILED${NC}"
            ((FAILED++))
        fi
    else
        if eval "$command" > /dev/null 2>&1; then
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${RED}FAILED${NC}"
            ((FAILED++))
        fi
    fi
}

skip_test() {
    local name="$1"
    local reason="$2"
    echo -e "Skipping $name... ${YELLOW}SKIPPED${NC} ($reason)"
    ((SKIPPED++))
}

echo "=== Building Tests ==="
make tests unit_tests 2>&1 | tail -5

echo ""
echo "=== Unit Tests ==="

# Unit tests
if [[ -f "tests/unit/test_quantum_state" ]]; then
    run_test "Quantum State Tests" "./tests/unit/test_quantum_state"
else
    skip_test "Quantum State Tests" "not built"
fi

if [[ -f "tests/unit/test_quantum_gates" ]]; then
    run_test "Quantum Gates Tests" "./tests/unit/test_quantum_gates"
else
    skip_test "Quantum Gates Tests" "not built"
fi

if [[ -f "tests/unit/test_memory_align" ]]; then
    run_test "Memory Alignment Tests" "./tests/unit/test_memory_align"
else
    skip_test "Memory Alignment Tests" "not built"
fi

if [[ -f "tests/unit/test_simd_dispatch" ]]; then
    run_test "SIMD Dispatch Tests" "./tests/unit/test_simd_dispatch"
else
    skip_test "SIMD Dispatch Tests" "not built"
fi

if [[ -f "tests/unit/test_stride_gates" ]]; then
    run_test "Stride Gates Tests" "./tests/unit/test_stride_gates"
else
    skip_test "Stride Gates Tests" "not built"
fi

if [[ -f "tests/unit/test_tensor_network" ]]; then
    run_test "Tensor Network Tests" "./tests/unit/test_tensor_network"
else
    skip_test "Tensor Network Tests" "not built"
fi

echo ""
echo "=== Integration Tests ==="

if [[ -f "tests/quantum_sim_test" ]]; then
    run_test "Quantum Sim Integration" "./tests/quantum_sim_test"
else
    skip_test "Quantum Sim Integration" "not built"
fi

if [[ -f "tests/test_comprehensive" ]]; then
    run_test "Comprehensive Test" "./tests/test_comprehensive"
else
    skip_test "Comprehensive Test" "not built"
fi

echo ""
echo "========================================"
echo "  Test Results"
echo "========================================"
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
