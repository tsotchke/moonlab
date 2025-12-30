#!/bin/bash
# Test Coverage Report Generator for Moonlab Quantum Simulator
# Generates HTML coverage reports using lcov/genhtml

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     MOONLAB QUANTUM SIMULATOR - TEST COVERAGE REPORT      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Clean previous coverage data
echo "[1/6] Cleaning previous coverage data..."
find . -name "*.gcda" -delete
find . -name "*.gcno" -delete
rm -rf coverage_report
mkdir -p coverage_report

# Build with coverage instrumentation
echo "[2/6] Building with coverage instrumentation..."
make clean
make CFLAGS="$(make -s print-CFLAGS) --coverage" LDFLAGS="$(make -s print-LDFLAGS) --coverage" tests unit_tests

# Run all tests
echo "[3/6] Running all tests..."
echo ""
echo "=== Unit Tests ==="
make test_unit || true
echo ""
echo "=== Integration Tests ==="
make test || true

# Generate coverage data
echo ""
echo "[4/6] Generating coverage data..."
lcov --capture --directory . --output-file coverage.info --rc lcov_branch_coverage=1

# Remove system/external files from coverage
echo "[5/6] Filtering coverage data..."
lcov --remove coverage.info '/usr/*' --output-file coverage.info
lcov --remove coverage.info '*/tests/*' --output-file coverage.info
lcov --remove coverage.info '*/examples/*' --output-file coverage.info

# Generate HTML report
echo "[6/6] Generating HTML report..."
genhtml coverage.info --output-directory coverage_report --rc lcov_branch_coverage=1

# Calculate coverage statistics
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                 COVERAGE SUMMARY                          ║"
echo "╠═══════════════════════════════════════════════════════════╣"

# Extract coverage percentages
TOTAL_LINES=$(lcov --summary coverage.info 2>&1 | grep "lines" | awk '{print $2}' | tr -d '%')
TOTAL_FUNCS=$(lcov --summary coverage.info 2>&1 | grep "functions" | awk '{print $2}' | tr -d '%')

echo "║  Line Coverage:     ${TOTAL_LINES}%"
echo "║  Function Coverage: ${TOTAL_FUNCS}%"
echo "║"
echo "║  Report Location: ./coverage_report/index.html"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Open report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening coverage report in browser..."
    open coverage_report/index.html
fi

# Check if target coverage is met
TARGET_COVERAGE=80
if (( $(echo "$TOTAL_LINES >= $TARGET_COVERAGE" | bc -l) )); then
    echo "✓ SUCCESS: Coverage target of ${TARGET_COVERAGE}% achieved!"
    exit 0
else
    echo "⚠ WARNING: Coverage ${TOTAL_LINES}% is below target of ${TARGET_COVERAGE}%"
    exit 1
fi