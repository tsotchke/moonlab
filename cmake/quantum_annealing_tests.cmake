# Quantum-annealing v1.2.1 tests live separately because cmake/tests.cmake may
# carry an operator's in-flight noise-marginal work.  Keep this include
# additive until that work is integrated.
add_executable(test_quantum_annealing tests/unit/test_quantum_annealing.c)
target_link_libraries(test_quantum_annealing PRIVATE quantumsim ${MATH_LIBRARY})
add_test(NAME unit_quantum_annealing COMMAND test_quantum_annealing)
set_tests_properties(unit_quantum_annealing PROPERTIES
    LABELS "algorithms;quantum_annealing"
    TIMEOUT 120)

if(QSIM_BUILD_BENCHMARKS)
    add_executable(bench_quantum_annealing
        benchmarks/quantum_annealing_bench.c)
    target_link_libraries(bench_quantum_annealing PRIVATE
        quantumsim ${MATH_LIBRARY})
endif()
