# cmake/tests.cmake — tests + ctest registrations for libquantumsim.
#
# Included from the root CMakeLists.txt at the body of the
#   if(QSIM_BUILD_TESTS)
#       include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
#   endif()
# block.  Uses include() rather than add_subdirectory() so the file
# runs in the same scope and continues to read variables (QSIM_HAS_*,
# QSIM_OPENMP_LIBRARIES, ...) and macros (qsim_target_link_openmp,
# qsim_label_tests) defined at the top of the root file.
#
# Extracted from the root CMakeLists.txt as part of the v0.2.x
# architectural cleanup (audit task #128).  Keeping the test target
# definitions in their own file makes the root CMakeLists.txt easier
# to navigate; before extraction it was 2549 lines, of which ~990 was
# this block.

    enable_testing()

    # Helper: append one or more labels to a list of named tests.  Wraps
    # the get/set dance so the test-grouping section at the bottom of
    # this block stays readable.  Labels stack; e.g. unit_kagome_ed_large
    # may be both "long" and "algorithms".
    function(qsim_label_tests label)
        foreach(t IN LISTS ARGN)
            if(TEST ${t})
                get_test_property(${t} LABELS existing)
                if(existing STREQUAL "NOTFOUND")
                    set_tests_properties(${t} PROPERTIES LABELS "${label}")
                else()
                    set_tests_properties(${t} PROPERTIES
                                          LABELS "${existing};${label}")
                endif()
            endif()
        endforeach()
    endfunction()

    # Main test executable
    add_executable(qsim_test tests/quantum_sim_test.c)
    target_link_libraries(qsim_test PRIVATE quantumsim)
    add_test(NAME quantum_sim_test COMMAND qsim_test)
    set_tests_properties(quantum_sim_test PROPERTIES LABELS "long")

    # Health tests
    add_executable(health_tests_test tests/health_tests_test.c)
    target_link_libraries(health_tests_test PRIVATE quantumsim)
    add_test(NAME health_tests COMMAND health_tests_test)

    # Bell test demo
    add_executable(bell_test_demo tests/bell_test_demo.c)
    target_link_libraries(bell_test_demo PRIVATE quantumsim)
    add_test(NAME bell_test COMMAND bell_test_demo)

    # Gate test
    add_executable(gate_test tests/gate_test.c)
    target_link_libraries(gate_test PRIVATE quantumsim)
    add_test(NAME gate_test COMMAND gate_test)

    # Correlation test
    add_executable(correlation_test tests/correlation_test.c)
    target_link_libraries(correlation_test PRIVATE quantumsim)
    add_test(NAME correlation_test COMMAND correlation_test)

    # Unit tests
    add_executable(test_quantum_state tests/unit/test_quantum_state.c)
    target_link_libraries(test_quantum_state PRIVATE quantumsim)
    add_test(NAME unit_quantum_state COMMAND test_quantum_state)

    add_executable(test_constants tests/unit/test_constants.c)
    target_link_libraries(test_constants PRIVATE ${MATH_LIBRARY})
    add_test(NAME unit_constants COMMAND test_constants)

    add_executable(test_correctness_properties tests/unit/test_correctness_properties.c)
    target_link_libraries(test_correctness_properties PRIVATE quantumsim)
    add_test(NAME unit_correctness_properties COMMAND test_correctness_properties)

    add_executable(test_quantum_gates tests/unit/test_quantum_gates.c)
    target_link_libraries(test_quantum_gates PRIVATE quantumsim)
    add_test(NAME unit_quantum_gates COMMAND test_quantum_gates)

    add_executable(test_memory_align tests/unit/test_memory_align.c
                   src/optimization/memory_align.c)
    target_include_directories(test_memory_align PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_memory_align PRIVATE ${MATH_LIBRARY} Threads::Threads ${QSIM_CLANG_BUILTINS})
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_memory_align PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_memory_align PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_memory_align COMMAND test_memory_align)

    add_executable(test_simd_dispatch tests/unit/test_simd_dispatch.c
                   src/optimization/simd_dispatch.c)
    target_include_directories(test_simd_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_simd_dispatch PRIVATE ${MATH_LIBRARY} Threads::Threads ${QSIM_CLANG_BUILTINS})
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_simd_dispatch PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(test_simd_dispatch PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_simd_dispatch COMMAND test_simd_dispatch)

    # Tensor-network unit test — the 800+ line suite that exercises
    # tensors, SVD, MPS, gate application via tensor networks,
    # measurement, and entanglement. Already in-tree but historically
    # not wired into ctest.
    add_executable(test_tensor_network tests/unit/test_tensor_network.c)
    target_link_libraries(test_tensor_network PRIVATE quantumsim)
    add_test(NAME unit_tensor_network COMMAND test_tensor_network)

    # Comprehensive end-to-end suite: gates, entanglement (von Neumann /
    # Renyi / concurrence), amplitude/phase damping noise, Grover, MPS.
    add_executable(test_comprehensive tests/test_comprehensive.c)
    target_link_libraries(test_comprehensive PRIVATE quantumsim)
    add_test(NAME comprehensive COMMAND test_comprehensive)

    # DMRG ground-state solver regression test.
    add_executable(test_dmrg tests/test_dmrg.c)
    target_link_libraries(test_dmrg PRIVATE quantumsim)
    add_test(NAME dmrg COMMAND test_dmrg)

    # MPS vs exact state-vector drift test — catches divergences as
    # bond dimension or system size grows.
    add_executable(test_mps_vs_exact tests/test_mps_vs_exact.c)
    target_link_libraries(test_mps_vs_exact PRIVATE quantumsim)
    add_test(NAME mps_vs_exact COMMAND test_mps_vs_exact)

    # Time-evolution long-run regression.
    add_executable(test_long_evolution tests/test_long_evolution.c)
    target_link_libraries(test_long_evolution PRIVATE quantumsim)
    add_test(NAME long_evolution COMMAND test_long_evolution)
    set_tests_properties(long_evolution PROPERTIES LABELS "long")

    # Fast-measurement correctness test.
    add_executable(test_fast_measurement tests/test_fast_measurement.c)
    target_link_libraries(test_fast_measurement PRIVATE quantumsim)
    add_test(NAME fast_measurement COMMAND test_fast_measurement)

    # Measurement-subsystem unit test: probability queries, projective
    # collapse, Bell correlations, expectation values.
    add_executable(test_measurement tests/unit/test_measurement.c)
    target_link_libraries(test_measurement PRIVATE quantumsim)
    add_test(NAME unit_measurement COMMAND test_measurement)

    # Entanglement unit test: product-state zero entropy, Bell-state
    # maximal entropy, fidelity, purity, partial trace = I/2.
    add_executable(test_entanglement tests/unit/test_entanglement.c)
    target_link_libraries(test_entanglement PRIVATE quantumsim)
    add_test(NAME unit_entanglement COMMAND test_entanglement)

    # Noise-channel unit test: involution of bit/phase flips, depolarizing
    # p=0 no-op, norm preservation across every channel.
    add_executable(test_noise tests/unit/test_noise.c)
    target_link_libraries(test_noise PRIVATE quantumsim)
    add_test(NAME unit_noise COMMAND test_noise)

    # Grover algorithm smoke: on 3/4/5 qubits, success probability on the
    # marked state must exceed 0.8 after pi/4 * sqrt(N) iterations.
    add_executable(test_grover tests/unit/test_grover.c)
    target_link_libraries(test_grover PRIVATE quantumsim)
    add_test(NAME unit_grover COMMAND test_grover)

    # VQE H2 ground-state smoke.
    add_executable(test_vqe tests/unit/test_vqe.c)
    target_link_libraries(test_vqe PRIVATE quantumsim)
    add_test(NAME unit_vqe COMMAND test_vqe)

    # QAOA MaxCut smoke.
    add_executable(test_qaoa tests/unit/test_qaoa.c)
    target_link_libraries(test_qaoa PRIVATE quantumsim)
    add_test(NAME unit_qaoa COMMAND test_qaoa)

    # QPE phase-recovery smoke.
    add_executable(test_qpe tests/unit/test_qpe.c)
    target_link_libraries(test_qpe PRIVATE quantumsim)
    add_test(NAME unit_qpe COMMAND test_qpe)

    # Clifford stabilizer backend: H2=I, X, S4=I, Bell, 100-qubit GHZ.
    add_executable(test_clifford_pauli_api tests/unit/test_clifford_pauli_api.c)
    target_link_libraries(test_clifford_pauli_api PRIVATE quantumsim)
    add_test(NAME unit_clifford_pauli_api COMMAND test_clifford_pauli_api)

    # Clifford-assisted MPS reduction-limit tests: pure Clifford (tableau-only
    # Bell/GHZ), pure MPS (rotations on |0>), and mixed H + T rotations.
    add_executable(test_ca_mps_limits tests/unit/test_ca_mps_limits.c)
    target_link_libraries(test_ca_mps_limits PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_limits COMMAND test_ca_mps_limits)

    # CA-PEPS row-major-MPS implementation: lifecycle + adjacency validation
    # + parity against the underlying CA-MPS for a mixed Clifford + rotation
    # circuit on a 3x3 lattice.
    add_executable(test_ca_peps tests/unit/test_ca_peps.c)
    target_link_libraries(test_ca_peps PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_peps COMMAND test_ca_peps)

    # Pauli-frame sampler: per-gate commutation rules + batched kernels +
    # depolarising channel rate.
    add_executable(test_pauli_frame tests/unit/test_pauli_frame.c)
    target_link_libraries(test_pauli_frame PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_pauli_frame COMMAND test_pauli_frame)

    # End-to-end CA-MPS correctness: random mixed Clifford + rx/ry/rz/CZ/CNOT
    # circuits compared against the dense state-vector backend for every
    # single-qubit Pauli expectation and a handful of ZZ pairs.
    add_executable(test_ca_mps_vs_sv tests/unit/test_ca_mps_vs_sv.c)
    target_link_libraries(test_ca_mps_vs_sv PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_vs_sv COMMAND test_ca_mps_vs_sv)

    # CA-MPS marginal Z probabilities + T-dagger correctness against the
    # dense state-vector backend.  Deterministic n=4 H+T+T-dag+S+CNOT
    # circuit; every single-qubit P(Z=+1) must agree to <1e-12.
    add_executable(test_ca_mps_prob tests/unit/test_ca_mps_prob.c)
    target_link_libraries(test_ca_mps_prob PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_prob COMMAND test_ca_mps_prob)

    # Direct unit tests for the SVD compression layer that every MPS
    # truncation goes through.  Closes a 19-Apr audit gap: previously
    # exercised only indirectly via mps_vs_exact and the CA-MPS suite.
    add_executable(test_svd_compress tests/unit/test_svd_compress.c)
    target_link_libraries(test_svd_compress PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_svd_compress COMMAND test_svd_compress)

    # Pin complex-Hermitian Jacobi correctness in matrix_math.c.
    # Pre-2026-04-26 the rotations were real-valued; eigenvectors of
    # complex-Hermitian inputs only diagonalised the real part of A.
    add_executable(test_hermitian_eigen tests/unit/test_hermitian_eigen.c)
    target_link_libraries(test_hermitian_eigen PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_hermitian_eigen COMMAND test_hermitian_eigen)

    # QWZ phase diagram via qgt_phase_diagram_chern: integer Chern jumps
    # at m = -2, 0, +2 (gap closures); compare to analytic phases.
    add_executable(test_qgt_phase_diagram tests/unit/test_qgt_phase_diagram.c)
    target_link_libraries(test_qgt_phase_diagram PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qgt_phase_diagram COMMAND test_qgt_phase_diagram)

    # 2D phase-diagram mechanics, exercised on QWZ stacked along a
    # no-op y-axis.  The Haldane (t2, phi) physics test is queued for
    # when the Haldane Bloch implementation stabilises against grid-
    # size oscillation.
    add_executable(test_qgt_phase_diagram_2d tests/unit/test_qgt_phase_diagram_2d.c)
    target_link_libraries(test_qgt_phase_diagram_2d PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qgt_phase_diagram_2d COMMAND test_qgt_phase_diagram_2d)

    # Adaptive-bond TDVP step 1 (v0.4 / Phase 3B): backwards-compat
    # regression on the tdvp_adaptive_bond_config_t surface and the
    # tdvp_config_default / tdvp_config_adaptive helpers.  See
    # docs/research/adaptive_bond_tdvp.md for the full roadmap.
    add_executable(test_tdvp_adaptive_config tests/unit/test_tdvp_adaptive_config.c)
    target_link_libraries(test_tdvp_adaptive_config PRIVATE quantumsim)
    add_test(NAME unit_tdvp_adaptive_config COMMAND test_tdvp_adaptive_config)

    # Adaptive-bond TDVP steps 3+4 (v0.4 / Phase 3B): end-to-end smoke
    # over the entropy-feedback PID controller + per-bond state array.
    # Verifies engine allocation/free, a short imaginary-time
    # Heisenberg run, and per-bond chi staying inside the configured
    # [chi_floor, chi_ceiling] band.
    add_executable(test_tdvp_adaptive_pid tests/unit/test_tdvp_adaptive_pid.c)
    target_link_libraries(test_tdvp_adaptive_pid PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_tdvp_adaptive_pid COMMAND test_tdvp_adaptive_pid)

    # Adaptive-bond TDVP step 6a (v0.4 / Phase 3B): real-time energy
    # conservation under the entropy-feedback PID controller.
    # Symplectic 2TDVP must preserve <H> to within the integrator's
    # error envelope; the PID changes which singular values are kept
    # but never the Hamiltonian, so this is the first sharp acceptance
    # criterion from docs/research/adaptive_bond_tdvp.md.
    add_executable(test_tdvp_adaptive_energy_conservation
                   tests/unit/test_tdvp_adaptive_energy_conservation.c)
    target_link_libraries(test_tdvp_adaptive_energy_conservation
                          PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_tdvp_adaptive_energy_conservation
             COMMAND test_tdvp_adaptive_energy_conservation)

    # Adaptive-bond TDVP step 6b (v0.4 / Phase 3B): imaginary-time
    # ground-state convergence on the 8-site TFIM at g=1.  Runs DMRG
    # for a reference energy at chi=32, then asserts that 30
    # imag-time TDVP steps with the adaptive controller land within
    # 3% of the DMRG ground state.
    add_executable(test_tdvp_adaptive_tfim_ground
                   tests/unit/test_tdvp_adaptive_tfim_ground.c)
    target_link_libraries(test_tdvp_adaptive_tfim_ground
                          PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_tdvp_adaptive_tfim_ground
             COMMAND test_tdvp_adaptive_tfim_ground)

    # Adaptive-bond TDVP step 6c (v0.4 / Phase 3B): PID stability
    # sweep.  3x3x3 grid over (kp, ki, kd) around the reference
    # defaults; for each gain triplet, run 5 real-time TDVP steps
    # and measure max |chi(t+1) - chi(t)| across bonds.  At least
    # 80% of grid points must stay within 4 bond-units / step.
    add_executable(test_tdvp_adaptive_pid_stability
                   tests/unit/test_tdvp_adaptive_pid_stability.c)
    target_link_libraries(test_tdvp_adaptive_pid_stability
                          PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_tdvp_adaptive_pid_stability
             COMMAND test_tdvp_adaptive_pid_stability)

    # Adaptive-bond TDVP second-SVD re-truncation (v0.4.1): drives the
    # entropy-feedback PID into target_chi < first->bond_dim so the
    # second SVD pass in tdvp_truncate_bond actually fires, then pins
    # invariants on the resulting per-bond chi history, truncation
    # error, and final energy.  Branch-coverage smoke for the
    # otherwise-implicit second-pass code path documented in
    # docs/research/adaptive_bond_tdvp.md.
    add_executable(test_tdvp_adaptive_second_svd
                   tests/unit/test_tdvp_adaptive_second_svd.c)
    target_link_libraries(test_tdvp_adaptive_second_svd
                          PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_tdvp_adaptive_second_svd
             COMMAND test_tdvp_adaptive_second_svd)

    # CA-MPS bond-dimension advantage: a random Clifford circuit on n qubits
    # produces a stabilizer state that plain MPS needs bond dim ~2^(n/2) to
    # represent, while CA-MPS factors it entirely into the tableau so the
    # MPS factor stays at bond dim 1.
    add_executable(test_ca_mps_bond_advantage tests/unit/test_ca_mps_bond_advantage.c)
    target_link_libraries(test_ca_mps_bond_advantage PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_bond_advantage COMMAND test_ca_mps_bond_advantage)

    # CA-MPS expect_pauli_sum integration test: builds the Heisenberg
    # Hamiltonian as a Pauli sum and validates against dense state-vector
    # <psi|H|psi> after a random Clifford preparation.  Foundation for
    # VQE-on-CA-MPS and DMRG-on-CA-MPS drivers.
    add_executable(test_ca_mps_heisenberg tests/unit/test_ca_mps_heisenberg.c)
    target_link_libraries(test_ca_mps_heisenberg PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_heisenberg COMMAND test_ca_mps_heisenberg)

    # CA-MPS imaginary-time ground-state search: Trotter-Suzuki on the
    # Heisenberg bond-by-bond Gibbs factors, verify convergence to exact
    # ED on N=2/4/6 open chains.  Calls zheev directly for the dense ED
    # reference so it needs the LAPACK link (Accelerate on macOS).
    if(NOT QSIM_PLATFORM_WINDOWS)
        add_executable(test_ca_mps_imag_time tests/unit/test_ca_mps_imag_time.c)
        target_link_libraries(test_ca_mps_imag_time PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_ca_mps_imag_time PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_ca_mps_imag_time PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_ca_mps_imag_time COMMAND test_ca_mps_imag_time)
        set_tests_properties(unit_ca_mps_imag_time PROPERTIES TIMEOUT 120)

        # CA-MPS variational-D Clifford-only search.  Validates that the
        # greedy local-Clifford search reduces <psi|H|psi> on a small
        # TFIM problem (n=4, g=2.5) starting from |psi> = |0...0>.
        add_executable(test_ca_mps_var_d tests/unit/test_ca_mps_var_d.c)
        target_link_libraries(test_ca_mps_var_d PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_ca_mps_var_d PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_ca_mps_var_d PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_ca_mps_var_d COMMAND test_ca_mps_var_d)
        set_tests_properties(unit_ca_mps_var_d PROPERTIES TIMEOUT 60)

        # CA-MPS alternating variational-D (imag-time |phi> + Clifford D).
        # Headline experiment: TFIM at criticality (g=1) on n=6 must
        # converge to within 5% of the exact GS energy with bounded
        # |phi> entropy.
        add_executable(test_ca_mps_var_d_alt tests/unit/test_ca_mps_var_d_alt.c)
        target_link_libraries(test_ca_mps_var_d_alt PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_ca_mps_var_d_alt PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_ca_mps_var_d_alt PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_ca_mps_var_d_alt COMMAND test_ca_mps_var_d_alt)
        set_tests_properties(unit_ca_mps_var_d_alt PROPERTIES TIMEOUT 120)

        # Composite-2-gate move feature regression test.
        add_executable(test_ca_mps_var_d_composite tests/unit/test_ca_mps_var_d_composite.c)
        target_link_libraries(test_ca_mps_var_d_composite PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_ca_mps_var_d_composite PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_ca_mps_var_d_composite PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_ca_mps_var_d_composite COMMAND test_ca_mps_var_d_composite)
        set_tests_properties(unit_ca_mps_var_d_composite PROPERTIES TIMEOUT 60)

        # 1+1D Z2 lattice gauge theory Pauli-sum builder.  Pins the term-
        # count breakdown, individual term coefficients, Gauss-law operator
        # support, and Wilson-line construction.
        add_executable(test_z2_lgt_pauli_sum tests/unit/test_z2_lgt_pauli_sum.c)
        target_link_libraries(test_z2_lgt_pauli_sum PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_z2_lgt_pauli_sum COMMAND test_z2_lgt_pauli_sum)
        set_tests_properties(unit_z2_lgt_pauli_sum PROPERTIES TIMEOUT 30)

        # Stabilizer-subgroup warmstart for var-D.  Validates the
        # symplectic Gauss-Jordan Clifford builder on Bell, GHZ, and
        # 1+1D Z2 LGT Gauss-law generators.
        add_executable(test_gauge_warmstart tests/unit/test_gauge_warmstart.c)
        target_link_libraries(test_gauge_warmstart PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_gauge_warmstart COMMAND test_gauge_warmstart)
        set_tests_properties(unit_gauge_warmstart PROPERTIES TIMEOUT 30)

        # QGT integrator agreement (qgt_berry_grid vs _pt vs _proj on QWZ).
        add_executable(test_qgt_integrators tests/unit/test_qgt_integrators.c)
        target_link_libraries(test_qgt_integrators PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_integrators COMMAND test_qgt_integrators)
        set_tests_properties(unit_qgt_integrators PROPERTIES TIMEOUT 30 LABELS "topology")

        # Kane-Mele 4-band model + Z_2 invariant (v0.3 QGT extension).
        add_executable(test_qgt_kane_mele tests/unit/test_qgt_kane_mele.c)
        target_link_libraries(test_qgt_kane_mele PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_kane_mele COMMAND test_qgt_kane_mele)
        set_tests_properties(unit_qgt_kane_mele PROPERTIES TIMEOUT 30 LABELS "topology")

        # BHZ 4-band model + Z_2 invariant (v0.3 QGT extension).
        add_executable(test_qgt_bhz tests/unit/test_qgt_bhz.c)
        target_link_libraries(test_qgt_bhz PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_bhz COMMAND test_qgt_bhz)
        set_tests_properties(unit_qgt_bhz PROPERTIES TIMEOUT 30 LABELS "topology")

        # Kitaev p-wave chain + 1D BdG Z_2 invariant (v0.3 QGT extension).
        add_executable(test_qgt_kitaev_chain tests/unit/test_qgt_kitaev_chain.c)
        target_link_libraries(test_qgt_kitaev_chain PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_kitaev_chain COMMAND test_qgt_kitaev_chain)
        set_tests_properties(unit_qgt_kitaev_chain PROPERTIES TIMEOUT 30 LABELS "topology")

        # Hofstadter butterfly Chern sub-bands (v0.3 QGT extension).
        add_executable(test_qgt_hofstadter tests/unit/test_qgt_hofstadter.c)
        target_link_libraries(test_qgt_hofstadter PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_hofstadter COMMAND test_qgt_hofstadter)
        set_tests_properties(unit_qgt_hofstadter PROPERTIES TIMEOUT 60 LABELS "topology")

        # Cross-check momentum-space FHS / projector-trace vs real-space
        # Bianco-Resta on QWZ -- two independent topology calculations
        # must agree.
        add_executable(test_qgt_vs_chern_marker tests/unit/test_qgt_vs_chern_marker.c)
        target_link_libraries(test_qgt_vs_chern_marker PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_qgt_vs_chern_marker COMMAND test_qgt_vs_chern_marker)
        set_tests_properties(unit_qgt_vs_chern_marker PROPERTIES TIMEOUT 60 LABELS "topology")

        # MPDO noise simulator scaffold (v0.3 noise extension).
        add_executable(test_mpdo_smoke tests/unit/test_mpdo_smoke.c)
        target_link_libraries(test_mpdo_smoke PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_mpdo_smoke COMMAND test_mpdo_smoke)
        set_tests_properties(unit_mpdo_smoke PROPERTIES TIMEOUT 30 LABELS "noise")

        # Smoke test for the direct 1q/2q-gate fast path inside
        # moonlab_ca_mps_apply_pauli_imag (the perf optimisation that
        # bypasses the bond-dim-2 MPO build for low-weight Paulis).
        add_executable(test_imag_pauli_2q_smoke tests/unit/test_imag_pauli_2q_smoke.c)
        target_link_libraries(test_imag_pauli_2q_smoke PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_imag_pauli_2q_smoke COMMAND test_imag_pauli_2q_smoke)
        set_tests_properties(unit_imag_pauli_2q_smoke PROPERTIES TIMEOUT 30)

        # Round-trip test for tn_mps_from_statevector.  Closes the ICC
        # dead-code triage entry for that public function.
        add_executable(test_tn_mps_from_statevector
            tests/unit/test_tn_mps_from_statevector.c)
        target_link_libraries(test_tn_mps_from_statevector
            PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_tn_mps_from_statevector
            COMMAND test_tn_mps_from_statevector)
        set_tests_properties(unit_tn_mps_from_statevector
            PROPERTIES TIMEOUT 30)

        # Centralised moonlab_status_t registry test.  Closes audit
        # task #73 (was previously marked complete without code).
        add_executable(test_moonlab_status
            tests/unit/test_moonlab_status.c)
        target_link_libraries(test_moonlab_status
            PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_moonlab_status COMMAND test_moonlab_status)
        set_tests_properties(unit_moonlab_status PROPERTIES TIMEOUT 10)

        # Smoke harness for tensor-network public API surfaces that
        # ICC found had no in-tree caller (tensor_einsum,
        # svd_left_canonicalize, svd_right_canonicalize,
        # tn_expectation_2q).  Closes four entries on the dead-code
        # triage queue documented in AUDIT.md.
        add_executable(test_tn_dead_code_smoke
            tests/unit/test_tn_dead_code_smoke.c)
        target_link_libraries(test_tn_dead_code_smoke
            PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_tn_dead_code_smoke
            COMMAND test_tn_dead_code_smoke)
        set_tests_properties(unit_tn_dead_code_smoke
            PROPERTIES TIMEOUT 30)

        # MBL dead-code smoke (construct_lioms, scan_phase_diagram).
        # Closes two more dead-code triage entries.
        add_executable(test_mbl_smoke tests/unit/test_mbl_smoke.c)
        target_link_libraries(test_mbl_smoke
            PRIVATE quantumsim ${MATH_LIBRARY})
        add_test(NAME unit_mbl_smoke COMMAND test_mbl_smoke)
        set_tests_properties(unit_mbl_smoke PROPERTIES TIMEOUT 30)
    endif()

    # CA-MPS kagome 12-site Heisenberg ground state, matching PRB 83, 212401
    # (2011) Table I cluster "12" to <= 1e-2.  Runs a Trotter-Suzuki
    # imaginary-time schedule at chi_max = 256.  Tagged "long" and skipped
    # on Linux aarch64 and Windows due to runtime / BLAS specificity.
    add_executable(test_ca_mps_kagome12 tests/unit/test_ca_mps_kagome12.c)
    target_link_libraries(test_ca_mps_kagome12 PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_ca_mps_kagome12 COMMAND test_ca_mps_kagome12)
    set_tests_properties(unit_ca_mps_kagome12 PROPERTIES
        LABELS "long"
        TIMEOUT 300
    )

    add_executable(test_clifford tests/unit/test_clifford.c)
    target_link_libraries(test_clifford PRIVATE quantumsim)
    add_test(NAME unit_clifford COMMAND test_clifford)

    # Gate-fusion DAG: random-circuit parity vs gate-by-gate execution.
    add_executable(test_fusion tests/unit/test_fusion.c)
    target_link_libraries(test_fusion PRIVATE quantumsim)
    add_test(NAME unit_fusion COMMAND test_fusion)

    # Quantum Volume harness: HOP above 2/3 on noiseless simulator.
    add_executable(test_quantum_volume tests/unit/test_quantum_volume.c)
    target_link_libraries(test_quantum_volume PRIVATE quantumsim)
    add_test(NAME unit_quantum_volume COMMAND test_quantum_volume)

    # Clifford-backed surface code: syndrome response at d=7,9,15.
    add_executable(test_surface_code_clifford tests/unit/test_surface_code_clifford.c)
    target_link_libraries(test_surface_code_clifford PRIVATE quantumsim)
    add_test(NAME unit_surface_code_clifford COMMAND test_surface_code_clifford)

    # Local Chern marker: QWZ topological vs trivial phases.
    add_executable(test_chern_marker tests/unit/test_chern_marker.c)
    target_link_libraries(test_chern_marker PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_chern_marker COMMAND test_chern_marker)

    # matrix_math ground-truth: matmul, trace, Hermitian check, eigenvalues.
    # Eigenvector correctness is tested ONLY on the real-symmetric path;
    # the complex-Hermitian branch is known unsound (see matrix_math.h).
    add_executable(test_matrix_math tests/unit/test_matrix_math.c)
    target_link_libraries(test_matrix_math PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_matrix_math COMMAND test_matrix_math)

    # Matrix-free KPM Chern marker: parity with dense + large-L demo.
    # The aarch64 OpenBLAS path accumulates enough numerical drift in
    # the Newton-Schulz + Chebyshev-KPM stack to fail the parity-with-
    # dense pin (|c_dense - c_kpm| ~ 10^11 instead of < 0.1).  Label
    # `aarch64_flake` so the aarch64 CI tier can skip via -LE.
    add_executable(test_chern_kpm tests/unit/test_chern_kpm.c)
    target_link_libraries(test_chern_kpm PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_chern_kpm COMMAND test_chern_kpm)
    set_tests_properties(unit_chern_kpm PROPERTIES LABELS "aarch64_flake")
    # FHS momentum-space Chern integrator: cross-validates against
    # Newton-Schulz + KPM real-space markers on QWZ sweep.  Closes the
    # paper §2.7 / §4.3 three-paths claim.
    add_executable(test_chern_fhs tests/unit/test_chern_fhs.c)
    target_link_libraries(test_chern_fhs PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_chern_fhs COMMAND test_chern_fhs)
    # test_mpo_kpm calls zheev_ directly so it needs LAPACK (Accelerate on
    # macOS, -llapack on Linux).  Windows clang-cl doesn't ship a LAPACK
    # ABI we can link, so skip the target entirely there.
    if(NOT QSIM_PLATFORM_WINDOWS)
        add_executable(test_mpo_kpm tests/unit/test_mpo_kpm.c)
        target_link_libraries(test_mpo_kpm PRIVATE quantumsim ${MATH_LIBRARY})
        if(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_mpo_kpm PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
    endif()
    # test_gpu_backend_discovery links against gpu_is_available /
    # gpu_compute_init etc., which only exist when the unified
    # dispatcher (src/optimization/gpu/gpu_backend.c) is compiled
    # into the library -- and that source is conditional on at least
    # one backend being enabled (see the "Only add it when..." guard
    # above).  Match the guard here so plain Linux / WASM builds
    # don't try to link against undefined dispatcher symbols.
    if(QSIM_HAS_METAL OR QSIM_ENABLE_CUDA OR QSIM_ENABLE_OPENCL
       OR QSIM_ENABLE_VULKAN OR QSIM_ENABLE_CUQUANTUM OR QSIM_ENABLE_WEBGPU)
        add_executable(test_gpu_backend_discovery tests/unit/test_gpu_backend_discovery.c)
        target_link_libraries(test_gpu_backend_discovery PRIVATE quantumsim)
        add_test(NAME gpu_backend_discovery COMMAND test_gpu_backend_discovery)
    endif()
    add_executable(test_tensor_adversarial tests/unit/test_tensor_adversarial.c)
    target_link_libraries(test_tensor_adversarial PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME tensor_adversarial COMMAND test_tensor_adversarial)
    add_executable(test_differentiable tests/unit/test_differentiable.c)
    target_link_libraries(test_differentiable PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME differentiable COMMAND test_differentiable)

    add_executable(test_bell_variants tests/unit/test_bell_variants.c)
    target_link_libraries(test_bell_variants PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_bell_variants COMMAND test_bell_variants)

    add_executable(test_zne tests/unit/test_zne.c)
    target_link_libraries(test_zne PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_zne COMMAND test_zne)

    add_executable(test_povm tests/unit/test_povm.c)
    target_link_libraries(test_povm PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_povm COMMAND test_povm)

    add_executable(test_mutual_info tests/unit/test_mutual_info.c)
    target_link_libraries(test_mutual_info PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mutual_info COMMAND test_mutual_info)

    add_executable(test_qrng_di tests/unit/test_qrng_di.c)
    target_link_libraries(test_qrng_di PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qrng_di COMMAND test_qrng_di)

    add_executable(test_composite_noise tests/unit/test_composite_noise.c)
    target_link_libraries(test_composite_noise PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_composite_noise COMMAND test_composite_noise)

    add_executable(test_sha3 tests/unit/test_sha3.c)
    target_link_libraries(test_sha3 PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_sha3 COMMAND test_sha3)

    add_executable(test_mlkem_poly tests/unit/test_mlkem_poly.c)
    target_link_libraries(test_mlkem_poly PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem_poly COMMAND test_mlkem_poly)

    add_executable(test_mlkem tests/unit/test_mlkem.c)
    target_link_libraries(test_mlkem PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem COMMAND test_mlkem)

    add_executable(test_aes_drbg tests/unit/test_aes_drbg.c)
    target_link_libraries(test_aes_drbg PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_aes_drbg COMMAND test_aes_drbg)

    add_executable(test_mlkem_nist_kat tests/unit/test_mlkem_nist_kat.c)
    target_link_libraries(test_mlkem_nist_kat PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_mlkem_nist_kat COMMAND test_mlkem_nist_kat)
    add_executable(test_manifest tests/unit/test_manifest.c)
    target_link_libraries(test_manifest PRIVATE quantumsim)
    add_test(NAME unit_manifest COMMAND test_manifest)
    if(NOT QSIM_PLATFORM_WINDOWS AND QSIM_PLATFORM_MACOS)
        target_link_libraries(test_mpo_kpm PRIVATE "-framework Accelerate")
    endif()
    if(NOT QSIM_PLATFORM_WINDOWS)
        add_test(NAME unit_mpo_kpm COMMAND test_mpo_kpm)
        set_tests_properties(unit_mpo_kpm PROPERTIES LABELS "long")
    endif()

    # Quantum geometric tensor: Berry / Chern via FHS link variables.
    add_executable(test_qgt tests/unit/test_qgt.c)
    target_link_libraries(test_qgt PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qgt COMMAND test_qgt)

    # Shor-ECDLP resource estimator: reproduce Gidney-Drake-Boneh 2026
    # secp256k1 numbers + FTQC overhead model.
    add_executable(test_shor_ecdlp tests/unit/test_shor_ecdlp.c)
    target_link_libraries(test_shor_ecdlp PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_shor_ecdlp COMMAND test_shor_ecdlp)

    # Moonlab-Eshkol bridge parity: moonlab_eshkol_zgemm vs cblas_zgemm.
    # Always built; the test tolerates NOT_BUILT and NO_GPU by design.
    add_executable(test_gpu_eshkol tests/unit/test_gpu_eshkol.cpp)
    set_target_properties(test_gpu_eshkol PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_gpu_eshkol PRIVATE quantumsim)
    if(QSIM_PLATFORM_LINUX AND BLAS_LIB)
        target_link_libraries(test_gpu_eshkol PRIVATE ${BLAS_LIB})
    endif()
    if(QSIM_HAS_ACCELERATE)
        target_link_libraries(test_gpu_eshkol PRIVATE "-framework Accelerate")
    endif()
    add_test(NAME unit_gpu_eshkol COMMAND test_gpu_eshkol)

    # Chemistry module smoke.
    add_executable(test_chemistry tests/unit/test_chemistry.c)
    target_link_libraries(test_chemistry PRIVATE quantumsim)
    add_test(NAME unit_chemistry COMMAND test_chemistry)

    # Topological QC smoke.
    add_executable(test_topological tests/unit/test_topological.c)
    target_link_libraries(test_topological PRIVATE quantumsim)
    add_test(NAME unit_topological COMMAND test_topological)

    # Many-body localization smoke.
    add_executable(test_mbl tests/unit/test_mbl.c)
    target_link_libraries(test_mbl PRIVATE quantumsim)
    add_test(NAME unit_mbl COMMAND test_mbl)

    # SIMD / Accelerate parity against scalar reference.
    add_executable(test_simd_parity tests/unit/test_simd_parity.c)
    target_link_libraries(test_simd_parity PRIVATE quantumsim)
    if(QSIM_HAS_ACCELERATE)
        target_compile_definitions(test_simd_parity PRIVATE HAS_ACCELERATE=1)
    endif()
    add_test(NAME unit_simd_parity COMMAND test_simd_parity)

    # Metal GPU kernel parity against CPU (macOS + Metal only).
    add_executable(test_metal_parity tests/unit/test_metal_parity.c)
    target_link_libraries(test_metal_parity PRIVATE quantumsim)
    if(QSIM_HAS_METAL)
        target_compile_definitions(test_metal_parity PRIVATE HAS_METAL=1)
    endif()
    add_test(NAME unit_metal_parity COMMAND test_metal_parity)

    # 2D lattice + MPO-2D smoke (square / triangular / honeycomb).
    add_executable(test_lattice_2d tests/unit/test_lattice_2d.c)
    target_link_libraries(test_lattice_2d PRIVATE quantumsim)
    add_test(NAME unit_lattice_2d COMMAND test_lattice_2d)

    # Kagome Heisenberg ED vs published references.  Both tests call
    # LAPACK (zheev / dstev) directly rather than through quantumsim, so
    # they need an explicit LAPACK link -- Accelerate on macOS, -llapack on
    # Linux.  Windows clang-cl doesn't ship a LAPACK ABI we can link, so
    # skip these targets there (same pattern as test_mpo_kpm).
    if(NOT QSIM_PLATFORM_WINDOWS)
        # N=12 test: Läuchli-Sudan-Sørensen PRB 83, 212401 (2011) Table I
        # cluster "12" (rectangular 2x2 torus, E = -5.444875216 J).  Dense
        # ED via mpo_2d_heisenberg_dmi_create -> mpo_to_matrix -> zheev.
        add_executable(test_kagome_ed tests/unit/test_kagome_ed.c)
        target_link_libraries(test_kagome_ed PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_kagome_ed PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_kagome_ed PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_kagome_ed COMMAND test_kagome_ed)
        # Dense zheev on a 4096x4096 complex Hermitian matrix takes ~2s on
        # M2 Ultra but can hit 5+ min on slower CI runners (e.g. macos-15-intel
        # / dockerized aarch64).  Default ctest 300s timeout isn't enough on
        # those tiers; tag "long" + raise to 600s.
        set_tests_properties(unit_kagome_ed PROPERTIES
            LABELS "long"
            TIMEOUT 600
        )

        # N=18 test: PRB 83, 212401 Table I cluster "18 b" (rectangular 2x3
        # torus, E = -8.048270773 J).  Matrix-free Lanczos with full
        # reorthogonalization on a bond-list Heisenberg.  Peak memory ~840
        # MB, runtime ~8s on M2 Ultra.  Tagged "long" for CI.
        add_executable(test_kagome_ed_large tests/unit/test_kagome_ed_large.c)
        target_link_libraries(test_kagome_ed_large PRIVATE quantumsim ${MATH_LIBRARY})
        if(APPLE)
            target_link_libraries(test_kagome_ed_large PRIVATE "-framework Accelerate")
        elseif(QSIM_PLATFORM_LINUX AND LAPACK_LIB AND BLAS_LIB)
            target_link_libraries(test_kagome_ed_large PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
        endif()
        add_test(NAME unit_kagome_ed_large COMMAND test_kagome_ed_large)
        set_tests_properties(unit_kagome_ed_large PROPERTIES
            LABELS "long;memory_heavy"
            TIMEOUT 120
        )
    endif()

    # libirrep sector-ED bridge (since v0.6.1).  Validates the
    # space-group + rep-table + Lanczos-on-orbits pipeline at
    # N = 12, 18, 24 kagome (Sz = 0 sector).  When libirrep is OFF
    # the test gracefully exits 77 (CTest "skip") so the CI matrix
    # without libirrep stays green.
    add_executable(test_libirrep_sector_ed tests/unit/test_libirrep_sector_ed.c)
    target_link_libraries(test_libirrep_sector_ed PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_libirrep_sector_ed COMMAND test_libirrep_sector_ed)
    set_tests_properties(unit_libirrep_sector_ed PROPERTIES
        LABELS "long;libirrep"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
    )

    # libirrep CSS-code bridge (since v0.6.1).  Surface code d = 3, 5
    # built via irrep_surface_init + irrep_surface_build, plumbed
    # through the moonlab_libirrep_qec_t opaque handle.  Foundation
    # for the v0.6.2 expansion to toric / color / BB / hypergraph
    # families behind the same surface.
    add_executable(test_libirrep_css tests/unit/test_libirrep_css.c)
    target_link_libraries(test_libirrep_css PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_libirrep_css COMMAND test_libirrep_css)
    set_tests_properties(unit_libirrep_css PROPERTIES
        LABELS "libirrep"
        TIMEOUT 120
        SKIP_RETURN_CODE 77
    )

    # QGTL-shaped circuit-ingestion surface (since v0.6.6).  Validates
    # the moonlab_qgtl_* contract QGTL plugs into to route circuits
    # through moonlab's state-vector backend before paying for IBM /
    # Rigetti / IonQ shots.
    add_executable(test_qgtl_backend tests/unit/test_qgtl_backend.c)
    target_link_libraries(test_qgtl_backend PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_qgtl_backend COMMAND test_qgtl_backend)
    set_tests_properties(unit_qgtl_backend PROPERTIES
        LABELS "qgtl"
        TIMEOUT 60
    )

    # Multi-decoder bench harness scaffold (since v0.6.7).  Five
    # slots: GREEDY + MWPM_EXACT in-tree, SBNN + LIBIRREP_SS +
    # PYMATCHING return NOT_BUILT until v0.6.8 wires external deps.
    add_executable(test_decoder_bench tests/unit/test_decoder_bench.c)
    target_link_libraries(test_decoder_bench PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_decoder_bench COMMAND test_decoder_bench)
    set_tests_properties(unit_decoder_bench PROPERTIES
        LABELS "qec"
        TIMEOUT 30
    )

    # Distributed scheduler MVP (since v0.7.0).  In-process worker
    # fan-out atop the QGTL ingestion surface.  Bell + GHZ verified
    # across 3 / 4 workers.
    add_executable(test_scheduler tests/unit/test_scheduler.c)
    target_link_libraries(test_scheduler PRIVATE quantumsim ${MATH_LIBRARY})
    add_test(NAME unit_scheduler COMMAND test_scheduler)
    set_tests_properties(unit_scheduler PROPERTIES
        LABELS "distributed"
        TIMEOUT 60
    )

    # MPI scheduler transport (since v0.7.4) -- only built when
    # QSIM_ENABLE_MPI=ON, runs under mpirun -n 4 if available.
    if(QSIM_HAS_MPI)
        add_executable(test_scheduler_mpi tests/unit/test_scheduler_mpi.c)
        target_link_libraries(test_scheduler_mpi PRIVATE quantumsim MPI::MPI_C ${MATH_LIBRARY})
        find_program(MPIRUN_EXECUTABLE mpirun)
        if(MPIRUN_EXECUTABLE)
            add_test(NAME unit_scheduler_mpi
                     COMMAND ${MPIRUN_EXECUTABLE} -n 4
                             $<TARGET_FILE:test_scheduler_mpi>)
        else()
            # Fall back to single-rank run if mpirun is unavailable.
            add_test(NAME unit_scheduler_mpi COMMAND test_scheduler_mpi)
        endif()
        set_tests_properties(unit_scheduler_mpi PROPERTIES
            LABELS "distributed;mpi"
            TIMEOUT 120
        )

        # State-vector sharding (since v0.7.6) -- partitioned_state_t
        # driven through dist_hadamard + dist_cnot across MPI ranks.
        add_executable(test_partitioned_state tests/unit/test_partitioned_state.c)
        target_link_libraries(test_partitioned_state PRIVATE quantumsim MPI::MPI_C ${MATH_LIBRARY})
        if(MPIRUN_EXECUTABLE)
            add_test(NAME unit_partitioned_state
                     COMMAND ${MPIRUN_EXECUTABLE} -n 2
                             $<TARGET_FILE:test_partitioned_state>)
        else()
            add_test(NAME unit_partitioned_state COMMAND test_partitioned_state)
        endif()
        set_tests_properties(unit_partitioned_state PROPERTIES
            LABELS "distributed;mpi"
            TIMEOUT 60
        )
    endif()

    # Skyrmion braid path generators.
    add_executable(test_skyrmion tests/unit/test_skyrmion.c)
    target_link_libraries(test_skyrmion PRIVATE quantumsim)
    add_test(NAME unit_skyrmion COMMAND test_skyrmion)

    # Visualization tests — only when QSIM_BUILD_VISUALIZATION=ON.
    if(QSIM_BUILD_VISUALIZATION)
        # Feynman diagram builder smoke.
        add_executable(test_feynman tests/unit/test_feynman.c)
        target_link_libraries(test_feynman PRIVATE quantumsim)
        add_test(NAME unit_feynman COMMAND test_feynman)

        # Circuit diagram builder smoke.
        add_executable(test_circuit_diagram tests/unit/test_circuit_diagram.c)
        target_link_libraries(test_circuit_diagram PRIVATE quantumsim)
        add_test(NAME unit_circuit_diagram COMMAND test_circuit_diagram)
    endif()

    # Entropy-pool initialization regression checks.
    add_executable(test_entropy_pool tests/unit/test_entropy_pool.c)
    target_link_libraries(test_entropy_pool PRIVATE quantumsim)
    add_test(NAME unit_entropy_pool COMMAND test_entropy_pool)

    # ARM helper probing should use the compiled helper path, not ./tools.
    add_executable(test_hardware_entropy_probe
        tests/unit/test_hardware_entropy_probe.c
        src/applications/hardware_entropy.c
    )
    add_dependencies(test_hardware_entropy_probe hw_rng_probe)
    target_link_libraries(test_hardware_entropy_probe PRIVATE Threads::Threads)
    # This test compiles hardware_entropy.c directly rather than linking
    # quantumsim, so it doesn't inherit the library's include dirs OR the
    # library's system-library links.  Add src/ explicitly for the Windows
    # shim headers and add bcrypt on Windows for BCryptGenRandom, both of
    # which quantumsim already has in its target_link/include properties.
    target_include_directories(test_hardware_entropy_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    if(QSIM_PLATFORM_WINDOWS)
        target_link_libraries(test_hardware_entropy_probe PRIVATE bcrypt)
    endif()
    target_compile_definitions(test_hardware_entropy_probe PRIVATE
        MOONLAB_TESTING=1
        MOONLAB_HW_RNG_PROBE_PATH=\"$<TARGET_FILE:hw_rng_probe>\"
    )
    add_test(NAME unit_hardware_entropy_probe COMMAND test_hardware_entropy_probe)

    # QRNG statistical quality: chi^2 / monobit / serial correlation
    # on 256 KiB drawn through the stable ABI.
    add_executable(test_qrng_statistics tests/unit/test_qrng_statistics.c)
    target_link_libraries(test_qrng_statistics PRIVATE quantumsim)
    add_test(NAME unit_qrng_statistics COMMAND test_qrng_statistics)
    set_tests_properties(unit_qrng_statistics PROPERTIES LABELS "long")

    if(QSIM_BUILD_BENCHMARKS)
    # Performance benchmarks (build-only; no ctest entry so CI stays
    # fast). Run directly for local profiling:
    #   ./build/bench_state_operations
    #   ./build/bench_tensor_networks
    add_executable(bench_ca_mps tests/performance/bench_ca_mps.c)
    target_link_libraries(bench_ca_mps PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_apply_conjugated_imag tests/performance/bench_apply_conjugated_imag.c)
    target_link_libraries(bench_apply_conjugated_imag PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_warmstart_pivot_scaling tests/performance/bench_warmstart_pivot_scaling.c)
    target_link_libraries(bench_warmstart_pivot_scaling PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_topology_phase_diagrams tests/performance/bench_topology_phase_diagrams.c)
    target_link_libraries(bench_topology_phase_diagrams PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_warmstart_empirical_entropy tests/performance/bench_warmstart_empirical_entropy.c)
    target_link_libraries(bench_warmstart_empirical_entropy PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_state_operations tests/performance/bench_state_operations.c)
    target_link_libraries(bench_state_operations PRIVATE quantumsim)
    add_executable(bench_tensor_networks tests/performance/bench_tensor_networks.c)
    target_link_libraries(bench_tensor_networks PRIVATE quantumsim)
    add_executable(bench_fusion tests/performance/bench_fusion.c)
    target_link_libraries(bench_fusion PRIVATE quantumsim)
    add_executable(bench_quantum_volume tests/performance/bench_quantum_volume.c)
    target_link_libraries(bench_quantum_volume PRIVATE quantumsim)
    add_executable(bench_clifford tests/performance/bench_clifford.c)
    target_link_libraries(bench_clifford PRIVATE quantumsim)
    add_executable(bench_chern_kpm tests/performance/bench_chern_kpm.c)
    target_link_libraries(bench_chern_kpm PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_chern_mosaic tests/performance/bench_chern_mosaic.c)
    target_link_libraries(bench_chern_mosaic PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_chern_mosaic_hq tests/performance/bench_chern_mosaic_hq.c)
    target_link_libraries(bench_chern_mosaic_hq PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_tensor_matmul_eshkol tests/performance/bench_tensor_matmul_eshkol.c)
    target_link_libraries(bench_tensor_matmul_eshkol PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_dmrg_workspace tests/performance/bench_dmrg_workspace.c)
    target_link_libraries(bench_dmrg_workspace PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_diff_adjoint tests/performance/bench_diff_adjoint.c)
    target_link_libraries(bench_diff_adjoint PRIVATE quantumsim ${MATH_LIBRARY})
    add_executable(bench_hadamard tests/performance/bench_hadamard.c)
    target_link_libraries(bench_hadamard PRIVATE quantumsim ${MATH_LIBRARY})
    qsim_target_link_openmp(bench_hadamard)

    add_executable(bench_hadamard_metal tests/performance/bench_hadamard_metal.c)
    target_link_libraries(bench_hadamard_metal PRIVATE quantumsim)
    if(QSIM_HAS_METAL)
        target_compile_definitions(bench_hadamard_metal PRIVATE HAS_METAL=1)
    endif()

    # State-vector gate throughput, JSON-emitting paper companion.
    add_executable(bench_state_throughput tests/performance/bench_state_throughput.c)
    target_link_libraries(bench_state_throughput PRIVATE quantumsim ${MATH_LIBRARY})
    qsim_target_link_openmp(bench_state_throughput)

    # CLOPS benchmark: variational circuits-per-second.
    add_executable(bench_clops tests/performance/bench_clops.c)
    target_link_libraries(bench_clops PRIVATE quantumsim ${MATH_LIBRARY})
    qsim_target_link_openmp(bench_clops)

    # XEB benchmark: cross-entropy / Porter-Thomas on random circuits.
    add_executable(bench_xeb tests/performance/bench_xeb.c)
    target_link_libraries(bench_xeb PRIVATE quantumsim ${MATH_LIBRARY})
    qsim_target_link_openmp(bench_xeb)

    # Pauli-frame surface-code throughput.
    add_executable(bench_pauli_frame tests/performance/bench_pauli_frame.c)
    target_link_libraries(bench_pauli_frame PRIVATE quantumsim ${MATH_LIBRARY})
    qsim_target_link_openmp(bench_pauli_frame)

    # Cross-backend TFIM ground-state validation: ED vs DMRG vs CA-MPS var-D
    # on the same Hamiltonian.  Closes the moonlab paper's cross-validation
    # audit point.
    add_executable(bench_cross_backend_tfim
        tests/performance/bench_cross_backend_tfim.c)
    target_link_libraries(bench_cross_backend_tfim PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Cross-backend XXZ Heisenberg ground-state validation -- companion
    # to the TFIM bench above.  XXZ is the honest test where var-D's
    # SU(2)-symmetric collapse to plain DMRG is measured rather than
    # cherry-picked away.
    add_executable(bench_cross_backend_xxz
        tests/performance/bench_cross_backend_xxz.c)
    target_link_libraries(bench_cross_backend_xxz PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Cross-backend kagome-12 AFM Heisenberg ground-state validation.
    # Anchors against the Lauchli-Sudan-Sorensen / libirrep cluster-12
    # reference E_0 = -5.44487522 J.  Three engines on a frustrated
    # 2D model rather than the easy 1D ones above.
    add_executable(bench_cross_backend_kagome12
        tests/performance/bench_cross_backend_kagome12.c)
    target_link_libraries(bench_cross_backend_kagome12 PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Optional: Eshkol fp64-on-Metal backend (SF64 / Ozaki-II CRT).
    # Enable via -DQSIM_ENABLE_ESHKOL=ON and -DQSIM_ESHKOL_ROOT=/path/to/eshkol.
    if(QSIM_ENABLE_ESHKOL)
        set(_eshkol_inc   "${QSIM_ESHKOL_ROOT}/inc")
        set(_eshkol_libfn "${QSIM_ESHKOL_ROOT}/build/libeshkol-static.a")
        if(NOT EXISTS "${_eshkol_libfn}")
            message(FATAL_ERROR
                "QSIM_ENABLE_ESHKOL=ON but ${_eshkol_libfn} not found. "
                "Build Eshkol first (cmake -S ${QSIM_ESHKOL_ROOT} -B "
                "${QSIM_ESHKOL_ROOT}/build && cmake --build "
                "${QSIM_ESHKOL_ROOT}/build).")
        endif()
        add_executable(bench_eshkol_gemm tests/performance/bench_eshkol_gemm.cpp)
        set_target_properties(bench_eshkol_gemm PROPERTIES CXX_STANDARD 17)
        target_include_directories(bench_eshkol_gemm PRIVATE "${_eshkol_inc}")
        target_compile_definitions(bench_eshkol_gemm PRIVATE HAS_ESHKOL=1)
        target_link_libraries(bench_eshkol_gemm PRIVATE
            "${_eshkol_libfn}"
            "-framework Accelerate"
            "-framework Metal"
            "-framework MetalPerformanceShaders"
            "-framework Foundation"
            "-framework CoreGraphics"
        )
        message(STATUS "Eshkol: ON (lib=${_eshkol_libfn})")
    endif()
    endif()

    # Python bindings smoke — imports moonlab.core, constructs a Bell
    # state, asserts exact amplitudes. Requires python3 on PATH plus
    # numpy (the smoke test imports it transitively through
    # moonlab.core).  Gate the registration on numpy being importable
    # so CI tiers that don't pre-install Python scientific packages
    # (e.g. the plain macos-arm64 / macos-debug / macos-werror tiers
    # that only brew-install libomp + ninja) don't fail here; the
    # dedicated bindings-smoke job does pip-install numpy first and
    # will still run the tests.
    find_program(PYTHON3_EXECUTABLE python3)
    if(PYTHON3_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON3_EXECUTABLE} -c "import numpy"
            RESULT_VARIABLE _numpy_probe
            OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(PYTHON3_EXECUTABLE AND _numpy_probe EQUAL 0)
        add_test(NAME python_bindings_smoke
                 COMMAND ${PYTHON3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_bindings_smoke.py)
        set_tests_properties(python_bindings_smoke PROPERTIES
            ENVIRONMENT
                "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )

        # Full pytest suite — covers measurement entropy-ctx, algorithms
        # (VQE/QAOA/Grover/BellTest), and state ops.  Gated on pytest
        # being importable; otherwise skipped so bare-Python installs
        # don't break the default build.
        execute_process(
            COMMAND ${PYTHON3_EXECUTABLE} -c "import pytest"
            RESULT_VARIABLE _pytest_probe
            OUTPUT_QUIET ERROR_QUIET)
        if(_pytest_probe EQUAL 0)
            add_test(NAME python_bindings_pytest
                     COMMAND ${PYTHON3_EXECUTABLE} -m pytest -q
                             --deselect tests/test_algorithms.py::TestVQEH2
                             --deselect tests/test_algorithms.py::TestGroverSearch::test_grover_search_medium
                             --deselect tests/test_algorithms.py::TestCHSHTest::test_chsh_high_statistics
                             ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/tests
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python)
            set_tests_properties(python_bindings_pytest PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
            )
        endif()
    endif()

    # Binding-version sync gate.  Verifies every binding manifest's
    # version field matches VERSION.txt at the repo root.  Cheap and
    # fast; runs on every ctest invocation because version skew is the
    # kind of mistake that hides until release time.
    add_test(NAME bindings_version_sync
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_binding_versions.sh)
    set_tests_properties(bindings_version_sync PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # Rust bindings smoke — cargo test in bindings/rust/moonlab, only
    # if the crate root and Cargo are present.
    find_program(CARGO_EXECUTABLE cargo)
    if(CARGO_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/rust/moonlab/Cargo.toml")
        add_test(NAME rust_bindings_smoke
                 COMMAND ${CARGO_EXECUTABLE} test --manifest-path
                         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/rust/moonlab/Cargo.toml
                         --no-fail-fast)
        set_tests_properties(rust_bindings_smoke PROPERTIES
            ENVIRONMENT
                "MOONLAB_LIB_DIR=$<TARGET_FILE_DIR:quantumsim>;DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )
    endif()

    # WebGPU parity smoke — runs the existing randomized eval script.
    # Requires pnpm, emscripten-built moonlab.wasm, a pnpm-built TS dist,
    # and optionally a WebGPU-capable runtime (Chromium 113+ or Deno 2).
    #
    # We gate on both the script existing AND the compiled dist already
    # being present, because CMake does not own the pnpm build graph;
    # a fresh clone without `pnpm -C bindings/javascript/packages/core
    # build` would otherwise fail this test on first run.
    #
    # With -DQSIM_BUILD_JS_DIST=ON, we invoke the pnpm build ourselves
    # via a custom_target so dist/index.mjs lands on disk before
    # ctest runs.  Default OFF keeps routine builds fast.
    find_program(PNPM_EXECUTABLE pnpm)
    set(_js_dist "${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core/dist/index.mjs")
    if(QSIM_BUILD_JS_DIST AND PNPM_EXECUTABLE)
        add_custom_command(
            OUTPUT ${_js_dist}
            COMMAND ${PNPM_EXECUTABLE} install --ignore-scripts
            COMMAND ${PNPM_EXECUTABLE} run build:ts
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core
            COMMENT "Building TS/JS dist via pnpm (QSIM_BUILD_JS_DIST=ON)"
            VERBATIM
        )
        add_custom_target(js_dist ALL DEPENDS ${_js_dist})
    endif()
    if(PNPM_EXECUTABLE AND EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core/scripts/webgpu-unified-smoke.mjs"
        AND (EXISTS "${_js_dist}" OR QSIM_BUILD_JS_DIST))
        add_test(NAME webgpu_unified_smoke
                 COMMAND ${PNPM_EXECUTABLE} -C
                         ${CMAKE_CURRENT_SOURCE_DIR}/bindings/javascript/packages/core
                         exec node scripts/webgpu-unified-smoke.mjs)
    endif()

    # Distributed computing tests (MPI). Each test executable is only
    # added if the corresponding source file is present — the MPI test
    # sources are planned per the 0.x release roadmap but not all are
    # in the tree yet. This prevents CMake generate errors in builds
    # that enable MPI before the matching test is written.
    if(QSIM_HAS_MPI)
        set(_mpi_tests_added 0)
        foreach(_mpi_test_name IN ITEMS
                distributed_gates
                state_partition
                collective_ops)
            set(_mpi_test_src "tests/unit/test_${_mpi_test_name}.c")
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_mpi_test_src}")
                add_executable(test_${_mpi_test_name} ${_mpi_test_src})
                target_link_libraries(test_${_mpi_test_name}
                                      PRIVATE quantumsim MPI::MPI_C)
                # --oversubscribe (Open MPI >= 5) or --use-hwthread-cpus
                # lets us run 4 MPI procs on CI runners with fewer than
                # 4 physical cores (macos-14 exposes 3 cores).  The flag
                # is a no-op when there are enough slots.
                add_test(NAME ${_mpi_test_name}
                         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                                 --oversubscribe
                                 ${MPIEXEC_PREFLAGS}
                                 $<TARGET_FILE:test_${_mpi_test_name}>
                                 ${MPIEXEC_POSTFLAGS})
                math(EXPR _mpi_tests_added "${_mpi_tests_added} + 1")
            endif()
        endforeach()
        if(_mpi_tests_added GREATER 0)
            message(STATUS "MPI Tests: Enabled (${_mpi_tests_added} test(s), 4 MPI processes each)")
        else()
            message(STATUS "MPI Tests: None present in tests/unit/ yet")
        endif()
    endif()

    # Downstream-ABI smoke test — dlopens libquantumsim and verifies the
    # symbols declared in src/applications/moonlab_export.h. Only meaningful
    # for shared builds (dlopen needs a dylib/so).
    if(QSIM_BUILD_SHARED AND NOT QSIM_PLATFORM_WINDOWS)
        add_executable(test_moonlab_export_abi tests/abi/test_moonlab_export_abi.c)
        target_link_libraries(test_moonlab_export_abi PRIVATE ${CMAKE_DL_LIBS})
        add_test(NAME abi_moonlab_export
                 COMMAND test_moonlab_export_abi)
        # Make the freshly-built libquantumsim discoverable to the test at
        # run time without requiring `make install`.
        set_tests_properties(abi_moonlab_export PROPERTIES
            ENVIRONMENT
                "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:quantumsim>:$ENV{LD_LIBRARY_PATH}"
        )
    endif()

    # ==========================================================================
    # Per-subsystem ctest labels.  Selectable via:
    #   ctest -L core           # state-vector + gates + simd dispatch
    #   ctest -L tn             # tensor-network MPS / DMRG / TDVP
    #   ctest -L ca_mps         # CA-MPS hybrid + CA-PEPS
    #   ctest -L topology       # Chern markers, surface code, skyrmions
    #   ctest -L clifford       # Aaronson-Gottesman + Pauli-frame
    #   ctest -L algorithms     # VQE / QAOA / Grover / QPE / chemistry / Shor
    #   ctest -L qrng           # QRNG, hardware entropy, Bell certification
    #   ctest -L crypto         # ML-KEM, AES-DRBG, SHA-3
    #   ctest -L bell           # Bell-test variants + CHSH aggregate
    #   ctest -L gpu            # Metal / WebGPU / GPU backend discovery
    #   ctest -L viz            # circuit + Feynman diagram renderers
    #   ctest -L bindings       # Python / Rust / version-sync gate
    #   ctest -L abi            # stable-ABI dlsym smoke + status registry
    #   ctest -L examples       # smoke-tested example binaries
    # Labels stack; `unit_kagome_ed_large` is `long;algorithms`.
    # ==========================================================================
    qsim_label_tests(core
        unit_quantum_state unit_quantum_gates unit_constants
        unit_correctness_properties unit_measurement unit_entanglement
        unit_noise unit_composite_noise unit_fusion unit_povm
        unit_mutual_info unit_zne unit_simd_dispatch unit_simd_parity
        unit_memory_align gate_test fast_measurement
        comprehensive correlation_test unit_hermitian_eigen
        unit_matrix_math unit_svd_compress unit_metal_parity)
    qsim_label_tests(tn
        unit_tensor_network unit_tn_dead_code_smoke
        unit_tn_mps_from_statevector tensor_adversarial dmrg
        mps_vs_exact unit_lattice_2d)
    qsim_label_tests(ca_mps
        unit_ca_mps_bond_advantage unit_ca_mps_heisenberg
        unit_ca_mps_imag_time unit_ca_mps_kagome12 unit_ca_mps_limits
        unit_ca_mps_prob unit_ca_mps_var_d unit_ca_mps_var_d_alt
        unit_ca_mps_var_d_composite unit_ca_mps_vs_sv unit_ca_peps
        unit_gauge_warmstart unit_z2_lgt_pauli_sum)
    qsim_label_tests(topology
        unit_chern_marker unit_chern_kpm unit_chern_fhs unit_qgt
        unit_qgt_phase_diagram unit_qgt_phase_diagram_2d
        unit_topological unit_skyrmion unit_surface_code_clifford
        unit_pauli_frame unit_mpo_kpm)
    qsim_label_tests(clifford
        unit_clifford unit_clifford_pauli_api unit_pauli_frame
        unit_surface_code_clifford)
    qsim_label_tests(algorithms
        unit_grover unit_qaoa unit_qpe unit_vqe unit_chemistry
        unit_quantum_volume unit_kagome_ed unit_kagome_ed_large
        unit_shor_ecdlp unit_differentiable unit_mbl unit_mbl_smoke
        differentiable)
    qsim_label_tests(qrng
        unit_qrng_di unit_qrng_statistics unit_entropy_pool
        unit_hardware_entropy_probe health_tests)
    qsim_label_tests(crypto
        unit_mlkem unit_mlkem_nist_kat unit_mlkem_poly unit_aes_drbg
        unit_sha3)
    qsim_label_tests(bell
        unit_bell_variants bell_test)
    qsim_label_tests(gpu
        gpu_backend_discovery unit_gpu_eshkol unit_metal_parity
        webgpu_unified_smoke)
    qsim_label_tests(viz
        unit_circuit_diagram unit_feynman unit_manifest)
    qsim_label_tests(bindings
        bindings_version_sync python_bindings_pytest
        python_bindings_smoke rust_bindings_smoke)
    qsim_label_tests(abi
        abi_moonlab_export unit_moonlab_status)
    qsim_label_tests(examples
        example_ca_peps_2d_tfim_smoke example_grover_hash_collision
        example_phase3_phase4_benchmark example_qaoa_maxcut
        example_quantum_spin_chain_small example_vqe_h2_molecule)

    message(STATUS "Tests: Enabled")
