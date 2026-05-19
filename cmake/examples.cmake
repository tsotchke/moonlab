# cmake/examples.cmake — example program targets for libquantumsim.
#
# Included from the root CMakeLists.txt at the body of the
#   if(QSIM_BUILD_EXAMPLES)
#       include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/examples.cmake)
#   endif()
# block.  Same-scope include() so the file reads root-defined
# variables (QSIM_HAS_*) and macros (qsim_target_link_openmp).

    # Sharded GHZ example (since v0.7.9) -- demonstrates >32-qubit
    # state-vector reach via MPI partitions.  Built only when MPI
    # is enabled; runs under `mpirun -n 4`.
    if(QSIM_HAS_MPI)
        add_executable(large_state_ghz examples/distributed/large_state_ghz.c)
        target_link_libraries(large_state_ghz PRIVATE quantumsim MPI::MPI_C)

        # Sharded QFT (since v0.8.1) -- N-qubit textbook QFT under
        # dist_hadamard + dist_cphase + dist_swap across MPI ranks.
        add_executable(large_state_qft examples/distributed/large_state_qft.c)
        target_link_libraries(large_state_qft PRIVATE quantumsim MPI::MPI_C)
    endif()

    # Grover examples
    add_executable(grover_hash_collision examples/quantum/grover_hash_collision.c)
    target_link_libraries(grover_hash_collision PRIVATE quantumsim)

    add_executable(grover_large_scale_demo examples/quantum/grover_large_scale_demo.c)
    target_link_libraries(grover_large_scale_demo PRIVATE quantumsim)

    add_executable(grover_large_scale_optimized examples/quantum/grover_large_scale_optimized.c)
    target_link_libraries(grover_large_scale_optimized PRIVATE quantumsim)

    add_executable(grover_password_crack examples/quantum/grover_password_crack.c)
    target_link_libraries(grover_password_crack PRIVATE quantumsim)

    add_executable(phase3_phase4_benchmark examples/quantum/phase3_phase4_benchmark.c)
    target_link_libraries(phase3_phase4_benchmark PRIVATE quantumsim)

    # Application examples
    add_executable(vqe_h2_molecule examples/applications/vqe_h2_molecule.c)
    target_link_libraries(vqe_h2_molecule PRIVATE quantumsim)

    add_executable(qaoa_maxcut examples/applications/qaoa_maxcut.c)
    target_link_libraries(qaoa_maxcut PRIVATE quantumsim)

    add_executable(portfolio_optimization examples/applications/portfolio_optimization.c)
    target_link_libraries(portfolio_optimization PRIVATE quantumsim)

    add_executable(tsp_logistics examples/applications/tsp_logistics.c)
    target_link_libraries(tsp_logistics PRIVATE quantumsim)

    add_executable(pqc_qrng_demo examples/applications/pqc_qrng_demo.c)
    target_link_libraries(pqc_qrng_demo PRIVATE quantumsim)

    add_executable(diff_vqe_demo examples/tensor_network/diff_vqe_demo.c)
    target_link_libraries(diff_vqe_demo PRIVATE quantumsim ${MATH_LIBRARY})

    add_executable(quantum_spin_chain examples/tensor_network/quantum_spin_chain.c)
    target_link_libraries(quantum_spin_chain PRIVATE quantumsim)
    if(QSIM_HAS_OPENMP)
        qsim_target_link_openmp(quantum_spin_chain)
        if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
            target_compile_options(quantum_spin_chain PRIVATE -Xpreprocessor -fopenmp)
        endif()
        target_compile_definitions(quantum_spin_chain PRIVATE HAS_OPENMP=1)
    endif()

    add_executable(quantum_critical_point examples/tensor_network/quantum_critical_point.c)
    target_link_libraries(quantum_critical_point PRIVATE quantumsim)

    add_executable(example_ca_mps_clifford_advantage
        examples/tensor_network/ca_mps_clifford_advantage.c)
    target_link_libraries(example_ca_mps_clifford_advantage PRIVATE quantumsim)

    add_executable(example_ca_mps_imag_time
        examples/tensor_network/ca_mps_imag_time.c)
    target_link_libraries(example_ca_mps_imag_time PRIVATE quantumsim ${MATH_LIBRARY})

    # Variational-D oracle proof: does a hand-supplied Clifford reduce
    # |phi> bond dim for the TFIM ground state?  Gating experiment for the
    # full var-D implementation (see docs/research/ca_mps.md §5.3).
    add_executable(example_ca_mps_oracle_proof
        examples/tensor_network/ca_mps_oracle_proof.c)
    target_link_libraries(example_ca_mps_oracle_proof PRIVATE quantumsim ${MATH_LIBRARY})

    # Head-to-head: variational-D CA-MPS vs plain DMRG on a TFIM phase
    # sweep.  This is the paper figure -- entropy ratio S(phi)/S(psi)
    # across the ferromagnetic / critical / paramagnetic regimes.
    add_executable(example_ca_mps_var_d_vs_plain_dmrg
        examples/tensor_network/ca_mps_var_d_vs_plain_dmrg.c)
    target_link_libraries(example_ca_mps_var_d_vs_plain_dmrg PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Bond-dim cap scan: at fixed (n, g), what chi do we need for a
    # given accuracy?  Plain DMRG vs variational-D CA-MPS.  This is
    # the paper's strongest claim -- the chi required for a fixed
    # accuracy threshold is much smaller for var-D than for plain DMRG.
    add_executable(example_ca_mps_var_d_chi_scan
        examples/tensor_network/ca_mps_var_d_chi_scan.c)
    target_link_libraries(example_ca_mps_var_d_chi_scan PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Greedy Clifford disentangler: take a converged plain-DMRG ground
    # state and directly minimise its half-cut entropy via Clifford
    # gates applied to the MPS.  Different angle on the var-D claim --
    # entropy as direct objective, no imag-time, no warmstart heuristics.
    add_executable(example_ca_mps_disentangler
        examples/tensor_network/ca_mps_disentangler.c)
    target_link_libraries(example_ca_mps_disentangler PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Generalisation: var-D vs plain DMRG on the XXZ Heisenberg model.
    # If the entropy-reduction claim is real and not TFIM-specific, it
    # should also reduce |phi> entropy on Heisenberg ground states across
    # the Delta sweep (XY -> SU(2) -> Ising-like).
    add_executable(example_ca_mps_var_d_heisenberg
        examples/tensor_network/ca_mps_var_d_heisenberg.c)
    target_link_libraries(example_ca_mps_var_d_heisenberg PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Hard test: var-D on kagome 12-site Heisenberg AFM (frustrated
    # spin liquid GS).  Compares the four warmstarts against the
    # IDENTITY baseline (which is plain MPS imag-time).  This is the
    # "can it help on a hard problem" test for the paper.
    add_executable(example_ca_mps_var_d_kagome12
        examples/tensor_network/ca_mps_var_d_kagome12.c)
    target_link_libraries(example_ca_mps_var_d_kagome12 PRIVATE
        quantumsim ${MATH_LIBRARY})

    # HEP demo: var-D absorbs Gauss-law constraints on 1+1D Z2 lattice
    # gauge theory.  Bridges the var-D method to lattice-QCD-style
    # simulations -- the SaaS / hardware-development / serious-mathematics
    # vignette that no competing simulator currently offers.
    add_executable(example_z2_gauge_var_d
        examples/hep/z2_gauge_var_d.c)
    target_link_libraries(example_z2_gauge_var_d PRIVATE
        quantumsim ${MATH_LIBRARY})

    # CA-PEPS 2D TFIM imag-time evolution.  Exercises the row-major-MPS-
    # backed CA-PEPS path end-to-end on an Lx-by-Ly square lattice and
    # emits a ca_peps_2d_tfim_v1 JSON for paper figures.
    add_executable(example_ca_peps_2d_tfim
        examples/tensor_network/ca_peps_2d_tfim.c)
    target_link_libraries(example_ca_peps_2d_tfim PRIVATE
        quantumsim ${MATH_LIBRARY})

    # CA-PEPS 2D TFIM ground-state cross-checked against dense ED.  Runs
    # imag-time evolution on the same Pauli sum that vqe_exact_ground_state_energy
    # diagonalises, and reports |E_imag - E_ED| as the parity figure that
    # sits underneath the §3.6 paper claim of correctness for the row-
    # major-MPS-backed CA-PEPS at 2D TFIM scales.
    add_executable(example_ca_peps_2d_tfim_vs_ed
        examples/tensor_network/ca_peps_2d_tfim_vs_ed.c)
    target_link_libraries(example_ca_peps_2d_tfim_vs_ed PRIVATE
        quantumsim ${MATH_LIBRARY})

    # CA-PEPS variational-D ground-state on the same 2D TFIM benchmark.
    # Uses the alternating Clifford-disentangle + imag-time loop with
    # FERRO/H_ALL warmstart selected by field-strength regime; the
    # paper §3.5 var-D claim is now testable in 2D, not just 1D.
    add_executable(example_ca_peps_2d_tfim_var_d_vs_ed
        examples/tensor_network/ca_peps_2d_tfim_var_d_vs_ed.c)
    target_link_libraries(example_ca_peps_2d_tfim_var_d_vs_ed PRIVATE
        quantumsim ${MATH_LIBRARY})

    # v0.3 QGT topology examples: each demonstrates one new model
    # primitive (Kane-Mele, BHZ, Hofstadter, Kitaev p-wave chain) and
    # prints the topological invariant across its phase diagram.
    add_executable(example_qgt_kane_mele
        examples/topological/qgt_kane_mele.c)
    target_link_libraries(example_qgt_kane_mele PRIVATE
        quantumsim ${MATH_LIBRARY})
    add_executable(example_qgt_bhz
        examples/topological/qgt_bhz.c)
    target_link_libraries(example_qgt_bhz PRIVATE
        quantumsim ${MATH_LIBRARY})
    add_executable(example_qgt_hofstadter
        examples/topological/qgt_hofstadter.c)
    target_link_libraries(example_qgt_hofstadter PRIVATE
        quantumsim ${MATH_LIBRARY})
    add_executable(example_qgt_kitaev_z2
        examples/topological/qgt_kitaev_z2.c)
    target_link_libraries(example_qgt_kitaev_z2 PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Bell-CHSH multi-run aggregate: runs the CHSH test N>=5 times and
    # reports median + IQR + min/max + violation rate.  Closes the
    # paper §4.2 todo about reporting a statistically-honest CHSH
    # value rather than a single sampled run.
    add_executable(example_bell_chsh_aggregate
        examples/applications/bell_chsh_aggregate.c)
    target_link_libraries(example_bell_chsh_aggregate PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Bell-variants harness: CHSH + Mermin-3 + Mermin-Klyshko-{4,5}.
    # Closes paper §4.2 follow-on ("Mermin and Mermin-Klyshko N-party
    # Bell tests are exposed under the same harness for N-qubit GHZ
    # states") -- the bell_chsh_aggregate harness above only drives
    # the 2-qubit CHSH variant.
    add_executable(example_bell_variants_aggregate
        examples/applications/bell_variants_aggregate.c)
    target_link_libraries(example_bell_variants_aggregate PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Surface-code threshold sweep: code-capacity i.i.d. depolarising
    # noise on the open-boundary planar surface code, greedy nearest-
    # neighbour MWPM decoder, distances {3, 5, 7}.  Closes paper §4.4
    # todo about replacing the imported Wang 2003 threshold value with
    # a Moonlab-measured curve.
    add_executable(example_surface_code_threshold
        examples/applications/surface_code_threshold.c)
    target_link_libraries(example_surface_code_threshold PRIVATE
        quantumsim ${MATH_LIBRARY})

    # Execute a representative subset of examples under ctest so CI
    # catches link-time success followed by runtime crashes. Heavy
    # examples (quantum_spin_chain at 100 qubits, grover_large_scale)
    # are excluded — they're correct but multi-minute runs.
    if(QSIM_BUILD_TESTS)
        add_test(NAME example_vqe_h2_molecule
                 COMMAND vqe_h2_molecule)
        add_test(NAME example_qaoa_maxcut
                 COMMAND qaoa_maxcut)
        add_test(NAME example_grover_hash_collision
                 COMMAND grover_hash_collision)
        add_test(NAME example_quantum_spin_chain_small
                 COMMAND quantum_spin_chain 8 3 16)
        add_test(NAME example_phase3_phase4_benchmark
                 COMMAND phase3_phase4_benchmark 16)
        # CA-PEPS 2D TFIM smoke at 2x2 with short imag-time evolution;
        # full 3x3 sweep is the paper-grade benchmark archived under
        # benchmarks/results/.
        add_test(NAME example_ca_peps_2d_tfim_smoke
                 COMMAND example_ca_peps_2d_tfim
                         /tmp/ca_peps_tfim_smoke.json 2 2 8 0.1 30)
        # quantum_critical_point is a multi-minute finite-size-scaling
        # demo and is deliberately excluded from the CI smoke.
        set_tests_properties(
            example_vqe_h2_molecule
            example_qaoa_maxcut
            example_grover_hash_collision
            example_quantum_spin_chain_small
            example_phase3_phase4_benchmark
            example_ca_peps_2d_tfim_smoke
            PROPERTIES TIMEOUT 120
        )
    endif()

    message(STATUS "Examples: Enabled")
