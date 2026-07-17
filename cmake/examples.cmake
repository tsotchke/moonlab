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

        # Sharded random RZ+CNOT circuit (since v0.8.2) -- depth
        # layers of per-qubit RZ + alternating-parity CNOT chain.
        # Exercises dist_rz + dist_cnot under cross-rank traffic.
        add_executable(large_state_random_circuit examples/distributed/large_state_random_circuit.c)
        target_link_libraries(large_state_random_circuit PRIVATE quantumsim MPI::MPI_C)
    endif()

    # ------------------------------------------------------------------
    # CUDA examples (native moonlab CUDA state-vector backend).  Gated on
    # QSIM_HAS_CUDA, set by CMakeLists.txt's QSIM_ENABLE_CUDA block once
    # nvcc + the CUDA Toolkit are actually found.
    # ------------------------------------------------------------------
    if(QSIM_HAS_CUDA)
        # Standalone CUDA-backend smoke test: Bell state via the raw
        # moonlab_cuda_state_* API, no moonlab core dependency.  Compiled
        # as a .cu translation unit (nvcc) -- the tightest possible loop
        # to verify the CUDA backend on a fresh Jetson/discrete host.
        add_executable(bell_jetson examples/cuda/bell_jetson.cu)
        target_link_libraries(bell_jetson PRIVATE quantumsim)
        set_target_properties(bell_jetson PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION OFF
        )

        # Same Bell-state circuit as bell_jetson, but compiled as plain C
        # against the installed library -- proves the CUDA backend's
        # headers are C-callable and the symbols are exported from
        # libquantumsim.
        add_executable(bell_lib examples/cuda/bell_lib.c)
        target_link_libraries(bell_lib PRIVATE quantumsim)

        # v1.1 step 9 "transparent GPU dispatch" proof: Bell state built
        # with the SAME gate_hadamard()/gate_cnot() API the CPU tests use
        # -- the only difference is quantum_state_create_gpu() at
        # construction time.
        add_executable(state_gpu_bell examples/cuda/state_gpu_bell.c)
        target_link_libraries(state_gpu_bell PRIVATE quantumsim)

        # v1.1 step 9b: the full single- and two-qubit gate surface (H, X,
        # Y, Z, S, S^dagger, T, T^dagger, RX, RY, RZ, phase, CNOT, CZ,
        # SWAP, CPHASE, CY, CRX, CRY, CRZ) dispatches transparently to
        # GPU; asserts CPU/GPU amplitude parity to 1e-10.
        add_executable(state_gpu_circuit examples/cuda/state_gpu_circuit.c)
        target_link_libraries(state_gpu_circuit PRIVATE quantumsim)

        # v1.1 follow-up #4: Toffoli / Fredkin / MCX / MCZ multi-control
        # dispatch via the CUDA kernels; asserts CPU/GPU parity.
        add_executable(state_gpu_multiq examples/cuda/state_gpu_multiq.c)
        target_link_libraries(state_gpu_multiq PRIVATE quantumsim)

        if(QSIM_HAS_MPI)
            # v1.1 follow-up #6: sharded MPI + CUDA GHZ state -- each
            # rank's local shard lives on a CUDA GPU.  dist_hadamard(0) is
            # a local gate and dispatches entirely on GPU with no MPI;
            # dist_cnot across the partition boundary syncs GPU<->host
            # around the MPI exchange.  GPU counterpart to the plain
            # large_state_ghz example above.
            add_executable(large_state_ghz_gpu examples/cuda/large_state_ghz_gpu.c)
            target_link_libraries(large_state_ghz_gpu PRIVATE quantumsim MPI::MPI_C)

            # v1.1 step 11: MPI + CUDA together (trivially-parallel case)
            # -- each rank creates an independent GPU-backed state, runs a
            # different parameter point of a small variational circuit
            # locally, and MPI_Allreduce finds the cluster-wide minimum.
            add_executable(mpi_gpu_search examples/cuda/mpi_gpu_search.c)
            target_link_libraries(mpi_gpu_search PRIVATE quantumsim MPI::MPI_C)
        endif()
    endif()

    # ------------------------------------------------------------------
    # Metal GPU examples (macOS only).  Gated on QSIM_HAS_METAL, set by
    # CMakeLists.txt when QSIM_ENABLE_METAL is ON and the Metal/Foundation
    # frameworks are found.  Both link only against quantumsim -- the
    # Objective-C++ Metal bridge (gpu_metal.mm) is already compiled into
    # the shared library, so a plain C translation unit can call
    # metal_compute_init() etc. without linking the frameworks itself.
    # ------------------------------------------------------------------
    if(QSIM_HAS_METAL)
        # THE RIGHT WAY to benchmark Metal GPU: batches many independent
        # Grover searches into a single kernel launch and compares CPU
        # sequential / CPU OpenMP-parallel / GPU batch throughput.
        add_executable(metal_batch_benchmark examples/quantum/metal_batch_benchmark.c)
        target_link_libraries(metal_batch_benchmark PRIVATE quantumsim)
        if(QSIM_HAS_OPENMP)
            qsim_target_link_openmp(metal_batch_benchmark)
            if(QSIM_PLATFORM_MACOS AND DEFINED QSIM_OPENMP_LIBRARIES)
                target_compile_options(metal_batch_benchmark PRIVATE -Xpreprocessor -fopenmp)
            endif()
            target_compile_definitions(metal_batch_benchmark PRIVATE HAS_OPENMP=1)
        endif()

        # Metal GPU vs CPU benchmark: single Grover search plus individual
        # kernel-level benchmarks (Hadamard / oracle / diffusion).
        add_executable(metal_gpu_benchmark examples/quantum/metal_gpu_benchmark.c)
        target_link_libraries(metal_gpu_benchmark PRIVATE quantumsim)
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

    # QGT singularity == QEC node: one eps^2=0, tying the quantum-geometric-tensor and
    # topological-error-correction layers together on the Qi-Wu-Zhang model.
    add_executable(qgt_qec_node examples/applications/qgt_qec_node.c)
    target_link_libraries(qgt_qec_node PRIVATE quantumsim)

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

    # Open-core overlay demo (since v1.0.3): exercises every public
    # plug-in surface (backend, vendor-noise profile, decoder, and
    # scheduler completion hook) in a single executable.  Reference
    # for private overlays + sibling libraries that consume moonlab.
    add_executable(example_open_core_overlay_demo
        examples/extensions/open_core_overlay_demo.c)
    target_link_libraries(example_open_core_overlay_demo PRIVATE
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
        # Open-core overlay demo: registers all four runtime surfaces
        # and runs three jobs; smoke-test catches link + dispatch
        # regressions in the plug-in registry pipeline.
        add_test(NAME example_open_core_overlay_demo
                 COMMAND example_open_core_overlay_demo)
        # quantum_critical_point is a multi-minute finite-size-scaling
        # demo and is deliberately excluded from the CI smoke.
        set_tests_properties(
            example_vqe_h2_molecule
            example_qaoa_maxcut
            example_grover_hash_collision
            example_quantum_spin_chain_small
            example_phase3_phase4_benchmark
            example_ca_peps_2d_tfim_smoke
            example_open_core_overlay_demo
            PROPERTIES TIMEOUT 120
        )

        # `qsim_label_tests` is defined in cmake/tests.cmake, which the root
        # CMakeLists.txt includes BEFORE this file.  The "examples" call
        # over there (`qsim_label_tests(examples ...)`) therefore always
        # ran against a test set that didn't include these targets yet --
        # `if(TEST ...)` silently skipped every name, so `ctest -L examples`
        # returned zero tests despite the tests existing.  Re-issue the
        # same labeling here, after the targets above are actually
        # registered, so the "examples" label is real.
        qsim_label_tests(examples
            example_vqe_h2_molecule
            example_qaoa_maxcut
            example_grover_hash_collision
            example_quantum_spin_chain_small
            example_phase3_phase4_benchmark
            example_ca_peps_2d_tfim_smoke
            example_open_core_overlay_demo
        )
    endif()

    message(STATUS "Examples: Enabled")
