//! Build script for moonlab-sys
//!
//! Uses bindgen to generate Rust FFI bindings from C headers
//! and links against the libquantumsim library.

use std::env;
use std::path::{Path, PathBuf};

fn openmp_link_library_exists(dir: &Path) -> bool {
    dir.join("libomp.so").exists()
        || dir.join("libomp.a").exists()
        || dir.join("libomp.dylib").exists()
        || dir.join("omp.lib").exists()
        || dir.join("libomp.lib").exists()
}

fn emit_openmp_search_dir_if_present(dir: impl AsRef<Path>) -> bool {
    let dir = dir.as_ref();
    if openmp_link_library_exists(dir) {
        println!("cargo:rustc-link-search=native={}", dir.display());
        true
    } else {
        false
    }
}

fn main() {
    // Get the project root (3 levels up from bindings/rust/moonlab-sys)
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir)
        .parent().unwrap()  // rust/
        .parent().unwrap()  // bindings/
        .parent().unwrap()  // project root
        .to_path_buf();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}/src/quantum/state.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/quantum/gates.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/quantum/measurement.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/algorithms/grover.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/algorithms/vqe.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/algorithms/qaoa.h", project_root.display());
    println!("cargo:rerun-if-changed={}/src/visualization/feynman_diagram.h", project_root.display());

    // Link against the quantum simulator library. Prefer MOONLAB_LIB_DIR
    // if the build system sets it (e.g. CMake CTest), fall back to the
    // project root. Static preferred, but fall through to shared if
    // only libquantumsim.{so,dylib,dll} is present.
    let lib_dir = env::var("MOONLAB_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("build"));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let static_present = lib_dir.join("libquantumsim.a").exists();
    if static_present {
        println!("cargo:rustc-link-lib=static=quantumsim");
    } else {
        println!("cargo:rustc-link-lib=dylib=quantumsim");
        // Embed the dylib directory in the binary's rpath so cargo test
        // works without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    // Link OpenMP (required for parallel operations). Ubuntu's libomp-dev
    // installs libomp under /usr/lib/llvm-*/lib, which rust-lld does not search
    // by default.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=MOONLAB_OMP_DIR");
    println!("cargo:rerun-if-env-changed=MOONLAB_OPENMP_LIB_DIR");

    let mut found_openmp_dir = false;
    for var in ["MOONLAB_OMP_DIR", "MOONLAB_OPENMP_LIB_DIR"] {
        if let Ok(dir) = env::var(var) {
            println!("cargo:rustc-link-search=native={dir}");
            found_openmp_dir = true;
            break;
        }
    }

    if !found_openmp_dir && target_os == "macos" {
        if let Ok(brew) = env::var("HOMEBREW_PREFIX") {
            found_openmp_dir =
                emit_openmp_search_dir_if_present(PathBuf::from(&brew).join("opt/libomp/lib"));
        }
        if !found_openmp_dir {
            for p in ["/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"] {
                if emit_openmp_search_dir_if_present(p) {
                    found_openmp_dir = true;
                    break;
                }
            }
        }
    }

    if !found_openmp_dir && target_os == "linux" {
        for major in (10..=21).rev() {
            let candidate = PathBuf::from(format!("/usr/lib/llvm-{major}/lib"));
            if emit_openmp_search_dir_if_present(candidate) {
                found_openmp_dir = true;
                break;
            }
        }
        if !found_openmp_dir {
            for p in [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",
                "/usr/local/lib",
            ] {
                if emit_openmp_search_dir_if_present(p) {
                    break;
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=omp");

    // Link system libraries (macOS)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    // Link math and the platform C++ standard library.
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=m");
    }
    match target_os.as_str() {
        "linux" | "android" | "freebsd" | "openbsd" | "netbsd" => {
            println!("cargo:rustc-link-lib=stdc++");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        _ => {}
    }

    // Create wrapper header that includes all needed headers
    let wrapper_content = format!(r#"
// Wrapper header for bindgen
#include "{root}/src/quantum/state.h"
#include "{root}/src/quantum/gates.h"
#include "{root}/src/quantum/measurement.h"
#include "{root}/src/quantum/entanglement.h"
#include "{root}/src/quantum/noise.h"
#include "{root}/src/algorithms/grover.h"
#include "{root}/src/algorithms/vqe.h"
#include "{root}/src/algorithms/qaoa.h"
#include "{root}/src/utils/quantum_entropy.h"
#include "{root}/src/utils/config.h"
#include "{root}/src/visualization/feynman_diagram.h"
#include "{root}/src/applications/moonlab_export.h"
#include "{root}/src/applications/quantum_volume.h"
#include "{root}/src/backends/clifford/clifford.h"
#include "{root}/src/algorithms/topology_realspace/chern_kpm.h"
#include "{root}/src/algorithms/topology_realspace/chern_marker.h"
#include "{root}/src/algorithms/quantum_geometry/qgt.h"
#include "{root}/src/optimization/fusion/fusion.h"
"#, root = project_root.display());

    let wrapper_path = PathBuf::from(&manifest_dir).join("wrapper.h");
    std::fs::write(&wrapper_path, wrapper_content).expect("Failed to write wrapper.h");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_str().unwrap())
        // Include paths
        .clang_arg(format!("-I{}", project_root.display()))
        .clang_arg(format!("-I{}/src", project_root.display()))
        // Core quantum types
        .allowlist_type("quantum_state_t")
        .allowlist_type("complex_t")
        .allowlist_type("qs_error_t")
        .allowlist_type("measurement_result_t")
        .allowlist_type("measurement_basis_t")
        // State functions
        .allowlist_function("quantum_state_init")
        .allowlist_function("quantum_state_free")
        .allowlist_function("quantum_state_clone")
        .allowlist_function("quantum_state_reset")
        .allowlist_function("quantum_state_is_normalized")
        .allowlist_function("quantum_state_normalize")
        .allowlist_function("quantum_state_entropy")
        .allowlist_function("quantum_state_purity")
        .allowlist_function("quantum_state_fidelity")
        .allowlist_function("quantum_state_entanglement_entropy")
        // Gate functions - single qubit
        .allowlist_function("gate_pauli_x")
        .allowlist_function("gate_pauli_y")
        .allowlist_function("gate_pauli_z")
        .allowlist_function("gate_hadamard")
        .allowlist_function("gate_s")
        .allowlist_function("gate_s_dagger")
        .allowlist_function("gate_t")
        .allowlist_function("gate_t_dagger")
        .allowlist_function("gate_rx")
        .allowlist_function("gate_ry")
        .allowlist_function("gate_rz")
        .allowlist_function("gate_phase")
        .allowlist_function("gate_u3")
        // Gate functions - two qubit
        .allowlist_function("gate_cnot")
        .allowlist_function("gate_cz")
        .allowlist_function("gate_cy")
        .allowlist_function("gate_swap")
        .allowlist_function("gate_crx")
        .allowlist_function("gate_cry")
        .allowlist_function("gate_crz")
        .allowlist_function("gate_cphase")
        // Gate functions - multi qubit
        .allowlist_function("gate_toffoli")
        .allowlist_function("gate_fredkin")
        .allowlist_function("gate_mcx")
        .allowlist_function("gate_mcz")
        .allowlist_function("gate_qft")
        .allowlist_function("gate_iqft")
        // Measurement functions
        .allowlist_function("quantum_measure")
        .allowlist_function("quantum_measure_multi")
        .allowlist_function("quantum_measure_all")
        .allowlist_function("quantum_measure_all_fast")
        .allowlist_function("quantum_peek_probability")
        .allowlist_function("measurement_probability_one")
        .allowlist_function("measurement_probability_zero")
        .allowlist_function("measurement_all_probabilities")
        .allowlist_function("measurement_expectation_z")
        .allowlist_function("measurement_expectation_x")
        .allowlist_function("measurement_expectation_y")
        .allowlist_function("measurement_correlation_zz")
        // Grover's algorithm
        .allowlist_type("grover_config_t")
        .allowlist_type("grover_result_t")
        .allowlist_function("grover_search")
        .allowlist_function("grover_optimal_iterations")
        .allowlist_function("grover_oracle")
        .allowlist_function("grover_diffusion")
        // VQE types and functions
        .allowlist_type("pauli_term_t")
        .allowlist_type("pauli_hamiltonian_t")
        .allowlist_type("vqe_ansatz_t")
        .allowlist_type("vqe_ansatz_type_t")
        .allowlist_type("vqe_optimizer_t")
        .allowlist_type("vqe_optimizer_type_t")
        .allowlist_type("vqe_solver_t")
        .allowlist_type("vqe_result_t")
        .allowlist_function("vqe_create_h2_hamiltonian")
        .allowlist_function("vqe_create_lih_hamiltonian")
        .allowlist_function("vqe_create_hardware_efficient_ansatz")
        .allowlist_function("vqe_create_uccsd_ansatz")
        .allowlist_function("vqe_optimizer_create")
        .allowlist_function("vqe_optimizer_free")
        .allowlist_function("vqe_solver_create")
        .allowlist_function("vqe_solver_free")
        .allowlist_function("vqe_solve")
        .allowlist_function("vqe_compute_energy")
        .allowlist_function("vqe_compute_gradient")
        .allowlist_function("vqe_apply_ansatz")
        .allowlist_function("pauli_hamiltonian_free")
        .allowlist_function("vqe_ansatz_free")
        // QAOA types and functions
        .allowlist_type("ising_model_t")
        .allowlist_type("graph_t")
        .allowlist_type("qaoa_config_t")
        .allowlist_type("qaoa_result_t")
        .allowlist_type("qaoa_solver_t")
        .allowlist_function("ising_encode_maxcut")
        .allowlist_function("qaoa_solver_create")
        .allowlist_function("qaoa_solver_free")
        .allowlist_function("qaoa_solve")
        .allowlist_function("qaoa_compute_expectation")
        .allowlist_function("qaoa_apply_circuit")
        .allowlist_function("qaoa_sample_solution")
        .allowlist_function("qaoa_approximation_ratio")
        .allowlist_function("graph_create")
        .allowlist_function("graph_free")
        .allowlist_function("graph_add_edge")
        .allowlist_function("ising_model_free")
        // Entropy context
        .allowlist_type("quantum_entropy_ctx_t")
        .allowlist_type("quantum_entropy_fn")
        .allowlist_function("quantum_entropy_init")
        .allowlist_function("quantum_entropy_get_bytes")
        // Entanglement
        .allowlist_function("entanglement_entropy_bipartition")
        .allowlist_function("entanglement_concurrence_2qubit")
        .allowlist_function("entanglement_negativity_2qubit")
        .allowlist_function("entanglement_renyi_entropy")
        // Configuration
        .allowlist_type("qsim_config_t")
        .allowlist_type("qsim_backend_t")
        .allowlist_function("qsim_config_init")
        .allowlist_function("qsim_config_set_backend")
        // Feynman diagrams
        .allowlist_type("particle_type_t")
        .allowlist_type("feynman_vertex_t")
        .allowlist_type("feynman_propagator_t")
        .allowlist_type("feynman_diagram_t")
        .allowlist_type("feynman_options_t")
        .allowlist_function("feynman_create")
        .allowlist_function("feynman_free")
        .allowlist_function("feynman_set_title")
        .allowlist_function("feynman_set_loop_order")
        .allowlist_function("feynman_add_vertex")
        .allowlist_function("feynman_add_vertex_labeled")
        .allowlist_function("feynman_add_external_vertex")
        .allowlist_function("feynman_add_fermion")
        .allowlist_function("feynman_add_antifermion")
        .allowlist_function("feynman_add_photon")
        .allowlist_function("feynman_add_gluon")
        .allowlist_function("feynman_add_w_boson")
        .allowlist_function("feynman_add_z_boson")
        .allowlist_function("feynman_add_higgs")
        .allowlist_function("feynman_add_scalar")
        .allowlist_function("feynman_add_propagator")
        .allowlist_function("feynman_add_incoming")
        .allowlist_function("feynman_add_outgoing")
        .allowlist_function("feynman_create_qed_vertex")
        .allowlist_function("feynman_create_ee_to_mumu")
        .allowlist_function("feynman_create_compton")
        .allowlist_function("feynman_create_pair_annihilation")
        .allowlist_function("feynman_create_electron_self_energy")
        .allowlist_function("feynman_create_vacuum_polarization")
        .allowlist_function("feynman_create_moller_scattering")
        .allowlist_function("feynman_create_bhabha_scattering")
        .allowlist_function("feynman_default_options")
        .allowlist_function("feynman_publication_options")
        .allowlist_function("feynman_render_ascii")
        .allowlist_function("feynman_print_ascii")
        .allowlist_function("feynman_render_svg")
        .allowlist_function("feynman_save_svg")
        .allowlist_function("feynman_render_latex")
        .allowlist_function("feynman_save_latex")
        .allowlist_function("feynman_get_vertex")
        .allowlist_function("feynman_update_bounds")
        .allowlist_function("feynman_particle_type_name")
        .allowlist_function("feynman_tikz_style")
        // Stable ABI surface (dlsym-able across 0.x)
        .allowlist_function("moonlab_abi_version")
        .allowlist_function("moonlab_qrng_bytes")
        .allowlist_function("moonlab_qwz_chern")
        // Clifford stabilizer backend
        .allowlist_type("clifford_tableau_t")
        .allowlist_type("clifford_error_t")
        .allowlist_function("clifford_tableau_create")
        .allowlist_function("clifford_tableau_free")
        .allowlist_function("clifford_num_qubits")
        .allowlist_function("clifford_h")
        .allowlist_function("clifford_s")
        .allowlist_function("clifford_s_dag")
        .allowlist_function("clifford_x")
        .allowlist_function("clifford_y")
        .allowlist_function("clifford_z")
        .allowlist_function("clifford_cnot")
        .allowlist_function("clifford_cz")
        .allowlist_function("clifford_swap")
        .allowlist_function("clifford_measure")
        .allowlist_function("clifford_sample_all")
        // Quantum Volume
        .allowlist_type("qv_result_t")
        .allowlist_function("quantum_volume_run")
        // Real-space Chern marker (dense + matrix-free)
        .allowlist_type("chern_system_t")
        .allowlist_function("chern_qwz_create")
        .allowlist_function("chern_system_free")
        .allowlist_function("chern_build_projector")
        .allowlist_function("chern_local_marker")
        .allowlist_function("chern_bulk_sum")
        .allowlist_type("chern_kpm_system_t")
        .allowlist_function("chern_kpm_create")
        .allowlist_function("chern_kpm_free")
        .allowlist_function("chern_kpm_local_marker")
        .allowlist_function("chern_kpm_bulk_sum")
        .allowlist_function("chern_kpm_bulk_map")
        .allowlist_function("chern_kpm_set_modulation")
        .allowlist_function("chern_kpm_cn_modulation")
        // Quantum geometric tensor
        .allowlist_type("qgt_system_t")
        .allowlist_type("qgt_system_1d_t")
        .allowlist_type("qgt_berry_grid_t")
        .allowlist_function("qgt_create")
        .allowlist_function("qgt_free")
        .allowlist_function("qgt_model_qwz")
        .allowlist_function("qgt_model_haldane")
        .allowlist_function("qgt_model_ssh")
        .allowlist_function("qgt_create_1d")
        .allowlist_function("qgt_free_1d")
        .allowlist_function("qgt_winding_1d")
        .allowlist_function("qgt_berry_grid")
        .allowlist_function("qgt_berry_grid_free")
        .allowlist_function("qgt_metric_at")
        .allowlist_function("qgt_wilson_loop")
        // Gate-fusion DAG
        .allowlist_type("fuse_circuit_t")
        .allowlist_type("fuse_gate_t")
        .allowlist_type("fuse_gate_kind_t")
        .allowlist_type("fuse_stats_t")
        .allowlist_function("fuse_circuit_create")
        .allowlist_function("fuse_circuit_free")
        .allowlist_function("fuse_circuit_len")
        .allowlist_function("fuse_circuit_num_qubits")
        .allowlist_function("fuse_append_h")
        .allowlist_function("fuse_append_x")
        .allowlist_function("fuse_append_y")
        .allowlist_function("fuse_append_z")
        .allowlist_function("fuse_append_s")
        .allowlist_function("fuse_append_sdg")
        .allowlist_function("fuse_append_t")
        .allowlist_function("fuse_append_tdg")
        .allowlist_function("fuse_append_phase")
        .allowlist_function("fuse_append_rx")
        .allowlist_function("fuse_append_ry")
        .allowlist_function("fuse_append_rz")
        .allowlist_function("fuse_append_u3")
        .allowlist_function("fuse_append_cnot")
        .allowlist_function("fuse_append_cz")
        .allowlist_function("fuse_append_cy")
        .allowlist_function("fuse_append_swap")
        .allowlist_function("fuse_append_cphase")
        .allowlist_function("fuse_append_crx")
        .allowlist_function("fuse_append_cry")
        .allowlist_function("fuse_append_crz")
        .allowlist_function("fuse_compile")
        .allowlist_function("fuse_execute")
        // Layout settings
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .size_t_is_usize(true)
        .generate_comments(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings to OUT_DIR
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Clean up wrapper
    let _ = std::fs::remove_file(wrapper_path);
}
