//! Build script for moonlab-sys
//!
//! Uses bindgen to generate Rust FFI bindings from C headers
//! and links against the libquantumsim library.

use std::env;
use std::path::PathBuf;

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

    // Link against the quantum simulator static library
    println!("cargo:rustc-link-search=native={}", project_root.display());
    println!("cargo:rustc-link-lib=static=quantumsim");

    // Link OpenMP (required for parallel operations)
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
    println!("cargo:rustc-link-lib=dylib=omp");

    // Link system libraries (macOS)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    // Link math and C++ standard library
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=c++");

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
        .allowlist_function("entanglement_entropy")
        .allowlist_function("concurrence")
        .allowlist_function("negativity")
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
