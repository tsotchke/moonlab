//! Build script for moonlab-sys
//!
//! Uses bindgen to generate Rust FFI bindings from C headers
//! and links against the libquantumsim library.

use std::env;
use std::path::{Path, PathBuf};

fn normalize_header_root(candidate: &Path) -> Option<PathBuf> {
    if candidate.join("quantum/state.h").is_file() {
        return Some(candidate.to_path_buf());
    }
    let installed = candidate.join("quantumsim");
    if installed.join("quantum/state.h").is_file() {
        return Some(installed);
    }
    None
}

// MSVC static and shared builds can both produce a bare `quantumsim.lib`:
// CMakeLists.txt does not rename the STATIC target's output, so a static
// build's archive is *also* named `quantumsim.lib`, identically to a
// shared build's DLL import library. The file's presence alone is
// therefore ambiguous on Windows -- it is only an import library (shared)
// when `quantumsim.dll` sits next to it; with no `.dll` alongside it, it
// is the static archive. `has_shared_library` and `has_static_library`
// must agree on this so a directory containing only the static archive
// is never linked as `dylib=quantumsim` (which would ask the linker to
// resolve DLL-only symbols against an archive that has none).
fn has_dll(dir: &Path) -> bool {
    dir.join("quantumsim.dll").is_file()
}

fn has_shared_library(dir: &Path) -> bool {
    dir.join("libquantumsim.so").is_file() || dir.join("libquantumsim.dylib").is_file() || has_dll(dir)
}

fn has_static_library(dir: &Path) -> bool {
    dir.join("libquantumsim.a").is_file()
        || dir.join("quantumsim_static.lib").is_file()
        // A bare `.lib` with no matching `.dll` is the MSVC static
        // archive, not an import library.
        || (dir.join("quantumsim.lib").is_file() && !has_dll(dir))
}

/// Returns the `cargo:rustc-link-lib=<kind>=<name>` pair to emit for
/// whatever Moonlab library is actually present in `dir`, or `None` if
/// nothing usable is there. Distinguishing `quantumsim` from
/// `quantumsim_static` matters on MSVC, where the link name must match
/// the archive's real filename exactly.
fn classify_library(dir: &Path) -> Option<(&'static str, &'static str)> {
    if dir.join("libquantumsim.so").is_file() || dir.join("libquantumsim.dylib").is_file() {
        return Some(("dylib", "quantumsim"));
    }
    if has_dll(dir) {
        return Some(("dylib", "quantumsim"));
    }
    if dir.join("libquantumsim.a").is_file() {
        return Some(("static", "quantumsim"));
    }
    if dir.join("quantumsim_static.lib").is_file() {
        return Some(("static", "quantumsim_static"));
    }
    if dir.join("quantumsim.lib").is_file() {
        // No matching .dll => the MSVC static archive.
        return Some(("static", "quantumsim"));
    }
    None
}

fn link_from_directory(dir: &Path) {
    let Some((kind, name)) = classify_library(dir) else {
        panic!(
            "MOONLAB_LIB_DIR={} does not contain a Moonlab shared or static library",
            dir.display()
        );
    };

    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib={kind}={name}");
    if kind == "dylib" && env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let source_headers = project_root.join("src");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MOONLAB_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MOONLAB_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let explicit_lib_dir = env::var_os("MOONLAB_LIB_DIR").map(PathBuf::from);
    let explicit_include_dir = env::var_os("MOONLAB_INCLUDE_DIR").map(PathBuf::from);
    let source_tree_available = source_headers.join("quantum/state.h").is_file();

    // Installed crates discover the native SDK through pkg-config. Contributors
    // can still point at an arbitrary build/install tree with the two MOONLAB_*
    // variables, and in-repository builds retain their source-tree fallback.
    let need_pkg_config =
        explicit_lib_dir.is_none() || (explicit_include_dir.is_none() && !source_tree_available);
    let pkg = if need_pkg_config {
        pkg_config::Config::new()
            .atleast_version("1.1.0")
            .cargo_metadata(explicit_lib_dir.is_none())
            .probe("quantumsim")
            .ok()
    } else {
        None
    };

    if let Some(lib_dir) = explicit_lib_dir.as_deref() {
        link_from_directory(lib_dir);
    } else if pkg.is_none() {
        let development_lib_dir = project_root.join("build");
        if source_tree_available
            && (has_shared_library(&development_lib_dir)
                || has_static_library(&development_lib_dir))
        {
            link_from_directory(&development_lib_dir);
        } else {
            panic!(
                "Moonlab native SDK not found. Install Moonlab so pkg-config can find \
                 quantumsim, or set MOONLAB_LIB_DIR and MOONLAB_INCLUDE_DIR."
            );
        }
    } else if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        for lib_dir in &pkg.as_ref().unwrap().link_paths {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
    }

    let header_root = explicit_include_dir
        .as_deref()
        .and_then(normalize_header_root)
        .or_else(|| {
            pkg.as_ref().and_then(|library| {
                library
                    .include_paths
                    .iter()
                    .find_map(|path| normalize_header_root(path))
            })
        })
        .or_else(|| source_tree_available.then_some(source_headers.clone()))
        .unwrap_or_else(|| {
            panic!(
                "Moonlab C headers not found. Set MOONLAB_INCLUDE_DIR to either \
                 <prefix>/include or <prefix>/include/quantumsim."
            )
        });

    for header in [
        "quantum/state.h",
        "quantum/gates.h",
        "quantum/measurement.h",
        "algorithms/grover.h",
        "algorithms/vqe.h",
        "algorithms/qaoa.h",
        "visualization/feynman_diagram.h",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            header_root.join(header).display()
        );
    }

    // Create wrapper header that includes all needed headers
    let wrapper_content = r#"
// Wrapper header for bindgen
#include "quantum/state.h"
#include "quantum/gates.h"
#include "quantum/measurement.h"
#include "quantum/entanglement.h"
#include "quantum/noise.h"
#include "quantum/noise_mpdo.h"
#include "algorithms/grover.h"
#include "algorithms/bell_tests.h"
#include "algorithms/vqe.h"
#include "algorithms/qaoa.h"
#include "utils/quantum_entropy.h"
#include "utils/config.h"
#include "visualization/feynman_diagram.h"
#include "applications/moonlab_export.h"
#include "applications/quantum_volume.h"
#include "backends/clifford/clifford.h"
#include "algorithms/topology_realspace/chern_kpm.h"
#include "algorithms/topology_realspace/chern_marker.h"
#include "algorithms/quantum_geometry/qgt.h"
#include "optimization/fusion/fusion.h"
#include "algorithms/tensor_network/tn_state.h"
#include "algorithms/tensor_network/dmrg.h"
#include "algorithms/tensor_network/tdvp.h"
#include "algorithms/tensor_network/ca_mps.h"
#include "algorithms/tensor_network/ca_peps.h"
#include "algorithms/topological/topological.h"
#include "integration/libirrep_bridge.h"
#include "applications/moonlab_qgtl_backend.h"
#include "applications/decoder_bench.h"
#include "applications/vendor_noise_backend.h"
#include "distributed/scheduler.h"
#include "control/control_plane.h"
"#;

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_path = out_path.join("wrapper.h");
    std::fs::write(&wrapper_path, wrapper_content).expect("Failed to write wrapper.h");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_str().unwrap())
        // Include paths
        .clang_arg(format!("-I{}", header_root.display()))
        .clang_arg(format!("-I{}", header_root.parent().unwrap().display()))
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
        .allowlist_function("vqe_create_h2o_hamiltonian")
        .allowlist_function("vqe_create_hardware_efficient_ansatz")
        .allowlist_function("vqe_create_uccsd_ansatz")
        .allowlist_function("vqe_optimizer_create")
        .allowlist_function("vqe_optimizer_free")
        .allowlist_function("vqe_solver_create")
        .allowlist_function("vqe_solver_free")
        .allowlist_function("vqe_solve")
        .allowlist_function("vqe_compute_energy")
        .allowlist_function("vqe_compute_gradient")
        .allowlist_function("moonlab_vqe_gradient")
        .allowlist_function("vqe_apply_ansatz")
        .allowlist_function("vqe_exact_ground_state_energy")
        .allowlist_function("vqe_hartree_to_kcalmol")
        .allowlist_function("pauli_hamiltonian_create")
        .allowlist_function("pauli_hamiltonian_add_term")
        .allowlist_function("pauli_hamiltonian_free")
        .allowlist_function("vqe_ansatz_free")
        // QAOA types and functions
        .allowlist_type("ising_model_t")
        .allowlist_type("graph_t")
        .allowlist_type("qaoa_config_t")
        .allowlist_type("qaoa_result_t")
        .allowlist_type("qaoa_solver_t")
        .allowlist_function("ising_encode_maxcut")
        .allowlist_function("ising_model_create")
        .allowlist_function("ising_model_free")
        .allowlist_function("ising_model_set_coupling")
        .allowlist_function("ising_model_set_field")
        .allowlist_function("ising_model_evaluate")
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
        // Entropy context
        .allowlist_type("quantum_entropy_ctx_t")
        .allowlist_type("quantum_entropy_fn")
        .allowlist_function("quantum_entropy_init")
        .allowlist_function("quantum_entropy_get_bytes")
        .allowlist_function("quantum_entropy_ctx_create_hw")
        .allowlist_function("quantum_entropy_ctx_destroy")
        // Entanglement
        .allowlist_function("entanglement_entropy_bipartition")
        .allowlist_function("entanglement_concurrence_2qubit")
        .allowlist_function("entanglement_negativity_2qubit")
        .allowlist_function("entanglement_renyi_entropy")
        .allowlist_function("entanglement_mutual_information")
        // Surface code (Clifford-tableau variant; Rust wrapper from 0.5.12).
        .allowlist_type("surface_code_clifford_t")
        .allowlist_function("surface_code_clifford_create")
        .allowlist_function("surface_code_clifford_free")
        .allowlist_function("surface_code_clifford_data_index")
        .allowlist_function("surface_code_clifford_apply_error")
        .allowlist_function("surface_code_clifford_measure_z_syndromes")
        .allowlist_function("surface_code_clifford_measure_x_syndromes")
        .allowlist_function("surface_code_clifford_syndrome_weight")
        // libirrep bridge (since v0.6.0) -- behind QSIM_ENABLE_LIBIRREP
        // at the C build level; the symbols always link (no-op stubs
        // when libirrep isn't found), so the Rust surface is uniform.
        .allowlist_type("moonlab_libirrep_qec_t")
        .allowlist_type("moonlab_libirrep_lattice_kind_t")
        .allowlist_type("moonlab_libirrep_wallpaper_t")
        .allowlist_function("moonlab_libirrep_available")
        .allowlist_function("moonlab_libirrep_kagome12_e0")
        .allowlist_function("moonlab_libirrep_heisenberg_sector_e0")
        .allowlist_function("moonlab_libirrep_surface_code_new")
        .allowlist_function("moonlab_libirrep_toric_code_new")
        .allowlist_function("moonlab_libirrep_color_steane_new")
        .allowlist_function("moonlab_libirrep_color_hamming_15_7_3_new")
        .allowlist_function("moonlab_libirrep_bb_72_12_6_new")
        .allowlist_function("moonlab_libirrep_bb_144_12_12_new")
        .allowlist_function("moonlab_libirrep_bb_288_12_18_new")
        .allowlist_function("moonlab_libirrep_hgp_repetition_new")
        .allowlist_function("moonlab_libirrep_qec_free")
        .allowlist_function("moonlab_libirrep_qec_n_qubits")
        .allowlist_function("moonlab_libirrep_qec_n_x_stabs")
        .allowlist_function("moonlab_libirrep_qec_n_z_stabs")
        .allowlist_function("moonlab_libirrep_qec_logical_qubits")
        .allowlist_function("moonlab_libirrep_qec_distance")
        .allowlist_function("moonlab_libirrep_qec_get_x_check_row")
        .allowlist_function("moonlab_libirrep_qec_get_z_check_row")
        // QGTL ingestion surface (since v0.6.6).
        .allowlist_type("moonlab_qgtl_circuit")
        .allowlist_type("moonlab_qgtl_gate_t")
        .allowlist_type("moonlab_qgtl_exec_options_t")
        .allowlist_type("moonlab_qgtl_results_t")
        .allowlist_function("moonlab_qgtl_circuit_create")
        .allowlist_function("moonlab_qgtl_circuit_free")
        .allowlist_function("moonlab_qgtl_add_gate")
        .allowlist_function("moonlab_qgtl_execute")
        .allowlist_function("moonlab_qgtl_results_free")
        .allowlist_function("moonlab_qgtl_circuit_num_qubits")
        .allowlist_function("moonlab_qgtl_circuit_num_gates")
        // Portable circuit serialization (since v0.8.3).
        .allowlist_function("moonlab_qgtl_circuit_serialize")
        .allowlist_function("moonlab_qgtl_circuit_deserialize")
        .allowlist_function("moonlab_qgtl_circuit_save")
        .allowlist_function("moonlab_qgtl_circuit_load")
        // TCP control plane (since v0.8.7).
        .allowlist_function("moonlab_control_serve")
        .allowlist_function("moonlab_control_submit_circuit")
        .allowlist_function("moonlab_control_submit_circuit_shots")
        // Lifecycle API (since v0.8.13).
        .allowlist_function("moonlab_control_server_open")
        .allowlist_function("moonlab_control_server_run")
        .allowlist_function("moonlab_control_server_shutdown")
        .allowlist_function("moonlab_control_server_close")
        // HMAC-SHA3-256 auth (since v0.8.15).
        .allowlist_function("moonlab_control_server_set_secret")
        .allowlist_function("moonlab_control_submit_circuit_auth")
        .allowlist_function("moonlab_control_hmac_sha3_256")
        // TLS transport (since v0.8.17).
        .allowlist_function("moonlab_control_server_use_tls")
        .allowlist_function("moonlab_control_submit_circuit_tls")
        // mTLS (since v0.8.19).
        .allowlist_function("moonlab_control_server_require_client_cert")
        .allowlist_function("moonlab_control_submit_circuit_mtls")
        // HEALTH + rate limit (since v0.8.21).
        .allowlist_function("moonlab_control_server_set_rate_limit")
        .allowlist_function("moonlab_control_submit_health")
        // METRICS (since v0.8.23).
        .allowlist_function("moonlab_control_submit_metrics")
        // Per-request socket timeout (since v0.8.26).
        .allowlist_function("moonlab_control_server_set_request_timeout")
        // Concurrent-connection cap (since v0.9.0).
        .allowlist_function("moonlab_control_server_set_max_concurrent")
        // Tenant identity + admission hook (since v1.0.3).
        .allowlist_function("moonlab_control_submit_circuit_auth_tenant")
        .allowlist_function("moonlab_control_server_set_admission_hook")
        .allowlist_type("moonlab_admission_hook_fn")
        // Decoder-bench dispatcher (since v0.6.7).
        .allowlist_type("moonlab_decoder_kind_t")
        .allowlist_type("moonlab_decoder_code_t")
        .allowlist_type("moonlab_decoder_input_t")
        .allowlist_function("moonlab_decoder_decode")
        .allowlist_function("moonlab_decoder_slot_available")
        .allowlist_function("moonlab_decoder_slot_name")
        // Decoder runtime registry (since v1.0.3).
        .allowlist_type("moonlab_decoder_fn")
        .allowlist_type("moonlab_decoder_entry_t")
        .allowlist_function("moonlab_register_decoder")
        .allowlist_function("moonlab_unregister_decoder")
        .allowlist_function("moonlab_lookup_decoder")
        .allowlist_function("moonlab_decoder_decode_by_name")
        .allowlist_function("moonlab_num_decoders")
        .allowlist_function("moonlab_list_decoders")
        // Vendor-noise profile registry (since v1.0.3).
        .allowlist_type("moonlab_vendor_noise_profile_t")
        .allowlist_function("moonlab_register_vendor_noise_profile")
        .allowlist_function("moonlab_unregister_vendor_noise_profile")
        .allowlist_function("moonlab_lookup_vendor_noise_profile")
        .allowlist_function("moonlab_num_vendor_noise_profiles")
        .allowlist_function("moonlab_list_vendor_noise_profiles")
        // Scheduler completion hook (since v1.0.3).
        .allowlist_type("moonlab_completion_hook_fn")
        .allowlist_function("moonlab_scheduler_set_completion_hook")
        // Distributed scheduler (since v0.7.0).
        .allowlist_type("moonlab_job")
        .allowlist_type("moonlab_job_results_t")
        .allowlist_function("moonlab_job_create")
        .allowlist_function("moonlab_job_free")
        .allowlist_function("moonlab_job_add_gate")
        .allowlist_function("moonlab_job_set_num_shots")
        .allowlist_function("moonlab_job_set_num_workers")
        .allowlist_function("moonlab_job_set_rng_seed")
        .allowlist_function("moonlab_job_num_qubits")
        .allowlist_function("moonlab_job_num_gates")
        .allowlist_function("moonlab_job_num_shots")
        .allowlist_function("moonlab_job_num_workers")
        .allowlist_function("moonlab_scheduler_run")
        .allowlist_function("moonlab_job_results_free")
        .allowlist_function("moonlab_job_to_json")
        // CA-PEPS 2D Clifford-assisted simulator (since 0.2.1; Rust wrapper from 0.4.11).
        .allowlist_type("ca_peps_error_t")
        .allowlist_type("moonlab_ca_peps_t")
        .allowlist_function("moonlab_ca_peps_create")
        .allowlist_function("moonlab_ca_peps_free")
        .allowlist_function("moonlab_ca_peps_clone")
        .allowlist_function("moonlab_ca_peps_lx")
        .allowlist_function("moonlab_ca_peps_ly")
        .allowlist_function("moonlab_ca_peps_num_qubits")
        .allowlist_function("moonlab_ca_peps_max_bond_dim")
        .allowlist_function("moonlab_ca_peps_current_bond_dim")
        .allowlist_function("moonlab_ca_peps_max_half_cut_entropy")
        .allowlist_function("moonlab_ca_peps_h")
        .allowlist_function("moonlab_ca_peps_s")
        .allowlist_function("moonlab_ca_peps_sdag")
        .allowlist_function("moonlab_ca_peps_x")
        .allowlist_function("moonlab_ca_peps_y")
        .allowlist_function("moonlab_ca_peps_z")
        .allowlist_function("moonlab_ca_peps_cnot")
        .allowlist_function("moonlab_ca_peps_cz")
        .allowlist_function("moonlab_ca_peps_rx")
        .allowlist_function("moonlab_ca_peps_ry")
        .allowlist_function("moonlab_ca_peps_rz")
        .allowlist_function("moonlab_ca_peps_t_gate")
        .allowlist_function("moonlab_ca_peps_t_dagger")
        .allowlist_function("moonlab_ca_peps_phase")
        .allowlist_function("moonlab_ca_peps_normalize")
        .allowlist_function("moonlab_ca_peps_norm")
        .allowlist_function("moonlab_ca_peps_expect_pauli")
        .allowlist_function("moonlab_ca_peps_prob_z")
        // Single-qubit Kraus noise channels (since 0.2.1; Rust wrapper from 0.4.8).
        .allowlist_function("noise_depolarizing_single")
        .allowlist_function("noise_depolarizing_two_qubit")
        .allowlist_function("noise_amplitude_damping")
        .allowlist_function("noise_phase_damping")
        .allowlist_function("noise_pure_dephasing")
        .allowlist_function("noise_bit_flip")
        .allowlist_function("noise_phase_flip")
        .allowlist_function("noise_bit_phase_flip")
        .allowlist_function("noise_thermal_relaxation")
        .allowlist_function("noise_readout_error")
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
        // MPDO noise simulator (v0.3)
        .allowlist_type("moonlab_mpdo_t")
        .allowlist_type("mpdo_complex_t")
        .allowlist_type("mpdo_error_t")
        .allowlist_function("moonlab_mpdo_create")
        .allowlist_function("moonlab_mpdo_free")
        .allowlist_function("moonlab_mpdo_clone")
        .allowlist_function("moonlab_mpdo_num_qubits")
        .allowlist_function("moonlab_mpdo_max_bond_dim")
        .allowlist_function("moonlab_mpdo_current_bond_dim")
        .allowlist_function("moonlab_mpdo_trace")
        .allowlist_function("moonlab_mpdo_apply_kraus_1q")
        .allowlist_function("moonlab_mpdo_apply_depolarizing_1q")
        .allowlist_function("moonlab_mpdo_apply_amplitude_damping_1q")
        .allowlist_function("moonlab_mpdo_apply_phase_damping_1q")
        .allowlist_function("moonlab_mpdo_apply_bit_flip_1q")
        .allowlist_function("moonlab_mpdo_apply_phase_flip_1q")
        .allowlist_function("moonlab_mpdo_apply_bit_phase_flip_1q")
        .allowlist_function("moonlab_mpdo_expect_pauli_1q")
        // TDVP + MPO + MPS surface (v0.4 adaptive-bond TDVP).
        .allowlist_type("tn_mps_state_t")
        .allowlist_type("tn_state_config_t")
        .allowlist_type("tn_canonical_form_t")
        .allowlist_type("mpo_t")
        .allowlist_type("tdvp_config_t")
        .allowlist_type("tdvp_adaptive_bond_config_t")
        .allowlist_type("tdvp_result_t")
        .allowlist_type("tdvp_history_t")
        .allowlist_type("tdvp_engine_t")
        .allowlist_type("tdvp_evolution_type_t")
        .allowlist_type("tdvp_variant_t")
        .allowlist_type("integrator_type_t")
        .allowlist_function("tn_state_config_create")
        .allowlist_function("tn_mps_free")
        .allowlist_function("mpo_heisenberg_create")
        .allowlist_function("mpo_tfim_create")
        .allowlist_function("mpo_free")
        .allowlist_function("dmrg_init_random_mps")
        .allowlist_function("tdvp_engine_create")
        .allowlist_function("tdvp_engine_free")
        .allowlist_function("tdvp_step")
        .allowlist_function("tdvp_evolve_to")
        .allowlist_function("tdvp_set_dt")
        .allowlist_function("tdvp_get_time")
        .allowlist_function("tdvp_bond_chi")
        .allowlist_function("tdvp_result_clear")
        .allowlist_function("tdvp_history_create")
        .allowlist_function("tdvp_history_free")
        .allowlist_function("tdvp_history_add")
        .allowlist_function("tdvp_history_add_with_observable")
        .allowlist_function("tdvp_evolve_to_with_observable")
        .allowlist_function("tdvp_evolve_with_observables")
        // v0.4.1 stable-ABI TDVP wrapper surface.
        .allowlist_function("moonlab_tdvp_create_heisenberg")
        .allowlist_function("moonlab_tdvp_create_tfim")
        .allowlist_function("moonlab_tdvp_step")
        .allowlist_function("moonlab_tdvp_evolve_to")
        .allowlist_function("moonlab_tdvp_current_time")
        .allowlist_function("moonlab_tdvp_current_energy")
        .allowlist_function("moonlab_tdvp_current_norm")
        .allowlist_function("moonlab_tdvp_current_max_bond_dim")
        .allowlist_function("moonlab_tdvp_num_bonds")
        .allowlist_function("moonlab_tdvp_bond_chi")
        .allowlist_function("moonlab_tdvp_history_num_steps")
        .allowlist_function("moonlab_tdvp_history_get_step")
        .allowlist_function("moonlab_tdvp_history_get_bond_chi")
        .allowlist_function("moonlab_tdvp_engine_free")
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
        .allowlist_function("qgt_berry_grid_proj")
        .allowlist_function("qgt_berry_grid_pt")
        .allowlist_function("qgt_berry_grid_free")
        .allowlist_function("qgt_metric_at")
        .allowlist_function("qgt_wilson_loop")
        // v0.3: n-band Bloch surface + 4-band Z_2 + Pfaffian-sign 1D BdG Z_2.
        .allowlist_type("qgt_system_n_t")
        .allowlist_function("qgt_create_nband")
        .allowlist_function("qgt_free_nband")
        .allowlist_function("qgt_berry_grid_nband")
        .allowlist_function("qgt_z2_invariant")
        .allowlist_function("qgt_z2_invariant_1d_bdg")
        .allowlist_function("qgt_model_kane_mele")
        .allowlist_function("qgt_model_bhz")
        .allowlist_function("qgt_model_kitaev_chain")
        .allowlist_function("qgt_model_hofstadter")
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
        // CA-MPS handle + gate surface (since 0.2.1).
        .allowlist_type("moonlab_ca_mps_t")
        .allowlist_function("moonlab_ca_mps_create")
        .allowlist_function("moonlab_ca_mps_free")
        .allowlist_function("moonlab_ca_mps_clone")
        .allowlist_function("moonlab_ca_mps_num_qubits")
        .allowlist_function("moonlab_ca_mps_max_bond_dim")
        .allowlist_function("moonlab_ca_mps_current_bond_dim")
        .allowlist_function("moonlab_ca_mps_h")
        .allowlist_function("moonlab_ca_mps_s")
        .allowlist_function("moonlab_ca_mps_sdag")
        .allowlist_function("moonlab_ca_mps_x")
        .allowlist_function("moonlab_ca_mps_y")
        .allowlist_function("moonlab_ca_mps_z")
        .allowlist_function("moonlab_ca_mps_cnot")
        .allowlist_function("moonlab_ca_mps_cz")
        .allowlist_function("moonlab_ca_mps_swap")
        .allowlist_function("moonlab_ca_mps_rx")
        .allowlist_function("moonlab_ca_mps_ry")
        .allowlist_function("moonlab_ca_mps_rz")
        .allowlist_function("moonlab_ca_mps_t_gate")
        .allowlist_function("moonlab_ca_mps_t_dagger")
        .allowlist_function("moonlab_ca_mps_phase")
        .allowlist_function("moonlab_ca_mps_normalize")
        .allowlist_function("moonlab_ca_mps_norm")
        // Born-rule sequential sampling (since v0.10.0).
        .allowlist_function("moonlab_ca_mps_sample_z")
        // var-D, gauge warmstart, Z2 LGT, status (since 0.2.1).
        .allowlist_function("moonlab_ca_mps_var_d_run")
        // var-D v2 with explicit convergence_eps (since 0.2.4).
        .allowlist_function("moonlab_ca_mps_var_d_run_v2")
        .allowlist_function("moonlab_ca_mps_gauge_warmstart")
        .allowlist_function("moonlab_z2_lgt_1d_build")
        .allowlist_function("moonlab_z2_lgt_1d_gauss_law")
        .allowlist_function("moonlab_status_string")
        // DMRG scalar-energy convenience entries (since 0.10.0).
        .allowlist_function("moonlab_dmrg_tfim_energy")
        .allowlist_function("moonlab_dmrg_heisenberg_energy")
        // Bell tests + CHSH/Mermin/Mermin-Klyshko variants
        // (since 0.2.0; safe Rust wrapper from 0.4.7).
        .allowlist_type("bell_state_type_t")
        .allowlist_type("bell_test_result_t")
        .allowlist_type("bell_measurement_settings_t")
        .allowlist_function("create_bell_state")
        .allowlist_function("create_bell_state_phi_plus")
        .allowlist_function("create_bell_state_phi_minus")
        .allowlist_function("create_bell_state_psi_plus")
        .allowlist_function("create_bell_state_psi_minus")
        .allowlist_function("bell_get_optimal_settings")
        .allowlist_function("calculate_chsh_parameter")
        .allowlist_function("bell_test_chsh")
        .allowlist_function("bell_test_mermin_ghz")
        .allowlist_function("bell_test_mermin_klyshko")
        // Layout settings
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .size_t_is_usize(true)
        .generate_comments(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings to OUT_DIR
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
