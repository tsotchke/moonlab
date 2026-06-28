//! Variational Quantum Eigensolver: H2 ground state at the
//! equilibrium bond length, plus a custom 1-qubit Hamiltonian.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example vqe_demo -p moonlab

use moonlab::vqe::{OptimizerType, PauliHamiltonian, VqeSolver};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("H2 molecule at R = 0.74 A");
    let h = PauliHamiltonian::h2(0.74).unwrap();
    let exact = h.exact_ground_state_energy();
    println!("  exact ground-state E_0 (direct diag):  {:+.6} Ha", exact);
    println!("  num_qubits = {}   num_terms = {}", h.num_qubits(), h.num_terms());

    let mut solver = VqeSolver::new(h, 2, OptimizerType::Adam).unwrap();
    let r = solver.solve().unwrap();
    println!();
    println!("  VQE (Adam, 2 layers):");
    println!("    E_VQE      = {:+.6} Ha   ({:+.3} kcal/mol)",
             r.ground_state_energy, r.ground_state_energy_kcal_mol);
    println!("    converged  = {}", r.converged);
    println!("    iterations = {}", r.iterations);
    println!("    optimal params = {} entries", r.optimal_parameters.len());

    banner("Custom 1-qubit Hamiltonian: H = 0.5 Z");
    let custom = PauliHamiltonian::builder(1, 1)
        .unwrap()
        .add_term(0.5, "Z").unwrap()
        .build();
    let e_custom = custom.exact_ground_state_energy();
    println!("  exact ground state of 0.5 Z = -0.5");
    println!("  via vqe_exact_ground_state_energy: {:+.6}", e_custom);
    assert!((e_custom + 0.5).abs() < 1e-10, "exact GS off");

    banner("LiH at R = 1.6 A");
    let h_lih = PauliHamiltonian::lih(1.6).unwrap();
    println!("  num_qubits = {}   num_terms = {}",
             h_lih.num_qubits(), h_lih.num_terms());
    let exact_lih = h_lih.exact_ground_state_energy();
    println!("  exact ground-state E_0:  {:+.6} Ha", exact_lih);
    println!();
}
