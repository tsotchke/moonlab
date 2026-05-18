//! Matrix-product density operator (MPDO) noise simulation: Rust demo.
//!
//! Rust counterpart to `examples/applications/mpdo_noise_demo.py`.
//! Reproduces canonical analytical results from the operator-sum
//! representation of single-qubit noise channels (Nielsen and Chuang
//! 2010, ch. 8) and verifies the matrix-product density-operator
//! engine of Verstraete, Garcia-Ripoll, and Cirac (Phys. Rev. Lett.
//! 93, 207204, 2004) against the closed-form expressions to machine
//! precision.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build_release \
//!     cargo run --example mpdo_demo -p moonlab

use moonlab::mpdo::{Mpdo, PauliCode};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn demo_initial_state() {
    banner("1. Initial state |0...0><0...0| at chi = 1");
    let rho = Mpdo::new(4, 16).expect("Mpdo::new failed");
    println!("  qubits           = {}", rho.num_qubits());
    println!("  max_bond_dim     = {}", rho.max_bond_dim());
    println!("  current_bond_dim = {}", rho.current_bond_dim());
    println!("  Tr(rho)          = {:.12}  (expected 1)", rho.trace());
    for q in 0..rho.num_qubits() {
        let z = rho.expect_pauli(q, PauliCode::Z).unwrap();
        println!("  <Z_{q}>            = {z:+.12}  (expected +1)");
    }
}

fn demo_depolarising() {
    banner("2. Depolarising channel: <Z> -> 1 - 4 p / 3");
    println!("  {:>5}  {:>12}  {:>12}  {:>10}", "p", "<Z>", "1 - 4p/3", "|err|");
    for &p in &[0.0_f64, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_depolarizing(0, p).unwrap();
        let z = rho.expect_pauli(0, PauliCode::Z).unwrap();
        let reference = 1.0 - 4.0 * p / 3.0;
        let err = (z - reference).abs();
        println!("  {p:5.2}  {z:+12.6}  {reference:+12.6}  {err:.2e}");
    }
}

fn demo_amplitude_damping() {
    banner("3. Amplitude damping (T_1) from |1>");
    println!("  After bit-flip(p=1) we are at |1><1|, then amplitude-damp.");
    println!("  {:>5}  {:>12}  {:>12}  {:>10}", "gamma", "<Z>", "2 gamma - 1", "|err|");
    for &gamma in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_bit_flip(0, 1.0).unwrap();          // |0><0| -> |1><1|
        rho.apply_amplitude_damping(0, gamma).unwrap();
        let z = rho.expect_pauli(0, PauliCode::Z).unwrap();
        let reference = 2.0 * gamma - 1.0;
        let err = (z - reference).abs();
        println!("  {gamma:5.2}  {z:+12.6}  {reference:+12.6}  {err:.2e}");
    }
}

fn demo_phase_damping() {
    banner("4. Phase damping (T_2) on |0>: preserves <Z> exactly");
    println!("  {:>6}  {:>12}", "lambda", "<Z>");
    for &lam in &[0.0_f64, 0.1, 0.25, 0.5, 0.75, 1.0] {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_phase_damping(0, lam).unwrap();
        let z = rho.expect_pauli(0, PauliCode::Z).unwrap();
        println!("  {lam:6.2}  {z:+12.6}");
    }
}

fn demo_clone_independence() {
    banner("5. Clone independence");
    let mut rho = Mpdo::new(2, 16).unwrap();
    rho.apply_depolarizing(0, 0.3).unwrap();
    let mut sigma = rho.clone();
    sigma.apply_amplitude_damping(0, 1.0).unwrap();  // reset clone to |0>
    println!(
        "  rho:   <Z_0> = {:+.6}  (depolarised)",
        rho.expect_pauli(0, PauliCode::Z).unwrap(),
    );
    println!(
        "  sigma: <Z_0> = {:+.6}  (reset to |0>)",
        sigma.expect_pauli(0, PauliCode::Z).unwrap(),
    );
}

fn demo_named_channels_inventory() {
    banner("6. Named single-qubit channel inventory");
    println!("  Each channel applied at p = 0.25 to a fresh |0><0|; <Z> reported.");
    println!("  {:>22}  {:>10}", "channel", "<Z>");
    let mut rows: Vec<(&'static str, f64)> = Vec::new();
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_depolarizing(0, 0.25).unwrap();
        rows.push(("depolarizing(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_amplitude_damping(0, 0.25).unwrap();
        rows.push(("amplitude_damping(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_phase_damping(0, 0.25).unwrap();
        rows.push(("phase_damping(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_bit_flip(0, 0.25).unwrap();
        rows.push(("bit_flip(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_phase_flip(0, 0.25).unwrap();
        rows.push(("phase_flip(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    {
        let mut rho = Mpdo::new(1, 16).unwrap();
        rho.apply_bit_phase_flip(0, 0.25).unwrap();
        rows.push(("bit_phase_flip(0.25)", rho.expect_pauli(0, PauliCode::Z).unwrap()));
    }
    for (name, z) in rows {
        println!("  {name:>22}  {z:+10.6}");
    }
}

fn main() {
    demo_initial_state();
    demo_depolarising();
    demo_amplitude_damping();
    demo_phase_damping();
    demo_clone_independence();
    demo_named_channels_inventory();
    println!();
}
