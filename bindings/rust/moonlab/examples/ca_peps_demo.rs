//! Clifford-Assisted PEPS demonstration on a 2x3 lattice.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example ca_peps_demo -p moonlab

use moonlab::ca_peps::{CaPeps, PauliCode};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("CA-PEPS: |0..0> initial expectations");
    let s = CaPeps::new(2, 3, 8).unwrap();
    println!("  Lx = {}, Ly = {}, num_qubits = {}, chi_max = {}",
             s.lx(), s.ly(), s.num_qubits(), s.max_bond_dim());
    println!("  norm = {:.6}", s.norm());
    for q in 0..s.num_qubits() {
        let z = s.expect_pauli_single(q, PauliCode::Z).unwrap();
        print!("  <Z_{}> = {:+.6}", q, z);
        if (q + 1) % 3 == 0 { println!(); }
    }

    banner("CA-PEPS: Bell pair via H(0) + CNOT(0,1)");
    let mut bell = CaPeps::new(2, 1, 4).unwrap();
    bell.h(0).unwrap().cnot(0, 1).unwrap();
    let zz = bell.expect_pauli(&[PauliCode::Z as u8, PauliCode::Z as u8]).unwrap();
    println!("  <Z_0 Z_1>     = {:+.6} + {:+.6}i  (expect 1)", zz.0, zz.1);
    println!("  <Z_0>         = {:+.6}            (expect 0)",
             bell.expect_pauli_single(0, PauliCode::Z).unwrap());
    println!("  P(Z_0 = +1)   = {:.6}             (expect 0.5)",
             bell.prob_z(0).unwrap());

    banner("CA-PEPS: cloned state is independent");
    let mut a = CaPeps::new(2, 2, 4).unwrap();
    a.h(0).unwrap();  // first qubit in superposition
    let b = a.clone();  // capture mid-state
    a.h(0).unwrap();  // toggle |+> back to |0>
    let z_a = a.expect_pauli_single(0, PauliCode::Z).unwrap();
    let z_b = b.expect_pauli_single(0, PauliCode::Z).unwrap();
    println!("  a (after H twice): <Z_0> = {:+.6}   (expect +1 = |0>)", z_a);
    println!("  b (snapshot mid):  <Z_0> = {:+.6}   (expect 0 = |+>)", z_b);
    println!();
}
