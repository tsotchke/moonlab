//! Clifford-Assisted MPS demonstration: GHZ-state preparation,
//! gauge warmstart on a Z2 LGT problem, and variational-D ground
//! state search on a small TFIM.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example ca_mps_demo -p moonlab

use moonlab::{
    var_d_run, z2_lgt_1d_build, z2_lgt_1d_gauss_law,
    CaMps, VarDConfig, Warmstart,
};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("CA-MPS: |0...0> initial norm + bond dimension");
    let mut state = CaMps::new(6, 16).unwrap();
    println!("  n = {}, chi_max = {}, norm = {:.6}, bond_dim = {}",
             state.num_qubits(), 16, state.norm(), state.bond_dim());

    banner("CA-MPS: GHZ preparation via H + CNOT chain");
    state.h(0).unwrap();
    for q in 1..6u32 {
        state.cnot(0, q).unwrap();
    }
    state.normalize().unwrap();
    println!("  After H(0) + CNOT chain: norm = {:.6}, bond_dim = {}",
             state.norm(), state.bond_dim());
    println!("  GHZ is Clifford-prepared -- bond_dim stays at 1.");

    banner("Z2 LGT Hamiltonian on 4 matter sites");
    let (paulis_z2, coeffs_z2, nq_z2) = z2_lgt_1d_build(
        4, /*t=*/1.0, /*h=*/0.5, /*mass=*/0.0, /*gauss=*/1.0).unwrap();
    let num_terms_z2 = (coeffs_z2.len()) as u32;
    println!("  num_qubits = {}, num_terms = {}", nq_z2, num_terms_z2);
    let _ = paulis_z2;  // structure inspected in z2_lgt_demo
    let gauss_site_1 = z2_lgt_1d_gauss_law(4, 1).unwrap();
    let nonzero: Vec<(usize, u8)> = gauss_site_1.iter().enumerate()
        .filter(|&(_, &p)| p != 0)
        .map(|(i, &p)| (i, p))
        .collect();
    println!("  Gauss-law generator at site 1: {} non-identity entries",
             nonzero.len());

    banner("CA-MPS var-D ground state search on TFIM (n=6, g=1.0)");
    // 6-site TFIM: H = -sum Z_i Z_{i+1} - g sum X_i
    let n = 6;
    let g = 1.0;
    let mut paulis: Vec<u8> = Vec::new();
    let mut coeffs: Vec<f64> = Vec::new();
    for i in 0..n-1 {
        let mut row = vec![0u8; n];
        row[i] = 3; row[i+1] = 3;
        paulis.extend(&row);
        coeffs.push(-1.0);
    }
    for i in 0..n {
        let mut row = vec![0u8; n];
        row[i] = 1;
        paulis.extend(&row);
        coeffs.push(-g);
    }
    let cfg = VarDConfig {
        max_outer_iters: 4,
        imag_time_dtau: 0.05,
        imag_time_steps_per_outer: 8,
        clifford_passes_per_outer: 2,
        composite_2gate: false,
        warmstart: Warmstart::Identity,
        convergence_eps: 1e-6,
    };
    let num_terms = coeffs.len() as u32;
    let mut s = CaMps::new(n as u32, 16).unwrap();
    let e = var_d_run(&mut s, &paulis, &coeffs, num_terms, &[], 0, &cfg).unwrap();
    println!("  E_var-D = {:+.6}  (n=6, g=1.0 TFIM)", e);
    println!("  Critical TFIM ground-state energy density: -1.273 per bond");
    println!("  expected total E ~ -5 (for 5 bonds at criticality with field)");
    println!();
}
