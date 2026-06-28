//! Single-qubit gate-fusion DAG: build a hardware-efficient
//! ansatz, run the fuser, and show the compression ratio.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example fusion_demo -p moonlab

use moonlab::fusion::FusedCircuit;
use moonlab::QuantumState;

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

/// Build an L-layer hardware-efficient ansatz on n qubits: each
/// layer is `Rz Rx Rz` on every qubit followed by a CNOT ladder.
fn build_hea(n: i32, layers: u32) -> FusedCircuit {
    let mut c = FusedCircuit::new(n as usize).unwrap();
    // Pseudo-random angles; deterministic for run-to-run stability.
    let mut t = 0.1_f64;
    let mut next = || { t = (t * 1.61803).fract(); t };
    for _ in 0..layers {
        for q in 0..n {
            c.rz(q, next()).unwrap();
            c.rx(q, next()).unwrap();
            c.rz(q, next()).unwrap();
        }
        for q in 0..(n - 1) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

fn main() {
    banner("Hardware-efficient ansatz: fusion compression ratio");
    println!("  Each row: a fresh n-qubit, L-layer HEA built, fused,");
    println!("  and run on a fresh |0..0> state.  The fuser merges");
    println!("  consecutive 1q gates between two-qubit gate barriers.");
    println!();
    println!("  {:>3}   {:>3}   {:>10}   {:>10}   {:>10}   {:>10}",
             "n", "L", "in gates", "out gates", "ratio", "merges");
    println!("  {}", "-".repeat(60));

    for (n, l) in [(4i32, 3u32), (6, 3), (8, 5), (10, 5)] {
        let circuit = build_hea(n, l);
        let original_len = circuit.len();
        let (fused, stats) = circuit.compile().unwrap();
        let mut state = QuantumState::new(n as usize).unwrap();
        fused.execute(&mut state).unwrap();
        let ratio = stats.fused_gates as f64 / stats.original_gates as f64;
        println!(
            "  {:>3}   {:>3}   {:>10}   {:>10}   {:>10.3}   {:>10}",
            n, l, original_len, stats.fused_gates, ratio, stats.merges_applied,
        );
    }
    println!();
}
