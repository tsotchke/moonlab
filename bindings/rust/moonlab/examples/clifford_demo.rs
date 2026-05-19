//! Aaronson-Gottesman Clifford tableau: GHZ states at scales the
//! dense state-vector backend cannot reach.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example clifford_demo -p moonlab

use moonlab::clifford::CliffordTableau;

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("GHZ states via polynomial-time Clifford tableau");
    println!("  Each row: H(0) + CNOT(0, i) for i = 1 .. n-1, then sampleAll().");
    println!("  |GHZ_n> has support only on |0...0> and |1...1>; the");
    println!("  sample must be one of those two bitstrings exactly.");
    println!();
    println!("  {:>4}   {:>10}", "n", "bitstring");
    println!("  {}", "-".repeat(18));

    for n in [8usize, 32, 64] {
        let mut tab = CliffordTableau::new(n).unwrap();
        tab.set_rng_seed(0xC0FFEE * n as u64);
        tab.h(0).unwrap();
        for q in 1..n {
            tab.cnot(0, q).unwrap();
        }
        let bits = tab.sample_all().unwrap();
        // GHZ sample must be all-zeros or all-ones.
        let all_zero = bits == 0;
        let all_ones = if n == 64 { bits == u64::MAX } else { bits == (1u64 << n) - 1 };
        assert!(all_zero || all_ones, "GHZ sample not aligned: {bits:#066b}");
        println!("  {:>4}   0x{:016x}", n, bits);
    }

    banner("Deterministic measurement on |0...0>");
    let mut tab = CliffordTableau::new(8).unwrap();
    let result = tab.measure(3).unwrap();
    println!("  qubit 3 of |0...0>: outcome = {}, deterministic = {}",
             result.outcome, result.deterministic);
    assert_eq!(result.outcome, 0);
    println!();
}
