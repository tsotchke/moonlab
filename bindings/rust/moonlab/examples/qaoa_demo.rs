//! QAOA on small MaxCut instances.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example qaoa_demo -p moonlab

use moonlab::qaoa::{Graph, IsingModel, QaoaSolver};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("MaxCut on the triangle (K_3)");
    let g = Graph::new(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
    let ising = IsingModel::from_maxcut(&g).unwrap();
    println!("  All 3 edges have equal weight 1.  Optimal cut: 2 of 3 edges.");
    println!();
    println!("  All bitstrings and their Ising energies:");
    for bits in 0u64..8 {
        let e = ising.evaluate(bits);
        println!("    {:03b}   E = {:+.3}", bits, e);
    }

    banner("QAOA on the triangle at depths p = 1, 2, 3");
    println!("  {:>4}   {:>14}   {:>10}   {:>10}", "p", "best energy", "approx %", "best bits");
    println!("  {}", "-".repeat(48));
    for p in 1..=3 {
        let g_p = Graph::new(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
        let ising_p = IsingModel::from_maxcut(&g_p).unwrap();
        let mut solver = QaoaSolver::new(ising_p, p).unwrap();
        let r = solver.solve().unwrap();
        let pct = r.approximation_ratio * 100.0;
        println!(
            "  {:>4}   {:>14.4}   {:>9.2}%   {:>010b}",
            p, r.best_energy, pct,
            r.best_bitstring as u32,
        );
    }
    println!();
}
