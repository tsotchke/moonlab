//! Grover's quantum search across a small range of register sizes.
//! Sweeps n in [4, 8] and reports P(success) vs the textbook
//! 1 - 1/N ceiling.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example grover_demo -p moonlab

use moonlab::grover::{optimal_iterations, search};
use moonlab::QuantumState;

fn main() {
    let bar: String = "=".repeat(60);
    println!("\n{bar}");
    println!("  Grover's search: P(success) vs textbook ceiling 1 - 1/N");
    println!("{bar}\n");
    println!("  {:>3}   {:>4}   {:>11}   {:>15}   {:>12}",
             "n", "iter", "P(success)", "ceiling (1-1/N)", "found marked");
    println!("  {}", "-".repeat(58));

    for n in 4..=8usize {
        let optimal = optimal_iterations(n);
        let n_states = 1u64 << n;
        // Pick the marked state to be 0b...1010..1 -- something
        // not 0 to make the test less degenerate.
        let marked = (1u64 << (n - 1)) | 1;
        let mut state = QuantumState::new(n).unwrap();
        let r = search(&mut state, marked, None).unwrap();
        let ceiling = 1.0 - 1.0 / (n_states as f64);
        println!(
            "  {:>3}   {:>4}   {:>11.4}   {:>15.4}   {:>12}",
            n, optimal, r.success_probability, ceiling,
            if r.found_marked_state { "yes" } else { "no" },
        );
        assert_eq!(r.iterations_performed, optimal);
    }
    println!();
}
