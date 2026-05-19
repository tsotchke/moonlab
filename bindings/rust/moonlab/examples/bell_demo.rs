//! Bell-inequality demonstration: CHSH + Mermin-GHZ + Mermin-Klyshko
//! across the four canonical entangled states.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example bell_demo -p moonlab

use moonlab::{
    bell::{chsh_test, create_bell_state, mermin_ghz_test,
           mermin_klyshko_test, BellState},
    QuantumState,
};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    let num_meas = 4000usize;

    banner("CHSH inequality across the four Bell pairs");
    println!("  classical bound: 2.0   |   quantum (Tsirelson) bound: 2.828");
    println!();
    for (label, kind) in [
        ("|Phi+>", BellState::PhiPlus),
        ("|Phi->", BellState::PhiMinus),
        ("|Psi+>", BellState::PsiPlus),
        ("|Psi->", BellState::PsiMinus),
    ] {
        let mut state = QuantumState::new(2).unwrap();
        create_bell_state(&mut state, 0, 1, kind).unwrap();
        let r = chsh_test(&mut state, 0, 1, num_meas).unwrap();
        let badge = if r.violates_classical { "VIOLATES classical" } else { "trivial" };
        let q_badge = if r.confirms_quantum { "near quantum bound" } else { "off bound" };
        println!(
            "  {:>7}  S = {:+.4}   ({}, {})",
            label, r.chsh_value, badge, q_badge,
        );
    }

    banner("Mermin-GHZ inequality on |GHZ_3>");
    println!("  classical bound: 2.0   |   quantum bound: 4.0");
    println!();
    let mut state = QuantumState::new(3).unwrap();
    state.h(0).cnot(0, 1).cnot(0, 2);
    let r = mermin_ghz_test(&mut state, 0, 1, 2, num_meas).unwrap();
    println!(
        "  |M| = {:+.4}   (classical violation by {:.2}x)",
        r.chsh_value.abs(),
        r.chsh_value.abs() / r.classical_bound,
    );

    banner("Mermin-Klyshko inequality on |GHZ_n>");
    println!("  classical bound (normalised): 1.0");
    println!();
    for n in [2usize, 3, 4, 5] {
        let mut state = QuantumState::new(n).unwrap();
        state.h(0);
        for q in 1..n {
            state.cnot(0, q);
        }
        let mn = mermin_klyshko_test(&mut state, n, num_meas).unwrap();
        let ideal = 2f64.powf((n - 1) as f64 / 2.0);
        println!(
            "  n = {n}   |M_N| = {:.4}   (ideal = {:.4}, ratio = {:.3})",
            mn, ideal, mn / ideal,
        );
    }
    println!();
}
