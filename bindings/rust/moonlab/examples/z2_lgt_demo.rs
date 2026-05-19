//! 1+1D Z2 lattice gauge theory: build the Pauli-sum Hamiltonian
//! and inspect its structure.  Mirrors the var-D x gauge-warmstart
//! demonstration in the moonlab paper (Appendix F).
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example z2_lgt_demo -p moonlab

use moonlab::z2_lgt;

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn pauli_str(row: &[u8]) -> String {
    row.iter().map(|&p| match p {
        0 => 'I', 1 => 'X', 2 => 'Y', 3 => 'Z', _ => '?',
    }).collect()
}

fn main() {
    banner("Z2 LGT Hamiltonian on N=4 matter sites");
    let h = z2_lgt::build(
        /*num_matter_sites=*/4,
        /*t_hop=*/1.0,
        /*h_link=*/0.5,
        /*mass=*/0.0,
        /*gauss_penalty=*/2.0,
    ).unwrap();
    println!("  matter sites = 4 (N=4)");
    println!("  num_qubits   = {}   (matter + link qubits)", h.num_qubits);
    println!("  num_terms    = {}", h.num_terms);
    println!();
    println!("  First 6 Pauli-string terms:");
    println!("  {:>4}   {:>10}   {:<30}", "idx", "coeff", "pauli");
    println!("  {}", "-".repeat(50));
    let n = h.num_qubits as usize;
    for k in 0..6.min(h.num_terms as usize) {
        let row = &h.paulis[k * n .. (k + 1) * n];
        println!("  {:>4}   {:>+10.4}   {:<30}", k, h.coeffs[k], pauli_str(row));
    }

    banner("Gauss-law generators at the interior matter sites");
    println!("  G_x = X_link(x-1) Z_matter(x) X_link(x)");
    println!("  Defined for interior sites 1..N-2; edges have boundary terms.");
    println!();
    for x in 1..3 {  // interior sites for N=4: x=1, 2
        let g = z2_lgt::gauss_law(4, x).unwrap();
        let support: Vec<(usize, char)> = g.iter().enumerate()
            .filter(|&(_, &p)| p != 0)
            .map(|(i, &p)| (i, "IXYZ".chars().nth(p as usize).unwrap()))
            .collect();
        let s = support.iter()
            .map(|(i, c)| format!("{}{}", c, i))
            .collect::<Vec<_>>().join(" * ");
        println!("  G_{x} = {}", s);
    }
    println!();
}
