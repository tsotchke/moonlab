//! Quantum geometric tensor: topological phase diagrams (Rust).
//!
//! Reproduces the principal phase boundaries of seven canonical
//! lattice models using the Moonlab v0.3 Rust bindings to the
//! quantum-geometric-tensor module.  The Python sibling lives at
//! `examples/topological/qgt_phase_diagrams.py` and exercises the
//! same closed-form invariants.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build_release \
//!     cargo run --example topology_demo -p moonlab
//!
//! Models covered, with primary references inline:
//!     SSH                 Su, Schrieffer, Heeger, PRL 42, 1698 (1979)
//!     Qi-Wu-Zhang         Qi, Wu, Zhang,        PRB 74, 085308 (2006)
//!     Kane-Mele           Kane, Mele,           PRL 95, 146802 (2005)
//!     Bernevig-Hughes-Zhang Bernevig, Hughes, Zhang,
//!                                              Science 314, 1757 (2006)
//!     Kitaev p-wave chain Kitaev,               Phys.-Uspekhi 44, 131 (2001)
//!     Hofstadter          Hofstadter,           PRB 14, 2239 (1976)

use moonlab::{
    bhz_z2, chern_qwz_parallel_transport, chern_qwz_proj, hofstadter_chern,
    kane_mele_z2, kitaev_chain_z2, qwz_chern, ssh_winding,
};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn section_ssh() {
    banner("SSH chain (1D winding)");
    println!("Topological for |t2| > |t1|; trivial otherwise.");
    println!("  t1=0.5, t2=1.0  ->  W = {:+}", ssh_winding(0.5, 1.0, 64));
    println!("  t1=1.0, t2=0.5  ->  W = {:+}", ssh_winding(1.0, 0.5, 64));
}

fn section_qwz() {
    banner("Qi-Wu-Zhang model: three Chern integrators must agree");
    println!("Phase boundaries at m in {{-2, 0, +2}}; gap closures otherwise.");
    println!("  {:>5} {:>5} {:>5} {:>5}", "m", "FHS", "proj", "p.t.");
    let masses = [-3.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    for m in masses {
        let fhs = qwz_chern(m, 32);
        let proj = chern_qwz_proj(m, 32).unwrap();
        let pt = chern_qwz_parallel_transport(m, 32).unwrap();
        println!("  {m:>+5.1} {fhs:>+5} {proj:>+5} {pt:>+5}");
    }
}

fn section_kane_mele() {
    banner("Kane-Mele Z_2 (S_z conserving)");
    println!("QSH (Z_2 = 1) when |lambda_v| < 3 sqrt(3) |lambda_so|.");
    let lambda_so = 0.06_f64;
    let boundary = 3.0 * 3.0_f64.sqrt() * lambda_so;
    println!("  Boundary at |lambda_v| = 3 sqrt(3) * {lambda_so} = {boundary:.5}");
    for &lambda_v in &[0.05_f64, 0.10, 0.15, 0.20, 0.25, 0.40] {
        let z2 = kane_mele_z2(1.0, lambda_so, 0.0, lambda_v, 24).unwrap();
        let label = if z2 == 1 { "QSH" } else { "trivial" };
        println!("  lambda_v = {lambda_v:.2} ->  Z_2 = {z2}  ({label})");
    }
}

fn section_bhz() {
    banner("Bernevig-Hughes-Zhang (HgTe quantum well)");
    println!("Lattice regularisation gives QSH for 0 < M / B < 8.");
    let (a, b) = (1.0_f64, 1.0_f64);
    for &m in &[-1.0_f64, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0] {
        let z2 = bhz_z2(a, b, m, 24).unwrap();
        let label = if z2 == 1 { "QSH" } else { "trivial" };
        println!("  M = {m:>+5.1}  ->  Z_2 = {z2}  ({label})");
    }
}

fn section_kitaev_chain() {
    banner("Kitaev p-wave chain (Pfaffian-sign Z_2)");
    println!("Topological with Majorana edges when |mu| < 2 |t|.");
    let (t, delta) = (1.0_f64, 1.0_f64);
    for &mu in &[-3.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
        let z2 = kitaev_chain_z2(t, mu, delta).unwrap();
        let label = if z2 == 1 { "Majorana" } else { "trivial" };
        println!("  mu = {mu:>+4.1}  ->  Z_2 = {z2}  ({label})");
    }
}

fn section_hofstadter() {
    banner("Hofstadter magnetic sub-band Chern numbers (phi = 1/q)");
    println!("For phi = 1/q the lowest sub-band carries Chern = +1 (TKNN 1982).");
    for q in 3..=7 {
        let c = hofstadter_chern(1, q, 1, 1.0, 24).unwrap();
        println!("  q = {q}: lowest band Chern = {c:+}");
    }
}

fn main() {
    section_ssh();
    section_qwz();
    section_kane_mele();
    section_bhz();
    section_kitaev_chain();
    section_hofstadter();
    println!();
}
