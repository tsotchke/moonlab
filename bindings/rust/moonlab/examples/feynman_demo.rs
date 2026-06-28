//! Feynman diagram rendering: build a few canonical QED processes
//! and dump them as ASCII / SVG.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example feynman_demo -p moonlab

use moonlab::feynman::FeynmanDiagram;

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("e+ e- -> mu+ mu- (tree-level QED)");
    let d = FeynmanDiagram::ee_to_mumu().unwrap();
    println!("  ASCII rendering:");
    println!("{}", d.render_ascii());

    banner("Compton scattering: gamma e- -> gamma e-");
    let d = FeynmanDiagram::compton_scattering().unwrap();
    println!("{}", d.render_ascii());

    banner("Pair annihilation: e+ e- -> gamma gamma");
    let d = FeynmanDiagram::pair_annihilation().unwrap();
    println!("{}", d.render_ascii());

    banner("QED vertex (tree-level)");
    let d = FeynmanDiagram::qed_vertex().unwrap();
    println!("{}", d.render_ascii());
    println!();
}
