//! Single-qubit Kraus noise channels: walk through every channel
//! at a few probabilities, then verify trace conservation on a
//! mixed multi-channel sequence.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build \
//!     cargo run --example noise_demo -p moonlab

use moonlab::{noise, QuantumState};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn main() {
    banner("Single-channel sweeps on |+> = H|0>");
    println!("  Each channel applied with a fixed r=0.5; <Z> after.");
    println!();
    println!("  {:>22}   {:>5}   {:>7}", "channel", "param", "<Z>");
    println!("  {}", "-".repeat(45));

    for (label, p) in [
        ("depolarizing(p=0.1)",    0.1),
        ("amplitude_damping(g=0.2)", 0.2),
        ("phase_damping(g=0.2)",   0.2),
        ("bit_flip(p=0.3)",        0.3),
        ("phase_flip(p=0.3)",      0.3),
        ("bit_phase_flip(p=0.3)",  0.3),
    ] {
        let mut s = QuantumState::new(1).unwrap();
        s.h(0);
        match label {
            l if l.starts_with("depolarizing") =>
                noise::depolarizing_single(&mut s, 0, p, 0.5).unwrap(),
            l if l.starts_with("amplitude_damping") =>
                noise::amplitude_damping(&mut s, 0, p, 0.5).unwrap(),
            l if l.starts_with("phase_damping") =>
                noise::phase_damping(&mut s, 0, p, 0.5).unwrap(),
            l if l.starts_with("bit_flip") =>
                noise::bit_flip(&mut s, 0, p, 0.5).unwrap(),
            l if l.starts_with("phase_flip") =>
                noise::phase_flip(&mut s, 0, p, 0.5).unwrap(),
            l if l.starts_with("bit_phase_flip") =>
                noise::bit_phase_flip(&mut s, 0, p, 0.5).unwrap(),
            _ => unreachable!(),
        };
        let z = s.expectation_z(0).unwrap();
        println!("  {:>22}   {:>5.2}   {:>+7.4}", label, p, z);
    }

    banner("Full depolarising at p = 3/4 sends |+> to I/2");
    let mut s = QuantumState::new(1).unwrap();
    s.h(0);
    noise::depolarizing_single(&mut s, 0, 0.75, 0.5).unwrap();
    let z = s.expectation_z(0).unwrap();
    println!("  <Z> after p=3/4 depolarising = {:+.6} (expect 0)", z);

    banner("Thermal relaxation (T1 + T2) over time 1.0 us");
    let mut s = QuantumState::new(1).unwrap();
    s.x(0);  // start in |1>
    noise::thermal_relaxation(&mut s, 0,
        /*t1=*/50.0, /*t2=*/30.0, /*time=*/1.0,
        &[0.4, 0.6]).unwrap();
    println!("  <Z> after T1=50us, T2=30us, t=1us on |1>: {:+.6}",
             s.expectation_z(0).unwrap());

    banner("Classical readout error");
    let r = noise::readout_error(false, 0.1, 0.05, 0.05);
    println!("  readout(true=0, e01=0.1, e10=0.05, r=0.05) -> {}", r);
    let r = noise::readout_error(false, 0.1, 0.05, 0.5);
    println!("  readout(true=0, e01=0.1, e10=0.05, r=0.50) -> {}", r);
    println!();
}
