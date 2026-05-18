//! Adaptive-bond TDVP demonstration (Rust).
//!
//! Rust counterpart to `bindings/python/moonlab/tdvp.py` and the
//! pytest suite at `bindings/python/tests/test_tdvp.py`.  Three
//! sections:
//!
//!   1. Real-time evolution on the 8-site Heisenberg chain --
//!      reports the maximum relative energy drift across 5 steps
//!      under the entropy-feedback PID controller.
//!
//!   2. Imaginary-time evolution on the 8-site critical TFIM
//!      (g = 1) -- 30 steps at dt = 0.05, reporting the energy at
//!      every fifth step.
//!
//!   3. PID stability micro-sweep at three (kp, ki, kd) triplets.
//!
//! Run with::
//!
//!     MOONLAB_LIB_DIR=$(pwd)/build_release \
//!     cargo run --example tdvp_demo -p moonlab

use moonlab::tdvp::{
    EvolutionType, Mpo, Mps, TdvpConfig, TdvpEngine,
};

fn banner(title: &str) {
    let bar: String = "=".repeat(title.len() + 4);
    println!("\n{bar}\n  {title}\n{bar}");
}

fn section_real_time() {
    banner("1. Real-time Heisenberg (energy conservation)");
    let n: u32 = 8;
    let mpo = Mpo::heisenberg(n, 1.0, 1.0, 0.0).unwrap();
    let mps = Mps::random(n, 8, 32, 1e-12).unwrap();

    let mut cfg = TdvpConfig::adaptive(1e-3);
    cfg.evolution_type = EvolutionType::RealTime;
    cfg.dt = 0.02;
    cfg.adaptive_bond.chi_ceiling = 32;
    cfg.max_bond_dim = 32;

    let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
    let mut energies = Vec::new();
    for _ in 0..5 {
        energies.push(engine.step().unwrap().energy);
    }
    let e0 = energies[0];
    let max_drift = energies
        .iter()
        .map(|e| (e - e0).abs() / e0.abs().max(1e-12))
        .fold(0.0_f64, f64::max);
    println!("  E0 = {e0:+.6}");
    for (i, e) in energies.iter().enumerate() {
        println!("  step {i}: E = {e:+.6}");
    }
    println!("  max relative drift = {max_drift:.3e}");
}

fn section_imag_time() {
    banner("2. Imag-time TFIM (g = 1, critical point)");
    let n: u32 = 8;
    let mpo = Mpo::tfim(n, 1.0, 1.0).unwrap();
    let mps = Mps::random(n, 8, 32, 1e-12).unwrap();

    let mut cfg = TdvpConfig::adaptive(1e-3);
    cfg.evolution_type = EvolutionType::ImaginaryTime;
    cfg.dt = 0.05;
    cfg.adaptive_bond.chi_ceiling = 32;
    cfg.max_bond_dim = 32;

    let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
    for step in 0..30 {
        let r = engine.step().unwrap();
        if step % 5 == 0 || step == 29 {
            println!(
                "  step {step:2}: E = {:+.6}, mean chi = {:.2}",
                r.energy,
                if r.bond_chi_distribution.is_empty() {
                    0.0
                } else {
                    r.bond_chi_distribution
                        .iter()
                        .map(|&c| c as f64)
                        .sum::<f64>()
                        / r.bond_chi_distribution.len() as f64
                },
            );
        }
    }
}

fn section_stability() {
    banner("3. PID stability micro-sweep");
    println!("  {:>5} {:>6} {:>5}  {:>9}", "kp", "ki", "kd", "max osc");
    for &(kp, ki, kd) in &[
        (0.25_f64, 0.05_f64, 0.10_f64),
        (0.50, 0.05, 0.10),
        (1.00, 0.05, 0.10),
    ] {
        let n: u32 = 8;
        let mpo = Mpo::heisenberg(n, 1.0, 1.0, 0.0).unwrap();
        let mps = Mps::random(n, 8, 32, 1e-12).unwrap();

        let mut cfg = TdvpConfig::adaptive(1e-3);
        cfg.evolution_type = EvolutionType::RealTime;
        cfg.dt = 0.02;
        cfg.adaptive_bond.kp = kp;
        cfg.adaptive_bond.ki = ki;
        cfg.adaptive_bond.kd = kd;
        cfg.adaptive_bond.chi_ceiling = 32;
        cfg.max_bond_dim = 32;

        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let mut prev: Vec<u32> = Vec::new();
        let mut max_osc: u32 = 0;
        for _ in 0..5 {
            let r = engine.step().unwrap();
            if !prev.is_empty() && prev.len() == r.bond_chi_distribution.len() {
                for (a, b) in prev.iter().zip(r.bond_chi_distribution.iter())
                {
                    let d = if a > b { a - b } else { b - a };
                    if d > max_osc {
                        max_osc = d;
                    }
                }
            }
            prev = r.bond_chi_distribution;
        }
        println!("  {kp:>5.2} {ki:>6.2} {kd:>5.2}  {max_osc:>9}");
    }
}

fn main() {
    section_real_time();
    section_imag_time();
    section_stability();
    println!();
}
