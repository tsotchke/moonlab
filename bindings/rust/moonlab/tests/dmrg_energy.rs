//! Rust binding parity tests for `moonlab::dmrg::*`.
//!
//! Mirrors the Python tests in `bindings/python/tests/test_dmrg.py`.

use moonlab::dmrg::{heisenberg_ground_energy, tfim_ground_energy};

#[test]
fn tfim_critical_point_returns_finite_negative_energy() {
    let e = tfim_ground_energy(8, 1.0, 32, 10);
    assert!(e.is_finite(), "expected finite energy, got {}", e);
    assert!(e < 0.0, "expected negative energy at criticality, got {}", e);
}

#[test]
fn tfim_larger_field_more_negative() {
    // -sum ZZ - g sum X: at large g the X-term dominates so E(g=2.0) < E(g=0.1).
    let e_small = tfim_ground_energy(8, 0.1, 32, 10);
    let e_large = tfim_ground_energy(8, 2.0, 32, 10);
    assert!(e_small.is_finite() && e_large.is_finite());
    assert!(
        e_large < e_small,
        "expected E(g=2.0) < E(g=0.1), got {} vs {}",
        e_large,
        e_small
    );
}

#[test]
fn heisenberg_isotropic_finite_energy() {
    let e = heisenberg_ground_energy(8, 1.0, 1.0, 0.0, 32, 10);
    assert!(e.is_finite(), "expected finite energy, got {}", e);
}

#[test]
fn heisenberg_xx_only_finite_energy() {
    // Delta = 0 reduces to the free-fermion XX chain; should still
    // return a finite energy.
    let e = heisenberg_ground_energy(8, 1.0, 0.0, 0.0, 32, 10);
    assert!(e.is_finite());
}

#[test]
fn tfim_invalid_input_returns_sentinel() {
    // num_sites < 2 is invalid.
    let e = tfim_ground_energy(1, 1.0, 16, 5);
    assert!(e > 1e300, "expected DBL_MAX sentinel, got {}", e);

    // max_bond_dim < 1 is invalid.
    let e = tfim_ground_energy(4, 1.0, 0, 5);
    assert!(e > 1e300, "expected DBL_MAX sentinel, got {}", e);

    // num_sweeps = 0 is invalid.
    let e = tfim_ground_energy(4, 1.0, 16, 0);
    assert!(e > 1e300, "expected DBL_MAX sentinel, got {}", e);
}
