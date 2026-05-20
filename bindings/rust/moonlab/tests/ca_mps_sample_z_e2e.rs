//! Rust parity for the CA-MPS Born-rule sequential sampler.
//!
//! Mirrors `tests/unit/test_ca_mps_sample.c` and the Python
//! `test_sample_z_*` cases.  Bell-pair and GHZ_4 support are exact;
//! we sample 4096 shots and check the empirical support.

use moonlab::ca_mps::CaMps;

/// Reproducible LCG matching the C test so distributions line up.
struct Lcg(u32);
impl Lcg {
    fn next_u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f64) / 16_777_216.0
    }
    fn fill(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_u()).collect()
    }
}

#[test]
fn bell_sample_z_support_and_marginals() {
    let mut s = CaMps::new(2, 16).expect("new");
    s.h(0).unwrap();
    s.cnot(0, 1).unwrap();

    let mut rng = Lcg(0xDEAD_BEEF);
    let shots = 4096u32;
    let randoms = rng.fill((shots as usize) * 2);
    let bits = s.sample_z(shots, &randoms).expect("sample_z");

    assert_eq!(bits.len(), (shots as usize) * 2);

    // Bell support is {00, 11}; verify and check rough 50/50 split.
    let mut n00 = 0usize;
    let mut n11 = 0usize;
    let mut other = 0usize;
    for s in 0..(shots as usize) {
        match (bits[s * 2], bits[s * 2 + 1]) {
            (0, 0) => n00 += 1,
            (1, 1) => n11 += 1,
            _ => other += 1,
        }
    }
    assert_eq!(other, 0, "Bell support escaped: {} non-{{00,11}} shots", other);
    let p00 = n00 as f64 / shots as f64;
    let p11 = n11 as f64 / shots as f64;
    assert!((p00 - 0.5).abs() < 0.08, "P(00) = {}", p00);
    assert!((p11 - 0.5).abs() < 0.08, "P(11) = {}", p11);
}

#[test]
fn ghz4_sample_z_support_only_zero_and_all_ones() {
    let n: u32 = 4;
    let mut s = CaMps::new(n, 16).unwrap();
    s.h(0).unwrap();
    for k in 1..n {
        s.cnot(k - 1, k).unwrap();
    }

    let mut rng = Lcg(0xCAFE_BABE);
    let shots = 4096u32;
    let randoms = rng.fill((shots as usize) * (n as usize));
    let bits = s.sample_z(shots, &randoms).unwrap();

    let n_us = n as usize;
    for shot in 0..(shots as usize) {
        let row = &bits[shot * n_us..(shot + 1) * n_us];
        let all_zero = row.iter().all(|&b| b == 0);
        let all_ones = row.iter().all(|&b| b == 1);
        assert!(all_zero || all_ones,
                "GHZ_4 sample {} escaped support: {:?}", shot, row);
    }
}

#[test]
fn sample_z_rejects_wrong_random_length() {
    let s = CaMps::new(3, 8).unwrap();
    let too_short = vec![0.0_f64; 5];   // need 4 shots * 3 qubits = 12
    let r = s.sample_z(4, &too_short);
    assert!(r.is_err(), "expected length-mismatch error");
}
