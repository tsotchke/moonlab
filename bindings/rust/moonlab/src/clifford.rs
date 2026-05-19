//! Standalone Aaronson-Gottesman Clifford tableau (since v0.2.0;
//! safe Rust wrapper since v0.4.6).
//!
//! An n-qubit Clifford state is represented by 2n Pauli operators
//! (n stabilisers and n destabilisers) with sign bits, occupying
//! `2n * (2n + 1)` bits.  Every Clifford gate acts on the tableau
//! in `O(n)` elementary bit operations and a Z-basis measurement in
//! `O(n^2)`.  Useful for `n > 32` GHZ / Bell / stabiliser-code
//! experiments that exceed the dense state-vector ceiling.
//!
//! Mirrors the v0.2.1 Python `moonlab.clifford.Clifford` and the
//! v0.4.5 TypeScript `CliffordTableau`.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::clifford::CliffordTableau;
//!
//! // GHZ state on 64 qubits.
//! let mut c = CliffordTableau::new(64).unwrap();
//! c.h(0).unwrap();
//! for q in 1..64 {
//!     c.cnot(0, q).unwrap();
//! }
//!
//! // Sample one bitstring: all zeros or all ones, with equal prob.
//! let bits = c.sample_all().unwrap();
//! assert!(bits == 0 || bits == u64::MAX);
//! ```
//!
//! Reference: Aaronson and Gottesman, "Improved simulation of
//! stabilizer circuits", Phys. Rev. A 70, 052328 (2004),
//! arXiv:quant-ph/0406196.

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    clifford_cnot, clifford_cz, clifford_h, clifford_measure,
    clifford_num_qubits, clifford_s, clifford_s_dag, clifford_sample_all,
    clifford_swap, clifford_tableau_create, clifford_tableau_free,
    clifford_tableau_t, clifford_x, clifford_y, clifford_z,
};
use std::ptr;

/// Outcome of a single-qubit Z-basis measurement on a
/// [`CliffordTableau`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MeasureResult {
    /// Z-basis outcome bit: `0` or `1`.
    pub outcome: u8,
    /// `true` if every stabiliser commuted with `Z_q` and the
    /// outcome was fixed by the tableau alone; `false` if a
    /// stabiliser anticommuted and the result was drawn from the
    /// RNG state.
    pub deterministic: bool,
}

/// Owned handle to an Aaronson-Gottesman Clifford tableau.
///
/// Initial state is `|0...0>`.  The struct also owns a splitmix64
/// RNG state used by `measure` / `sample_all`; seed it explicitly
/// with [`CliffordTableau::set_rng_seed`] for reproducibility.
pub struct CliffordTableau {
    handle: *mut clifford_tableau_t,
    num_qubits: usize,
    rng_state: u64,
}

// SAFETY: CliffordTableau owns its handle and is not shared across
// threads without external synchronization.
unsafe impl Send for CliffordTableau {}

impl CliffordTableau {
    /// Allocate a fresh tableau in `|0..0>` on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(QuantumError::InvalidQubit { index: 0, max: 1 });
        }
        let handle = unsafe { clifford_tableau_create(num_qubits) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_qubits));
        }
        // Seed splitmix64 from std::process time + handle address.
        // Never zero; callers can override via set_rng_seed.
        let seed = (handle as usize as u64).wrapping_mul(0x9e3779b97f4a7c15) | 1;
        Ok(Self {
            handle,
            num_qubits,
            rng_state: seed,
        })
    }

    /// Number of qubits the tableau was constructed for.
    pub fn num_qubits(&self) -> usize {
        unsafe { clifford_num_qubits(self.handle) }
    }

    /// Override the splitmix64 RNG state used by [`Self::measure`]
    /// and [`Self::sample_all`].  Both calls advance the state in
    /// place; this setter lets the caller pin it for
    /// reproducibility.  A zero seed is replaced with 1.
    pub fn set_rng_seed(&mut self, seed: u64) -> &mut Self {
        self.rng_state = if seed == 0 { 1 } else { seed };
        self
    }

    // ---- Single-qubit Clifford gates --------------------------------------

    /// Hadamard on qubit `q`.
    pub fn h(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_h(self.handle, q) };
        self.guard_rc(rc, "clifford_h")
    }
    /// Phase gate `S` on qubit `q`.
    pub fn s(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_s(self.handle, q) };
        self.guard_rc(rc, "clifford_s")
    }
    /// `S^dagger` on qubit `q`.
    pub fn sdag(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_s_dag(self.handle, q) };
        self.guard_rc(rc, "clifford_s_dag")
    }
    /// Pauli `X` on qubit `q`.
    pub fn x(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_x(self.handle, q) };
        self.guard_rc(rc, "clifford_x")
    }
    /// Pauli `Y` on qubit `q`.
    pub fn y(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_y(self.handle, q) };
        self.guard_rc(rc, "clifford_y")
    }
    /// Pauli `Z` on qubit `q`.
    pub fn z(&mut self, q: usize) -> Result<&mut Self> {
        self.check_qubit(q)?;
        let rc = unsafe { clifford_z(self.handle, q) };
        self.guard_rc(rc, "clifford_z")
    }

    // ---- Two-qubit Clifford gates -----------------------------------------

    /// CNOT with the named control and target qubits.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.check_qubit(control)?;
        self.check_qubit(target)?;
        let rc = unsafe { clifford_cnot(self.handle, control, target) };
        self.guard_rc(rc, "clifford_cnot")
    }
    /// Symmetric CZ.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        let rc = unsafe { clifford_cz(self.handle, a, b) };
        self.guard_rc(rc, "clifford_cz")
    }
    /// SWAP `a` <-> `b`.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        let rc = unsafe { clifford_swap(self.handle, a, b) };
        self.guard_rc(rc, "clifford_swap")
    }

    // ---- Measurement / sampling -------------------------------------------

    /// Z-basis measurement on qubit `q`.  Mutates the tableau and
    /// advances the internal splitmix64 RNG state.
    pub fn measure(&mut self, q: usize) -> Result<MeasureResult> {
        self.check_qubit(q)?;
        let mut rng = self.rng_state;
        let mut outcome: std::os::raw::c_int = 0;
        let mut kind: std::os::raw::c_int = 0;
        let rc = unsafe {
            clifford_measure(self.handle, q, &mut rng, &mut outcome, &mut kind)
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("clifford_measure rc={rc}")));
        }
        self.rng_state = rng;
        Ok(MeasureResult {
            outcome: if outcome == 0 { 0 } else { 1 },
            deterministic: kind == 0,
        })
    }

    /// Draw an n-bit computational-basis sample.  Mutates the
    /// tableau and advances the RNG state.  Caps at 64 qubits
    /// because the C entry point returns a single `uint64_t`.
    pub fn sample_all(&mut self) -> Result<u64> {
        if self.num_qubits > 64 {
            return Err(QuantumError::UnsupportedOperation(format!(
                "sample_all() supports up to 64 qubits; got {}.  \
                 Use measure(q) in a loop for wider tableaus.",
                self.num_qubits
            )));
        }
        let mut rng = self.rng_state;
        let mut result: u64 = 0;
        let rc =
            unsafe { clifford_sample_all(self.handle, &mut rng, &mut result) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "clifford_sample_all rc={rc}"
            )));
        }
        self.rng_state = rng;
        Ok(result)
    }

    // ---- Internal helpers --------------------------------------------------

    fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.num_qubits {
            return Err(QuantumError::InvalidQubit {
                index: q,
                max: self.num_qubits,
            });
        }
        Ok(())
    }

    fn guard_rc(&mut self, rc: i32, name: &'static str) -> Result<&mut Self> {
        if rc != 0 {
            Err(QuantumError::Ffi(format!("{name} rc={rc}")))
        } else {
            Ok(self)
        }
    }
}

impl Drop for CliffordTableau {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { clifford_tableau_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_num_qubits() {
        let c = CliffordTableau::new(8).unwrap();
        assert_eq!(c.num_qubits(), 8);
    }

    #[test]
    fn reject_zero_qubits() {
        assert!(CliffordTableau::new(0).is_err());
    }

    #[test]
    fn ground_state_measurements_are_deterministic_zero() {
        let mut c = CliffordTableau::new(4).unwrap();
        // |0000> measured in Z is deterministic 0 on every qubit.
        for q in 0..4 {
            let r = c.measure(q).unwrap();
            assert_eq!(r.outcome, 0, "qubit {q}");
            assert!(r.deterministic, "qubit {q} should be deterministic");
        }
    }

    #[test]
    fn ghz_sample_collapses_to_aligned_string() {
        // n-qubit GHZ via H_0 + ladder of CNOTs.  A single computational-
        // basis sample on GHZ must come out either all zeros or all ones.
        let n: usize = 6;
        let mut c = CliffordTableau::new(n).unwrap();
        c.h(0).unwrap();
        for q in 1..n {
            c.cnot(0, q).unwrap();
        }
        let bits = c.sample_all().unwrap();
        let mask = (1u64 << n) - 1;
        assert!(
            bits == 0 || bits == mask,
            "GHZ sample = 0x{bits:x} not in {{0, 0x{mask:x}}}"
        );
    }

    #[test]
    fn measure_advances_rng_state_on_random_branch() {
        // |+> = H|0> measured in Z is the random branch.  Pin a seed
        // and verify two runs from the same seed agree.
        let mut a = CliffordTableau::new(1).unwrap();
        a.set_rng_seed(0xdeadbeefcafebabe);
        a.h(0).unwrap();
        let ra = a.measure(0).unwrap();
        assert!(!ra.deterministic);

        let mut b = CliffordTableau::new(1).unwrap();
        b.set_rng_seed(0xdeadbeefcafebabe);
        b.h(0).unwrap();
        let rb = b.measure(0).unwrap();
        assert_eq!(ra.outcome, rb.outcome);
    }

    #[test]
    fn cnot_index_range_check() {
        let mut c = CliffordTableau::new(2).unwrap();
        assert!(c.cnot(0, 5).is_err());
        assert!(c.cnot(5, 0).is_err());
    }
}
