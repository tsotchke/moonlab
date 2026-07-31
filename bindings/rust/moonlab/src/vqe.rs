//! Variational Quantum Eigensolver (since v0.2.0; safe Rust wrapper
//! since v0.4.9).
//!
//! Wraps `src/algorithms/vqe.{c,h}` with idiomatic Rust around the
//! Pauli-Hamiltonian builder, the hardware-efficient ansatz, the five
//! classical optimizers (Adam, L-BFGS, COBYLA, gradient descent,
//! quantum natural gradient), the `vqe_solve` driver, and the
//! parameter-space geometry (Fubini-Study metric, Berry curvature,
//! natural-gradient direction).  Mirrors the Python
//! `moonlab.algorithms.VQE` surface.
//!
//! The C entry points need a `quantum_entropy_ctx_t` for shot-noise
//! sampling, so [`VqeSolver`] leases a hardware-entropy context for
//! its lifetime via the same RAII pattern as [`crate::bell`] and
//! [`crate::grover`].
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::vqe::{PauliHamiltonian, VqeSolver, OptimizerType};
//!
//! let hamiltonian = PauliHamiltonian::h2(0.74).unwrap();
//! let mut solver = VqeSolver::new(
//!     hamiltonian, 2, OptimizerType::Adam,
//! ).unwrap();
//! let result = solver.solve().unwrap();
//! println!("E_0 = {:.6} Ha", result.ground_state_energy);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_vqe_berry_curvature, moonlab_vqe_natural_gradient,
    moonlab_vqe_qgt, pauli_hamiltonian_add_term, pauli_hamiltonian_create,
    pauli_hamiltonian_free, pauli_hamiltonian_t, quantum_entropy_ctx_create_hw,
    quantum_entropy_ctx_destroy, quantum_entropy_ctx_t,
    vqe_ansatz_free, vqe_ansatz_t, vqe_compute_energy,
    vqe_create_h2_hamiltonian, vqe_create_hardware_efficient_ansatz,
    vqe_create_lih_hamiltonian, vqe_exact_ground_state_energy,
    vqe_hartree_to_kcalmol, vqe_optimizer_create, vqe_optimizer_free,
    vqe_optimizer_t, vqe_optimizer_type_t, vqe_result_t, vqe_solve,
    vqe_solver_create, vqe_solver_free, vqe_solver_t,
};
use std::ffi::CString;
use std::ptr;

/// Lease a hardware-entropy context for the lifetime of `Self`.
/// The C-side VQE solver creator demands a non-NULL entropy ctx; we
/// take one in [`VqeSolver::new`] and release it on drop.
struct EntropyGuard {
    ctx: *mut quantum_entropy_ctx_t,
}

impl EntropyGuard {
    fn new() -> Result<Self> {
        let ctx = unsafe { quantum_entropy_ctx_create_hw() };
        if ctx.is_null() {
            return Err(QuantumError::Ffi(
                "quantum_entropy_ctx_create_hw returned NULL".to_string(),
            ));
        }
        Ok(Self { ctx })
    }
}

impl Drop for EntropyGuard {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { quantum_entropy_ctx_destroy(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

/// Classical optimizer for [`VqeSolver`].  Maps to
/// `vqe_optimizer_type_t` in the C header.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum OptimizerType {
    /// Constrained gradient-free optimizer.
    Cobyla = 0,
    /// Limited-memory BFGS (quasi-Newton).
    Lbfgs = 1,
    /// Adam adaptive-moment optimizer (default).
    Adam = 2,
    /// Plain gradient descent.
    GradientDescent = 3,
    /// Quantum natural gradient: gradient descent preconditioned by the
    /// Fubini-Study metric ([`VqeSolver::quantum_geometric_tensor`]).
    Qng = 4,
}

/// Owned Pauli-string Hamiltonian over `num_qubits` qubits.
///
/// `H = sum_i c_i P_i` where each `P_i` is a Pauli string of length
/// `num_qubits` over `{I, X, Y, Z}`.  Construct via the prebuilt
/// molecule helpers ([`h2`](Self::h2), [`lih`](Self::lih)) or via
/// [`builder`](Self::builder) for a custom Hamiltonian.
pub struct PauliHamiltonian {
    ptr: *mut pauli_hamiltonian_t,
}

impl PauliHamiltonian {
    /// H2 molecule Hamiltonian in the STO-3G basis under
    /// Jordan-Wigner.  `bond_distance` is the H-H bond in Angstroms.
    pub fn h2(bond_distance: f64) -> Result<Self> {
        let ptr = unsafe { vqe_create_h2_hamiltonian(bond_distance) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "vqe_create_h2_hamiltonian returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    /// LiH molecule Hamiltonian (4 qubits) at the given Li-H bond
    /// distance in Angstroms.
    pub fn lih(bond_distance: f64) -> Result<Self> {
        let ptr = unsafe { vqe_create_lih_hamiltonian(bond_distance) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "vqe_create_lih_hamiltonian returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Start a custom Hamiltonian over `num_qubits` qubits with
    /// space for `num_terms` Pauli strings.
    pub fn builder(num_qubits: usize, num_terms: usize) -> Result<PauliHamiltonianBuilder> {
        let ptr = unsafe { pauli_hamiltonian_create(num_qubits, num_terms) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "pauli_hamiltonian_create returned NULL".to_string(),
            ));
        }
        Ok(PauliHamiltonianBuilder { ptr, num_terms, next: 0 })
    }

    /// Number of qubits the Hamiltonian acts on.
    pub fn num_qubits(&self) -> usize {
        unsafe { (*self.ptr).num_qubits }
    }

    /// Number of Pauli-string terms in the sum.
    pub fn num_terms(&self) -> usize {
        unsafe { (*self.ptr).num_terms }
    }

    /// Exact ground-state energy by direct diagonalisation.  Returns
    /// `NaN` on failure.  Intended for small Hamiltonians (n <= 10)
    /// to validate VQE convergence.
    pub fn exact_ground_state_energy(&self) -> f64 {
        unsafe { vqe_exact_ground_state_energy(self.ptr) }
    }
}

impl Drop for PauliHamiltonian {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { pauli_hamiltonian_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Fluent builder for a custom [`PauliHamiltonian`].
///
/// Append exactly `num_terms` terms then call [`build`](Self::build).
pub struct PauliHamiltonianBuilder {
    ptr: *mut pauli_hamiltonian_t,
    num_terms: usize,
    next: usize,
}

impl PauliHamiltonianBuilder {
    /// Append one Pauli-string term `c * P` where `pauli` is a
    /// string over `{I, X, Y, Z}` of length `num_qubits`.
    pub fn add_term(mut self, coefficient: f64, pauli: &str) -> Result<Self> {
        if self.next >= self.num_terms {
            return Err(QuantumError::Ffi(format!(
                "Hamiltonian capacity {} exceeded",
                self.num_terms
            )));
        }
        let c_pauli = CString::new(pauli).map_err(|e| {
            QuantumError::Ffi(format!("Pauli string contains NUL: {e}"))
        })?;
        let rc = unsafe {
            pauli_hamiltonian_add_term(
                self.ptr,
                coefficient,
                c_pauli.as_ptr(),
                self.next,
            )
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "pauli_hamiltonian_add_term rc={rc} on '{pauli}'"
            )));
        }
        self.next += 1;
        Ok(self)
    }

    /// Consume the builder and return the finished Hamiltonian.
    pub fn build(self) -> PauliHamiltonian {
        let ptr = self.ptr;
        std::mem::forget(self);
        PauliHamiltonian { ptr }
    }
}

impl Drop for PauliHamiltonianBuilder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { pauli_hamiltonian_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// VQE optimisation result.  Mirrors `vqe_result_t` minus the
/// `optimal_parameters` raw pointer; the parameters are copied into
/// an owned `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct VqeResult {
    /// Final ground-state energy in Hartree (includes nuclear-repulsion).
    pub ground_state_energy: f64,
    /// Same energy converted to kcal/mol.
    pub ground_state_energy_kcal_mol: f64,
    /// Optimal variational parameters at convergence.
    pub optimal_parameters: Vec<f64>,
    /// Number of optimizer iterations consumed.
    pub iterations: usize,
    /// Final gradient norm at exit.
    pub convergence_tolerance: f64,
    /// `true` if the optimizer met the tolerance criterion.
    pub converged: bool,
}

/// Variational quantum eigensolver bundling
/// `(hamiltonian, ansatz, optimizer, entropy)`.
///
/// Construction takes ownership of the Hamiltonian; cleanup of every
/// C-side allocation happens on drop, including the entropy ctx.
pub struct VqeSolver {
    solver: *mut vqe_solver_t,
    ansatz: *mut vqe_ansatz_t,
    optimizer: *mut vqe_optimizer_t,
    // Hamiltonian must outlive `solver` (the C solver borrows the
    // pointer) -- we keep it owned by `Self` and only release it on
    // drop *after* solver+ansatz+optimizer are freed.
    hamiltonian: PauliHamiltonian,
    _entropy: EntropyGuard,
}

impl VqeSolver {
    /// Construct a VQE solver over `hamiltonian` using a hardware-
    /// efficient ansatz of depth `num_layers` and the given
    /// classical optimizer.
    pub fn new(
        hamiltonian: PauliHamiltonian,
        num_layers: usize,
        optimizer_type: OptimizerType,
    ) -> Result<Self> {
        let entropy = EntropyGuard::new()?;
        let num_qubits = hamiltonian.num_qubits();

        let ansatz = unsafe {
            vqe_create_hardware_efficient_ansatz(num_qubits, num_layers)
        };
        if ansatz.is_null() {
            return Err(QuantumError::Ffi(
                "vqe_create_hardware_efficient_ansatz returned NULL".to_string(),
            ));
        }

        let optimizer = unsafe {
            vqe_optimizer_create(optimizer_type as vqe_optimizer_type_t)
        };
        if optimizer.is_null() {
            unsafe { vqe_ansatz_free(ansatz) };
            return Err(QuantumError::Ffi(
                "vqe_optimizer_create returned NULL".to_string(),
            ));
        }

        let solver = unsafe {
            vqe_solver_create(hamiltonian.ptr, ansatz, optimizer, entropy.ctx)
        };
        if solver.is_null() {
            unsafe {
                vqe_optimizer_free(optimizer);
                vqe_ansatz_free(ansatz);
            }
            return Err(QuantumError::Ffi(
                "vqe_solver_create returned NULL".to_string(),
            ));
        }

        Ok(Self {
            solver,
            ansatz,
            optimizer,
            hamiltonian,
            _entropy: entropy,
        })
    }

    /// Number of qubits the Hamiltonian (and ansatz) act on.
    pub fn num_qubits(&self) -> usize {
        self.hamiltonian.num_qubits()
    }

    /// Run the classical-quantum optimisation loop to convergence.
    pub fn solve(&mut self) -> Result<VqeResult> {
        let r: vqe_result_t = unsafe { vqe_solve(self.solver) };

        let optimal_parameters = if r.optimal_parameters.is_null()
            || r.num_parameters == 0
        {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    r.optimal_parameters,
                    r.num_parameters,
                )
                .to_vec()
            }
        };

        let energy_kcal = unsafe { vqe_hartree_to_kcalmol(r.ground_state_energy) };

        Ok(VqeResult {
            ground_state_energy: r.ground_state_energy,
            ground_state_energy_kcal_mol: energy_kcal,
            optimal_parameters,
            iterations: r.iterations,
            convergence_tolerance: r.convergence_tolerance,
            converged: r.converged != 0,
        })
    }

    /// Compute `E(theta) = <psi(theta)|H|psi(theta)>` for an
    /// arbitrary parameter vector.  Useful for plotting the energy
    /// landscape independent of the optimizer.
    pub fn compute_energy(&self, parameters: &[f64]) -> f64 {
        unsafe { vqe_compute_energy(self.solver, parameters.as_ptr()) }
    }

    /// Number of variational parameters the ansatz carries.  Every
    /// geometry entry point below takes and returns arrays sized by it.
    pub fn num_parameters(&self) -> usize {
        unsafe { (*self.ansatz).num_parameters }
    }

    /// Quantum geometric (Fubini-Study) metric of the trial state:
    /// `g_ij = Re[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>]`.
    ///
    /// Returns the row-major `n x n` metric (element `(i, j)` at index
    /// `i * n + j`), where `n = parameters.len()`.  The matrix is real,
    /// symmetric and positive semidefinite: it is the Riemannian metric
    /// the quantum natural gradient descends against.
    ///
    /// Exact analytic derivatives (generator insertion) for the built-in
    /// ansaetze; central differences for a custom circuit.
    pub fn quantum_geometric_tensor(&self, parameters: &[f64]) -> Result<Vec<f64>> {
        let n = parameters.len();
        let mut out = vec![0.0_f64; n * n];
        let rc = unsafe {
            moonlab_vqe_qgt(self.solver, parameters.as_ptr(), out.as_mut_ptr(), n)
        };
        geometry_rc("moonlab_vqe_qgt", rc, n, self.num_parameters())?;
        Ok(out)
    }

    /// Berry curvature of the trial state, the imaginary half of the
    /// quantum geometric tensor:
    /// `F_ij = -2 Im[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>]`.
    ///
    /// Returns the row-major `n x n` curvature.  Real and antisymmetric
    /// (`F_ii = 0`, `F_ji = -F_ij`); a purely real ansatz gives `F = 0`
    /// identically, so a nonzero curvature requires phase-bearing gates
    /// such as the RZ layer of the hardware-efficient ansatz.
    pub fn berry_curvature(&self, parameters: &[f64]) -> Result<Vec<f64>> {
        let n = parameters.len();
        let mut out = vec![0.0_f64; n * n];
        let rc = unsafe {
            moonlab_vqe_berry_curvature(
                self.solver,
                parameters.as_ptr(),
                out.as_mut_ptr(),
                n,
            )
        };
        geometry_rc("moonlab_vqe_berry_curvature", rc, n, self.num_parameters())?;
        Ok(out)
    }

    /// Natural-gradient direction `(g + regularization * I)^-1 * gradient`,
    /// the energy gradient preconditioned by the Tikhonov-regularised
    /// quantum geometric tensor.
    ///
    /// `parameters` and `gradient` must have the same length.  A typical
    /// `regularization` is `1e-3`; it keeps the metric invertible where
    /// the ansatz is locally degenerate.
    pub fn natural_gradient_direction(
        &self,
        parameters: &[f64],
        gradient: &[f64],
        regularization: f64,
    ) -> Result<Vec<f64>> {
        let n = parameters.len();
        if gradient.len() != n {
            return Err(QuantumError::Ffi(format!(
                "natural_gradient_direction: gradient has {} entries, \
                 parameters {n}",
                gradient.len()
            )));
        }
        let mut out = vec![0.0_f64; n];
        let rc = unsafe {
            moonlab_vqe_natural_gradient(
                self.solver,
                parameters.as_ptr(),
                gradient.as_ptr(),
                regularization,
                out.as_mut_ptr(),
                n,
            )
        };
        geometry_rc("moonlab_vqe_natural_gradient", rc, n, self.num_parameters())?;
        Ok(out)
    }
}

/// Translate the shared `0 / -1 / -2 / -3` return contract of the ABI
/// geometry entry points into a [`QuantumError`].
fn geometry_rc(what: &str, rc: i32, requested: usize, ansatz: usize) -> Result<()> {
    match rc {
        0 => Ok(()),
        -1 => Err(QuantumError::NullPointer),
        -2 => Err(QuantumError::Ffi(format!(
            "{what}: {requested} parameters requested, ansatz has {ansatz}"
        ))),
        _ => Err(QuantumError::Ffi(format!("{what} failed (rc={rc})"))),
    }
}

impl Drop for VqeSolver {
    fn drop(&mut self) {
        unsafe {
            if !self.solver.is_null() {
                vqe_solver_free(self.solver);
                self.solver = ptr::null_mut();
            }
            if !self.optimizer.is_null() {
                vqe_optimizer_free(self.optimizer);
                self.optimizer = ptr::null_mut();
            }
            if !self.ansatz.is_null() {
                vqe_ansatz_free(self.ansatz);
                self.ansatz = ptr::null_mut();
            }
        }
        // `hamiltonian` and `_entropy` drop in field order.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2_hamiltonian_has_expected_layout() {
        let h = PauliHamiltonian::h2(0.74).unwrap();
        assert_eq!(h.num_qubits(), 2);
        assert!(h.num_terms() >= 4);
        let exact = h.exact_ground_state_energy();
        // H2 STO-3G at R=0.74A is approximately -1.137 Ha; we don't
        // pin the exact value -- just that it's a sane, negative number.
        assert!(
            exact < 0.0 && exact > -2.0,
            "exact ground state E = {} Ha is out of range",
            exact
        );
    }

    #[test]
    fn custom_hamiltonian_builder_round_trip() {
        // Build the toy 1-qubit Hamiltonian H = 0.5 Z and exact-diag
        // it -- the ground state energy is -0.5.
        let h = PauliHamiltonian::builder(1, 1)
            .unwrap()
            .add_term(0.5, "Z")
            .unwrap()
            .build();
        assert_eq!(h.num_qubits(), 1);
        assert_eq!(h.num_terms(), 1);
        let e0 = h.exact_ground_state_energy();
        assert!((e0 + 0.5).abs() < 1e-10, "E_0 = {} != -0.5", e0);
    }

    #[test]
    fn solver_runs_h2_under_adam_optimizer() {
        let h = PauliHamiltonian::h2(0.74).unwrap();
        let exact = h.exact_ground_state_energy();
        let mut solver = VqeSolver::new(h, 2, OptimizerType::Adam).unwrap();
        let result = solver.solve().unwrap();
        // VQE on Adam in default-iteration mode won't always reach
        // chemical accuracy; we only assert the result is bounded
        // around the exact value and the parameter vector is filled.
        assert!(
            result.ground_state_energy >= exact - 0.5,
            "E_VQE = {} below E_exact - 0.5 = {}; suggests a wiring bug",
            result.ground_state_energy,
            exact - 0.5
        );
        assert!(
            result.ground_state_energy <= exact + 5.0,
            "E_VQE = {} too far above exact = {}",
            result.ground_state_energy,
            exact
        );
        assert!(!result.optimal_parameters.is_empty());
    }

    /// 2-qubit H2 under a 1-layer hardware-efficient ansatz: 4
    /// parameters, the fixture every geometry test below shares.
    fn h2_qng_solver() -> VqeSolver {
        let h = PauliHamiltonian::h2(0.74).unwrap();
        VqeSolver::new(h, 1, OptimizerType::Qng).unwrap()
    }

    const THETA: [f64; 4] = [0.31, -0.87, 1.24, 0.55];

    #[test]
    fn hardware_efficient_ansatz_has_four_parameters() {
        let solver = h2_qng_solver();
        assert_eq!(solver.num_parameters(), 4);
    }

    #[test]
    fn quantum_geometric_tensor_is_symmetric_psd() {
        let solver = h2_qng_solver();
        let n = THETA.len();
        let g = solver.quantum_geometric_tensor(&THETA).unwrap();
        assert_eq!(g.len(), n * n);
        assert!(g.iter().all(|x| x.is_finite()), "QGT has non-finite entries");

        // Symmetric to round-off.
        for i in 0..n {
            for j in 0..n {
                let d = (g[i * n + j] - g[j * n + i]).abs();
                assert!(d < 1e-9, "g[{i}][{j}] != g[{j}][{i}]: asym {d:.3e}");
            }
        }

        // Positive semidefinite: v^T g v >= 0 for a spanning set of
        // directions -- the axes, and every signed pair combination,
        // which together certify the diagonal and the 2x2 minors.
        for i in 0..n {
            assert!(
                g[i * n + i] >= -1e-12,
                "negative diagonal g[{i}][{i}] = {}",
                g[i * n + i]
            );
            for j in (i + 1)..n {
                for &s in &[1.0_f64, -1.0] {
                    let q = g[i * n + i] + g[j * n + j] + 2.0 * s * g[i * n + j];
                    assert!(
                        q >= -1e-12,
                        "PSD violated on ({i},{j}) sign {s}: v^T g v = {q:.3e}"
                    );
                }
            }
        }

        // And on random directions, which catch indefiniteness the
        // pairwise minors miss.  Deterministic LCG, no rand dependency.
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        for _ in 0..64 {
            let mut v = vec![0.0_f64; n];
            for slot in v.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = ((seed >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            }
            let mut q = 0.0;
            for i in 0..n {
                for j in 0..n {
                    q += v[i] * g[i * n + j] * v[j];
                }
            }
            assert!(q >= -1e-12, "PSD violated: v^T g v = {q:.3e}");
        }
    }

    #[test]
    fn berry_curvature_is_antisymmetric_with_zero_diagonal() {
        let solver = h2_qng_solver();
        let n = THETA.len();
        let f = solver.berry_curvature(&THETA).unwrap();
        assert_eq!(f.len(), n * n);
        assert!(f.iter().all(|x| x.is_finite()), "F has non-finite entries");

        for i in 0..n {
            assert!(
                f[i * n + i].abs() < 1e-12,
                "F[{i}][{i}] = {} is not zero",
                f[i * n + i]
            );
            for j in 0..n {
                let s = (f[i * n + j] + f[j * n + i]).abs();
                assert!(s < 1e-9, "F[{i}][{j}] + F[{j}][{i}] = {s:.3e}");
            }
        }
    }

    #[test]
    fn natural_gradient_solves_the_regularized_metric_system() {
        let solver = h2_qng_solver();
        let n = THETA.len();
        let g = solver.quantum_geometric_tensor(&THETA).unwrap();

        let gradient: Vec<f64> = (0..n).map(|k| 0.1 * (k + 1) as f64).collect();
        let eps = 1e-2;
        let dir = solver
            .natural_gradient_direction(&THETA, &gradient, eps)
            .unwrap();
        assert_eq!(dir.len(), n);

        // (g + eps I) dir must reproduce the gradient.
        for i in 0..n {
            let mut lhs = eps * dir[i];
            for j in 0..n {
                lhs += g[i * n + j] * dir[j];
            }
            let resid = (lhs - gradient[i]).abs();
            assert!(
                resid < 1e-8,
                "row {i}: (g + eps I) x = {lhs}, grad = {}, residual {resid:.3e}",
                gradient[i]
            );
        }
    }

    #[test]
    fn geometry_rejects_a_parameter_count_mismatch() {
        let solver = h2_qng_solver();
        let wrong = [0.1, 0.2, 0.3, 0.4, 0.5];
        assert!(solver.quantum_geometric_tensor(&wrong).is_err());
        assert!(solver.berry_curvature(&wrong).is_err());
        assert!(solver
            .natural_gradient_direction(&wrong, &wrong, 1e-3)
            .is_err());
        // Mismatched gradient length is caught before the FFI call.
        assert!(solver
            .natural_gradient_direction(&THETA, &[0.1, 0.2], 1e-3)
            .is_err());
    }
}
