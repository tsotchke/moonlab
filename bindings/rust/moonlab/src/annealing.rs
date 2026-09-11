//! Complete transverse-field Ising/QUBO quantum annealing (Moonlab v1.2.1).

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_anneal_config_t, moonlab_anneal_result_best_bitstring,
    moonlab_anneal_result_best_energy, moonlab_anneal_result_effective_seed,
    moonlab_anneal_result_expected_energy, moonlab_anneal_result_final_norm,
    moonlab_anneal_result_free, moonlab_anneal_result_ground_bitstring,
    moonlab_anneal_result_ground_degeneracy, moonlab_anneal_result_ground_energy,
    moonlab_anneal_result_most_likely_bitstring, moonlab_anneal_result_num_qubits,
    moonlab_anneal_result_num_samples, moonlab_anneal_result_problem_gap,
    moonlab_anneal_result_residual_energy, moonlab_anneal_result_sample,
    moonlab_anneal_result_success_probability, moonlab_anneal_result_t,
    moonlab_quantum_anneal_ising, moonlab_quantum_anneal_qubo, moonlab_qubo_to_ising,
};
use std::ptr;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnealSchedule {
    Linear = 0,
    Quadratic = 1,
    Cosine = 2,
    Piecewise = 3,
}

#[derive(Debug, Clone)]
pub struct AnnealConfig {
    pub total_time: f64,
    pub num_steps: usize,
    pub num_samples: usize,
    pub seed: u64,
    pub schedule: AnnealSchedule,
    pub driver_strength: f64,
    pub problem_strength: f64,
    pub second_order: bool,
}

impl Default for AnnealConfig {
    fn default() -> Self {
        Self {
            total_time: 10.0,
            num_steps: 1000,
            num_samples: 1024,
            seed: 0,
            schedule: AnnealSchedule::Cosine,
            driver_strength: 1.0,
            problem_strength: 1.0,
            second_order: true,
        }
    }
}

impl AnnealConfig {
    fn ffi(&self) -> moonlab_anneal_config_t {
        moonlab_anneal_config_t {
            total_time: self.total_time,
            num_steps: self.num_steps,
            num_samples: self.num_samples,
            seed: self.seed,
            schedule: self.schedule as _,
            driver_strength: self.driver_strength,
            problem_strength: self.problem_strength,
            second_order: i32::from(self.second_order),
            schedule_points: ptr::null(),
            num_schedule_points: 0,
            reverse_anneal: 0,
            initial_bitstring: 0,
            reverse_s_target: 0.0,
            reverse_hold_fraction: 0.0,
            anneal_offsets: ptr::null(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnnealResult {
    pub num_qubits: usize,
    pub effective_seed: u64,
    pub best_bitstring: u64,
    pub most_likely_bitstring: u64,
    pub ground_bitstring: u64,
    pub ground_degeneracy: usize,
    pub best_energy: f64,
    pub ground_energy: f64,
    pub expected_energy: f64,
    pub success_probability: f64,
    pub residual_energy: f64,
    pub problem_gap: f64,
    pub final_norm: f64,
    pub samples: Vec<u64>,
    pub sample_energies: Vec<f64>,
}

struct ResultGuard(*mut moonlab_anneal_result_t);
impl Drop for ResultGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { moonlab_anneal_result_free(self.0) };
            self.0 = ptr::null_mut();
        }
    }
}

fn collect(raw: *mut moonlab_anneal_result_t) -> Result<AnnealResult> {
    let guard = ResultGuard(raw);
    if guard.0.is_null() {
        return Err(QuantumError::Ffi("annealing returned NULL".to_string()));
    }
    let count = unsafe { moonlab_anneal_result_num_samples(guard.0) };
    let mut samples = Vec::with_capacity(count);
    let mut sample_energies = Vec::with_capacity(count);
    for index in 0..count {
        let mut bits = 0u64;
        let mut energy = 0.0;
        let rc = unsafe { moonlab_anneal_result_sample(guard.0, index, &mut bits, &mut energy) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "moonlab_anneal_result_sample({index}) rc={rc}"
            )));
        }
        samples.push(bits);
        sample_energies.push(energy);
    }
    Ok(unsafe {
        AnnealResult {
            num_qubits: moonlab_anneal_result_num_qubits(guard.0),
            effective_seed: moonlab_anneal_result_effective_seed(guard.0),
            best_bitstring: moonlab_anneal_result_best_bitstring(guard.0),
            most_likely_bitstring: moonlab_anneal_result_most_likely_bitstring(guard.0),
            ground_bitstring: moonlab_anneal_result_ground_bitstring(guard.0),
            ground_degeneracy: moonlab_anneal_result_ground_degeneracy(guard.0),
            best_energy: moonlab_anneal_result_best_energy(guard.0),
            ground_energy: moonlab_anneal_result_ground_energy(guard.0),
            expected_energy: moonlab_anneal_result_expected_energy(guard.0),
            success_probability: moonlab_anneal_result_success_probability(guard.0),
            residual_energy: moonlab_anneal_result_residual_energy(guard.0),
            problem_gap: moonlab_anneal_result_problem_gap(guard.0),
            final_norm: moonlab_anneal_result_final_norm(guard.0),
            samples,
            sample_energies,
        }
    })
}

fn square_dimension(matrix: &[f64]) -> Result<usize> {
    let n = (matrix.len() as f64).sqrt() as usize;
    if n == 0 || n.checked_mul(n) != Some(matrix.len()) {
        return Err(QuantumError::Ffi(
            "matrix must be non-empty and square".to_string(),
        ));
    }
    Ok(n)
}

pub fn anneal_ising(
    fields: &[f64],
    couplings: &[f64],
    offset: f64,
    config: &AnnealConfig,
) -> Result<AnnealResult> {
    let n = square_dimension(couplings)?;
    if fields.len() != n {
        return Err(QuantumError::Ffi(
            "fields length must match coupling matrix".to_string(),
        ));
    }
    let ffi_config = config.ffi();
    let mut raw = ptr::null_mut();
    let rc = unsafe {
        moonlab_quantum_anneal_ising(
            n,
            fields.as_ptr(),
            couplings.as_ptr(),
            offset,
            &ffi_config,
            &mut raw,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_quantum_anneal_ising rc={rc}"
        )));
    }
    collect(raw)
}

pub fn anneal_qubo(qubo: &[f64], offset: f64, config: &AnnealConfig) -> Result<AnnealResult> {
    let n = square_dimension(qubo)?;
    let ffi_config = config.ffi();
    let mut raw = ptr::null_mut();
    let rc =
        unsafe { moonlab_quantum_anneal_qubo(n, qubo.as_ptr(), offset, &ffi_config, &mut raw) };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_quantum_anneal_qubo rc={rc}"
        )));
    }
    collect(raw)
}

pub fn qubo_to_ising(qubo: &[f64], offset: f64) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let n = square_dimension(qubo)?;
    let mut h = vec![0.0; n];
    let mut j = vec![0.0; n * n];
    let mut converted_offset = 0.0;
    let rc = unsafe {
        moonlab_qubo_to_ising(
            n,
            qubo.as_ptr(),
            offset,
            h.as_mut_ptr(),
            j.as_mut_ptr(),
            &mut converted_offset,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("moonlab_qubo_to_ising rc={rc}")));
    }
    Ok((h, j, converted_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_qubo_replays() {
        let config = AnnealConfig {
            total_time: 12.0,
            num_steps: 1200,
            num_samples: 128,
            seed: 0x1234_5678_9abc_def0,
            ..AnnealConfig::default()
        };
        let q = [-1.0, 1.0, 1.0, -1.0];
        let a = anneal_qubo(&q, 1.0, &config).unwrap();
        let b = anneal_qubo(&q, 1.0, &config).unwrap();
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.sample_energies, b.sample_energies);
        assert_eq!(a.ground_degeneracy, 2);
        assert_eq!(a.ground_energy, 0.0);
        assert!(a.success_probability > 0.95);
    }

    #[test]
    fn qubo_conversion_is_exact() {
        let q = [-1.5, 0.7, 0.3, 2.0];
        let (h, j, off) = qubo_to_ising(&q, 0.25).unwrap();
        for bits in 0..4u64 {
            let x0 = (bits & 1) as f64;
            let x1 = ((bits >> 1) & 1) as f64;
            let q_energy = 0.25 + x0 * q[0] * x0 + x0 * q[1] * x1 + x1 * q[2] * x0 + x1 * q[3] * x1;
            let z0 = 1.0 - 2.0 * x0;
            let z1 = 1.0 - 2.0 * x1;
            let i_energy = off + h[0] * z0 + h[1] * z1 + j[1] * z0 * z1;
            assert!((q_energy - i_energy).abs() < 1e-13);
        }
    }
}
