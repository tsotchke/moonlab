//! Decoder-bench Rust binding -- since v0.7.3.
//!
//! Wraps `src/applications/decoder_bench.{c,h}` (v0.6.7 + v0.6.9 +
//! v0.7.2 wiring).  Five-slot dispatcher for the QEC decoder zoo.
//!
//! ```no_run
//! use moonlab::decoder::{DecoderSlot, decode, CodeGeometry};
//!
//! let d = 5;
//! let mut s = vec![0u8; d * d];
//! s[0]  = 1;
//! s[12] = 1;  // (2, 2) on a d=5 lattice
//! let corr = decode(DecoderSlot::MwpmExact,
//!                   &CodeGeometry { distance: d as i32,
//!                                   num_qubits: (2 * d * d) as i32,
//!                                   is_toric: true },
//!                   &s, 0).unwrap();
//! assert_eq!(corr.iter().map(|&b| b as i32).sum::<i32>(), 4);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_decoder_code_t, moonlab_decoder_decode, moonlab_decoder_input_t,
    moonlab_decoder_slot_available, moonlab_decoder_slot_name,
};
use std::ffi::CStr;

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecoderSlot {
    Greedy      = 0,
    MwpmExact   = 1,
    Sbnn        = 2,
    LibirrepSs  = 3,
    Pymatching  = 4,
}

pub const MOONLAB_DECODER_OK: i32 = 0;
pub const MOONLAB_DECODER_NOT_BUILT: i32 = -401;
pub const MOONLAB_DECODER_BAD_ARG: i32 = -402;
pub const MOONLAB_DECODER_INFEASIBLE: i32 = -403;
pub const MOONLAB_DECODER_OOM: i32 = -404;

#[derive(Debug, Clone)]
pub struct CodeGeometry {
    pub distance: i32,
    pub num_qubits: i32,
    pub is_toric: bool,
}

pub fn slot_available(slot: DecoderSlot) -> bool {
    unsafe { moonlab_decoder_slot_available(slot as u32) == 1 }
}

pub fn slot_name(slot: DecoderSlot) -> &'static str {
    unsafe {
        let p = moonlab_decoder_slot_name(slot as u32);
        if p.is_null() { "unknown" } else {
            CStr::from_ptr(p).to_str().unwrap_or("unknown")
        }
    }
}

/// Decode a syndrome with the requested slot.  Returns the
/// length-`num_qubits` correction byte vector (0 / 1 per data qubit).
pub fn decode(slot: DecoderSlot,
              code: &CodeGeometry,
              syndromes: &[u8],
              rng_seed: u64) -> Result<Vec<u8>> {
    let c_code = moonlab_decoder_code_t {
        distance: code.distance,
        num_qubits: code.num_qubits,
        is_toric: if code.is_toric { 1 } else { 0 },
    };
    let mut corrections = vec![0u8; code.num_qubits as usize];
    let input = moonlab_decoder_input_t {
        code: &c_code as *const _,
        syndromes: syndromes.as_ptr(),
        corrections: corrections.as_mut_ptr(),
        num_stabilisers: syndromes.len() as i32,
        rng_seed,
    };
    let rc = unsafe { moonlab_decoder_decode(slot as u32, &input) };
    if rc == MOONLAB_DECODER_OK {
        Ok(corrections)
    } else if rc == MOONLAB_DECODER_NOT_BUILT {
        Err(QuantumError::Ffi(format!(
            "decoder slot {slot:?} not built (rebuild with QSIM_ENABLE_*)"
        )))
    } else {
        Err(QuantumError::Ffi(format!("decode({slot:?}): rc={rc}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_names_roundtrip() {
        assert_eq!(slot_name(DecoderSlot::Greedy), "greedy");
        assert_eq!(slot_name(DecoderSlot::MwpmExact), "mwpm_exact");
        assert_eq!(slot_name(DecoderSlot::Sbnn), "sbnn");
    }

    #[test]
    fn slot_availability() {
        assert!(slot_available(DecoderSlot::Greedy));
        assert!(slot_available(DecoderSlot::MwpmExact));
        assert!(!slot_available(DecoderSlot::Sbnn));
        assert!(!slot_available(DecoderSlot::Pymatching));
    }

    #[test]
    fn greedy_zero_syndrome() {
        let d = 3;
        let s = vec![0u8; d * d];
        let corr = decode(
            DecoderSlot::Greedy,
            &CodeGeometry { distance: d as i32, num_qubits: (2 * d * d) as i32, is_toric: true },
            &s, 0,
        ).unwrap();
        assert_eq!(corr.iter().map(|&b| b as i32).sum::<i32>(), 0);
    }

    #[test]
    fn mwpm_exact_l1_4() {
        let d = 5usize;
        let mut s = vec![0u8; d * d];
        s[0]  = 1;
        s[12] = 1;
        let corr = decode(
            DecoderSlot::MwpmExact,
            &CodeGeometry { distance: d as i32, num_qubits: (2 * d * d) as i32, is_toric: true },
            &s, 0,
        ).unwrap();
        assert_eq!(corr.iter().map(|&b| b as i32).sum::<i32>(), 4);
    }

    #[test]
    fn sbnn_not_built() {
        let d = 3;
        let s = vec![0u8; d * d];
        let r = decode(
            DecoderSlot::Sbnn,
            &CodeGeometry { distance: d as i32, num_qubits: (2 * d * d) as i32, is_toric: true },
            &s, 0,
        );
        assert!(r.is_err());
    }
}
