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
    moonlab_decoder_code_t, moonlab_decoder_decode, moonlab_decoder_decode_by_name,
    moonlab_decoder_input_t, moonlab_decoder_slot_available,
    moonlab_decoder_slot_name, moonlab_list_decoders, moonlab_lookup_decoder,
    moonlab_num_decoders, moonlab_register_decoder, moonlab_unregister_decoder,
};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

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

/// Decode through a name-keyed registry lookup -- mirror of
/// [`decode`] but with arbitrary registered names (built-in or
/// custom).  Since v1.0.3.
pub fn decode_by_name(name: &str,
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
    let c_name = CString::new(name)
        .map_err(|e| QuantumError::Ffi(format!("bad decoder name: {e}")))?;
    let rc = unsafe {
        moonlab_decoder_decode_by_name(c_name.as_ptr(), &input)
    };
    if rc == MOONLAB_DECODER_OK {
        Ok(corrections)
    } else {
        Err(QuantumError::Ffi(format!("decode_by_name({name:?}): rc={rc}")))
    }
}

/// Number of decoders currently registered.
pub fn num_decoders() -> i32 {
    unsafe { moonlab_num_decoders() }
}

/// Names of all currently-registered decoders.
pub fn list_decoders() -> Vec<String> {
    let n = num_decoders();
    if n <= 0 { return Vec::new(); }
    let mut buf: Vec<*const c_char> = vec![ptr::null(); n as usize];
    let written = unsafe {
        moonlab_list_decoders(buf.as_mut_ptr(), n) as usize
    };
    buf.into_iter().take(written).filter_map(|p| {
        if p.is_null() { None } else {
            unsafe { CStr::from_ptr(p).to_str().ok().map(|s| s.to_owned()) }
        }
    }).collect()
}

/// Description (or empty string) of a registered decoder.  None if
/// the name is not registered.
pub fn lookup_decoder(name: &str) -> Option<String> {
    let c_name = CString::new(name).ok()?;
    let p = unsafe { moonlab_lookup_decoder(c_name.as_ptr()) };
    if p.is_null() { return None; }
    unsafe {
        let entry = &*p;
        if entry.description.is_null() { Some(String::new()) }
        else { CStr::from_ptr(entry.description).to_str().ok().map(|s| s.to_owned()) }
    }
}

/// Caller-supplied decoder closure.  Receives the same arguments as
/// [`decode`] (geometry + syndromes + seed) and must return the
/// length-`num_qubits` correction byte vector.  Errors flatten to
/// `MOONLAB_DECODER_OOM`.
pub type DecoderClosure =
    Box<dyn Fn(&CodeGeometry, &[u8], u64) -> Result<Vec<u8>> + Send + Sync>;

// Keep registered trampolines + closures alive for the program
// lifetime.  Indexed by name so unregister can free them; never
// shrink (entries become tombstones) so the C runtime's stale
// ctx is always still valid.
struct RegistrySlot {
    closure: Box<DecoderClosure>,
}
static REGISTERED_DECODERS: Mutex<Vec<(String, Box<RegistrySlot>)>> =
    Mutex::new(Vec::new());

extern "C" fn rust_decoder_trampoline(
    input: *const moonlab_decoder_input_t,
    ctx: *mut c_void,
) -> i32 {
    if input.is_null() || ctx.is_null() { return MOONLAB_DECODER_BAD_ARG; }
    let slot: &RegistrySlot = unsafe { &*(ctx as *const RegistrySlot) };
    let in_ref = unsafe { &*input };
    let code_ref = unsafe { &*in_ref.code };
    let geom = CodeGeometry {
        distance: code_ref.distance,
        num_qubits: code_ref.num_qubits,
        is_toric: code_ref.is_toric != 0,
    };
    let syndromes = unsafe {
        std::slice::from_raw_parts(in_ref.syndromes, in_ref.num_stabilisers as usize)
    };
    let result = (slot.closure)(&geom, syndromes, in_ref.rng_seed);
    match result {
        Ok(corr) => {
            let n_q = code_ref.num_qubits as usize;
            if corr.len() < n_q { return MOONLAB_DECODER_OOM; }
            for i in 0..n_q {
                unsafe { *in_ref.corrections.add(i) = corr[i]; }
            }
            MOONLAB_DECODER_OK
        }
        Err(_) => MOONLAB_DECODER_OOM,
    }
}

/// Register a Rust closure as a decoder under `name`.  Re-registering
/// the same name replaces the previous slot.  The closure is leaked
/// (kept alive for program lifetime) so the C runtime's stable ctx
/// pointer remains valid even after unregister.  Since v1.0.3.
pub fn register_decoder<F>(name: &str,
                           description: &str,
                           f: F) -> Result<()>
where
    F: Fn(&CodeGeometry, &[u8], u64) -> Result<Vec<u8>> + Send + Sync + 'static,
{
    let c_name = CString::new(name)
        .map_err(|e| QuantumError::Ffi(format!("bad decoder name: {e}")))?;
    let c_desc = if description.is_empty() { None } else {
        Some(CString::new(description)
             .map_err(|e| QuantumError::Ffi(format!("bad description: {e}")))?)
    };
    let slot = Box::new(RegistrySlot {
        closure: Box::new(Box::new(f) as DecoderClosure),
    });
    let ctx_ptr: *mut c_void = (&*slot) as *const RegistrySlot as *mut c_void;
    let rc = unsafe {
        moonlab_register_decoder(
            c_name.as_ptr(),
            Some(rust_decoder_trampoline),
            ctx_ptr,
            c_desc.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null()),
        )
    };
    if rc != MOONLAB_DECODER_OK {
        return Err(QuantumError::Ffi(format!("register_decoder({name:?}): rc={rc}")));
    }
    REGISTERED_DECODERS.lock().unwrap().push((name.to_owned(), slot));
    Ok(())
}

/// Remove a decoder from the registry.  The closure remains alive in
/// case the registry's ctx pointer is dereferenced by a racing call.
pub fn unregister_decoder(name: &str) -> Result<()> {
    let c_name = CString::new(name)
        .map_err(|e| QuantumError::Ffi(format!("bad decoder name: {e}")))?;
    let rc = unsafe { moonlab_unregister_decoder(c_name.as_ptr()) };
    if rc != MOONLAB_DECODER_OK {
        return Err(QuantumError::Ffi(format!("unregister_decoder({name:?}): rc={rc}")));
    }
    Ok(())
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
        // GREEDY + MWPM_EXACT always available; SBNN gated by build
        // flag (default off); PYMATCHING + LIBIRREP_SS depend on
        // optional linkage.
        assert!(slot_available(DecoderSlot::Greedy));
        assert!(slot_available(DecoderSlot::MwpmExact));
        assert!(!slot_available(DecoderSlot::Sbnn));
        // PYMATCHING + LIBIRREP_SS: either availability is acceptable.
        let _ = slot_available(DecoderSlot::Pymatching);
        let _ = slot_available(DecoderSlot::LibirrepSs);
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

    #[test]
    fn registry_lists_builtins() {
        // num_decoders triggers the auto-init pthread_once, so the
        // five built-ins are present.
        let names = list_decoders();
        assert!(names.contains(&"greedy".to_string()),
                "greedy missing from registry; got {names:?}");
        assert!(names.contains(&"mwpm_exact".to_string()));
        assert!(names.contains(&"sbnn".to_string()));
        assert!(names.contains(&"libirrep_single_shot".to_string()));
        assert!(names.contains(&"pymatching".to_string()));
    }

    #[test]
    fn registry_lookup_unknown_returns_none() {
        assert!(lookup_decoder("definitely-not-a-real-decoder").is_none());
    }

    #[test]
    fn registry_register_dispatch_unregister() {
        // Decoder that fills the corrections vector with a sentinel
        // value -- proves the closure ran and bytes round-tripped.
        register_decoder(
            "rust-sentinel-test",
            "round-trip sanity decoder",
            |geom, _syndromes, _seed| {
                let mut out = vec![0u8; geom.num_qubits as usize];
                out[0] = 0xC3;
                Ok(out)
            },
        ).unwrap();
        assert!(list_decoders().contains(&"rust-sentinel-test".to_string()));
        let desc = lookup_decoder("rust-sentinel-test").unwrap();
        assert!(desc.contains("round-trip"));

        let d = 3;
        let corr = decode_by_name(
            "rust-sentinel-test",
            &CodeGeometry { distance: d as i32, num_qubits: (2 * d * d) as i32, is_toric: true },
            &vec![0u8; d * d], 0,
        ).unwrap();
        assert_eq!(corr[0], 0xC3);
        assert!(corr[1..].iter().all(|&b| b == 0));

        unregister_decoder("rust-sentinel-test").unwrap();
        assert!(!list_decoders().contains(&"rust-sentinel-test".to_string()));
    }

    #[test]
    fn registry_unregister_unknown_errs() {
        assert!(unregister_decoder("never-registered").is_err());
    }
}
