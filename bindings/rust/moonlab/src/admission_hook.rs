//! Safe wrapper for the v1.0.3 control-plane admission hook
//! (`moonlab_control_server_set_admission_hook`).
//!
//! A Rust-built overlay installs its admission policy as a closure
//! (`Fn(&AdmissionRequest) -> AdmissionDecision`) and the wrapper
//! hands the C side a `extern "C"` trampoline that bridges the
//! call.  The closure runs on the C worker thread, so it must be
//! `Send + Sync + 'static`.
//!
//! Mirrors `bindings/python/moonlab/control_plane.py`
//! `ControlPlaneServer.set_admission_hook`.
//!
//! Example:
//!
//! ```no_run
//! use moonlab::admission_hook::{AdmissionHook, AdmissionDecision};
//! use moonlab_sys::moonlab_control_server_t;
//!
//! # let server: *mut moonlab_control_server_t = std::ptr::null_mut();
//! let hook = AdmissionHook::new(|req| {
//!     if req.tenant_id() == Some("acme-corp") {
//!         AdmissionDecision::Refused(-405)   // MOONLAB_CONTROL_REJECTED
//!     } else {
//!         AdmissionDecision::Admitted
//!     }
//! });
//! unsafe { hook.install(server)?; }
//! // Drop `hook` only AFTER the server is shut down -- the C side
//! // keeps the function pointer alive across worker dispatches.
//! # Ok::<(), moonlab::QuantumError>(())
//! ```

use crate::QuantumError;
use moonlab_sys::moonlab_control_server_t;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

/// What the admission hook saw about a single incoming request.
/// Lifetimes tie strings to the C buffers; the closure must NOT
/// retain references past its return.
pub struct AdmissionRequest<'a> {
    tenant_id: Option<&'a str>,
    verb:      &'a str,
    num_qubits: i32,
    num_shots:  i32,
}

impl<'a> AdmissionRequest<'a> {
    /// Tenant identifier from the `AUTH <tenant>:<hmac>` form, or
    /// `None` for the legacy `AUTH <hmac>` form / no auth.
    pub fn tenant_id(&self) -> Option<&str> { self.tenant_id }
    /// Verb header: `"CIRCUIT"`, `"SHOTS"`, etc.
    pub fn verb(&self) -> &str { self.verb }
    /// Qubit count if known at admission time; `-1` otherwise.
    pub fn num_qubits(&self) -> i32 { self.num_qubits }
    /// Shot count for SHOTS requests, `0` for non-shots.
    pub fn num_shots(&self) -> i32 { self.num_shots }
}

/// Outcome of an admission decision.  Negative codes propagate to
/// the wire as `ERR <code> <msg>`; positive numbers are coerced to
/// `Admitted` since the C side treats only zero as "allow".
pub enum AdmissionDecision {
    /// Allow the request to proceed.
    Admitted,
    /// Refuse with this status code.  Recommended values:
    /// - `-405` `MOONLAB_CONTROL_REJECTED` (paid-tier gate, lockout)
    /// - `-408` `MOONLAB_CONTROL_RATE_LIMITED` (quota exhausted)
    /// Overlay-defined codes pass through verbatim.
    Refused(i32),
}

type HookFn = dyn Fn(&AdmissionRequest) -> AdmissionDecision + Send + Sync + 'static;

/// Owned admission-hook handle.  Drop after the server is closed.
pub struct AdmissionHook {
    inner: Arc<HookFn>,
    installed_on: std::cell::Cell<Option<*mut moonlab_control_server_t>>,
}

unsafe impl Send for AdmissionHook {}
unsafe impl Sync for AdmissionHook {}

impl AdmissionHook {
    /// Build a hook from a `Send + Sync` closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&AdmissionRequest) -> AdmissionDecision + Send + Sync + 'static,
    {
        AdmissionHook {
            inner: Arc::new(f),
            installed_on: std::cell::Cell::new(None),
        }
    }

    /// Install the hook on `server` (an opened
    /// `moonlab_control_server_t*` from `moonlab_control_server_open`).
    /// Caller must keep `self` alive until the server is closed; the
    /// C side stores a raw `void *` pointer into our `Arc` clone.
    ///
    /// # Safety
    /// `server` must be a valid handle from `moonlab_control_server_open`
    /// that has not been closed.  Concurrent calls from multiple
    /// threads on the same handle are not safe.
    pub unsafe fn install(
        &self,
        server: *mut moonlab_control_server_t,
    ) -> Result<(), QuantumError> {
        if server.is_null() {
            return Err(QuantumError::Ffi("install: server handle is null".into()));
        }
        // Hand the C side a raw `Box<Arc<HookFn>>` thin pointer.
        // The outer Box gives us a thin (single-word) pointer that
        // C can store as void*; the inner Arc shares ownership with
        // self.inner so the closure stays alive after we leak.
        let arc = Arc::clone(&self.inner);
        let boxed: Box<Arc<HookFn>> = Box::new(arc);
        let ctx = Box::into_raw(boxed) as *mut c_void;

        let rc = unsafe {
            moonlab_sys::moonlab_control_server_set_admission_hook(
                server,
                Some(trampoline),
                ctx,
            )
        };
        if rc != 0 {
            // Reclaim the Box so we don't leak it on the error path.
            let _ = unsafe { Box::from_raw(ctx as *mut Arc<HookFn>) };
            return Err(QuantumError::Ffi(format!(
                "set_admission_hook rc={rc}"
            )));
        }
        self.installed_on.set(Some(server));
        Ok(())
    }

    /// Clear an installed hook.  The server reverts to "no hook".
    ///
    /// # Safety
    /// Same as `install`: handle must still be valid + not closed.
    pub unsafe fn uninstall(
        &self,
        server: *mut moonlab_control_server_t,
    ) -> Result<(), QuantumError> {
        let rc = unsafe {
            moonlab_sys::moonlab_control_server_set_admission_hook(
                server, None, std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "set_admission_hook(NULL) rc={rc}"
            )));
        }
        self.installed_on.set(None);
        Ok(())
    }
}

/// extern "C" trampoline -- C calls this with our Arc pointer as
/// the ctx; we cast back and invoke the Rust closure.  Panics in
/// the closure are caught and reported as `MOONLAB_CONTROL_REJECTED`
/// so a panicking hook can't tear down the server.
extern "C" fn trampoline(
    tenant_id_p: *const c_char,
    verb_p:      *const c_char,
    num_qubits:  c_int,
    num_shots:   c_int,
    ctx:         *mut c_void,
) -> c_int {
    if ctx.is_null() { return 0; }
    // SAFETY: ctx points to a Box<Arc<HookFn>> we leaked in
    // install().  Borrow through the box without taking ownership;
    // the AdmissionHook destructor will reclaim when the user
    // drops the wrapper.
    let arc: &Arc<HookFn> = unsafe { &*(ctx as *const Arc<HookFn>) };
    let hook: &HookFn = arc.as_ref();

    let tenant_str: Option<&str> = if tenant_id_p.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(tenant_id_p) }.to_str().ok()
    };
    let verb_str: &str = if verb_p.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(verb_p) }.to_str().unwrap_or("")
    };

    let req = AdmissionRequest {
        tenant_id:  tenant_str,
        verb:       verb_str,
        num_qubits: num_qubits as i32,
        num_shots:  num_shots as i32,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hook(&req)
    }));
    match result {
        Ok(AdmissionDecision::Admitted)   => 0,
        Ok(AdmissionDecision::Refused(c)) => c,
        Err(_) => -405,  // MOONLAB_CONTROL_REJECTED -- catch all panics
    }
}
