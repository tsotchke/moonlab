//! Control-plane Rust client -- since v0.8.9.
//!
//! Pure-stdlib TCP client for the v0.8.7 `moonlab_control_serve()`
//! server.  Uses `std::net::TcpStream` only -- no `tokio`, no `hyper`.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::qgtl::{QgtlCircuit, GateType};
//! use moonlab::control_plane::submit_circuit;
//!
//! let mut c = QgtlCircuit::new(2).unwrap();
//! c.add_gate(GateType::H, 0, -1, &[]).unwrap();
//! c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
//! let probs = submit_circuit("127.0.0.1", 8765, &c.serialize().unwrap()).unwrap();
//! assert!((probs[0] - 0.5).abs() < 1e-9);
//! assert!((probs[3] - 0.5).abs() < 1e-9);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_control_hmac_sha3_256, moonlab_control_server_close,
    moonlab_control_server_open, moonlab_control_server_run,
    moonlab_control_server_set_max_concurrent, moonlab_control_server_set_rate_limit,
    moonlab_control_server_set_request_timeout, moonlab_control_server_set_secret,
    moonlab_control_server_shutdown, moonlab_control_server_t,
    moonlab_control_submit_circuit_mtls, moonlab_control_submit_circuit_tls,
    moonlab_control_submit_health, moonlab_control_submit_metrics,
};
use std::ffi::CString;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::raw::c_char;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Compute `HMAC-SHA3-256(secret, msg)` via the moonlab C
/// implementation.  Used by `submit_circuit_auth` to construct the
/// `AUTH <token>` prelude.  Exposed publicly so callers handling the
/// socket directly can use the same code path.
pub fn hmac_sha3_256(secret: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    unsafe {
        moonlab_control_hmac_sha3_256(
            secret.as_ptr(),
            secret.len(),
            msg.as_ptr(),
            msg.len(),
            out.as_mut_ptr(),
        );
    }
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Send a `moonlab-circuit v1` text payload to a control-plane server
/// and collect the probability vector back.
///
/// Wire format matches v0.8.7 / v0.8.8 byte-for-byte.
///
/// # Errors
///
/// Returns `QuantumError::Ffi` if the server responds with `ERR`, the
/// connection is closed prematurely, or the response framing is
/// malformed.
pub fn submit_circuit(host: &str, port: u16, circuit_text: &str) -> Result<Vec<f64>> {
    submit_circuit_full(host, port, circuit_text, None, None, Duration::from_secs(30))
}

/// Same as [`submit_circuit`] with an explicit socket timeout.
pub fn submit_circuit_with_timeout(
    host: &str,
    port: u16,
    circuit_text: &str,
    timeout: Duration,
) -> Result<Vec<f64>> {
    submit_circuit_full(host, port, circuit_text, None, None, timeout)
}

/// Same as [`submit_circuit`] but with an optional shared secret for
/// HMAC-SHA3-256 authentication (since v0.8.16).  When `secret` is
/// `None`, behaves identically to [`submit_circuit_with_timeout`].
pub fn submit_circuit_auth(
    host: &str,
    port: u16,
    circuit_text: &str,
    secret: Option<&[u8]>,
) -> Result<Vec<f64>> {
    submit_circuit_full(host, port, circuit_text, secret, None, Duration::from_secs(30))
}

/// Submit with a tenant identifier carried in the AUTH line (since
/// v1.0.3).  The wire prelude becomes `AUTH <tenant_id>:<hmac>\n`;
/// the server plumbs `tenant_id` through to its scheduler
/// completion hook so a private overlay can attribute the run to a
/// billing account.
///
/// `tenant_id` must be 1..=63 chars from `[A-Za-z0-9_.-]` and
/// `secret` must be `Some` (the server uses HMAC to authenticate
/// the request regardless of which tenant_id is claimed).  Passing
/// a `tenant_id` with `None` secret returns `QuantumError::Ffi`.
pub fn submit_circuit_auth_tenant(
    host: &str,
    port: u16,
    circuit_text: &str,
    secret: &[u8],
    tenant_id: &str,
) -> Result<Vec<f64>> {
    validate_tenant_id(tenant_id)?;
    submit_circuit_full(
        host, port, circuit_text,
        Some(secret), Some(tenant_id),
        Duration::from_secs(30),
    )
}

fn validate_tenant_id(tenant_id: &str) -> Result<()> {
    if tenant_id.is_empty() || tenant_id.len() > 63 {
        return Err(QuantumError::Ffi(format!(
            "tenant_id length {} out of range [1, 63]", tenant_id.len()
        )));
    }
    for c in tenant_id.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-';
        if !ok {
            return Err(QuantumError::Ffi(format!(
                "tenant_id contains illegal char {c:?}; allowed [A-Za-z0-9_.-]"
            )));
        }
    }
    Ok(())
}

fn submit_circuit_full(
    host: &str,
    port: u16,
    circuit_text: &str,
    secret: Option<&[u8]>,
    tenant_id: Option<&str>,
    timeout: Duration,
) -> Result<Vec<f64>> {
    /* Body identical to the legacy submit_circuit_with_timeout path
     * with an optional AUTH prelude when `secret` is Some. */
    let addr_str = format!("{host}:{port}");
    let mut addrs = addr_str
        .to_socket_addrs()
        .map_err(|e| QuantumError::Ffi(format!("resolve {addr_str}: {e}")))?;
    let addr: SocketAddr = addrs
        .next()
        .ok_or_else(|| QuantumError::Ffi(format!("no addresses for {addr_str}")))?;

    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| QuantumError::Ffi(format!("connect {addr}: {e}")))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| QuantumError::Ffi(format!("set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| QuantumError::Ffi(format!("set_write_timeout: {e}")))?;

    let bytes = circuit_text.as_bytes();
    let header = format!("CIRCUIT {}\n", bytes.len());
    if let Some(key) = secret {
        let tok = hmac_sha3_256(key, header.as_bytes());
        let auth = if let Some(tid) = tenant_id {
            format!("AUTH {}:{}\n", tid, hex_encode(&tok))
        } else {
            format!("AUTH {}\n", hex_encode(&tok))
        };
        stream
            .write_all(auth.as_bytes())
            .map_err(|e| QuantumError::Ffi(format!("send AUTH: {e}")))?;
    }
    stream
        .write_all(header.as_bytes())
        .map_err(|e| QuantumError::Ffi(format!("send header: {e}")))?;
    stream
        .write_all(bytes)
        .map_err(|e| QuantumError::Ffi(format!("send body: {e}")))?;

    // Read response header line ("OK <N>\n" or "ERR <code> <msg>\n").
    let mut reader = BufReader::new(stream);
    let mut resp_hdr = String::new();
    reader
        .read_line(&mut resp_hdr)
        .map_err(|e| QuantumError::Ffi(format!("recv header: {e}")))?;

    if let Some(rest) = resp_hdr.strip_prefix("OK ") {
        let num_str = rest.trim();
        let num: usize = num_str
            .parse()
            .map_err(|e| QuantumError::Ffi(format!("malformed OK header {resp_hdr:?}: {e}")))?;
        if num == 0 || num > (1usize << 30) {
            return Err(QuantumError::Ffi(format!(
                "implausible num_probs {num}"
            )));
        }
        let mut raw = vec![0u8; num * 8];
        reader
            .read_exact(&mut raw)
            .map_err(|e| QuantumError::Ffi(format!("recv body: {e}")))?;

        let mut probs = Vec::with_capacity(num);
        for chunk in raw.chunks_exact(8) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            probs.push(f64::from_le_bytes(buf));
        }
        Ok(probs)
    } else if resp_hdr.starts_with("ERR ") {
        Err(QuantumError::Ffi(format!(
            "server rejected: {}",
            resp_hdr.trim_end()
        )))
    } else {
        Err(QuantumError::Ffi(format!(
            "unrecognized response: {resp_hdr:?}"
        )))
    }
}

/// Submit a `moonlab-circuit v1` text payload and request
/// `num_shots` measurement samples instead of the probability vector.
/// Returns a vector of `u64` bitstrings -- bit 0 is qubit 0.
///
/// Since v0.8.12 (Rust binding) / v0.8.11 (C server).
pub fn submit_circuit_shots(
    host: &str,
    port: u16,
    circuit_text: &str,
    num_shots: i32,
) -> Result<Vec<u64>> {
    submit_circuit_shots_with_timeout(
        host, port, circuit_text, num_shots, Duration::from_secs(60))
}

/// Same as [`submit_circuit_shots`] but with explicit socket timeout.
pub fn submit_circuit_shots_with_timeout(
    host: &str,
    port: u16,
    circuit_text: &str,
    num_shots: i32,
    timeout: Duration,
) -> Result<Vec<u64>> {
    if num_shots <= 0 || num_shots > (1 << 20) {
        return Err(QuantumError::Ffi(format!(
            "num_shots {num_shots} out of range [1, 2^20]"
        )));
    }

    let addr_str = format!("{host}:{port}");
    let mut addrs = addr_str
        .to_socket_addrs()
        .map_err(|e| QuantumError::Ffi(format!("resolve {addr_str}: {e}")))?;
    let addr: SocketAddr = addrs
        .next()
        .ok_or_else(|| QuantumError::Ffi(format!("no addresses for {addr_str}")))?;

    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| QuantumError::Ffi(format!("connect {addr}: {e}")))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| QuantumError::Ffi(format!("set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| QuantumError::Ffi(format!("set_write_timeout: {e}")))?;

    let bytes = circuit_text.as_bytes();
    let header = format!("SHOTS {} {}\n", num_shots, bytes.len());
    stream
        .write_all(header.as_bytes())
        .map_err(|e| QuantumError::Ffi(format!("send header: {e}")))?;
    stream
        .write_all(bytes)
        .map_err(|e| QuantumError::Ffi(format!("send body: {e}")))?;

    let mut reader = BufReader::new(stream);
    let mut resp_hdr = String::new();
    reader
        .read_line(&mut resp_hdr)
        .map_err(|e| QuantumError::Ffi(format!("recv header: {e}")))?;

    if let Some(rest) = resp_hdr.strip_prefix("SAMPLES ") {
        let n: usize = rest
            .trim()
            .parse()
            .map_err(|e| QuantumError::Ffi(format!("malformed SAMPLES header {resp_hdr:?}: {e}")))?;
        if n == 0 || n > (1usize << 20) {
            return Err(QuantumError::Ffi(format!("implausible shots_back {n}")));
        }
        let mut raw = vec![0u8; n * 8];
        reader
            .read_exact(&mut raw)
            .map_err(|e| QuantumError::Ffi(format!("recv body: {e}")))?;

        let mut outs = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(8) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            outs.push(u64::from_le_bytes(buf));
        }
        Ok(outs)
    } else if resp_hdr.starts_with("ERR ") {
        Err(QuantumError::Ffi(format!("server rejected: {}", resp_hdr.trim_end())))
    } else {
        Err(QuantumError::Ffi(format!("unrecognized response: {resp_hdr:?}")))
    }
}

/// In-process control-plane server (since v0.8.14).
///
/// Owns the C lifecycle handle, runs `moonlab_control_server_run` on a
/// background thread, exposes the bound port, and shuts down + joins
/// cleanly via Drop.
///
/// ## Example
///
/// ```no_run
/// use moonlab::control_plane::{ControlPlaneServer, submit_circuit};
/// use moonlab::qgtl::{GateType, QgtlCircuit};
///
/// let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
/// let port = srv.port();
///
/// let mut c = QgtlCircuit::new(2).unwrap();
/// c.add_gate(GateType::H, 0, -1, &[]).unwrap();
/// c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
/// let probs = submit_circuit("127.0.0.1", port, &c.serialize().unwrap()).unwrap();
/// assert!((probs[0] - 0.5).abs() < 1e-9);
///
/// // srv drops here -- background thread is signalled and joined.
/// ```
pub struct ControlPlaneServer {
    handle: Arc<AtomicPtr<moonlab_control_server_t>>,
    port:   u16,
    runner: Option<JoinHandle<i32>>,
}

// SAFETY: the C lifecycle API is internally synchronized around the
// listen socket fd and self-pipe.  The opaque handle pointer is only
// dereferenced inside the C library, which is thread-safe for the
// shutdown/close path (the run loop and shutdown writes are
// independent fds).
unsafe impl Send for ControlPlaneServer {}
unsafe impl Sync for ControlPlaneServer {}

impl ControlPlaneServer {
    /// Open a listener on `host:port` and start the worker thread.
    /// Pass `port = 0` for an OS-chosen port (read back via `.port()`).
    pub fn open(host: &str, port: u16) -> Result<Self> {
        Self::open_with_max_iters(host, port, i32::MAX)
    }

    /// Same as `open` but caps the number of served connections.
    pub fn open_with_max_iters(host: &str, port: u16, max_iters: i32) -> Result<Self> {
        let host_c = CString::new(host)
            .map_err(|e| QuantumError::Ffi(format!("invalid host: {e}")))?;
        let mut handle_raw: *mut moonlab_control_server_t = std::ptr::null_mut();
        let mut bound_port: u16 = 0;

        let rc = unsafe {
            moonlab_control_server_open(
                host_c.as_ptr(),
                port,
                &mut handle_raw as *mut *mut moonlab_control_server_t,
                &mut bound_port as *mut u16,
            )
        };
        if rc != 0 || handle_raw.is_null() {
            return Err(QuantumError::Ffi(format!("server_open rc={rc}")));
        }

        let handle = Arc::new(AtomicPtr::new(handle_raw));
        let runner_handle = Arc::clone(&handle);
        let runner = std::thread::spawn(move || -> i32 {
            let ptr = runner_handle.load(Ordering::SeqCst);
            unsafe { moonlab_control_server_run(ptr, max_iters) }
        });

        Ok(Self {
            handle,
            port: bound_port,
            runner: Some(runner),
        })
    }

    /// Bound TCP port (OS-chosen when constructed with `port = 0`).
    pub fn port(&self) -> u16 { self.port }

    /// Signal the server to stop after the current in-flight request.
    /// Idempotent.  The Drop impl also calls this, so explicit usage
    /// is only required when you want to observe the join before drop.
    pub fn shutdown(&self) {
        let ptr = self.handle.load(Ordering::SeqCst);
        if !ptr.is_null() {
            unsafe { moonlab_control_server_shutdown(ptr); }
        }
    }
}

impl ControlPlaneServer {
    /// Configure a per-source-IP rate limit (since v0.8.22).
    /// `rate_rps = 0` disables.  Burst <= 0 defaults to 2 * rate_rps.
    pub fn set_rate_limit(&self, rate_rps: i32, burst: i32) -> Result<()> {
        let ptr = self.handle.load(Ordering::SeqCst);
        if ptr.is_null() {
            return Err(QuantumError::Ffi("server handle is null".into()));
        }
        let rc = unsafe { moonlab_control_server_set_rate_limit(ptr, rate_rps, burst) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("set_rate_limit rc={rc}")));
        }
        Ok(())
    }

    /// Set per-request socket timeout in seconds (since v0.8.27 /
    /// v0.8.26 server).  `0` disables (legacy default).
    pub fn set_request_timeout(&self, timeout_secs: i32) -> Result<()> {
        let ptr = self.handle.load(Ordering::SeqCst);
        if ptr.is_null() {
            return Err(QuantumError::Ffi("server handle is null".into()));
        }
        let rc = unsafe { moonlab_control_server_set_request_timeout(ptr, timeout_secs) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("set_request_timeout rc={rc}")));
        }
        Ok(())
    }

    /// Cap the number of concurrent in-flight requests (since v0.9.0).
    /// `0` disables.  Excess clients receive `ERR -409 server busy\n`.
    pub fn set_max_concurrent(&self, max_concurrent: i32) -> Result<()> {
        let ptr = self.handle.load(Ordering::SeqCst);
        if ptr.is_null() {
            return Err(QuantumError::Ffi("server handle is null".into()));
        }
        let rc = unsafe { moonlab_control_server_set_max_concurrent(ptr, max_concurrent) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("set_max_concurrent rc={rc}")));
        }
        Ok(())
    }
}

/// Scrape the v0.8.23 METRICS endpoint and return the Prometheus
/// text-format exposition body.  Since v0.8.24 (Rust binding).
pub fn submit_metrics(host: &str, port: u16) -> Result<String> {
    let host_c = CString::new(host)
        .map_err(|e| QuantumError::Ffi(format!("invalid host: {e}")))?;
    let mut text_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    let rc = unsafe {
        moonlab_control_submit_metrics(host_c.as_ptr(), port, &mut text_ptr as *mut _)
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("submit_metrics rc={rc}")));
    }
    if text_ptr.is_null() {
        return Err(QuantumError::Ffi("submit_metrics returned NULL body".into()));
    }
    let body = unsafe { std::ffi::CStr::from_ptr(text_ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { libc::free(text_ptr as *mut libc::c_void); }
    Ok(body)
}

/// HEALTH probe -- since v0.8.22 (Rust binding) / v0.8.21 (C server).
/// Returns `Ok(true)` if the server is alive, `Ok(false)` on rate-limit,
/// or `Err` on transport failure.
pub fn submit_health(host: &str, port: u16) -> Result<bool> {
    let host_c = CString::new(host)
        .map_err(|e| QuantumError::Ffi(format!("invalid host: {e}")))?;
    let rc = unsafe { moonlab_control_submit_health(host_c.as_ptr(), port) };
    match rc {
        0 => Ok(true),
        -408 => Ok(false), // MOONLAB_CONTROL_RATE_LIMITED
        _ => Err(QuantumError::Ffi(format!("submit_health rc={rc}"))),
    }
}

impl Drop for ControlPlaneServer {
    fn drop(&mut self) {
        self.shutdown();
        if let Some(jh) = self.runner.take() {
            let _ = jh.join();
        }
        let ptr = self.handle.swap(std::ptr::null_mut(), Ordering::SeqCst);
        if !ptr.is_null() {
            unsafe { moonlab_control_server_close(ptr); }
        }
    }
}

/// Submit a circuit over a TLS-wrapped TCP connection -- since v0.8.18.
///
/// FFI thunk over the C `moonlab_control_submit_circuit_tls`, which is
/// built into the library when `-DQSIM_ENABLE_TLS=ON`.  Returns
/// `QuantumError::Ffi` if the library was built without TLS.
///
/// # Parameters
///
/// - `ca_path`: PEM CA bundle pin against the server's cert.  Required
///   for production; pass `None` together with `insecure = true` for
///   development / self-signed certificates only.
/// - `insecure`: skip peer verification.  Tests / dev only.
/// - `secret`: optional HMAC-SHA3-256 shared secret.  Composes with
///   TLS -- the server must have both `use_tls` and `set_secret`
///   configured when both are provided.
/// mTLS client variant -- presents a client cert + key alongside the
/// optional server-CA pin.  Required when the server was configured
/// with `moonlab_control_server_require_client_cert`.  Since v0.8.20.
pub fn submit_circuit_mtls(
    host: &str,
    port: u16,
    circuit_text: &str,
    server_ca_path: Option<&str>,
    client_cert_path: &str,
    client_key_path: &str,
    insecure: bool,
    secret: Option<&[u8]>,
) -> Result<Vec<f64>> {
    let host_c = CString::new(host)
        .map_err(|e| QuantumError::Ffi(format!("invalid host: {e}")))?;
    let ca_c = match server_ca_path {
        Some(p) => Some(CString::new(p)
            .map_err(|e| QuantumError::Ffi(format!("invalid server_ca_path: {e}")))?),
        None => None,
    };
    let cert_c = CString::new(client_cert_path)
        .map_err(|e| QuantumError::Ffi(format!("invalid client_cert_path: {e}")))?;
    let key_c  = CString::new(client_key_path)
        .map_err(|e| QuantumError::Ffi(format!("invalid client_key_path: {e}")))?;
    let bytes = circuit_text.as_bytes();

    let mut probs_ptr: *mut f64 = std::ptr::null_mut();
    let mut num_probs: usize = 0;
    let (secret_ptr, secret_len) = match secret {
        Some(s) => (s.as_ptr(), s.len()),
        None    => (std::ptr::null(), 0usize),
    };

    let rc = unsafe {
        moonlab_control_submit_circuit_mtls(
            host_c.as_ptr(),
            port,
            ca_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null()),
            cert_c.as_ptr(),
            key_c.as_ptr(),
            if insecure { 1 } else { 0 },
            secret_ptr,
            secret_len,
            bytes.as_ptr() as *const c_char,
            bytes.len(),
            &mut probs_ptr as *mut *mut f64,
            &mut num_probs as *mut usize,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("submit_circuit_mtls rc={rc}")));
    }
    if probs_ptr.is_null() || num_probs == 0 {
        return Err(QuantumError::Ffi("submit_circuit_mtls returned no probs".into()));
    }
    let probs = unsafe {
        std::slice::from_raw_parts(probs_ptr, num_probs).to_vec()
    };
    unsafe { libc::free(probs_ptr as *mut libc::c_void); }
    Ok(probs)
}

pub fn submit_circuit_tls(
    host: &str,
    port: u16,
    circuit_text: &str,
    ca_path: Option<&str>,
    insecure: bool,
    secret: Option<&[u8]>,
) -> Result<Vec<f64>> {
    let host_c = CString::new(host)
        .map_err(|e| QuantumError::Ffi(format!("invalid host: {e}")))?;
    let ca_c = match ca_path {
        Some(p) => Some(CString::new(p)
            .map_err(|e| QuantumError::Ffi(format!("invalid ca_path: {e}")))?),
        None => None,
    };
    let bytes = circuit_text.as_bytes();

    let mut probs_ptr: *mut f64 = std::ptr::null_mut();
    let mut num_probs: usize = 0;
    let (secret_ptr, secret_len) = match secret {
        Some(s) => (s.as_ptr(), s.len()),
        None    => (std::ptr::null(), 0usize),
    };

    let rc = unsafe {
        moonlab_control_submit_circuit_tls(
            host_c.as_ptr(),
            port,
            ca_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null()),
            if insecure { 1 } else { 0 },
            secret_ptr,
            secret_len,
            bytes.as_ptr() as *const c_char,
            bytes.len(),
            &mut probs_ptr as *mut *mut f64,
            &mut num_probs as *mut usize,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("submit_circuit_tls rc={rc}")));
    }
    if probs_ptr.is_null() || num_probs == 0 {
        return Err(QuantumError::Ffi("submit_circuit_tls returned no probs".into()));
    }
    // Copy out + free the C buffer.
    let probs = unsafe {
        std::slice::from_raw_parts(probs_ptr, num_probs).to_vec()
    };
    unsafe { libc::free(probs_ptr as *mut libc::c_void); }
    Ok(probs)
}
