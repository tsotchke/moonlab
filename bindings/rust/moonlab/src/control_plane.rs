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
    moonlab_control_server_close, moonlab_control_server_open,
    moonlab_control_server_run, moonlab_control_server_shutdown,
    moonlab_control_server_t,
};
use std::ffi::CString;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

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
    submit_circuit_with_timeout(host, port, circuit_text, Duration::from_secs(30))
}

/// Same as [`submit_circuit`] but with an explicit socket timeout.
pub fn submit_circuit_with_timeout(
    host: &str,
    port: u16,
    circuit_text: &str,
    timeout: Duration,
) -> Result<Vec<f64>> {
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
