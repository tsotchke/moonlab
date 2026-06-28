//! TLS client e2e (v0.8.18): generate a self-signed cert via openssl
//! CLI, point the C server at it via use_tls(), submit a Bell circuit
//! through submit_circuit_tls() with insecure=true, verify the
//! probability vector.

use moonlab::control_plane::submit_circuit_tls;
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::{
    moonlab_control_server_close, moonlab_control_server_open,
    moonlab_control_server_run, moonlab_control_server_t,
    moonlab_control_server_use_tls,
};
use std::ffi::CString;
use std::process::Command;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn bell_circuit_round_trips_over_tls() {
    /* Per-test cert paths.  Multiple TLS tests run in parallel
     * under `cargo test`; sharing /tmp/moonlab_rs_tls.* races on
     * openssl-generates-cert vs server-reads-cert and is the root
     * cause of the rust_bindings_smoke flake. */
    let pid = std::process::id();
    let cert_path = format!("/tmp/moonlab_rs_tls_{pid}.crt");
    let key_path  = format!("/tmp/moonlab_rs_tls_{pid}.key");

    // Self-signed cert via openssl CLI -- avoids pulling in the
    // openssl crate.  Always present on macOS / Linux dev environments.
    let status = Command::new("openssl")
        .args([
            "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", &key_path, "-out", &cert_path,
            "-days", "1", "-nodes",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ])
        .status()
        .expect("openssl CLI available");
    assert!(status.success());

    let host = CString::new("127.0.0.1").unwrap();
    let cert_c = CString::new(cert_path.clone()).unwrap();
    let key_c  = CString::new(key_path.clone()).unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));

    let runner = {
        let bp = Arc::clone(&bind_port);
        thread::spawn(move || -> i32 {
            let mut handle: *mut moonlab_control_server_t = std::ptr::null_mut();
            let mut port_out: u16 = 0;
            let rc = unsafe {
                moonlab_control_server_open(
                    host.as_ptr(),
                    0,
                    &mut handle as *mut _,
                    &mut port_out as *mut u16,
                )
            };
            if rc != 0 { return rc; }
            let tls_rc = unsafe {
                moonlab_control_server_use_tls(handle, cert_c.as_ptr(), key_c.as_ptr())
            };
            if tls_rc != 0 {
                unsafe { moonlab_control_server_close(handle); }
                return tls_rc;
            }
            bp.store(port_out, Ordering::SeqCst);
            let run_rc = unsafe { moonlab_control_server_run(handle, 1) };
            unsafe { moonlab_control_server_close(handle); }
            run_rc
        })
    };

    let deadline = Instant::now() + Duration::from_secs(3);
    let port = loop {
        let p = bind_port.load(Ordering::SeqCst);
        if p != 0 { break p; }
        if Instant::now() >= deadline { panic!("server bind timeout"); }
        thread::sleep(Duration::from_millis(20));
    };

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    let probs = submit_circuit_tls("127.0.0.1", port, &text, None, true, None)
        .expect("TLS submission");
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!( probs[1].abs() < 1e-9);
    assert!( probs[2].abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);

    let rc = runner.join().unwrap();
    assert_eq!(rc, 0);

    let _ = std::fs::remove_file(&cert_path);
    let _ = std::fs::remove_file(&key_path);
}
