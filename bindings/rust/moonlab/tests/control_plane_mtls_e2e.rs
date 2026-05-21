//! mTLS e2e (v0.8.20): generate a CA + server-cert + client-cert via
//! openssl CLI, configure the server with use_tls + require_client_cert,
//! drive a Bell circuit through submit_circuit_mtls and verify the
//! signature.  Also confirms a no-client-cert TLS attempt is rejected.

use moonlab::control_plane::{submit_circuit_mtls, submit_circuit_tls};
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::{
    moonlab_control_server_close, moonlab_control_server_open,
    moonlab_control_server_require_client_cert, moonlab_control_server_run,
    moonlab_control_server_t, moonlab_control_server_use_tls,
};
use std::ffi::CString;
use std::process::Command;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn run(args: &[&str]) {
    let st = Command::new(args[0]).args(&args[1..]).status().expect("spawn");
    assert!(st.success(), "{args:?} failed");
}

#[test]
fn bell_round_trips_over_mtls() {
    /* Per-process dir so parallel cargo test runs don't race on
     * cert generation. */
    let pid = std::process::id();
    let dir = format!("/tmp/moonlab_rs_mtls_{pid}");
    let _ = std::fs::create_dir_all(&dir);
    let ca_crt  = format!("{dir}/ca.crt");
    let ca_key  = format!("{dir}/ca.key");
    let srv_crt = format!("{dir}/server.crt");
    let srv_key = format!("{dir}/server.key");
    let cli_crt = format!("{dir}/client.crt");
    let cli_key = format!("{dir}/client.key");
    let srv_csr = format!("{dir}/server.csr");
    let cli_csr = format!("{dir}/client.csr");

    // Generate CA.
    run(&["openssl", "req", "-x509", "-newkey", "rsa:2048",
         "-keyout", &ca_key, "-out", &ca_crt,
         "-days", "1", "-nodes", "-subj", "/CN=moonlab-rs-ca"]);

    // Server cert signed by CA.
    run(&["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", &srv_key, "-out", &srv_csr,
         "-subj", "/CN=127.0.0.1"]);
    run(&["openssl", "x509", "-req", "-in", &srv_csr,
         "-CA", &ca_crt, "-CAkey", &ca_key, "-CAcreateserial",
         "-out", &srv_crt, "-days", "1"]);

    // Client cert signed by same CA.
    run(&["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", &cli_key, "-out", &cli_csr,
         "-subj", "/CN=moonlab-rs-client"]);
    run(&["openssl", "x509", "-req", "-in", &cli_csr,
         "-CA", &ca_crt, "-CAkey", &ca_key, "-CAcreateserial",
         "-out", &cli_crt, "-days", "1"]);

    let host = CString::new("127.0.0.1").unwrap();
    let ca_crt_c  = CString::new(ca_crt.clone()).unwrap();
    let srv_crt_c = CString::new(srv_crt).unwrap();
    let srv_key_c = CString::new(srv_key).unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));

    let runner = {
        let bp = Arc::clone(&bind_port);
        thread::spawn(move || -> i32 {
            let mut handle: *mut moonlab_control_server_t = std::ptr::null_mut();
            let mut port_out: u16 = 0;
            let rc = unsafe {
                moonlab_control_server_open(
                    host.as_ptr(), 0,
                    &mut handle as *mut _,
                    &mut port_out as *mut u16,
                )
            };
            if rc != 0 { return rc; }
            let rc_tls = unsafe {
                moonlab_control_server_use_tls(handle, srv_crt_c.as_ptr(), srv_key_c.as_ptr())
            };
            if rc_tls != 0 {
                unsafe { moonlab_control_server_close(handle); }
                return rc_tls;
            }
            let rc_req = unsafe {
                moonlab_control_server_require_client_cert(handle, ca_crt_c.as_ptr())
            };
            if rc_req != 0 {
                unsafe { moonlab_control_server_close(handle); }
                return rc_req;
            }
            bp.store(port_out, Ordering::SeqCst);
            let run_rc = unsafe { moonlab_control_server_run(handle, 2) };
            unsafe { moonlab_control_server_close(handle); }
            run_rc
        })
    };

    let deadline = Instant::now() + Duration::from_secs(3);
    let port = loop {
        let p = bind_port.load(Ordering::SeqCst);
        if p != 0 { break p; }
        if Instant::now() >= deadline { panic!("bind timeout"); }
        thread::sleep(Duration::from_millis(20));
    };

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    // Path 1: client presents CA-signed cert.
    let probs = submit_circuit_mtls(
        "127.0.0.1", port, &text,
        Some(&ca_crt),
        &cli_crt, &cli_key,
        true,            // insecure=true: self-signed CA, skip hostname pin
        None,
    ).expect("mTLS submission");
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);

    // Path 2: client without cert -> rejected.
    let unauth = submit_circuit_tls(
        "127.0.0.1", port, &text, Some(&ca_crt), true, None);
    assert!(unauth.is_err(), "no-cert client should be rejected, got {unauth:?}");

    let rc = runner.join().unwrap();
    assert_eq!(rc, 0);
    let _ = std::fs::remove_dir_all(dir);
}
