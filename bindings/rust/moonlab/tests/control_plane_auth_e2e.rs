//! HMAC-SHA3-256 auth e2e (v0.8.16): Rust client authenticates against
//! the C server which has been configured with a shared secret.

use moonlab::control_plane::{submit_circuit, submit_circuit_auth};
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::{
    moonlab_control_server_close, moonlab_control_server_open,
    moonlab_control_server_run, moonlab_control_server_set_secret,
    moonlab_control_server_t,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const SECRET: &[u8] = b"moonlab-shared-2026";

#[test]
fn auth_client_round_trip() {
    let host = CString::new("127.0.0.1").unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));

    // Server thread owns the handle entirely.  Three accept budget:
    // matching / wrong / unauth requests fill the quota and the run
    // returns naturally.
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

            // Install the secret before publishing the port.
            unsafe {
                moonlab_control_server_set_secret(
                    handle, SECRET.as_ptr(), SECRET.len());
            }
            bp.store(port_out, Ordering::SeqCst);

            let run_rc = unsafe { moonlab_control_server_run(handle, 3) };
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

    // 1) Matching secret -> OK Bell.
    let probs = submit_circuit_auth("127.0.0.1", port, &text, Some(SECRET))
        .expect("auth submission");
    assert!((probs[0] - 0.5).abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9);

    // 2) Wrong secret -> Err.
    let bad = submit_circuit_auth("127.0.0.1", port, &text, Some(b"wrong"));
    assert!(bad.is_err(), "wrong secret should be rejected, got {bad:?}");

    // 3) No secret -> Err.
    let unauth = submit_circuit("127.0.0.1", port, &text);
    assert!(unauth.is_err(), "missing AUTH should be rejected, got {unauth:?}");

    let rc = runner.join().unwrap();
    assert_eq!(rc, 0);
}
