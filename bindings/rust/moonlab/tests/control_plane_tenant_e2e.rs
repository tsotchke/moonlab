//! Tenant-form AUTH e2e (v1.0.3): Rust client sends
//! `AUTH <tenant_id>:<hmac>` against a C server that has the
//! HMAC secret installed.  Mirrors:
//!   - bindings/python/tests/test_control_plane.py
//!     (test_tenant_form_auth_round_trip, etc.)
//!   - tests/integration/test_control_plane_tenant.c (C side)

use moonlab::control_plane::submit_circuit_auth_tenant;
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

const SECRET: &[u8] = b"rust-tenant-smoke-2026";

fn bell_text() -> String {
    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    c.serialize().unwrap()
}

fn boot_server(max_accepts: i32) -> (Arc<AtomicU16>, thread::JoinHandle<i32>) {
    let host = CString::new("127.0.0.1").unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));
    let bp = Arc::clone(&bind_port);
    let runner = thread::spawn(move || -> i32 {
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
        unsafe {
            moonlab_control_server_set_secret(
                handle, SECRET.as_ptr(), SECRET.len());
        }
        bp.store(port_out, Ordering::SeqCst);
        let run_rc = unsafe { moonlab_control_server_run(handle, max_accepts) };
        unsafe { moonlab_control_server_close(handle); }
        run_rc
    });
    (bind_port, runner)
}

fn wait_for_port(bind_port: &AtomicU16) -> u16 {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let p = bind_port.load(Ordering::SeqCst);
        if p != 0 { return p; }
        if Instant::now() >= deadline { panic!("bind timeout"); }
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn tenant_round_trip() {
    let (bind_port, runner) = boot_server(1);
    let port = wait_for_port(&bind_port);

    let text = bell_text();
    let probs = submit_circuit_auth_tenant(
        "127.0.0.1", port, &text, SECRET, "acme-corp",
    ).expect("AUTH acme-corp:hex bell submit");
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);
    assert!(probs[1] < 1e-9 && probs[2] < 1e-9);

    runner.join().unwrap();
}

#[test]
fn three_tenants_in_sequence() {
    let (bind_port, runner) = boot_server(3);
    let port = wait_for_port(&bind_port);

    let text = bell_text();
    for tenant in ["acme-corp", "beta-startup", "gamma.industries"] {
        let probs = submit_circuit_auth_tenant(
            "127.0.0.1", port, &text, SECRET, tenant,
        ).unwrap_or_else(|e| panic!("submit tenant={tenant}: {e}"));
        assert!((probs[0] - 0.5).abs() < 1e-9, "{tenant} P[00]={}", probs[0]);
        assert!((probs[3] - 0.5).abs() < 1e-9, "{tenant} P[11]={}", probs[3]);
    }
    runner.join().unwrap();
}

#[test]
fn illegal_char_tenant_id_rejected_clientside() {
    // No server boot needed; the helper validates before connecting.
    let text = bell_text();
    let rc = submit_circuit_auth_tenant(
        "127.0.0.1", 1, &text, SECRET, "acme;rm -rf /",
    );
    assert!(rc.is_err(),
        "illegal-char tenant_id should be rejected client-side; got {:?}",
        rc.map(|p| p.len()));
}

#[test]
fn empty_tenant_id_rejected_clientside() {
    let text = bell_text();
    let rc = submit_circuit_auth_tenant(
        "127.0.0.1", 1, &text, SECRET, "",
    );
    assert!(rc.is_err(), "empty tenant_id should be rejected");
}

#[test]
fn oversize_tenant_id_rejected_clientside() {
    let text = bell_text();
    let big = "x".repeat(64);
    let rc = submit_circuit_auth_tenant(
        "127.0.0.1", 1, &text, SECRET, &big,
    );
    assert!(rc.is_err(), "64-char tenant_id should be rejected (max 63)");
}
