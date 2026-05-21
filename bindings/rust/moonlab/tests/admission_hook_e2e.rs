//! Admission-hook e2e (v1.0.3): a Rust closure installed on a real
//! moonlab-control-server refuses one tenant_id and admits another.
//! Mirrors:
//!   - bindings/python/tests/test_control_plane.py
//!     test_admission_hook_refuses_acme_allows_beta
//!   - tests/integration/test_control_plane_tenant.c (C side)

use moonlab::admission_hook::{AdmissionDecision, AdmissionHook};
use moonlab::control_plane::submit_circuit_auth_tenant;
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::{
    moonlab_control_server_close, moonlab_control_server_open,
    moonlab_control_server_run, moonlab_control_server_set_secret,
    moonlab_control_server_t,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const SECRET: &[u8] = b"rust-admission-smoke-2026";

fn bell_text() -> String {
    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    c.serialize().unwrap()
}

#[test]
fn admission_hook_refuses_acme_allows_beta() {
    // Track which tenants the hook saw -- gives us a way to assert
    // the hook was actually wired into the server lifecycle.
    let fired = Arc::new(AtomicUsize::new(0));
    let hook = {
        let fired = Arc::clone(&fired);
        AdmissionHook::new(move |req| {
            fired.fetch_add(1, Ordering::SeqCst);
            if req.tenant_id() == Some("acme-corp") {
                AdmissionDecision::Refused(-405)
            } else {
                AdmissionDecision::Admitted
            }
        })
    };

    // Boot a server, install the hook, run 2 accepts.
    let host = CString::new("127.0.0.1").unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));
    let runner = {
        let bp = Arc::clone(&bind_port);
        let host = host.clone();
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
            unsafe {
                moonlab_control_server_set_secret(
                    handle, SECRET.as_ptr(), SECRET.len());
            }
            // Install the admission hook before the runner thread
            // starts accepting.
            unsafe { hook.install(handle).expect("install"); }
            bp.store(port_out, Ordering::SeqCst);
            // Two accepts: acme refusal, beta success.
            let run_rc = unsafe { moonlab_control_server_run(handle, 2) };
            unsafe { moonlab_control_server_close(handle); }
            // Hook drops here, after server close -> safe.
            drop(hook);
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

    let text = bell_text();

    // acme-corp -> refused
    let rc = submit_circuit_auth_tenant(
        "127.0.0.1", port, &text, SECRET, "acme-corp",
    );
    assert!(rc.is_err(),
        "acme-corp should be refused; got {:?}", rc.map(|p| p.len()));

    // beta-startup -> admitted, Bell distribution
    let probs = submit_circuit_auth_tenant(
        "127.0.0.1", port, &text, SECRET, "beta-startup",
    ).expect("beta-startup should be admitted");
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00]={}", probs[0]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11]={}", probs[3]);

    runner.join().unwrap();
    assert_eq!(fired.load(Ordering::SeqCst), 2,
        "hook should have fired twice (acme + beta)");
}

#[test]
fn admission_hook_panic_is_caught() {
    // A panicking hook is treated as MOONLAB_CONTROL_REJECTED so it
    // cannot tear down the server.
    let hook = AdmissionHook::new(|_req| {
        panic!("intentional fault inside admission hook");
    });

    let host = CString::new("127.0.0.1").unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));
    let runner = {
        let bp = Arc::clone(&bind_port);
        let host = host.clone();
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
            unsafe {
                moonlab_control_server_set_secret(
                    handle, SECRET.as_ptr(), SECRET.len());
            }
            unsafe { hook.install(handle).expect("install"); }
            bp.store(port_out, Ordering::SeqCst);
            let run_rc = unsafe { moonlab_control_server_run(handle, 1) };
            unsafe { moonlab_control_server_close(handle); }
            drop(hook);
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

    let text = bell_text();
    let rc = submit_circuit_auth_tenant(
        "127.0.0.1", port, &text, SECRET, "alpha",
    );
    assert!(rc.is_err(),
        "panicking hook should refuse the request; got {:?}",
        rc.map(|p| p.len()));

    runner.join().unwrap();
}
