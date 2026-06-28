//! End-to-end: spawn the C control-plane server in a worker thread,
//! drive `submit_circuit` from the Rust client, verify Bell signature.

use moonlab::control_plane::submit_circuit;
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::moonlab_control_serve;
use std::ffi::CString;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn bell_circuit_round_trips_over_tcp() {
    let host = CString::new("127.0.0.1").unwrap();

    // The C control-plane writes the OS-chosen port through the
    // `out_port` parameter after bind() but before accept().  Pass an
    // AtomicU16's underlying storage as `*mut u16` so the main thread
    // can observe the port while the server thread is still blocked
    // in accept().  AtomicU16 is layout-compatible with u16 on every
    // arch moonlab targets (x86_64, aarch64).
    let bind_port = Arc::new(AtomicU16::new(0));

    let server = {
        let bp = Arc::clone(&bind_port);
        thread::spawn(move || unsafe {
            let port_ptr = bp.as_ref() as *const AtomicU16 as *mut u16;
            moonlab_control_serve(host.as_ptr(), 0, 2, port_ptr)
        })
    };

    let deadline = Instant::now() + Duration::from_secs(3);
    let port = loop {
        let p = bind_port.load(Ordering::SeqCst);
        if p != 0 { break p; }
        if Instant::now() >= deadline {
            panic!("server failed to bind within 3 s");
        }
        thread::sleep(Duration::from_millis(20));
    };

    // ---- Round 1: Bell over TCP ----
    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    let probs = submit_circuit("127.0.0.1", port, &text)
        .expect("submit_circuit returned Err");
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!( probs[1].abs() < 1e-9, "P[01] = {}", probs[1]);
    assert!( probs[2].abs() < 1e-9, "P[10] = {}", probs[2]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);

    // ---- Round 2: garbage rejected ----
    let result = submit_circuit("127.0.0.1", port, "this is not a circuit\n");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("server rejected"));

    let rc = server.join().unwrap();
    assert_eq!(rc, 0);
}
