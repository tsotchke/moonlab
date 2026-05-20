//! Shots-mode end-to-end test for the Rust control-plane client.
//! Server-thread spawned in-process, 1024 Bell samples requested,
//! every outcome must be 0 or 3 (the Bell support).

use moonlab::control_plane::submit_circuit_shots;
use moonlab::qgtl::{GateType, QgtlCircuit};
use moonlab_sys::moonlab_control_serve;
use std::ffi::CString;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn bell_shots_round_trip() {
    let host = CString::new("127.0.0.1").unwrap();
    let bind_port = Arc::new(AtomicU16::new(0));

    let server = {
        let bp = Arc::clone(&bind_port);
        thread::spawn(move || unsafe {
            let port_ptr = bp.as_ref() as *const AtomicU16 as *mut u16;
            moonlab_control_serve(host.as_ptr(), 0, 1, port_ptr)
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

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    const N_SHOTS: i32 = 1024;
    let outcomes = submit_circuit_shots("127.0.0.1", port, &text, N_SHOTS)
        .expect("shots submission failed");
    assert_eq!(outcomes.len(), N_SHOTS as usize);

    let mut n_bell = 0usize;
    let mut n_off  = 0usize;
    for o in &outcomes {
        if *o == 0 || *o == 3 { n_bell += 1; } else { n_off += 1; }
    }
    assert_eq!(n_off, 0, "off-Bell outcomes: {n_off}");
    assert_eq!(n_bell, N_SHOTS as usize);

    let rc = server.join().unwrap();
    assert_eq!(rc, 0);
}
