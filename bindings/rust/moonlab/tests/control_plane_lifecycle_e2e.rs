//! Lifecycle wrapper e2e (v0.8.14): `ControlPlaneServer` spawns the
//! C server on a worker thread, submits a Bell circuit via the client,
//! drops cleanly via Drop.

use moonlab::control_plane::{submit_circuit, ControlPlaneServer};
use moonlab::qgtl::{GateType, QgtlCircuit};

#[test]
fn lifecycle_wrapper_round_trip() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).expect("open");
    let port = srv.port();
    assert!(port != 0, "expected OS-chosen port");

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    let probs = submit_circuit("127.0.0.1", port, &text).expect("submit");
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.5).abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9);

    // Drop drives shutdown + join + close.
    drop(srv);
}

#[test]
fn lifecycle_shutdown_signaled_externally() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).expect("open");
    let _port = srv.port();
    srv.shutdown();
    drop(srv); // join should complete promptly.
}
