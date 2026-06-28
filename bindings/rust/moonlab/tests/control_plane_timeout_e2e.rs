//! Per-request timeout binding parity (v0.8.27).  Sets the timeout
//! via the wrapper method and confirms a normal Bell still works.

use moonlab::control_plane::{submit_circuit, ControlPlaneServer};
use moonlab::qgtl::{GateType, QgtlCircuit};

#[test]
fn set_request_timeout_via_wrapper() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_request_timeout(2).expect("set_request_timeout");

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    let probs = submit_circuit("127.0.0.1", srv.port(), &text).expect("submit");
    assert!((probs[0] - 0.5).abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9);
    drop(srv);
}
