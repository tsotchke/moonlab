//! Concurrency-ceiling binding parity (v0.9.0).  Confirms the
//! wrapper method drives the new C API and that the cap is observed
//! end-to-end by scraping METRICS.

use moonlab::control_plane::{submit_circuit, submit_metrics, ControlPlaneServer};
use moonlab::qgtl::{GateType, QgtlCircuit};
use std::sync::mpsc;
use std::thread;

#[test]
fn set_max_concurrent_via_wrapper() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_max_concurrent(4).expect("set_max_concurrent");
    drop(srv);
}

#[test]
fn cap_rejects_some_of_six_parallel() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_max_concurrent(2).expect("set_max_concurrent");
    let port = srv.port();

    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    let text = c.serialize().unwrap();

    let (tx, rx) = mpsc::channel();
    for _ in 0..6 {
        let tx = tx.clone();
        let text = text.clone();
        thread::spawn(move || {
            let r = submit_circuit("127.0.0.1", port, &text);
            tx.send(r.is_ok()).unwrap();
        });
    }
    let mut ok = 0;
    let mut denied = 0;
    for _ in 0..6 {
        if rx.recv().unwrap() { ok += 1; } else { denied += 1; }
    }
    assert_eq!(ok + denied, 6);
    assert!(denied >= 1, "cap=2 with 6 parallel must reject some");

    let body = submit_metrics("127.0.0.1", port).expect("METRICS scrape");
    let counter_line = body
        .lines()
        .find(|line| line.starts_with("moonlab_control_max_concurrent_rejected_total "))
        .expect("counter present");
    let value: u64 = counter_line
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap();
    assert!(value >= denied as u64, "counter={value}, denied={denied}");
    drop(srv);
}
