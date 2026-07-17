//! Safe-wrapper parity for the server-side security configuration
//! calls: `ControlPlaneServer::set_secret` / `use_tls` /
//! `require_client_cert`.  These wrap `moonlab_control_server_set_secret`,
//! `moonlab_control_server_use_tls`, and
//! `moonlab_control_server_require_client_cert` respectively -- prior to
//! these wrappers existing, the only way to reach those three C entry
//! points from Rust was the raw `moonlab_sys` FFI (see
//! `control_plane_auth_e2e.rs`, `control_plane_tls_e2e.rs`, and
//! `control_plane_mtls_e2e.rs`, which configure security by calling
//! `moonlab_sys::moonlab_control_server_*` directly on the server
//! thread). This file exercises the same C behavior end-to-end but
//! entirely through the safe `ControlPlaneServer` API.

use moonlab::control_plane::{
    submit_circuit, submit_circuit_auth, submit_circuit_mtls, submit_circuit_tls,
    ControlPlaneServer,
};
use moonlab::qgtl::{GateType, QgtlCircuit};
use std::process::Command;

const SECRET: &[u8] = b"moonlab-safe-wrapper-secret-2026";

fn bell_circuit_text() -> String {
    let mut c = QgtlCircuit::new(2).unwrap();
    c.add_gate(GateType::H, 0, -1, &[]).unwrap();
    c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
    c.serialize().unwrap()
}

fn run(args: &[&str]) {
    let st = Command::new(args[0]).args(&args[1..]).status().expect("spawn openssl");
    assert!(st.success(), "{args:?} failed");
}

#[test]
fn set_secret_via_wrapper() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_secret(SECRET).expect("set_secret");
    drop(srv);
}

#[test]
fn set_secret_round_trip() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_secret(SECRET).expect("set_secret");
    let port = srv.port();
    let text = bell_circuit_text();

    // Matching secret -> OK Bell state.
    let probs = submit_circuit_auth("127.0.0.1", port, &text, Some(SECRET))
        .expect("auth submission with matching secret");
    assert!((probs[0] - 0.5).abs() < 1e-9);
    assert!((probs[3] - 0.5).abs() < 1e-9);

    // Wrong secret -> rejected.
    let bad = submit_circuit_auth("127.0.0.1", port, &text, Some(b"wrong-secret"));
    assert!(bad.is_err(), "wrong secret should be rejected, got {bad:?}");

    // No AUTH line at all -> rejected.
    let unauth = submit_circuit("127.0.0.1", port, &text);
    assert!(unauth.is_err(), "missing AUTH should be rejected, got {unauth:?}");

    drop(srv);
}

#[test]
fn use_tls_round_trip() {
    let pid = std::process::id();
    let cert_path = format!("/tmp/moonlab_rs_safe_tls_{pid}.crt");
    let key_path = format!("/tmp/moonlab_rs_safe_tls_{pid}.key");

    run(&[
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", &key_path, "-out", &cert_path,
        "-days", "1", "-nodes",
        "-subj", "/CN=127.0.0.1",
        "-addext", "subjectAltName=IP:127.0.0.1",
    ]);

    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.use_tls(&cert_path, &key_path).expect("use_tls");
    let port = srv.port();
    let text = bell_circuit_text();

    let probs = submit_circuit_tls("127.0.0.1", port, &text, None, true, None)
        .expect("TLS submission");
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);

    drop(srv);
    let _ = std::fs::remove_file(&cert_path);
    let _ = std::fs::remove_file(&key_path);
}

#[test]
fn require_client_cert_round_trip() {
    let pid = std::process::id();
    let dir = format!("/tmp/moonlab_rs_safe_mtls_{pid}");
    let _ = std::fs::create_dir_all(&dir);
    let ca_crt = format!("{dir}/ca.crt");
    let ca_key = format!("{dir}/ca.key");
    let srv_crt = format!("{dir}/server.crt");
    let srv_key = format!("{dir}/server.key");
    let cli_crt = format!("{dir}/client.crt");
    let cli_key = format!("{dir}/client.key");
    let srv_csr = format!("{dir}/server.csr");
    let cli_csr = format!("{dir}/client.csr");

    run(&["openssl", "req", "-x509", "-newkey", "rsa:2048",
         "-keyout", &ca_key, "-out", &ca_crt,
         "-days", "1", "-nodes", "-subj", "/CN=moonlab-rs-safe-ca"]);

    run(&["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", &srv_key, "-out", &srv_csr,
         "-subj", "/CN=127.0.0.1"]);
    run(&["openssl", "x509", "-req", "-in", &srv_csr,
         "-CA", &ca_crt, "-CAkey", &ca_key, "-CAcreateserial",
         "-out", &srv_crt, "-days", "1"]);

    run(&["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", &cli_key, "-out", &cli_csr,
         "-subj", "/CN=moonlab-rs-safe-client"]);
    run(&["openssl", "x509", "-req", "-in", &cli_csr,
         "-CA", &ca_crt, "-CAkey", &ca_key, "-CAcreateserial",
         "-out", &cli_crt, "-days", "1"]);

    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.use_tls(&srv_crt, &srv_key).expect("use_tls");
    srv.require_client_cert(&ca_crt).expect("require_client_cert");
    let port = srv.port();
    let text = bell_circuit_text();

    // Client presents a CA-signed cert -> accepted.
    let probs = submit_circuit_mtls(
        "127.0.0.1", port, &text,
        Some(&ca_crt),
        &cli_crt, &cli_key,
        true, // insecure=true: self-signed CA, skip hostname pin
        None,
    ).expect("mTLS submission");
    assert!((probs[0] - 0.5).abs() < 1e-9, "P[00] = {}", probs[0]);
    assert!((probs[3] - 0.5).abs() < 1e-9, "P[11] = {}", probs[3]);

    // Client without a cert -> rejected.
    let unauth = submit_circuit_tls("127.0.0.1", port, &text, Some(&ca_crt), true, None);
    assert!(unauth.is_err(), "no-cert client should be rejected, got {unauth:?}");

    drop(srv);
    let _ = std::fs::remove_dir_all(dir);
}
