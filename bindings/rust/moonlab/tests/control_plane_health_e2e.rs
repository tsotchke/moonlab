//! HEALTH + rate-limit binding e2e (v0.8.22).

use moonlab::control_plane::{submit_health, ControlPlaneServer};

#[test]
fn health_probe() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    let alive = submit_health("127.0.0.1", srv.port()).expect("HEALTH");
    assert!(alive, "server should answer HEALTH alive");
    drop(srv);
}

#[test]
fn rate_limit_via_wrapper() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    srv.set_rate_limit(3, 3).expect("set_rate_limit");

    let mut ok_count   = 0;
    let mut lim_count  = 0;
    for _ in 0..8 {
        match submit_health("127.0.0.1", srv.port()) {
            Ok(true)  => ok_count  += 1,
            Ok(false) => lim_count += 1,
            Err(_)    => { /* transport noise; ignore */ }
        }
    }
    assert!(ok_count  >= 3, "burst should let through 3 (got {ok_count})");
    assert!(lim_count > 0,  "rate limiter should kick in (got {lim_count})");
    drop(srv);
}
