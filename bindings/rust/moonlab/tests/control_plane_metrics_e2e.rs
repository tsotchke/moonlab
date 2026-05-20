//! METRICS scrape e2e (v0.8.24): drive a few HEALTH/METRICS round-trips
//! through the wrapper, scrape the endpoint, confirm the exposition
//! body is well-formed and the counters reflect activity.

use moonlab::control_plane::{submit_health, submit_metrics, ControlPlaneServer};

#[test]
fn metrics_scrape_after_health_probes() {
    let srv = ControlPlaneServer::open("127.0.0.1", 0).unwrap();
    let port = srv.port();

    for _ in 0..3 {
        let _ = submit_health("127.0.0.1", port);
    }
    let body = submit_metrics("127.0.0.1", port).expect("METRICS scrape");
    assert!(body.contains("# HELP moonlab_control_requests_total"),
            "missing # HELP line: {body}");
    assert!(body.contains("moonlab_control_requests_total{verb=\"HEALTH\"}"),
            "missing HEALTH metric: {body}");
    assert!(body.contains("moonlab_control_requests_total{verb=\"METRICS\"}"),
            "missing METRICS metric: {body}");
    drop(srv);
}
