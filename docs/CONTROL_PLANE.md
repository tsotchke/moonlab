# MoonLab control plane -- production deployment guide

> Audience: operators standing up a MoonLab compute node behind a load
> balancer, in Kubernetes, or as a long-lived simulator daemon.
> This document tracks the surface as of v0.9.0.

## 1. What the control plane is

A TCP server that accepts moonlab-circuit v1 text payloads and
returns simulation results.  Embeds the state-vector / MPS / CA-MPS /
MPDO backends; speaks one line-delimited request format with a small
RPC vocabulary; supports plain TCP, TLS, and mutual TLS.

```
       client app                          moonlab control plane
       ----------                          ---------------------
                                          +----------------------+
   moonlab_control_submit_*  --tcp-->     | accept loop          |
                                          |   rate-limit (-408)  |
                                          |   max-concurrent (-409)
                                          |   tls handshake      |
                                          +----------------------+
                                                |
                                                v
                                          +----------------------+
                                          | per-connection       |
                                          | worker thread        |
                                          |   AUTH (HMAC-SHA3)   |
                                          |   verb dispatch      |
                                          |   simulator backend  |
                                          |   reply              |
                                          +----------------------+
```

## 2. Wire protocol

Every connection is a sequence of length-prefixed line-delimited
messages.  Each verb is one round trip on its own connection.

### 2.1 Common header

The first line of every request from a client is

    <VERB>[ <args>]\n

where `<VERB>` is one of `CIRCUIT`, `SHOTS`, `HEALTH`, `METRICS`,
`AUTH`, or `CIRCUIT_AUTH`.

Server responses begin with one of:

    OK <n>\n        -- success; <n> body bytes follow (binary)
    OK\n            -- success; no body
    ERR <code> <msg>\n

### 2.2 Status codes

| Code | Name                              | Meaning                                |
|------|-----------------------------------|----------------------------------------|
|    0 | `MOONLAB_CONTROL_OK`              | success                                |
| -400 | `MOONLAB_CONTROL_BAD_ARG`         | malformed request                      |
| -401 | `MOONLAB_CONTROL_AUTH_REQUIRED`   | server requires HMAC; client did not   |
| -402 | `MOONLAB_CONTROL_AUTH_BAD`        | HMAC verification failed               |
| -403 | `MOONLAB_CONTROL_IO_ERROR`        | socket-level failure                   |
| -405 | `MOONLAB_CONTROL_REJECTED`        | server rejected the payload            |
| -407 | `MOONLAB_CONTROL_OOM`             | out of memory                          |
| -408 | `MOONLAB_CONTROL_RATE_LIMITED`    | per-IP token bucket refused            |
| -409 | `MOONLAB_CONTROL_SERVER_BUSY`     | concurrent-connection cap hit          |

### 2.3 Verbs

```
CIRCUIT <body_len>\n               -- run circuit; body = moonlab-circuit v1 text
<body bytes>
            <- OK <n>\n<binary probabilities, n = 2^q * 8 bytes>

SHOTS <shots> <body_len>\n         -- sample shots from circuit
<body bytes>
            <- OK <n>\n<binary outcome counts>

HEALTH\n                           -- liveness ping
            <- OK\n

METRICS\n                          -- Prometheus text exposition
            <- METRICS <n>\n<n bytes of metrics text>

AUTH <hexdigest>\n                 -- HMAC-SHA3-256 of next request body
                                      under the shared secret

CIRCUIT_AUTH <body_len> <hexdigest>\n
<body bytes>
```

The `CIRCUIT_AUTH` form is a single-shot variant that bundles auth +
payload; both forms compute the digest over the raw body bytes.

## 3. C API surface

```c
#include <moonlab/control/control_plane.h>

moonlab_control_server_t *srv;
uint16_t port;
moonlab_control_server_open("0.0.0.0", 7070, &srv, &port);

/* Auth (optional but recommended). */
const uint8_t secret[] = {...};
moonlab_control_server_set_secret(srv, secret, sizeof secret);

/* TLS / mTLS (optional). */
moonlab_control_server_use_tls(srv, "cert.pem", "key.pem");
moonlab_control_server_require_client_cert(srv, "ca.pem");

/* Defensive caps. */
moonlab_control_server_set_rate_limit(srv,
    /*tokens_per_sec=*/  10,
    /*burst_capacity=*/ 20);
moonlab_control_server_set_request_timeout(srv, /*secs=*/ 30);
moonlab_control_server_set_max_concurrent(srv,  /*workers=*/ 64);

/* Serve N requests, or moonlab_control_server_shutdown() to exit. */
moonlab_control_server_run(srv, /*max_iters=*/ INT_MAX);

moonlab_control_server_close(srv);
```

Client-side helpers `moonlab_control_submit_circuit`, `_shots`,
`_health`, `_metrics`, `_auth`, `_tls`, and `_mtls` all return a
`MOONLAB_CONTROL_*` status code.

## 4. Defensive layers

Each request flows through these checks in order.  Failing any one
short-circuits the request before it reaches a worker thread.

| Layer                  | Reject code | Configured by                                    | Metric                                            |
|------------------------|-------------|--------------------------------------------------|---------------------------------------------------|
| TLS handshake          | --          | `set_use_tls` / `require_client_cert`            | `moonlab_control_tls_handshake_failed_total`      |
| Per-IP rate limit      | -408        | `set_rate_limit(rps, burst)`                     | `moonlab_control_rate_limited_total`              |
| Concurrency ceiling    | -409        | `set_max_concurrent(N)`                          | `moonlab_control_max_concurrent_rejected_total`   |
| Per-request timeout    | -403        | `set_request_timeout(secs)`                      | (returns -403 IO_ERROR)                           |
| HMAC authentication    | -401/-402   | `set_secret(buf, len)`                           | (returns -401/-402)                               |

### 4.1 TLS handshake counter

`moonlab_control_tls_handshake_failed_total` tracks every connection
that reached `SSL_accept` but failed -- bad client cert, protocol
mismatch, plain TCP to a TLS-only port, etc.  A non-zero rate after
provisioning is a sign of misconfigured clients or a stale CA bundle.

### 4.2 Concurrency ceiling

When `max_concurrent > 0`, the accept loop tracks an `_Atomic int
active_workers`.  If a new connection arrives while
`active_workers >= max_concurrent`, the server writes
`ERR -409 server busy\n` to the socket, closes it, and bumps
`moonlab_control_max_concurrent_rejected_total`.

The ceiling is *connections-in-flight*, not *requests-per-second*.
Pair with `set_request_timeout` so a single misbehaving client can't
hold a slot indefinitely.

### 4.3 mTLS peer audit log

When `require_client_cert` is set and a client connects with a valid
cert chain, the audit-log line carries the peer Subject CN:

    [moonlab.control] verb=CIRCUIT n_qubits=2 body=47 shots=0 wall_ms=1.12 rc=0 peer_cn=worker-3.cluster.local

In JSON-format mode (`MOONLAB_CONTROL_LOG_FORMAT=json`) the field
appears as `"peer_cn":"worker-3.cluster.local"`.  Connections that
arrive without a client cert never reach the worker; the metric
counters above capture the rejection.

## 5. Metrics endpoint

`GET /` analog: connect to the control port and send `METRICS\n`.
Response is Prometheus 0.0.4 text exposition.  The per-verb request
counter carries a `verb` label; defensive counters are unlabelled:

```
moonlab_control_requests_total{verb="CIRCUIT"}   Successful CIRCUIT requests
moonlab_control_requests_total{verb="SHOTS"}     Successful SHOTS requests
moonlab_control_requests_total{verb="HEALTH"}    HEALTH probes served
moonlab_control_requests_total{verb="METRICS"}   METRICS scrapes served
moonlab_control_rejected_total                   Requests rejected post-accept
moonlab_control_rate_limited_total               Connections refused by token bucket
moonlab_control_tls_handshake_failed_total       SSL_accept failures
moonlab_control_max_concurrent_rejected_total    Connections refused by concurrency cap
```

A typical scrape config:

```yaml
scrape_configs:
  - job_name: 'moonlab-control'
    static_configs:
      - targets: ['moonlab-0.cluster.local:7070']
    metrics_path: /
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'moonlab_control_(.*)'
        action: keep
```

(The MoonLab control plane speaks its own line-protocol on `/`, not
HTTP -- if your Prometheus build can't drive raw TCP, use the
`moonlab-control-exporter` sidecar in `tools/exporter/` to bridge.)

## 6. Bindings

### 6.1 Python

```python
from moonlab.control_plane import ControlPlaneServer

with ControlPlaneServer(
    host="0.0.0.0", port=7070,
    secret=os.environ["MOONLAB_SECRET"].encode(),
    tls_cert="cert.pem", tls_key="key.pem",
    require_client_ca="ca.pem",
    rate_limit_rps=10, rate_limit_burst=20,
    request_timeout_secs=30,
    max_concurrent=64,
) as srv:
    srv.run_forever()
```

Runtime tuning:

```python
srv.set_max_concurrent(128)
srv.set_request_timeout(60)
srv.set_rate_limit(20, 40)
```

### 6.2 Rust

```rust
use moonlab::control_plane::ControlPlaneServer;

let srv = ControlPlaneServer::open("0.0.0.0", 7070)?;
srv.set_secret(&secret)?;
srv.use_tls("cert.pem", "key.pem")?;
srv.require_client_cert("ca.pem")?;
srv.set_rate_limit(10, 20)?;
srv.set_request_timeout(30)?;
srv.set_max_concurrent(64)?;
srv.run(i32::MAX)?;
```

### 6.3 JavaScript / WASM

Browser deployments do not host control planes; the JS binding ships
the client surface only.

## 7. Operational runbook

### 7.1 Tuning the concurrency cap

Start with `max_concurrent = number_of_simulator_cores * 2`.  Each
worker thread spends most of its time inside the simulator backend
(BLAS / openmp); doubling the core count gives the I/O path enough
room to keep BLAS saturated without thrashing.  Monitor
`moonlab_control_max_concurrent_rejected_total` -- a steady non-zero
rate means clients are queueing.  Either raise the cap or scale out.

### 7.2 Rate-limit tuning

The rate limiter is per-source-IP (IPv4 only as of v0.9.0).  In
front-of-an-LB deployments where every request appears to come from
the LB's IP, disable rate limiting with
`set_rate_limit(0, 0)` and rely on the LB.

### 7.3 mTLS rollout

```
day -7   issue worker certs from internal CA; deploy to clients
day  0   moonlab_control_server_require_client_cert(srv, "internal-ca.pem")
day +1   check moonlab_control_tls_handshake_failed_total; any spike
         indicates a worker that didn't pick up its cert
```

The audit-log `peer_cn` field is the canonical "which worker did
this" record once mTLS is on; pipe stdout through your aggregator and
group by `peer_cn`.

### 7.4 Graceful shutdown

```c
moonlab_control_server_shutdown(srv);   /* writes to self-pipe */
/* wait for run() thread to return */
moonlab_control_server_close(srv);
```

The shutdown signal is async-signal-safe; call it from a SIGTERM
handler.

### 7.5 Debugging

```
# Tail-and-parse JSON request log
MOONLAB_CONTROL_LOG=1 MOONLAB_CONTROL_LOG_FORMAT=json ./moonlab-control \
  | jq 'select(.rc != 0)'

# Watch the cap counter live
watch -n1 'printf "METRICS\n" | nc 127.0.0.1 7070 \
  | grep max_concurrent_rejected_total'
```

## 8. Versioning

The control-plane wire protocol is stable across MoonLab 0.x; new
verbs and status codes are additive.  Clients that see an unknown
`OK <n>\n<bytes>` should treat the trailing bytes as opaque and pass
through.
