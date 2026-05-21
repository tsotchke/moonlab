# Moonlab Operations Runbook

> **Audience:** SREs operating a Moonlab Community Edition or Moonlab
> Commercial deployment.  Concrete: every section answers "what
> command do I run, what file do I edit, what log line means what."
>
> **Not in scope:** marketing-style overview or product features.
> See README.md for those.  See `COMMUNITY_EDITION.md` for the
> public/private product boundary and `docs/CONTROL_PLANE.md` for
> the protocol reference.

## 1. Deploying a single node

Reference deployment uses the supplied docker-compose stack:

```bash
cd deploy/docker
docker-compose up -d control-plane exporter
docker-compose logs -f control-plane
```

The stack brings up:

| Service        | Image                              | Port  | Purpose                              |
|----------------|------------------------------------|-------|--------------------------------------|
| `control-plane`| `Dockerfile.control-plane`         | 8765  | Accepts CIRCUIT / SHOTS jobs         |
| `exporter`     | `Dockerfile.exporter`              | 9100  | Prometheus scrape endpoint (METRICS) |
| `gateway`      | `Dockerfile.gateway`               | 8080  | Optional WebSocket front-end for browser clients |

Health-check the control plane from the host:

```bash
printf 'HEALTH\n' | nc 127.0.0.1 8765   # expect: OK alive
```

For Kubernetes deployment see `deploy/helm/moonlab/`.  The default
values render two-replica control plane behind a NodePort, with
mTLS turned ON and a placeholder secret you MUST replace before
production use.

## 2. First-time configuration

Required env vars on the `control-plane` container:

```
MOONLAB_CONTROL_SECRET           hex-encoded HMAC shared secret
MOONLAB_CONTROL_TLS_CERT         /etc/moonlab/server.crt
MOONLAB_CONTROL_TLS_KEY          /etc/moonlab/server.key
MOONLAB_CONTROL_MAX_CONCURRENT   bound on in-flight jobs (e.g. 32)
MOONLAB_CONTROL_RATE_LIMIT_RPS   per-source-IP rate ceiling (e.g. 20)
MOONLAB_CONTROL_REQUEST_TIMEOUT  per-request socket timeout (seconds, e.g. 30)
```

Optional but recommended:

```
MOONLAB_CONTROL_TLS_CLIENT_CA    /etc/moonlab/clients-ca.crt   # mTLS
MOONLAB_LOG_FORMAT               json                          # structured logs
```

Public moonlab does not enforce any tenant policy.  To accept the
`AUTH <tenant>:<hmac>` form and bill / quota per tenant, run a
private overlay process that:

1. Builds against `libquantumsim` and links the same
   `moonlab_control_server_t` instance.
2. Calls `moonlab_control_server_set_admission_hook(srv, fn, ctx)`
   to install a quota check (see Section 5).
3. Calls `moonlab_scheduler_set_completion_hook(fn, ctx)` to install
   a billing / audit consumer (see Section 6).

The Moonlab Commercial deployment ships such an overlay; Community
Edition operators wire their own.

## 3. Rotating the HMAC secret

The HMAC shared secret authenticates clients to the server.  Rotate
quarterly or after any suspected leak:

1. Generate a fresh 32-byte secret on the operator workstation:
   ```bash
   openssl rand -hex 32 | tee new-secret.hex
   ```
2. Pre-stage the new secret on every server host (e.g. via the
   secrets store backing Kubernetes secrets).
3. Reload the control plane.  The default deployment treats
   `MOONLAB_CONTROL_SECRET` as picked up on startup; rolling
   restart the pods:
   ```bash
   kubectl rollout restart deployment/moonlab-control-plane
   ```
4. Distribute the new secret to client tenants (out-of-band; do
   not commit to the public repo -- the `public-hygiene` CI gate
   greps for the obvious shapes).
5. After the cutover window, revoke the old secret.

There is **no on-the-wire renegotiation** of the secret.  Clients
using the old secret receive `ERR -405 bad token` and the connection
closes.  Clients must reconnect with the new secret.

## 4. Rotating TLS certificates

The control plane re-reads the certificate paths on
`SIGHUP`.  In Kubernetes the cert-manager `Certificate` resource
combined with `secret`-mounted cert files plus a sidecar that signals
SIGHUP on rotation handles this automatically.  Reference values in
`deploy/helm/moonlab/templates/cert-rotation.yaml`.

To rotate manually:

1. Place the new `server.crt` + `server.key` at the configured paths.
2. `kill -HUP $(pidof moonlab_control_plane)` (or
   `kubectl exec -- kill -HUP 1`).
3. Verify the new certificate is in effect:
   ```bash
   openssl s_client -connect controlplane:8443 -servername moonlab.example </dev/null \
       | openssl x509 -noout -dates
   ```

## 5. Installing a per-tenant admission hook (v1.0.3)

The admission hook gates every authenticated request before
dispatch.  Use it for:

- Per-tenant daily / monthly shot budgets.
- Per-tenant rate limits (beyond the per-IP one).
- Paid-tier qubit-count caps.
- Emergency tenant lockout.

C signature:

```c
int my_admission(const char *tenant_id,
                 const char *verb,
                 int         num_qubits,
                 int         num_shots,
                 void       *ctx);
// return 0 to allow; MOONLAB_CONTROL_RATE_LIMITED (-408) for over-quota;
//        MOONLAB_CONTROL_REJECTED  (-405) for tier-blocked / locked-out.

moonlab_control_server_set_admission_hook(server, my_admission, &state);
```

The hook fires after AUTH succeeds and the verb header is parsed
but **before** the circuit body is read, so refused jobs do not
consume bandwidth or simulator time.

Typical overlay structure:

1. Look up `tenant_id` in a tenant table (database, in-memory cache,
   etcd, ...).
2. Compute remaining quota (shots this billing period - count of
   `num_shots`).
3. If over: return `MOONLAB_CONTROL_RATE_LIMITED` and log
   `tenant_id, verb, requested_shots, budget_remaining`.
4. If allowed: return 0; the run will then fire the completion hook
   (Section 6) where you debit the budget.

**Common pitfalls:**

- Hook return value `0` means "allow"; any non-zero value refuses
  and is sent to the client as `ERR <rc> <msg>`.
- The hook is called on the worker thread; it must be thread-safe.
- A blocking hook stalls the worker; if your tenant lookup is over
  the network, cache aggressively.

## 6. Installing the completion hook (billing / audit / dashboard)

The completion hook fires after every **successful** scheduler run
(failed dispatches do not fire it):

```c
void my_completion(const moonlab_job_t          *job,
                   const moonlab_job_results_t  *results,
                   const char                   *backend_name,
                   void                         *ctx);

moonlab_scheduler_set_completion_hook(my_completion, &state);
```

Inside the hook:

```c
const char *tenant_id  = moonlab_scheduler_current_tenant_id();
const char *request_id = moonlab_scheduler_current_request_id();
const int   n_qubits   = results->num_qubits;
const int   n_shots    = results->total_shots;
```

- Use a worker queue inside the callback to hand off to the
  billing / audit / dashboard sinks; do NOT block the scheduler.
- Treat the hook as best-effort delivery: if your sink is down,
  buffer to disk and ship later.  The scheduler does not retry
  hook failures.
- The completion hook is single-slot; install your overlay's hook
  at process start and never swap it mid-run.

## 7. Reading the Prometheus metrics

The `METRICS` verb returns Prometheus text-format counters.  Scrape
from the exporter sidecar or directly:

```bash
printf 'METRICS\n' | nc 127.0.0.1 8765
```

Stable counters (since v0.8.23):

| Metric                                          | Meaning                                                   |
|-------------------------------------------------|-----------------------------------------------------------|
| `moonlab_control_requests_total{verb=...}`      | Per-verb request counter (CIRCUIT, SHOTS, HEALTH, METRICS)|
| `moonlab_control_rejected_total`                | Requests refused (bad input, auth, execute failure)       |
| `moonlab_control_rate_limited_total`            | Requests refused by the per-IP token bucket               |
| `moonlab_control_tls_handshake_failed_total`    | TLS handshakes that failed in SSL_accept                  |
| `moonlab_control_max_concurrent_rejected_total` | Connections refused by the bounded thread-pool ceiling    |

Alerting suggestions:

- `rate(moonlab_control_rejected_total[5m]) > 1.0` -- something is
  driving auth failures; check log lines tagged `bad token` /
  `missing AUTH` for the source IPs.
- `rate(moonlab_control_max_concurrent_rejected_total[5m]) > 0` --
  the cap is being hit; either scale out replicas or raise
  `MOONLAB_CONTROL_MAX_CONCURRENT`.
- `rate(moonlab_control_tls_handshake_failed_total[5m]) > 0.1` --
  client cert problems; correlate with the mTLS audit log.

## 8. Debugging common errors

### `ERR -405 bad token`

HMAC mismatch.  Verify:

- The client is using the same secret as the server.
- The client sends `AUTH <hex>\n CIRCUIT <n>\n<body>` in that order
  (a TLS wrapper that buffers / coalesces can break the HMAC).
- The HMAC is computed over the **verb line including the trailing
  newline**, not the body.

### `ERR -405 missing AUTH`

Server has a secret configured but client sent CIRCUIT / SHOTS
without a preceding AUTH line.  Either:

- The client wasn't upgraded; use `moonlab_control_submit_circuit_auth`
  (or `_auth_tenant`) instead of `_submit_circuit`.
- The server has `MOONLAB_CONTROL_SECRET` set when it shouldn't
  (e.g. a dev environment).  Clear it for unauthenticated mode.

### `ERR -408 rate limited`

Per-IP token bucket exhausted.  Look at
`moonlab_control_rate_limited_total` and the access log to find the
offending source.  Either:

- Raise `MOONLAB_CONTROL_RATE_LIMIT_RPS` if this is legitimate
  traffic.
- Add the source to an upstream allow-list and bypass the bucket
  for it.

### `ERR -409 server busy`

Concurrent-cap hit.  Either:

- Scale out replicas (preferred).
- Raise `MOONLAB_CONTROL_MAX_CONCURRENT`; each in-flight job consumes
  roughly `2^N * 16 bytes` of state-vector memory for an N-qubit
  dense run.  At 32 qubits that is 64 GiB per job, so the cap
  protects the host from OOM.

### `ERR -405 tenant rejected`

A v1.0.3+ admission hook refused this tenant.  Logs on the overlay
side identify which check fired (quota, lockout, tier).  No
moonlab-side action; this is by design.

### Connection accepted but no response

Probably a `MOONLAB_CONTROL_REQUEST_TIMEOUT` firing on a slow body
upload.  Check `dmesg` and the access log for `TIMEOUT` lines.
Raise the timeout if your clients legitimately send large circuits
slowly.

## 9. Capacity planning

Per-replica resource budget at the default `MAX_CONCURRENT=32` cap:

- **Memory:** state-vector workloads dominate at high qubit counts.
  Plan for `worst-case-qubits * 16 bytes * MAX_CONCURRENT`.  For 32
  qubits at 32 concurrent: 2 TiB.  Use `MAX_QUBITS` cap (env var)
  or admission-hook tier gating to bound per-tenant qubit count.
- **CPU:** the scheduler uses OpenMP fan-out, default
  `num_workers=1` per job.  Tune `OMP_NUM_THREADS` for the
  underlying core count.  Each MPS / CA-MPS path is roughly
  single-threaded; state-vector kernels vectorise.
- **Network:** results return as binary doubles (probabilities,
  8 bytes * 2^qubits) or shot outcomes (8 bytes * num_shots).
  Plan for `~10 MB/s` per replica at typical workloads.

## 10. Backup and disaster recovery

Moonlab the runtime is stateless; there is nothing to back up on
the moonlab process side.  The data that must survive a failure
lives in:

- The overlay's billing / audit store (Section 6).
- The customer-facing tenant directory (Section 5).
- The TLS cert + HMAC secret (Sections 3 & 4).

The runbook for restoring these depends on your overlay; this
document does not prescribe one.  Moonlab Commercial customers
have an SLA-backed restore playbook; Community Edition operators
build their own.

## 11. Public-CI hygiene gate

The `public-hygiene` CI lane in `.github/workflows/ci.yml` does two
things on every PR:

1. Greps for proprietary-marker prefixes
   (`// PROPRIETARY:`, `* TSOTCHKE-INTERNAL:`, etc.) and fails on
   match.
2. Greps for credential shapes (AWS / GitHub / OpenAI / JWT / PEM)
   and fails on match.

If a legitimate PR trips one, the fix is either:

- The marker form was descriptive (e.g. a doc that quotes
  "PROPRIETARY:" verbatim).  Anchor the test pattern more
  narrowly or rephrase the doc.
- The credential shape was a test fixture (a JWT signed with the
  published example secret).  Add an explicit allowlist entry in
  the workflow.

Do not relax the patterns to make the test pass; both gates are
there to catch the next operator's mistake.

## See also

- `docs/CONTROL_PLANE.md` -- protocol reference + status codes.
- `docs/STABLE_ABI.md` -- which C symbols are stable across the v1.x line.
- `docs/EXTENSION_SURFACES.md` -- integration guide for overlay authors.
- `COMMUNITY_EDITION.md` -- public / commercial product boundary.
- `examples/extensions/open_core_overlay_demo.c` -- reference
  overlay code exercising all four plug-in surfaces.
