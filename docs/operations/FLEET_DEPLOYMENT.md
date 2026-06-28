# Moonlab Fleet Deployment

> **Audience:** SREs rolling Moonlab Community Edition (or a private
> overlay built on it) onto many hosts.  This document focuses on
> the multi-replica concerns that the single-node `RUNBOOK.md` does
> not cover.
>
> **In scope:** N-replica control plane behind a load balancer,
> secret distribution to a fleet, image registry workflow,
> per-replica observability, fleet-wide rate-limit semantics,
> rolling upgrades.
>
> **Not in scope:** the live-hardware (QGTL) path; live calibration
> scrapers; per-tenant billing implementation.  Those live in the
> private overlay that consumes the public hooks documented here.

## 1. Topology

The standard multi-node Moonlab deployment looks like this:

```
                              +--------------------+
            tenants  ----->   | external load LB   |   (TLS-terminating)
                              +---------+----------+
                                        |
                  +---------------------+---------------------+
                  |                     |                     |
       +----------+------+    +---------+--------+   +--------+---------+
       |   control-plane |    |  control-plane   |   |  control-plane   |
       |    replica-0    |    |    replica-1     |   |    replica-2     |
       |    127:7070     |    |    127:7070      |   |    127:7070      |
       +-+--------+------+    +-+-------+--------+   +-+-------+--------+
         |        |             |       |              |       |
         | exporter sidecar     | exporter sidecar     | exporter sidecar
         |        |             |       |              |       |
         v        v             v       v              v       v
            Prometheus federation             (single aggregator)
                              |
                              v
                   +----------+---------+
                   | overlay services   |
                   | - billing meter    |   <-- consumes scheduler
                   | - audit log shard  |       completion hook
                   | - quota service    |   <-- backs admission hook
                   +--------------------+
```

Every control-plane replica is **independent** -- they share no
state beyond what their backing overlay services hold.  This is
intentional: a control plane is a stateless compute node.  All
tenant state lives in the overlay layer.

## 2. Building the image

The reference Dockerfile builds a self-contained binary against
Debian 12 slim.  From the repo root:

```bash
docker build -f deploy/docker/Dockerfile.control-plane \
             -t moonlab/control-plane:1.0.3 \
             --build-arg MOONLAB_VERSION=1.0.3 \
             .
docker build -f deploy/docker/Dockerfile.exporter \
             -t moonlab/control-exporter:1.0.3 \
             .
docker build -f deploy/docker/Dockerfile.gateway \
             -t moonlab/websocket-gateway:1.0.3 \
             .
```

Tag and push to your registry (replace with yours):

```bash
docker tag moonlab/control-plane:1.0.3 myregistry.example/moonlab/control-plane:1.0.3
docker push myregistry.example/moonlab/control-plane:1.0.3
# ... same for control-exporter and websocket-gateway
```

The Community Edition image carries no proprietary code; the CI
hygiene gate (see `.github/workflows/ci.yml`) blocks proprietary
markers from landing in the public source the image is built from.

Image SHA pinning is recommended for fleet upgrades:

```yaml
image:
  repository: myregistry.example/moonlab/control-plane
  tag:        "1.0.3"
  digest:     "sha256:..."
```

## 3. Secret distribution

Four distinct secrets must reach every replica:

| Secret                       | Where it lives in the image           | Rotated by              |
|------------------------------|---------------------------------------|-------------------------|
| `hmac.bin`                   | `/etc/moonlab/hmac.bin` (raw bytes)   | rotation procedure §4   |
| `tls.crt` + `tls.key`        | `/etc/moonlab/server.{crt,key}`       | cert-manager / manual   |
| (optional) `clients-ca.crt`  | `/etc/moonlab/clients-ca.crt`         | client-CA rotation      |
| (optional) overlay creds     | overlay-defined                       | overlay-defined         |

In Kubernetes, two Secrets back this:

```bash
kubectl create secret generic moonlab-hmac \
        --from-file=hmac.bin=/path/to/hmac.bin

kubectl create secret tls moonlab-tls \
        --cert=/path/to/server.crt \
        --key=/path/to/server.key

# Optional mTLS:
kubectl create secret generic moonlab-client-ca \
        --from-file=ca.crt=/path/to/clients-ca.crt
```

Reference the secrets in `values.yaml`:

```yaml
controlPlane:
  auth:
    enabled: true
    secretName: moonlab-hmac
  tls:
    enabled: true
    secretName: moonlab-tls
    requireClientCert: true                  # mTLS
    clientCaSecretName: moonlab-client-ca
```

Rotate by replacing the Secret and triggering a rolling restart:

```bash
kubectl create secret generic moonlab-hmac --from-file=hmac.bin=/path/to/new.bin \
        --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/moonlab-control-plane
kubectl rollout status  deployment/moonlab-control-plane
```

The old secret is honored until the last old-image pod terminates.
Plan a 30-second overlap window for in-flight requests; clients
that re-use a stale secret receive `ERR -405` and must reconnect.

## 4. Per-replica state vs shared state

Moonlab v1.0.3 has these stateful surfaces in the control plane:

| Surface                   | Per-replica or shared?            | Implication for fleet     |
|---------------------------|-----------------------------------|---------------------------|
| HMAC shared secret        | Per-replica copy of the same blob | Identical via Secret      |
| TLS server cert           | Per-replica copy                  | Identical via Secret      |
| Active worker counter     | Per-replica                       | Cap N applies per replica |
| Per-IP rate limiter       | Per-replica                       | See "rate-limit math"     |
| Per-tenant admission      | Overlay-defined                   | Must consult shared store |
| Tenant identity (AUTH)    | Stateless on the moonlab side     | Free                      |
| Completion hook output    | Overlay-defined                   | Must ship to shared sink  |

### Rate-limit math

The per-IP token bucket lives in process memory.  With N replicas:

- Effective per-IP burst capacity = `N * burst`.
- Effective per-IP refill rate = `N * rate_rps`.

This is intentional: a tenant's burst budget scales with the fleet
size, which is desirable for legitimate clients hitting the load
balancer.  If you need a strict fleet-wide cap, install an
admission hook (Section 5) backed by a shared store (Redis,
etcd, your billing service).

### Concurrent-cap math

`MOONLAB_CONTROL_MAX_CONCURRENT` caps in-flight jobs per replica.
With three replicas at `max_concurrent: 32`, the fleet handles
96 simultaneous jobs.  Memory budget on a single host is bounded
by the *replica* cap, not the fleet cap; consult
`docs/operations/RUNBOOK.md` Section 9 for sizing.

## 5. Tenant table + admission hook

The admission hook (`docs/EXTENSION_SURFACES.md` Section 6 and
`docs/operations/RUNBOOK.md` Section 5) is the fleet-wide
enforcement point.  The hook itself lives **per replica** -- the
overlay installs it via `moonlab_control_server_set_admission_hook`
when the daemon boots.  Inside the hook, the overlay queries
fleet-shared state:

```c
int my_admission(const char *tenant_id, const char *verb,
                 int n_qubits, int n_shots, void *ctx) {
    /* Shared store: Redis / etcd / billing API.  Cache for a few
     * seconds in process to bound the per-request latency. */
    tenant_quota_t q;
    if (fetch_tenant_quota_cached(tenant_id, &q) != 0)
        return MOONLAB_CONTROL_BAD_ARG;     /* unknown tenant */
    if (q.locked_out)
        return MOONLAB_CONTROL_REJECTED;
    if (q.shots_remaining < n_shots)
        return MOONLAB_CONTROL_RATE_LIMITED;
    return 0;
}
```

Distribute the shared store as a sidecar (Redis on the same node)
or as a cluster service.  Each replica connects to it
independently; concurrent decrements are the overlay's problem.

## 6. Observability

Every replica's Prometheus exporter emits the metrics enumerated in
RUNBOOK Section 7.  Federate to a central Prometheus:

```yaml
# central Prometheus prometheus.yml
scrape_configs:
  - job_name: moonlab-fleet
    metrics_path: /metrics
    static_configs:
      - targets:
        - moonlab-control-0.example:9090
        - moonlab-control-1.example:9090
        - moonlab-control-2.example:9090
```

Or use kubernetes_sd for auto-discovery.

### Fleet alerts

- `sum(rate(moonlab_control_rejected_total[5m])) > 5` -- a tenant
  may be misconfigured.
- `sum(rate(moonlab_control_rate_limited_total[5m])) > 10` -- the
  per-IP bucket is firing fleet-wide; consider raising the cap or
  scaling out.
- `avg(moonlab_control_active_workers) > 0.8 * max_concurrent` --
  the fleet is at 80% saturation; add replicas before customers
  see `ERR -409`.

## 7. Rolling upgrade

The control plane has no on-disk state to migrate, so a rolling
upgrade is the standard Kubernetes rollout:

```bash
# 1. Push the new image
docker push myregistry.example/moonlab/control-plane:1.0.4

# 2. Update the helm values
helm upgrade moonlab ./deploy/helm/moonlab \
     --set image.tag=1.0.4 \
     --wait

# 3. Watch the rollout
kubectl rollout status deployment/moonlab-control-plane
```

The default rolling strategy keeps `replicas - 1` pods serving while
one pod restarts.  In-flight requests on the restarting pod are
held until the request timeout fires (`MOONLAB_CONTROL_REQUEST_TIMEOUT`)
and then the connection closes with `ERR -403`.  Tenants typically
retry; their next connection reaches a healthy replica.

For zero-cutoff upgrades, use blue/green: bring up a parallel
deployment of the new version, drain traffic from the old set, then
delete it.

## 8. Validation procedure after a fleet rollout

After every rollout, run this from a host that can reach the LB:

```bash
# 1. HEALTH probe (unauth, hits LB)
for i in $(seq 1 9); do
    printf 'HEALTH\n' | nc -w 1 lb.example.com 8443
done
# Expect 9x "OK alive" -- if any pod is unhealthy, traffic gets uneven.

# 2. Tenant smoke (requires the test helper from this repo)
./build/test_control_plane_tenant_smoke lb.example.com 8443 acme-corp

# 3. Metrics check
curl -s lb.example.com:9090/metrics | grep -E 'requests_total|rejected_total'
# CIRCUIT counter should be non-zero on every replica.
```

If any check fails, roll back:

```bash
helm rollback moonlab
```

## 9. Capacity benchmarks per host

Validated on the moonlab smoke host (Apple M-series, 8-core arm64,
2 GB free):

| Workload                        | Per-replica throughput  | Latency P99 |
|---------------------------------|-------------------------|-------------|
| Bell pair, 2 qubits             | ~1500 jobs/sec          | < 5 ms      |
| 16-qubit dense + 1024 shots     | ~25 jobs/sec            | ~40 ms      |
| 28-qubit dense + 64 shots       | ~1 job/sec              | ~1 s        |
| CA-MPS Heisenberg N=12 chi=64   | ~0.2 jobs/sec           | ~5 s        |

Multiply by replica count for fleet throughput.  These are floor
numbers under a single workload; mixed workloads bottleneck on
memory at high qubit counts (see RUNBOOK §9).

## 10. Failure-mode quick reference

| Symptom on a replica          | Likely cause                              | First action                                  |
|------------------------------|-------------------------------------------|-----------------------------------------------|
| `ERR -405 bad token` spike    | HMAC drift between replicas               | Re-apply Secret, rolling restart              |
| `ERR -409 server busy` spike  | One replica's `MAX_CONCURRENT` saturated  | Add replica, raise cap, or shed load          |
| TLS handshake failures        | Cert about to expire / drifted across LB  | Run cert-renew + rolling restart              |
| Replica missing from metrics  | Pod fell out; LB still routes to it       | `kubectl describe pod ...` + `kubectl logs`   |
| Replica accepting but slow    | Worker thread oversubscribed              | Check `active_workers` metric                 |

## See also

- `docs/operations/RUNBOOK.md` -- single-node operational doc.
- `docs/CONTROL_PLANE.md` -- wire protocol reference.
- `docs/EXTENSION_SURFACES.md` -- overlay integration guide.
- `COMMUNITY_EDITION.md` -- public / commercial product boundary.
- `deploy/helm/moonlab/values.yaml` -- canonical k8s config knobs.
