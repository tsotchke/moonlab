# Moonlab v1.0 production stack (Docker)

A three-container compose stack that brings up:

| Service             | Port | Role                                                |
|---------------------|------|------------------------------------------------------|
| `moonlab-control`   | 7070 | Control-plane daemon (line protocol over TCP)        |
| `moonlab-exporter`  | 9090 | HTTP `/metrics` Prometheus bridge                    |
| `prometheus`        | 9091 | Prometheus UI scraping the exporter                  |

## Quickstart

```
docker compose -f deploy/docker/docker-compose.yml up --build
```

First boot builds two images (control-plane in Debian-slim,
exporter in python-slim).  Subsequent boots are fast.

Smoke-test once the stack is up:

```
# HEALTH probe
printf 'HEALTH\n' | nc -w 1 127.0.0.1 7070
# -> OK alive

# METRICS via the line protocol directly
printf 'METRICS\n' | nc -w 1 127.0.0.1 7070 | head

# METRICS via Prometheus HTTP bridge
curl -s http://127.0.0.1:9090/metrics | head

# Prometheus UI
open http://127.0.0.1:9091
```

## Submit a circuit

A 2-qubit Bell pair from Python (uses the `moonlab` Python binding):

```python
from moonlab.qgtl   import QgtlCircuit, GateType
from moonlab.control_plane import submit_circuit

c = (QgtlCircuit(num_qubits=2)
     .add_gate(GateType.H,    target=0)
     .add_gate(GateType.CNOT, target=1, control=0))

probs = submit_circuit("127.0.0.1", 7070, c.serialize())
print(probs)   # [0.5, 0.0, 0.0, 0.5]
```

Equivalent calls exist for the Rust (`moonlab::control_plane::submit_circuit`)
and Node.js (`@moonlab/quantum-core` `controlPlaneSubmitCircuit`)
bindings.

## Hardening the deployment

The default `CMD` in `Dockerfile.control-plane` already enables:

- dual-stack v4/v6 listener (`--host ::`)
- 64 in-flight workers cap (`--max-concurrent 64`)
- 60s per-request timeout (`--request-timeout 60`)
- 100 req/s rate limit per source IP with burst of 200 (`--rate-limit-rps 100`)

To enable TLS + mTLS in front of the control plane:

```yaml
# docker-compose.yml override
services:
  moonlab-control:
    volumes:
      - /etc/moonlab/tls:/etc/moonlab/tls:ro
      - /etc/moonlab/secret:/etc/moonlab/secret:ro
    command:
      - --host=::
      - --port=7070
      - --tls-cert=/etc/moonlab/tls/server.pem
      - --tls-key=/etc/moonlab/tls/server.key
      - --client-ca=/etc/moonlab/tls/ca.pem
      - --secret-file=/etc/moonlab/secret/hmac.bin
      - --max-concurrent=64
      - --request-timeout=60
      - --rate-limit-rps=100
      - --rate-limit-burst=200
```

Bind-mount your PEM bundle + HMAC secret from the host; the
exporter still sees plain-TCP `moonlab-control:7070` inside the
private network, which is fine -- the network is bridge-isolated.
For external TLS scrape, point a separate exporter at the
host-exposed port instead and pass `--tls-ca`, `--client-cert`,
`--client-key` to it (see `tools/exporter/README.md`).

## Cleaning up

```
docker compose -f deploy/docker/docker-compose.yml down
docker rmi moonlab/control-plane:0.10.0 moonlab/control-exporter:0.10.0
```

## Architecture

```
   client --line proto--> :7070  moonlab-control (private network)
                                   |
                                   v line proto
                                 :7070
                              moonlab-exporter -- HTTP /metrics --> :9090
                                                       |
                                                       v
                                                  prometheus :9091
```

The control plane speaks the moonlab line protocol on raw TCP.
Browsers + Prometheus need HTTP; the exporter sidecar bridges.
Native Python / Rust / Node clients use the line protocol
directly, bypassing the exporter.
