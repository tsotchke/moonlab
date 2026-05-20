# moonlab-control-exporter

HTTP `/metrics` sidecar that bridges the Moonlab control plane's
line-protocol METRICS endpoint to a Prometheus-scrapable HTTP one.

Required because the control plane speaks its own line protocol over
raw TCP, not HTTP, so most Prometheus builds cannot scrape it
directly.

## Quick start

```
python3 moonlab_control_exporter.py \
  --target 127.0.0.1:7070 \
  --listen 0.0.0.0:9090
```

Then point Prometheus at `http://<sidecar-host>:9090/metrics`.

## Sample `prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'moonlab-control'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
```

## Flags

| Flag           | Default          | Meaning                                  |
|----------------|------------------|------------------------------------------|
| `--target`     | (required)       | Moonlab control endpoint, `host:port`    |
| `--listen`     | `0.0.0.0:9090`   | HTTP bind for Prometheus, `host:port`    |
| `--timeout`    | `5.0`            | Per-scrape socket timeout in seconds     |
| `--log-level`  | `INFO`           | `DEBUG / INFO / WARNING / ERROR`         |

## Failure modes

| HTTP status | Cause                                                 |
|------------:|--------------------------------------------------------|
|         200 | Scrape succeeded; body is the Prometheus text exposition |
|         404 | Request to a path other than `/metrics` or `/`           |
|         502 | Upstream connection refused, timed out, or malformed     |

The exporter is single-process and threading; one worker per HTTP
request.  TLS-only control endpoints are not supported by this
sidecar -- for those, scrape the control plane directly with a TLS
exporter (e.g. Vector or a Python client that wraps the TLS submit).
