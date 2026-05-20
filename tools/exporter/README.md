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

| Flag                 | Default          | Meaning                                                        |
|----------------------|------------------|----------------------------------------------------------------|
| `--target`           | (required)       | Moonlab control endpoint, `host:port`                          |
| `--listen`           | `0.0.0.0:9090`   | HTTP bind for Prometheus, `host:port`                          |
| `--timeout`          | `5.0`            | Per-scrape socket timeout in seconds                           |
| `--tls-ca`           | `none`           | CA bundle PEM that verifies the control-plane server cert      |
| `--client-cert`      | `none`           | Client cert PEM for mTLS scraping (requires `--client-key`)    |
| `--client-key`       | `none`           | Client key PEM for mTLS scraping (requires `--client-cert`)    |
| `--tls-insecure`     | `false`          | Skip server-cert verification (development only)               |
| `--tls-server-name`  | (target host)    | SNI / server-name override                                     |
| `--log-level`        | `INFO`           | `DEBUG / INFO / WARNING / ERROR`                               |

## TLS / mTLS scraping

```
python3 moonlab_control_exporter.py \
  --target moonlab-0.cluster.local:7070 \
  --tls-ca   /etc/moonlab/ca.pem \
  --client-cert /etc/moonlab/scraper.pem \
  --client-key  /etc/moonlab/scraper.key \
  --listen 0.0.0.0:9090
```

Any of `--tls-ca`, `--client-cert`, or `--tls-insecure` enables TLS;
all combinations of (CA-verified, mTLS, insecure-skip) are supported.

## Failure modes

| HTTP status | Cause                                                 |
|------------:|--------------------------------------------------------|
|         200 | Scrape succeeded; body is the Prometheus text exposition |
|         404 | Request to a path other than `/metrics` or `/`           |
|         502 | Upstream connection refused, timed out, or malformed     |

The exporter is single-process and threading; one worker per HTTP
request.

## Tests

```
python3 -m pytest tools/exporter/tests/ -v
```

The test suite covers plain TCP scrape, large bodies, malformed
upstream replies, unreachable upstream, TLS with `--tls-insecure`,
TLS with an explicit `--tls-ca`, and the HTTP handler returning
200 / 404 / 502 paths.
