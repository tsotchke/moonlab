# Archived Moonlab Documentation: moonlab Helm chart

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# moonlab Helm chart

Helm v3 chart for the Moonlab v1.0 cloud platform.

Deploys the three core services from `deploy/docker/`:

| Component               | Default | Service port |
|-------------------------|---------|--------------|
| Control-plane daemon    | on      | 7070         |
| Prometheus exporter     | on      | 9090         |
| WebSocket gateway       | off     | 8765         |

Image registry defaults to `docker.io/moonlab/{component}:1.0.5`.  Build
the images locally first (`docker compose -f deploy/docker/docker-compose.yml build`)
and push them to your registry, or override `image.registry` /
`image.repository` to point at a published copy.

## Verification status (v1.0.2)

The chart passes `helm lint deploy/helm/moonlab` and `helm template
my-moonlab deploy/helm/moonlab` (default + tls-enabled values) without
errors.  Apply against a real cluster (kind / minikube / production
k8s) is NOT recorded in this repo; it is on the v1.1 list.  Treat
this chart as a working starting template that needs cluster-specific
review of resource requests, storage classes, and image-registry
provenance before going to production.

## Install

[archived fence delimiter: ```]
helm install my-moonlab ./deploy/helm/moonlab
[archived fence delimiter: ```]

Override the defensive layer caps and rate limit:

[archived fence delimiter: ```]
helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.maxConcurrent=128 \
  --set controlPlane.rateLimitRps=200 \
  --set controlPlane.rateLimitBurst=400
[archived fence delimiter: ```]

## TLS / mTLS

[archived fence delimiter: ```]
kubectl create secret tls moonlab-tls \
  --cert=server.pem --key=server.key
kubectl create secret generic moonlab-tls-ca \
  --from-file=ca.crt=ca.pem

helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.tls.enabled=true \
  --set controlPlane.tls.secretName=moonlab-tls \
  --set controlPlane.tls.requireClientCert=true
[archived fence delimiter: ```]

When `requireClientCert: true`, the chart expects the same secret to
also contain `ca.crt`.  Combine cert+key+ca into one secret if you
prefer.

## HMAC-SHA3 auth

[archived fence delimiter: ```]
kubectl create secret generic moonlab-hmac --from-file=hmac.bin=secret.bin

helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.auth.enabled=true \
  --set controlPlane.auth.secretName=moonlab-hmac
[archived fence delimiter: ```]

## Browser clients

Turn on the WebSocket gateway:

[archived fence delimiter: ```]
helm install my-moonlab ./deploy/helm/moonlab \
  --set websocketGateway.enabled=true \
  --set websocketGateway.service.type=LoadBalancer
[archived fence delimiter: ```]

Pair with an Ingress + cert-manager for `wss://` termination.

## Liveness + readiness

The control-plane pod runs the `HEALTH` line-protocol probe every
10s (liveness) and 5s (readiness).  The exporter pod runs an HTTP
`/metrics` check on the same cadence.

## Cleaning up

[archived fence delimiter: ```]
helm uninstall my-moonlab
[archived fence delimiter: ```]
```
