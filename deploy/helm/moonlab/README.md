# moonlab Helm chart

Helm v3 chart for the Moonlab v1.0 cloud platform.

Deploys the three core services from `deploy/docker/`:

| Component               | Default | Service port |
|-------------------------|---------|--------------|
| Control-plane daemon    | on      | 7070         |
| Prometheus exporter     | on      | 9090         |
| WebSocket gateway       | off     | 8765         |

Image registry defaults to `docker.io/moonlab/{component}:1.0.4`.  Build
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

```
helm install my-moonlab ./deploy/helm/moonlab
```

Override the defensive layer caps and rate limit:

```
helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.maxConcurrent=128 \
  --set controlPlane.rateLimitRps=200 \
  --set controlPlane.rateLimitBurst=400
```

## TLS / mTLS

```
kubectl create secret tls moonlab-tls \
  --cert=server.pem --key=server.key
kubectl create secret generic moonlab-tls-ca \
  --from-file=ca.crt=ca.pem

helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.tls.enabled=true \
  --set controlPlane.tls.secretName=moonlab-tls \
  --set controlPlane.tls.requireClientCert=true
```

When `requireClientCert: true`, the chart expects the same secret to
also contain `ca.crt`.  Combine cert+key+ca into one secret if you
prefer.

## HMAC-SHA3 auth

```
kubectl create secret generic moonlab-hmac --from-file=hmac.bin=secret.bin

helm install my-moonlab ./deploy/helm/moonlab \
  --set controlPlane.auth.enabled=true \
  --set controlPlane.auth.secretName=moonlab-hmac
```

## Browser clients

Turn on the WebSocket gateway:

```
helm install my-moonlab ./deploy/helm/moonlab \
  --set websocketGateway.enabled=true \
  --set websocketGateway.service.type=LoadBalancer
```

Pair with an Ingress + cert-manager for `wss://` termination.

## Liveness + readiness

The control-plane pod runs the `HEALTH` line-protocol probe every
10s (liveness) and 5s (readiness).  The exporter pod runs an HTTP
`/metrics` check on the same cadence.

## Cleaning up

```
helm uninstall my-moonlab
```
