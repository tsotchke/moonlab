# moonlab-websocket-gateway

Browser-facing WebSocket bridge for the Moonlab control plane.

Browsers cannot open raw TCP sockets, so a JS client that wants to
submit circuits to a remote moonlab cluster has to go through some
HTTP-family transport.  This sidecar terminates a WebSocket and
re-encodes each message as a moonlab line-protocol round trip.

## Quick start

```
pip install websockets
python3 moonlab_websocket_gateway.py \
  --target  127.0.0.1:7070 \
  --listen  0.0.0.0:8765
```

Then from a browser:

```javascript
const ws = new WebSocket("ws://gateway-host:8765/");
ws.onopen = () => {
  ws.send(JSON.stringify({
    verb:    "CIRCUIT",
    circuit: "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
  }));
};
ws.onmessage = (evt) => {
  const r = JSON.parse(evt.data);
  if (r.status === "OK") console.log("probs:", r.probs);
  else                    console.warn("err:", r.code, r.message);
};
```

## Protocol

One JSON object per direction.

### Request envelope

| Field      | Required           | Notes                                      |
|------------|--------------------|--------------------------------------------|
| `verb`     | yes                | `CIRCUIT` / `SHOTS` / `HEALTH` / `METRICS` |
| `circuit`  | for CIRCUIT/SHOTS  | moonlab-circuit v1 text                    |
| `shots`    | for SHOTS          | positive int                               |
| `secret`   | optional           | HMAC-SHA3-256 shared secret (hex or utf-8) |

### Reply envelope

| Field      | Type   | When               |
|------------|--------|--------------------|
| `status`   | string | `OK` or `ERR`      |
| `code`     | int    | 0 on OK; negative `MOONLAB_CONTROL_*` on ERR |
| `message`  | string | ERR / HEALTH path  |
| `probs`    | number[]| CIRCUIT OK        |
| `counts`   | number[]| SHOTS OK          |
| `body`     | string | METRICS OK        |

## Tests

```
python3 -m pytest tools/gateway/tests/ -v
```

The test suite uses an in-process fake control plane (no
libquantumsim dependency) to exercise every verb path plus the
error paths (unknown verb, bad JSON).

## Docker

The gateway ships as a second container next to the
moonlab-control-exporter in `deploy/docker/`.  See
`deploy/docker/README.md` for the production stack.

## Security

The gateway DOES NOT terminate TLS itself.  Deploy behind an
nginx / Caddy / Cloudflare reverse proxy that handles `wss://` and
talks plain `ws://` to the gateway on the private network.  The
gateway forwards the optional `secret` field to the moonlab control
plane via `AUTH <hmac-sha3-256>` so the auth check happens inside the
control plane, not at the gateway.
