#!/usr/bin/env python3
"""Cross-host WebSocket load test against moonlab_websocket_gateway.

Uses only stdlib (asyncio + the WebSocket framing in `websockets`
package; falls back to raw stdlib if needed) so it runs on the
fleet without a moonlab build.  Counts successful Bell-2q OK
responses through the gateway.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

try:
    import websockets
except ImportError:
    print("Install: pip install websockets", file=sys.stderr)
    sys.exit(2)

BELL_2Q = "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n"


async def worker(uri: str, secret_hex: str, tenant: str | None,
                 deadline: float, results: dict):
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        try:
            async with websockets.connect(uri, open_timeout=5,
                                          close_timeout=2) as ws:
                msg = {"verb": "CIRCUIT", "circuit": BELL_2Q,
                       "secret": secret_hex}
                if tenant:
                    msg["tenant"] = tenant
                await ws.send(json.dumps(msg))
                reply = json.loads(await ws.recv())
                if reply.get("status") == "OK":
                    results["lats"].append(time.perf_counter() - t0)
                    results["ok"] += 1
                else:
                    results["err"] += 1
        except Exception:
            results["err"] += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True,
                    help="ws://gateway-host:8765/")
    ap.add_argument("--secret-hex", required=True,
                    help="hex-encoded HMAC secret")
    ap.add_argument("--tenant", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duration", type=float, default=5.0)
    args = ap.parse_args()

    deadline = time.perf_counter() + args.duration
    results = {"lats": [], "ok": 0, "err": 0}

    tasks = [asyncio.create_task(
        worker(args.uri, args.secret_hex, args.tenant, deadline, results))
             for _ in range(args.workers)]
    await asyncio.gather(*tasks)

    lats = sorted(results["lats"])
    n = len(lats)
    if n == 0:
        print(f"WSLOAD uri={args.uri} workers={args.workers} "
              f"reqs=0 errors={results['err']}")
        return 1
    p50 = lats[n // 2] * 1000
    p99 = lats[min(n - 1, int(n * 0.99))] * 1000
    rps = n / args.duration
    print(f"WSLOAD uri={args.uri} workers={args.workers} "
          f"duration={args.duration:.2f} "
          f"reqs={n} rps={rps:.1f} p50_ms={p50:.2f} "
          f"p99_ms={p99:.2f} errors={results['err']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
