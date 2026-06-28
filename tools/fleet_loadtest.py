#!/usr/bin/env python3
"""Cross-host load test client for moonlab-control-server.

Self-contained -- stdlib only (hmac + hashlib + socket + threading).
Speaks the v1.0.5 wire protocol directly (HMAC-SHA3-256 + AUTH
tenant_id:hex form).  Runnable on any host that has python3.6+,
no moonlab build required.

Usage:
    ./fleet_loadtest.py --host <tailscale-ip> --port 17075 \\
        --secret <ascii> [--tenant <id>] --workers N --duration S

Reports req/sec + P50/P90/P99 latency + error count.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import socket
import ssl
import struct
import sys
import threading
import time


# Canonical Bell-2q circuit in moonlab-circuit v1 text format.
BELL_2Q = b"# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n"


def hmac_sha3_256(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, digestmod=hashlib.sha3_256).hexdigest()


def submit_bell_circuit(host: str, port: int, secret: bytes,
                        tenant, tls_ctx=None) -> float:
    """Submit one Bell 2q CIRCUIT job; return wall-clock seconds."""
    encoded = BELL_2Q
    verb_line = f"CIRCUIT {len(encoded)}\n".encode("ascii")
    t0 = time.perf_counter()
    raw = socket.create_connection((host, port), timeout=10.0)
    s = tls_ctx.wrap_socket(raw, server_hostname=host) if tls_ctx else raw
    try:
        tok = hmac_sha3_256(secret, verb_line)
        if tenant:
            auth = f"AUTH {tenant}:{tok}\n".encode("ascii")
        else:
            auth = f"AUTH {tok}\n".encode("ascii")
        s.sendall(auth)
        s.sendall(verb_line)
        s.sendall(encoded)
        # Server replies "OK <n>\n" then n * 8 bytes of little-endian f64.
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError("server closed before OK header")
            buf += chunk
        header, _, rest = buf.partition(b"\n")
        if not header.startswith(b"OK "):
            raise RuntimeError(f"server error: {header!r}")
        n = int(header.split()[1])
        need = n * 8
        while len(rest) < need:
            chunk = s.recv(need - len(rest))
            if not chunk:
                raise RuntimeError("server closed mid-payload")
            rest += chunk
    finally:
        s.close()
    return time.perf_counter() - t0


def worker(host, port, secret, tenant, deadline, lats, errs_box,
           tls_ctx=None):
    while time.perf_counter() < deadline:
        try:
            lats.append(submit_bell_circuit(host, port, secret, tenant,
                                            tls_ctx=tls_ctx))
        except Exception:
            errs_box[0] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--tenant", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--tls", action="store_true",
                    help="wrap socket in TLS")
    ap.add_argument("--tls-ca", default=None,
                    help="CA cert path to verify server (TLS only)")
    ap.add_argument("--tls-insecure", action="store_true",
                    help="TLS without cert verification (smoke only)")
    args = ap.parse_args()

    secret = args.secret.encode("ascii")
    deadline = time.perf_counter() + args.duration

    tls_ctx = None
    if args.tls:
        tls_ctx = ssl.create_default_context()
        if args.tls_ca:
            tls_ctx.load_verify_locations(cafile=args.tls_ca)
        if args.tls_insecure:
            tls_ctx.check_hostname = False
            tls_ctx.verify_mode = ssl.CERT_NONE

    threads = []
    per_worker_lats = [[] for _ in range(args.workers)]
    per_worker_errs = [[0] for _ in range(args.workers)]
    for i in range(args.workers):
        t = threading.Thread(
            target=worker,
            args=(args.host, args.port, secret, args.tenant,
                  deadline, per_worker_lats[i], per_worker_errs[i]),
            kwargs={"tls_ctx": tls_ctx},
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    all_lats: list[float] = []
    for ls in per_worker_lats:
        all_lats.extend(ls)
    all_lats.sort()
    errs = sum(b[0] for b in per_worker_errs)
    n = len(all_lats)
    if n == 0:
        print(f"FLEETLOAD host={args.host} workers={args.workers} "
              f"duration={args.duration} reqs=0 errors={errs}")
        return 1
    rps = n / args.duration
    p50 = all_lats[int(n * 0.50)] * 1000
    p90 = all_lats[int(n * 0.90)] * 1000
    p99 = all_lats[min(n - 1, int(n * 0.99))] * 1000
    tenant_s = args.tenant or "(none)"
    print(f"FLEETLOAD host={args.host} workers={args.workers} "
          f"duration={args.duration:.2f} tenant={tenant_s} "
          f"reqs={n} rps={rps:.1f} p50_ms={p50:.2f} "
          f"p90_ms={p90:.2f} p99_ms={p99:.2f} errors={errs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
