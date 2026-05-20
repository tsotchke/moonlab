#!/usr/bin/env python3
"""HTTP /metrics sidecar for the Moonlab control plane.

Prometheus expects to scrape HTTP at a `/metrics` endpoint.  The
Moonlab control plane speaks its own line-protocol over TCP, not
HTTP, so a thin bridge is needed.  This script binds an HTTP port,
forwards each `GET /metrics` to a Moonlab control endpoint over the
line protocol, and returns the body verbatim.

Usage::

    moonlab_control_exporter --target 127.0.0.1:7070 --listen 0.0.0.0:9090

Then point Prometheus at ``http://<host>:9090/metrics``.

The control endpoint must be reachable on TCP (TLS clients are
deferred -- mTLS scrapes happen direct, not via this sidecar).
"""

from __future__ import annotations

import argparse
import http.server
import logging
import socket
import sys
from typing import Tuple

LOG = logging.getLogger("moonlab.exporter")


def scrape(host: str, port: int, timeout_secs: float) -> bytes:
    """Send METRICS to the control plane and return the response body."""
    with socket.create_connection((host, port), timeout=timeout_secs) as s:
        s.sendall(b"METRICS\n")
        # Header line is "METRICS <n>\n<n bytes>" on success, or "ERR ...".
        header = b""
        while b"\n" not in header:
            chunk = s.recv(64)
            if not chunk:
                raise IOError("control-plane closed during header read")
            header += chunk
        line, sep, rest = header.partition(b"\n")
        if not line.startswith(b"METRICS "):
            raise IOError(f"control-plane returned: {line!r}")
        n = int(line[len(b"METRICS "):].decode("ascii"))
        body = rest
        while len(body) < n:
            chunk = s.recv(min(4096, n - len(body)))
            if not chunk:
                raise IOError(
                    f"control-plane closed during body read "
                    f"({len(body)} / {n} bytes)"
                )
            body += chunk
        return body[:n]


def make_handler(target_host: str, target_port: int, timeout_secs: float):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (Python stdlib API)
            if self.path not in ("/metrics", "/"):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found\n")
                return
            try:
                body = scrape(target_host, target_port, timeout_secs)
            except Exception as exc:
                LOG.warning("scrape failed: %s", exc)
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(f"upstream error: {exc}\n".encode("utf-8"))
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/plain; version=0.0.4; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            LOG.info("%s -- %s", self.address_string(), fmt % args)

    return Handler


def parse_endpoint(arg: str) -> Tuple[str, int]:
    host, _, port = arg.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(
            f"expected host:port, got {arg!r}")
    return host, int(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=parse_endpoint, required=True,
                        help="Moonlab control endpoint to scrape, host:port")
    parser.add_argument("--listen", type=parse_endpoint,
                        default=("0.0.0.0", 9090),
                        help="HTTP bind for Prometheus, host:port "
                             "(default 0.0.0.0:9090)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Per-scrape socket timeout in seconds")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    target_host, target_port = args.target
    listen_host, listen_port = args.listen

    handler = make_handler(target_host, target_port, args.timeout)
    server = http.server.ThreadingHTTPServer(
        (listen_host, listen_port), handler)
    LOG.info("listening on http://%s:%d (target %s:%d)",
             listen_host, listen_port, target_host, target_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutdown via SIGINT")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
