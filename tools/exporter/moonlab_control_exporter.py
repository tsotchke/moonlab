#!/usr/bin/env python3
"""HTTP /metrics sidecar for the Moonlab control plane.

Prometheus expects to scrape HTTP at a `/metrics` endpoint.  The
Moonlab control plane speaks its own line-protocol over TCP, not
HTTP, so a thin bridge is needed.  This script binds an HTTP port,
forwards each `GET /metrics` to a Moonlab control endpoint over the
line protocol, and returns the body verbatim.

Plain TCP::

    moonlab_control_exporter --target 127.0.0.1:7070 --listen 0.0.0.0:9090

TLS / mTLS::

    moonlab_control_exporter \\
        --target moonlab-0.cluster.local:7070 \\
        --tls-ca   /etc/moonlab/ca.pem        \\
        --client-cert /etc/moonlab/scraper.pem \\
        --client-key  /etc/moonlab/scraper.key \\
        --listen 0.0.0.0:9090

Then point Prometheus at ``http://<host>:9090/metrics``.
"""

from __future__ import annotations

import argparse
import http.server
import logging
import socket
import ssl
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

LOG = logging.getLogger("moonlab.exporter")


@dataclass(frozen=True)
class TlsConfig:
    """Optional TLS scrape configuration.  ``ca_path`` is the CA bundle
    used to verify the server cert; the cert+key pair is sent to the
    server when present (mTLS).  ``insecure`` disables server-cert
    verification -- use for self-signed dev certs only."""
    ca_path: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    insecure: bool = False
    server_hostname: Optional[str] = None


def _build_ssl_context(cfg: TlsConfig) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    if cfg.ca_path:
        ctx.load_verify_locations(cafile=cfg.ca_path)
    if cfg.client_cert:
        ctx.load_cert_chain(certfile=cfg.client_cert, keyfile=cfg.client_key)
    if cfg.insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _read_until_newline(sock: socket.socket, cap: int = 4096) -> Tuple[bytes, bytes]:
    """Read until the first newline, return (line, leftover_bytes)."""
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(min(64, cap - len(buf)))
        if not chunk:
            raise IOError("control-plane closed during header read")
        buf += chunk
        if len(buf) >= cap:
            raise IOError(f"control-plane header exceeds {cap} bytes")
    line, _, rest = buf.partition(b"\n")
    return line, rest


def _recv_exact(sock: socket.socket, want: int, seeded: bytes = b"") -> bytes:
    """Receive exactly ``want`` bytes, accumulating onto ``seeded``."""
    buf = seeded
    while len(buf) < want:
        chunk = sock.recv(min(4096, want - len(buf)))
        if not chunk:
            raise IOError(
                f"control-plane closed during body read "
                f"({len(buf)} / {want} bytes)"
            )
        buf += chunk
    return buf[:want]


def scrape(host: str, port: int, timeout_secs: float,
           tls: Optional[TlsConfig] = None) -> bytes:
    """Send METRICS to the control plane and return the response body.

    On wire failure raises ``IOError``.  On a server-side rejection
    (ERR line) raises ``IOError`` with the rejection string."""
    raw = socket.create_connection((host, port), timeout=timeout_secs)
    sock: socket.socket
    if tls is not None:
        ctx = _build_ssl_context(tls)
        sock = ctx.wrap_socket(raw, server_hostname=tls.server_hostname or host)
    else:
        sock = raw
    try:
        sock.sendall(b"METRICS\n")
        line, rest = _read_until_newline(sock)
        if not line.startswith(b"METRICS "):
            raise IOError(f"control-plane returned: {line!r}")
        n = int(line[len(b"METRICS "):].decode("ascii"))
        body = _recv_exact(sock, n, seeded=rest)
        return body
    finally:
        try:
            sock.close()
        except OSError:
            pass


def make_handler(target_host: str, target_port: int, timeout_secs: float,
                 tls: Optional[TlsConfig] = None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (Python stdlib API)
            if self.path not in ("/metrics", "/"):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found\n")
                return
            try:
                body = scrape(target_host, target_port, timeout_secs, tls)
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
    parser.add_argument("--tls-ca", default=None,
                        help="CA bundle PEM for verifying the control-plane "
                             "server cert.  Enables TLS scraping.")
    parser.add_argument("--client-cert", default=None,
                        help="Client cert PEM for mTLS scraping.  Requires "
                             "--client-key.")
    parser.add_argument("--client-key", default=None,
                        help="Client private key PEM for mTLS scraping.  "
                             "Requires --client-cert.")
    parser.add_argument("--tls-insecure", action="store_true",
                        help="Skip server-cert verification.  Dev only.")
    parser.add_argument("--tls-server-name", default=None,
                        help="SNI / server-name override (defaults to "
                             "--target host).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    if (args.client_cert is None) != (args.client_key is None):
        parser.error("--client-cert and --client-key must be supplied together")

    tls = None
    if args.tls_ca or args.client_cert or args.tls_insecure:
        tls = TlsConfig(
            ca_path=args.tls_ca,
            client_cert=args.client_cert,
            client_key=args.client_key,
            insecure=args.tls_insecure,
            server_hostname=args.tls_server_name,
        )

    target_host, target_port = args.target
    listen_host, listen_port = args.listen

    handler = make_handler(target_host, target_port, args.timeout, tls)
    server = http.server.ThreadingHTTPServer(
        (listen_host, listen_port), handler)
    LOG.info("listening on http://%s:%d (target %s:%d, tls=%s)",
             listen_host, listen_port, target_host, target_port,
             "yes" if tls else "no")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutdown via SIGINT")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
