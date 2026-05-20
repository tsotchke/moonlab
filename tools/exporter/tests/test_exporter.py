"""Tests for the moonlab control-plane Prometheus exporter sidecar.

Covers the wire-level scrape implementation and the HTTP handler.
TLS coverage uses a Python in-process TLS server -- the moonlab C
control plane is exercised by the separate
``bindings/python/tests/test_control_plane.py`` suite.
"""

from __future__ import annotations

import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Tuple

import pytest

# Make the sidecar importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import moonlab_control_exporter as ex  # noqa: E402


# ---------------------------------------------------------------------------
# Fake control plane (Python-only, no libquantumsim dependency)
# ---------------------------------------------------------------------------

class FakeControlPlane:
    """Listens on a chosen TCP port and replies to ``METRICS\\n`` with a
    canned body wrapped in the proper line-protocol framing.  Other
    verbs receive an ``ERR -400 bad verb`` line.  Optional TLS wrap
    via ``ssl_context``."""

    def __init__(self, body: bytes,
                 ssl_context: ssl.SSLContext | None = None,
                 reply_with_corrupt_header: bool = False) -> None:
        self.body = body
        self.ssl_context = ssl_context
        self.reply_with_corrupt_header = reply_with_corrupt_header
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port: int = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "FakeControlPlane":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        # Poke the accept loop so it observes the stop flag.
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                pass
        except OSError:
            pass
        self._thread.join(timeout=3)
        self._srv.close()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(2)
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(64)
                if not chunk:
                    return
                buf += chunk
            line, _, _ = buf.partition(b"\n")
            if line.strip() != b"METRICS":
                conn.sendall(b"ERR -400 bad verb\n")
                return
            if self.reply_with_corrupt_header:
                conn.sendall(b"NOT_THE_RIGHT_VERB junk\n")
                return
            header = f"METRICS {len(self.body)}\n".encode("ascii")
            conn.sendall(header + self.body)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _serve(self) -> None:
        self._srv.settimeout(0.2)
        while not self._stop.is_set():
            try:
                raw, _ = self._srv.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return
            if self.ssl_context is not None:
                try:
                    raw = self.ssl_context.wrap_socket(raw, server_side=True)
                except (ssl.SSLError, OSError):
                    continue
            self._handle(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANNED_METRICS = (
    "# HELP moonlab_control_requests_total Total control-plane requests.\n"
    "# TYPE moonlab_control_requests_total counter\n"
    'moonlab_control_requests_total{verb="CIRCUIT"} 7\n'
    'moonlab_control_requests_total{verb="HEALTH"} 3\n'
    "# TYPE moonlab_control_max_concurrent_rejected_total counter\n"
    "moonlab_control_max_concurrent_rejected_total 0\n"
).encode("utf-8")


def _gen_self_signed_cert(tmp_path: Path) -> Tuple[Path, Path]:
    """Generate a self-signed cert+key pair via openssl into tmp_path.

    Modern Python ssl requires an explicit Subject Alt Name (IP=...)
    for IP-address hostnames; a CN-only cert is no longer enough."""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert),
         "-days", "1",
         "-subj", "/CN=127.0.0.1",
         "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True,
    )
    return cert, key


# ---------------------------------------------------------------------------
# Scrape unit tests
# ---------------------------------------------------------------------------

def test_scrape_plain_returns_body():
    with FakeControlPlane(CANNED_METRICS) as srv:
        body = ex.scrape("127.0.0.1", srv.port, timeout_secs=2.0)
    assert body == CANNED_METRICS


def test_scrape_handles_large_body():
    # Force the recv loop to make multiple passes.
    big = (b"moonlab_control_test_total 1\n" * 5000)
    with FakeControlPlane(big) as srv:
        body = ex.scrape("127.0.0.1", srv.port, timeout_secs=2.0)
    assert body == big


def test_scrape_raises_on_corrupt_header():
    with FakeControlPlane(CANNED_METRICS, reply_with_corrupt_header=True) as srv:
        with pytest.raises(IOError, match="control-plane returned"):
            ex.scrape("127.0.0.1", srv.port, timeout_secs=2.0)


def test_scrape_raises_on_unreachable_server():
    # 1 is reserved (root) and connecting will fail fast.
    with pytest.raises(OSError):
        ex.scrape("127.0.0.1", 1, timeout_secs=0.5)


# ---------------------------------------------------------------------------
# TLS scrape (uses a Python SSL server -- no libquantumsim needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    subprocess.run(["which", "openssl"], capture_output=True).returncode != 0,
    reason="openssl CLI not available",
)
def test_scrape_tls_with_insecure_skip(tmp_path):
    cert, key = _gen_self_signed_cert(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))

    with FakeControlPlane(CANNED_METRICS, ssl_context=ctx) as srv:
        body = ex.scrape(
            "127.0.0.1", srv.port,
            timeout_secs=2.0,
            tls=ex.TlsConfig(insecure=True),
        )
    assert body == CANNED_METRICS


@pytest.mark.skipif(
    subprocess.run(["which", "openssl"], capture_output=True).returncode != 0,
    reason="openssl CLI not available",
)
def test_scrape_tls_with_explicit_ca(tmp_path):
    cert, key = _gen_self_signed_cert(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))

    with FakeControlPlane(CANNED_METRICS, ssl_context=ctx) as srv:
        body = ex.scrape(
            "127.0.0.1", srv.port,
            timeout_secs=2.0,
            tls=ex.TlsConfig(ca_path=str(cert)),
        )
    assert body == CANNED_METRICS


# ---------------------------------------------------------------------------
# End-to-end HTTP handler
# ---------------------------------------------------------------------------

def test_http_handler_returns_200_with_body():
    import http.server

    with FakeControlPlane(CANNED_METRICS) as srv:
        handler = ex.make_handler("127.0.0.1", srv.port, 2.0, None)
        http_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=3)
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/plain")
            assert resp.read() == CANNED_METRICS
        finally:
            http_srv.shutdown()
            http_srv.server_close()


def test_http_handler_returns_502_when_upstream_down():
    import http.server

    handler = ex.make_handler("127.0.0.1", 1, 0.5, None)
    http_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = http_srv.server_address[1]
    t = threading.Thread(target=http_srv.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=3)
        assert ei.value.code == 502
    finally:
        http_srv.shutdown()
        http_srv.server_close()


def test_http_handler_returns_404_on_unknown_path():
    import http.server

    with FakeControlPlane(CANNED_METRICS) as srv:
        handler = ex.make_handler("127.0.0.1", srv.port, 2.0, None)
        http_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/wat", timeout=3)
            assert ei.value.code == 404
        finally:
            http_srv.shutdown()
            http_srv.server_close()
