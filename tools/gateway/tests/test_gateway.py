"""Tests for the moonlab WebSocket-to-line-protocol gateway."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import moonlab_websocket_gateway as gw  # noqa: E402


class FakeControlPlane:
    """Speaks the moonlab line protocol on plain TCP for the gateway."""

    def __init__(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port: int = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        # Captures the AUTH line of every request that started with one,
        # so tests can assert the gateway emitted the right wire form.
        self.auth_lines: list[str] = []

    def __enter__(self) -> "FakeControlPlane":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
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
            verb_line, _, rest = buf.partition(b"\n")
            text = verb_line.decode("ascii", errors="replace")
            # v1.0.3: AUTH prelude.  Capture for assertions, advance to
            # the next newline-delimited line as the actual verb.
            if text.startswith("AUTH "):
                self.auth_lines.append(text)
                buf = rest
                while b"\n" not in buf:
                    chunk = conn.recv(64)
                    if not chunk:
                        return
                    buf += chunk
                verb_line, _, rest = buf.partition(b"\n")
                text = verb_line.decode("ascii", errors="replace")
            if text == "HEALTH":
                conn.sendall(b"OK alive\n")
                return
            if text == "METRICS":
                body = b"moonlab_control_requests_total{verb=\"HEALTH\"} 1\n"
                conn.sendall(f"METRICS {len(body)}\n".encode("ascii") + body)
                return
            if text.startswith("CIRCUIT "):
                buf2 = bytearray()
                for v in (0.5, 0.0, 0.0, 0.5):
                    buf2 += struct.pack("<d", v)
                conn.sendall(b"OK 4\n" + bytes(buf2))
                return
            if text.startswith("SHOTS "):
                buf2 = bytearray()
                for v in (500, 0, 0, 500):
                    buf2 += struct.pack("<Q", v)
                conn.sendall(b"SAMPLES 4\n" + bytes(buf2))
                return
            conn.sendall(b"ERR -400 unknown verb\n")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _serve(self) -> None:
        self._srv.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return
            self._handle(conn)


@pytest.fixture
def gateway_pair():
    """Start a fake control plane + the gateway in a dedicated event loop."""
    pytest.importorskip("websockets")
    import websockets as ws_lib

    with FakeControlPlane() as cp:
        target = ("127.0.0.1", cp.port)

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        async def start():
            async def handler(ws, *_):
                await gw._serve(ws, target, timeout=2.0)
            srv = await ws_lib.serve(handler, "127.0.0.1", 0)
            return srv

        srv_handle = asyncio.run_coroutine_threadsafe(start(), loop).result()
        actual_port = srv_handle.sockets[0].getsockname()[1]

        try:
            yield actual_port, cp
        finally:
            async def shutdown():
                srv_handle.close()
                await srv_handle.wait_closed()
            try:
                asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=3)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=3)


async def _send(port: int, payload: dict) -> dict:
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
        await ws.send(json.dumps(payload))
        reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    return reply


def test_health_round_trip(gateway_pair):
    port, _ = gateway_pair
    reply = asyncio.run(_send(port, {"verb": "HEALTH"}))
    assert reply == {"status": "OK", "code": 0, "message": "alive"}


def test_metrics_round_trip(gateway_pair):
    port, _ = gateway_pair
    reply = asyncio.run(_send(port, {"verb": "METRICS"}))
    assert reply["status"] == "OK"
    assert "moonlab_control_requests_total" in reply["body"]


def test_circuit_round_trip(gateway_pair):
    port, _ = gateway_pair
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
    }))
    assert reply["status"] == "OK"
    assert reply["probs"] == [0.5, 0.0, 0.0, 0.5]


def test_shots_round_trip(gateway_pair):
    port, _ = gateway_pair
    reply = asyncio.run(_send(port, {
        "verb": "SHOTS",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "shots": 1000,
    }))
    assert reply["status"] == "OK"
    assert reply["counts"] == [500, 0, 0, 500]


def test_tenant_form_auth_propagates_through_gateway(gateway_pair):
    """v1.0.3: browser request with `tenant` field -> gateway emits
    `AUTH <tenant>:<hmac>\\n` upstream.  Verify the fake control
    plane saw the exact wire form, including a 64-hex HMAC."""
    port, cp = gateway_pair
    import re
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "secret": "shared-key-bytes".encode("utf-8").hex(),
        "tenant": "acme-corp",
    }))
    assert reply["status"] == "OK", reply
    assert reply["probs"] == [0.5, 0.0, 0.0, 0.5]
    assert len(cp.auth_lines) == 1
    assert re.fullmatch(r"AUTH acme-corp:[0-9a-f]{64}", cp.auth_lines[0]), \
        f"unexpected upstream AUTH line: {cp.auth_lines[0]!r}"


def test_tenant_without_secret_rejected_clientside(gateway_pair):
    port, cp = gateway_pair
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "tenant": "acme-corp",
    }))
    assert reply["status"] == "ERR"
    assert reply["code"] == -400
    assert "secret" in reply["message"].lower()
    assert cp.auth_lines == [], "gateway should not have contacted upstream"


def test_tenant_illegal_char_rejected_clientside(gateway_pair):
    port, cp = gateway_pair
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "secret": "shared-key-bytes".encode("utf-8").hex(),
        "tenant": "acme;rm -rf /",
    }))
    assert reply["status"] == "ERR"
    assert reply["code"] == -400
    assert cp.auth_lines == []


def test_tenant_oversize_rejected_clientside(gateway_pair):
    port, cp = gateway_pair
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "secret": "shared-key-bytes".encode("utf-8").hex(),
        "tenant": "x" * 64,
    }))
    assert reply["status"] == "ERR"
    assert reply["code"] == -400
    assert cp.auth_lines == []


def test_legacy_auth_form_when_no_tenant(gateway_pair):
    """Sanity: with `secret` but no `tenant`, gateway emits the
    legacy `AUTH <hex>\\n` form, not the tenant variant."""
    port, cp = gateway_pair
    import re
    reply = asyncio.run(_send(port, {
        "verb": "CIRCUIT",
        "circuit": "# moonlab-circuit v1\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n",
        "secret": "shared-key-bytes".encode("utf-8").hex(),
    }))
    assert reply["status"] == "OK"
    assert len(cp.auth_lines) == 1
    assert re.fullmatch(r"AUTH [0-9a-f]{64}", cp.auth_lines[0]), \
        f"unexpected upstream AUTH line: {cp.auth_lines[0]!r}"


def test_unknown_verb_returns_400(gateway_pair):
    port, _ = gateway_pair
    reply = asyncio.run(_send(port, {"verb": "BANANA"}))
    assert reply["status"] == "ERR"
    assert reply["code"] == -400


def test_bad_json_returns_400(gateway_pair):
    port, _ = gateway_pair
    import websockets

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
            await ws.send("not-json")
            return json.loads(await ws.recv())
    reply = asyncio.run(go())
    assert reply["status"] == "ERR"
    assert reply["code"] == -400
