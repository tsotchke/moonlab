#!/usr/bin/env python3
"""Browser-facing WebSocket gateway for the Moonlab control plane.

Bridges the moonlab line protocol over a WebSocket so browser
clients (which cannot open raw TCP sockets) can submit circuits to
a moonlab control plane.

Protocol over the WebSocket: one JSON object per request, one JSON
object per reply.  The gateway translates to/from the moonlab line
protocol on the TCP side.

Request envelope::

    { "verb": "CIRCUIT" | "SHOTS" | "HEALTH" | "METRICS",
      "circuit": "<moonlab-circuit-v1 text>",
      "shots":   <int>,             // SHOTS verb only
      "secret":  "<hex-bytes>"      // optional HMAC-SHA3 shared secret
    }

Reply envelope::

    { "status":  "OK" | "ERR",
      "code":    <int>,
      "message": "<str>",
      "probs":   [<float>, ...],     // CIRCUIT path
      "counts":  [<int>, ...],       // SHOTS path
      "body":    "<str>"             // METRICS path
    }

Usage::

    moonlab-websocket-gateway --target 127.0.0.1:7070 --listen 0.0.0.0:8765

Then from a browser::

    const ws = new WebSocket("ws://gateway:8765/");
    ws.onopen = () => ws.send(JSON.stringify({verb: "CIRCUIT", circuit: text}));
    ws.onmessage = (e) => { ... JSON.parse(e.data).probs ... };
"""

from __future__ import annotations

import argparse
import asyncio
import binascii
import hashlib
import hmac
import json
import logging
import socket
import struct
from typing import Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:  # pragma: no cover -- runtime requirement
    raise SystemExit(
        "moonlab-websocket-gateway requires the `websockets` package.\n"
        "  pip install websockets"
    )

LOG = logging.getLogger("moonlab.gateway")


# ---------------------------------------------------------------------------
# Moonlab line-protocol client (synchronous; called inside thread executor)
# ---------------------------------------------------------------------------

def _hmac_sha3_256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, digestmod=hashlib.sha3_256).digest()


def _recv_until_newline(sock: socket.socket, cap: int = 4096) -> bytes:
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(64)
        if not chunk:
            raise IOError("control-plane closed during header read")
        buf += chunk
        if len(buf) > cap:
            raise IOError(f"header exceeds {cap} bytes")
    return buf


def _recv_exact(sock: socket.socket, want: int, seed: bytes) -> bytes:
    buf = seed
    while len(buf) < want:
        chunk = sock.recv(min(4096, want - len(buf)))
        if not chunk:
            raise IOError(
                f"control-plane closed during body read ({len(buf)}/{want})"
            )
        buf += chunk
    return buf[:want]


def _line_submit(target_host: str, target_port: int, timeout: float,
                 verb_line: bytes, body: bytes = b"",
                 secret: bytes | None = None) -> tuple[str, bytes, str]:
    """Open a TCP connection, send the verb (with optional AUTH prelude)
    and body, return (header_verb, body_bytes, remainder_after_verb)."""
    with socket.create_connection((target_host, target_port), timeout=timeout) as s:
        if secret is not None:
            tok = _hmac_sha3_256(secret, verb_line).hex()
            s.sendall(f"AUTH {tok}\n".encode("ascii"))
        s.sendall(verb_line)
        if body:
            s.sendall(body)

        head = _recv_until_newline(s)
        line, _, rest = head.partition(b"\n")
        text = line.decode("ascii", errors="replace")

        parts = text.split(" ", 1)
        head_verb = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        if head_verb in ("OK", "SAMPLES", "METRICS"):
            try:
                n = int(remainder)
            except ValueError:
                return head_verb, b"", remainder
            body_bytes = _recv_exact(s, n if head_verb == "METRICS" else n * 8, rest)
            return head_verb, body_bytes, remainder
        return head_verb, b"", remainder


# ---------------------------------------------------------------------------
# Request handlers (called per WebSocket message)
# ---------------------------------------------------------------------------

def _parse_secret(raw: Any) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return binascii.unhexlify(raw)
        except binascii.Error:
            return raw.encode("utf-8")
    if isinstance(raw, list):
        return bytes(raw)
    raise ValueError(f"unsupported secret type: {type(raw).__name__}")


def _handle_circuit(target: tuple[str, int], timeout: float, req: dict) -> dict:
    text = req.get("circuit", "")
    body = text.encode("utf-8")
    secret = _parse_secret(req.get("secret"))
    verb = f"CIRCUIT {len(body)}\n".encode("ascii")
    head, payload, rest = _line_submit(target[0], target[1], timeout, verb, body, secret)
    if head == "OK":
        num = len(payload) // 8
        probs = list(struct.unpack(f"<{num}d", payload))
        return {"status": "OK", "code": 0, "probs": probs}
    return _err_reply(head, rest)


def _handle_shots(target: tuple[str, int], timeout: float, req: dict) -> dict:
    text = req.get("circuit", "")
    shots = int(req.get("shots", 0))
    if shots <= 0:
        return {"status": "ERR", "code": -400, "message": "shots must be > 0"}
    body = text.encode("utf-8")
    secret = _parse_secret(req.get("secret"))
    verb = f"SHOTS {shots} {len(body)}\n".encode("ascii")
    head, payload, rest = _line_submit(target[0], target[1], timeout, verb, body, secret)
    if head == "SAMPLES":
        num = len(payload) // 8
        counts = list(struct.unpack(f"<{num}Q", payload))
        return {"status": "OK", "code": 0, "counts": counts}
    return _err_reply(head, rest)


def _handle_health(target: tuple[str, int], timeout: float, req: dict) -> dict:
    head, _, rest = _line_submit(target[0], target[1], timeout, b"HEALTH\n")
    if head == "OK":
        return {"status": "OK", "code": 0, "message": "alive"}
    return _err_reply(head, rest)


def _handle_metrics(target: tuple[str, int], timeout: float, req: dict) -> dict:
    head, payload, rest = _line_submit(target[0], target[1], timeout, b"METRICS\n")
    if head == "METRICS":
        return {"status": "OK", "code": 0, "body": payload.decode("utf-8", "replace")}
    return _err_reply(head, rest)


def _err_reply(head: str, rest: str) -> dict:
    """Translate an `ERR <code> <msg>` server reply into a JSON envelope."""
    if head != "ERR":
        return {
            "status": "ERR", "code": -403,
            "message": f"unexpected framing: {head}",
        }
    parts = rest.split(" ", 1)
    try:
        code = int(parts[0])
    except (ValueError, IndexError):
        code = -405
    msg = parts[1] if len(parts) > 1 else ""
    return {"status": "ERR", "code": code, "message": msg}


HANDLERS = {
    "CIRCUIT": _handle_circuit,
    "SHOTS":   _handle_shots,
    "HEALTH":  _handle_health,
    "METRICS": _handle_metrics,
}


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def _serve(ws, target: tuple[str, int], timeout: float):
    peer = ws.remote_address if ws.remote_address else "?"
    LOG.info("client connected: %s", peer)
    try:
        async for msg in ws:
            try:
                req = json.loads(msg)
            except json.JSONDecodeError as e:
                await ws.send(json.dumps({
                    "status": "ERR", "code": -400,
                    "message": f"bad json: {e}",
                }))
                continue
            verb = req.get("verb", "").upper()
            handler = HANDLERS.get(verb)
            if not handler:
                await ws.send(json.dumps({
                    "status": "ERR", "code": -400,
                    "message": f"unknown verb: {verb!r}",
                }))
                continue
            loop = asyncio.get_running_loop()
            try:
                reply = await loop.run_in_executor(
                    None, handler, target, timeout, req)
            except Exception as exc:
                LOG.warning("%s: upstream error: %s", verb, exc)
                reply = {
                    "status": "ERR", "code": -403,
                    "message": f"upstream: {exc}",
                }
            await ws.send(json.dumps(reply))
    except ConnectionClosed:
        pass
    finally:
        LOG.info("client disconnected: %s", peer)


def parse_endpoint(arg: str) -> tuple[str, int]:
    host, _, port = arg.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(
            f"expected host:port, got {arg!r}")
    return host, int(port)


async def main_async(args):
    target = args.target
    timeout = args.timeout
    listen_host, listen_port = args.listen

    LOG.info("gateway listening on ws://%s:%d (target %s:%d)",
             listen_host, listen_port, target[0], target[1])

    async def handler(ws, *_):
        await _serve(ws, target, timeout)

    async with websockets.serve(handler, listen_host, listen_port):
        await asyncio.Future()  # serve forever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=parse_endpoint, required=True,
                        help="Moonlab control endpoint, host:port")
    parser.add_argument("--listen", type=parse_endpoint,
                        default=("0.0.0.0", 8765),
                        help="WebSocket bind, host:port (default 0.0.0.0:8765)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-request line-protocol socket timeout in seconds")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("shutdown via SIGINT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
