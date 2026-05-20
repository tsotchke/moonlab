"""Control-plane Python client -- since v0.8.8.

Talks to a `moonlab_control_serve()` server (v0.8.7 native) over a
plain TCP socket using the moonlab-circuit v1 wire format.  No third-
party dependencies -- stdlib socket / struct only.

HMAC-SHA3-256 authentication (since v0.8.16 / v0.8.15 server) uses
``hashlib.sha3_256`` (Python 3.6+ stdlib).

Example:

    >>> from moonlab.qgtl import QgtlCircuit, GateType
    >>> from moonlab.control_plane import submit_circuit
    >>> c = QgtlCircuit(num_qubits=2)
    >>> c.add_gate(GateType.H, target=0)
    >>> c.add_gate(GateType.CNOT, target=1, control=0)
    >>> probs = submit_circuit("127.0.0.1", 8765, c.serialize())
    >>> probs
    [0.5, 0.0, 0.0, 0.5]
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac as _stdlib_hmac
import socket
import struct
import threading
from typing import List, Optional, Union

from .core import _lib


_HMAC_BLOCK_SIZE  = 136      # SHA3-256 rate.
_HMAC_DIGEST_SIZE = 32


def _hmac_sha3_256(key: bytes, msg: bytes) -> bytes:
    """HMAC-SHA3-256 per FIPS 198.

    Stdlib `hmac.new(...)` defaults to MD5 in older Python; we feed in
    `digestmod=hashlib.sha3_256` and rely on the well-tested stdlib
    HMAC construction.  Block size = 136 bytes (SHA3-256 rate).
    """
    h = _stdlib_hmac.new(key, msg, digestmod=hashlib.sha3_256)
    return h.digest()


class ControlPlaneError(RuntimeError):
    """Server-side rejection or transport failure."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly `n` bytes or raise."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ControlPlaneError(
                f"short read: expected {n} bytes, got {n - remaining}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_until_newline(sock: socket.socket, cap: int = 4096) -> str:
    """Read bytes until '\\n' (included) or cap; decode as ASCII."""
    out = bytearray()
    while len(out) < cap:
        c = sock.recv(1)
        if not c:
            raise ControlPlaneError("short read: peer closed before newline")
        out.append(c[0])
        if c == b"\n":
            break
    return out.decode("ascii", errors="replace")


def submit_circuit(host: str,
                   port: int,
                   circuit_text: str,
                   timeout: Optional[float] = 30.0,
                   secret: Optional[Union[bytes, str]] = None) -> List[float]:
    """Submit a moonlab-circuit v1 text payload to a control-plane
    server at ``host:port`` and return the probability vector.

    Parameters
    ----------
    host : str
        Server hostname or IPv4 literal.
    port : int
        TCP port.
    circuit_text : str
        Serialized moonlab-circuit v1 text (from
        :meth:`moonlab.qgtl.QgtlCircuit.serialize`).
    timeout : float, optional
        Socket timeout in seconds.  Default 30 s.  Pass ``None`` for
        blocking I/O.
    secret : bytes or str, optional
        HMAC-SHA3-256 shared secret (since v0.8.16).  When provided,
        sends an ``AUTH <token>`` prelude before the verb line; the
        server must be configured with the same secret via
        ``moonlab_control_server_set_secret``.

    Returns
    -------
    list[float]
        Length-``2^num_qubits`` probability vector.

    Raises
    ------
    ControlPlaneError
        Server returned ERR, the connection was closed prematurely, or
        the response framing was malformed.
    """
    encoded = circuit_text.encode("utf-8")
    verb_line = f"CIRCUIT {len(encoded)}\n".encode("ascii")

    with socket.create_connection((host, port), timeout=timeout) as sock:
        if secret is not None:
            key = secret.encode("utf-8") if isinstance(secret, str) else secret
            tok = _hmac_sha3_256(key, verb_line).hex()
            sock.sendall(f"AUTH {tok}\n".encode("ascii"))
        sock.sendall(verb_line)
        sock.sendall(encoded)

        resp_hdr = _recv_until_newline(sock)
        if resp_hdr.startswith("OK "):
            try:
                num_probs = int(resp_hdr[3:].strip())
            except ValueError as e:
                raise ControlPlaneError(
                    f"malformed OK header: {resp_hdr!r}"
                ) from e
            if num_probs <= 0 or num_probs > (1 << 30):
                raise ControlPlaneError(
                    f"implausible num_probs {num_probs} in OK header"
                )
            raw = _recv_exact(sock, num_probs * 8)
            # Little-endian f64 -- matches the C server's `send_all` of
            # `double *probabilities` on every host moonlab supports.
            probs = struct.unpack(f"<{num_probs}d", raw)
            return list(probs)
        elif resp_hdr.startswith("ERR "):
            raise ControlPlaneError(f"server rejected: {resp_hdr.strip()}")
        else:
            raise ControlPlaneError(f"unrecognized response: {resp_hdr!r}")


def submit_circuit_shots(host: str,
                         port: int,
                         circuit_text: str,
                         num_shots: int,
                         timeout: Optional[float] = 30.0,
                         secret: Optional[Union[bytes, str]] = None) -> List[int]:
    """Submit a moonlab-circuit v1 payload requesting `num_shots`
    measurement samples instead of the full probability vector.

    Returns a list of integer bitstring outcomes -- bit 0 is qubit 0,
    bit 1 is qubit 1, ...  Use ``outcome & (1 << k)`` to test qubit k.

    Wire format: ``SHOTS <num_shots> <bytes>\\n<circuit-text>`` ->
    ``SAMPLES <num_shots>\\n<num_shots * 8-byte LE uint64>``.
    Since v0.8.12 (Python) / v0.8.11 (C server).
    """
    if num_shots <= 0 or num_shots > (1 << 20):
        raise ControlPlaneError(
            f"num_shots {num_shots} out of range [1, 2^20]"
        )

    encoded = circuit_text.encode("utf-8")
    verb_line = f"SHOTS {num_shots} {len(encoded)}\n".encode("ascii")

    with socket.create_connection((host, port), timeout=timeout) as sock:
        if secret is not None:
            key = secret.encode("utf-8") if isinstance(secret, str) else secret
            tok = _hmac_sha3_256(key, verb_line).hex()
            sock.sendall(f"AUTH {tok}\n".encode("ascii"))
        sock.sendall(verb_line)
        sock.sendall(encoded)

        resp_hdr = _recv_until_newline(sock)
        if resp_hdr.startswith("SAMPLES "):
            try:
                shots_back = int(resp_hdr[len("SAMPLES "):].strip())
            except ValueError as e:
                raise ControlPlaneError(
                    f"malformed SAMPLES header: {resp_hdr!r}"
                ) from e
            if shots_back <= 0 or shots_back > (1 << 20):
                raise ControlPlaneError(
                    f"implausible shots_back {shots_back}"
                )
            raw = _recv_exact(sock, shots_back * 8)
            outcomes = struct.unpack(f"<{shots_back}Q", raw)
            return list(outcomes)
        elif resp_hdr.startswith("ERR "):
            raise ControlPlaneError(f"server rejected: {resp_hdr.strip()}")
        else:
            raise ControlPlaneError(f"unrecognized response: {resp_hdr!r}")


# ----- v0.8.14 server lifecycle binding -----
# Probe at module load -- a libquantumsim older than v0.8.13 doesn't
# export these, in which case ControlPlaneServer raises a clear error.
try:
    _lib.moonlab_control_server_open.argtypes = [
        ctypes.c_char_p, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint16),
    ]
    _lib.moonlab_control_server_open.restype = ctypes.c_int
    _lib.moonlab_control_server_run.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.moonlab_control_server_run.restype = ctypes.c_int
    _lib.moonlab_control_server_shutdown.argtypes = [ctypes.c_void_p]
    _lib.moonlab_control_server_shutdown.restype = None
    _lib.moonlab_control_server_close.argtypes = [ctypes.c_void_p]
    _lib.moonlab_control_server_close.restype = None
    _SERVER_API_AVAILABLE = True
except AttributeError:
    _SERVER_API_AVAILABLE = False


class ControlPlaneServer:
    """In-process control-plane server, since v0.8.14.

    Wraps the C lifecycle API.  Use as a context manager so the
    server is opened on enter, run on a background daemon thread,
    and shut down + joined on exit.  The bound port is available on
    the ``port`` attribute as soon as ``__enter__`` returns.

    Example::

        from moonlab.qgtl import QgtlCircuit, GateType
        from moonlab.control_plane import (
            ControlPlaneServer, submit_circuit,
        )

        with ControlPlaneServer(host="127.0.0.1", port=0) as srv:
            c = QgtlCircuit(num_qubits=2)
            c.add_gate(GateType.H, target=0)
            c.add_gate(GateType.CNOT, target=1, control=0)
            probs = submit_circuit("127.0.0.1", srv.port, c.serialize())
            assert abs(probs[0] - 0.5) < 1e-9
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 max_iters: int = (1 << 30)) -> None:
        if not _SERVER_API_AVAILABLE:
            raise ControlPlaneError(
                "ControlPlaneServer unavailable -- libquantumsim is older "
                "than v0.8.13 (no moonlab_control_server_* exports)"
            )
        self._host = host.encode("utf-8")
        self._port_in = int(port)
        self._max_iters = int(max_iters)
        self._handle: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._actual_port = ctypes.c_uint16(0)
        self._run_rc: Optional[int] = None

    @property
    def port(self) -> int:
        """Bound TCP port (the OS-chosen one when constructed with port=0)."""
        return int(self._actual_port.value)

    def __enter__(self) -> "ControlPlaneServer":
        handle = ctypes.c_void_p(0)
        rc = _lib.moonlab_control_server_open(
            self._host, self._port_in,
            ctypes.byref(handle), ctypes.byref(self._actual_port))
        if rc != 0 or not handle.value:
            raise ControlPlaneError(f"server_open rc={rc}")
        self._handle = handle.value

        def runner() -> None:
            self._run_rc = _lib.moonlab_control_server_run(
                ctypes.c_void_p(self._handle), self._max_iters)

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """Signal the server to stop after the current in-flight request."""
        if self._handle is not None:
            _lib.moonlab_control_server_shutdown(ctypes.c_void_p(self._handle))

    def __exit__(self, *exc) -> None:
        self.shutdown()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._handle is not None:
            _lib.moonlab_control_server_close(ctypes.c_void_p(self._handle))
            self._handle = None
        if self._run_rc is not None and self._run_rc != 0:
            raise ControlPlaneError(f"server_run rc={self._run_rc}")


__all__ = [
    "submit_circuit",
    "submit_circuit_shots",
    "ControlPlaneServer",
    "ControlPlaneError",
]
