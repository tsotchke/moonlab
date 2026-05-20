"""Control-plane Python client -- since v0.8.8.

Talks to a `moonlab_control_serve()` server (v0.8.7 native) over a
plain TCP socket using the moonlab-circuit v1 wire format.  No third-
party dependencies -- stdlib socket / struct only.

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

import socket
import struct
from typing import List, Optional


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
                   timeout: Optional[float] = 30.0) -> List[float]:
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

    with socket.create_connection((host, port), timeout=timeout) as sock:
        header = f"CIRCUIT {len(encoded)}\n".encode("ascii")
        sock.sendall(header)
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
                         timeout: Optional[float] = 30.0) -> List[int]:
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

    with socket.create_connection((host, port), timeout=timeout) as sock:
        header = f"SHOTS {num_shots} {len(encoded)}\n".encode("ascii")
        sock.sendall(header)
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


__all__ = [
    "submit_circuit",
    "submit_circuit_shots",
    "ControlPlaneError",
]
