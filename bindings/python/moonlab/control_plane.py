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
import ssl
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


_TENANT_OK_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


def _validate_tenant_id(tenant_id: str) -> None:
    """Mirror the server-side rule: 1..63 chars, [A-Za-z0-9_.-]."""
    if not isinstance(tenant_id, str):
        raise ControlPlaneError(f"tenant_id must be str, got {type(tenant_id)!r}")
    if not (1 <= len(tenant_id) <= 63):
        raise ControlPlaneError(
            f"tenant_id length {len(tenant_id)} out of range [1, 63]"
        )
    for c in tenant_id:
        if c not in _TENANT_OK_CHARS:
            raise ControlPlaneError(
                f"tenant_id contains illegal char {c!r}; allowed [A-Za-z0-9_.-]"
            )


def submit_circuit(host: str,
                   port: int,
                   circuit_text: str,
                   timeout: Optional[float] = 30.0,
                   secret: Optional[Union[bytes, str]] = None,
                   tenant_id: Optional[str] = None) -> List[float]:
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
    tenant_id : str, optional
        Tenant identifier for multi-tenant deployments (since
        v1.0.3).  When set together with ``secret``, the prelude
        becomes ``AUTH <tenant_id>:<token>\n``; the server plumbs
        the tenant through to its scheduler completion hook (used
        by the overlay's billing / audit / quota layer).  Requires
        ``secret``; pass without it raises ``ControlPlaneError``.
        Characters restricted to ``[A-Za-z0-9_.-]``; length 1..63.

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

    if tenant_id is not None:
        if secret is None:
            raise ControlPlaneError(
                "tenant_id requires secret; the server uses HMAC to "
                "authenticate the request regardless of tenant identity"
            )
        _validate_tenant_id(tenant_id)

    with socket.create_connection((host, port), timeout=timeout) as sock:
        if secret is not None:
            key = secret.encode("utf-8") if isinstance(secret, str) else secret
            tok = _hmac_sha3_256(key, verb_line).hex()
            if tenant_id is not None:
                auth_line = f"AUTH {tenant_id}:{tok}\n".encode("ascii")
            else:
                auth_line = f"AUTH {tok}\n".encode("ascii")
            sock.sendall(auth_line)
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

# v0.8.25 extended server config -- probe defensively; binding still
# loads (with NotImplementedError on the methods) if the loaded
# libquantumsim predates v0.8.15/v0.8.17/v0.8.21.
try:
    _lib.moonlab_control_server_set_secret.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
    ]
    _lib.moonlab_control_server_set_secret.restype = ctypes.c_int
    _SECRET_API_AVAILABLE = True
except AttributeError:
    _SECRET_API_AVAILABLE = False

try:
    _lib.moonlab_control_server_use_tls.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    _lib.moonlab_control_server_use_tls.restype = ctypes.c_int
    _lib.moonlab_control_server_require_client_cert.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
    ]
    _lib.moonlab_control_server_require_client_cert.restype = ctypes.c_int
    _TLS_API_AVAILABLE = True
except AttributeError:
    _TLS_API_AVAILABLE = False

try:
    _lib.moonlab_control_server_set_rate_limit.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ]
    _lib.moonlab_control_server_set_rate_limit.restype = ctypes.c_int
    _RATE_LIMIT_API_AVAILABLE = True
except AttributeError:
    _RATE_LIMIT_API_AVAILABLE = False

try:
    _lib.moonlab_control_server_set_request_timeout.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
    ]
    _lib.moonlab_control_server_set_request_timeout.restype = ctypes.c_int
    _REQUEST_TIMEOUT_API_AVAILABLE = True
except AttributeError:
    _REQUEST_TIMEOUT_API_AVAILABLE = False

try:
    _lib.moonlab_control_server_set_max_concurrent.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
    ]
    _lib.moonlab_control_server_set_max_concurrent.restype = ctypes.c_int
    _MAX_CONCURRENT_API_AVAILABLE = True
except AttributeError:
    _MAX_CONCURRENT_API_AVAILABLE = False

# v1.0.3 admission hook -- private overlays implement per-tenant policy
# (quotas, paid-tier gating, lockouts) in pure Python and install it
# via ControlPlaneServer.set_admission_hook(callback).
_AdmissionHookCB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,   # tenant_id (NUL-terminated, may be NULL)
    ctypes.c_char_p,   # verb (CIRCUIT / SHOTS / ...)
    ctypes.c_int,      # num_qubits (-1 if unknown at admission time)
    ctypes.c_int,      # num_shots (0 if not a shots request)
    ctypes.c_void_p,   # ctx (we always pass NULL; the python closure
                       # holds whatever state the user wants)
)
try:
    _lib.moonlab_control_server_set_admission_hook.argtypes = [
        ctypes.c_void_p, _AdmissionHookCB, ctypes.c_void_p,
    ]
    _lib.moonlab_control_server_set_admission_hook.restype = ctypes.c_int
    _ADMISSION_HOOK_API_AVAILABLE = True
except AttributeError:
    _ADMISSION_HOOK_API_AVAILABLE = False


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
                 max_iters: int = (1 << 30),
                 secret: Optional[Union[bytes, str]] = None,
                 tls_cert: Optional[str] = None,
                 tls_key: Optional[str] = None,
                 client_ca: Optional[str] = None,
                 rate_limit_rps: int = 0,
                 rate_limit_burst: int = 0,
                 request_timeout_secs: int = 0,
                 max_concurrent: int = 0) -> None:
        """Spin up an in-process control-plane server.

        Optional kwargs (all applied before the worker thread starts
        accept()ing, so they're free of TOCTTOU races):

        - ``secret``: HMAC-SHA3-256 shared secret (v0.8.15+).
        - ``tls_cert`` + ``tls_key``: PEM paths for TLS (v0.8.17+).
        - ``client_ca``: enable mTLS; requires ``tls_cert``+``tls_key`` set.
        - ``rate_limit_rps`` / ``rate_limit_burst``: token bucket (v0.8.21+).
        """
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
        # v1.0.3 admission hook -- holds the live ctypes thunk so it
        # isn't GC'd while the server still has a function pointer to
        # it.  None until set_admission_hook() is called.
        self._admission_cb_ref = None
        # v0.8.25 config -- applied in __enter__ after open() but before
        # the runner thread starts.
        self._cfg_secret    = secret
        self._cfg_tls_cert  = tls_cert
        self._cfg_tls_key   = tls_key
        self._cfg_client_ca = client_ca
        self._cfg_rate_rps  = int(rate_limit_rps)
        self._cfg_burst     = int(rate_limit_burst)
        self._cfg_timeout   = int(request_timeout_secs)
        self._cfg_max_conc  = int(max_concurrent)

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

        # Apply v0.8.25 config before the runner starts -- avoids races
        # with the accept loop.
        if self._cfg_secret is not None:
            self.set_secret(self._cfg_secret)
        if self._cfg_tls_cert is not None or self._cfg_tls_key is not None:
            self.use_tls(self._cfg_tls_cert, self._cfg_tls_key)
        if self._cfg_client_ca is not None:
            self.require_client_cert(self._cfg_client_ca)
        if self._cfg_rate_rps > 0:
            self.set_rate_limit(self._cfg_rate_rps, self._cfg_burst)
        if self._cfg_timeout > 0:
            self.set_request_timeout(self._cfg_timeout)
        if self._cfg_max_conc > 0:
            self.set_max_concurrent(self._cfg_max_conc)

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

    def set_secret(self, secret: Union[bytes, str, None]) -> None:
        """Set / clear the HMAC-SHA3-256 shared secret (since v0.8.25).
        Pass ``None`` or an empty value to disable authentication.

        Must be called before any client connects.
        """
        if not _SECRET_API_AVAILABLE:
            raise ControlPlaneError(
                "set_secret unavailable -- libquantumsim predates v0.8.15"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        key = b"" if secret is None else (
            secret.encode("utf-8") if isinstance(secret, str) else secret
        )
        rc = _lib.moonlab_control_server_set_secret(
            ctypes.c_void_p(self._handle), key, len(key))
        if rc != 0:
            raise ControlPlaneError(f"set_secret rc={rc}")

    def use_tls(self,
                cert_path: Optional[str],
                key_path: Optional[str]) -> None:
        """Configure TLS (since v0.8.25 / v0.8.17 server).  Pass both
        ``None`` to disable an already-configured TLS server."""
        if not _TLS_API_AVAILABLE:
            raise ControlPlaneError(
                "use_tls unavailable -- libquantumsim predates v0.8.17 "
                "or was built without QSIM_ENABLE_TLS=ON"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        cp = None if cert_path is None else cert_path.encode("utf-8")
        kp = None if key_path  is None else key_path.encode("utf-8")
        rc = _lib.moonlab_control_server_use_tls(
            ctypes.c_void_p(self._handle), cp, kp)
        if rc != 0:
            raise ControlPlaneError(f"use_tls rc={rc}")

    def require_client_cert(self, ca_path: Optional[str]) -> None:
        """Require CA-signed client certificates for mTLS (since v0.8.25
        / v0.8.19 server).  Must be called after ``use_tls``.  Pass
        ``None`` to disable the requirement."""
        if not _TLS_API_AVAILABLE:
            raise ControlPlaneError(
                "require_client_cert unavailable -- libquantumsim predates "
                "v0.8.19 or was built without QSIM_ENABLE_TLS=ON"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        cp = None if ca_path is None else ca_path.encode("utf-8")
        rc = _lib.moonlab_control_server_require_client_cert(
            ctypes.c_void_p(self._handle), cp)
        if rc != 0:
            raise ControlPlaneError(f"require_client_cert rc={rc}")

    def set_request_timeout(self, timeout_secs: int) -> None:
        """Set per-request socket timeout in seconds (since v0.8.27 /
        v0.8.26 server).  0 disables the timeout (legacy default)."""
        if not _REQUEST_TIMEOUT_API_AVAILABLE:
            raise ControlPlaneError(
                "set_request_timeout unavailable -- libquantumsim "
                "predates v0.8.26"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        rc = _lib.moonlab_control_server_set_request_timeout(
            ctypes.c_void_p(self._handle), int(timeout_secs))
        if rc != 0:
            raise ControlPlaneError(f"set_request_timeout rc={rc}")

    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Cap the number of concurrent in-flight requests (since v0.9.0
        Python / v0.9.0 server).  0 disables the cap.  Excess clients
        receive `ERR -409 server busy`."""
        if not _MAX_CONCURRENT_API_AVAILABLE:
            raise ControlPlaneError(
                "set_max_concurrent unavailable -- libquantumsim "
                "predates v0.9.0"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        rc = _lib.moonlab_control_server_set_max_concurrent(
            ctypes.c_void_p(self._handle), int(max_concurrent))
        if rc != 0:
            raise ControlPlaneError(f"set_max_concurrent rc={rc}")

    def set_rate_limit(self, rate_rps: int, burst: int = 0) -> None:
        """Configure the per-source-IP token-bucket rate limit (since
        v0.8.25 / v0.8.21 server).  Pass ``rate_rps=0`` to disable.
        ``burst <= 0`` defaults to ``2 * rate_rps``."""
        if not _RATE_LIMIT_API_AVAILABLE:
            raise ControlPlaneError(
                "set_rate_limit unavailable -- libquantumsim predates v0.8.21"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        rc = _lib.moonlab_control_server_set_rate_limit(
            ctypes.c_void_p(self._handle), int(rate_rps), int(burst))
        if rc != 0:
            raise ControlPlaneError(f"set_rate_limit rc={rc}")

    def set_admission_hook(self, callback) -> None:
        """Install a per-tenant admission hook (since v1.0.3).

        The callback fires AFTER auth + verb parsing but BEFORE the
        body is read; returning 0 admits the request, returning a
        negative status code (e.g.  ``MOONLAB_CONTROL_RATE_LIMITED``
        ``-408``, ``MOONLAB_CONTROL_REJECTED`` ``-405``) refuses it
        with ``ERR <code> <msg>``.  Overlay-defined codes are
        allowed and sent verbatim to the client.

        Signature::

            callback(tenant_id: str | None, verb: str,
                     num_qubits: int, num_shots: int) -> int

        The callback fires on the worker thread and must be
        thread-safe.  ``tenant_id`` is ``None`` when the request
        used the legacy ``AUTH <hex>`` form; ``verb`` is one of
        ``"CIRCUIT"``, ``"SHOTS"``.  ``num_qubits`` is -1 when not
        known at admission time; ``num_shots`` is 0 on non-shots
        requests.

        Pass ``None`` to clear an installed hook.
        """
        if not _ADMISSION_HOOK_API_AVAILABLE:
            raise ControlPlaneError(
                "set_admission_hook unavailable -- libquantumsim predates v1.0.3"
            )
        if self._handle is None:
            raise ControlPlaneError("server not opened")
        if callback is None:
            # Install a null hook -- the C layer takes NULL as "no hook".
            # ctypes won't let us pass python None for a CFUNCTYPE parameter,
            # so cast a 0 function pointer through c_void_p.
            self._admission_cb_ref = None
            rc = _lib.moonlab_control_server_set_admission_hook(
                ctypes.c_void_p(self._handle),
                ctypes.cast(0, _AdmissionHookCB),
                None)
            if rc != 0:
                raise ControlPlaneError(f"set_admission_hook rc={rc}")
            return

        if not callable(callback):
            raise ControlPlaneError("callback must be callable or None")

        # Wrap the python callable so the C side sees the right shape.
        def _trampoline(tenant_id_p, verb_p, n_qubits, n_shots, _ctx):
            try:
                tid = tenant_id_p.decode("utf-8") if tenant_id_p else None
                verb = verb_p.decode("ascii") if verb_p else ""
                rc = callback(tid, verb, int(n_qubits), int(n_shots))
                return int(rc) if rc is not None else 0
            except Exception:
                # Don't let a python exception unwind into C; treat it
                # as a refusal so the server is not destabilised.
                return -405

        cb = _AdmissionHookCB(_trampoline)
        # Keep a reference so ctypes doesn't GC the thunk while the
        # server still holds a function pointer to it.
        self._admission_cb_ref = cb
        rc = _lib.moonlab_control_server_set_admission_hook(
            ctypes.c_void_p(self._handle), cb, None)
        if rc != 0:
            raise ControlPlaneError(f"set_admission_hook rc={rc}")

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


def submit_metrics(host: str,
                   port: int,
                   timeout: Optional[float] = 5.0) -> str:
    """Scrape the v0.8.23 METRICS endpoint and return the Prometheus
    text-format exposition body.

    No AUTH or TLS-cert required (monitoring scrapers run without
    credentials by convention).

    Raises :class:`ControlPlaneError` on transport failure or
    malformed response framing.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(b"METRICS\n")
        resp_hdr = _recv_until_newline(sock, cap=64)
        if not resp_hdr.startswith("METRICS "):
            raise ControlPlaneError(
                f"unrecognized METRICS response: {resp_hdr!r}"
            )
        try:
            num_bytes = int(resp_hdr[len("METRICS "):].strip())
        except ValueError as e:
            raise ControlPlaneError(
                f"malformed METRICS header: {resp_hdr!r}"
            ) from e
        if num_bytes <= 0 or num_bytes > (1 << 18):
            raise ControlPlaneError(f"implausible metrics size {num_bytes}")
        body = _recv_exact(sock, num_bytes)
    return body.decode("utf-8")


def submit_health(host: str,
                  port: int,
                  timeout: Optional[float] = 5.0) -> bool:
    """Submit the v0.8.21 HEALTH probe and return True if the server
    is alive.  Bypasses AUTH and TLS-cert layers so load-balancer
    probes work against any control plane.

    Raises :class:`ControlPlaneError` on transport failure;
    returns False if the server returned `ERR -408 rate limited`.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(b"HEALTH\n")
        resp = _recv_until_newline(sock, cap=128)
    if resp.startswith("OK alive"):
        return True
    if resp.startswith("ERR -408"):
        return False
    raise ControlPlaneError(f"unrecognized HEALTH response: {resp!r}")


def submit_circuit_tls(host: str,
                       port: int,
                       circuit_text: str,
                       ca_path: Optional[str] = None,
                       insecure: bool = False,
                       timeout: Optional[float] = 30.0,
                       secret: Optional[Union[bytes, str]] = None,
                       client_cert_path: Optional[str] = None,
                       client_key_path: Optional[str] = None) -> List[float]:
    """Submit a circuit over a TLS-wrapped TCP connection -- since v0.8.18.

    Mirrors the C ``moonlab_control_submit_circuit_tls`` API.

    Parameters
    ----------
    ca_path : str, optional
        PEM CA bundle to pin against the server's certificate.  Required
        for production deployments; pass ``insecure=True`` to skip
        verification (development / self-signed certs only).
    insecure : bool
        When True, peer verification is disabled.  Only safe for tests.
    secret : bytes or str, optional
        HMAC-SHA3-256 shared secret.  When provided, sends an
        ``AUTH <token>`` line inside the encrypted channel before the
        verb line.  Composes with the TLS wrapper -- the server must
        be configured with both ``use_tls`` and ``set_secret``.
    """
    encoded = circuit_text.encode("utf-8")
    verb_line = f"CIRCUIT {len(encoded)}\n".encode("ascii")

    if insecure:
        ctx = ssl._create_unverified_context()  # noqa: SLF001
    else:
        ctx = ssl.create_default_context(cafile=ca_path)

    # mTLS: present client cert when both files are provided (v0.8.20).
    if client_cert_path is not None or client_key_path is not None:
        if not (client_cert_path and client_key_path):
            raise ControlPlaneError(
                "client_cert_path and client_key_path must both be provided"
            )
        ctx.load_cert_chain(certfile=client_cert_path, keyfile=client_key_path)

    with socket.create_connection((host, port), timeout=timeout) as raw_sock:
        with ctx.wrap_socket(raw_sock, server_hostname=host) as sock:
            if secret is not None:
                key = secret.encode("utf-8") if isinstance(secret, str) else secret
                tok = _hmac_sha3_256(key, verb_line).hex()
                sock.sendall(f"AUTH {tok}\n".encode("ascii"))
            sock.sendall(verb_line)
            sock.sendall(encoded)

            resp_hdr = _recv_until_newline(sock)
            if resp_hdr.startswith("OK "):
                try:
                    num = int(resp_hdr[3:].strip())
                except ValueError as e:
                    raise ControlPlaneError(
                        f"malformed OK header: {resp_hdr!r}"
                    ) from e
                if num <= 0 or num > (1 << 30):
                    raise ControlPlaneError(f"implausible num_probs {num}")
                raw = _recv_exact(sock, num * 8)
                probs = struct.unpack(f"<{num}d", raw)
                return list(probs)
            elif resp_hdr.startswith("ERR "):
                raise ControlPlaneError(f"server rejected: {resp_hdr.strip()}")
            else:
                raise ControlPlaneError(f"unrecognized response: {resp_hdr!r}")


__all__ = [
    "submit_circuit",
    "submit_circuit_shots",
    "submit_circuit_tls",
    "submit_health",
    "submit_metrics",
    "ControlPlaneServer",
    "ControlPlaneError",
]
