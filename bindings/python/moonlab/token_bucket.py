"""Thread-safe token-bucket rate limiter (since v1.0.3).

Native python port of ``src/utils/token_bucket.{c,h}``; semantics
match the C primitive's CHECK suite at the byte level so that an
overlay can use it in its admission_hook callback without crossing
the FFI boundary.

Typical use inside an admission hook::

    from moonlab.token_bucket import TokenBucket
    from moonlab.control_plane import ControlPlaneServer

    buckets = {}  # tenant_id -> TokenBucket

    def admission(tenant_id, verb, num_qubits, num_shots):
        if tenant_id is None:
            return -405  # require AUTH <tenant>:<hmac>
        b = buckets.setdefault(
            tenant_id, TokenBucket(burst=1000, refill_per_sec=100))
        if not b.take(num_shots if num_shots > 0 else 1):
            return -408  # MOONLAB_CONTROL_RATE_LIMITED
        return 0

    with ControlPlaneServer(...) as srv:
        srv.set_admission_hook(admission)
        ...
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Lock-free-style token bucket.

    All public methods are thread-safe.  Operations serialise on a
    single ``threading.Lock`` rather than relying on python's GIL
    quirks; this matches the lock-free CAS loop in the C primitive
    closely enough that the semantics agree byte-for-byte on the
    test cases.

    Parameters
    ----------
    burst : int
        Maximum tokens the bucket can hold.  Must be > 0.
    refill_per_sec : int
        Tokens added per wall-clock second.  Pass 0 to disable
        time-based refill (one-shot budget; replenish via
        ``refill()``).
    """

    __slots__ = ("_burst", "_refill", "_tokens", "_last", "_lock")

    def __init__(self, burst: int, refill_per_sec: int) -> None:
        if burst <= 0:
            raise ValueError(f"burst must be > 0, got {burst}")
        if refill_per_sec < 0:
            raise ValueError(
                f"refill_per_sec must be >= 0, got {refill_per_sec}")
        self._burst = float(burst)
        self._refill = float(refill_per_sec)
        # Start the bucket FULL, matching the C primitive.
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _accrue_locked(self, now: float) -> None:
        """Lazily add tokens for elapsed wall-clock time."""
        if self._refill == 0.0:
            return
        dt = now - self._last
        if dt <= 0.0:
            return
        self._tokens = min(self._burst, self._tokens + dt * self._refill)
        self._last = now

    def take(self, n: int = 1) -> bool:
        """Remove ``n`` tokens if available.  Returns True on
        success, False on insufficient balance (no tokens removed)."""
        if n <= 0:
            return True   # zero-cost takes are always allowed
        with self._lock:
            now = time.monotonic()
            self._accrue_locked(now)
            if self._tokens < n:
                return False
            self._tokens -= n
            return True

    def refill(self, n: int) -> None:
        """Add ``n`` tokens, clipped at ``burst``."""
        if n <= 0:
            return
        with self._lock:
            self._tokens = min(self._burst, self._tokens + float(n))

    def peek(self) -> int:
        """Return the current (lazily refilled) integer balance."""
        with self._lock:
            now = time.monotonic()
            self._accrue_locked(now)
            return int(self._tokens)
