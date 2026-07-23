"""Moonlab error-mitigation bindings.

Pythonic access to the error-mitigation surface the C core ships:

- :class:`ZNE` -- zero-noise extrapolation.  You supply a callable that
  runs your circuit at a given noise scale and returns an expectation
  value; ZNE sweeps the requested scales and extrapolates the result to
  the zero-noise limit using the C estimator :c:func:`zne_extrapolate`
  (linear, Richardson, or exponential).
- :class:`PEC` -- probabilistic error cancellation.  Wraps the C
  quasi-probability primitives (:c:func:`pec_one_norm_cost`,
  :c:func:`pec_sample_index`, :c:func:`pec_aggregate`) so a signed
  Monte-Carlo estimator of a mitigated observable can be built from a
  quasi-probability decomposition.
- :class:`MeasurementMitigation` -- readout-error mitigation.  Calibrates
  the assignment (confusion) matrix by preparing each computational-basis
  state on the real simulator and sampling its measured distribution,
  then corrects raw shot counts by the (regularised) inverse.  With an
  ideal readout the assignment matrix is the identity and correction is a
  no-op; under a readout-noise model it removes the bias.

ZNE and PEC bind to the stable-ABI symbols declared in
``src/mitigation/zne.h`` (``@since 0.2.0``); the Python layer mirrors the
C signatures exactly.  MeasurementMitigation is real post-processing over
the measurement engine in :mod:`moonlab.core`.

Quick example -- zero-noise extrapolation::

    >>> from moonlab.error_mitigation import ZNE
    >>> def run_at(scale):
    ...     # your noisy circuit; return an expectation value
    ...     return 0.9 - 0.05 * scale
    >>> zne = ZNE(noise_factors=[1, 2, 3])
    >>> zne.extrapolate(run_at)          # doctest: +SKIP
    0.95
"""

from __future__ import annotations

import ctypes
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .core import _lib

__all__ = [
    "ZNE",
    "PEC",
    "MeasurementMitigation",
    "ZNE_LINEAR",
    "ZNE_RICHARDSON",
    "ZNE_EXPONENTIAL",
]

# Mirror of zne_method_t in src/mitigation/zne.h.
ZNE_LINEAR = 0
ZNE_RICHARDSON = 1
ZNE_EXPONENTIAL = 2

_METHOD_NAMES = {
    "linear": ZNE_LINEAR,
    "richardson": ZNE_RICHARDSON,
    "exponential": ZNE_EXPONENTIAL,
}


def _resolve_method(method) -> int:
    if isinstance(method, str):
        key = method.strip().lower()
        if key not in _METHOD_NAMES:
            raise ValueError(
                f"unknown ZNE method {method!r}; "
                f"expected one of {sorted(_METHOD_NAMES)} or an int code"
            )
        return _METHOD_NAMES[key]
    code = int(method)
    if code not in (ZNE_LINEAR, ZNE_RICHARDSON, ZNE_EXPONENTIAL):
        raise ValueError(f"unknown ZNE method code {code}")
    return code


# ---- C signature bindings (declared once, at import) --------------------

_lib.zne_extrapolate.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # scales
    ctypes.POINTER(ctypes.c_double),  # expectations
    ctypes.c_size_t,                  # n
    ctypes.c_int,                     # method
    ctypes.POINTER(ctypes.c_double),  # stderr_out (nullable)
]
_lib.zne_extrapolate.restype = ctypes.c_double

_lib.pec_one_norm_cost.argtypes = [ctypes.c_void_p]
_lib.pec_one_norm_cost.restype = ctypes.c_double

_lib.pec_sample_index.argtypes = [
    ctypes.c_void_p,                  # pec_quasi_prob_t*
    ctypes.c_double,                  # uniform
    ctypes.POINTER(ctypes.c_size_t),  # index_out
]
_lib.pec_sample_index.restype = ctypes.c_double

_lib.pec_aggregate.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # signs
    ctypes.POINTER(ctypes.c_double),  # measurements
    ctypes.c_size_t,                  # n
    ctypes.c_double,                  # gamma
    ctypes.POINTER(ctypes.c_double),  # stderr_out (nullable)
]
_lib.pec_aggregate.restype = ctypes.c_double


class _CPecQuasiProb(ctypes.Structure):
    """Mirror of pec_quasi_prob_t in src/mitigation/zne.h."""

    _fields_ = [
        ("num_terms", ctypes.c_size_t),
        ("etas", ctypes.POINTER(ctypes.c_double)),
    ]


# ---- Zero-noise extrapolation -------------------------------------------

class ZNE:
    """Zero-noise extrapolation.

    Parameters
    ----------
    noise_factors:
        The noise scales lambda_i to evaluate.  ``lambda = 1`` is the
        native (hardware/model) noise level; ``lambda > 1`` amplifies it.
        Must contain at least two distinct positive values.
    method:
        ``"linear"``, ``"richardson"`` (default), or ``"exponential"`` --
        or the corresponding integer code.
    """

    def __init__(self, noise_factors: Sequence[float], method="richardson"):
        scales = [float(x) for x in noise_factors]
        if len(scales) < 2:
            raise ValueError("ZNE needs at least two noise factors")
        if len(set(scales)) != len(scales):
            raise ValueError("ZNE noise factors must be distinct")
        if any(s <= 0.0 for s in scales):
            raise ValueError("ZNE noise factors must be positive")
        self.noise_factors: List[float] = scales
        self.method: int = _resolve_method(method)
        self.last_stderr: Optional[float] = None

    def extrapolate(self, circuit: Callable[[float], float]) -> float:
        """Sweep ``circuit`` across the noise factors and extrapolate to 0.

        ``circuit(noise_scale)`` must run the circuit at the given noise
        scale and return a real expectation value.  Returns the
        zero-noise estimate; :attr:`last_stderr` holds the fit residual
        standard deviation (0 for the Richardson estimator).
        """
        expectations = [float(circuit(s)) for s in self.noise_factors]
        return self.extrapolate_data(self.noise_factors, expectations)

    def extrapolate_data(
        self, scales: Sequence[float], expectations: Sequence[float]
    ) -> float:
        """Extrapolate an already-measured ``(scale, expectation)`` set."""
        scl = np.ascontiguousarray(scales, dtype=np.float64)
        exp = np.ascontiguousarray(expectations, dtype=np.float64)
        if scl.shape != exp.shape or scl.ndim != 1 or scl.size < 2:
            raise ValueError("scales and expectations must be equal-length 1-D (>=2)")
        stderr = ctypes.c_double(0.0)
        value = _lib.zne_extrapolate(
            scl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            exp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(scl.size),
            ctypes.c_int(self.method),
            ctypes.byref(stderr),
        )
        self.last_stderr = stderr.value
        if stderr.value < 0.0:
            raise ValueError("zne_extrapolate rejected its arguments")
        return float(value)


# ---- Probabilistic error cancellation -----------------------------------

class PEC:
    """Probabilistic error cancellation over a quasi-probability decomposition.

    A noise inverse is written as ``N^{-1} = sum_i eta_i U_i`` with signed
    ``eta_i``.  Construct with the signed coefficients; :meth:`sample`
    draws a basis index with probability ``|eta_i| / gamma`` and returns
    its sign, and :meth:`aggregate` turns a batch of (sign, measurement)
    pairs into an unbiased mitigated estimate.
    """

    def __init__(self, etas: Sequence[float]):
        self._etas = np.ascontiguousarray(etas, dtype=np.float64)
        if self._etas.ndim != 1 or self._etas.size == 0:
            raise ValueError("etas must be a non-empty 1-D array")
        self._qp = _CPecQuasiProb(
            num_terms=self._etas.size,
            etas=self._etas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    @property
    def one_norm_cost(self) -> float:
        """gamma = sum_i |eta_i|; the PEC sampling overhead is gamma^2."""
        return float(_lib.pec_one_norm_cost(ctypes.byref(self._qp)))

    def sample(self, uniform: float) -> tuple:
        """Draw an index with probability |eta_i|/gamma; return (index, sign)."""
        idx = ctypes.c_size_t(0)
        sign = _lib.pec_sample_index(
            ctypes.byref(self._qp), ctypes.c_double(float(uniform)), ctypes.byref(idx)
        )
        return int(idx.value), float(sign)

    def aggregate(
        self, signs: Sequence[float], measurements: Sequence[float]
    ) -> float:
        """Unbiased mitigated estimate from a batch of PEC samples."""
        s = np.ascontiguousarray(signs, dtype=np.float64)
        m = np.ascontiguousarray(measurements, dtype=np.float64)
        if s.shape != m.shape or s.ndim != 1 or s.size == 0:
            raise ValueError("signs and measurements must be equal-length 1-D")
        stderr = ctypes.c_double(0.0)
        value = _lib.pec_aggregate(
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(s.size),
            ctypes.c_double(self.one_norm_cost),
            ctypes.byref(stderr),
        )
        self.last_stderr = float(stderr.value)
        return float(value)


# ---- Readout-error (measurement) mitigation -----------------------------

class MeasurementMitigation:
    """Readout-error mitigation via the measured assignment matrix.

    :meth:`calibrate` prepares each computational-basis state on the real
    simulator and samples its measured distribution to build the
    assignment matrix ``A[measured, prepared] = P(measure | prepared)``.
    :meth:`correct` maps raw shot counts to corrected counts by applying
    the (Tikhonov-regularised) inverse of ``A`` and clipping to the
    probability simplex.  With an ideal readout ``A`` is the identity and
    correction is a no-op; under a readout-noise model it removes the bias.

    The full assignment matrix is ``2**num_qubits`` square, so this is
    intended for small registers (the default cap is 12 qubits); raise
    ``max_qubits`` deliberately if you have the memory.
    """

    def __init__(self, num_qubits: int, max_qubits: int = 12):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        if num_qubits > max_qubits:
            raise ValueError(
                f"{num_qubits} qubits needs a {2**num_qubits}x{2**num_qubits} "
                f"assignment matrix; raise max_qubits to opt in"
            )
        self.num_qubits = int(num_qubits)
        self.dim = 1 << self.num_qubits
        self.assignment: Optional[np.ndarray] = None

    def calibrate(self, shots: int = 10000) -> "MeasurementMitigation":
        """Build the assignment matrix by preparing and sampling each basis state."""
        from .core import QuantumState  # local import avoids a cycle at load

        if shots < 1:
            raise ValueError("shots must be >= 1")
        A = np.zeros((self.dim, self.dim), dtype=np.float64)
        for prepared in range(self.dim):
            counts = np.zeros(self.dim, dtype=np.float64)
            for _ in range(shots):
                state = QuantumState(self.num_qubits)
                for q in range(self.num_qubits):
                    if (prepared >> q) & 1:
                        state.x(q)
                measured = state.measure_all_fast()
                counts[measured] += 1.0
            A[:, prepared] = counts / float(shots)
        self.assignment = A
        return self

    def correct(self, raw_counts: Mapping) -> Dict[str, float]:
        """Correct raw shot counts (dict of bitstring/int -> count)."""
        if self.assignment is None:
            raise RuntimeError("call calibrate() before correct()")
        observed = np.zeros(self.dim, dtype=np.float64)
        for key, count in raw_counts.items():
            observed[self._key_to_index(key)] += float(count)
        total = observed.sum()
        if total <= 0.0:
            raise ValueError("raw_counts is empty")

        # Tikhonov-regularised least squares: minimise ||A x - b|| with a
        # small ridge so a near-singular assignment matrix stays invertible.
        ridge = 1e-9 * np.trace(self.assignment) / self.dim
        ata = self.assignment.T @ self.assignment + ridge * np.eye(self.dim)
        corrected = np.linalg.solve(ata, self.assignment.T @ observed)

        # Project onto the non-negative simplex scaled to the shot total.
        corrected = np.clip(corrected, 0.0, None)
        s = corrected.sum()
        if s > 0.0:
            corrected *= total / s
        return {
            format(i, f"0{self.num_qubits}b"): float(c)
            for i, c in enumerate(corrected)
            if c > 0.0
        }

    def _key_to_index(self, key) -> int:
        if isinstance(key, str):
            idx = int(key, 2)
        else:
            idx = int(key)
        if not (0 <= idx < self.dim):
            raise ValueError(f"count key {key!r} out of range for {self.num_qubits} qubits")
        return idx
