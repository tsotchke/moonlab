"""Stim `.stim` circuit format: parse, serialise, lower, sample.

Wraps ``src/qec/stim_circuit.h``.  A :class:`StimCircuit` is a parsed
Stim circuit that can be re-serialised, lowered to moonlab's Pauli-frame
op list, queried for its detector / observable parity sets, and sampled
directly -- so an existing Stim/Sinter/PyMatching harness can be pointed
at moonlab without a translation layer.

Example:
    >>> from moonlab.qec import StimCircuit
    >>> c = StimCircuit.from_text('''
    ...     R 0 1
    ...     X_ERROR(0.01) 0
    ...     CX 0 1
    ...     M 0 1
    ...     DETECTOR rec[-1] rec[-2]
    ... ''')
    >>> c.num_detectors
    1
    >>> det, obs = c.sample_detectors(1000, seed=7)
    >>> det.shape
    (1, 1000)
"""

from __future__ import annotations

import ctypes
import os
from typing import Tuple

import numpy as np

from ._ffi import (
    PF_OP_DTYPE,
    SIZE_T_DTYPE,
    MoonlabStimError,
    PFCircuitOp,
    as_ptr,
    check_status,
    lib,
    take_text,
)

__all__ = ["StimCircuit"]

_COORD_CHUNK = 16


class StimCircuit:
    """A parsed Stim circuit.

    Construct with :meth:`from_text` / :meth:`from_file` (or by passing
    circuit source straight to the constructor).  The handle is released
    when the object is collected.
    """

    __slots__ = ("_handle",)

    def __init__(self, text: str) -> None:
        err = MoonlabStimError()
        handle = lib.moonlab_stim_circuit_parse(
            text.encode("utf-8"), ctypes.byref(err))
        if not handle:
            raise err.as_exception("moonlab_stim_circuit_parse failed")
        self._handle = handle

    # -- construction -------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "StimCircuit":
        """Parse `.stim` circuit source."""
        return cls(text)

    @classmethod
    def from_file(cls, path) -> "StimCircuit":
        """Parse a `.stim` file from disk."""
        err = MoonlabStimError()
        handle = lib.moonlab_stim_circuit_parse_file(
            os.fsencode(path), ctypes.byref(err))
        if not handle:
            raise err.as_exception(f"could not read {path!s}")
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            lib.moonlab_stim_circuit_free(handle)
            self._handle = None

    # -- serialisation ------------------------------------------------

    def to_text(self, flatten: bool = False) -> str:
        """Serialise back to `.stim` text in Stim's canonical spelling.

        ``flatten=False`` keeps REPEAT blocks as parsed; ``flatten=True``
        expands them.  Both re-parse to a semantically identical circuit.
        """
        return take_text(lib.moonlab_stim_circuit_to_text(
            self._handle, 1 if flatten else 0))

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return (f"StimCircuit(qubits={self.num_qubits}, "
                f"measurements={self.num_measurements}, "
                f"detectors={self.num_detectors}, "
                f"observables={self.num_observables})")

    # -- introspection ------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """One past the largest qubit index the circuit touches."""
        return int(lib.moonlab_stim_circuit_num_qubits(self._handle))

    @property
    def num_measurements(self) -> int:
        """Measurement record length, counting MPAD entries."""
        return int(lib.moonlab_stim_circuit_num_measurements(self._handle))

    @property
    def num_detectors(self) -> int:
        return int(lib.moonlab_stim_circuit_num_detectors(self._handle))

    @property
    def num_observables(self) -> int:
        """One past the largest OBSERVABLE_INCLUDE index."""
        return int(lib.moonlab_stim_circuit_num_observables(self._handle))

    @property
    def num_ticks(self) -> int:
        return int(lib.moonlab_stim_circuit_num_ticks(self._handle))

    def _coords(self, fn, index: int, what: str) -> np.ndarray:
        buf = np.zeros(_COORD_CHUNK, dtype=np.float64)
        n = fn(self._handle, index, as_ptr(buf, ctypes.c_double), buf.size)
        if n < 0:
            raise IndexError(f"{what} index {index} out of range")
        if n > buf.size:
            buf = np.zeros(int(n), dtype=np.float64)
            n = fn(self._handle, index, as_ptr(buf, ctypes.c_double), buf.size)
            if n < 0:
                raise IndexError(f"{what} index {index} out of range")
        return buf[:int(n)].copy()

    def qubit_coords(self, qubit: int) -> np.ndarray:
        """QUBIT_COORDS for ``qubit`` (empty when none were declared)."""
        return self._coords(
            lib.moonlab_stim_circuit_qubit_coords, qubit, "qubit")

    def detector_coords(self, detector: int) -> np.ndarray:
        """DETECTOR coordinates for ``detector`` (SHIFT_COORDS folded in)."""
        return self._coords(
            lib.moonlab_stim_circuit_detector_coords, detector, "detector")

    # -- lowering -----------------------------------------------------

    @property
    def num_ops(self) -> int:
        """Number of ``pf_circuit_op_t`` entries the lowering produces."""
        return int(check_status(
            lib.moonlab_stim_circuit_num_ops(self._handle),
            "moonlab_stim_circuit_num_ops"))

    @property
    def num_channel_args(self) -> int:
        """Number of doubles the lowering's channel-argument table needs."""
        return int(check_status(
            lib.moonlab_stim_circuit_num_channel_args(self._handle),
            "moonlab_stim_circuit_num_channel_args"))

    def lower(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lower to the flat Pauli-frame op list.

        REPEAT blocks are expanded.  Returns ``(ops, chan_args)`` where
        ``ops`` is a structured array with fields ``kind``, ``q0``,
        ``q1``, ``p`` laid out exactly as ``pf_circuit_op_t``, and
        ``chan_args`` holds the PAULI_CHANNEL_1 / _2 probability table
        that PF_OP_PAULI_CHANNEL ops index through their ``p`` field.

        The dtype carries ``pf_circuit_op_t``'s interior padding so the
        array can be handed straight back to the C sampler; compare op
        lists field by field rather than through ``tobytes()``, since
        the padding bytes are not part of the op.
        """
        n_ops = self.num_ops
        n_chan = self.num_channel_args
        ops = np.zeros(n_ops, dtype=PF_OP_DTYPE)
        chan = np.zeros(n_chan, dtype=np.float64)
        err = MoonlabStimError()
        rc = lib.moonlab_stim_circuit_lower(
            self._handle,
            ops.ctypes.data_as(ctypes.POINTER(PFCircuitOp)), n_ops,
            as_ptr(chan, ctypes.c_double), n_chan,
            ctypes.byref(err))
        if rc < 0:
            raise err.as_exception("moonlab_stim_circuit_lower failed")
        if int(rc) != n_ops:
            raise RuntimeError(
                f"moonlab_stim_circuit_lower wrote {int(rc)} ops, "
                f"num_ops reported {n_ops}")
        return ops, chan

    def _csr(self, fn, count: int, what: str) -> Tuple[np.ndarray, np.ndarray]:
        n_idx = int(check_status(fn(self._handle, None, 0, None, 0), what))
        offsets = np.zeros(count + 1, dtype=SIZE_T_DTYPE)
        indices = np.zeros(n_idx, dtype=np.uint32)
        rc = fn(self._handle,
                as_ptr(offsets, ctypes.c_size_t), offsets.size,
                as_ptr(indices, ctypes.c_uint32), indices.size)
        check_status(rc, what)
        return offsets, indices

    def detector_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Detector parity sets over measurement-record indices.

        Returns ``(offsets, indices)``; detector ``d`` covers
        ``indices[offsets[d]:offsets[d + 1]]``.
        """
        return self._csr(lib.moonlab_stim_circuit_detector_csr,
                         self.num_detectors,
                         "moonlab_stim_circuit_detector_csr")

    def observable_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Logical-observable parity sets, laid out as :meth:`detector_csr`."""
        return self._csr(lib.moonlab_stim_circuit_observable_csr,
                         self.num_observables,
                         "moonlab_stim_circuit_observable_csr")

    def measurement_inversions(self) -> np.ndarray:
        """One byte per measurement record: 1 when Stim's `!` inverts it."""
        n = self.num_measurements
        out = np.zeros(n, dtype=np.uint8)
        rc = lib.moonlab_stim_circuit_measurement_inversions(
            self._handle, as_ptr(out, ctypes.c_uint8), out.size)
        check_status(rc, "moonlab_stim_circuit_measurement_inversions")
        return out

    # -- sampling -----------------------------------------------------

    def sample_measurements(self, shots: int, seed: int = 0,
                            threads: int = 0) -> np.ndarray:
        """Sample raw measurement records.

        Returns a ``(num_measurements, shots)`` uint8 array
        (measurement-major, matching the C buffer layout).  Stim's `!`
        inversion mask is already applied.
        """
        shots = int(shots)
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")
        n_meas = self.num_measurements
        out = np.zeros((n_meas, shots), dtype=np.uint8)
        if shots == 0 or n_meas == 0:
            return out
        rc = lib.moonlab_stim_circuit_sample_measurements(
            self._handle, shots, ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
            int(threads), as_ptr(out, ctypes.c_uint8))
        check_status(rc, "moonlab_stim_circuit_sample_measurements")
        return out

    def sample_detectors(self, shots: int, seed: int = 0,
                         threads: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Sample detectors and logical observables.

        Mirrors Stim's
        ``compile_detector_sampler().sample(..., separate_observables=True)``
        transposed: returns ``(detectors, observables)`` with shapes
        ``(num_detectors, shots)`` and ``(num_observables, shots)``.
        """
        shots = int(shots)
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")
        n_det = self.num_detectors
        n_obs = self.num_observables
        det = np.zeros((n_det, shots), dtype=np.uint8)
        obs = np.zeros((n_obs, shots), dtype=np.uint8)
        if shots == 0:
            return det, obs
        rc = lib.moonlab_stim_circuit_sample_detectors(
            self._handle, shots, ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
            int(threads),
            as_ptr(det, ctypes.c_uint8) if n_det else None,
            as_ptr(obs, ctypes.c_uint8) if n_obs else None)
        check_status(rc, "moonlab_stim_circuit_sample_detectors")
        return det, obs
