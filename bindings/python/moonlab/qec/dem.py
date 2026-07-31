"""Stim detector error models and moonlab's union-find decoder.

Wraps ``src/qec/stim_dem.h``.  A :class:`DetectorErrorModel` is a DEM
imported into moonlab's edge-list form: each graphlike component of each
``error`` mechanism is an edge, parallel mechanisms are merged with
``p = p1(1-p2) + p2(1-p1)`` (what PyMatching does internally), and the
``^`` decompositions Stim emits with ``decompose_errors=True`` become
correlation links that :meth:`DetectorErrorModel.make_decoder` feeds to
the two-pass correlated decoder.

Export runs the same merge backwards, so an edge list survives a round
trip through DEM text and the text loads in Stim and PyMatching.

Example:
    >>> from moonlab.qec import DetectorErrorModel
    >>> dem = DetectorErrorModel.from_text('''
    ...     error(0.01) D0 D1 L0
    ...     error(0.01) D1
    ... ''')
    >>> dem.num_edges
    2
    >>> decoder = dem.make_decoder(correlated=False)
    >>> import numpy as np
    >>> decoder.decode_batch(np.array([[1], [1]], dtype=np.uint8))
    array([[1]], dtype=uint8)
"""

from __future__ import annotations

import ctypes
import os
from typing import Tuple

import numpy as np

from ._ffi import (
    UF_BOUNDARY,
    MoonlabStimError,
    as_ptr,
    check_status,
    lib,
    take_text,
)

__all__ = ["DetectorErrorModel", "UFDecoder", "text_from_edges", "UF_BOUNDARY"]

_COORD_CHUNK = 16


class UFDecoder:
    """moonlab's union-find decoder over a detector error model.

    Built by :meth:`DetectorErrorModel.make_decoder`.  Decoding is
    per-shot independent, so batches are split across threads.
    """

    __slots__ = ("_handle", "_num_detectors", "_num_observables", "_source")

    def __init__(self, handle, num_detectors: int, num_observables: int,
                 source=None) -> None:
        if not handle:
            raise MemoryError("moonlab_dem_make_uf_decoder returned NULL")
        self._handle = handle
        self._num_detectors = int(num_detectors)
        self._num_observables = int(num_observables)
        # Keep the model that produced us alive for the decoder's lifetime.
        self._source = source

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            lib.moonlab_uf_decoder_free(handle)
            self._handle = None

    def __repr__(self) -> str:
        return (f"UFDecoder(detectors={self._num_detectors}, "
                f"observables={self._num_observables}, "
                f"edges={self.num_edges})")

    @property
    def num_detectors(self) -> int:
        return self._num_detectors

    @property
    def num_observables(self) -> int:
        return self._num_observables

    @property
    def num_edges(self) -> int:
        return int(lib.moonlab_uf_decoder_num_edges(self._handle))

    def decode_batch(self, det: np.ndarray, threads: int = 0) -> np.ndarray:
        """Decode a detector-major batch.

        Args:
            det: ``(num_detectors, shots)`` uint8 detection events.
            threads: OpenMP block count; <= 0 selects all cores.

        Returns:
            ``(num_observables, shots)`` uint8 predicted observable flips.
        """
        det = np.ascontiguousarray(det, dtype=np.uint8)
        if det.ndim != 2:
            raise ValueError(
                f"det must be 2-D (num_detectors, shots), got shape {det.shape}")
        if det.shape[0] != self._num_detectors:
            raise ValueError(
                f"det has {det.shape[0]} detector rows, decoder expects "
                f"{self._num_detectors}")
        shots = int(det.shape[1])
        out = np.zeros((self._num_observables, shots), dtype=np.uint8)
        if shots == 0:
            return out
        rc = lib.moonlab_uf_decode_batch(
            self._handle, as_ptr(det, ctypes.c_uint8), shots, int(threads),
            as_ptr(out, ctypes.c_uint8))
        if int(rc) != shots:
            raise RuntimeError(
                f"moonlab_uf_decode_batch returned {int(rc)}, expected {shots}")
        return out

    def decode_shots_bit_packed(self, bit_packed_detection_event_data,
                                threads: int = 0) -> np.ndarray:
        """Decode sinter's bit-packed shot layout.

        Args:
            bit_packed_detection_event_data: ``(shots, ceil(ndet / 8))``
                uint8, little-endian bit order -- sinter's contract.

        Returns:
            ``(shots, ceil(nobs / 8))`` uint8, packed the same way.
        """
        data = np.asarray(bit_packed_detection_event_data, dtype=np.uint8)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError(
                "bit_packed_detection_event_data must be 1-D or 2-D, got "
                f"shape {data.shape}")
        shots = int(data.shape[0])
        n_det = self._num_detectors
        n_obs = self._num_observables
        if data.shape[1] * 8 < n_det:
            raise ValueError(
                f"bit-packed data has {data.shape[1]} bytes per shot, too "
                f"few for {n_det} detectors")
        if shots == 0:
            return np.zeros((0, (n_obs + 7) // 8), dtype=np.uint8)
        # (shots, ndet) -> detector-major (ndet, shots) -> decode ->
        # (nobs, shots) -> shot-major (shots, nobs) -> repack.
        bits = np.unpackbits(data, axis=1, count=n_det, bitorder="little")
        det = np.ascontiguousarray(bits.T)
        obs = self.decode_batch(det, threads=threads)
        obs_shot_major = np.ascontiguousarray(obs.T)
        if n_obs == 0:
            return np.zeros((shots, 0), dtype=np.uint8)
        return np.packbits(obs_shot_major, axis=1, bitorder="little")


class DetectorErrorModel:
    """A Stim detector error model in moonlab's edge-list form."""

    __slots__ = ("_handle",)

    def __init__(self, text: str) -> None:
        err = MoonlabStimError()
        handle = lib.moonlab_dem_parse(text.encode("utf-8"), ctypes.byref(err))
        if not handle:
            raise err.as_exception("moonlab_dem_parse failed")
        self._handle = handle

    # -- construction -------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "DetectorErrorModel":
        """Parse DEM text (``str(stim_dem)`` works directly)."""
        return cls(text)

    @classmethod
    def from_file(cls, path) -> "DetectorErrorModel":
        """Parse a `.dem` file from disk."""
        err = MoonlabStimError()
        handle = lib.moonlab_dem_parse_file(
            os.fsencode(path), ctypes.byref(err))
        if not handle:
            raise err.as_exception(f"could not read {path!s}")
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    @classmethod
    def from_edges(cls,
                   num_detectors: int,
                   num_observables: int,
                   edge_a,
                   edge_b,
                   edge_prob,
                   edge_obs,
                   corr_a=None,
                   corr_b=None,
                   corr_joint_p=None) -> "DetectorErrorModel":
        """Build a model from an edge list.

        Serialises through :func:`text_from_edges` and re-parses, so the
        result is exactly what a Stim or PyMatching consumer would see.
        Each correlation link ``(u, v, q)`` becomes one decomposed
        ``error(q) <u> ^ <v>`` mechanism and every edge carries the
        residual probability left after peeling its links off, which
        makes ``edges -> text -> parse -> edges`` a fixed point.
        """
        text = text_from_edges(num_detectors, num_observables,
                               edge_a, edge_b, edge_prob, edge_obs,
                               corr_a, corr_b, corr_joint_p)
        return cls(text)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            lib.moonlab_dem_free(handle)
            self._handle = None

    # -- serialisation ------------------------------------------------

    def to_text(self) -> str:
        """Serialise back to DEM text that Stim and PyMatching accept."""
        return take_text(lib.moonlab_dem_to_text(self._handle))

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return (f"DetectorErrorModel(detectors={self.num_detectors}, "
                f"observables={self.num_observables}, "
                f"edges={self.num_edges}, "
                f"correlations={self.num_correlations}, "
                f"hyperedges={self.num_hyperedges})")

    # -- introspection ------------------------------------------------

    @property
    def num_detectors(self) -> int:
        return int(lib.moonlab_dem_num_detectors(self._handle))

    @property
    def num_observables(self) -> int:
        return int(lib.moonlab_dem_num_observables(self._handle))

    @property
    def num_edges(self) -> int:
        return int(lib.moonlab_dem_num_edges(self._handle))

    @property
    def num_correlations(self) -> int:
        return int(lib.moonlab_dem_num_correlations(self._handle))

    @property
    def num_hyperedges(self) -> int:
        """Components with more than two detectors: not graphlike.

        Counted rather than silently dropped.  A well decomposed Stim
        DEM (``decompose_errors=True``) reports 0.
        """
        return int(lib.moonlab_dem_num_hyperedges(self._handle))

    @property
    def num_detectorless(self) -> int:
        """Components that flip observables without lighting a detector."""
        return int(lib.moonlab_dem_num_detectorless(self._handle))

    def edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray]:
        """Copy out the edge list.

        Returns ``(edge_a, edge_b, edge_weight, edge_obs, edge_prob)``
        where ``edge_b`` is :data:`UF_BOUNDARY` for a boundary edge,
        ``edge_weight`` is the log-likelihood ratio ``ln((1-p)/p)``, and
        ``edge_obs`` is a uint64 observable bitmask.
        """
        n = self.num_edges
        ea = np.zeros(n, dtype=np.uint32)
        eb = np.zeros(n, dtype=np.uint32)
        ew = np.zeros(n, dtype=np.float64)
        eo = np.zeros(n, dtype=np.uint64)
        ep = np.zeros(n, dtype=np.float64)
        rc = lib.moonlab_dem_edges(
            self._handle,
            as_ptr(ea, ctypes.c_uint32), as_ptr(eb, ctypes.c_uint32),
            as_ptr(ew, ctypes.c_double), as_ptr(eo, ctypes.c_uint64),
            as_ptr(ep, ctypes.c_double), n)
        check_status(rc, "moonlab_dem_edges")
        return ea, eb, ew, eo, ep

    def correlations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy out the mechanism correlation links.

        Returns ``(corr_a, corr_b, corr_joint_p)``: edge-index pairs and
        the combined probability that the mechanism linking them fires.
        """
        n = self.num_correlations
        ca = np.zeros(n, dtype=np.uint32)
        cb = np.zeros(n, dtype=np.uint32)
        cq = np.zeros(n, dtype=np.float64)
        rc = lib.moonlab_dem_correlations(
            self._handle,
            as_ptr(ca, ctypes.c_uint32), as_ptr(cb, ctypes.c_uint32),
            as_ptr(cq, ctypes.c_double), n)
        check_status(rc, "moonlab_dem_correlations")
        return ca, cb, cq

    def detector_coords(self, detector: int) -> np.ndarray:
        """Coordinates from a ``detector(...)`` instruction."""
        buf = np.zeros(_COORD_CHUNK, dtype=np.float64)
        n = lib.moonlab_dem_detector_coords(
            self._handle, detector, as_ptr(buf, ctypes.c_double), buf.size)
        if n < 0:
            raise IndexError(f"detector index {detector} out of range")
        if n > buf.size:
            buf = np.zeros(int(n), dtype=np.float64)
            n = lib.moonlab_dem_detector_coords(
                self._handle, detector, as_ptr(buf, ctypes.c_double), buf.size)
            if n < 0:
                raise IndexError(f"detector index {detector} out of range")
        return buf[:int(n)].copy()

    # -- decoding -----------------------------------------------------

    def make_decoder(self, correlated: bool = True) -> UFDecoder:
        """Build a :class:`UFDecoder` straight from this model.

        ``correlated=True`` selects the two-pass decoder, which consumes
        the links Stim's ``^`` decompositions carry.
        """
        handle = lib.moonlab_dem_make_uf_decoder(
            self._handle, 1 if correlated else 0)
        return UFDecoder(handle, self.num_detectors, self.num_observables,
                         source=self)


def text_from_edges(num_detectors: int,
                    num_observables: int,
                    edge_a,
                    edge_b,
                    edge_prob,
                    edge_obs,
                    corr_a=None,
                    corr_b=None,
                    corr_joint_p=None) -> str:
    """Serialise an edge list straight to DEM text.

    The exact inverse of the import merge: each correlation link emits
    one decomposed mechanism and each edge emits the residual left after
    peeling every link touching it off its probability.
    """
    ea = np.ascontiguousarray(edge_a, dtype=np.uint32)
    eb = np.ascontiguousarray(edge_b, dtype=np.uint32)
    ep = np.ascontiguousarray(edge_prob, dtype=np.float64)
    eo = np.ascontiguousarray(edge_obs, dtype=np.uint64)
    n_edges = ea.size
    if not (eb.size == ep.size == eo.size == n_edges):
        raise ValueError(
            f"edge arrays disagree: a={ea.size} b={eb.size} "
            f"prob={ep.size} obs={eo.size}")

    if corr_a is None and corr_b is None and corr_joint_p is None:
        ca = cb = cq = None
        n_corr = 0
    else:
        ca = np.ascontiguousarray(
            [] if corr_a is None else corr_a, dtype=np.uint32)
        cb = np.ascontiguousarray(
            [] if corr_b is None else corr_b, dtype=np.uint32)
        cq = np.ascontiguousarray(
            [] if corr_joint_p is None else corr_joint_p, dtype=np.float64)
        if not (ca.size == cb.size == cq.size):
            raise ValueError(
                f"correlation arrays disagree: a={ca.size} b={cb.size} "
                f"q={cq.size}")
        n_corr = ca.size

    err = MoonlabStimError()
    ptr = lib.moonlab_dem_text_from_edges(
        int(num_detectors), int(num_observables),
        as_ptr(ea, ctypes.c_uint32), as_ptr(eb, ctypes.c_uint32),
        as_ptr(ep, ctypes.c_double), as_ptr(eo, ctypes.c_uint64), n_edges,
        as_ptr(ca, ctypes.c_uint32), as_ptr(cb, ctypes.c_uint32),
        as_ptr(cq, ctypes.c_double), n_corr,
        ctypes.byref(err))
    if not ptr:
        raise err.as_exception("moonlab_dem_text_from_edges failed")
    return take_text(ptr)
