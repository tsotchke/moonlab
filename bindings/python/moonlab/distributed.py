"""Moonlab distributed (MPI) simulation bindings.

Backs the distributed-simulation API taught in
``documents/guides/distributed-simulation.md`` with the real MPI/distributed
C engine (``src/distributed/``).  The state vector is partitioned across MPI
ranks; gates on high-order ("partition") qubits trigger inter-rank exchange,
while measurement and expectation values are computed via MPI collectives.

Public surface
--------------
- :func:`init_mpi` -- initialise the MPI bridge and return a
  :class:`DistributedContext`.  Also records it as the process-wide default
  context used by :class:`DistributedState` when none is passed explicitly.
- :func:`finalize_mpi` -- free the default context and finalise MPI.
- :func:`is_mpi_available` -- ``True`` iff the linked libquantumsim was built
  with ``-DQSIM_ENABLE_MPI=ON`` (i.e. the distributed symbols are present).
- :class:`DistributedContext` -- thin wrapper over ``distributed_ctx_t*``
  exposing ``rank``, ``size``, ``is_root`` and ``processor_name``.
- :class:`DistributedState` -- a partitioned state vector supporting the
  gate / measurement / expectation surface the guide teaches.

Build reality
-------------
The distributed symbols (``mpi_bridge_init``, ``partition_state_create``,
``dist_hadamard``, ``collective_measure_all`` ...) are compiled into
libquantumsim ONLY when it is built with MPI support.  A default CPU wheel
has none of them.  This module always imports; :func:`init_mpi` and
:class:`DistributedState` raise an informative :class:`RuntimeError` when the
symbols are absent.

Matching the guide's Python example::

    from moonlab.distributed import DistributedState, init_mpi, finalize_mpi

    init_mpi()
    state = DistributedState(num_qubits=32)
    state.h(0)
    state.cnot(0, 1)
    result = state.measure_all()
    if state.rank == 0:
        print(f"Measurement result: {result}")
    finalize_mpi()

Run under MPI with, e.g., ``mpirun -np 4 python my_sim.py``.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from .core import _lib

__all__ = [
    "DistributedContext",
    "DistributedState",
    "MpiUnavailableError",
    "init_mpi",
    "finalize_mpi",
    "is_mpi_available",
]


# ---------------------------------------------------------------- #
# MPI-availability detection.                                       #
# ---------------------------------------------------------------- #
#
# The entire src/distributed/ translation unit is excluded from the build
# unless QSIM_ENABLE_MPI=ON, so the mere presence of mpi_bridge_init in the
# loaded library is a reliable signal that the distributed engine is linked
# in.  We probe the small set of entry points this module actually calls so
# a partially-built library fails at import-binding time rather than at first
# use with a cryptic AttributeError.
_REQUIRED_SYMBOLS = (
    "mpi_bridge_init",
    "mpi_bridge_init_no_args",
    "mpi_bridge_finalize",
    "mpi_bridge_free",
    "mpi_get_rank",
    "mpi_get_size",
    "mpi_is_root",
    "mpi_get_processor_name",
    "mpi_bridge_error_string",
    "partition_state_create",
    "partition_state_free",
    "dist_hadamard",
    "dist_pauli_x",
    "dist_pauli_y",
    "dist_pauli_z",
    "dist_rx",
    "dist_ry",
    "dist_rz",
    "dist_phase",
    "dist_s_gate",
    "dist_t_gate",
    "dist_cnot",
    "dist_cz",
    "dist_cphase",
    "dist_swap",
    "dist_iswap",
    "dist_gate_error_string",
    "collective_measure_all",
    "collective_measure_qubit",
    "collective_sample",
    "collective_sample_many",
    "collective_get_probability",
    "collective_get_qubit_probability",
    "collective_expectation_x",
    "collective_expectation_y",
    "collective_expectation_z",
    "collective_expectation_pauli",
    "collective_correlation_zz",
    "collective_error_string",
)


def _detect_mpi() -> bool:
    for name in _REQUIRED_SYMBOLS:
        if not hasattr(_lib, name):
            return False
    return True


_HAS_MPI = _detect_mpi()

_NO_MPI_MESSAGE = (
    "MoonLab was built without MPI support; the distributed engine "
    "(mpi_bridge_init, partition_state_create, dist_*, collective_*) is not "
    "present in the linked libquantumsim. Rebuild with -DQSIM_ENABLE_MPI=ON "
    "(and -DQSIM_BUILD_SHARED=ON), e.g.:\n"
    "    cmake -S . -B build-dist -DCMAKE_BUILD_TYPE=Release "
    "-DQSIM_ENABLE_MPI=ON -DQSIM_BUILD_SHARED=ON\n"
    "    cmake --build build-dist\n"
    "then point MOONLAB_LIB_DIR at build-dist."
)


class MpiUnavailableError(RuntimeError):
    """Raised when a distributed operation is requested on a non-MPI build."""


def is_mpi_available() -> bool:
    """Return ``True`` iff the linked libquantumsim has the MPI/distributed
    symbols (i.e. it was built with ``-DQSIM_ENABLE_MPI=ON``)."""
    return _HAS_MPI


def _require_mpi() -> None:
    if not _HAS_MPI:
        raise MpiUnavailableError(_NO_MPI_MESSAGE)


# ---------------------------------------------------------------- #
# ctypes struct definitions (only meaningful when MPI is present).  #
# ---------------------------------------------------------------- #


class _MpiInitOptions(ctypes.Structure):
    """Mirrors mpi_init_options_t from mpi_bridge.h."""

    _fields_ = [
        ("require_thread_multiple", ctypes.c_int),
        ("enable_rdma", ctypes.c_int),
        ("enable_gpu_direct", ctypes.c_int),
        ("comm_mode", ctypes.c_int),  # mpi_comm_mode_t enum
    ]


class _MeasurementResult(ctypes.Structure):
    """Mirrors dist_measurement_result_t from collective_ops.h."""

    _fields_ = [
        ("outcome", ctypes.c_uint64),
        ("probability", ctypes.c_double),
        ("measured_qubit", ctypes.c_int),
        ("collapsed", ctypes.c_int),
    ]


class _MeasurementConfig(ctypes.Structure):
    """Mirrors measurement_config_t from collective_ops.h."""

    _fields_ = [
        ("collapse_state", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("use_hardware_rng", ctypes.c_int),
    ]


# ---------------------------------------------------------------- #
# ABI signature setup -- only bound when the symbols exist.         #
# ---------------------------------------------------------------- #

if _HAS_MPI:
    _VOIDP = ctypes.c_void_p
    _U32 = ctypes.c_uint32
    _DBL = ctypes.c_double

    # -- MPI bridge --------------------------------------------------------
    _lib.mpi_bridge_init.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
        ctypes.POINTER(_MpiInitOptions),
    ]
    _lib.mpi_bridge_init.restype = _VOIDP

    _lib.mpi_bridge_init_no_args.argtypes = [ctypes.POINTER(_MpiInitOptions)]
    _lib.mpi_bridge_init_no_args.restype = _VOIDP

    _lib.mpi_bridge_free.argtypes = [_VOIDP]
    _lib.mpi_bridge_free.restype = None

    _lib.mpi_bridge_finalize.argtypes = []
    _lib.mpi_bridge_finalize.restype = None

    _lib.mpi_get_rank.argtypes = [_VOIDP]
    _lib.mpi_get_rank.restype = ctypes.c_int
    _lib.mpi_get_size.argtypes = [_VOIDP]
    _lib.mpi_get_size.restype = ctypes.c_int
    _lib.mpi_is_root.argtypes = [_VOIDP]
    _lib.mpi_is_root.restype = ctypes.c_int
    _lib.mpi_get_processor_name.argtypes = [_VOIDP]
    _lib.mpi_get_processor_name.restype = ctypes.c_char_p
    _lib.mpi_bridge_error_string.argtypes = [ctypes.c_int]
    _lib.mpi_bridge_error_string.restype = ctypes.c_char_p

    # -- partition ---------------------------------------------------------
    _lib.partition_state_create.argtypes = [_VOIDP, _U32, _VOIDP]
    _lib.partition_state_create.restype = _VOIDP
    _lib.partition_state_free.argtypes = [_VOIDP]
    _lib.partition_state_free.restype = None

    # -- single-qubit gates: int fn(state*, uint32 target) -----------------
    for _name in ("dist_hadamard", "dist_pauli_x", "dist_pauli_y",
                  "dist_pauli_z", "dist_s_gate", "dist_t_gate"):
        _fn = getattr(_lib, _name)
        _fn.argtypes = [_VOIDP, _U32]
        _fn.restype = ctypes.c_int

    # -- single-qubit rotations: int fn(state*, uint32 target, double) -----
    for _name in ("dist_rx", "dist_ry", "dist_rz", "dist_phase"):
        _fn = getattr(_lib, _name)
        _fn.argtypes = [_VOIDP, _U32, _DBL]
        _fn.restype = ctypes.c_int

    # -- two-qubit gates: int fn(state*, uint32 q1, uint32 q2) -------------
    for _name in ("dist_cnot", "dist_cz", "dist_swap", "dist_iswap"):
        _fn = getattr(_lib, _name)
        _fn.argtypes = [_VOIDP, _U32, _U32]
        _fn.restype = ctypes.c_int

    _lib.dist_cphase.argtypes = [_VOIDP, _U32, _U32, _DBL]
    _lib.dist_cphase.restype = ctypes.c_int

    _lib.dist_gate_error_string.argtypes = [ctypes.c_int]
    _lib.dist_gate_error_string.restype = ctypes.c_char_p

    # -- collective ops ----------------------------------------------------
    _lib.collective_measure_all.argtypes = [
        _VOIDP, ctypes.POINTER(_MeasurementResult),
        ctypes.POINTER(_MeasurementConfig),
    ]
    _lib.collective_measure_all.restype = ctypes.c_int

    _lib.collective_measure_qubit.argtypes = [
        _VOIDP, _U32, ctypes.POINTER(_MeasurementResult),
        ctypes.POINTER(_MeasurementConfig),
    ]
    _lib.collective_measure_qubit.restype = ctypes.c_int

    _lib.collective_sample.argtypes = [
        _VOIDP, ctypes.POINTER(_MeasurementResult),
        ctypes.POINTER(_MeasurementConfig),
    ]
    _lib.collective_sample.restype = ctypes.c_int

    _lib.collective_sample_many.argtypes = [
        _VOIDP, ctypes.POINTER(ctypes.c_uint64), _U32,
        ctypes.POINTER(_MeasurementConfig),
    ]
    _lib.collective_sample_many.restype = ctypes.c_int

    _lib.collective_get_probability.argtypes = [
        _VOIDP, ctypes.c_uint64, ctypes.POINTER(_DBL),
    ]
    _lib.collective_get_probability.restype = ctypes.c_int

    _lib.collective_get_qubit_probability.argtypes = [
        _VOIDP, _U32, ctypes.POINTER(_DBL),
    ]
    _lib.collective_get_qubit_probability.restype = ctypes.c_int

    for _name in ("collective_expectation_x", "collective_expectation_y",
                  "collective_expectation_z"):
        _fn = getattr(_lib, _name)
        _fn.argtypes = [_VOIDP, _U32, ctypes.POINTER(_DBL)]
        _fn.restype = ctypes.c_int

    _lib.collective_expectation_pauli.argtypes = [
        _VOIDP, ctypes.c_char_p, ctypes.POINTER(_DBL),
    ]
    _lib.collective_expectation_pauli.restype = ctypes.c_int

    _lib.collective_correlation_zz.argtypes = [
        _VOIDP, _U32, _U32, ctypes.POINTER(_DBL),
    ]
    _lib.collective_correlation_zz.restype = ctypes.c_int

    _lib.collective_error_string.argtypes = [ctypes.c_int]
    _lib.collective_error_string.restype = ctypes.c_char_p


# ---------------------------------------------------------------- #
# Error helpers.                                                    #
# ---------------------------------------------------------------- #


def _gate_check(rc: int) -> None:
    if rc != 0:
        msg = _lib.dist_gate_error_string(rc)
        raise RuntimeError(
            "distributed gate failed: "
            + (msg.decode("utf-8", "replace") if msg else f"code {rc}")
        )


def _collective_check(rc: int) -> None:
    if rc != 0:
        msg = _lib.collective_error_string(rc)
        raise RuntimeError(
            "collective operation failed: "
            + (msg.decode("utf-8", "replace") if msg else f"code {rc}")
        )


# ---------------------------------------------------------------- #
# Process-wide default context (set by init_mpi).                   #
# ---------------------------------------------------------------- #

_default_ctx: "Optional[DistributedContext]" = None


# ---------------------------------------------------------------- #
# DistributedContext.                                              #
# ---------------------------------------------------------------- #


class DistributedContext:
    """Wraps the MPI bridge ``distributed_ctx_t*``.

    Prefer :func:`init_mpi` over constructing this directly.  A context owns
    an MPI communicator view; free it with :meth:`free` (or via
    :func:`finalize_mpi` for the process-wide default).
    """

    def __init__(self, handle: int):
        if not handle:
            raise RuntimeError(
                "mpi_bridge_init returned NULL; MPI initialisation failed "
                "(is the process running under mpirun / an MPI launcher?)"
            )
        self._h = ctypes.c_void_p(handle)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    @property
    def rank(self) -> int:
        """This process's MPI rank (0 .. size-1)."""
        return int(_lib.mpi_get_rank(self._h))

    @property
    def size(self) -> int:
        """Total number of MPI ranks."""
        return int(_lib.mpi_get_size(self._h))

    @property
    def is_root(self) -> bool:
        """True on rank 0."""
        return bool(_lib.mpi_is_root(self._h))

    @property
    def processor_name(self) -> str:
        """MPI processor / host name for this rank."""
        raw = _lib.mpi_get_processor_name(self._h)
        return raw.decode("utf-8", "replace") if raw else ""

    def free(self) -> None:
        """Release the context.  Does not finalise MPI."""
        h = getattr(self, "_h", None)
        if h:
            _lib.mpi_bridge_free(h)
            self._h = ctypes.c_void_p(0)

    def __repr__(self) -> str:
        return f"DistributedContext(rank={self.rank}, size={self.size})"


def init_mpi(require_thread_multiple: bool = False,
             enable_rdma: bool = False) -> DistributedContext:
    """Initialise the MPI bridge and return a :class:`DistributedContext`.

    Calls ``mpi_bridge_init`` (which calls ``MPI_Init_thread`` if MPI is not
    already initialised).  The returned context is recorded as the
    process-wide default used by :class:`DistributedState` when it is
    constructed without an explicit ``ctx``.

    Safe to call once per process (a second call returns the same context).
    On a non-MPI build this raises :class:`MpiUnavailableError`.

    Parameters
    ----------
    require_thread_multiple : bool
        Request ``MPI_THREAD_MULTIPLE`` from the runtime.
    enable_rdma : bool
        Hint the bridge to enable RDMA optimisations where available.
    """
    _require_mpi()

    global _default_ctx
    if _default_ctx is not None:
        return _default_ctx

    opts = _MpiInitOptions(
        require_thread_multiple=1 if require_thread_multiple else 0,
        enable_rdma=1 if enable_rdma else 0,
        enable_gpu_direct=0,
        comm_mode=0,  # MPI_COMM_BLOCKING
    )
    # Pass NULL argc/argv: MPI_Init_thread(NULL, NULL, ...) is standard-legal.
    handle = _lib.mpi_bridge_init(None, None, ctypes.byref(opts))
    _default_ctx = DistributedContext(handle)
    return _default_ctx


def finalize_mpi() -> None:
    """Free the process-wide default context and finalise MPI.

    No-op on a non-MPI build.  Safe to call multiple times.
    """
    if not _HAS_MPI:
        return
    global _default_ctx
    if _default_ctx is not None:
        _default_ctx.free()
        _default_ctx = None
    _lib.mpi_bridge_finalize()


# ---------------------------------------------------------------- #
# DistributedState.                                                #
# ---------------------------------------------------------------- #


class DistributedState:
    """A quantum state vector partitioned across MPI ranks.

    The full 2^n amplitude array is split by high-order qubits across all
    ranks in the context.  Gates on low-order (local) qubits run without
    communication; gates on high-order (partition) qubits trigger inter-rank
    exchange.  Measurement and expectation values are MPI collectives and
    return the same value on every rank.

    Parameters
    ----------
    num_qubits : int
        Total qubits; the global state has ``2**num_qubits`` amplitudes
        distributed across ``ctx.size`` ranks.
    ctx : DistributedContext, optional
        The MPI context.  Defaults to the process-wide context created by
        :func:`init_mpi`; if none exists yet, :func:`init_mpi` is called.

    Example
    -------
    >>> init_mpi()
    >>> state = DistributedState(num_qubits=4)
    >>> state.h(0)
    >>> state.cnot(0, 1)
    >>> outcome = state.measure_all()
    """

    def __init__(self, num_qubits: int, ctx: Optional[DistributedContext] = None):
        _require_mpi()
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")

        if ctx is None:
            ctx = _default_ctx if _default_ctx is not None else init_mpi()
        self._ctx = ctx
        self._num_qubits = int(num_qubits)

        # config=NULL -> library defaults.  Initialises to |0...0>.
        h = _lib.partition_state_create(ctx.handle, ctypes.c_uint32(num_qubits), None)
        if not h:
            raise RuntimeError(
                f"partition_state_create failed for {num_qubits} qubits on "
                f"{ctx.size} ranks (out of memory, or num_qubits < log2(size))"
            )
        self._h = ctypes.c_void_p(h)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.partition_state_free(h)
            self._h = ctypes.c_void_p(0)

    def __repr__(self) -> str:
        return (
            f"DistributedState(num_qubits={self._num_qubits}, "
            f"rank={self.rank}, size={self.size})"
        )

    # -- context queries --------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def context(self) -> DistributedContext:
        return self._ctx

    @property
    def rank(self) -> int:
        """This process's MPI rank."""
        return self._ctx.rank

    @property
    def size(self) -> int:
        """Total number of MPI ranks."""
        return self._ctx.size

    # -- single-qubit gates ----------------------------------------------

    def h(self, target: int) -> "DistributedState":
        """Hadamard on ``target``."""
        _gate_check(_lib.dist_hadamard(self._h, ctypes.c_uint32(target)))
        return self

    def x(self, target: int) -> "DistributedState":
        """Pauli-X on ``target``."""
        _gate_check(_lib.dist_pauli_x(self._h, ctypes.c_uint32(target)))
        return self

    def y(self, target: int) -> "DistributedState":
        """Pauli-Y on ``target``."""
        _gate_check(_lib.dist_pauli_y(self._h, ctypes.c_uint32(target)))
        return self

    def z(self, target: int) -> "DistributedState":
        """Pauli-Z on ``target``."""
        _gate_check(_lib.dist_pauli_z(self._h, ctypes.c_uint32(target)))
        return self

    def s(self, target: int) -> "DistributedState":
        """S (phase, pi/2) gate on ``target``."""
        _gate_check(_lib.dist_s_gate(self._h, ctypes.c_uint32(target)))
        return self

    def t(self, target: int) -> "DistributedState":
        """T (phase, pi/4) gate on ``target``."""
        _gate_check(_lib.dist_t_gate(self._h, ctypes.c_uint32(target)))
        return self

    def rx(self, target: int, theta: float) -> "DistributedState":
        """Rotation about X by ``theta`` on ``target``."""
        _gate_check(_lib.dist_rx(self._h, ctypes.c_uint32(target), float(theta)))
        return self

    def ry(self, target: int, theta: float) -> "DistributedState":
        """Rotation about Y by ``theta`` on ``target``."""
        _gate_check(_lib.dist_ry(self._h, ctypes.c_uint32(target), float(theta)))
        return self

    def rz(self, target: int, theta: float) -> "DistributedState":
        """Rotation about Z by ``theta`` on ``target``."""
        _gate_check(_lib.dist_rz(self._h, ctypes.c_uint32(target), float(theta)))
        return self

    def phase(self, target: int, phi: float) -> "DistributedState":
        """Phase shift diag(1, e^{i phi}) on ``target``."""
        _gate_check(_lib.dist_phase(self._h, ctypes.c_uint32(target), float(phi)))
        return self

    # -- two-qubit gates --------------------------------------------------

    def cnot(self, control: int, target: int) -> "DistributedState":
        """Controlled-NOT with ``control`` and ``target``."""
        _gate_check(_lib.dist_cnot(self._h, ctypes.c_uint32(control),
                                   ctypes.c_uint32(target)))
        return self

    # Qiskit-style alias.
    cx = cnot

    def cz(self, qubit1: int, qubit2: int) -> "DistributedState":
        """Controlled-Z on ``qubit1``, ``qubit2``."""
        _gate_check(_lib.dist_cz(self._h, ctypes.c_uint32(qubit1),
                                 ctypes.c_uint32(qubit2)))
        return self

    def cphase(self, control: int, target: int, phi: float) -> "DistributedState":
        """Controlled phase e^{i phi} on the |11> subspace."""
        _gate_check(_lib.dist_cphase(self._h, ctypes.c_uint32(control),
                                     ctypes.c_uint32(target), float(phi)))
        return self

    def swap(self, qubit1: int, qubit2: int) -> "DistributedState":
        """SWAP ``qubit1`` and ``qubit2``."""
        _gate_check(_lib.dist_swap(self._h, ctypes.c_uint32(qubit1),
                                   ctypes.c_uint32(qubit2)))
        return self

    def iswap(self, qubit1: int, qubit2: int) -> "DistributedState":
        """iSWAP ``qubit1`` and ``qubit2``."""
        _gate_check(_lib.dist_iswap(self._h, ctypes.c_uint32(qubit1),
                                    ctypes.c_uint32(qubit2)))
        return self

    # -- measurement ------------------------------------------------------

    @staticmethod
    def _config(seed: Optional[int], collapse: bool) -> "Optional[_MeasurementConfig]":
        if seed is None:
            # NULL config -> library defaults (collapse on, system entropy).
            # We still build a config when the caller asks to suppress
            # collapse, since that departs from the default.
            if collapse:
                return None
            return _MeasurementConfig(collapse_state=0, seed=0, use_hardware_rng=0)
        return _MeasurementConfig(
            collapse_state=1 if collapse else 0,
            seed=int(seed),
            use_hardware_rng=0,
        )

    def measure_all(self, seed: Optional[int] = None, collapse: bool = True) -> int:
        """Measure all qubits; sample a basis state and (optionally) collapse.

        Returns the measured basis-state index.  Collective across all ranks:
        every rank returns the same outcome.
        """
        res = _MeasurementResult()
        cfg = self._config(seed, collapse)
        cfg_ptr = ctypes.byref(cfg) if cfg is not None else None
        _collective_check(
            _lib.collective_measure_all(self._h, ctypes.byref(res), cfg_ptr)
        )
        return int(res.outcome)

    def measure(self, qubit: int, seed: Optional[int] = None,
                collapse: bool = True) -> int:
        """Measure a single ``qubit``, projecting onto |0> or |1>.

        Returns 0 or 1.  Collective across all ranks.
        """
        res = _MeasurementResult()
        cfg = self._config(seed, collapse)
        cfg_ptr = ctypes.byref(cfg) if cfg is not None else None
        _collective_check(
            _lib.collective_measure_qubit(
                self._h, ctypes.c_uint32(qubit), ctypes.byref(res), cfg_ptr)
        )
        return int(res.outcome)

    def sample(self, seed: Optional[int] = None) -> int:
        """Draw one basis-state sample WITHOUT collapsing the state.

        Returns the sampled basis-state index.
        """
        res = _MeasurementResult()
        cfg = None if seed is None else _MeasurementConfig(
            collapse_state=0, seed=int(seed), use_hardware_rng=0)
        cfg_ptr = ctypes.byref(cfg) if cfg is not None else None
        _collective_check(
            _lib.collective_sample(self._h, ctypes.byref(res), cfg_ptr)
        )
        return int(res.outcome)

    def sample_many(self, num_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Draw ``num_samples`` basis-state samples without collapse.

        Returns a ``(num_samples,)`` uint64 NumPy array of basis-state indices.
        """
        if num_samples <= 0:
            return np.empty((0,), dtype=np.uint64)
        out = np.zeros(num_samples, dtype=np.uint64)
        cfg = None if seed is None else _MeasurementConfig(
            collapse_state=0, seed=int(seed), use_hardware_rng=0)
        cfg_ptr = ctypes.byref(cfg) if cfg is not None else None
        _collective_check(
            _lib.collective_sample_many(
                self._h,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                ctypes.c_uint32(num_samples),
                cfg_ptr,
            )
        )
        return out

    # -- probabilities ----------------------------------------------------

    def probability(self, basis_state: int) -> float:
        """Probability |amplitude[basis_state]|^2 of a specific basis state."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_get_probability(
                self._h, ctypes.c_uint64(basis_state), ctypes.byref(out))
        )
        return float(out.value)

    def qubit_probability(self, qubit: int) -> float:
        """Probability of measuring ``qubit`` as |1>."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_get_qubit_probability(
                self._h, ctypes.c_uint32(qubit), ctypes.byref(out))
        )
        return float(out.value)

    # Doc-guide alias: distributed_measure_probability(state, q).
    measure_probability = qubit_probability

    # -- expectation values ----------------------------------------------

    def expectation_x(self, qubit: int) -> float:
        """<X> on ``qubit``."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_expectation_x(
                self._h, ctypes.c_uint32(qubit), ctypes.byref(out))
        )
        return float(out.value)

    def expectation_y(self, qubit: int) -> float:
        """<Y> on ``qubit``."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_expectation_y(
                self._h, ctypes.c_uint32(qubit), ctypes.byref(out))
        )
        return float(out.value)

    def expectation_z(self, qubit: int) -> float:
        """<Z> on ``qubit``."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_expectation_z(
                self._h, ctypes.c_uint32(qubit), ctypes.byref(out))
        )
        return float(out.value)

    def expectation_pauli(self, pauli_string: str) -> float:
        """<P> for a Pauli string like ``"XYZI"`` (one char per qubit)."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_expectation_pauli(
                self._h, pauli_string.encode("ascii"), ctypes.byref(out))
        )
        return float(out.value)

    def correlation_zz(self, qubit_i: int, qubit_j: int) -> float:
        """<Z_i Z_j> two-qubit correlation."""
        out = ctypes.c_double(0.0)
        _collective_check(
            _lib.collective_correlation_zz(
                self._h, ctypes.c_uint32(qubit_i), ctypes.c_uint32(qubit_j),
                ctypes.byref(out))
        )
        return float(out.value)
