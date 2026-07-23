"""Moonlab noise-channel bindings.

Pythonic access to the Kraus-operator noise engine the C core ships in
``src/quantum/noise.{h,c}``.  Every channel here is a thin wrapper over a
real C entry point -- there is no Python-side re-implementation of the
channel physics.  The C channels are Monte-Carlo trajectory unravellings:
each ``apply`` call realises one Kraus branch selected from a uniform
random draw, and averaging the reduced density operator over many shots
recovers the CPTP map.  This mirrors how a real device produces one
sample per shot.

Channels
--------
- :class:`DepolarizingChannel` -- ``noise_depolarizing_single``: with
  probability ``p`` applies a uniformly random Pauli (X, Y, or Z).
- :class:`AmplitudeDamping` -- ``noise_amplitude_damping``: T1 energy
  relaxation, :math:`|1\\rangle \\to |0\\rangle`.
- :class:`PhaseDamping` -- ``noise_phase_damping``: T2 dephasing without
  energy loss.
- :class:`BitFlip` / :class:`PhaseFlip` / :class:`BitPhaseFlip` --
  ``noise_bit_flip`` / ``noise_phase_flip`` / ``noise_bit_phase_flip``:
  the elementary Pauli channels (X, Z, Y with probability ``p``).
- :class:`ThermalRelaxation` -- ``noise_thermal_relaxation``: combined
  T1/T2 relaxation for a given gate time.  Finite excited-state
  population (finite temperature) is modelled as generalized amplitude
  damping by conjugating the C channel with X on the excitation branch.
- :class:`ReadoutError` -- ``noise_readout_error``: classical
  measurement bit-flip on top of the real measurement engine.
- :class:`DeviceNoiseModel` -- ``noise_model_t`` + ``noise_apply_model``:
  a per-qubit composite profile (depolarizing + T1/T2 thermal) applied by
  the C ``noise_apply_model`` after a gate, plus two-qubit depolarizing
  via ``noise_apply_model_two_qubit``.

All random draws come from a per-object :class:`numpy.random.Generator`;
pass ``seed=`` for reproducibility.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence, Union

import numpy as np

from .core import _lib, CQuantumState, QuantumState, QuantumError

__all__ = [
    "DepolarizingChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "BitFlip",
    "PhaseFlip",
    "BitPhaseFlip",
    "ThermalRelaxation",
    "ReadoutError",
    "DeviceNoiseModel",
]

# ============================================================================
# C SIGNATURES -- src/quantum/noise.h
# ============================================================================
_S = ctypes.POINTER(CQuantumState)

_lib.noise_depolarizing_single.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_depolarizing_single.restype = None

_lib.noise_depolarizing_two_qubit.argtypes = [
    _S, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_depolarizing_two_qubit.restype = None

_lib.noise_amplitude_damping.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_amplitude_damping.restype = None

_lib.noise_phase_damping.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_phase_damping.restype = None

_lib.noise_bit_flip.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_bit_flip.restype = None

_lib.noise_phase_flip.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_phase_flip.restype = None

_lib.noise_bit_phase_flip.argtypes = [_S, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_lib.noise_bit_phase_flip.restype = None

_lib.noise_thermal_relaxation.argtypes = [
    _S, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double)]
_lib.noise_thermal_relaxation.restype = None

_lib.noise_readout_error.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
_lib.noise_readout_error.restype = ctypes.c_int

# noise_model_t management + composite application.
_lib.noise_model_create.argtypes = []
_lib.noise_model_create.restype = ctypes.c_void_p

_lib.noise_model_destroy.argtypes = [ctypes.c_void_p]
_lib.noise_model_destroy.restype = None

_lib.noise_model_set_depolarizing.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.noise_model_set_depolarizing.restype = None

_lib.noise_model_set_amplitude_damping.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.noise_model_set_amplitude_damping.restype = None

_lib.noise_model_set_phase_damping.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.noise_model_set_phase_damping.restype = None

_lib.noise_model_set_thermal.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_lib.noise_model_set_thermal.restype = None

_lib.noise_model_set_gate_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.noise_model_set_gate_time.restype = None

_lib.noise_model_set_readout_error.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_lib.noise_model_set_readout_error.restype = None

# noise_apply_model reads a length-6 (single) / length-10 (two-qubit)
# random-value buffer; noise_model_t* is passed opaquely as void*.
_lib.noise_apply_model.argtypes = [
    _S, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.noise_apply_model.restype = None

_lib.noise_apply_model_two_qubit.argtypes = [
    _S, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.noise_apply_model_two_qubit.restype = None


def _as_cstate(state: QuantumState):
    """Return a ctypes byref to the underlying CQuantumState.

    A GPU-backed state must be synced to host first because the C noise
    channels that touch the amplitudes buffer read/write the host mirror.
    ``noise_*`` push the result back to the device themselves.
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"expected moonlab.QuantumState, got {type(state).__name__}")
    if getattr(state, "is_gpu", False):
        state.sync_to_host()
    return ctypes.byref(state._state)


class _SingleParamChannel:
    """Common machinery for the single-parameter trajectory channels.

    Subclasses set ``_cfunc`` (the C entry point) and ``_pname`` (the
    parameter name used in messages).  ``apply`` draws one uniform sample
    and forwards it as the C ``random_value`` -- one Monte-Carlo
    trajectory per call.
    """

    _cfunc = None
    _pname = "parameter"

    def __init__(self, value: float, seed: Optional[int] = None):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{self._pname} must be in [0, 1], got {value}")
        self._value = value
        self._rng = np.random.default_rng(seed)

    @property
    def value(self) -> float:
        return self._value

    def apply(self, state: QuantumState, qubit: int) -> QuantumState:
        """Apply one trajectory of the channel to ``qubit`` of ``state``.

        Returns ``state`` for chaining.  A ``value`` of 0 is an exact
        no-op (the C channel returns without touching the state).
        """
        if not 0 <= qubit < state.num_qubits:
            raise ValueError(f"qubit {qubit} out of range for {state.num_qubits}-qubit state")
        r = float(self._rng.random())
        type(self)._cfunc(_as_cstate(state), int(qubit), self._value, r)
        return state


class DepolarizingChannel(_SingleParamChannel):
    """Depolarizing channel: with probability ``p`` apply a random Pauli.

    :math:`\\mathcal E_p(\\rho) = (1-p)\\rho + (p/3)(X\\rho X + Y\\rho Y + Z\\rho Z)`.
    Wraps ``noise_depolarizing_single``.  ``p = 1`` fully depolarizes the
    single-qubit reduced state (Bloch vector to 0) when averaged over
    trajectories.
    """

    _cfunc = _lib.noise_depolarizing_single
    _pname = "probability"

    def __init__(self, probability: Optional[float] = None, *,
                 error_rate: Optional[float] = None, seed: Optional[int] = None):
        if probability is None:
            probability = error_rate
        if probability is None:
            raise TypeError("DepolarizingChannel requires a probability")
        super().__init__(probability, seed=seed)

    @property
    def probability(self) -> float:
        return self._value


class AmplitudeDamping(_SingleParamChannel):
    """T1 energy relaxation :math:`|1\\rangle \\to |0\\rangle`.

    Wraps ``noise_amplitude_damping``.  Kraus operators
    :math:`K_0 = \\mathrm{diag}(1, \\sqrt{1-\\gamma})`,
    :math:`K_1 = \\sqrt{\\gamma}\\,|0\\rangle\\langle 1|`.  ``gamma = 1``
    deterministically drives a populated :math:`|1\\rangle` to
    :math:`|0\\rangle`.
    """

    _cfunc = _lib.noise_amplitude_damping
    _pname = "gamma"

    def __init__(self, gamma: float, *, seed: Optional[int] = None):
        super().__init__(gamma, seed=seed)

    @property
    def gamma(self) -> float:
        return self._value


class PhaseDamping(_SingleParamChannel):
    """T2 dephasing without energy loss.

    Wraps ``noise_phase_damping``.  Kraus operators
    :math:`K_0 = \\mathrm{diag}(1, \\sqrt{1-\\gamma})`,
    :math:`K_1 = \\mathrm{diag}(0, \\sqrt{\\gamma})`.  Destroys off-diagonal
    coherence while leaving computational-basis populations invariant on
    average.
    """

    _cfunc = _lib.noise_phase_damping
    _pname = "gamma"

    def __init__(self, gamma: float, *, seed: Optional[int] = None):
        super().__init__(gamma, seed=seed)

    @property
    def gamma(self) -> float:
        return self._value


class BitFlip(_SingleParamChannel):
    """Bit-flip channel: apply X with probability ``p``. Wraps ``noise_bit_flip``."""

    _cfunc = _lib.noise_bit_flip
    _pname = "probability"

    def __init__(self, probability: float, *, seed: Optional[int] = None):
        super().__init__(probability, seed=seed)

    @property
    def probability(self) -> float:
        return self._value


class PhaseFlip(_SingleParamChannel):
    """Phase-flip channel: apply Z with probability ``p``. Wraps ``noise_phase_flip``."""

    _cfunc = _lib.noise_phase_flip
    _pname = "probability"

    def __init__(self, probability: float, *, seed: Optional[int] = None):
        super().__init__(probability, seed=seed)

    @property
    def probability(self) -> float:
        return self._value


class BitPhaseFlip(_SingleParamChannel):
    """Bit-phase-flip channel: apply Y with probability ``p``. Wraps ``noise_bit_phase_flip``."""

    _cfunc = _lib.noise_bit_phase_flip
    _pname = "probability"

    def __init__(self, probability: float, *, seed: Optional[int] = None):
        super().__init__(probability, seed=seed)

    @property
    def probability(self) -> float:
        return self._value


class ThermalRelaxation:
    """Combined T1/T2 thermal relaxation for a gate of duration ``gate_time``.

    Wraps ``noise_thermal_relaxation``, which internally derives an
    amplitude-damping rate :math:`\\gamma_1 = 1 - e^{-t/T_1}` and a residual
    pure-dephasing rate from the T2 budget, then applies
    ``noise_amplitude_damping`` followed by ``noise_phase_damping``.  T2 is
    clamped to :math:`\\le 2 T_1` by the C layer (physical constraint).

    ``excited_population`` (finite-temperature steady state) is modelled as
    generalized amplitude damping: with that probability the relaxation
    branch is conjugated by X (``gate_pauli_x``), so the qubit relaxes
    toward :math:`|1\\rangle` instead of :math:`|0\\rangle`.  ``T1``, ``T2``
    and ``gate_time`` share the same time unit (seconds in the guide
    examples); only their ratios enter.
    """

    def __init__(self, t1: float, t2: float, gate_time: float,
                 excited_population: float = 0.0, *, seed: Optional[int] = None):
        if t1 <= 0.0 or t2 <= 0.0:
            raise ValueError("t1 and t2 must be positive")
        if gate_time <= 0.0:
            raise ValueError("gate_time must be positive")
        if not 0.0 <= excited_population <= 1.0:
            raise ValueError("excited_population must be in [0, 1]")
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.gate_time = float(gate_time)
        self.excited_population = float(excited_population)
        self._rng = np.random.default_rng(seed)

    def apply(self, state: QuantumState, qubit: int) -> QuantumState:
        if not 0 <= qubit < state.num_qubits:
            raise ValueError(f"qubit {qubit} out of range for {state.num_qubits}-qubit state")
        cstate = _as_cstate(state)
        rvals = (ctypes.c_double * 2)(
            float(self._rng.random()), float(self._rng.random()))

        excite = (self.excited_population > 0.0
                  and float(self._rng.random()) < self.excited_population)
        if excite:
            # Generalized amplitude damping, excitation branch: conjugate
            # the zero-temperature relaxation with X so the fixed point is
            # |1> rather than |0>.
            _lib.gate_pauli_x(cstate, int(qubit))
            _lib.noise_thermal_relaxation(
                cstate, int(qubit), self.t1, self.t2, self.gate_time, rvals)
            _lib.gate_pauli_x(cstate, int(qubit))
        else:
            _lib.noise_thermal_relaxation(
                cstate, int(qubit), self.t1, self.t2, self.gate_time, rvals)
        if getattr(state, "is_gpu", False):
            state.sync_from_host()
        return state


class ReadoutError:
    """Classical measurement bit-flip on top of the real measurement engine.

    Wraps ``noise_readout_error``.  Construct either from the two flip
    probabilities (``error_0_to_1`` = P(read 1 | prepared 0),
    ``error_1_to_0`` = P(read 0 | prepared 1)) or from a 2x2 assignment
    (confusion) matrix ``[[P(0|0), P(1|0)], [P(0|1), P(1|1)]]``.
    :meth:`measure` first performs a real projective measurement of the
    state, then applies the classical flip.
    """

    def __init__(self, confusion_matrix: Optional[Sequence[Sequence[float]]] = None, *,
                 error_0_to_1: Optional[float] = None,
                 error_1_to_0: Optional[float] = None,
                 seed: Optional[int] = None):
        if confusion_matrix is not None:
            m = np.asarray(confusion_matrix, dtype=float)
            if m.shape != (2, 2):
                raise ValueError("confusion_matrix must be 2x2")
            self.error_0_to_1 = float(m[0, 1])
            self.error_1_to_0 = float(m[1, 0])
        else:
            self.error_0_to_1 = float(error_0_to_1 or 0.0)
            self.error_1_to_0 = float(error_1_to_0 or 0.0)
        for v in (self.error_0_to_1, self.error_1_to_0):
            if not 0.0 <= v <= 1.0:
                raise ValueError("readout error probabilities must be in [0, 1]")
        self._rng = np.random.default_rng(seed)

    def apply_outcome(self, outcome: int) -> int:
        """Flip a classical bit ``outcome`` (0/1) through the C readout channel."""
        r = float(self._rng.random())
        return int(_lib.noise_readout_error(
            int(outcome), self.error_0_to_1, self.error_1_to_0, r))

    def measure(self, state: QuantumState, qubit: int) -> int:
        """Projectively measure ``qubit`` then apply the classical readout flip."""
        from .core import Measurement
        true_outcome = Measurement.measure(state, qubit)
        return self.apply_outcome(true_outcome)


class DeviceNoiseModel:
    """Per-qubit composite device noise applied by the C ``noise_apply_model``.

    Builds one C ``noise_model_t`` per qubit, each carrying a
    single-qubit depolarizing rate and (optionally) a T1/T2 thermal
    profile for the configured gate time.  :meth:`apply_single` runs the C
    ``noise_apply_model`` (depolarizing then thermal relaxation) after a
    single-qubit gate; :meth:`apply_two` runs ``noise_apply_model_two_qubit``
    (two-qubit depolarizing then the per-qubit single-qubit profile) after
    a two-qubit gate.

    Args:
        num_qubits: register width.
        t1, t2: scalar or per-qubit list of relaxation times (same unit as
            ``gate_time``).  Omit both to disable thermal relaxation.
        gate_time: gate duration used for the thermal rates.
        single_qubit_error: single-qubit depolarizing probability per gate.
        two_qubit_error: two-qubit depolarizing probability per gate.
        readout_error: symmetric readout flip probability (stored on each
            per-qubit model; used by :meth:`readout` / build a
            :class:`ReadoutError`).
        topology: optional list of allowed (a, b) qubit pairs; informational.
    """

    def __init__(self, num_qubits: int, *,
                 t1: Union[float, Sequence[float], None] = None,
                 t2: Union[float, Sequence[float], None] = None,
                 gate_time: float = 20e-9,
                 single_qubit_error: float = 0.0,
                 two_qubit_error: float = 0.0,
                 readout_error: float = 0.0,
                 topology: Optional[Sequence] = None,
                 seed: Optional[int] = None):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = int(num_qubits)
        self.gate_time = float(gate_time)
        self.single_qubit_error = float(single_qubit_error)
        self.two_qubit_error = float(two_qubit_error)
        self.readout_error = float(readout_error)
        self.topology = list(topology) if topology is not None else None
        self._rng = np.random.default_rng(seed)

        t1_list = self._broadcast(t1, num_qubits)
        t2_list = self._broadcast(t2, num_qubits)
        self.t1 = t1_list
        self.t2 = t2_list

        # One C noise_model_t per qubit.  These are owned by this object
        # and freed in __del__ via noise_model_destroy.
        self._models = []
        for q in range(self.num_qubits):
            m = _lib.noise_model_create()
            if not m:
                self._free_models()
                raise QuantumError("noise_model_create returned NULL")
            m = ctypes.c_void_p(m)
            if self.single_qubit_error > 0.0:
                _lib.noise_model_set_depolarizing(m, self.single_qubit_error)
            if t1_list is not None and t2_list is not None:
                _lib.noise_model_set_thermal(m, float(t1_list[q]), float(t2_list[q]))
                _lib.noise_model_set_gate_time(m, self.gate_time)
            if self.readout_error > 0.0:
                _lib.noise_model_set_readout_error(m, self.readout_error, self.readout_error)
            self._models.append(m)

    @staticmethod
    def _broadcast(v, n):
        if v is None:
            return None
        if np.isscalar(v):
            return [float(v)] * n
        v = list(v)
        if len(v) != n:
            raise ValueError(f"expected {n} values, got {len(v)}")
        return [float(x) for x in v]

    def _free_models(self):
        for m in getattr(self, "_models", []):
            if m:
                _lib.noise_model_destroy(m)
        self._models = []

    def __del__(self):
        self._free_models()

    def apply_single(self, state: QuantumState, qubit: int) -> QuantumState:
        """Apply the per-qubit composite noise profile after a single-qubit gate."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"qubit {qubit} out of range for {self.num_qubits}-qubit model")
        cstate = _as_cstate(state)
        # noise_apply_model consumes up to 5 random values (depolarizing,
        # amplitude, phase, then 2 for thermal); over-provision to 8.
        rvals = (ctypes.c_double * 8)(*[float(self._rng.random()) for _ in range(8)])
        _lib.noise_apply_model(cstate, int(qubit), self._models[qubit], rvals)
        if getattr(state, "is_gpu", False):
            state.sync_from_host()
        return state

    def apply_two(self, state: QuantumState, qubit_a: int, qubit_b: int) -> QuantumState:
        """Apply two-qubit depolarizing + per-qubit noise after a two-qubit gate.

        Uses ``noise_apply_model_two_qubit`` with a model carrying the
        two-qubit depolarizing rate, then the single-qubit profile on each
        qubit.
        """
        for q in (qubit_a, qubit_b):
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} out of range for {self.num_qubits}-qubit model")
        cstate = _as_cstate(state)
        # Build a scratch model that also carries the two-qubit rate and the
        # qubit_a single-qubit profile (noise_apply_model_two_qubit applies
        # the same model to both qubits).
        m = self._models[qubit_a]
        if self.two_qubit_error > 0.0:
            # Temporarily set the 2q depolarizing rate on the model struct.
            # noise_model_t.two_qubit_depolarizing_rate has no setter, so we
            # route the 2q rate through a dedicated scratch model.
            scratch = ctypes.c_void_p(_lib.noise_model_create())
            _lib.noise_model_set_depolarizing(scratch, self.single_qubit_error)
            self._set_two_qubit_rate(scratch, self.two_qubit_error)
            rvals = (ctypes.c_double * 12)(*[float(self._rng.random()) for _ in range(12)])
            _lib.noise_apply_model_two_qubit(cstate, int(qubit_a), int(qubit_b), scratch, rvals)
            _lib.noise_model_destroy(scratch)
        else:
            self.apply_single(state, qubit_a)
            self.apply_single(state, qubit_b)
            return state
        if getattr(state, "is_gpu", False):
            state.sync_from_host()
        return state

    @staticmethod
    def _set_two_qubit_rate(model_ptr, rate):
        """Write two_qubit_depolarizing_rate directly (no C setter exists).

        The field sits after the three single-qubit rates and the three
        thermal fields in noise_model_t; mirror that layout to poke it.
        """
        # Layout: int enabled; double depol, amp, phase, t1, t2, gate_time,
        #         two_qubit_depolarizing_rate, readout_0, readout_1.
        class _NoiseModel(ctypes.Structure):
            _fields_ = [
                ("enabled", ctypes.c_int),
                ("depolarizing_rate", ctypes.c_double),
                ("amplitude_damping_rate", ctypes.c_double),
                ("phase_damping_rate", ctypes.c_double),
                ("t1", ctypes.c_double),
                ("t2", ctypes.c_double),
                ("gate_time", ctypes.c_double),
                ("two_qubit_depolarizing_rate", ctypes.c_double),
                ("readout_error_0", ctypes.c_double),
                ("readout_error_1", ctypes.c_double),
            ]
        view = ctypes.cast(model_ptr, ctypes.POINTER(_NoiseModel))
        view.contents.two_qubit_depolarizing_rate = float(rate)

    def readout(self, qubit: int = 0, *, seed: Optional[int] = None) -> ReadoutError:
        """Build a :class:`ReadoutError` from this device's readout rate."""
        return ReadoutError(error_0_to_1=self.readout_error,
                            error_1_to_0=self.readout_error, seed=seed)
