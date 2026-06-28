"""Distributed scheduler Python binding -- since v0.7.1.

Wraps the C contract from `src/distributed/scheduler.{c,h}` shipped
in v0.7.0.  Mirrors the QGTL binding's structure: a `Job` class with
fluent `add_gate` + `set_*` builders, an `execute()` that runs the
worker fan-out and returns a `JobResults` dataclass.

Example:

    >>> from moonlab.scheduler import Job
    >>> from moonlab.qgtl import GateType
    >>> j = Job(num_qubits=2)
    >>> j.add_gate(GateType.H, target=0).add_gate(GateType.CNOT, target=1, control=0)
    >>> j.set_num_shots(1024).set_num_workers(4).set_rng_seed(0xdeadbeef)
    >>> r = j.execute()
    >>> all(o in (0, 3) for o in r.outcomes)
    True

@since v0.7.1
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Sequence

from .core import _lib
from .qgtl import GateType


MOONLAB_SCHED_OK = 0
MOONLAB_SCHED_BAD_ARG = -501
MOONLAB_SCHED_OOM = -502
MOONLAB_SCHED_INTERNAL = -503
MOONLAB_SCHED_BUFFER_TOO_SMALL = -504
MOONLAB_SCHED_BACKEND_NOT_FOUND = -506
MOONLAB_SCHED_BACKEND_BUSY = -507


class SchedulerError(RuntimeError):
    """Any negative-status return from the scheduler API."""


# ---- FFI signatures -----------------------------------------------

class _JobResults(ctypes.Structure):
    _fields_ = [
        ("num_qubits",       ctypes.c_int),
        ("total_shots",      ctypes.c_int),
        ("outcomes",         ctypes.POINTER(ctypes.c_uint64)),
        ("num_workers_used", ctypes.c_int),
        ("worker_seconds",   ctypes.POINTER(ctypes.c_double)),
    ]


_lib.moonlab_job_create.argtypes = [ctypes.c_int]
_lib.moonlab_job_create.restype = ctypes.c_void_p

_lib.moonlab_job_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_job_free.restype = None

_lib.moonlab_job_add_gate.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_job_add_gate.restype = ctypes.c_int

_lib.moonlab_job_set_num_shots.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.moonlab_job_set_num_shots.restype = ctypes.c_int

_lib.moonlab_job_set_num_workers.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.moonlab_job_set_num_workers.restype = ctypes.c_int

_lib.moonlab_job_set_rng_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_lib.moonlab_job_set_rng_seed.restype = ctypes.c_int

_lib.moonlab_job_num_qubits.argtypes = [ctypes.c_void_p]
_lib.moonlab_job_num_qubits.restype = ctypes.c_int
_lib.moonlab_job_num_gates.argtypes = [ctypes.c_void_p]
_lib.moonlab_job_num_gates.restype = ctypes.c_int
_lib.moonlab_job_num_shots.argtypes = [ctypes.c_void_p]
_lib.moonlab_job_num_shots.restype = ctypes.c_int
_lib.moonlab_job_num_workers.argtypes = [ctypes.c_void_p]
_lib.moonlab_job_num_workers.restype = ctypes.c_int

_lib.moonlab_scheduler_run.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(_JobResults)]
_lib.moonlab_scheduler_run.restype = ctypes.c_int

_lib.moonlab_job_results_free.argtypes = [ctypes.POINTER(_JobResults)]
_lib.moonlab_job_results_free.restype = None

_lib.moonlab_job_to_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.moonlab_job_to_json.restype = ctypes.c_int

# Backend plug-in surface (since v1.1).
_HAS_BACKEND_API = hasattr(_lib, "moonlab_job_set_backend")
if _HAS_BACKEND_API:
    _lib.moonlab_job_set_backend.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.moonlab_job_set_backend.restype = ctypes.c_int
    _lib.moonlab_job_backend.argtypes = [ctypes.c_void_p]
    _lib.moonlab_job_backend.restype = ctypes.c_char_p
    _lib.moonlab_num_backends.argtypes = []
    _lib.moonlab_num_backends.restype = ctypes.c_int
    _lib.moonlab_list_backends.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
    _lib.moonlab_list_backends.restype = ctypes.c_int

# Vendor-noise emulator registration (since v1.1).
_HAS_VENDOR_NOISE = hasattr(_lib, "moonlab_register_vendor_noise_backends")
if _HAS_VENDOR_NOISE:
    _lib.moonlab_register_vendor_noise_backends.argtypes = []
    _lib.moonlab_register_vendor_noise_backends.restype = ctypes.c_int

# Vendor-noise profile runtime registry (since v1.0.3).
_HAS_NOISE_PROFILE_REG = hasattr(_lib, "moonlab_register_vendor_noise_profile")


class _VendorNoiseProfile(ctypes.Structure):
    _fields_ = [
        ("p_gate_1q",   ctypes.c_double),
        ("p_gate_2q",   ctypes.c_double),
        ("p_readout",   ctypes.c_double),
        ("description", ctypes.c_char_p),
    ]


if _HAS_NOISE_PROFILE_REG:
    _lib.moonlab_register_vendor_noise_profile.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(_VendorNoiseProfile)]
    _lib.moonlab_register_vendor_noise_profile.restype = ctypes.c_int

    _lib.moonlab_unregister_vendor_noise_profile.argtypes = [ctypes.c_char_p]
    _lib.moonlab_unregister_vendor_noise_profile.restype = ctypes.c_int

    _lib.moonlab_lookup_vendor_noise_profile.argtypes = [ctypes.c_char_p]
    _lib.moonlab_lookup_vendor_noise_profile.restype = ctypes.POINTER(
        _VendorNoiseProfile)

    _lib.moonlab_num_vendor_noise_profiles.argtypes = []
    _lib.moonlab_num_vendor_noise_profiles.restype = ctypes.c_int

    _lib.moonlab_list_vendor_noise_profiles.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
    _lib.moonlab_list_vendor_noise_profiles.restype = ctypes.c_int

# Scheduler completion hook (since v1.0.3).
_HAS_COMPLETION_HOOK = hasattr(_lib, "moonlab_scheduler_set_completion_hook")
# Hook signature: (const moonlab_job_t*, const moonlab_job_results_t*,
#                  const char *backend_name, void *ctx) -> void.
_CompletionHookCFn = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,                       # job
    ctypes.POINTER(_JobResults),           # results
    ctypes.c_char_p,                       # backend name
    ctypes.c_void_p,                       # ctx (we don't thread Python ctx through)
)

if _HAS_COMPLETION_HOOK:
    _lib.moonlab_scheduler_set_completion_hook.argtypes = [
        _CompletionHookCFn, ctypes.c_void_p]
    _lib.moonlab_scheduler_set_completion_hook.restype = ctypes.c_int


def _check(rc: int, ctx: str) -> None:
    if rc != MOONLAB_SCHED_OK:
        raise SchedulerError(f"{ctx}: rc={rc}")


@dataclass
class JobResults:
    num_qubits: int
    total_shots: int
    outcomes: list[int]
    num_workers_used: int
    worker_seconds: list[float]


class Job:
    """Distributed-execution job: a circuit + shot count + worker count.

    Fluent builders return `self` so circuits compose cleanly."""

    __slots__ = ("_handle", "_n")

    def __init__(self, num_qubits: int) -> None:
        h = _lib.moonlab_job_create(int(num_qubits))
        if not h:
            raise SchedulerError(
                f"job_create({num_qubits}): NULL "
                f"(num_qubits must be in [1, 32])")
        self._handle = h
        self._n = int(num_qubits)

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.moonlab_job_free(self._handle)
            self._handle = None

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def num_gates(self) -> int:
        return int(_lib.moonlab_job_num_gates(self._handle))

    @property
    def num_shots(self) -> int:
        return int(_lib.moonlab_job_num_shots(self._handle))

    @property
    def num_workers(self) -> int:
        return int(_lib.moonlab_job_num_workers(self._handle))

    def add_gate(self,
                 type: GateType,
                 target: int,
                 control: int = -1,
                 params: Optional[Sequence[float]] = None) -> "Job":
        if params is not None:
            buf = (ctypes.c_double * len(params))(*params)
            params_ptr = buf
        else:
            params_ptr = None
        rc = _lib.moonlab_job_add_gate(
            self._handle, int(type), int(target), int(control), params_ptr)
        _check(rc, f"add_gate({type.name}, target={target}, control={control})")
        return self

    def set_num_shots(self, n: int) -> "Job":
        _check(_lib.moonlab_job_set_num_shots(self._handle, int(n)),
               f"set_num_shots({n})")
        return self

    def set_num_workers(self, n: int) -> "Job":
        _check(_lib.moonlab_job_set_num_workers(self._handle, int(n)),
               f"set_num_workers({n})")
        return self

    def set_rng_seed(self, seed: int) -> "Job":
        _check(_lib.moonlab_job_set_rng_seed(self._handle, int(seed)),
               f"set_rng_seed({seed})")
        return self

    def set_backend(self, backend_name: Optional[str]) -> "Job":
        """Pin this job to a registered backend.

        Pass None to clear (routes to the default ``"simulator"``).
        Vendor-noise emulators are registered via
        :func:`register_vendor_noise_backends`.  Raises ``SchedulerError``
        if the backend isn't found at the next ``execute()``.
        """
        if not _HAS_BACKEND_API:
            raise SchedulerError(
                "set_backend requires libquantumsim >= v1.1 "
                "(symbol moonlab_job_set_backend not exported)")
        name = backend_name.encode("utf-8") if backend_name else None
        _check(_lib.moonlab_job_set_backend(self._handle, name),
               f"set_backend({backend_name!r})")
        return self

    @property
    def backend(self) -> Optional[str]:
        """Backend name pinned via :py:meth:`set_backend`, or None for
        the default simulator."""
        if not _HAS_BACKEND_API:
            return None
        raw = _lib.moonlab_job_backend(self._handle)
        return raw.decode("utf-8") if raw else None

    def execute(self) -> JobResults:
        """Run the job's worker fan-out and return merged outcomes."""
        res = _JobResults()
        rc = _lib.moonlab_scheduler_run(self._handle, ctypes.byref(res))
        _check(rc, "scheduler_run")
        total = res.total_shots
        nw = res.num_workers_used
        outcomes = [int(res.outcomes[i]) for i in range(total)]
        worker_seconds = [float(res.worker_seconds[i]) for i in range(nw)]
        out = JobResults(
            num_qubits=res.num_qubits,
            total_shots=total,
            outcomes=outcomes,
            num_workers_used=nw,
            worker_seconds=worker_seconds,
        )
        _lib.moonlab_job_results_free(ctypes.byref(res))
        return out

    def to_json(self) -> str:
        """Serialise the job to a JSON string (moonlab/job/v0.7.0 schema)."""
        needed = _lib.moonlab_job_to_json(self._handle, None, 0)
        if needed < 0:
            raise SchedulerError(f"to_json size-probe: rc={needed}")
        buf = ctypes.create_string_buffer(needed + 1)
        _lib.moonlab_job_to_json(self._handle, buf, needed + 1)
        return buf.value.decode("utf-8")


def register_vendor_noise_backends() -> None:
    """Install the three pre-baked vendor-noise emulator backends
    (``ibm-falcon``, ``rigetti-aspen``, ``ionq-forte``) into the
    scheduler's backend registry.

    Idempotent.  Raises :class:`SchedulerError` on failure.

    Example::

        >>> from moonlab.scheduler import Job, register_vendor_noise_backends
        >>> register_vendor_noise_backends()
        >>> j = Job(num_qubits=2)
        >>> j.add_gate(GateType.H, 0).add_gate(GateType.CNOT, 1, 0)
        >>> j.set_num_shots(8192).set_backend("ibm-falcon")
        >>> r = j.execute()
        # Roughly 5% of outcomes will be off-Bell -- the device noise
        # signature you would see on a real IBM Falcon r5.11.
    """
    if not _HAS_VENDOR_NOISE:
        raise SchedulerError(
            "register_vendor_noise_backends requires libquantumsim >= v1.1 "
            "(symbol moonlab_register_vendor_noise_backends not exported)")
    _check(_lib.moonlab_register_vendor_noise_backends(),
           "register_vendor_noise_backends")


def list_backends() -> list[str]:
    """Return the names of all currently-registered scheduler backends.

    The default ``"simulator"`` backend is always present after the
    first scheduler call.
    """
    if not _HAS_BACKEND_API:
        return []
    n = int(_lib.moonlab_num_backends())
    if n <= 0:
        return []
    arr = (ctypes.c_char_p * n)()
    written = int(_lib.moonlab_list_backends(arr, n))
    return [arr[i].decode("utf-8") for i in range(written)]


@dataclass
class VendorNoiseProfile:
    """Snapshot of a hardware-noise profile -- 1q, 2q, and readout
    Pauli-channel rates plus a human-readable label.  Returned by
    :func:`lookup_vendor_noise_profile`; passed to
    :func:`register_vendor_noise_profile`."""
    p_gate_1q: float
    p_gate_2q: float
    p_readout: float
    description: str = ""


def register_vendor_noise_profile(name: str,
                                  profile: VendorNoiseProfile) -> None:
    """Install or replace a noise profile under ``name``.

    Lets a live-calibration scraper push today's device snapshot into
    the registry; existing backends registered against ``name`` see the
    update on the next ``execute()`` (the backend's ctx is the name, not
    a profile pointer, so updates take effect in place)."""
    if not _HAS_NOISE_PROFILE_REG:
        raise SchedulerError(
            "register_vendor_noise_profile requires libquantumsim with the "
            "v1.0.3+ vendor-noise profile registry compiled in")
    cprof = _VendorNoiseProfile(
        p_gate_1q=float(profile.p_gate_1q),
        p_gate_2q=float(profile.p_gate_2q),
        p_readout=float(profile.p_readout),
        description=(profile.description or "").encode("utf-8"),
    )
    _check(_lib.moonlab_register_vendor_noise_profile(
        name.encode("utf-8"), ctypes.byref(cprof)),
        f"register_vendor_noise_profile({name!r})")


def unregister_vendor_noise_profile(name: str) -> None:
    """Remove a noise profile from the registry."""
    if not _HAS_NOISE_PROFILE_REG:
        raise SchedulerError(
            "unregister_vendor_noise_profile requires libquantumsim >= v1.0.3")
    _check(_lib.moonlab_unregister_vendor_noise_profile(name.encode("utf-8")),
           f"unregister_vendor_noise_profile({name!r})")


def lookup_vendor_noise_profile(name: str) -> Optional[VendorNoiseProfile]:
    """Look up a noise profile by name.  Returns None if not registered."""
    if not _HAS_NOISE_PROFILE_REG:
        return None
    p = _lib.moonlab_lookup_vendor_noise_profile(name.encode("utf-8"))
    if not p:
        return None
    raw = p.contents
    desc = raw.description.decode("utf-8") if raw.description else ""
    return VendorNoiseProfile(
        p_gate_1q=float(raw.p_gate_1q),
        p_gate_2q=float(raw.p_gate_2q),
        p_readout=float(raw.p_readout),
        description=desc,
    )


def list_vendor_noise_profiles() -> list[str]:
    """Return the names of all currently-registered noise profiles."""
    if not _HAS_NOISE_PROFILE_REG:
        return []
    n = int(_lib.moonlab_num_vendor_noise_profiles())
    if n <= 0:
        return []
    arr = (ctypes.c_char_p * n)()
    written = int(_lib.moonlab_list_vendor_noise_profiles(arr, n))
    return [arr[i].decode("utf-8") for i in range(written)]


# ---- Completion hook ---------------------------------------------

# Module-global holder for the active Python hook so the ctypes
# trampoline does not get garbage-collected while installed.
_active_completion_hook: Optional[_CompletionHookCFn] = None
_active_completion_callback = None


def set_completion_hook(callback: Optional[callable]) -> None:
    """Install a Python callback that fires after every successful
    ``scheduler.run`` (and only successful runs -- failed dispatches
    do not fire the hook).

    The callback receives ``(num_qubits, total_shots, backend_name)``;
    if ``callback`` is None the hook is cleared.

    Use cases: billing meter, audit log, customer dashboard.  Runs
    synchronously on the caller thread, so keep the work short."""
    global _active_completion_hook, _active_completion_callback
    if not _HAS_COMPLETION_HOOK:
        raise SchedulerError(
            "set_completion_hook requires libquantumsim with the v1.0.3+ "
            "scheduler completion hook compiled in")

    if callback is None:
        # Pass a NULL fn pointer to clear; the C scheduler checks for
        # NULL before firing, so this is the right idempotent disable.
        # ctypes accepts None for a CFUNCTYPE argument and passes NULL.
        _lib.moonlab_scheduler_set_completion_hook(
            ctypes.cast(None, _CompletionHookCFn), None)
        _active_completion_hook = None
        _active_completion_callback = None
        return

    def trampoline(job, results, backend_name, ctx):
        try:
            res = results.contents
            backend = backend_name.decode("utf-8") if backend_name else None
            callback(res.num_qubits, res.total_shots, backend)
        except Exception:
            # Hook errors must not propagate into the C runtime.
            import traceback
            traceback.print_exc()

    cfn = _CompletionHookCFn(trampoline)
    _check(_lib.moonlab_scheduler_set_completion_hook(cfn, None),
           "set_completion_hook")
    _active_completion_hook = cfn
    _active_completion_callback = callback


def clear_completion_hook() -> None:
    """Detach the active completion hook.  Equivalent to
    ``set_completion_hook(None)``."""
    set_completion_hook(None)


__all__ = [
    "Job",
    "JobResults",
    "SchedulerError",
    "VendorNoiseProfile",
    "register_vendor_noise_backends",
    "register_vendor_noise_profile",
    "unregister_vendor_noise_profile",
    "lookup_vendor_noise_profile",
    "list_vendor_noise_profiles",
    "list_backends",
    "set_completion_hook",
    "clear_completion_hook",
]
