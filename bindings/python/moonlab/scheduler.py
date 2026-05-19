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


__all__ = ["Job", "JobResults", "SchedulerError"]
