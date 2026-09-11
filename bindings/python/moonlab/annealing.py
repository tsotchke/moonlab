"""Complete transverse-field quantum annealing bindings (Moonlab v1.2.1)."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from .core import _lib


class AnnealError(RuntimeError):
    """Native annealing input, allocation, schedule, or evolution failure."""


class AnnealSchedule(IntEnum):
    LINEAR = 0
    QUADRATIC = 1
    COSINE = 2
    PIECEWISE = 3


@dataclass(frozen=True)
class SchedulePoint:
    t: float
    s: float


@dataclass(frozen=True)
class AnnealConfig:
    total_time: float = 10.0
    num_steps: int = 1000
    num_samples: int = 1024
    seed: int = 0
    schedule: AnnealSchedule = AnnealSchedule.COSINE
    driver_strength: float = 1.0
    problem_strength: float = 1.0
    second_order: bool = True
    schedule_points: Sequence[tuple[float, float] | SchedulePoint] | None = None
    reverse_anneal: bool = False
    initial_bitstring: int = 0
    reverse_s_target: float = 0.0
    reverse_hold_fraction: float = 0.0
    anneal_offsets: Sequence[float] | None = None


@dataclass(frozen=True)
class AnnealResult:
    num_qubits: int
    effective_seed: int
    best_bitstring: int
    most_likely_bitstring: int
    ground_bitstring: int
    ground_degeneracy: int
    best_energy: float
    ground_energy: float
    expected_energy: float
    success_probability: float
    residual_energy: float
    problem_gap: float
    final_norm: float
    samples: list[int]
    sample_energies: list[float]


@dataclass(frozen=True)
class ZephyrGraph:
    m: int
    t: int
    num_qubits: int
    num_couplers: int
    couplers: list[tuple[int, int]]


@dataclass(frozen=True)
class ZephyrEmbedding:
    num_logical: int
    chains: list[list[int]]


class _CSchedulePoint(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("s", ctypes.c_double),
    ]


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("total_time", ctypes.c_double),
        ("num_steps", ctypes.c_size_t),
        ("num_samples", ctypes.c_size_t),
        ("seed", ctypes.c_uint64),
        ("schedule", ctypes.c_int),
        ("driver_strength", ctypes.c_double),
        ("problem_strength", ctypes.c_double),
        ("second_order", ctypes.c_int),
        ("schedule_points", ctypes.POINTER(_CSchedulePoint)),
        ("num_schedule_points", ctypes.c_size_t),
        ("reverse_anneal", ctypes.c_int),
        ("initial_bitstring", ctypes.c_uint64),
        ("reverse_s_target", ctypes.c_double),
        ("reverse_hold_fraction", ctypes.c_double),
        ("anneal_offsets", ctypes.POINTER(ctypes.c_double)),
    ]


class _CZephyrGraph(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_size_t),
        ("t", ctypes.c_size_t),
        ("num_qubits", ctypes.c_size_t),
        ("num_couplers", ctypes.c_size_t),
        ("coupler_u", ctypes.POINTER(ctypes.c_uint32)),
        ("coupler_v", ctypes.POINTER(ctypes.c_uint32)),
        ("adjacency_matrix", ctypes.POINTER(ctypes.c_uint8)),
    ]


class _CZephyrEmbedding(ctypes.Structure):
    _fields_ = [
        ("num_logical", ctypes.c_size_t),
        ("chain_lengths", ctypes.POINTER(ctypes.c_size_t)),
        ("chains", ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t))),
    ]



class _CResult(ctypes.Structure):
    pass


_ResultPtr = ctypes.POINTER(_CResult)
_DoublePtr = ctypes.POINTER(ctypes.c_double)

_lib.moonlab_quantum_anneal_ising.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    _DoublePtr,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_anneal_ising.restype = ctypes.c_int
_lib.moonlab_quantum_anneal_qubo.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_anneal_qubo.restype = ctypes.c_int
_lib.moonlab_qubo_to_ising.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    ctypes.c_double,
    _DoublePtr,
    _DoublePtr,
    _DoublePtr,
]
_lib.moonlab_qubo_to_ising.restype = ctypes.c_int
_lib.moonlab_anneal_result_free.argtypes = [_ResultPtr]
_lib.moonlab_anneal_result_free.restype = None

_lib.moonlab_anneal_schedule_make_pause.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(_CSchedulePoint),
]
_lib.moonlab_anneal_schedule_make_pause.restype = ctypes.c_int

_lib.moonlab_anneal_schedule_make_quench.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(_CSchedulePoint),
]
_lib.moonlab_anneal_schedule_make_quench.restype = ctypes.c_int

_lib.moonlab_quantum_reverse_anneal_ising.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    _DoublePtr,
    ctypes.c_double,
    ctypes.c_uint64,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_reverse_anneal_ising.restype = ctypes.c_int

_lib.moonlab_quantum_reverse_anneal_qubo.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    ctypes.c_double,
    ctypes.c_uint64,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_reverse_anneal_qubo.restype = ctypes.c_int

_lib.moonlab_zephyr_graph_create.argtypes = [ctypes.c_size_t]
_lib.moonlab_zephyr_graph_create.restype = ctypes.POINTER(_CZephyrGraph)

_lib.moonlab_zephyr_graph_free.argtypes = [ctypes.POINTER(_CZephyrGraph)]
_lib.moonlab_zephyr_graph_free.restype = None

_lib.moonlab_zephyr_has_coupler.argtypes = [
    ctypes.POINTER(_CZephyrGraph),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.moonlab_zephyr_has_coupler.restype = ctypes.c_int

_lib.moonlab_zephyr_find_clique_embedding.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.moonlab_zephyr_find_clique_embedding.restype = ctypes.POINTER(_CZephyrEmbedding)

_lib.moonlab_zephyr_embedding_free.argtypes = [ctypes.POINTER(_CZephyrEmbedding)]
_lib.moonlab_zephyr_embedding_free.restype = None

_lib.moonlab_zephyr_embed_ising.argtypes = [
    ctypes.POINTER(_CZephyrGraph),
    ctypes.POINTER(_CZephyrEmbedding),
    _DoublePtr,
    _DoublePtr,
    ctypes.c_double,
    _DoublePtr,
    _DoublePtr,
]
_lib.moonlab_zephyr_embed_ising.restype = ctypes.c_int

_lib.moonlab_zephyr_unembed_samples.argtypes = [
    ctypes.POINTER(_CZephyrEmbedding),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    _DoublePtr,
]
_lib.moonlab_zephyr_unembed_samples.restype = ctypes.c_int

_UINT64_GETTERS = (
    "effective_seed",
    "best_bitstring",
    "most_likely_bitstring",
    "ground_bitstring",
)
for _name in _UINT64_GETTERS:
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_uint64
for _name in ("num_qubits", "num_samples", "ground_degeneracy"):
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_size_t
for _name in (
    "best_energy",
    "ground_energy",
    "expected_energy",
    "success_probability",
    "residual_energy",
    "problem_gap",
    "final_norm",
):
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_double
_lib.moonlab_anneal_result_sample.argtypes = [
    _ResultPtr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_anneal_result_sample.restype = ctypes.c_int


def _matrix(values: Sequence[Sequence[float]], name: str) -> tuple[int, ctypes.Array]:
    rows = [list(row) for row in values]
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise AnnealError(f"{name} must be a non-empty square matrix")
    flat = [float(value) for row in rows for value in row]
    return n, (ctypes.c_double * (n * n))(*flat)


def _cconfig(config: AnnealConfig | None, num_qubits: int = 0) -> _CConfig:
    c = config or AnnealConfig()
    if not 0 <= int(c.seed) <= (1 << 64) - 1:
        raise AnnealError("seed must be in [0, 2^64-1]")

    pts_ptr = None
    num_pts = 0
    pts_arr = None
    if c.schedule_points is not None:
        raw_pts = []
        for p in c.schedule_points:
            if isinstance(p, SchedulePoint):
                raw_pts.append(_CSchedulePoint(float(p.t), float(p.s)))
            else:
                raw_pts.append(_CSchedulePoint(float(p[0]), float(p[1])))
        num_pts = len(raw_pts)
        pts_arr = (_CSchedulePoint * num_pts)(*raw_pts)
        pts_ptr = ctypes.cast(pts_arr, ctypes.POINTER(_CSchedulePoint))

    offsets_ptr = None
    off_arr = None
    if c.anneal_offsets is not None:
        n_off = len(c.anneal_offsets)
        off_arr = (ctypes.c_double * n_off)(*[float(x) for x in c.anneal_offsets])
        offsets_ptr = ctypes.cast(off_arr, ctypes.POINTER(ctypes.c_double))

    cc = _CConfig(
        float(c.total_time),
        int(c.num_steps),
        int(c.num_samples),
        int(c.seed),
        int(c.schedule),
        float(c.driver_strength),
        float(c.problem_strength),
        int(bool(c.second_order)),
        pts_ptr,
        int(num_pts),
        int(bool(c.reverse_anneal)),
        int(c.initial_bitstring),
        float(c.reverse_s_target),
        float(c.reverse_hold_fraction),
        offsets_ptr,
    )
    cc._keepalive = (pts_arr, off_arr)
    return cc


def _collect(ptr: _ResultPtr) -> AnnealResult:
    try:
        n_samples = int(_lib.moonlab_anneal_result_num_samples(ptr))
        samples: list[int] = []
        energies: list[float] = []
        for index in range(n_samples):
            bits = ctypes.c_uint64()
            energy = ctypes.c_double()
            rc = _lib.moonlab_anneal_result_sample(
                ptr, index, ctypes.byref(bits), ctypes.byref(energy)
            )
            if rc != 0:
                raise AnnealError(f"result sample {index} failed with status {rc}")
            samples.append(int(bits.value))
            energies.append(float(energy.value))

        def get(name: str):
            return getattr(_lib, f"moonlab_anneal_result_{name}")(ptr)

        return AnnealResult(
            num_qubits=int(get("num_qubits")),
            effective_seed=int(get("effective_seed")),
            best_bitstring=int(get("best_bitstring")),
            most_likely_bitstring=int(get("most_likely_bitstring")),
            ground_bitstring=int(get("ground_bitstring")),
            ground_degeneracy=int(get("ground_degeneracy")),
            best_energy=float(get("best_energy")),
            ground_energy=float(get("ground_energy")),
            expected_energy=float(get("expected_energy")),
            success_probability=float(get("success_probability")),
            residual_energy=float(get("residual_energy")),
            problem_gap=float(get("problem_gap")),
            final_norm=float(get("final_norm")),
            samples=samples,
            sample_energies=energies,
        )
    finally:
        _lib.moonlab_anneal_result_free(ptr)


def anneal_ising(
    fields: Sequence[float],
    couplings: Sequence[Sequence[float]],
    *,
    offset: float = 0.0,
    config: AnnealConfig | None = None,
) -> AnnealResult:
    """Anneal a symmetric Ising model ``sum J_ij Z_i Z_j + sum h_i Z_i``."""
    h = [float(value) for value in fields]
    n, J = _matrix(couplings, "couplings")
    if len(h) != n:
        raise AnnealError("fields length must match the coupling matrix")
    h_array = (ctypes.c_double * n)(*h)
    c = _cconfig(config)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_anneal_ising(
        n, h_array, J, float(offset), ctypes.byref(c), ctypes.byref(out)
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_anneal_ising failed with status {rc}")
    return _collect(out)


def anneal_qubo(
    Q: Sequence[Sequence[float]], *, offset: float = 0.0, config: AnnealConfig | None = None
) -> AnnealResult:
    """Anneal the full-matrix objective ``x.T @ Q @ x + offset``."""
    n, q_array = _matrix(Q, "Q")
    c = _cconfig(config)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_anneal_qubo(
        n, q_array, float(offset), ctypes.byref(c), ctypes.byref(out)
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_anneal_qubo failed with status {rc}")
    return _collect(out)


def qubo_to_ising(
    Q: Sequence[Sequence[float]], *, offset: float = 0.0
) -> tuple[list[float], list[list[float]], float]:
    """Convert ``x.T @ Q @ x + offset`` to an exactly equivalent Ising model."""
    n, q_array = _matrix(Q, "Q")
    h = (ctypes.c_double * n)()
    J = (ctypes.c_double * (n * n))()
    converted_offset = ctypes.c_double()
    rc = _lib.moonlab_qubo_to_ising(n, q_array, float(offset), h, J, ctypes.byref(converted_offset))
    if rc != 0:
        raise AnnealError(f"moonlab_qubo_to_ising failed with status {rc}")
    return (
        [float(h[i]) for i in range(n)],
        [[float(J[i * n + j]) for j in range(n)] for i in range(n)],
        float(converted_offset.value),
    )


def make_pause_schedule(
    s_pause: float, pause_start_frac: float, pause_duration_frac: float
) -> list[SchedulePoint]:
    """Create a 4-point pause schedule."""
    pts = (_CSchedulePoint * 4)()
    rc = _lib.moonlab_anneal_schedule_make_pause(
        float(s_pause), float(pause_start_frac), float(pause_duration_frac), pts
    )
    if rc != 0:
        raise AnnealError(f"make_pause_schedule failed with status {rc}")
    return [SchedulePoint(float(p.t), float(p.s)) for p in pts]


def make_quench_schedule(
    s_quench: float, quench_start_frac: float
) -> list[SchedulePoint]:
    """Create a 3-point quench schedule."""
    pts = (_CSchedulePoint * 3)()
    rc = _lib.moonlab_anneal_schedule_make_quench(
        float(s_quench), float(quench_start_frac), pts
    )
    if rc != 0:
        raise AnnealError(f"make_quench_schedule failed with status {rc}")
    return [SchedulePoint(float(p.t), float(p.s)) for p in pts]


def reverse_anneal_ising(
    fields: Sequence[float],
    couplings: Sequence[Sequence[float]],
    *,
    initial_bitstring: int,
    s_target: float,
    hold_fraction: float,
    offset: float = 0.0,
    config: AnnealConfig | None = None,
) -> AnnealResult:
    """Reverse anneal an Ising model starting from a classical bitstring."""
    h = [float(value) for value in fields]
    n, J = _matrix(couplings, "couplings")
    if len(h) != n:
        raise AnnealError("fields length must match the coupling matrix")
    h_array = (ctypes.c_double * n)(*h)
    c = _cconfig(config, num_qubits=n)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_reverse_anneal_ising(
        n,
        h_array,
        J,
        float(offset),
        int(initial_bitstring),
        float(s_target),
        float(hold_fraction),
        ctypes.byref(c),
        ctypes.byref(out),
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_reverse_anneal_ising failed with status {rc}")
    return _collect(out)


def reverse_anneal_qubo(
    Q: Sequence[Sequence[float]],
    *,
    initial_bitstring: int,
    s_target: float,
    hold_fraction: float,
    offset: float = 0.0,
    config: AnnealConfig | None = None,
) -> AnnealResult:
    """Reverse anneal a QUBO problem starting from a classical bitstring."""
    n, q_array = _matrix(Q, "Q")
    c = _cconfig(config, num_qubits=n)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_reverse_anneal_qubo(
        n,
        q_array,
        float(offset),
        int(initial_bitstring),
        float(s_target),
        float(hold_fraction),
        ctypes.byref(c),
        ctypes.byref(out),
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_reverse_anneal_qubo failed with status {rc}")
    return _collect(out)


def zephyr_graph_create(m: int) -> ZephyrGraph:
    """Create a D-Wave Advantage2 Zephyr graph topology Z_m."""
    ptr = _lib.moonlab_zephyr_graph_create(int(m))
    if not ptr:
        raise AnnealError(f"moonlab_zephyr_graph_create failed for m={m}")
    try:
        g = ptr.contents
        n_couplers = int(g.num_couplers)
        couplers = [
            (int(g.coupler_u[i]), int(g.coupler_v[i])) for i in range(n_couplers)
        ]
        return ZephyrGraph(
            m=int(g.m),
            t=int(g.t),
            num_qubits=int(g.num_qubits),
            num_couplers=n_couplers,
            couplers=couplers,
        )
    finally:
        _lib.moonlab_zephyr_graph_free(ptr)


def zephyr_find_clique_embedding(num_logical: int, m: int) -> ZephyrEmbedding:
    """Generate a deterministic clique (complete graph K_k) embedding into Zephyr Z_m."""
    ptr = _lib.moonlab_zephyr_find_clique_embedding(int(num_logical), int(m))
    if not ptr:
        raise AnnealError(
            f"moonlab_zephyr_find_clique_embedding failed for num_logical={num_logical}, m={m}"
        )
    try:
        emb = ptr.contents
        chains: list[list[int]] = []
        for i in range(int(emb.num_logical)):
            length = int(emb.chain_lengths[i])
            chain = [int(emb.chains[i][c]) for c in range(length)]
            chains.append(chain)
        return ZephyrEmbedding(num_logical=int(emb.num_logical), chains=chains)
    finally:
        _lib.moonlab_zephyr_embedding_free(ptr)


def zephyr_embed_ising(
    m: int,
    embedding: ZephyrEmbedding,
    fields: Sequence[float],
    couplings: Sequence[Sequence[float]],
    chain_strength: float,
) -> tuple[list[float], list[list[float]]]:
    """Embed a logical Ising problem onto the physical Zephyr graph."""
    g_ptr = _lib.moonlab_zephyr_graph_create(int(m))
    if not g_ptr:
        raise AnnealError(f"Failed to create Zephyr graph m={m}")
    try:
        n_phys = int(g_ptr.contents.num_qubits)
        n_log = embedding.num_logical
        if len(fields) != n_log:
            raise AnnealError("fields length must match number of logical variables")
        n, J_arr = _matrix(couplings, "couplings")
        if n != n_log:
            raise AnnealError("couplings size must match number of logical variables")

        h_arr = (ctypes.c_double * n_log)(*[float(x) for x in fields])

        c_emb = _CZephyrEmbedding()
        c_emb.num_logical = n_log
        c_emb.chain_lengths = (ctypes.c_size_t * n_log)(
            *[len(c) for c in embedding.chains]
        )
        chain_ptrs = (ctypes.POINTER(ctypes.c_size_t) * n_log)()
        keep_chains = []
        for i, c in enumerate(embedding.chains):
            arr = (ctypes.c_size_t * len(c))(*c)
            keep_chains.append(arr)
            chain_ptrs[i] = arr
        c_emb.chains = chain_ptrs

        out_h = (ctypes.c_double * n_phys)()
        out_J = (ctypes.c_double * (n_phys * n_phys))()

        rc = _lib.moonlab_zephyr_embed_ising(
            g_ptr,
            ctypes.byref(c_emb),
            h_arr,
            J_arr,
            float(chain_strength),
            out_h,
            out_J,
        )
        if rc != 0:
            raise AnnealError(f"moonlab_zephyr_embed_ising failed with status {rc}")

        phys_h = [float(out_h[i]) for i in range(n_phys)]
        phys_J = [
            [float(out_J[i * n_phys + j]) for j in range(n_phys)] for i in range(n_phys)
        ]
        return phys_h, phys_J
    finally:
        _lib.moonlab_zephyr_graph_free(g_ptr)


def zephyr_unembed_samples(
    embedding: ZephyrEmbedding,
    physical_samples: Sequence[int],
) -> tuple[list[int], list[float]]:
    """Decode physical bitstrings back to logical bitstrings via majority voting on chains."""
    n_log = embedding.num_logical
    n_samples = len(physical_samples)
    if n_samples == 0:
        return [], []

    c_emb = _CZephyrEmbedding()
    c_emb.num_logical = n_log
    c_emb.chain_lengths = (ctypes.c_size_t * n_log)(
        *[len(c) for c in embedding.chains]
    )
    chain_ptrs = (ctypes.POINTER(ctypes.c_size_t) * n_log)()
    keep_chains = []
    for i, c in enumerate(embedding.chains):
        arr = (ctypes.c_size_t * len(c))(*c)
        keep_chains.append(arr)
        chain_ptrs[i] = arr
    c_emb.chains = chain_ptrs

    phys_arr = (ctypes.c_uint64 * n_samples)(*[int(s) for s in physical_samples])
    log_arr = (ctypes.c_uint64 * n_samples)()
    break_arr = (ctypes.c_double * n_samples)()

    rc = _lib.moonlab_zephyr_unembed_samples(
        ctypes.byref(c_emb),
        n_samples,
        phys_arr,
        log_arr,
        break_arr,
    )
    if rc != 0:
        raise AnnealError(f"moonlab_zephyr_unembed_samples failed with status {rc}")

    logical = [int(log_arr[i]) for i in range(n_samples)]
    breaks = [float(break_arr[i]) for i in range(n_samples)]
    return logical, breaks


__all__ = [
    "AnnealError",
    "AnnealSchedule",
    "SchedulePoint",
    "AnnealConfig",
    "AnnealResult",
    "ZephyrGraph",
    "ZephyrEmbedding",
    "anneal_ising",
    "anneal_qubo",
    "qubo_to_ising",
    "make_pause_schedule",
    "make_quench_schedule",
    "reverse_anneal_ising",
    "reverse_anneal_qubo",
    "zephyr_graph_create",
    "zephyr_find_clique_embedding",
    "zephyr_embed_ising",
    "zephyr_unembed_samples",
]

