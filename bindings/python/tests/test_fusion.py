"""Pytest suite for the v0.4.4 gate-fusion DAG Python wrapper.

Validates the FusedCircuit binding around
``src/optimization/fusion/fusion.{c,h}`` end to end:
construction, gate-append surface, fuser-pass statistics,
state-vector execution, and equivalence against the unfused path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from moonlab import QuantumState
from moonlab.fusion import FuseStats, FusedCircuit


# ---- Lifecycle / accessors --------------------------------------------------


def test_create_and_len():
    c = FusedCircuit(num_qubits=4)
    assert c.num_qubits == 4
    assert len(c) == 0


def test_fluent_append_increments_len():
    c = (
        FusedCircuit(3)
        .h(0)
        .rz(0, 0.3)
        .rx(0, 0.4)
        .cnot(0, 1)
        .ry(1, 0.5)
    )
    assert len(c) == 5


def test_invalid_num_qubits_rejected():
    with pytest.raises(ValueError):
        FusedCircuit(0)


# ---- Compile statistics -----------------------------------------------------


def test_compile_returns_stats_with_run_fusion():
    """Three single-qubit gates on the same qubit should fuse into one."""
    c = (
        FusedCircuit(2)
        .h(0).rz(0, 0.3).rx(0, 0.4)   # three 1q on qubit 0 -> fuses
        .cnot(0, 1)
    )
    fused, stats = c.compile()
    assert isinstance(stats, FuseStats)
    assert stats.original_gates == 4
    # Three single-qubit gates collapse into one FUSED_1Q, plus the
    # CNOT survives, so the fused circuit has 2 gates and the fuser
    # records two merges (rz into h, rx into h*rz).
    assert stats.fused_gates == 2
    assert stats.merges_applied == 2
    assert len(fused) == stats.fused_gates


def test_compile_passthrough_when_no_runs():
    """Alternating multi-qubit gates leave nothing to fuse."""
    c = (
        FusedCircuit(3)
        .h(0).cnot(0, 1).h(1).cnot(1, 2).h(2)
    )
    fused, stats = c.compile()
    assert stats.original_gates == 5
    # Each single-qubit gate is the only one on its qubit between
    # multi-qubit boundaries; nothing fuses (zero merges).
    assert stats.merges_applied == 0
    assert stats.fused_gates == 5


# ---- Execution against state vector -----------------------------------------


def _bell_state_probabilities() -> tuple[float, float, float, float]:
    """Reference probabilities for |Phi+> = (|00> + |11>)/sqrt(2)."""
    return (0.5, 0.0, 0.0, 0.5)


def test_execute_unfused_bell_state():
    """H_0 + CNOT(0,1) on |00> produces the |Phi+> Bell state."""
    circuit = FusedCircuit(2).h(0).cnot(0, 1)
    state = QuantumState(num_qubits=2)
    circuit.execute(state)

    probs = state.probabilities()
    assert len(probs) == 4
    for measured, expected in zip(probs, _bell_state_probabilities()):
        assert measured == pytest.approx(expected, abs=1e-10)


def test_execute_fused_matches_unfused():
    """Fusing a single-qubit run must produce the same state vector
    as the unfused execution (gate fusion is exact)."""
    def build():
        return (
            FusedCircuit(3)
            .h(0).rz(0, 0.7).rx(0, 0.3)
            .cnot(0, 1)
            .ry(1, 0.2).rz(1, 0.9)
            .cnot(1, 2)
            .rx(2, 0.4)
        )

    state_unfused = QuantumState(num_qubits=3)
    build().execute(state_unfused)
    probs_unfused = state_unfused.probabilities()

    state_fused = QuantumState(num_qubits=3)
    fused, _stats = build().compile()
    fused.execute(state_fused)
    probs_fused = state_fused.probabilities()

    for u, f in zip(probs_unfused, probs_fused):
        assert u == pytest.approx(f, abs=1e-10)


def test_execute_two_qubit_parameterised_gates():
    """Cover the cphase / crx / cry / crz / swap appenders."""
    c = (
        FusedCircuit(3)
        .h(0).h(1).h(2)
        .cphase(0, 1, 0.5)
        .crx(0, 1, 0.2)
        .cry(1, 2, 0.3)
        .crz(1, 2, 0.4)
        .swap(0, 2)
    )
    state = QuantumState(num_qubits=3)
    c.execute(state)
    # Probabilities sum to 1 -> the C call chain ran without leaking
    # or producing NaN.
    total = sum(state.probabilities())
    assert total == pytest.approx(1.0, abs=1e-10)


def test_u3_appender():
    """U3 is the universal single-qubit gate; cover the three-param call."""
    c = FusedCircuit(1).u3(0, math.pi / 2, math.pi / 4, math.pi / 4)
    state = QuantumState(num_qubits=1)
    c.execute(state)
    total = sum(state.probabilities())
    assert total == pytest.approx(1.0, abs=1e-10)


# ---- Module exports ---------------------------------------------------------


def test_top_level_exports():
    import moonlab
    assert hasattr(moonlab, "FusedCircuit")
    assert hasattr(moonlab, "FuseStats")
    assert moonlab.FusedCircuit is FusedCircuit
    assert moonlab.FuseStats is FuseStats
