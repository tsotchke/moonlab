"""Python tests for moonlab.qgtl mirror the C test_qgtl_backend suite.

Validates Bell, GHZ, shot sampling, Grover, RY, and error paths."""

from __future__ import annotations

import pytest

from moonlab.qgtl import QgtlCircuit, GateType, QgtlError


def test_bell_pair():
    c = QgtlCircuit(num_qubits=2)
    c.add_gate(GateType.H, target=0)
    c.add_gate(GateType.CNOT, target=1, control=0)
    assert c.num_gates == 2
    r = c.execute(return_probabilities=True)
    assert r.probabilities is not None
    assert abs(r.probabilities[0] - 0.5) < 1e-9
    assert abs(r.probabilities[3] - 0.5) < 1e-9
    assert r.probabilities[1] < 1e-9
    assert r.probabilities[2] < 1e-9


def test_ghz_3():
    c = (QgtlCircuit(num_qubits=3)
         .add_gate(GateType.H, target=0)
         .add_gate(GateType.CNOT, target=1, control=0)
         .add_gate(GateType.CNOT, target=2, control=1))
    r = c.execute(return_probabilities=True)
    assert abs(r.probabilities[0] - 0.5) < 1e-9
    assert abs(r.probabilities[7] - 0.5) < 1e-9
    for b in range(1, 7):
        assert r.probabilities[b] < 1e-9


def test_shot_sampling():
    c = (QgtlCircuit(num_qubits=2)
         .add_gate(GateType.H, target=0)
         .add_gate(GateType.CNOT, target=1, control=0))
    r = c.execute(num_shots=1024, rng_seed=0xdeadbeef)
    assert r.outcomes is not None
    assert len(r.outcomes) == 1024
    assert all(o in (0, 3) for o in r.outcomes)
    n00 = sum(1 for o in r.outcomes if o == 0)
    # Statistical: n00 ~ Binomial(1024, 0.5), 5sigma is +/- 80.
    assert 1024 // 2 - 80 < n00 < 1024 // 2 + 80


def test_grover_n2():
    c = QgtlCircuit(num_qubits=2)
    c.add_gate(GateType.H, target=0).add_gate(GateType.H, target=1)
    c.add_gate(GateType.CZ, target=1, control=0)
    c.add_gate(GateType.H, target=0).add_gate(GateType.H, target=1)
    c.add_gate(GateType.Z, target=0).add_gate(GateType.Z, target=1)
    c.add_gate(GateType.CZ, target=1, control=0)
    c.add_gate(GateType.H, target=0).add_gate(GateType.H, target=1)
    r = c.execute(return_probabilities=True)
    assert r.probabilities[3] > 0.99


def test_ry_half_pi():
    import math
    c = QgtlCircuit(num_qubits=1)
    c.add_gate(GateType.RY, target=0, params=[math.pi / 2])
    r = c.execute(return_probabilities=True)
    assert abs(r.probabilities[0] - 0.5) < 1e-9
    assert abs(r.probabilities[1] - 0.5) < 1e-9


def test_rejects_zero_qubits():
    with pytest.raises(QgtlError):
        QgtlCircuit(num_qubits=0)


def test_rejects_oversize():
    with pytest.raises(QgtlError):
        QgtlCircuit(num_qubits=33)


def test_rejects_out_of_range_target():
    c = QgtlCircuit(num_qubits=2)
    with pytest.raises(QgtlError):
        c.add_gate(GateType.H, target=2)


def test_rejects_cnot_self():
    c = QgtlCircuit(num_qubits=2)
    with pytest.raises(QgtlError):
        c.add_gate(GateType.CNOT, target=0, control=0)


def test_gate_type_enum_matches_qgtl_numbering():
    """Numerical contract: QGTL passes gate_type_t cast directly."""
    assert int(GateType.I) == 0
    assert int(GateType.X) == 1
    assert int(GateType.H) == 4
    assert int(GateType.RX) == 7
    assert int(GateType.CNOT) == 10
    assert int(GateType.CZ) == 12
    assert int(GateType.SWAP) == 13
