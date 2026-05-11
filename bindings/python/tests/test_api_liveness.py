"""Focused API coverage for Python binding methods used through public APIs."""

import numpy as np

from moonlab import Clifford
from moonlab.core import Complex
from moonlab.ml import QuantumPCA
from moonlab.visualization import CircuitDiagram


def test_complex_to_python_returns_builtin_complex():
    value = Complex(1.25, -0.5).to_python()

    assert value == complex(1.25, -0.5)


def test_clifford_s_dag_inverts_s_gate():
    state = Clifford(num_qubits=1, seed=0xCAFE)

    state.h(0)
    state.s(0)
    state.s_dag(0)
    state.h(0)

    outcome, kind = state.measure(0)
    assert outcome == 0
    assert kind == 0


def test_quantum_pca_fit_transform_projects_components():
    X = np.array(
        [
            [1.0, 0.2],
            [0.8, 0.1],
            [-0.9, -0.2],
            [-1.1, -0.1],
        ],
        dtype=float,
    )
    model = QuantumPCA(num_components=1, num_qubits=1)

    projected = model.fit_transform(X)

    assert projected.shape == (4, 1)
    assert model.components_.shape == (1, 2)
    np.testing.assert_allclose(projected, model.transform(X))


def test_circuit_diagram_set_qubit_label_updates_rendered_wire():
    circuit = CircuitDiagram(2).set_qubit_label(1, "ancilla")

    assert circuit.qubit_labels[1] == "ancilla"
    assert "ancilla:" in circuit.to_ascii()
