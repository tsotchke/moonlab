"""
Moonlab Quantum Computing Framework
====================================

Python bindings for the Moonlab quantum simulator.

Features:
- 32-qubit quantum state simulation
- Complete universal gate set
- VQE for molecular simulation
- QAOA for combinatorial optimization
- Bell-verified quantum behavior (CHSH = 2.828)
- 10,000× optimized for Apple Silicon

Quick Start:
    >>> from moonlab import QuantumState
    >>> state = QuantumState(num_qubits=2)
    >>> state.h(0)  # Hadamard on qubit 0
    >>> state.cnot(0, 1)  # Entangle qubits
    >>> result = state.measure(0)  # Measure qubit 0
"""

__version__ = "0.1.0-dev"
__author__ = "tsotchke"

from .core import (
    QuantumState,
    Gates,
    Measurement,
    QuantumError
)

from .algorithms import (
    VQE,
    QAOA,
    Grover,
    BellTest
)

__all__ = [
    'QuantumState',
    'Gates',
    'Measurement',
    'QuantumError',
    'VQE',
    'QAOA',
    'Grover',
    'BellTest',
]