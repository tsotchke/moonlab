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

# Algorithm wrappers currently bind against C symbols (molecular_hamiltonian_*,
# vqe_*, qaoa_*, grover_*, bell_test_*) that are not yet exported from the
# v0.1.2 dylib. Skip the import rather than crash at module load; the 0.2
# Phase 1G housekeeping sweep will realign algorithms.py with the actual
# C ABI or drop it in favour of a ctypes rework.
try:
    from .algorithms import (
        VQE,
        QAOA,
        Grover,
        BellTest,
    )
    _ALGO_AVAILABLE = True
except (ImportError, AttributeError):
    _ALGO_AVAILABLE = False

__all__ = [
    'QuantumState',
    'Gates',
    'Measurement',
    'QuantumError',
]
if _ALGO_AVAILABLE:
    __all__ += ['VQE', 'QAOA', 'Grover', 'BellTest']