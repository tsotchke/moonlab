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

__version__ = "0.2.0-dev"
__author__ = "tsotchke"

from .core import (
    QuantumState,
    Gates,
    Measurement,
    QuantumError
)

# Algorithm wrappers. These historically failed to load because of an
# ABI mismatch; the mismatch was resolved in the 0.2 sweep but we keep
# the import guard so a partial / stripped build of libquantumsim does
# not crash the module at import. The intended state is
# _ALGO_AVAILABLE == True on a full build.
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

from .benchmarks import quantum_volume, QuantumVolumeResult
from .clifford import Clifford
from .topology import (
    ChernKPM, qwz_chern, berry_grid_qwz,
    berry_grid_haldane, ssh_winding,
)

__all__ = [
    'QuantumState',
    'Gates',
    'Measurement',
    'QuantumError',
    'quantum_volume',
    'QuantumVolumeResult',
    'Clifford',
    'ChernKPM',
    'qwz_chern',
    'berry_grid_qwz',
    'berry_grid_haldane',
    'ssh_winding',
]
if _ALGO_AVAILABLE:
    __all__ += ['VQE', 'QAOA', 'Grover', 'BellTest']