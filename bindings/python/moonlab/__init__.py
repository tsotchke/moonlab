"""
Moonlab Quantum Computing Framework
====================================

Python bindings for the Moonlab quantum simulator.

Features:
- 32-qubit quantum state simulation
- Complete universal gate set
- VQE for molecular simulation
- QAOA for combinatorial optimization
- Bell inequality violation on explicit Bell states
  (CHSH ~ 2.87 measured; 2.828 is the Tsirelson bound)
- Native reverse-mode autograd (moonlab.diff) for parameterized
  circuits -- adjoint gradients without PyTorch
- SIMD-dispatched C core with an optional Metal GPU backend on
  Apple Silicon (see docs/benchmarks/reproducible-benchmarks.md)

Quick Start:
    >>> from moonlab import QuantumState
    >>> state = QuantumState(num_qubits=2)
    >>> state.h(0)  # Hadamard on qubit 0
    >>> state.cnot(0, 1)  # Entangle qubits
    >>> result = state.measure(0)  # Measure qubit 0
"""

__version__ = "1.2.0"
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
except (ImportError, AttributeError, OSError):
    _ALGO_AVAILABLE = False

# Everything below is optional in exactly the same sense as the guarded
# blocks further down this file: a stripped / size-trimmed libquantumsim
# build (e.g. WASM) can be missing any of these entry points, and that
# must not crash `import moonlab` itself. OSError is included alongside
# ImportError/AttributeError because a missing symbol surfaces as an
# OSError from ctypes' CDLL.__getattr__, not an ImportError.
try:
    from .benchmarks import quantum_volume, QuantumVolumeResult
    _BENCHMARKS_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _BENCHMARKS_AVAILABLE = False

try:
    from .clifford import Clifford
    _CLIFFORD_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _CLIFFORD_AVAILABLE = False

try:
    from .topology import (
        ChernKPM, qwz_chern, berry_grid_qwz,
        berry_grid_haldane, ssh_winding,
        # v0.3 additions
        chern_qwz_proj, chern_qwz_parallel_transport,
        kane_mele_z2, bhz_z2, kitaev_chain_z2, hofstadter_chern,
        # v0.3.2 curvature-grid variants
        berry_grid_qwz_proj, berry_grid_qwz_pt,
    )
    _TOPOLOGY_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _TOPOLOGY_AVAILABLE = False

try:
    from .diff import DiffCircuit, PauliTerm, OBS_Z, OBS_X, OBS_Y
    _DIFF_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _DIFF_AVAILABLE = False

try:
    from . import crypto
    _CRYPTO_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _CRYPTO_AVAILABLE = False

# Matrix-product density operator noise simulator (since v0.3.0).
# Optional import: a stripped libquantumsim build without these
# entry points is still a usable Moonlab.
try:
    from .mpdo import Mpdo
    _MPDO_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _MPDO_AVAILABLE = False

# Adaptive-bond time-dependent variational principle (since v0.4).
# Same optional-import policy as MPDO.
try:
    from . import tdvp as _tdvp_module
    tdvp = _tdvp_module
    _TDVP_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _TDVP_AVAILABLE = False

# Single-qubit gate-fusion DAG (since v0.4.4).
try:
    from .fusion import FusedCircuit, FuseStats
    _FUSION_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _FUSION_AVAILABLE = False

# Clifford-Assisted MPS + var-D + gauge-aware warmstart + Z2 LGT.
# Optional import: a stripped libquantumsim build without these
# entry points (e.g. WASM size-trimmed) should still import the
# top-level moonlab module.
try:
    from .ca_mps import (
        CAMPS,
        WARMSTART_IDENTITY,
        WARMSTART_H_ALL,
        WARMSTART_DUAL_TFIM,
        WARMSTART_FERRO_TFIM,
        WARMSTART_STABILIZER_SUBGROUP,
        var_d_run,
        gauge_warmstart,
        z2_lgt_1d_build,
        z2_lgt_1d_gauss_law,
        status_string,
    )
    _CAMPS_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _CAMPS_AVAILABLE = False

try:
    from .ca_peps import CAPEPS
    _CAPEPS_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _CAPEPS_AVAILABLE = False

try:
    from .surface_code import SurfaceCode
    _SURFACE_CODE_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _SURFACE_CODE_AVAILABLE = False

try:
    from .libirrep_qec import (
        LibirrepQecCode,
        LibirrepError,
        LibirrepNotBuiltError,
        is_available as libirrep_is_available,
    )
    _LIBIRREP_QEC_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _LIBIRREP_QEC_AVAILABLE = False

try:
    from .qgtl import QgtlCircuit, QgtlResults, QgtlError, GateType as QgtlGateType
    _QGTL_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _QGTL_AVAILABLE = False

try:
    from .scheduler import Job, JobResults, SchedulerError
    _SCHEDULER_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _SCHEDULER_AVAILABLE = False

try:
    from .decoder import (
        DecoderSlot, DecoderError, DecoderNotBuiltError,
        decode as decoder_decode,
        slot_available as decoder_slot_available,
        slot_name as decoder_slot_name,
    )
    _DECODER_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _DECODER_AVAILABLE = False

# Base __all__: names from .core, imported unconditionally above (if
# these fail to import, the whole package fails to import, so they are
# always present). Everything else is appended below, gated on the same
# availability flag that gated its import, so __all__ never advertises a
# name that isn't actually bound on this build.
__all__ = [
    'QuantumState',
    'Gates',
    'Measurement',
    'QuantumError',
]
if _ALGO_AVAILABLE:
    __all__ += ['VQE', 'QAOA', 'Grover', 'BellTest']
if _BENCHMARKS_AVAILABLE:
    __all__ += ['quantum_volume', 'QuantumVolumeResult']
if _CLIFFORD_AVAILABLE:
    __all__ += ['Clifford']
if _TOPOLOGY_AVAILABLE:
    __all__ += [
        'ChernKPM',
        'qwz_chern',
        'berry_grid_qwz',
        'berry_grid_haldane',
        'ssh_winding',
        'chern_qwz_proj',
        'chern_qwz_parallel_transport',
        'kane_mele_z2',
        'bhz_z2',
        'kitaev_chain_z2',
        'hofstadter_chern',
        'berry_grid_qwz_proj',
        'berry_grid_qwz_pt',
    ]
if _DIFF_AVAILABLE:
    __all__ += ['DiffCircuit', 'PauliTerm', 'OBS_Z', 'OBS_X', 'OBS_Y']
if _CRYPTO_AVAILABLE:
    __all__ += ['crypto']
if _MPDO_AVAILABLE:
    __all__ += ['Mpdo']
if _TDVP_AVAILABLE:
    __all__ += ['tdvp']
if _FUSION_AVAILABLE:
    __all__ += ['FusedCircuit', 'FuseStats']
if _CAMPS_AVAILABLE:
    __all__ += [
        'CAMPS',
        'WARMSTART_IDENTITY',
        'WARMSTART_H_ALL',
        'WARMSTART_DUAL_TFIM',
        'WARMSTART_FERRO_TFIM',
        'WARMSTART_STABILIZER_SUBGROUP',
        'var_d_run',
        'gauge_warmstart',
        'z2_lgt_1d_build',
        'z2_lgt_1d_gauss_law',
        'status_string',
    ]
if _CAPEPS_AVAILABLE:
    __all__ += ['CAPEPS']
if _SURFACE_CODE_AVAILABLE:
    __all__ += ['SurfaceCode']
if _LIBIRREP_QEC_AVAILABLE:
    __all__ += [
        'LibirrepQecCode',
        'LibirrepError',
        'LibirrepNotBuiltError',
        'libirrep_is_available',
    ]
if _QGTL_AVAILABLE:
    __all__ += ['QgtlCircuit', 'QgtlResults', 'QgtlError', 'QgtlGateType']
if _SCHEDULER_AVAILABLE:
    __all__ += ['Job', 'JobResults', 'SchedulerError']
if _DECODER_AVAILABLE:
    __all__ += [
        'DecoderSlot',
        'DecoderError',
        'DecoderNotBuiltError',
        'decoder_decode',
        'decoder_slot_available',
        'decoder_slot_name',
    ]