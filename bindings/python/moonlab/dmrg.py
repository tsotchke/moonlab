"""DMRG scalar-energy bindings.

Thin wrappers around the v0.10.0 stable-ABI entries:

- ``moonlab_dmrg_tfim_energy(num_sites, g, max_bond_dim, num_sweeps)``
  returns the ground-state energy of the transverse-field Ising model
  ``H = -J sum_i Z_i Z_{i+1} - g J sum_i X_i`` with ``J = 1``.

- ``moonlab_dmrg_heisenberg_energy(num_sites, J, Delta, h,
  max_bond_dim, num_sweeps)`` returns the ground-state energy of the
  anisotropic XXZ-with-field model ``H = J sum_i (X_i X_{i+1} +
  Y_i Y_{i+1} + Delta Z_i Z_{i+1}) - h sum_i Z_i``.

Both calls run two-site DMRG against an internally constructed MPO and
return ``DBL_MAX`` on parameter errors -- check the result against
``math.inf`` or a sentinel to detect bad inputs.

Heavier workflows that need the MPS handle, sweep history, or per-bond
truncation should use :mod:`moonlab.tdvp` (TDVP) or drop to the C ABI
directly.
"""

from __future__ import annotations

import ctypes

from .core import _lib

__all__ = ["tfim_ground_energy", "heisenberg_ground_energy"]


_lib.moonlab_dmrg_tfim_energy.argtypes = [
    ctypes.c_uint32,   # num_sites
    ctypes.c_double,   # g
    ctypes.c_uint32,   # max_bond_dim
    ctypes.c_uint32,   # num_sweeps
]
_lib.moonlab_dmrg_tfim_energy.restype = ctypes.c_double

_lib.moonlab_dmrg_heisenberg_energy.argtypes = [
    ctypes.c_uint32,   # num_sites
    ctypes.c_double,   # J
    ctypes.c_double,   # Delta
    ctypes.c_double,   # h
    ctypes.c_uint32,   # max_bond_dim
    ctypes.c_uint32,   # num_sweeps
]
_lib.moonlab_dmrg_heisenberg_energy.restype = ctypes.c_double


def tfim_ground_energy(
    num_sites: int,
    g: float,
    max_bond_dim: int = 32,
    num_sweeps: int = 10,
) -> float:
    """DMRG ground-state energy of the 1D transverse-field Ising model.

    Hamiltonian (J = 1)::

        H = -sum_i Z_i Z_{i+1} - g sum_i X_i

    Args:
        num_sites: Chain length (>= 2).
        g: Transverse field ratio h/J.  Critical point at g = 1.
        max_bond_dim: DMRG truncation cap.
        num_sweeps: Number of two-site DMRG sweeps.

    Returns:
        Ground-state energy.  ``inf`` (``DBL_MAX``) signals invalid input.
    """
    return float(_lib.moonlab_dmrg_tfim_energy(
        num_sites, g, max_bond_dim, num_sweeps))


def heisenberg_ground_energy(
    num_sites: int,
    J: float = 1.0,
    Delta: float = 1.0,
    h: float = 0.0,
    max_bond_dim: int = 32,
    num_sweeps: int = 10,
) -> float:
    """DMRG ground-state energy of the 1D XXZ-with-field chain.

    Hamiltonian::

        H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
            - h sum_i Z_i

    Args:
        num_sites: Chain length (>= 2).
        J: Exchange coupling.
        Delta: XXZ anisotropy.  Delta = 1 is isotropic Heisenberg.
        h: Longitudinal field strength.
        max_bond_dim: DMRG truncation cap.
        num_sweeps: Number of two-site DMRG sweeps.

    Returns:
        Ground-state energy.  ``inf`` (``DBL_MAX``) signals invalid input.
    """
    return float(_lib.moonlab_dmrg_heisenberg_energy(
        num_sites, J, Delta, h, max_bond_dim, num_sweeps))
