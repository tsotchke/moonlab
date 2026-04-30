"""Smoke + correctness tests for the moonlab.ca_mps Python bindings.

Pins:
    - CAMPS create/free + basic Clifford gates work.
    - The new ABI entry points (var_d_run, gauge_warmstart,
      z2_lgt_1d_build / gauss_law, status_string) round-trip correctly
      from Python through the v0.2.1 stable ABI.
"""

from __future__ import annotations

import numpy as np
import pytest

import moonlab


pytestmark = pytest.mark.skipif(
    not getattr(moonlab, "_CAMPS_AVAILABLE", False),
    reason="moonlab.ca_mps did not load (libquantumsim missing var-D/Z2 ABI?)",
)


def test_camps_basic_lifecycle():
    s = moonlab.CAMPS(num_qubits=4, max_bond_dim=16)
    assert s.num_qubits == 4
    s.h(0)
    s.cnot(0, 1)
    assert abs(s.norm - 1.0) < 1e-9


def test_status_string_canonical():
    assert moonlab.status_string(0, 0) == "SUCCESS"
    assert moonlab.status_string(1, -1) == "ERR_INVALID"
    # Fallback for unknown code: must be non-empty.
    assert "CLIFFORD" in moonlab.status_string(11, -42)


def test_z2_lgt_build_and_gauss_law():
    paulis, coeffs = moonlab.z2_lgt_1d_build(N=4, t=1.0, h=0.5,
                                                m=0.0, gauss_penalty=0.0)
    assert paulis.shape[1] == 7   # 2 * 4 - 1
    assert paulis.shape[0] == coeffs.shape[0]
    assert coeffs.shape[0] > 0

    g1 = moonlab.z2_lgt_1d_gauss_law(N=4, site_x=1)
    assert g1.shape == (7,)
    # G_1 = X_1 Z_2 X_3 -> bytes 0,1,3,1,0,0,0 (X=1, Z=3).
    assert g1[1] == 1 and g1[2] == 3 and g1[3] == 1
    assert g1[0] == 0 and g1[4] == 0 and g1[5] == 0 and g1[6] == 0


def test_gauge_warmstart_bell_pair():
    """Generators {XX, ZZ} stabilise the Bell state.  Apply the
    warmstart and verify the prepared state is normalised."""
    s = moonlab.CAMPS(num_qubits=2, max_bond_dim=8)
    gens = np.array([[1, 1],   # X X
                     [3, 3]],  # Z Z
                    dtype=np.uint8)
    moonlab.gauge_warmstart(s, gens)
    assert abs(s.norm - 1.0) < 1e-7


def test_gauge_warmstart_rejects_anticommuting():
    """Generators {X_0, Z_0} anticommute -> RuntimeError."""
    s = moonlab.CAMPS(num_qubits=1, max_bond_dim=4)
    gens = np.array([[1], [3]], dtype=np.uint8)
    with pytest.raises(RuntimeError):
        moonlab.gauge_warmstart(s, gens)


def test_var_d_run_short_smoke():
    """Run var-D for a single outer iter on a 4-qubit TFIM-like
    Pauli sum.  Pin: returns a finite energy and the state stays
    normalised throughout."""
    n = 4
    # H = -sum_i Z_i Z_{i+1}  (toy TFIM ferromagnet, J = 1, h = 0).
    paulis = np.zeros((n - 1, n), dtype=np.uint8)
    coeffs = np.full(n - 1, -1.0, dtype=np.float64)
    for i in range(n - 1):
        paulis[i, i] = 3      # Z
        paulis[i, i + 1] = 3  # Z

    s = moonlab.CAMPS(num_qubits=n, max_bond_dim=8)
    energy = moonlab.var_d_run(
        s, paulis, coeffs,
        max_outer_iters=2,
        imag_time_steps_per_outer=2,
        clifford_passes_per_outer=2,
        warmstart=moonlab.WARMSTART_IDENTITY,
    )
    assert np.isfinite(energy)
    assert abs(s.norm - 1.0) < 1e-3   # imag-time renormalises
